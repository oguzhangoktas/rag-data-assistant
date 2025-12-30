"""
Document Retriever with Semantic Search
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Provides semantic search over documents with optional reranking
and context building for RAG applications.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.rag.vector_store import VectorStore, SearchResult, get_vector_store
from src.rag.embeddings import EmbeddingGenerator, get_embedding_generator
from src.utils.config_loader import get_config
from src.utils.logger import get_logger, RAGLogger
from src.utils.exceptions import RetrievalException

logger = get_logger("retriever")
rag_logger = RAGLogger()


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    documents: List[SearchResult]
    query: str
    total_results: int
    latency_ms: float
    context: str  # Combined context for LLM
    sources: List[Dict[str, str]]  # Source citations


class Retriever:
    """
    Semantic document retriever with context building.
    
    Features:
    - Semantic search using embeddings
    - Optional reranking for better relevance
    - Context building for LLM prompts
    - Source citation tracking
    - Similarity threshold filtering
    
    Usage:
        retriever = Retriever()
        result = await retriever.retrieve("What is customer lifetime value?")
        print(result.context)  # Context for LLM
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance (or uses global)
            embedding_generator: EmbeddingGenerator instance (or uses global)
        """
        self._vector_store = vector_store or get_vector_store()
        self._embeddings = embedding_generator or get_embedding_generator()
        
        # Load config
        rag_config = get_config().get_rag_config()
        retrieval_config = rag_config.get("retrieval", {})
        
        self.top_k = retrieval_config.get("top_k", 5)
        self.similarity_threshold = retrieval_config.get("similarity_threshold", 0.7)
        self.reranking_enabled = retrieval_config.get("reranking", {}).get("enabled", False)
        self.max_context_tokens = rag_config.get("context_building", {}).get(
            "max_context_tokens", 4000
        )
        
        logger.info(
            f"Retriever initialized: top_k={self.top_k}, "
            f"threshold={self.similarity_threshold}, reranking={self.reranking_enabled}"
        )
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (overrides default)
            filter_metadata: Optional metadata filter
            min_score: Minimum similarity score (overrides default)
            
        Returns:
            RetrievalResult with documents and context
        """
        start_time = time.time()
        top_k = top_k or self.top_k
        min_score = min_score or self.similarity_threshold
        
        try:
            # Generate query embedding
            embedding_result = await self._embeddings.embed_text(query)
            query_embedding = embedding_result.embedding
            
            # Search vector store
            search_results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2 if self.reranking_enabled else top_k,  # Get more for reranking
                filter_metadata=filter_metadata,
            )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in search_results
                if r.score >= min_score
            ]
            
            # Rerank if enabled
            if self.reranking_enabled and len(filtered_results) > top_k:
                filtered_results = self._rerank(query, filtered_results, top_k)
            else:
                filtered_results = filtered_results[:top_k]
            
            # Build context and sources
            context = self._build_context(filtered_results)
            sources = self._extract_sources(filtered_results)
            
            latency_ms = (time.time() - start_time) * 1000
            
            rag_logger.log_retrieval(
                query=query,
                num_results=len(filtered_results),
                top_score=filtered_results[0].score if filtered_results else 0,
                latency_ms=latency_ms,
                request_id="",
            )
            
            return RetrievalResult(
                documents=filtered_results,
                query=query,
                total_results=len(filtered_results),
                latency_ms=latency_ms,
                context=context,
                sources=sources,
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalException(
                message=f"Retrieval failed: {e}",
                query=query,
            )
    
    def retrieve_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Synchronous version of retrieve.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filter
            min_score: Minimum similarity score
            
        Returns:
            RetrievalResult with documents and context
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.retrieve(query, top_k, filter_metadata, min_score)
        )
    
    def _rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank results using a simple scoring heuristic.
        
        In production, this could use a cross-encoder model.
        
        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        # Simple keyword-based reranking boost
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            overlap = len(query_words & content_words)
            
            # Boost score based on keyword overlap
            boost = overlap * 0.05
            result.score = min(1.0, result.score + boost)
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _build_context(
        self,
        results: List[SearchResult],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build context string from search results.
        
        Args:
            results: Search results to include
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documentation found."
        
        max_tokens = max_tokens or self.max_context_tokens
        
        context_parts = []
        estimated_tokens = 0
        
        for i, result in enumerate(results):
            # Format each chunk with source citation
            source = result.metadata.get("source", "Unknown")
            section = result.metadata.get("section", "")
            
            header = f"[Document {i+1}: {source}"
            if section:
                header += f", Section: {section}"
            header += "]"
            
            chunk = f"{header}\n{result.content}\n"
            
            # Estimate tokens (rough: 4 chars per token)
            chunk_tokens = len(chunk) // 4
            
            if estimated_tokens + chunk_tokens > max_tokens:
                # Truncate if exceeding limit
                if context_parts:
                    break
                else:
                    # At least include first chunk (truncated)
                    remaining_tokens = max_tokens - estimated_tokens
                    remaining_chars = remaining_tokens * 4
                    chunk = chunk[:remaining_chars] + "..."
            
            context_parts.append(chunk)
            estimated_tokens += chunk_tokens
        
        return "\n".join(context_parts)
    
    def _extract_sources(
        self,
        results: List[SearchResult],
    ) -> List[Dict[str, str]]:
        """
        Extract source citations from results.
        
        Args:
            results: Search results
            
        Returns:
            List of source citations
        """
        sources = []
        seen_sources = set()
        
        for result in results:
            source = result.metadata.get("source", "Unknown")
            
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append({
                    "source": source,
                    "section": result.metadata.get("section", ""),
                    "score": round(result.score, 4),
                })
        
        return sources
    
    async def retrieve_with_expansion(
        self,
        query: str,
        expand_queries: int = 2,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve with query expansion for better recall.
        
        Generates alternative queries and combines results.
        
        Args:
            query: Original query
            expand_queries: Number of query variations
            top_k: Results per query
            
        Returns:
            Combined RetrievalResult
        """
        # For now, just use the original query
        # In production, this could use LLM to generate query variations
        return await self.retrieve(query, top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "reranking_enabled": self.reranking_enabled,
            "max_context_tokens": self.max_context_tokens,
            "vector_store_count": self._vector_store.count(),
        }


# Global singleton instance
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
