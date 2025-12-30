"""
Embedding Generator using OpenAI
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Generates embeddings for documents and queries using OpenAI's embedding models
with caching support for cost optimization.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from functools import lru_cache

from openai import AsyncOpenAI, OpenAI

from src.utils.config_loader import get_config, get_settings
from src.utils.logger import get_logger, RAGLogger
from src.utils.exceptions import EmbeddingException

logger = get_logger("embeddings")
rag_logger = RAGLogger()


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: List[float]
    tokens_used: int
    model: str
    cached: bool = False


@dataclass
class BatchEmbeddingResult:
    """Result from batch embedding generation."""
    embeddings: List[List[float]]
    total_tokens: int
    model: str
    latency_ms: float
    cache_hits: int = 0


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings.
    
    Uses content hash as key to avoid recomputing embeddings
    for identical content.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self._cache: Dict[str, List[float]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None
        """
        key = self._hash_text(text)
        embedding = self._cache.get(key)
        
        if embedding:
            self._hits += 1
            return embedding
        
        self._misses += 1
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding.
        
        Args:
            text: Original text
            embedding: Generated embedding
        """
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        key = self._hash_text(text)
        self._cache[key] = embedding
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI's embedding models.
    
    Features:
    - Batch embedding generation
    - Caching for cost reduction
    - Token usage tracking
    - Async and sync support
    
    Usage:
        generator = EmbeddingGenerator()
        result = await generator.embed_text("Hello, world!")
        batch_result = await generator.embed_batch(["text1", "text2"])
    """
    
    # Pricing per 1000 tokens
    EMBEDDING_PRICES = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.0001,
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_size: int = 10000,
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model: Embedding model name
            api_key: OpenAI API key (or from env)
            cache_enabled: Whether to enable caching
            cache_size: Maximum cache size
        """
        settings = get_settings()
        rag_config = get_config().get_rag_config()
        
        self.model = model or rag_config.get("embeddings", {}).get(
            "model", "text-embedding-3-small"
        )
        self.dimensions = rag_config.get("embeddings", {}).get("dimensions", 1536)
        self.batch_size = rag_config.get("embeddings", {}).get("batch_size", 100)
        
        # Initialize OpenAI client
        api_key = api_key or settings.openai_api_key
        self._client = AsyncOpenAI(api_key=api_key)
        self._sync_client = OpenAI(api_key=api_key)
        
        # Initialize cache
        self.cache_enabled = cache_enabled
        self._cache = EmbeddingCache(max_size=cache_size) if cache_enabled else None
        
        # Track total usage
        self._total_tokens = 0
        self._total_requests = 0
        
        logger.info(
            f"EmbeddingGenerator initialized: model={self.model}, "
            f"cache_enabled={cache_enabled}"
        )
    
    async def embed_text(
        self,
        text: str,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache
            
        Returns:
            EmbeddingResult with embedding vector
        """
        # Check cache first
        if use_cache and self._cache:
            cached = self._cache.get(text)
            if cached:
                return EmbeddingResult(
                    embedding=cached,
                    tokens_used=0,
                    model=self.model,
                    cached=True,
                )
        
        try:
            response = await self._client.embeddings.create(
                model=self.model,
                input=text,
            )
            
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens
            
            # Update stats
            self._total_tokens += tokens_used
            self._total_requests += 1
            
            # Cache the result
            if use_cache and self._cache:
                self._cache.set(text, embedding)
            
            return EmbeddingResult(
                embedding=embedding,
                tokens_used=tokens_used,
                model=self.model,
                cached=False,
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingException(
                message=f"Failed to generate embedding: {e}",
                model=self.model,
            )
    
    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache
            
        Returns:
            BatchEmbeddingResult with all embeddings
        """
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                total_tokens=0,
                model=self.model,
                latency_ms=0,
            )
        
        start_time = time.time()
        
        # Separate cached and uncached texts
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        cache_hits = 0
        
        if use_cache and self._cache:
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached:
                    embeddings[i] = cached
                    cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
        
        total_tokens = 0
        
        # Generate embeddings for uncached texts in batches
        for batch_start in range(0, len(uncached_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]
            
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )
                
                total_tokens += response.usage.total_tokens
                
                for j, embedding_data in enumerate(response.data):
                    idx = batch_indices[j]
                    embedding = embedding_data.embedding
                    embeddings[idx] = embedding
                    
                    # Cache the result
                    if use_cache and self._cache:
                        self._cache.set(batch_texts[j], embedding)
                        
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise EmbeddingException(
                    message=f"Failed to generate batch embeddings: {e}",
                    model=self.model,
                )
        
        # Update stats
        self._total_tokens += total_tokens
        self._total_requests += 1
        
        latency_ms = (time.time() - start_time) * 1000
        
        rag_logger.log_embedding(
            num_texts=len(texts),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            request_id="",
        )
        
        # Filter out None values (shouldn't happen but just in case)
        final_embeddings = [e for e in embeddings if e is not None]
        
        return BatchEmbeddingResult(
            embeddings=final_embeddings,
            total_tokens=total_tokens,
            model=self.model,
            latency_ms=latency_ms,
            cache_hits=cache_hits,
        )
    
    def embed_text_sync(self, text: str) -> EmbeddingResult:
        """
        Synchronous version of embed_text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get(text)
            if cached:
                return EmbeddingResult(
                    embedding=cached,
                    tokens_used=0,
                    model=self.model,
                    cached=True,
                )
        
        try:
            response = self._sync_client.embeddings.create(
                model=self.model,
                input=text,
            )
            
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens
            
            self._total_tokens += tokens_used
            self._total_requests += 1
            
            if self._cache:
                self._cache.set(text, embedding)
            
            return EmbeddingResult(
                embedding=embedding,
                tokens_used=tokens_used,
                model=self.model,
                cached=False,
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingException(
                message=f"Failed to generate embedding: {e}",
                model=self.model,
            )
    
    def calculate_cost(self, tokens: int) -> float:
        """
        Calculate cost for token usage.
        
        Args:
            tokens: Number of tokens
            
        Returns:
            Cost in USD
        """
        price_per_1k = self.EMBEDDING_PRICES.get(self.model, 0.0001)
        return (tokens / 1000) * price_per_1k
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generator statistics."""
        stats = {
            "model": self.model,
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "estimated_cost_usd": round(self.calculate_cost(self._total_tokens), 6),
        }
        
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Embedding cache cleared")


# Global singleton instance
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get the global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
