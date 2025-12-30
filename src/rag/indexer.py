"""
Document Indexer for RAG Pipeline
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Provides batch document indexing with chunking, embedding, and vector store
insertion for the RAG pipeline.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.rag.vector_store import VectorStore, Document, get_vector_store
from src.rag.embeddings import EmbeddingGenerator, get_embedding_generator
from src.rag.chunker import DocumentChunker, Chunk
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.exceptions import DocumentIngestionException

logger = get_logger("indexer")


@dataclass
class IndexingResult:
    """Result from document indexing."""
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    total_tokens: int
    latency_ms: float
    errors: List[str]


@dataclass
class DocumentSource:
    """Represents a document to be indexed."""
    content: str
    source: str
    title: Optional[str] = None
    document_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentIndexer:
    """
    Indexes documents into the vector store for RAG retrieval.
    
    Features:
    - Batch document processing
    - Automatic chunking
    - Embedding generation
    - Deduplication
    - Incremental updates
    
    Usage:
        indexer = DocumentIndexer()
        result = await indexer.index_documents([doc1, doc2])
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        """
        Initialize the document indexer.
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            chunker: DocumentChunker instance
        """
        self._vector_store = vector_store or get_vector_store()
        self._embeddings = embedding_generator or get_embedding_generator()
        self._chunker = chunker or DocumentChunker()
        
        # Load config
        rag_config = get_config().get_rag_config()
        ingestion_config = rag_config.get("ingestion", {})
        
        self.batch_size = ingestion_config.get("batch_size", 50)
        self.deduplication_enabled = ingestion_config.get("deduplication", {}).get("enabled", True)
        
        # Track indexed document hashes for deduplication
        self._indexed_hashes: set = set()
        
        logger.info(
            f"DocumentIndexer initialized: batch_size={self.batch_size}, "
            f"dedup={self.deduplication_enabled}"
        )
    
    def _compute_hash(self, content: str) -> str:
        """Compute hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def index_documents(
        self,
        documents: List[DocumentSource],
        skip_duplicates: bool = True,
    ) -> IndexingResult:
        """
        Index a list of documents.
        
        Args:
            documents: List of DocumentSource objects
            skip_duplicates: Whether to skip duplicate content
            
        Returns:
            IndexingResult with statistics
        """
        start_time = time.time()
        errors: List[str] = []
        
        all_chunks: List[Tuple[Chunk, str]] = []  # (chunk, doc_hash)
        
        # Step 1: Chunk all documents
        for doc in documents:
            try:
                doc_hash = self._compute_hash(doc.content)
                
                # Skip if already indexed
                if skip_duplicates and self.deduplication_enabled:
                    if doc_hash in self._indexed_hashes:
                        logger.debug(f"Skipping duplicate document: {doc.source}")
                        continue
                
                # Create metadata
                metadata = doc.metadata or {}
                metadata.update({
                    "source": doc.source,
                    "title": doc.title or doc.source,
                    "document_type": doc.document_type or "unknown",
                    "content_hash": doc_hash,
                })
                
                # Chunk the document
                chunks = self._chunker.chunk_with_headers(doc.content, metadata)
                
                for chunk in chunks:
                    all_chunks.append((chunk, doc_hash))
                
            except Exception as e:
                error_msg = f"Failed to process document {doc.source}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if not all_chunks:
            return IndexingResult(
                documents_processed=len(documents),
                chunks_created=0,
                chunks_indexed=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                errors=errors,
            )
        
        # Step 2: Generate embeddings in batches
        chunk_texts = [chunk.content for chunk, _ in all_chunks]
        
        try:
            embedding_result = await self._embeddings.embed_batch(chunk_texts)
            embeddings = embedding_result.embeddings
            total_tokens = embedding_result.total_tokens
        except Exception as e:
            error_msg = f"Failed to generate embeddings: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return IndexingResult(
                documents_processed=len(documents),
                chunks_created=len(all_chunks),
                chunks_indexed=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                errors=errors,
            )
        
        # Step 3: Create Document objects and add to vector store
        vector_docs: List[Document] = []
        indexed_hashes: set = set()
        
        for i, (chunk, doc_hash) in enumerate(all_chunks):
            doc_id = f"{doc_hash[:8]}_{chunk.chunk_index}"
            
            vector_docs.append(Document(
                id=doc_id,
                content=chunk.content,
                metadata=chunk.metadata,
                embedding=embeddings[i] if i < len(embeddings) else None,
            ))
            
            indexed_hashes.add(doc_hash)
        
        # Step 4: Add to vector store
        try:
            indexed_count = self._vector_store.add_documents(
                vector_docs,
                batch_size=self.batch_size,
            )
            
            # Update tracked hashes
            self._indexed_hashes.update(indexed_hashes)
            
        except Exception as e:
            error_msg = f"Failed to add to vector store: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            indexed_count = 0
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Indexed {indexed_count} chunks from {len(documents)} documents "
            f"in {latency_ms:.2f}ms"
        )
        
        return IndexingResult(
            documents_processed=len(documents),
            chunks_created=len(all_chunks),
            chunks_indexed=indexed_count,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            errors=errors,
        )
    
    async def index_file(
        self,
        file_path: str,
        document_type: Optional[str] = None,
    ) -> IndexingResult:
        """
        Index a single file.
        
        Args:
            file_path: Path to the file
            document_type: Optional document type override
            
        Returns:
            IndexingResult with statistics
        """
        path = Path(file_path)
        
        if not path.exists():
            raise DocumentIngestionException(
                message=f"File not found: {file_path}",
                document_path=file_path,
            )
        
        # Detect document type from extension
        if document_type is None:
            ext = path.suffix.lower()
            type_map = {
                ".md": "markdown",
                ".markdown": "markdown",
                ".txt": "text",
                ".json": "json",
                ".csv": "csv",
            }
            document_type = type_map.get(ext, "text")
        
        # Read file content
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise DocumentIngestionException(
                message=f"Failed to read file: {e}",
                document_path=file_path,
                document_type=document_type,
            )
        
        # Create document source
        doc = DocumentSource(
            content=content,
            source=path.name,
            title=path.stem,
            document_type=document_type,
        )
        
        return await self.index_documents([doc])
    
    async def index_directory(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> IndexingResult:
        """
        Index all matching files in a directory.
        
        Args:
            directory_path: Path to directory
            file_patterns: File patterns to match (e.g., ["*.md", "*.txt"])
            recursive: Whether to search recursively
            
        Returns:
            Combined IndexingResult
        """
        dir_path = Path(directory_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise DocumentIngestionException(
                message=f"Directory not found: {directory_path}",
                document_path=directory_path,
            )
        
        # Default patterns
        if file_patterns is None:
            file_patterns = ["*.md", "*.txt", "*.json"]
        
        # Find all matching files
        files: List[Path] = []
        for pattern in file_patterns:
            if recursive:
                files.extend(dir_path.rglob(pattern))
            else:
                files.extend(dir_path.glob(pattern))
        
        logger.info(f"Found {len(files)} files to index in {directory_path}")
        
        # Index each file
        total_result = IndexingResult(
            documents_processed=0,
            chunks_created=0,
            chunks_indexed=0,
            total_tokens=0,
            latency_ms=0,
            errors=[],
        )
        
        for file_path in files:
            try:
                result = await self.index_file(str(file_path))
                
                total_result.documents_processed += result.documents_processed
                total_result.chunks_created += result.chunks_created
                total_result.chunks_indexed += result.chunks_indexed
                total_result.total_tokens += result.total_tokens
                total_result.latency_ms += result.latency_ms
                total_result.errors.extend(result.errors)
                
            except Exception as e:
                error_msg = f"Failed to index {file_path}: {e}"
                logger.error(error_msg)
                total_result.errors.append(error_msg)
        
        return total_result
    
    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._vector_store.clear()
        self._indexed_hashes.clear()
        logger.info("Index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        return {
            "indexed_documents": len(self._indexed_hashes),
            "total_chunks": self._vector_store.count(),
            "batch_size": self.batch_size,
            "deduplication_enabled": self.deduplication_enabled,
            "embedding_stats": self._embeddings.get_stats(),
            "vector_store_stats": self._vector_store.get_stats(),
        }
