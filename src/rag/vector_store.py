"""
Vector Store Interface using ChromaDB
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Provides a wrapper around ChromaDB for storing and querying document embeddings
with support for metadata filtering and batch operations.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.utils.config_loader import get_config, get_settings
from src.utils.logger import get_logger, RAGLogger
from src.utils.exceptions import VectorStoreException

logger = get_logger("vector_store")
rag_logger = RAGLogger()


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    distance: float


class VectorStore:
    """
    ChromaDB-based vector store for document embeddings.
    
    Features:
    - Document storage with metadata
    - Similarity search with filtering
    - Batch operations for efficiency
    - Automatic ID generation from content
    
    Usage:
        store = VectorStore()
        store.add_documents([doc1, doc2])
        results = store.search("query text", top_k=5)
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for local persistence
            host: ChromaDB server host (for client mode)
            port: ChromaDB server port (for client mode)
        """
        settings = get_settings()
        rag_config = get_config().get_rag_config()
        
        self.collection_name = collection_name or settings.chroma_collection
        self.persist_directory = persist_directory or rag_config.get("vector_store", {}).get(
            "settings", {}
        ).get("persist_directory", "./chroma_data")
        
        # Initialize ChromaDB client
        if host and port:
            # Client mode - connect to ChromaDB server
            self._client = chromadb.HttpClient(host=host, port=port)
            logger.info(f"Connected to ChromaDB server at {host}:{port}")
        else:
            # Persistent mode - local storage
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"Using persistent ChromaDB at {self.persist_directory}")
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        
        logger.info(
            f"VectorStore initialized: collection={self.collection_name}, "
            f"documents={self._collection.count()}"
        )
    
    def _generate_id(self, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Generate a unique ID for a document based on content hash.
        
        Args:
            content: Document content
            metadata: Optional metadata to include in hash
            
        Returns:
            Unique document ID
        """
        hash_input = content
        if metadata:
            hash_input += str(sorted(metadata.items()))
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects with embeddings
            batch_size: Size of batches for bulk insert
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        start_time = time.time()
        added_count = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = []
            embeddings = []
            contents = []
            metadatas = []
            
            for doc in batch:
                doc_id = doc.id or self._generate_id(doc.content, doc.metadata)
                ids.append(doc_id)
                contents.append(doc.content)
                metadatas.append(doc.metadata or {})
                
                if doc.embedding:
                    embeddings.append(doc.embedding)
            
            try:
                if embeddings:
                    self._collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=contents,
                        metadatas=metadatas,
                    )
                else:
                    self._collection.add(
                        ids=ids,
                        documents=contents,
                        metadatas=metadatas,
                    )
                
                added_count += len(batch)
                
            except Exception as e:
                logger.error(f"Failed to add batch: {e}")
                raise VectorStoreException(
                    message=f"Failed to add documents: {e}",
                    operation="add",
                    collection=self.collection_name,
                )
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Added {added_count} documents in {duration_ms:.2f}ms "
            f"({added_count / (duration_ms / 1000):.1f} docs/sec)"
        )
        
        return added_count
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_distances: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            include_distances: Whether to include distance scores
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )
            
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    score = 1 - distance
                    
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=results["documents"][0][i] if results.get("documents") else "",
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                        score=score,
                        distance=distance,
                    ))
            
            duration_ms = (time.time() - start_time) * 1000
            top_score = search_results[0].score if search_results else 0
            
            rag_logger.log_retrieval(
                query="[embedding query]",
                num_results=len(search_results),
                top_score=top_score,
                latency_ms=duration_ms,
                request_id="",
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreException(
                message=f"Search failed: {e}",
                operation="search",
                collection=self.collection_name,
            )
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search using text query (uses ChromaDB's default embedding).
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )
            
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    score = 1 - distance
                    
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=results["documents"][0][i] if results.get("documents") else "",
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                        score=score,
                        distance=distance,
                    ))
            
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"Text search completed in {duration_ms:.2f}ms")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise VectorStoreException(
                message=f"Text search failed: {e}",
                operation="search",
                collection=self.collection_name,
            )
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        try:
            result = self._collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"],
            )
            
            if result["ids"]:
                return Document(
                    id=result["ids"][0],
                    content=result["documents"][0] if result.get("documents") else "",
                    metadata=result["metadatas"][0] if result.get("metadatas") else {},
                    embedding=result["embeddings"][0] if result.get("embeddings") else None,
                )
            return None
            
        except Exception as e:
            logger.error(f"Get document failed: {e}")
            return None
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete documents by IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not doc_ids:
            return 0
        
        try:
            self._collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise VectorStoreException(
                message=f"Delete failed: {e}",
                operation="delete",
                collection=self.collection_name,
            )
    
    def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Update a document.
        
        Args:
            doc_id: Document ID
            content: New content (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)
            
        Returns:
            True if successful
        """
        try:
            update_args = {"ids": [doc_id]}
            
            if content is not None:
                update_args["documents"] = [content]
            if metadata is not None:
                update_args["metadatas"] = [metadata]
            if embedding is not None:
                update_args["embeddings"] = [embedding]
            
            self._collection.update(**update_args)
            return True
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
        logger.info(f"Cleared collection {self.collection_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.count(),
            "persist_directory": self.persist_directory,
        }


# Global singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
