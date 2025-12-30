"""
Document Chunker with Multiple Strategies
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Provides various chunking strategies for splitting documents into
optimal-sized chunks for embedding and retrieval.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("chunker")


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    metadata: Dict[str, Any]
    start_index: int
    end_index: int
    chunk_index: int
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough: 4 chars per token)."""
        return len(self.content) // 4


class BaseChunker(ABC):
    """Abstract base class for chunkers."""
    
    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        pass


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking with character-based splitting.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split text into fixed-size chunks."""
        chunks = []
        metadata = metadata or {}
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Don't split in the middle of a word
            if end < len(text):
                # Look for a space near the boundary
                space_index = text.rfind(" ", start, end)
                if space_index > start:
                    end = space_index
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunks.append(Chunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "fixed",
                    },
                    start_index=start,
                    end_index=end,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
            
            start = end - self.chunk_overlap
            if start <= chunks[-1].start_index if chunks else 0:
                start = end
        
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking that respects document structure.
    
    Splits on natural boundaries (paragraphs, sentences) while
    maintaining target chunk size.
    """
    
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        min_chunk_size: int = 100,
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to try, in order
            min_chunk_size: Minimum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.min_chunk_size = min_chunk_size
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split text recursively on natural boundaries."""
        metadata = metadata or {}
        
        # Recursively split
        chunks_text = self._split_recursive(text, self.separators)
        
        # Create Chunk objects
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks_text):
            start_idx = text.find(chunk_text, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            
            end_idx = start_idx + len(chunk_text)
            
            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "chunking_strategy": "recursive",
                },
                start_index=start_idx,
                end_index=end_idx,
                chunk_index=i,
            ))
            
            current_pos = end_idx
        
        return chunks
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Recursively split text on separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split on current separator
        if separator:
            splits = text.split(separator)
        else:
            # Last resort: character split
            splits = list(text)
        
        # Merge small chunks and split large ones
        final_chunks = []
        current_chunk = ""
        
        for split in splits:
            if not split.strip():
                continue
            
            # Add separator back (except for the first chunk)
            if current_chunk and separator:
                test_chunk = current_chunk + separator + split
            else:
                test_chunk = current_chunk + split if current_chunk else split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is ready
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    final_chunks.append(current_chunk)
                elif current_chunk:
                    # Chunk too small, try to merge with next
                    pass
                
                # Start new chunk
                if len(split) > self.chunk_size and remaining_separators:
                    # Recursively split this part
                    sub_chunks = self._split_recursive(split, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # Add remaining chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            final_chunks.append(current_chunk)
        elif current_chunk and final_chunks:
            # Merge with last chunk
            final_chunks[-1] = final_chunks[-1] + separator + current_chunk
        elif current_chunk:
            final_chunks.append(current_chunk)
        
        return final_chunks


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking that groups complete sentences.
    """
    
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 1,  # Overlap in sentences
    ):
        """
        Initialize sentence chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Number of sentences to overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split text into sentence-based chunks."""
        metadata = metadata or {}
        
        # Split into sentences
        sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks
        chunks = []
        current_sentences: List[str] = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_sentences:
                # Create chunk
                chunk_content = " ".join(current_sentences)
                chunks.append(Chunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "sentence_count": len(current_sentences),
                        "chunking_strategy": "sentence",
                    },
                    start_index=0,  # Would need to track properly
                    end_index=len(chunk_content),
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
                
                # Keep overlap sentences
                current_sentences = current_sentences[-self.chunk_overlap:] if self.chunk_overlap else []
                current_length = sum(len(s) for s in current_sentences)
            
            current_sentences.append(sentence)
            current_length += len(sentence)
        
        # Add remaining sentences
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunks.append(Chunk(
                content=chunk_content,
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "sentence_count": len(current_sentences),
                    "chunking_strategy": "sentence",
                },
                start_index=0,
                end_index=len(chunk_content),
                chunk_index=chunk_index,
            ))
        
        return chunks


class DocumentChunker:
    """
    Main chunker interface with multiple strategy support.
    
    Usage:
        chunker = DocumentChunker(strategy="recursive")
        chunks = chunker.chunk(document_text, metadata={"source": "doc.md"})
    """
    
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize the document chunker.
        
        Args:
            strategy: Chunking strategy to use
            chunk_size: Target chunk size (uses config default if not provided)
            chunk_overlap: Chunk overlap (uses config default if not provided)
        """
        rag_config = get_config().get_rag_config()
        chunking_config = rag_config.get("chunking", {})
        
        self.strategy = strategy or chunking_config.get("strategy", "recursive")
        self.chunk_size = chunk_size or chunking_config.get("settings", {}).get("chunk_size", 512)
        self.chunk_overlap = chunk_overlap or chunking_config.get("settings", {}).get("chunk_overlap", 50)
        
        # Initialize the appropriate chunker
        self._chunker = self._create_chunker()
        
        logger.info(
            f"DocumentChunker initialized: strategy={self.strategy}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
    
    def _create_chunker(self) -> BaseChunker:
        """Create the appropriate chunker based on strategy."""
        if self.strategy == "fixed":
            return FixedSizeChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.strategy == "recursive":
            return RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.strategy == "sentence":
            return SentenceChunker(
                chunk_size=self.chunk_size,
            )
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using recursive")
            return RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        chunks = self._chunker.chunk(text, metadata)
        
        logger.debug(
            f"Chunked document into {len(chunks)} chunks "
            f"(avg size: {sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0})"
        )
        
        return chunks
    
    def chunk_with_headers(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk text while preserving header context.
        
        Extracts headers and includes them in chunk metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
            
        Returns:
            List of Chunk objects with header metadata
        """
        metadata = metadata or {}
        
        # Extract sections by headers
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        
        sections = []
        last_end = 0
        current_headers: Dict[int, str] = {}
        
        for match in header_pattern.finditer(text):
            # Save previous section
            if last_end < match.start():
                section_text = text[last_end:match.start()].strip()
                if section_text:
                    sections.append((section_text, dict(current_headers)))
            
            # Update current header
            level = len(match.group(1))
            header_text = match.group(2).strip()
            
            # Clear lower-level headers
            current_headers = {k: v for k, v in current_headers.items() if k < level}
            current_headers[level] = header_text
            
            last_end = match.end()
        
        # Add final section
        if last_end < len(text):
            section_text = text[last_end:].strip()
            if section_text:
                sections.append((section_text, dict(current_headers)))
        
        # Chunk each section
        all_chunks = []
        chunk_index = 0
        
        for section_text, headers in sections:
            # Create header context
            header_path = " > ".join(
                headers[k] for k in sorted(headers.keys())
            )
            
            section_metadata = {
                **metadata,
                "section": header_path,
                "headers": headers,
            }
            
            section_chunks = self._chunker.chunk(section_text, section_metadata)
            
            for chunk in section_chunks:
                chunk.chunk_index = chunk_index
                all_chunks.append(chunk)
                chunk_index += 1
        
        return all_chunks
