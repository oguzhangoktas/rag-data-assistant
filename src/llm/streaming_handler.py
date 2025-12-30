"""
Streaming Handler for Server-Sent Events (SSE)
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Manages streaming LLM responses via Server-Sent Events for real-time
response delivery to clients.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("streaming_handler")


class StreamEventType(str, Enum):
    """Types of streaming events."""
    START = "start"
    CONTENT = "content"
    METADATA = "metadata"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """Represents a single streaming event."""
    event_type: StreamEventType
    data: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_sse(self) -> str:
        """
        Convert to Server-Sent Event format.
        
        Returns:
            SSE formatted string
        """
        event_data = {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        
        if self.metadata:
            event_data["metadata"] = self.metadata
        
        # SSE format: event: type\ndata: json\n\n
        return f"event: {self.event_type.value}\ndata: {json.dumps(event_data)}\n\n"


@dataclass
class StreamingStats:
    """Statistics for a streaming session."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_chunks: int = 0
    total_characters: int = 0
    error_count: int = 0
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def characters_per_second(self) -> float:
        """Get streaming speed."""
        duration_s = self.duration_ms / 1000
        if duration_s == 0:
            return 0
        return self.total_characters / duration_s


class StreamingHandler:
    """
    Handles streaming LLM responses via Server-Sent Events.
    
    Features:
    - SSE formatting
    - Chunk buffering for optimal delivery
    - Progress tracking
    - Error handling during streams
    - Heartbeat for connection keepalive
    
    Usage:
        handler = StreamingHandler()
        async for event in handler.stream_response(llm_stream):
            yield event.to_sse()
    """
    
    def __init__(
        self,
        chunk_size: int = 50,
        heartbeat_interval: float = 15.0,
        buffer_enabled: bool = True,
    ):
        """
        Initialize the streaming handler.
        
        Args:
            chunk_size: Minimum characters to buffer before sending
            heartbeat_interval: Seconds between heartbeat events
            buffer_enabled: Whether to buffer small chunks
        """
        self.chunk_size = chunk_size
        self.heartbeat_interval = heartbeat_interval
        self.buffer_enabled = buffer_enabled
        
        logger.debug(
            f"StreamingHandler initialized: chunk_size={chunk_size}, "
            f"heartbeat={heartbeat_interval}s"
        )
    
    async def stream_response(
        self,
        content_generator: AsyncGenerator[str, None],
        request_id: Optional[str] = None,
        include_metadata: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream LLM response as events.
        
        Args:
            content_generator: Async generator yielding content chunks
            request_id: Optional request ID for tracking
            include_metadata: Whether to include timing metadata
            
        Yields:
            StreamEvent objects
        """
        stats = StreamingStats()
        buffer = ""
        
        # Send start event
        yield StreamEvent(
            event_type=StreamEventType.START,
            data="",
            metadata={"request_id": request_id} if request_id else {},
        )
        
        try:
            async for chunk in content_generator:
                stats.total_chunks += 1
                stats.total_characters += len(chunk)
                
                if self.buffer_enabled:
                    buffer += chunk
                    
                    # Send when buffer is large enough
                    if len(buffer) >= self.chunk_size:
                        yield StreamEvent(
                            event_type=StreamEventType.CONTENT,
                            data=buffer,
                        )
                        buffer = ""
                else:
                    # Send each chunk immediately
                    yield StreamEvent(
                        event_type=StreamEventType.CONTENT,
                        data=chunk,
                    )
            
            # Send any remaining buffered content
            if buffer:
                yield StreamEvent(
                    event_type=StreamEventType.CONTENT,
                    data=buffer,
                )
            
            stats.end_time = time.time()
            
            # Send completion event with metadata
            done_metadata = {}
            if include_metadata:
                done_metadata = {
                    "request_id": request_id,
                    "total_chunks": stats.total_chunks,
                    "total_characters": stats.total_characters,
                    "duration_ms": round(stats.duration_ms, 2),
                    "chars_per_second": round(stats.characters_per_second, 2),
                }
            
            yield StreamEvent(
                event_type=StreamEventType.DONE,
                data="",
                metadata=done_metadata,
            )
            
        except Exception as e:
            stats.error_count += 1
            stats.end_time = time.time()
            
            logger.error(f"Streaming error: {e}")
            
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=str(e),
                metadata={"request_id": request_id, "error_type": type(e).__name__},
            )
    
    async def stream_with_heartbeat(
        self,
        content_generator: AsyncGenerator[str, None],
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream response with periodic heartbeats for keepalive.
        
        Args:
            content_generator: Async generator yielding content chunks
            request_id: Optional request ID
            
        Yields:
            StreamEvent objects including heartbeats
        """
        last_heartbeat = time.time()
        
        async for event in self.stream_response(content_generator, request_id):
            yield event
            
            # Send heartbeat if interval has passed
            current_time = time.time()
            if current_time - last_heartbeat >= self.heartbeat_interval:
                yield StreamEvent(
                    event_type=StreamEventType.METADATA,
                    data="heartbeat",
                    metadata={"heartbeat": True},
                )
                last_heartbeat = current_time
    
    def format_sse(
        self,
        data: str,
        event_type: str = "message",
        event_id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> str:
        """
        Format data as Server-Sent Event.
        
        Args:
            data: Event data
            event_type: Event type name
            event_id: Optional event ID
            retry: Optional retry interval in ms
            
        Returns:
            SSE formatted string
        """
        lines = []
        
        if event_id:
            lines.append(f"id: {event_id}")
        
        if retry is not None:
            lines.append(f"retry: {retry}")
        
        lines.append(f"event: {event_type}")
        
        # Handle multiline data
        for line in data.split("\n"):
            lines.append(f"data: {line}")
        
        lines.append("")
        lines.append("")
        
        return "\n".join(lines)
    
    async def create_sse_response(
        self,
        content_generator: AsyncGenerator[str, None],
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Create an SSE response generator for FastAPI.
        
        Args:
            content_generator: Async generator yielding content chunks
            request_id: Optional request ID
            
        Yields:
            SSE formatted strings
        """
        async for event in self.stream_with_heartbeat(content_generator, request_id):
            yield event.to_sse()


class StreamingBuffer:
    """
    Buffer for accumulating streaming content.
    
    Useful for building up context while streaming.
    """
    
    def __init__(self, max_size: int = 100000):
        """
        Initialize the buffer.
        
        Args:
            max_size: Maximum buffer size in characters
        """
        self._content: List[str] = []
        self._size = 0
        self._max_size = max_size
    
    def append(self, chunk: str) -> None:
        """
        Append a chunk to the buffer.
        
        Args:
            chunk: Content chunk to append
        """
        if self._size + len(chunk) > self._max_size:
            # Truncate old content if needed
            excess = self._size + len(chunk) - self._max_size
            while excess > 0 and self._content:
                removed = self._content.pop(0)
                self._size -= len(removed)
                excess -= len(removed)
        
        self._content.append(chunk)
        self._size += len(chunk)
    
    def get_content(self) -> str:
        """
        Get accumulated content.
        
        Returns:
            Full buffered content as string
        """
        return "".join(self._content)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._content = []
        self._size = 0
    
    @property
    def size(self) -> int:
        """Get current buffer size."""
        return self._size


async def stream_to_string(
    generator: AsyncGenerator[str, None],
    max_length: Optional[int] = None,
) -> str:
    """
    Consume a streaming generator and return full content.
    
    Args:
        generator: Async generator yielding string chunks
        max_length: Optional maximum length to collect
        
    Returns:
        Complete string from generator
    """
    parts = []
    total_length = 0
    
    async for chunk in generator:
        if max_length and total_length + len(chunk) > max_length:
            # Truncate to max length
            remaining = max_length - total_length
            if remaining > 0:
                parts.append(chunk[:remaining])
            break
        
        parts.append(chunk)
        total_length += len(chunk)
    
    return "".join(parts)
