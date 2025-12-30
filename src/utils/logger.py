"""
Structured Logging Configuration using Loguru
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from loguru import logger


def json_serializer(record: dict) -> str:
    """
    Custom JSON serializer for structured logging.
    
    Args:
        record: Loguru record dictionary
        
    Returns:
        JSON formatted log string
    """
    subset = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add extra fields if present
    if record.get("extra"):
        subset["extra"] = record["extra"]
    
    # Add exception info if present
    if record.get("exception"):
        subset["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
        }
    
    return json.dumps(subset, default=str)


def json_sink(message):
    """Sink that outputs JSON formatted logs."""
    record = message.record
    serialized = json_serializer(record)
    print(serialized, file=sys.stderr)


def setup_logger(
    level: str = "INFO",
    log_format: str = "text",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure the global logger with specified settings.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('text' or 'json')
        log_file: Optional file path for log output
        rotation: Log rotation size/time
        retention: Log retention period
    """
    # Remove default handler
    logger.remove()
    
    # Console format based on log_format setting
    if log_format == "json":
        # JSON format for production
        logger.add(
            json_sink,
            level=level,
            format="{message}",
            backtrace=True,
            diagnose=True,
        )
    else:
        # Human-readable format for development
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            level=level,
            format=console_format,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{module}:{function}:{line} | "
            "{message}"
        )
        logger.add(
            log_file,
            level=level,
            format=file_format,
            rotation=rotation,
            retention=retention,
            compression="gz",
            backtrace=True,
            diagnose=True,
        )
    
    logger.info(f"Logger initialized with level={level}, format={log_format}")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance, optionally with a specific name binding.
    
    Args:
        name: Optional name to bind to the logger
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggerContext:
    """
    Context manager for adding contextual information to logs.
    
    Usage:
        with LoggerContext(request_id="abc123", user_id="user1"):
            logger.info("Processing request")  # Will include request_id and user_id
    """
    
    def __init__(self, **kwargs):
        """
        Initialize with context key-value pairs.
        
        Args:
            **kwargs: Context fields to add to all logs in this context
        """
        self.context = kwargs
        self._token = None
    
    def __enter__(self):
        """Enter context and bind context fields to logger."""
        self._token = logger.contextualize(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and remove context bindings."""
        if self._token:
            self._token.__exit__(exc_type, exc_val, exc_tb)


class RequestLogger:
    """
    Logger specifically for API request/response logging.
    
    Provides structured logging for HTTP requests with timing and metadata.
    """
    
    def __init__(self):
        """Initialize request logger."""
        self.logger = get_logger("api")
    
    def log_request(
        self,
        method: str,
        path: str,
        request_id: str,
        user_id: Optional[str] = None,
        body_size: Optional[int] = None,
    ) -> None:
        """
        Log an incoming API request.
        
        Args:
            method: HTTP method
            path: Request path
            request_id: Unique request identifier
            user_id: Optional user identifier
            body_size: Optional request body size in bytes
        """
        self.logger.info(
            f"Request started",
            method=method,
            path=path,
            request_id=request_id,
            user_id=user_id,
            body_size=body_size,
        )
    
    def log_response(
        self,
        method: str,
        path: str,
        request_id: str,
        status_code: int,
        duration_ms: float,
        response_size: Optional[int] = None,
    ) -> None:
        """
        Log an API response.
        
        Args:
            method: HTTP method
            path: Request path
            request_id: Unique request identifier
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            response_size: Optional response body size in bytes
        """
        log_level = "INFO" if status_code < 400 else "WARNING" if status_code < 500 else "ERROR"
        
        getattr(self.logger, log_level.lower())(
            f"Request completed",
            method=method,
            path=path,
            request_id=request_id,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            response_size=response_size,
        )


class LLMLogger:
    """
    Logger specifically for LLM operations.
    
    Provides structured logging for LLM calls with token counts, latency, and costs.
    """
    
    def __init__(self):
        """Initialize LLM logger."""
        self.logger = get_logger("llm")
    
    def log_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        request_id: str,
    ) -> None:
        """
        Log an LLM request.
        
        Args:
            provider: LLM provider name
            model: Model name
            prompt_tokens: Number of tokens in prompt
            request_id: Unique request identifier
        """
        self.logger.info(
            "LLM request started",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            request_id=request_id,
        )
    
    def log_response(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        cost_usd: float,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Log an LLM response.
        
        Args:
            provider: LLM provider name
            model: Model name
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens used
            latency_ms: Request latency in milliseconds
            cost_usd: Estimated cost in USD
            request_id: Unique request identifier
            success: Whether the request was successful
            error: Optional error message if failed
        """
        log_data = {
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_usd, 6),
            "request_id": request_id,
            "success": success,
        }
        
        if error:
            log_data["error"] = error
        
        if success:
            self.logger.info("LLM request completed", **log_data)
        else:
            self.logger.error("LLM request failed", **log_data)


class RAGLogger:
    """
    Logger specifically for RAG operations.
    
    Provides structured logging for retrieval operations with relevance scores.
    """
    
    def __init__(self):
        """Initialize RAG logger."""
        self.logger = get_logger("rag")
    
    def log_retrieval(
        self,
        query: str,
        num_results: int,
        top_score: float,
        latency_ms: float,
        request_id: str,
    ) -> None:
        """
        Log a RAG retrieval operation.
        
        Args:
            query: Search query
            num_results: Number of results retrieved
            top_score: Highest similarity score
            latency_ms: Retrieval latency in milliseconds
            request_id: Unique request identifier
        """
        self.logger.info(
            "RAG retrieval completed",
            query_length=len(query),
            num_results=num_results,
            top_score=round(top_score, 4),
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )
    
    def log_embedding(
        self,
        num_texts: int,
        total_tokens: int,
        latency_ms: float,
        request_id: str,
    ) -> None:
        """
        Log an embedding generation operation.
        
        Args:
            num_texts: Number of texts embedded
            total_tokens: Total tokens processed
            latency_ms: Embedding latency in milliseconds
            request_id: Unique request identifier
        """
        self.logger.info(
            "Embedding generation completed",
            num_texts=num_texts,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )


# Initialize default logger on module import
setup_logger()
