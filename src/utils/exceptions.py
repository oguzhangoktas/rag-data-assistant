"""
Custom Exception Classes for RAG Data Assistant
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """
    Base exception class for all RAG Data Assistant errors.
    
    All custom exceptions inherit from this class for consistent error handling.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional machine-readable error code
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "RAG_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message={self.message})"


class LLMException(RAGException):
    """Exception raised for LLM-related errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM exception.
        
        Args:
            message: Error message
            provider: LLM provider name (openai, anthropic)
            model: Model name that caused the error
            error_code: Error code
            details: Additional details
        """
        details = details or {}
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        
        super().__init__(
            message=message,
            error_code=error_code or "LLM_ERROR",
            details=details,
        )
        self.provider = provider
        self.model = model


class LLMRateLimitException(LLMException):
    """Exception raised when LLM rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize rate limit exception.
        
        Args:
            message: Error message
            provider: LLM provider name
            retry_after: Seconds to wait before retrying
        """
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            provider=provider,
            error_code="LLM_RATE_LIMIT",
            details=details,
        )
        self.retry_after = retry_after


class LLMTimeoutException(LLMException):
    """Exception raised when LLM request times out."""
    
    def __init__(
        self,
        message: str = "LLM request timed out",
        provider: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize timeout exception.
        
        Args:
            message: Error message
            provider: LLM provider name
            timeout_seconds: Configured timeout value
        """
        details = kwargs.get("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=message,
            provider=provider,
            error_code="LLM_TIMEOUT",
            details=details,
        )
        self.timeout_seconds = timeout_seconds


class LLMAuthenticationException(LLMException):
    """Exception raised when LLM authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize authentication exception.
        
        Args:
            message: Error message
            provider: LLM provider name
        """
        super().__init__(
            message=message,
            provider=provider,
            error_code="LLM_AUTH_ERROR",
            details=kwargs.get("details"),
        )


class SQLGenerationException(RAGException):
    """Exception raised for SQL generation errors."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        generated_sql: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SQL generation exception.
        
        Args:
            message: Error message
            query: Original natural language query
            generated_sql: SQL that was generated (if any)
            error_code: Error code
            details: Additional details
        """
        details = details or {}
        if query:
            details["original_query"] = query
        if generated_sql:
            details["generated_sql"] = generated_sql
        
        super().__init__(
            message=message,
            error_code=error_code or "SQL_GENERATION_ERROR",
            details=details,
        )
        self.query = query
        self.generated_sql = generated_sql


class SQLValidationException(SQLGenerationException):
    """Exception raised when generated SQL fails validation."""
    
    def __init__(
        self,
        message: str,
        sql: str,
        validation_errors: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize SQL validation exception.
        
        Args:
            message: Error message
            sql: SQL that failed validation
            validation_errors: List of specific validation errors
        """
        details = kwargs.get("details", {})
        if validation_errors:
            details["validation_errors"] = validation_errors
        
        super().__init__(
            message=message,
            generated_sql=sql,
            error_code="SQL_VALIDATION_ERROR",
            details=details,
        )
        self.validation_errors = validation_errors or []


class SQLExecutionException(SQLGenerationException):
    """Exception raised when SQL execution fails."""
    
    def __init__(
        self,
        message: str,
        sql: str,
        db_error: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize SQL execution exception.
        
        Args:
            message: Error message
            sql: SQL that failed to execute
            db_error: Database error message
        """
        details = kwargs.get("details", {})
        if db_error:
            details["database_error"] = db_error
        
        super().__init__(
            message=message,
            generated_sql=sql,
            error_code="SQL_EXECUTION_ERROR",
            details=details,
        )
        self.db_error = db_error


class SQLSecurityException(SQLGenerationException):
    """Exception raised when SQL contains security risks."""
    
    def __init__(
        self,
        message: str = "SQL contains potentially unsafe operations",
        sql: Optional[str] = None,
        security_issues: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize SQL security exception.
        
        Args:
            message: Error message
            sql: SQL with security issues
            security_issues: List of security concerns
        """
        details = kwargs.get("details", {})
        if security_issues:
            details["security_issues"] = security_issues
        
        super().__init__(
            message=message,
            generated_sql=sql,
            error_code="SQL_SECURITY_ERROR",
            details=details,
        )
        self.security_issues = security_issues or []


class VectorStoreException(RAGException):
    """Exception raised for vector store operations."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        collection: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize vector store exception.
        
        Args:
            message: Error message
            operation: Operation that failed (insert, query, delete)
            collection: Collection name
            error_code: Error code
            details: Additional details
        """
        details = details or {}
        if operation:
            details["operation"] = operation
        if collection:
            details["collection"] = collection
        
        super().__init__(
            message=message,
            error_code=error_code or "VECTOR_STORE_ERROR",
            details=details,
        )
        self.operation = operation
        self.collection = collection


class EmbeddingException(VectorStoreException):
    """Exception raised for embedding generation errors."""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize embedding exception.
        
        Args:
            message: Error message
            model: Embedding model name
        """
        details = kwargs.get("details", {})
        if model:
            details["embedding_model"] = model
        
        super().__init__(
            message=message,
            operation="embedding",
            error_code="EMBEDDING_ERROR",
            details=details,
        )
        self.model = model


class RetrievalException(VectorStoreException):
    """Exception raised for retrieval/search errors."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize retrieval exception.
        
        Args:
            message: Error message
            query: Search query
        """
        details = kwargs.get("details", {})
        if query:
            details["search_query"] = query
        
        super().__init__(
            message=message,
            operation="retrieval",
            error_code="RETRIEVAL_ERROR",
            details=details,
        )
        self.query = query


class ConfigurationException(RAGException):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize configuration exception.
        
        Args:
            message: Error message
            config_file: Configuration file with the error
            config_key: Specific configuration key that's invalid
        """
        details = kwargs.get("details", {})
        if config_file:
            details["config_file"] = config_file
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
        )
        self.config_file = config_file
        self.config_key = config_key


class DocumentIngestionException(RAGException):
    """Exception raised during document ingestion."""
    
    def __init__(
        self,
        message: str,
        document_path: Optional[str] = None,
        document_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize document ingestion exception.
        
        Args:
            message: Error message
            document_path: Path to the document
            document_type: Type of document (pdf, markdown, etc.)
        """
        details = kwargs.get("details", {})
        if document_path:
            details["document_path"] = document_path
        if document_type:
            details["document_type"] = document_type
        
        super().__init__(
            message=message,
            error_code="DOCUMENT_INGESTION_ERROR",
            details=details,
        )
        self.document_path = document_path
        self.document_type = document_type


class CacheException(RAGException):
    """Exception raised for cache operations."""
    
    def __init__(
        self,
        message: str,
        cache_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize cache exception.
        
        Args:
            message: Error message
            cache_type: Type of cache (redis, semantic, embedding)
            operation: Cache operation that failed
        """
        details = kwargs.get("details", {})
        if cache_type:
            details["cache_type"] = cache_type
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details,
        )
        self.cache_type = cache_type
        self.operation = operation


class RateLimitException(RAGException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        period_seconds: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize rate limit exception.
        
        Args:
            message: Error message
            limit: Rate limit value
            period_seconds: Rate limit period
            retry_after: Seconds to wait before retrying
        """
        details = kwargs.get("details", {})
        if limit:
            details["limit"] = limit
        if period_seconds:
            details["period_seconds"] = period_seconds
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
        )
        self.limit = limit
        self.period_seconds = period_seconds
        self.retry_after = retry_after


class ValidationException(RAGException):
    """Exception raised for input validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize validation exception.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
        """
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)[:100]  # Truncate long values
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
        )
        self.field = field
        self.value = value
