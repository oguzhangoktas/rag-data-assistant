"""
Response Validator for LLM Outputs
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Validates and sanitizes LLM responses including SQL queries,
JSON outputs, and general text responses.
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger
from src.utils.exceptions import SQLValidationException, ValidationException

logger = get_logger("response_validator")


class ResponseType(str, Enum):
    """Types of LLM responses."""
    SQL = "sql"
    JSON = "json"
    TEXT = "text"
    CODE = "code"


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    response_type: ResponseType
    cleaned_response: str
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class ResponseValidator:
    """
    Validates and sanitizes LLM responses.
    
    Features:
    - SQL injection detection
    - JSON parsing and validation
    - Content sanitization
    - Format extraction (SQL from markdown)
    
    Usage:
        validator = ResponseValidator()
        result = validator.validate_sql(llm_response)
        if result.is_valid:
            safe_sql = result.cleaned_response
    """
    
    # SQL keywords that indicate dangerous operations
    DANGEROUS_SQL_KEYWORDS = [
        "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT",
        "ALTER", "CREATE", "GRANT", "REVOKE", "EXEC",
        "EXECUTE", "SHUTDOWN", "BACKUP", "RESTORE",
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r";\s*--",  # Comment injection
        r"'\s*OR\s+'",  # OR injection
        r"'\s*AND\s+'",  # AND injection
        r"UNION\s+SELECT",  # Union injection
        r"INTO\s+OUTFILE",  # File write
        r"LOAD_FILE",  # File read
        r"xp_cmdshell",  # Command execution
        r"sp_executesql",  # Dynamic SQL
    ]
    
    def __init__(self):
        """Initialize the response validator."""
        self._injection_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]
    
    def validate_sql(
        self,
        response: str,
        allow_multiple_statements: bool = False,
        max_length: int = 10000,
    ) -> ValidationResult:
        """
        Validate an SQL response from an LLM.
        
        Args:
            response: Raw LLM response containing SQL
            allow_multiple_statements: Whether to allow multiple SQL statements
            max_length: Maximum allowed SQL length
            
        Returns:
            ValidationResult with validation status and cleaned SQL
        """
        issues = []
        warnings = []
        metadata = {}
        
        # Extract SQL from response (may be in markdown code block)
        sql = self._extract_sql(response)
        metadata["original_length"] = len(response)
        metadata["extracted_length"] = len(sql)
        
        # Check if SQL was extracted
        if not sql.strip():
            return ValidationResult(
                is_valid=False,
                response_type=ResponseType.SQL,
                cleaned_response="",
                issues=["No SQL query found in response"],
                warnings=[],
                metadata=metadata,
            )
        
        # Length check
        if len(sql) > max_length:
            issues.append(f"SQL exceeds maximum length of {max_length} characters")
        
        # Check for dangerous keywords
        dangerous_found = self._check_dangerous_keywords(sql)
        if dangerous_found:
            issues.append(f"SQL contains dangerous keywords: {', '.join(dangerous_found)}")
            metadata["dangerous_keywords"] = dangerous_found
        
        # Check for SQL injection patterns
        injection_found = self._check_injection_patterns(sql)
        if injection_found:
            issues.append(f"SQL contains potential injection patterns")
            metadata["injection_patterns"] = injection_found
        
        # Check for multiple statements
        if not allow_multiple_statements and self._has_multiple_statements(sql):
            issues.append("Multiple SQL statements detected (not allowed)")
        
        # Check for SELECT statement
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            if sql_upper.startswith(("WITH", "(")):
                # CTEs and subqueries are OK
                pass
            else:
                warnings.append("Query does not start with SELECT")
        
        # Check for proper termination
        if not sql.strip().endswith(";"):
            warnings.append("SQL does not end with semicolon")
            sql = sql.strip() + ";"
        
        # Clean and normalize SQL
        cleaned_sql = self._clean_sql(sql)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            response_type=ResponseType.SQL,
            cleaned_response=cleaned_sql,
            issues=issues,
            warnings=warnings,
            metadata=metadata,
        )
    
    def validate_json(
        self,
        response: str,
        required_keys: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a JSON response from an LLM.
        
        Args:
            response: Raw LLM response containing JSON
            required_keys: Optional list of required keys
            schema: Optional JSON schema for validation
            
        Returns:
            ValidationResult with validation status and parsed JSON
        """
        issues = []
        warnings = []
        metadata = {}
        
        # Extract JSON from response
        json_str = self._extract_json(response)
        metadata["extracted_json"] = json_str[:100] if json_str else ""
        
        if not json_str:
            return ValidationResult(
                is_valid=False,
                response_type=ResponseType.JSON,
                cleaned_response="",
                issues=["No JSON found in response"],
                warnings=[],
                metadata=metadata,
            )
        
        # Try to parse JSON
        try:
            parsed = json.loads(json_str)
            metadata["parsed_type"] = type(parsed).__name__
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                response_type=ResponseType.JSON,
                cleaned_response=json_str,
                issues=[f"Invalid JSON: {str(e)}"],
                warnings=[],
                metadata=metadata,
            )
        
        # Check required keys
        if required_keys and isinstance(parsed, dict):
            missing_keys = [k for k in required_keys if k not in parsed]
            if missing_keys:
                issues.append(f"Missing required keys: {missing_keys}")
        
        # Re-serialize to ensure clean JSON
        cleaned_json = json.dumps(parsed, indent=2)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            response_type=ResponseType.JSON,
            cleaned_response=cleaned_json,
            issues=issues,
            warnings=warnings,
            metadata=metadata,
        )
    
    def validate_text(
        self,
        response: str,
        max_length: int = 10000,
        min_length: int = 1,
    ) -> ValidationResult:
        """
        Validate a text response from an LLM.
        
        Args:
            response: Raw LLM response
            max_length: Maximum allowed length
            min_length: Minimum required length
            
        Returns:
            ValidationResult with validation status
        """
        issues = []
        warnings = []
        metadata = {"length": len(response)}
        
        if len(response) < min_length:
            issues.append(f"Response too short (minimum {min_length} characters)")
        
        if len(response) > max_length:
            warnings.append(f"Response exceeds {max_length} characters, may be truncated")
        
        # Clean the response
        cleaned = self._clean_text(response)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            response_type=ResponseType.TEXT,
            cleaned_response=cleaned,
            issues=issues,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _extract_sql(self, text: str) -> str:
        """
        Extract SQL from text, handling markdown code blocks.
        
        Args:
            text: Text potentially containing SQL
            
        Returns:
            Extracted SQL string
        """
        # Try to extract from markdown code block
        sql_block_pattern = r"```(?:sql)?\s*([\s\S]*?)```"
        matches = re.findall(sql_block_pattern, text, re.IGNORECASE)
        
        if matches:
            # Return the first SQL block found
            return matches[0].strip()
        
        # If no code block, assume the entire response is SQL
        # but clean up any markdown artifacts
        text = text.strip()
        
        # Remove any leading/trailing backticks
        text = re.sub(r"^`+|`+$", "", text)
        
        return text.strip()
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text, handling markdown code blocks.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON string
        """
        # Try to extract from markdown code block
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, text, re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # Try to find JSON object or array
        json_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
        matches = re.findall(json_pattern, text)
        
        if matches:
            return matches[0].strip()
        
        return text.strip()
    
    def _check_dangerous_keywords(self, sql: str) -> List[str]:
        """
        Check for dangerous SQL keywords.
        
        Args:
            sql: SQL string to check
            
        Returns:
            List of dangerous keywords found
        """
        found = []
        sql_upper = sql.upper()
        
        for keyword in self.DANGEROUS_SQL_KEYWORDS:
            # Check for keyword as a whole word
            pattern = r"\b" + keyword + r"\b"
            if re.search(pattern, sql_upper):
                found.append(keyword)
        
        return found
    
    def _check_injection_patterns(self, sql: str) -> List[str]:
        """
        Check for SQL injection patterns.
        
        Args:
            sql: SQL string to check
            
        Returns:
            List of injection patterns found
        """
        found = []
        
        for i, pattern in enumerate(self._injection_patterns):
            if pattern.search(sql):
                found.append(self.SQL_INJECTION_PATTERNS[i])
        
        return found
    
    def _has_multiple_statements(self, sql: str) -> bool:
        """
        Check if SQL contains multiple statements.
        
        Args:
            sql: SQL string to check
            
        Returns:
            True if multiple statements detected
        """
        # Remove string literals to avoid false positives
        cleaned = re.sub(r"'[^']*'", "", sql)
        cleaned = re.sub(r'"[^"]*"', "", cleaned)
        
        # Count semicolons not at the end
        statements = [s.strip() for s in cleaned.split(";") if s.strip()]
        
        return len(statements) > 1
    
    def _clean_sql(self, sql: str) -> str:
        """
        Clean and normalize SQL string.
        
        Args:
            sql: SQL string to clean
            
        Returns:
            Cleaned SQL string
        """
        # Remove excessive whitespace
        sql = re.sub(r"\s+", " ", sql)
        
        # Ensure single semicolon at end
        sql = sql.rstrip(";").strip() + ";"
        
        return sql
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text response.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        
        return text
    
    def sanitize_for_display(self, text: str, max_length: int = 1000) -> str:
        """
        Sanitize text for safe display.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length
            
        Returns:
            Sanitized text
        """
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Remove any potential XSS
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        
        return text


def validate_sql_response(response: str) -> Tuple[bool, str, List[str]]:
    """
    Convenience function to validate SQL response.
    
    Args:
        response: LLM response containing SQL
        
    Returns:
        Tuple of (is_valid, cleaned_sql, issues)
    """
    validator = ResponseValidator()
    result = validator.validate_sql(response)
    return result.is_valid, result.cleaned_response, result.issues


def extract_sql(response: str) -> str:
    """
    Convenience function to extract SQL from response.
    
    Args:
        response: LLM response containing SQL
        
    Returns:
        Extracted SQL string
    """
    validator = ResponseValidator()
    result = validator.validate_sql(response)
    return result.cleaned_response
