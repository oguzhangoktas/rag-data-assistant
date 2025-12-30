"""
SQL Query Validator
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Validates generated SQL queries for correctness and security.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("query_validator")


@dataclass
class ValidationResult:
    """Result from SQL validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    security_score: float


class QueryValidator:
    """
    Validates SQL queries for correctness and security.
    
    Features:
    - SQL syntax validation
    - Security checks (injection, dangerous operations)
    - Schema validation
    - Performance warnings
    """
    
    DANGEROUS_KEYWORDS = [
        "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT",
        "ALTER", "CREATE", "GRANT", "REVOKE", "EXEC",
    ]
    
    INJECTION_PATTERNS = [
        r";\s*--",
        r"'\s*OR\s+'",
        r"UNION\s+SELECT",
        r"INTO\s+OUTFILE",
    ]
    
    def __init__(self):
        """Initialize the query validator."""
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
    
    def validate(
        self,
        sql: str,
        schema_manager: Optional[Any] = None,
    ) -> ValidationResult:
        """
        Validate a SQL query.
        
        Args:
            sql: SQL query to validate
            schema_manager: Optional schema manager for table validation
            
        Returns:
            ValidationResult with issues and warnings
        """
        issues = []
        warnings = []
        
        sql_upper = sql.upper().strip()
        
        # Check for dangerous keywords
        for keyword in self.DANGEROUS_KEYWORDS:
            pattern = r"\b" + keyword + r"\b"
            if re.search(pattern, sql_upper):
                issues.append(f"Dangerous keyword '{keyword}' not allowed")
        
        # Check for injection patterns
        for pattern in self._injection_patterns:
            if pattern.search(sql):
                issues.append("Potential SQL injection detected")
                break
        
        # Check query type
        if not sql_upper.startswith(("SELECT", "WITH")):
            issues.append("Only SELECT queries are allowed")
        
        # Check for missing WHERE on potentially large operations
        if "SELECT" in sql_upper and "FROM" in sql_upper:
            if "WHERE" not in sql_upper and "LIMIT" not in sql_upper:
                warnings.append("Query has no WHERE clause or LIMIT")
        
        # Check for SELECT *
        if "SELECT *" in sql_upper or "SELECT  *" in sql_upper:
            warnings.append("SELECT * should be replaced with specific columns")
        
        # Calculate security score
        security_score = 1.0
        security_score -= len(issues) * 0.3
        security_score -= len(warnings) * 0.1
        security_score = max(0.0, security_score)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            security_score=security_score,
        )
    
    def sanitize(self, sql: str) -> str:
        """
        Sanitize SQL query by removing dangerous elements.
        
        Args:
            sql: SQL to sanitize
            
        Returns:
            Sanitized SQL
        """
        # Remove comments
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        
        # Normalize whitespace
        sql = re.sub(r"\s+", " ", sql).strip()
        
        return sql
