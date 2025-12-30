"""
Safe SQL Executor
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Executes SQL queries with safety limits and result formatting.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sql_generator.query_validator import QueryValidator
from src.utils.database import DatabaseManager, get_database_manager
from src.utils.logger import get_logger
from src.utils.exceptions import SQLExecutionException

logger = get_logger("sql_executor")


@dataclass
class ExecutionResult:
    """Result from SQL execution."""
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    truncated: bool
    query: str


class SQLExecutor:
    """
    Executes SQL queries safely with limits and validation.
    
    Features:
    - Query validation before execution
    - Row limit enforcement
    - Timeout protection
    - Result formatting
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        max_rows: int = 1000,
        timeout_seconds: int = 30,
    ):
        """
        Initialize the SQL executor.
        
        Args:
            db_manager: Database manager instance
            max_rows: Maximum rows to return
            timeout_seconds: Query timeout in seconds
        """
        self._db = db_manager or get_database_manager()
        self._validator = QueryValidator()
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        
        logger.info(f"SQLExecutor initialized: max_rows={max_rows}, timeout={timeout_seconds}s")
    
    async def execute(
        self,
        sql: str,
        validate: bool = True,
    ) -> ExecutionResult:
        """
        Execute a SQL query safely.
        
        Args:
            sql: SQL query to execute
            validate: Whether to validate before execution
            
        Returns:
            ExecutionResult with data
        """
        # Validate query
        if validate:
            validation = self._validator.validate(sql)
            if not validation.is_valid:
                raise SQLExecutionException(
                    message=f"Query validation failed: {', '.join(validation.issues)}",
                    sql=sql,
                )
        
        # Execute query
        result = await self._db.execute_query_async(
            sql=sql,
            timeout_seconds=self.timeout_seconds,
            max_rows=self.max_rows,
        )
        
        return ExecutionResult(
            columns=result["columns"],
            rows=result["rows"],
            row_count=result["row_count"],
            execution_time_ms=result["execution_time_ms"],
            truncated=result.get("truncated", False),
            query=sql,
        )
    
    def execute_sync(self, sql: str, validate: bool = True) -> ExecutionResult:
        """Synchronous version of execute."""
        if validate:
            validation = self._validator.validate(sql)
            if not validation.is_valid:
                raise SQLExecutionException(
                    message=f"Query validation failed: {', '.join(validation.issues)}",
                    sql=sql,
                )
        
        result = self._db.execute_query_sync(sql=sql, max_rows=self.max_rows)
        
        return ExecutionResult(
            columns=result["columns"],
            rows=result["rows"],
            row_count=result["row_count"],
            execution_time_ms=result["execution_time_ms"],
            truncated=result.get("truncated", False),
            query=sql,
        )
    
    def format_results_for_llm(
        self,
        result: ExecutionResult,
        max_rows: int = 20,
    ) -> str:
        """
        Format execution results for LLM context.
        
        Args:
            result: Execution result
            max_rows: Max rows to include
            
        Returns:
            Formatted string for LLM
        """
        lines = [
            f"Query Results ({result.row_count} rows, {result.execution_time_ms:.2f}ms):",
            f"Columns: {', '.join(result.columns)}",
            "",
        ]
        
        for i, row in enumerate(result.rows[:max_rows]):
            row_str = " | ".join(str(v)[:50] for v in row.values())
            lines.append(f"{i+1}. {row_str}")
        
        if result.row_count > max_rows:
            lines.append(f"... and {result.row_count - max_rows} more rows")
        
        if result.truncated:
            lines.append(f"[Results truncated to {self.max_rows} rows]")
        
        return "\n".join(lines)
