"""
Database Connection Manager for PostgreSQL
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from src.utils.config_loader import get_settings
from src.utils.logger import get_logger
from src.utils.exceptions import SQLExecutionException

logger = get_logger("database")


class DatabaseManager:
    """
    Manages PostgreSQL database connections with both sync and async support.
    
    Features:
    - Connection pooling
    - Async and sync session management
    - Query execution with timeout
    - Schema introspection
    - Query history tracking
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        async_database_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
    ):
        """
        Initialize database manager.
        
        Args:
            database_url: Sync PostgreSQL connection URL
            async_database_url: Async PostgreSQL connection URL
            pool_size: Connection pool size
            max_overflow: Max connections beyond pool_size
            pool_timeout: Timeout waiting for connection from pool
        """
        settings = get_settings()
        
        self._database_url = database_url or settings.database_url
        self._async_database_url = async_database_url or settings.async_database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        
        # Sync engine and session factory
        self._engine = None
        self._session_factory = None
        
        # Async engine and session factory
        self._async_engine = None
        self._async_session_factory = None
        
        # Metadata for schema introspection
        self._metadata = MetaData()
        
        logger.info("DatabaseManager initialized")
    
    def _get_sync_engine(self):
        """Get or create sync database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self._database_url,
                poolclass=QueuePool,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_pre_ping=True,  # Verify connections before use
            )
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )
        return self._engine
    
    def _get_async_engine(self):
        """Get or create async database engine."""
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                self._async_database_url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_pre_ping=True,
            )
            self._async_session_factory = async_sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._async_engine
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Yields:
            AsyncSession for database operations
        """
        self._get_async_engine()
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    def get_sync_session(self) -> Session:
        """
        Get a sync database session.
        
        Returns:
            Session for database operations
        """
        self._get_sync_engine()
        return self._session_factory()
    
    async def execute_query_async(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 30,
        max_rows: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute a SQL query asynchronously with safety limits.
        
        Args:
            sql: SQL query to execute
            params: Optional query parameters
            timeout_seconds: Query timeout in seconds
            max_rows: Maximum rows to return
            
        Returns:
            Dictionary with columns, rows, row_count, and execution_time_ms
            
        Raises:
            SQLExecutionException: If query execution fails
        """
        import time
        start_time = time.time()
        
        try:
            async with self.get_async_session() as session:
                # Add LIMIT if not present and it's a SELECT query
                sql_upper = sql.strip().upper()
                if sql_upper.startswith("SELECT") and "LIMIT" not in sql_upper:
                    sql = f"{sql.rstrip(';')} LIMIT {max_rows}"
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    session.execute(text(sql), params or {}),
                    timeout=timeout_seconds,
                )
                
                # Fetch results
                rows = result.fetchall()
                columns = list(result.keys()) if result.keys() else []
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Convert rows to list of dicts
                data = [dict(zip(columns, row)) for row in rows]
                
                logger.info(
                    f"Query executed successfully",
                    row_count=len(data),
                    execution_time_ms=round(execution_time_ms, 2),
                )
                
                return {
                    "columns": columns,
                    "rows": data,
                    "row_count": len(data),
                    "execution_time_ms": round(execution_time_ms, 2),
                    "truncated": len(data) >= max_rows,
                }
                
        except asyncio.TimeoutError:
            raise SQLExecutionException(
                message=f"Query timed out after {timeout_seconds} seconds",
                sql=sql,
                details={"timeout_seconds": timeout_seconds},
            )
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise SQLExecutionException(
                message="Query execution failed",
                sql=sql,
                db_error=str(e),
            )
    
    def execute_query_sync(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000,
    ) -> Dict[str, Any]:
        """
        Execute a SQL query synchronously with safety limits.
        
        Args:
            sql: SQL query to execute
            params: Optional query parameters
            max_rows: Maximum rows to return
            
        Returns:
            Dictionary with columns, rows, row_count, and execution_time_ms
        """
        import time
        start_time = time.time()
        
        try:
            with self.get_sync_session() as session:
                # Add LIMIT if not present and it's a SELECT query
                sql_upper = sql.strip().upper()
                if sql_upper.startswith("SELECT") and "LIMIT" not in sql_upper:
                    sql = f"{sql.rstrip(';')} LIMIT {max_rows}"
                
                result = session.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = list(result.keys()) if result.keys() else []
                
                execution_time_ms = (time.time() - start_time) * 1000
                data = [dict(zip(columns, row)) for row in rows]
                
                return {
                    "columns": columns,
                    "rows": data,
                    "row_count": len(data),
                    "execution_time_ms": round(execution_time_ms, 2),
                    "truncated": len(data) >= max_rows,
                }
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise SQLExecutionException(
                message="Query execution failed",
                sql=sql,
                db_error=str(e),
            )
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get database schema information for SQL generation.
        
        Returns:
            Dictionary with tables, columns, and relationships
        """
        engine = self._get_sync_engine()
        inspector = inspect(engine)
        
        schema_info = {"tables": {}}
        
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column.get("nullable", True),
                })
            
            # Get primary keys
            pk = inspector.get_pk_constraint(table_name)
            primary_keys = pk.get("constrained_columns", []) if pk else []
            
            # Get foreign keys
            fks = inspector.get_foreign_keys(table_name)
            foreign_keys = [
                {
                    "column": fk["constrained_columns"][0] if fk["constrained_columns"] else None,
                    "references": f"{fk['referred_table']}.{fk['referred_columns'][0]}" 
                        if fk["referred_columns"] else None,
                }
                for fk in fks
            ]
            
            schema_info["tables"][table_name] = {
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            }
        
        logger.info(f"Retrieved schema info for {len(schema_info['tables'])} tables")
        return schema_info
    
    async def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible
        """
        try:
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Async database connections closed")
        
        if self._engine:
            self._engine.dispose()
            logger.info("Sync database connections closed")


# Global singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager singleton instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session.
    
    Yields:
        AsyncSession for database operations
    """
    db = get_database_manager()
    async with db.get_async_session() as session:
        yield session
