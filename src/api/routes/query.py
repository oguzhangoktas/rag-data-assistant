"""
Query API Routes
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Main endpoint for natural language queries.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.sql_generator.text_to_sql import get_text_to_sql
from src.sql_generator.sql_executor import SQLExecutor
from src.utils.logger import get_logger

logger = get_logger("api.query")
router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="Natural language question")
    execute: bool = Field(default=True, description="Whether to execute the generated SQL")
    stream: bool = Field(default=False, description="Whether to stream the response")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    question: str
    sql: str
    is_valid: bool
    explanation: str
    results: Optional[dict] = None
    sources: list = []
    metrics: dict = {}


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Process a natural language query and return SQL with results.
    
    Args:
        request: FastAPI request
        body: Query request body
        
    Returns:
        QueryResponse with SQL, results, and explanation
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"Processing query: {body.question[:100]}...")
    
    try:
        # Generate SQL
        text_to_sql = get_text_to_sql()
        result = await text_to_sql.generate_with_explanation(body.question)
        
        response = QueryResponse(
            question=body.question,
            sql=result.sql,
            is_valid=result.is_valid,
            explanation=result.explanation,
            sources=result.sources,
            metrics={
                "latency_ms": result.latency_ms,
                "tokens_used": result.tokens_used,
                "cost_usd": result.cost_usd,
            },
        )
        
        # Execute SQL if requested and valid
        if body.execute and result.is_valid:
            try:
                executor = SQLExecutor()
                exec_result = await executor.execute(result.sql)
                response.results = {
                    "columns": exec_result.columns,
                    "rows": exec_result.rows[:100],  # Limit for API response
                    "row_count": exec_result.row_count,
                    "execution_time_ms": exec_result.execution_time_ms,
                }
            except Exception as e:
                logger.warning(f"SQL execution failed: {e}")
                response.results = {"error": str(e)}
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/sql-only")
async def query_sql_only(request: Request, body: QueryRequest):
    """
    Generate SQL without execution.
    
    Args:
        request: FastAPI request
        body: Query request body
        
    Returns:
        Generated SQL query
    """
    try:
        text_to_sql = get_text_to_sql()
        result = await text_to_sql.generate(body.question)
        
        return {
            "question": body.question,
            "sql": result.sql,
            "is_valid": result.is_valid,
            "validation_issues": result.validation_issues,
        }
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
