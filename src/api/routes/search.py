"""
Search API Routes
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

logger = get_logger("api.search")
router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    min_score: float = Field(default=0.5, ge=0, le=1, description="Minimum score")


class SearchResult(BaseModel):
    """Single search result."""
    content: str
    source: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    total_results: int
    latency_ms: float


@router.post("/search", response_model=SearchResponse)
async def search_documents(body: SearchRequest):
    """Search documentation using semantic search."""
    try:
        retriever = get_retriever()
        result = await retriever.retrieve(
            query=body.query,
            top_k=body.top_k,
            min_score=body.min_score,
        )
        
        return SearchResponse(
            query=body.query,
            results=[
                SearchResult(
                    content=doc.content[:500],
                    source=doc.metadata.get("source", "Unknown"),
                    score=doc.score,
                    metadata=doc.metadata,
                )
                for doc in result.documents
            ],
            total_results=result.total_results,
            latency_ms=result.latency_ms,
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
