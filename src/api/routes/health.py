"""
Health Check API Routes
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "api": "up",
            "database": "up",
            "vector_store": "up",
        },
    )


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RAG Data Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
    }
