"""
FastAPI Application - Main Entry Point
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)

Main FastAPI application with all routes and middleware.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import query, document, feedback, search, health
from src.utils.config_loader import get_settings
from src.utils.logger import setup_logger, get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Data Assistant API...")
    setup_logger(level=get_settings().log_level)
    yield
    # Shutdown
    logger.info("Shutting down RAG Data Assistant API...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title="RAG Data Assistant API",
        description="Enterprise LLM-Powered Data Analytics Platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))
        
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms",
        )
        
        return response
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(document.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "An internal error occurred",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
