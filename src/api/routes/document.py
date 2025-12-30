"""
Document API Routes
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.rag.indexer import DocumentIndexer, DocumentSource
from src.utils.logger import get_logger

logger = get_logger("api.document")
router = APIRouter()


class DocumentUploadRequest(BaseModel):
    """Request for document upload."""
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source name")
    title: Optional[str] = Field(None, description="Document title")
    document_type: Optional[str] = Field(None, description="Document type")


class DocumentUploadResponse(BaseModel):
    """Response from document upload."""
    documents_processed: int
    chunks_created: int
    chunks_indexed: int
    errors: List[str]


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(body: DocumentUploadRequest):
    """Upload and index a document."""
    try:
        indexer = DocumentIndexer()
        doc = DocumentSource(
            content=body.content,
            source=body.source,
            title=body.title,
            document_type=body.document_type,
        )
        
        result = await indexer.index_documents([doc])
        
        return DocumentUploadResponse(
            documents_processed=result.documents_processed,
            chunks_created=result.chunks_created,
            chunks_indexed=result.chunks_indexed,
            errors=result.errors,
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/stats")
async def get_document_stats():
    """Get document indexing statistics."""
    try:
        indexer = DocumentIndexer()
        return indexer.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
