"""
Feedback API Routes
Author: Oguzhan Goktas (oguzhangoktas22@gmail.com)
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger("api.feedback")
router = APIRouter()


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    query_id: str = Field(..., description="Query ID to provide feedback for")
    score: int = Field(..., ge=1, le=5, description="Feedback score 1-5")
    sql_correction: Optional[str] = Field(None, description="Corrected SQL if any")
    comment: Optional[str] = Field(None, description="Additional comment")


class FeedbackResponse(BaseModel):
    """Feedback response model."""
    success: bool
    message: str


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest):
    """Submit feedback for a query."""
    try:
        logger.info(f"Received feedback for query {body.query_id}: score={body.score}")
        
        # In production, this would store feedback in database
        # and potentially update few-shot examples
        
        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
        )
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
