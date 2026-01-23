"""
Feedback Buffer Router for Candlestick Self-Learning.

API endpoints for managing feedback samples.
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from ..services.feedback_buffer_service import feedback_buffer_service


router = APIRouter(prefix="/feedback-buffer", tags=["3. Feedback Buffer"])


class OutcomeFeedbackRequest(BaseModel):
    """Request to add outcome feedback."""
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str = "neutral"
    category: str = "indecision"
    strength: str = "moderate"
    confidence: float = 0.5
    outcome_status: str  # success, partial, failed, expired
    max_favorable_percent: float = 0.0
    max_adverse_percent: float = 0.0
    final_move_percent: float = 0.0
    ohlc_data: List[Dict[str, Any]] = Field(default_factory=list)
    detection_time: str = ""
    completion_time: str = ""


class ClaudeFeedbackRequest(BaseModel):
    """Request to add Claude validation feedback."""
    id: str
    pattern_type: str
    symbol: str = ""
    timeframe: str = "H1"
    direction: str = "neutral"
    is_confirmed: bool
    confidence: float = 0.8
    ohlc_context: Optional[Dict[str, Any]] = None


@router.get("/statistics", summary="Get buffer statistics")
async def get_statistics():
    """
    Get statistics about the feedback buffer.

    Returns sample counts, balance, and readiness for training.
    """
    stats = feedback_buffer_service.get_statistics()
    return {
        "total_samples": stats.total_samples,
        "unused_samples": stats.unused_samples,
        "samples_by_type": stats.samples_by_type,
        "samples_by_pattern": stats.samples_by_pattern,
        "positive_samples": stats.positive_samples,
        "negative_samples": stats.negative_samples,
        "neutral_samples": stats.neutral_samples,
        "oldest_sample_age_hours": stats.oldest_sample_age_hours,
        "average_weight": stats.average_weight,
        "ready_for_training": stats.ready_for_training,
        "min_samples_required": stats.min_samples_required,
    }


@router.get("/samples", summary="Get feedback samples")
async def get_samples(
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    limit: int = Query(100, ge=1, le=500, description="Maximum samples to return")
):
    """Get feedback samples with optional filtering."""
    samples = []
    for sample in list(feedback_buffer_service._samples.values())[:limit]:
        if pattern_type and sample.pattern_type != pattern_type:
            continue
        samples.append(sample.to_dict())

    return {"samples": samples, "count": len(samples)}


@router.post("/outcome", summary="Add outcome feedback")
async def add_outcome_feedback(request: OutcomeFeedbackRequest):
    """
    Add feedback from a completed pattern outcome.

    Called by the outcome tracker when a pattern's tracking period ends.
    """
    sample_id = feedback_buffer_service.add_feedback_from_outcome(request.model_dump())

    if sample_id:
        return {"status": "added", "sample_id": sample_id}
    else:
        raise HTTPException(status_code=400, detail="Failed to add feedback")


@router.post("/claude", summary="Add Claude validation feedback")
async def add_claude_feedback(request: ClaudeFeedbackRequest):
    """
    Add feedback from Claude validation.

    Called when Claude validates or rejects a pattern.
    """
    sample_id = feedback_buffer_service.add_claude_feedback(
        pattern_data=request.model_dump(),
        is_confirmed=request.is_confirmed,
        confidence=request.confidence
    )

    if sample_id:
        return {"status": "added", "sample_id": sample_id}
    else:
        raise HTTPException(status_code=400, detail="Failed to add Claude feedback")


@router.post("/mark-used", summary="Mark samples as used")
async def mark_samples_used(sample_ids: List[str]):
    """Mark samples as used in training."""
    feedback_buffer_service.mark_as_used(sample_ids)
    return {"status": "marked", "count": len(sample_ids)}


@router.delete("/clear-used", summary="Clear used samples")
async def clear_used_samples():
    """Remove all samples that have been used in training."""
    feedback_buffer_service.clear_used_samples()
    return {"status": "cleared"}


@router.delete("/clear-all", summary="Clear all samples")
async def clear_all_samples():
    """Remove all samples from the buffer."""
    feedback_buffer_service.clear_all()
    return {"status": "cleared"}


@router.get("/ready", summary="Check if ready for training")
async def check_ready():
    """Check if the buffer has enough samples for training."""
    return {
        "ready": feedback_buffer_service.is_ready_for_training(),
        "stats": feedback_buffer_service.get_statistics().__dict__
    }
