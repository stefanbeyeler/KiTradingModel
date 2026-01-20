"""
Feedback Router - API endpoints for the feedback buffer.

Provides endpoints for:
- Viewing feedback buffer statistics
- Managing feedback samples
- Receiving feedback from TCN Inference Service
"""

from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from ..services.feedback_buffer_service import (
    feedback_buffer_service,
    FeedbackSample,
    BufferStatistics,
)

router = APIRouter(prefix="/feedback-buffer", tags=["3. Feedback Buffer"])


# ============================================================================
# Request Models
# ============================================================================

class OutcomeFeedbackRequest(BaseModel):
    """Request to add feedback from a pattern outcome."""

    pattern_id: str = Field(..., description="Unique pattern identifier")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Pattern timeframe")
    pattern_type: str = Field(..., description="Type of pattern")
    direction: str = Field(..., description="Pattern direction (bullish/bearish)")
    original_confidence: float = Field(..., ge=0, le=1, description="Original TCN confidence")
    outcome_status: str = Field(..., description="Outcome status: success, partial, failed, expired")
    outcome_score: float = Field(..., ge=-1, le=1, description="Outcome score from -1 to 1")
    max_favorable_percent: float = Field(default=0.0, description="Max favorable move %")
    max_adverse_percent: float = Field(default=0.0, description="Max adverse move %")
    ohlcv_data: List[dict] = Field(..., description="OHLCV candles used for detection")
    claude_validated: bool = Field(default=False, description="Whether Claude validated")
    claude_agreed: Optional[bool] = Field(default=None, description="Whether Claude agreed")
    claude_confidence: Optional[float] = Field(default=None, description="Claude's confidence")


class ManualFeedbackRequest(BaseModel):
    """Request to add manual feedback."""

    pattern_id: str = Field(..., description="Pattern identifier")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Pattern timeframe")
    pattern_type: str = Field(..., description="Type of pattern")
    direction: str = Field(..., description="Pattern direction")
    original_confidence: float = Field(..., ge=0, le=1, description="Original confidence")
    is_positive: bool = Field(..., description="Whether feedback is positive")
    ohlcv_data: List[dict] = Field(..., description="OHLCV data")
    reason: str = Field(default="", description="Reason for feedback")


# ============================================================================
# Response Models
# ============================================================================

class FeedbackSampleResponse(BaseModel):
    """Single feedback sample response."""

    sample_id: str
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str
    original_confidence: float
    feedback_type: str
    outcome_score: float
    sample_weight: float
    max_favorable_percent: float
    max_adverse_percent: float
    claude_validated: bool
    created_at: str
    used_for_training: bool


class FeedbackListResponse(BaseModel):
    """List of feedback samples response."""

    count: int
    samples: List[dict]


class BufferStatisticsResponse(BaseModel):
    """Buffer statistics response."""

    total_samples: int
    unused_samples: int
    used_samples: int
    by_feedback_type: dict
    by_pattern_type: dict
    positive_outcomes: int
    negative_outcomes: int
    neutral_outcomes: int
    oldest_sample_age_hours: float
    avg_sample_weight: float
    ready_for_training: bool
    min_samples_for_training: int
    last_update: Optional[str]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/statistics", response_model=BufferStatisticsResponse)
async def get_buffer_statistics():
    """
    Get feedback buffer statistics.

    Returns information about buffer size, sample distribution,
    and whether there are enough samples for training.
    """
    stats = feedback_buffer_service.get_statistics()
    return BufferStatisticsResponse(**stats.to_dict())


@router.get("/samples", response_model=FeedbackListResponse)
async def get_samples(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum samples to return"),
    unused_only: bool = Query(default=False, description="Only return unused samples"),
    pattern_type: Optional[str] = Query(default=None, description="Filter by pattern type"),
):
    """
    Get feedback samples from the buffer.
    """
    samples = feedback_buffer_service.get_samples(
        limit=limit,
        unused_only=unused_only,
        pattern_type=pattern_type,
    )

    return FeedbackListResponse(
        count=len(samples),
        samples=[s.to_dict() for s in samples]
    )


@router.post("/outcome")
async def add_outcome_feedback(request: OutcomeFeedbackRequest):
    """
    Add feedback from a completed pattern outcome.

    This endpoint is called by the TCN Inference Service when
    a pattern outcome is evaluated.
    """
    sample = await feedback_buffer_service.add_feedback_from_outcome(
        pattern_id=request.pattern_id,
        symbol=request.symbol,
        timeframe=request.timeframe,
        pattern_type=request.pattern_type,
        direction=request.direction,
        original_confidence=request.original_confidence,
        outcome_status=request.outcome_status,
        outcome_score=request.outcome_score,
        max_favorable_percent=request.max_favorable_percent,
        max_adverse_percent=request.max_adverse_percent,
        ohlcv_data=request.ohlcv_data,
        claude_validated=request.claude_validated,
        claude_agreed=request.claude_agreed,
        claude_confidence=request.claude_confidence,
    )

    if sample:
        return {
            "status": "ok",
            "message": "Feedback added successfully",
            "sample_id": sample.sample_id,
            "feedback_type": sample.feedback_type,
            "sample_weight": sample.sample_weight,
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Could not add feedback. Check pattern type and OHLCV data."
        )


@router.post("/manual")
async def add_manual_feedback(request: ManualFeedbackRequest):
    """
    Add manual feedback for a pattern.

    Use this to manually mark patterns as correct or incorrect.
    """
    sample = await feedback_buffer_service.add_manual_feedback(
        pattern_id=request.pattern_id,
        symbol=request.symbol,
        timeframe=request.timeframe,
        pattern_type=request.pattern_type,
        direction=request.direction,
        original_confidence=request.original_confidence,
        is_positive=request.is_positive,
        ohlcv_data=request.ohlcv_data,
        reason=request.reason,
    )

    if sample:
        return {
            "status": "ok",
            "message": "Manual feedback added",
            "sample_id": sample.sample_id,
            "feedback_type": sample.feedback_type,
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Could not add manual feedback. Check inputs."
        )


@router.post("/mark-used")
async def mark_samples_as_used(sample_ids: List[str]):
    """
    Mark samples as used for training.

    Call this after successfully using samples for incremental training.
    """
    marked = feedback_buffer_service.mark_as_used(sample_ids)

    return {
        "status": "ok",
        "message": f"Marked {marked} samples as used",
        "marked_count": marked,
    }


@router.delete("/clear-used")
async def clear_used_samples():
    """
    Remove all samples that have been used for training.
    """
    removed = feedback_buffer_service.clear_used_samples()

    return {
        "status": "ok",
        "message": f"Cleared {removed} used samples",
        "removed_count": removed,
    }


@router.delete("/clear-all")
async def clear_all_samples():
    """
    Clear all samples from the buffer.

    Warning: This removes all feedback data including unused samples!
    """
    removed = feedback_buffer_service.clear_all()

    return {
        "status": "ok",
        "message": f"Cleared all {removed} samples",
        "removed_count": removed,
    }


@router.get("/ready")
async def check_training_ready():
    """
    Check if buffer has enough samples for training.
    """
    stats = feedback_buffer_service.get_statistics()

    return {
        "ready": stats.ready_for_training,
        "unused_samples": stats.unused_samples,
        "min_required": stats.min_samples_for_training,
        "samples_needed": max(0, stats.min_samples_for_training - stats.unused_samples),
    }
