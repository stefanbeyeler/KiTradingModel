"""
Feedback Buffer Router for CNN-LSTM Training Service.

API endpoints for the feedback buffer used in self-learning.
"""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel
from loguru import logger

from ..services.feedback_buffer_service import feedback_buffer_service


router = APIRouter(prefix="/feedback-buffer", tags=["3. Feedback Buffer"])


class OutcomeFeedbackRequest(BaseModel):
    """Request to add outcome feedback."""
    prediction_id: str
    symbol: str
    timeframe: str
    status: str
    overall_accuracy: float = 0.0
    task_accuracies: Optional[dict] = None
    original_predictions: Optional[dict] = None
    price_outcome: Optional[dict] = None
    pattern_outcome: Optional[dict] = None
    regime_outcome: Optional[dict] = None
    ohlcv_context: Optional[list] = None
    price_at_prediction: float = 0.0
    final_price: float = 0.0
    max_favorable_move: float = 0.0
    max_adverse_move: float = 0.0


class UserFeedbackRequest(BaseModel):
    """Request to add user feedback."""
    prediction_id: str
    symbol: str
    timeframe: str
    feedback_type: str  # confirmed, corrected, rejected
    price_correction: Optional[dict] = None
    pattern_correction: Optional[List[str]] = None
    regime_correction: Optional[str] = None
    ohlcv_context: Optional[list] = None


@router.get("/statistics", summary="Get buffer statistics")
async def get_statistics():
    """
    Get statistics about the feedback buffer.

    Returns sample counts, task coverage, and accuracy metrics.
    """
    stats = feedback_buffer_service.get_statistics()
    return {
        "total_samples": stats.total_samples,
        "unused_samples": stats.unused_samples,
        "ready_for_training": stats.ready_for_training,
        "by_feedback_type": stats.by_feedback_type,
        "by_symbol": stats.by_symbol,
        "by_timeframe": stats.by_timeframe,
        "samples_with_price_label": stats.samples_with_price_label,
        "samples_with_pattern_label": stats.samples_with_pattern_label,
        "samples_with_regime_label": stats.samples_with_regime_label,
        "average_accuracy": round(stats.average_accuracy, 3),
        "task_averages": stats.task_averages,
        "oldest_sample": stats.oldest_sample,
        "newest_sample": stats.newest_sample,
    }


@router.get("/samples", summary="Get samples from buffer")
async def get_samples(
    unused_only: bool = Query(False, description="Only return unused samples"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum samples to return")
):
    """Get samples from the feedback buffer."""
    return feedback_buffer_service.get_samples(unused_only=unused_only, limit=limit)


@router.post("/outcome", summary="Add outcome feedback")
async def add_outcome(request: OutcomeFeedbackRequest):
    """
    Add feedback from a prediction outcome evaluation.

    This is typically called by the Outcome Tracker service.
    """
    outcome_data = request.model_dump()
    sample = feedback_buffer_service.add_outcome_feedback(outcome_data)

    if sample:
        return {
            "status": "added",
            "sample_id": sample.sample_id,
            "feedback_type": sample.feedback_type.value
        }
    else:
        return {"status": "failed", "message": "Could not add feedback"}


@router.post("/user", summary="Add user feedback")
async def add_user_feedback(request: UserFeedbackRequest):
    """
    Add feedback from user input.

    Used when users manually confirm, correct, or reject predictions.
    """
    sample = feedback_buffer_service.add_user_feedback(
        prediction_id=request.prediction_id,
        symbol=request.symbol,
        timeframe=request.timeframe,
        feedback_type=request.feedback_type,
        price_correction=request.price_correction,
        pattern_correction=request.pattern_correction,
        regime_correction=request.regime_correction,
        ohlcv_context=request.ohlcv_context,
    )

    if sample:
        return {
            "status": "added",
            "sample_id": sample.sample_id,
            "feedback_type": sample.feedback_type.value
        }
    else:
        return {"status": "failed", "message": "Could not add feedback"}


@router.post("/mark-used", summary="Mark samples as used")
async def mark_used(sample_ids: List[str]):
    """Mark specific samples as used for training."""
    feedback_buffer_service.mark_as_used(sample_ids)
    return {"status": "marked", "count": len(sample_ids)}


@router.delete("/clear-used", summary="Clear used samples")
async def clear_used():
    """Clear all samples that have been used for training."""
    feedback_buffer_service.clear_used_samples()
    return {"status": "cleared"}


@router.get("/ready", summary="Check if ready for training")
async def check_ready():
    """Check if the buffer has enough samples for training."""
    stats = feedback_buffer_service.get_statistics()
    return {
        "ready": stats.ready_for_training,
        "unused_samples": stats.unused_samples,
        "required_samples": 100  # Default min samples
    }
