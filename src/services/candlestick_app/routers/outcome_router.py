"""
Outcome Tracker Router for Candlestick Patterns.

API endpoints for tracking and querying pattern outcomes.
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from loguru import logger

from ..services.outcome_tracker_service import outcome_tracker_service


router = APIRouter(prefix="/outcomes", tags=["5. Outcome Tracking"])


class TrackPatternRequest(BaseModel):
    """Request to track a pattern."""
    id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str
    category: str
    strength: str = "moderate"
    confidence: float = 0.5
    price_at_detection: float = 0.0
    timestamp: str = ""
    ohlc_context: Optional[dict] = None


@router.get("/", summary="Get outcomes with filters")
async def get_outcomes(
    status: Optional[str] = Query(None, description="Filter by status (pending, success, partial, failed, expired)"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """
    Get tracked pattern outcomes with optional filters.
    """
    return outcome_tracker_service.get_outcomes(
        status=status,
        symbol=symbol,
        pattern_type=pattern_type,
        limit=limit
    )


@router.get("/active", summary="Get pending outcomes")
async def get_active_outcomes():
    """Get all outcomes currently being tracked (pending status)."""
    return outcome_tracker_service.get_outcomes(status="pending", limit=500)


@router.get("/completed", summary="Get completed outcomes")
async def get_completed_outcomes(
    limit: int = Query(100, ge=1, le=500)
):
    """Get all completed outcomes (not pending)."""
    all_outcomes = outcome_tracker_service.get_outcomes(limit=500)
    return [o for o in all_outcomes if o.get("status") != "pending"][:limit]


@router.get("/statistics", summary="Get outcome statistics")
async def get_statistics():
    """
    Get aggregated statistics about pattern outcomes.

    Returns success rates by pattern type, direction, and overall metrics.
    """
    return outcome_tracker_service.get_statistics()


@router.post("/track", summary="Start tracking a pattern")
async def track_pattern(request: TrackPatternRequest):
    """
    Start tracking a detected pattern for outcome evaluation.

    The pattern will be monitored for the evaluation period based on its timeframe.
    """
    pattern_data = request.model_dump()
    outcome = await outcome_tracker_service.track_pattern(pattern_data)

    if outcome:
        return {
            "status": "tracking",
            "pattern_id": outcome.pattern_id,
            "tracking_ends": outcome.tracking_ends
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to start tracking")


@router.post("/update", summary="Manually trigger outcome update")
async def trigger_update():
    """Manually trigger an update of all pending outcomes."""
    await outcome_tracker_service.update_outcomes()
    return {"status": "updated"}


@router.post("/loop/start", summary="Start background update loop")
async def start_loop():
    """Start the background loop that updates outcomes periodically."""
    return await outcome_tracker_service.start_loop()


@router.post("/loop/stop", summary="Stop background update loop")
async def stop_loop():
    """Stop the background update loop."""
    return await outcome_tracker_service.stop_loop()


@router.get("/loop/status", summary="Get loop status")
async def get_loop_status():
    """Get the status of the background update loop."""
    return {
        "running": outcome_tracker_service.is_running()
    }


@router.delete("/completed", summary="Clear completed outcomes")
async def clear_completed():
    """Clear all completed outcomes from tracking."""
    outcome_tracker_service.clear_completed()
    return {"status": "cleared"}
