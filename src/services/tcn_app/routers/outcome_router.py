"""
Outcome Router - API endpoints for pattern outcome tracking.

Provides endpoints for:
- Viewing active and completed outcomes
- Getting outcome statistics
- Manual outcome evaluation
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from loguru import logger

from ..services.outcome_tracker_service import (
    outcome_tracker_service,
    PatternOutcome,
    OutcomeStatistics,
    OutcomeStatus,
)

router = APIRouter(prefix="/outcomes", tags=["6. Outcome Tracking"])


# ============================================================================
# Response Models
# ============================================================================

class OutcomeResponse(BaseModel):
    """Single outcome response."""
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str
    category: str
    detected_at: str
    price_at_detection: float
    confidence: float
    price_target: Optional[float]
    invalidation_level: Optional[float]
    status: str
    tracking_started: Optional[str]
    tracking_ends: Optional[str]
    current_price: Optional[float]
    max_favorable_percent: float
    max_adverse_percent: float
    current_move_percent: float
    target_reached: bool
    invalidation_hit: bool
    outcome_reason: str
    last_update: Optional[str]


class OutcomeListResponse(BaseModel):
    """List of outcomes response."""
    count: int
    outcomes: list[dict]


class OutcomeStatisticsResponse(BaseModel):
    """Statistics response."""
    total_tracked: int
    total_completed: int
    pending: int
    success_count: int
    partial_count: int
    failed_count: int
    invalidated_count: int
    expired_count: int
    success_rate: float
    partial_rate: float
    failure_rate: float
    avg_favorable_percent: float
    avg_adverse_percent: float
    profit_factor: float
    by_pattern_type: dict
    by_direction: dict
    by_timeframe: dict
    last_update: Optional[str]


class UpdateLoopStatusResponse(BaseModel):
    """Update loop status response."""
    running: bool
    update_interval_seconds: int
    active_outcomes: int


# ============================================================================
# Endpoints
# ============================================================================

@router.get("", response_model=OutcomeListResponse)
async def get_outcomes(
    status: Optional[str] = Query(None, description="Filter by status: pending, success, partial, failed, invalidated, expired"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of results"),
):
    """
    Get pattern outcomes.

    Returns active (pending) outcomes first, then completed outcomes.
    """
    outcomes = []

    # Get active outcomes
    active = outcome_tracker_service.get_active_outcomes(symbol=symbol, limit=limit)

    if status is None or status == "pending":
        outcomes.extend([o.to_dict() for o in active])

    # Get completed outcomes
    if status != "pending":
        completed = outcome_tracker_service.get_completed_outcomes(
            symbol=symbol,
            status=status if status and status != "pending" else None,
            pattern_type=pattern_type,
            limit=limit - len(outcomes) if limit > len(outcomes) else 0
        )
        outcomes.extend([o.to_dict() for o in completed])

    return OutcomeListResponse(
        count=len(outcomes),
        outcomes=outcomes[:limit]
    )


@router.get("/active", response_model=OutcomeListResponse)
async def get_active_outcomes(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of results"),
):
    """
    Get active (pending) outcomes being tracked.
    """
    outcomes = outcome_tracker_service.get_active_outcomes(symbol=symbol, limit=limit)

    return OutcomeListResponse(
        count=len(outcomes),
        outcomes=[o.to_dict() for o in outcomes]
    )


@router.get("/completed", response_model=OutcomeListResponse)
async def get_completed_outcomes(
    status: Optional[str] = Query(None, description="Filter by status: success, partial, failed, invalidated, expired"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of results"),
):
    """
    Get completed outcomes.
    """
    # Validate status
    valid_statuses = ["success", "partial", "failed", "invalidated", "expired"]
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    outcomes = outcome_tracker_service.get_completed_outcomes(
        symbol=symbol,
        status=status,
        pattern_type=pattern_type,
        limit=limit
    )

    return OutcomeListResponse(
        count=len(outcomes),
        outcomes=[o.to_dict() for o in outcomes]
    )


@router.get("/statistics", response_model=OutcomeStatisticsResponse)
async def get_outcome_statistics():
    """
    Get aggregated outcome statistics.

    Returns success rates, average movements, and breakdowns by pattern type,
    direction, and timeframe.
    """
    stats = outcome_tracker_service.get_statistics()
    return OutcomeStatisticsResponse(**stats.to_dict())


@router.get("/{pattern_id}", response_model=OutcomeResponse)
async def get_outcome_by_id(pattern_id: str):
    """
    Get a specific outcome by pattern ID.
    """
    outcome = outcome_tracker_service.get_outcome(pattern_id)

    if not outcome:
        raise HTTPException(
            status_code=404,
            detail=f"Outcome not found for pattern: {pattern_id}"
        )

    return OutcomeResponse(**outcome.to_dict())


@router.post("/update")
async def trigger_update():
    """
    Manually trigger an outcome update.

    Updates all active outcomes with current prices.
    """
    result = await outcome_tracker_service.update_outcomes()

    return {
        "status": "ok",
        "message": "Outcome update completed",
        **result
    }


@router.get("/loop/status", response_model=UpdateLoopStatusResponse)
async def get_update_loop_status():
    """
    Get the status of the background update loop.
    """
    return UpdateLoopStatusResponse(
        running=outcome_tracker_service.is_update_running(),
        update_interval_seconds=outcome_tracker_service._update_interval,
        active_outcomes=len(outcome_tracker_service._outcomes)
    )


@router.post("/loop/start")
async def start_update_loop():
    """
    Start the background update loop.
    """
    success = await outcome_tracker_service.start_update_loop()

    if success:
        return {"status": "ok", "message": "Update loop started"}
    else:
        return {"status": "warning", "message": "Update loop already running"}


@router.post("/loop/stop")
async def stop_update_loop():
    """
    Stop the background update loop.
    """
    success = await outcome_tracker_service.stop_update_loop()

    if success:
        return {"status": "ok", "message": "Update loop stopped"}
    else:
        return {"status": "warning", "message": "Update loop not running"}
