"""
Auto-Optimization Router for Candlestick Pattern Service.

Provides API endpoints for controlling and monitoring the automatic
optimization of pattern detection rules based on Claude validation feedback.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from ..services.auto_optimization_service import auto_optimization_service


router = APIRouter(prefix="/auto-optimization", tags=["7. Auto-Optimization"])


# === Request/Response Models ===

class AutoOptimizationConfigUpdate(BaseModel):
    """Configuration update request."""
    enabled: Optional[bool] = Field(None, description="Enable/disable auto-optimization")
    auto_validate_new_patterns: Optional[bool] = Field(None, description="Auto-validate new patterns")
    auto_apply_recommendations: Optional[bool] = Field(None, description="Auto-apply high-confidence recommendations")
    auto_revalidate_after_adjustment: Optional[bool] = Field(None, description="Re-validate after parameter changes")
    validation_sample_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sampling rate for validation")
    min_confidence_for_auto_apply: Optional[float] = Field(None, ge=0.0, le=1.0, description="Min confidence for auto-apply")
    min_feedback_count_for_auto_apply: Optional[int] = Field(None, ge=1, description="Min feedback count for auto-apply")
    max_auto_adjustments_per_hour: Optional[int] = Field(None, ge=1, description="Max auto-adjustments per hour")
    batch_size: Optional[int] = Field(None, ge=1, le=20, description="Batch size for validation")
    batch_interval_seconds: Optional[int] = Field(None, ge=10, le=600, description="Interval between batches")


class RevalidateRequest(BaseModel):
    """Request to revalidate specific patterns."""
    pattern_ids: List[str] = Field(..., description="List of pattern IDs to revalidate")


# === Endpoints ===

@router.get("/status")
async def get_auto_optimization_status():
    """
    Get the current status of the auto-optimization service.

    Returns configuration, running state, and statistics.
    """
    return auto_optimization_service.get_status()


@router.post("/config")
async def update_config(config: AutoOptimizationConfigUpdate):
    """
    Update auto-optimization configuration.

    Only provided fields will be updated.
    """
    # Filter out None values
    updates = {k: v for k, v in config.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No configuration values provided")

    updated = auto_optimization_service.update_config(**updates)

    return {
        "status": "updated",
        "config": updated
    }


@router.post("/start")
async def start_auto_optimization():
    """
    Start the auto-optimization background service.

    This will begin processing pending validations and applying
    recommendations automatically based on configuration.
    """
    await auto_optimization_service.start()

    return {
        "status": "started",
        "config": auto_optimization_service.config.to_dict()
    }


@router.post("/stop")
async def stop_auto_optimization():
    """
    Stop the auto-optimization background service.

    Pending validations will be preserved for the next start.
    """
    await auto_optimization_service.stop()

    return {
        "status": "stopped",
        "pending_validations": len(auto_optimization_service._pending_validations)
    }


@router.post("/run-cycle")
async def run_optimization_cycle():
    """
    Manually trigger a single optimization cycle.

    This processes pending validations and applies recommendations
    without waiting for the background loop.
    """
    result = await auto_optimization_service.run_optimization_cycle()
    return result


@router.get("/history")
async def get_optimization_history(
    limit: int = Query(default=50, ge=1, le=200, description="Number of entries to return")
):
    """
    Get the history of automatic parameter adjustments.
    """
    return auto_optimization_service.get_optimization_history(limit=limit)


@router.post("/revalidate")
async def revalidate_patterns(request: RevalidateRequest):
    """
    Trigger Claude re-validation for specific patterns.

    Use this after applying feedback or parameter adjustments to verify
    if patterns are now being detected correctly.
    """
    if not request.pattern_ids:
        raise HTTPException(status_code=400, detail="No pattern IDs provided")

    if len(request.pattern_ids) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 patterns per request")

    result = await auto_optimization_service.trigger_revalidation_with_claude(
        pattern_ids=request.pattern_ids
    )

    return result


@router.post("/queue-patterns")
async def queue_patterns_for_validation(
    pattern_ids: List[str] = Body(..., description="Pattern IDs to queue for validation")
):
    """
    Queue specific patterns for automatic Claude validation.

    Unlike automatic sampling, this queues all specified patterns.
    """
    from ..services.pattern_history_service import pattern_history_service

    if not pattern_ids:
        raise HTTPException(status_code=400, detail="No pattern IDs provided")

    # Get patterns from history
    history = pattern_history_service.get_history(limit=1000)
    patterns = [p for p in history if p.get("id") in pattern_ids]

    if not patterns:
        raise HTTPException(status_code=404, detail="No matching patterns found in history")

    # Temporarily set sample rate to 1.0 to queue all
    original_rate = auto_optimization_service.config.validation_sample_rate
    auto_optimization_service.config.validation_sample_rate = 1.0

    queued = await auto_optimization_service.queue_patterns_for_validation(patterns)

    # Restore original rate
    auto_optimization_service.config.validation_sample_rate = original_rate

    return {
        "requested": len(pattern_ids),
        "found": len(patterns),
        "queued": queued
    }


@router.get("/queue")
async def get_validation_queue(
    limit: int = Query(default=50, ge=1, le=200, description="Maximale Anzahl anzuzeigender Einträge"),
    offset: int = Query(default=0, ge=0, description="Offset für Paginierung")
):
    """
    Get the current validation queue contents.

    Returns patterns waiting to be validated by Claude.
    """
    queue = auto_optimization_service._pending_validations
    total = len(queue)

    # Apply pagination
    paginated = queue[offset:offset + limit]

    # Simplify the output for display
    items = []
    for item in paginated:
        items.append({
            "id": item.get("id", "unknown"),
            "pattern_type": item.get("pattern_type", "unknown"),
            "symbol": item.get("symbol", "unknown"),
            "timeframe": item.get("timeframe", "unknown"),
            "timestamp": item.get("timestamp", "unknown"),
            "confidence": item.get("confidence", 0),
            "queued_at": item.get("queued_at", item.get("timestamp", "unknown"))
        })

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items
    }


@router.delete("/queue")
async def clear_validation_queue():
    """
    Clear all pending validations from the queue.
    """
    count = len(auto_optimization_service._pending_validations)
    auto_optimization_service._pending_validations.clear()
    auto_optimization_service._save_pending_validations()

    return {
        "status": "cleared",
        "removed": count
    }


@router.post("/queue/fill-from-history")
async def fill_queue_from_history(
    count: int = Query(default=10, ge=1, le=100, description="Anzahl der Patterns aus der History")
):
    """
    Fill the validation queue with recent patterns from history.

    This allows manual batch validation even when the automatic queue is empty.
    Patterns are selected from the most recent detections that haven't been
    validated by Claude yet.
    """
    from ..services.pattern_history_service import pattern_history_service
    from ..services.claude_validator_service import claude_validator_service

    # Get recent patterns from history
    history = pattern_history_service.get_history(limit=count * 3)  # Get more to filter

    if not history:
        raise HTTPException(status_code=404, detail="Keine Patterns in der History gefunden")

    # Filter out patterns that are already in queue
    existing_ids = {p.get("id") for p in auto_optimization_service._pending_validations}

    # Filter out patterns already validated by Claude (in cache or history)
    validated_ids = set()
    for val in claude_validator_service._validation_history:
        if val.get("pattern_id"):
            validated_ids.add(val.get("pattern_id"))

    patterns_to_add = []
    for pattern in history:
        pid = pattern.get("id", "")
        if pid and pid not in existing_ids and pid not in validated_ids:
            patterns_to_add.append(pattern)
            if len(patterns_to_add) >= count:
                break

    if not patterns_to_add:
        raise HTTPException(
            status_code=404,
            detail="Keine unvalidierten Patterns gefunden. Alle Patterns wurden bereits durch Claude geprüft."
        )

    # Add to queue (bypass sampling rate)
    for pattern in patterns_to_add:
        auto_optimization_service._pending_validations.append(pattern)

    auto_optimization_service._save_pending_validations()

    return {
        "status": "filled",
        "queued": len(patterns_to_add),
        "total_in_queue": len(auto_optimization_service._pending_validations)
    }


@router.post("/enable")
async def enable_auto_optimization(
    auto_validate: bool = Query(default=True, description="Enable auto-validation"),
    auto_apply: bool = Query(default=False, description="Enable auto-apply recommendations")
):
    """
    Quick endpoint to enable auto-optimization with specific settings.
    """
    auto_optimization_service.update_config(
        enabled=True,
        auto_validate_new_patterns=auto_validate,
        auto_apply_recommendations=auto_apply
    )

    await auto_optimization_service.start()

    return {
        "status": "enabled",
        "auto_validate": auto_validate,
        "auto_apply": auto_apply,
        "config": auto_optimization_service.config.to_dict()
    }


@router.post("/disable")
async def disable_auto_optimization():
    """
    Disable auto-optimization entirely.
    """
    await auto_optimization_service.stop()

    auto_optimization_service.update_config(
        enabled=False,
        auto_validate_new_patterns=False,
        auto_apply_recommendations=False
    )

    return {
        "status": "disabled",
        "config": auto_optimization_service.config.to_dict()
    }
