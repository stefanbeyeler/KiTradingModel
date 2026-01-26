"""
Drift Router - API endpoints for CNN-LSTM drift detection.

Provides endpoints for:
- Viewing drift status and statistics
- Checking for drift
- Managing drift configuration
"""

from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..services.drift_detection_service import (
    drift_detection_service,
    DriftSeverity,
    DriftConfig,
)


router = APIRouter(prefix="/drift", tags=["8. Drift Detection"])


# ============================================================================
# Response Models
# ============================================================================

class DriftEventResponse(BaseModel):
    """Single drift event response."""
    drift_type: str
    severity: str
    current_value: float
    baseline_value: float
    drop_percentage: float
    detected_at: str
    task: Optional[str] = None
    message: str


class DriftStatusResponse(BaseModel):
    """Drift status response."""
    overall_severity: str
    price_drift: Optional[DriftEventResponse] = None
    pattern_drift: Optional[DriftEventResponse] = None
    regime_drift: Optional[DriftEventResponse] = None
    overall_accuracy_drift: Optional[DriftEventResponse] = None
    confidence_drift: Optional[DriftEventResponse] = None
    distribution_drift: Optional[DriftEventResponse] = None
    active_drifts_count: int
    last_check: Optional[str] = None
    recommendation: str


class DriftStatisticsResponse(BaseModel):
    """Drift statistics response."""
    observations_count: int
    drift_events_count: int
    recent_drift_events: int
    current_severity: str
    last_check: Optional[str] = None
    recommendation: str
    task_accuracies: dict
    overall_accuracy: float
    severity_distribution: dict


class DriftConfigResponse(BaseModel):
    """Drift configuration response."""
    threshold_low: float
    threshold_medium: float
    threshold_high: float
    threshold_critical: float
    min_samples_for_detection: int
    baseline_window_hours: int
    recent_window_hours: int
    calibration_tolerance: float
    distribution_change_threshold: float
    price_weight: float
    pattern_weight: float
    regime_weight: float


class DriftConfigUpdateRequest(BaseModel):
    """Request to update drift configuration."""
    threshold_medium: Optional[float] = Field(None, ge=0.05, le=0.5)
    threshold_high: Optional[float] = Field(None, ge=0.1, le=0.5)
    threshold_critical: Optional[float] = Field(None, ge=0.2, le=0.7)
    min_samples: Optional[int] = Field(None, ge=10, le=1000)
    price_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    pattern_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    regime_weight: Optional[float] = Field(None, ge=0.0, le=1.0)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status", response_model=DriftStatusResponse)
async def get_drift_status():
    """
    Get current drift status.

    Returns the latest drift detection results including task-specific
    and overall drift information.
    """
    status = drift_detection_service.get_status()
    return status.to_dict()


@router.post("/check", response_model=DriftStatusResponse)
async def check_drift():
    """
    Trigger a drift check.

    Analyzes recent prediction outcomes and compares against baseline
    to detect performance degradation.
    """
    status = drift_detection_service.check_drift()
    logger.info(f"Drift check completed: {status.overall_severity.value}")
    return status.to_dict()


@router.get("/statistics", response_model=DriftStatisticsResponse)
async def get_drift_statistics():
    """
    Get drift detection statistics.

    Returns observation counts, task-specific accuracies, and
    drift event history.
    """
    return drift_detection_service.get_statistics()


@router.get("/history")
async def get_drift_history(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum events to return")
):
    """
    Get drift event history.

    Returns recent drift events that exceeded the configured thresholds.
    """
    events = drift_detection_service.get_drift_history(limit=limit)
    return {
        "count": len(events),
        "events": events
    }


@router.get("/config", response_model=DriftConfigResponse)
async def get_drift_config():
    """
    Get current drift detection configuration.
    """
    return drift_detection_service.config.to_dict()


@router.post("/config", response_model=DriftConfigResponse)
async def update_drift_config(request: DriftConfigUpdateRequest):
    """
    Update drift detection configuration.

    Allows adjusting thresholds and task weights for drift detection.
    """
    config = drift_detection_service.update_config(
        threshold_medium=request.threshold_medium,
        threshold_high=request.threshold_high,
        threshold_critical=request.threshold_critical,
        min_samples=request.min_samples,
        price_weight=request.price_weight,
        pattern_weight=request.pattern_weight,
        regime_weight=request.regime_weight
    )
    logger.info(f"Drift config updated")
    return config.to_dict()


@router.post("/reset-baseline")
async def reset_baseline():
    """
    Reset baseline metrics.

    Uses current observation data to establish a new baseline for
    drift comparison.
    """
    drift_detection_service.reset_baseline()
    return {
        "status": "ok",
        "message": "Baseline metrics reset from current observations"
    }


@router.delete("/observations")
async def clear_observations():
    """
    Clear all observations.

    Warning: This removes all drift detection data!
    """
    count = drift_detection_service.clear_observations()
    return {
        "status": "ok",
        "message": f"Cleared {count} observations"
    }
