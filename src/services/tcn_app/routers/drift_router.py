"""Drift detection endpoints for TCN Service."""

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from loguru import logger

from ..services.drift_detection_service import drift_detection_service

router = APIRouter()


class DriftConfigUpdate(BaseModel):
    """Drift configuration update request."""
    threshold_medium: Optional[float] = Field(None, ge=0, le=1, description="Medium severity threshold (0.1 = 10%)")
    threshold_high: Optional[float] = Field(None, ge=0, le=1, description="High severity threshold (0.2 = 20%)")
    threshold_critical: Optional[float] = Field(None, ge=0, le=1, description="Critical severity threshold (0.3 = 30%)")
    min_samples: Optional[int] = Field(None, ge=10, description="Minimum samples for detection")


@router.get("/drift/status")
async def get_drift_status():
    """
    Get current drift status.

    Returns the overall drift status including:
    - Overall severity level
    - Active drift events
    - Recommendations for action
    """
    status = drift_detection_service.get_status()
    return status.to_dict()


@router.post("/drift/check")
async def check_drift():
    """
    Trigger a drift check.

    Analyzes recent observations against baseline to detect:
    - Accuracy drift
    - Confidence calibration drift
    - Pattern distribution drift
    """
    status = drift_detection_service.check_drift()
    return {
        "status": "checked",
        "result": status.to_dict()
    }


@router.get("/drift/statistics")
async def get_drift_statistics():
    """Get drift detection statistics."""
    return drift_detection_service.get_statistics()


@router.get("/drift/history")
async def get_drift_history(limit: int = 20):
    """Get drift event history."""
    history = drift_detection_service.get_drift_history(limit)
    return {
        "events": history,
        "count": len(history)
    }


@router.post("/drift/reset-baseline")
async def reset_baseline():
    """Reset baseline metrics from current observations."""
    drift_detection_service.reset_baseline()
    return {
        "status": "reset",
        "message": "Baseline metrics have been reset"
    }


@router.put("/drift/config")
async def update_drift_config(config: DriftConfigUpdate):
    """Update drift detection configuration."""
    updated = drift_detection_service.update_config(
        threshold_medium=config.threshold_medium,
        threshold_high=config.threshold_high,
        threshold_critical=config.threshold_critical,
        min_samples=config.min_samples
    )
    return {
        "status": "updated",
        "config": {
            "threshold_low": updated.threshold_low,
            "threshold_medium": updated.threshold_medium,
            "threshold_high": updated.threshold_high,
            "threshold_critical": updated.threshold_critical,
            "min_samples_for_detection": updated.min_samples_for_detection
        }
    }
