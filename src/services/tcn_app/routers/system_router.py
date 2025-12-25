"""System and monitoring endpoints."""

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

from ..services.pattern_detection_service import pattern_detection_service

router = APIRouter()

VERSION = "1.0.0"
SERVICE_NAME = "tcn-pattern"
START_TIME = datetime.now()


class ModelReloadRequest(BaseModel):
    """Request to reload model."""
    model_path: Optional[str] = None


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "model_loaded": pattern_detection_service.is_model_loaded()
    }


@router.get("/info")
async def service_info():
    """
    Get detailed service information.
    """
    from ..models.tcn_model import TCNPatternClassifier

    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "started_at": START_TIME.isoformat(),
        "model": pattern_detection_service.get_model_info(),
        "pattern_classes": TCNPatternClassifier.PATTERN_CLASSES,
        "num_classes": len(TCNPatternClassifier.PATTERN_CLASSES)
    }


@router.get("/stats")
async def get_stats():
    """
    Get service statistics.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "model": pattern_detection_service.get_model_info()
    }


@router.post("/model/reload")
async def reload_model(request: ModelReloadRequest):
    """
    Hot-reload the TCN model.

    Called by the training service when a new model is available.
    Can also be called manually to reload the model from disk.

    Args:
        model_path: Path to the model file. If not provided, uses latest.pt
    """
    success = pattern_detection_service.reload_model(request.model_path)

    if success:
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model": pattern_detection_service.get_model_info()
        }
    else:
        return {
            "status": "error",
            "message": "Failed to reload model",
            "model": pattern_detection_service.get_model_info()
        }


@router.get("/model")
async def get_model_info():
    """Get information about the currently loaded model."""
    return pattern_detection_service.get_model_info()
