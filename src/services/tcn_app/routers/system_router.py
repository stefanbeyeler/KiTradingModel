"""System and monitoring endpoints."""

from fastapi import APIRouter
from datetime import datetime

from ..services.pattern_detection_service import pattern_detection_service
from ..services.tcn_training_service import tcn_training_service

router = APIRouter()

VERSION = "1.0.0"
SERVICE_NAME = "tcn-pattern"
START_TIME = datetime.now()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "model_loaded": pattern_detection_service.is_model_loaded(),
        "training_in_progress": tcn_training_service.is_training()
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
        "model": {
            "loaded": pattern_detection_service.is_model_loaded(),
            "pattern_classes": TCNPatternClassifier.PATTERN_CLASSES,
            "num_classes": len(TCNPatternClassifier.PATTERN_CLASSES),
            "parameters": pattern_detection_service.tcn_model.get_num_parameters()
        },
        "training": {
            "in_progress": tcn_training_service.is_training(),
            "available_models": len(tcn_training_service.list_models())
        }
    }


@router.get("/stats")
async def get_stats():
    """
    Get service statistics.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "model_loaded": pattern_detection_service.is_model_loaded(),
        "available_models": len(tcn_training_service.list_models()),
        "training_history_count": len(tcn_training_service.get_training_history())
    }
