"""System and monitoring endpoints for TCN Training Service."""

from fastapi import APIRouter
from loguru import logger

from ..services.training_service import training_service

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Always responds quickly, even during training.
    """
    return {
        "status": "healthy",
        "service": "tcn-train",
        "training_active": training_service.is_training()
    }


@router.get("/status")
async def get_service_status():
    """Get overall service status."""
    from ..services.training_scheduler import training_scheduler

    return {
        "service": "tcn-train",
        "status": "healthy",
        "training": training_service.get_training_status(),
        "scheduler": training_scheduler.get_status(),
        "models_count": len(training_service.list_models())
    }
