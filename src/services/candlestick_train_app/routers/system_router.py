"""System and monitoring endpoints for training service."""

import os
from datetime import datetime
from fastapi import APIRouter

from ..services.training_service import training_service, TORCH_AVAILABLE
from ..services.training_scheduler import training_scheduler

router = APIRouter()

VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_NAME = "candlestick-train"
START_TIME = datetime.now()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "pytorch_available": TORCH_AVAILABLE,
        "training_active": training_service.is_training(),
        "scheduler_running": training_scheduler.is_running(),
    }


@router.get("/info")
async def service_info():
    """
    Get detailed service information.
    """
    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "started_at": START_TIME.isoformat(),
        "pytorch_available": TORCH_AVAILABLE,
        "capabilities": {
            "pattern_types": 21,
            "supported_timeframes": ["M5", "M15", "H1", "H4", "D1"],
            "training_enabled": TORCH_AVAILABLE,
            "scheduled_training": training_scheduler.get_config().get("enabled", False),
        }
    }


@router.get("/stats")
async def get_stats():
    """
    Get service statistics.
    """
    jobs = training_service.get_all_jobs(limit=100)

    completed = sum(1 for j in jobs if j.status.value == "completed")
    failed = sum(1 for j in jobs if j.status.value == "failed")

    return {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "total_jobs": len(jobs),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "current_job": training_service.get_current_job().to_dict() if training_service.get_current_job() else None,
        "latest_model": training_service.get_latest_model(),
    }


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.
    """
    return {
        "status": "ready",
        "service": SERVICE_NAME
    }


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.
    """
    return {
        "status": "alive",
        "service": SERVICE_NAME
    }
