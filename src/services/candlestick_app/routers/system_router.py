"""System and monitoring endpoints."""

import os
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ..services.pattern_detection_service import candlestick_pattern_service
from ..services.pattern_history_service import pattern_history_service

router = APIRouter()

VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_NAME = "candlestick"
START_TIME = datetime.now()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "auto_scan_running": pattern_history_service.is_running()
    }


@router.get("/info")
async def service_info():
    """
    Get detailed service information.
    """
    patterns = candlestick_pattern_service.get_supported_patterns()

    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "started_at": START_TIME.isoformat(),
        "pattern_types": len(patterns),
        "supported_timeframes": ["M5", "M15", "H1", "H4", "D1"],
        "categories": {
            "reversal": [p["type"] for p in patterns if p["category"] == "reversal"],
            "continuation": [p["type"] for p in patterns if p["category"] == "continuation"],
            "indecision": [p["type"] for p in patterns if p["category"] == "indecision"],
        }
    }


@router.get("/stats")
async def get_stats():
    """
    Get service statistics.
    """
    history_stats = pattern_history_service.get_statistics()

    return {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "history": history_stats
    }


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.

    Returns 200 when the service is ready to accept requests.
    """
    return {
        "status": "ready",
        "service": SERVICE_NAME
    }


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.

    Returns 200 when the service is alive.
    """
    return {
        "status": "alive",
        "service": SERVICE_NAME
    }
