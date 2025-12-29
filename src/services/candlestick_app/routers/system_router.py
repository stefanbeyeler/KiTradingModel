"""System and monitoring endpoints."""

import os
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ..services.pattern_detection_service import candlestick_pattern_service
from ..services.pattern_history_service import pattern_history_service
from ..services.ai_validator_service import ai_validator_service

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


@router.get("/ai-status")
async def get_ai_status():
    """
    Get AI validator status.

    Shows whether an AI model is loaded for pattern validation
    and information about the current model.
    """
    status = ai_validator_service.get_status()

    return {
        "timestamp": datetime.now().isoformat(),
        "ai_validation": status,
        "detection_mode": "hybrid" if status["model_loaded"] else "rule-only",
        "description": (
            "Patterns werden durch Regelwerk erkannt und durch KI validiert"
            if status["model_loaded"]
            else "Patterns werden nur durch Regelwerk erkannt (kein KI-Modell geladen)"
        )
    }


@router.post("/ai-reload")
async def reload_ai_model():
    """
    Reload the AI model.

    Forces a reload of the latest trained model.
    Use this after training has completed.
    """
    success = ai_validator_service.initialize()

    return {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "status": ai_validator_service.get_status(),
        "message": "AI-Modell erfolgreich geladen" if success else "Kein Modell gefunden"
    }
