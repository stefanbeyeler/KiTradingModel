"""System and monitoring endpoints."""

from fastapi import APIRouter
from datetime import datetime
import sys
import os

from ..services.regime_detection_service import regime_detection_service
from ..services.signal_scoring_service import signal_scoring_service

# Import für Test-Health-Funktionalität
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from src.shared.health import get_test_unhealthy_status
from src.shared.test_health_router import create_test_health_router

router = APIRouter()

VERSION = "1.0.0"
SERVICE_NAME = "hmm-regime"
START_TIME = datetime.now()

# Test-Health-Router
test_health_router = create_test_health_router("hmm")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status("hmm")
    is_unhealthy = test_status.get("test_unhealthy", False)

    response = {
        "service": SERVICE_NAME,
        "status": "unhealthy" if is_unhealthy else "healthy",
        "version": VERSION,
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "models_loaded": len(regime_detection_service._models),
        "scorer_fitted": signal_scoring_service._scorer.is_fitted()
    }

    # Test-Status hinzufügen wenn aktiv
    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


@router.get("/info")
async def service_info():
    """
    Get detailed service information.
    """
    from ..models.hmm_regime_model import MarketRegime

    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "started_at": START_TIME.isoformat(),
        "components": {
            "hmm": {
                "loaded_models": list(regime_detection_service._models.keys()),
                "regimes": [r.value for r in MarketRegime]
            },
            "scorer": {
                "fitted": signal_scoring_service._scorer.is_fitted(),
                "feature_count": 18
            }
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
        "hmm_models_count": len(regime_detection_service._models),
        "scorer_fitted": signal_scoring_service._scorer.is_fitted()
    }
