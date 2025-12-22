"""System and monitoring endpoints."""

from fastapi import APIRouter
from datetime import datetime

from ..services.regime_detection_service import regime_detection_service
from ..services.signal_scoring_service import signal_scoring_service

router = APIRouter()

VERSION = "1.0.0"
SERVICE_NAME = "hmm-regime"
START_TIME = datetime.now()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
        "models_loaded": len(regime_detection_service._models),
        "scorer_fitted": signal_scoring_service._scorer.is_fitted()
    }


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
