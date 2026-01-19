"""System and monitoring endpoints."""

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from datetime import datetime
from loguru import logger
import sys
import os

from ..services.regime_detection_service import regime_detection_service
from ..services.signal_scoring_service import signal_scoring_service

# Import für Test-Health-Funktionalität
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from src.shared.health import get_test_unhealthy_status
from src.shared.test_health_router import create_test_health_router

router = APIRouter()


class ModelReloadRequest(BaseModel):
    """Request to reload a model."""
    model_type: str = Field(..., description="Model type: 'hmm' or 'scorer'")
    symbol: Optional[str] = Field(default=None, description="Symbol for HMM models")
    model_path: Optional[str] = Field(default=None, description="Optional path to model file")

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


@router.post("/model/reload")
async def reload_model(request: ModelReloadRequest):
    """
    Reload a model from disk.

    Called by HMM-Train service after successful deployment.
    This endpoint allows hot-reloading of models without restarting the service.

    Model Types:
    - **hmm**: Reload HMM model for a specific symbol
    - **scorer**: Reload the LightGBM signal scorer

    The model will be loaded from the current/ directory (symlinked to production version).
    """
    try:
        if request.model_type == "hmm" and request.symbol:
            # Clear cached model to force reload on next request
            if request.symbol in regime_detection_service._models:
                del regime_detection_service._models[request.symbol]
                logger.info(f"Cleared HMM model cache for {request.symbol}")

            # Optionally pre-load the model
            try:
                model = regime_detection_service.get_model(request.symbol)
                is_fitted = model.is_fitted() if model else False
            except Exception as e:
                logger.warning(f"Could not pre-load model for {request.symbol}: {e}")
                is_fitted = False

            return {
                "status": "ok",
                "message": f"HMM model cache cleared for {request.symbol}",
                "symbol": request.symbol,
                "model_loaded": is_fitted
            }

        elif request.model_type == "scorer":
            # Reload scorer from current/ directory
            from ..models.lightgbm_scorer import LightGBMSignalScorer

            # Determine path
            scorer_path = request.model_path
            if not scorer_path:
                # Default to current/ symlink
                scorer_path = os.path.join(
                    os.getenv("MODEL_DIR", "/app/data/models/hmm"),
                    "current",
                    "scorer_lightgbm.pkl"
                )

                # Fallback to legacy location
                if not os.path.exists(scorer_path):
                    scorer_path = os.path.join(
                        os.getenv("MODEL_DIR", "/app/data/models/hmm"),
                        "scorer_lightgbm.pkl"
                    )

            if os.path.exists(scorer_path):
                signal_scoring_service._scorer = LightGBMSignalScorer.load(scorer_path)
                logger.info(f"Reloaded scorer from {scorer_path}")
                return {
                    "status": "ok",
                    "message": "Scorer model reloaded",
                    "path": scorer_path,
                    "is_fitted": signal_scoring_service._scorer.is_fitted()
                }
            else:
                logger.warning(f"Scorer model not found at {scorer_path}")
                return {
                    "status": "warning",
                    "message": f"Scorer model not found at {scorer_path}",
                    "is_fitted": signal_scoring_service._scorer.is_fitted()
                }

        else:
            return {
                "status": "error",
                "message": f"Unknown model_type: {request.model_type}. Use 'hmm' or 'scorer'"
            }

    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/model/reload-all")
async def reload_all_models():
    """
    Reload all cached models.

    Clears all HMM model caches and reloads the scorer.
    Use this after a major model update or deployment.
    """
    try:
        # Clear all HMM models
        cleared_symbols = list(regime_detection_service._models.keys())
        regime_detection_service._models.clear()

        # Reload scorer
        from ..models.lightgbm_scorer import LightGBMSignalScorer

        scorer_paths = [
            os.path.join(os.getenv("MODEL_DIR", "/app/data/models/hmm"), "current", "scorer_lightgbm.pkl"),
            os.path.join(os.getenv("MODEL_DIR", "/app/data/models/hmm"), "scorer_lightgbm.pkl")
        ]

        scorer_loaded = False
        for scorer_path in scorer_paths:
            if os.path.exists(scorer_path):
                signal_scoring_service._scorer = LightGBMSignalScorer.load(scorer_path)
                scorer_loaded = True
                logger.info(f"Reloaded scorer from {scorer_path}")
                break

        logger.info(f"Cleared {len(cleared_symbols)} HMM model caches")

        return {
            "status": "ok",
            "message": "All models reloaded",
            "hmm_models_cleared": cleared_symbols,
            "scorer_reloaded": scorer_loaded
        }

    except Exception as e:
        logger.error(f"Reload all models failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
