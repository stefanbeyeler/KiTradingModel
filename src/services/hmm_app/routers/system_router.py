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
from ..services.accuracy_tracker import accuracy_tracker

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


# ==================== Self-Learning / Accuracy Tracking ====================


class SelfLearningConfigRequest(BaseModel):
    """Request to update self-learning configuration."""
    enabled: Optional[bool] = Field(default=None, description="Enable/disable self-learning")
    accuracy_threshold: Optional[float] = Field(default=None, ge=0.1, le=0.9, description="Accuracy threshold for retrain trigger (0.1-0.9)")
    cooldown_hours: Optional[int] = Field(default=None, ge=1, le=48, description="Cooldown between retrains in hours (1-48)")


@router.get("/accuracy/stats")
async def get_accuracy_stats(symbol: Optional[str] = None):
    """
    Get accuracy statistics for Self-Learning feedback loop.

    Returns rolling accuracy metrics used to determine when re-training is needed.

    **Parameters:**
    - **symbol**: Optional symbol to get stats for (default: global stats)

    **Response:**
    - Rolling accuracy over last N predictions
    - Per-regime accuracy breakdown
    - Self-learning configuration and status
    """
    if symbol:
        return accuracy_tracker.get_stats(symbol)
    return accuracy_tracker.get_all_stats()


@router.get("/accuracy/should-retrain")
async def should_trigger_retrain():
    """
    Check if accuracy has dropped below threshold and retrain should be triggered.

    Used by HMM-Train service to implement closed-loop feedback.

    **Response:**
    - **should_retrain**: Boolean indicating if retraining is recommended
    - **reason**: Explanation of why retrain is/isn't needed
    - **current_accuracy**: Current rolling accuracy
    - **threshold**: Configured accuracy threshold
    """
    stats = accuracy_tracker.get_all_stats()
    should_retrain = accuracy_tracker.should_trigger_retrain()

    reasons = []
    if not stats["self_learning"]["enabled"]:
        reasons.append("Self-learning is disabled")
    elif stats["global"]["evaluated_predictions"] < 50:
        reasons.append(f"Not enough evaluations ({stats['global']['evaluated_predictions']} < 50)")
    elif stats["global"]["rolling_accuracy"] >= stats["self_learning"]["accuracy_threshold"]:
        reasons.append(f"Accuracy ({stats['global']['rolling_accuracy']:.1%}) is above threshold ({stats['self_learning']['accuracy_threshold']:.1%})")
    else:
        reasons.append(f"Accuracy ({stats['global']['rolling_accuracy']:.1%}) dropped below threshold ({stats['self_learning']['accuracy_threshold']:.1%})")

        # Check cooldown
        if stats["self_learning"]["last_retrain_trigger"]:
            reasons.append(f"Last retrain: {stats['self_learning']['last_retrain_trigger']}")

    return {
        "should_retrain": should_retrain,
        "reason": "; ".join(reasons),
        "current_accuracy": stats["global"]["rolling_accuracy"],
        "threshold": stats["self_learning"]["accuracy_threshold"],
        "evaluated_predictions": stats["global"]["evaluated_predictions"],
        "self_learning_enabled": stats["self_learning"]["enabled"]
    }


@router.post("/accuracy/mark-retrain")
async def mark_retrain_triggered():
    """
    Mark that a retrain was triggered.

    Called by HMM-Train service after initiating a training job.
    Resets the cooldown timer.
    """
    accuracy_tracker.mark_retrain_triggered()
    return {
        "status": "ok",
        "message": "Retrain trigger recorded",
        "cooldown_until": accuracy_tracker._last_retrain_trigger.isoformat() if accuracy_tracker._last_retrain_trigger else None
    }


@router.post("/self-learning/config")
async def update_self_learning_config(request: SelfLearningConfigRequest):
    """
    Update self-learning configuration.

    **Parameters:**
    - **enabled**: Enable/disable automatic retrain triggers
    - **accuracy_threshold**: Accuracy below which retrain is triggered (default: 0.40)
    - **cooldown_hours**: Minimum hours between retrains (default: 6)
    """
    config = accuracy_tracker.set_config(
        enabled=request.enabled,
        threshold=request.accuracy_threshold,
        cooldown_hours=request.cooldown_hours
    )

    return {
        "status": "ok",
        "config": config
    }


@router.get("/self-learning/config")
async def get_self_learning_config():
    """
    Get current self-learning configuration.
    """
    stats = accuracy_tracker.get_all_stats()
    return {
        "enabled": stats["self_learning"]["enabled"],
        "accuracy_threshold": stats["self_learning"]["accuracy_threshold"],
        "cooldown_hours": stats["self_learning"]["cooldown_hours"],
        "last_retrain_trigger": stats["self_learning"]["last_retrain_trigger"]
    }
