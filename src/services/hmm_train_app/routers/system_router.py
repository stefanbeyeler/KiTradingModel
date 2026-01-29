"""System endpoints for HMM Training Service."""

from fastapi import APIRouter
from loguru import logger

from src.config.microservices import microservices_config
from ..services.training_service import training_service

router = APIRouter(tags=["System"])


@router.get("/info")
async def get_service_info():
    """Get service information."""
    return {
        "service": "hmm-train",
        "description": "HMM & LightGBM Scorer Training Service",
        "version": "1.0.0",
        "model_types": ["hmm", "scorer", "both"],
        "endpoints": {
            "training": "/api/v1/train",
            "health": "/health"
        },
        "status": training_service.get_status()
    }


@router.get("/config")
async def get_config():
    """Get service configuration."""
    import os

    return {
        "model_directory": os.getenv("MODEL_DIR", "/app/data/models/hmm"),
        "data_service_url": os.getenv("DATA_SERVICE_URL", microservices_config.data_service_url),
        "hmm_service_url": os.getenv("HMM_SERVICE_URL", microservices_config.hmm_service_url),
        "default_timeframe": "1h",
        "default_lookback_days": 365,
        "hmm_n_components": 4,
        "regimes": ["bull_trend", "bear_trend", "sideways", "high_volatility"]
    }
