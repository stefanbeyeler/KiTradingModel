"""System endpoints for NHITS Training Service."""

from fastapi import APIRouter
from loguru import logger

from ..services.training_service import training_service

router = APIRouter(tags=["System"])


@router.get("/info")
async def get_service_info():
    """Get service information."""
    return {
        "service": "nhits-train",
        "description": "NHITS Model Training Service",
        "version": "1.0.0",
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
        "model_directory": os.getenv("MODEL_DIR", "/app/data/models/nhits"),
        "data_service_url": os.getenv("DATA_SERVICE_URL", "http://trading-data:3001"),
        "nhits_service_url": os.getenv("NHITS_SERVICE_URL", "http://trading-nhits:3002"),
        "default_timeframes": ["H1", "D1"],
        "cpu_threads": {
            "omp": os.getenv("OMP_NUM_THREADS", "4"),
            "mkl": os.getenv("MKL_NUM_THREADS", "4"),
            "openblas": os.getenv("OPENBLAS_NUM_THREADS", "4")
        }
    }
