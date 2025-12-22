"""System and monitoring endpoints."""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from loguru import logger

from ..models.schemas import CacheStatsResponse, ModelInfoResponse
from ..services.embedding_service import embedding_service

router = APIRouter()

VERSION = "1.0.0"
SERVICE_NAME = "embedder"
START_TIME = datetime.now()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds()
    }


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about loaded embedding models.

    Returns details about each model type including:
    - Model name
    - Embedding dimension
    - Load status
    """
    try:
        info = embedding_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get embedding cache statistics.

    Returns:
    - Number of cached entries
    - Cache size in MB
    - Hit rate
    """
    try:
        stats = embedding_service.get_cache_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_cache():
    """
    Clear the embedding cache.

    Returns number of entries cleared.
    """
    try:
        count = embedding_service.clear_cache()
        logger.info(f"Cache cleared: {count} entries")
        return {
            "status": "cleared",
            "entries_removed": count
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup")
async def warmup_models():
    """
    Pre-load all embedding models.

    Useful for reducing first-request latency.
    """
    try:
        logger.info("Starting model warmup...")

        # Warmup text embedder
        await embedding_service.embed_text(["warmup text"])
        logger.info("Text embedder warmed up")

        # Warmup FinBERT
        await embedding_service.embed_text(["warmup financial text"], use_finbert=True)
        logger.info("FinBERT embedder warmed up")

        # Note: TimeSeries and Feature embedders are warmed up on first use
        # as they require actual data

        info = embedding_service.get_model_info()

        return {
            "status": "completed",
            "models_loaded": info
        }
    except Exception as e:
        logger.error(f"Error during warmup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dimensions")
async def get_embedding_dimensions():
    """
    Get embedding dimensions for all model types.

    Useful for configuring downstream services.
    """
    return {
        "text": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384
        },
        "financial_text": {
            "model": "ProsusAI/finbert",
            "dimension": 768
        },
        "timeseries": {
            "model": "ts2vec",
            "dimension": 320
        },
        "features": {
            "model": "feature_autoencoder",
            "dimension": 128
        }
    }
