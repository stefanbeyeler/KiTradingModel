"""
Embedder Service - Central Embedding Service for ML Models
=============================================================

Port: 3007 (konfigurierbar via PORT Umgebungsvariable)

Provides embeddings for:
- Text (Sentence Transformers)
- Financial Text (FinBERT)
- Time Series (TS2Vec)
- Features (Autoencoder)

Verwendet die standardisierte Service-Basis-Architektur.
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import settings
from .routers import text_router, timeseries_router, feature_router, system_router
from .services.embedding_service import embedding_service

# ============================================================
# Logging Configuration (Standard)
# ============================================================
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>EMBEDDER</cyan> | "
           "<level>{message}</level>",
    level=settings.log_level,
)
logger.add(
    f"logs/{settings.service_name}_{{time}}.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)

# ============================================================
# Service State
# ============================================================
class ServiceState:
    """Zentrale State-Verwaltung für den Embedder Service."""
    is_ready: bool = False
    startup_error: str | None = None
    models_loaded: bool = False


state = ServiceState()

# Startzeit für Uptime
from datetime import datetime
_start_time = datetime.utcnow()


def get_uptime_seconds() -> float:
    """Berechnet Uptime in Sekunden."""
    return (datetime.utcnow() - _start_time).total_seconds()


# ============================================================
# Lifespan Management (Modern FastAPI Pattern)
# ============================================================
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management für den Embedder Service.

    Startup:
    - Optional: Model Warmup
    - Lazy Loading ist default an (Modelle laden bei erster Anfrage)

    Shutdown:
    - Cache speichern
    - Ressourcen freigeben
    """
    logger.info("=" * 60)
    logger.info(f"  EMBEDDER SERVICE v{settings.version}")
    logger.info("=" * 60)
    logger.info(f"  Port: {settings.port}")
    logger.info(f"  GPU: {'Enabled' if settings.use_gpu else 'Disabled'}")
    logger.info(f"  Lazy Loading: {'Enabled' if settings.lazy_load_models else 'Disabled'}")
    logger.info(f"  Cache Size: {settings.cache_max_size}")
    logger.info("=" * 60)

    try:
        # Model Warmup (optional)
        if settings.warmup_on_startup:
            logger.info("Pre-warming embedding models...")
            try:
                await embedding_service.embed_text(["warmup"])
                state.models_loaded = True
                logger.success("Models warmed up successfully")
            except Exception as e:
                logger.warning(f"Model warmup failed: {e}")
                # Kein Fehler - Lazy Loading wird es später laden

        state.is_ready = True
        logger.success("Embedder Service is ready!")

    except Exception as e:
        state.startup_error = str(e)
        logger.error(f"Startup error: {e}")
        # Service startet trotzdem, wird aber "degraded" melden

    yield  # === SERVICE LÄUFT ===

    # === SHUTDOWN ===
    logger.info("=" * 60)
    logger.info("  EMBEDDER SERVICE SHUTDOWN")
    logger.info("=" * 60)

    try:
        # Cache-Statistiken loggen
        cache_stats = embedding_service.get_cache_stats()
        logger.info(f"Cache stats at shutdown: {cache_stats}")

        # Ressourcen freigeben
        embedding_service.cleanup()
        logger.info("Resources cleaned up")

    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title=settings.display_name,
    description="""
    Central embedding service for the KI Trading Model system.

    ## Features

    - **Text Embeddings**: General-purpose text embeddings using Sentence Transformers
    - **Financial Text**: Finance-optimized embeddings using FinBERT
    - **Time Series**: OHLCV sequence embeddings using TS2Vec architecture
    - **Feature Compression**: Technical indicator compression using autoencoder

    ## Embedding Dimensions

    | Type | Model | Dimension |
    |------|-------|-----------|
    | Text | all-MiniLM-L6-v2 | 384 |
    | Financial | FinBERT | 768 |
    | TimeSeries | TS2Vec | 320 |
    | Features | Autoencoder | 128 |

    ## Usage

    All embeddings are normalized (L2) for cosine similarity computation.
    Results are automatically cached for performance.

    ## Configuration

    - `PORT`: Service Port (default: 3007)
    - `EMBEDDER_WARMUP`: Pre-load models on startup (default: false)
    - `EMBEDDING_CACHE_SIZE`: Max cache entries (default: 10000)
    - `USE_GPU`: Enable GPU acceleration (default: true)
    """,
    version=settings.version,
    root_path=settings.root_path,
    lifespan=lifespan,
)

# ============================================================
# Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Health Check (Standardized)
# ============================================================
from src.shared.health import HealthResponse, HealthStatus


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Standardisierter Health-Check.

    Returns:
        HealthResponse mit Service-Status und Model-Informationen
    """
    # GPU Status
    gpu_available = False
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Model Status
    model_info = embedding_service.get_model_info()

    # Status bestimmen
    if state.startup_error:
        status = HealthStatus.UNHEALTHY
    elif not state.is_ready:
        status = HealthStatus.STARTING
    else:
        status = HealthStatus.HEALTHY

    return HealthResponse(
        status=status,
        service=settings.service_name,
        version=settings.version,
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=round(get_uptime_seconds(), 2),
        is_ready=state.is_ready,
        error=state.startup_error,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        model_loaded=state.models_loaded or model_info.get("text", {}).get("loaded", False),
        model_name=settings.text_model,
        extras={
            "models": model_info,
            "cache_size": settings.cache_max_size,
            "lazy_loading": settings.lazy_load_models,
        },
    )


# ============================================================
# Router Registration
# ============================================================
app.include_router(system_router, prefix="/api/v1", tags=["1. System & Monitoring"])
app.include_router(text_router, prefix="/api/v1/text", tags=["2. Text Embeddings"])
app.include_router(timeseries_router, prefix="/api/v1/timeseries", tags=["3. Time Series Embeddings"])
app.include_router(feature_router, prefix="/api/v1/features", tags=["4. Feature Embeddings"])


# ============================================================
# Root Endpoint
# ============================================================
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with service info and navigation."""
    return {
        "service": settings.display_name,
        "version": settings.version,
        "status": "healthy" if state.is_ready else "starting",
        "docs": f"{settings.root_path}/docs",
        "health": f"{settings.root_path}/health",
        "endpoints": {
            "text_embed": "/api/v1/text/embed",
            "timeseries_embed": "/api/v1/timeseries/embed",
            "features_embed": "/api/v1/features/embed",
            "models": "/api/v1/models",
            "cache_stats": "/api/v1/cache/stats",
            "dimensions": "/api/v1/dimensions",
        },
    }


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.services.embedder_app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )
