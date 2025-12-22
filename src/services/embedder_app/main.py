"""
Embedder Service - Central Embedding Service for ML Models

Port: 3007
Provides embeddings for:
- Text (Sentence Transformers)
- Financial Text (FinBERT)
- Time Series (TS2Vec)
- Features (Autoencoder)
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import text_router, timeseries_router, feature_router, system_router
from .services.embedding_service import embedding_service

# Configuration
VERSION = "1.0.0"
SERVICE_NAME = "embedder"
SERVICE_PORT = int(os.getenv("PORT", "3007"))
ROOT_PATH = os.getenv("ROOT_PATH", "")

# Logging setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>EMBEDDER</cyan> | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger.add(
    f"logs/embedder_service.log",
    rotation="10 MB",
    retention="7 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info(f"Starting Embedder Service v{VERSION} on port {SERVICE_PORT}")

    # Pre-warm models if configured
    if os.getenv("EMBEDDER_WARMUP", "false").lower() == "true":
        logger.info("Pre-warming embedding models...")
        try:
            await embedding_service.embed_text(["warmup"])
            logger.info("Models warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    yield

    logger.info("Shutting down Embedder Service...")


# Create FastAPI app
app = FastAPI(
    title="Embedder Service",
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
    """,
    version=VERSION,
    root_path=ROOT_PATH,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system_router, prefix="/api/v1", tags=["1. System & Monitoring"])
app.include_router(text_router, prefix="/api/v1/text", tags=["2. Text Embeddings"])
app.include_router(timeseries_router, prefix="/api/v1/timeseries", tags=["3. Time Series Embeddings"])
app.include_router(feature_router, prefix="/api/v1/features", tags=["4. Feature Embeddings"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Embedder Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "text": "/api/v1/text/embed",
            "timeseries": "/api/v1/timeseries/embed",
            "features": "/api/v1/features/embed",
            "models": "/api/v1/models",
            "cache": "/api/v1/cache/stats"
        }
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "healthy", "service": SERVICE_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.embedder_app.main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False
    )
