"""
CNN-LSTM Multi-Task Training Service

FastAPI Application für Training des Hybrid CNN-LSTM Modells.
Wird vom Watchdog Orchestrator gesteuert.

Port: 3017
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import training_router, system_router

# =============================================================================
# Configuration
# =============================================================================

SERVICE_NAME = os.getenv("SERVICE_NAME", "cnn-lstm-train")
SERVICE_VERSION = "1.0.0"
SERVICE_PORT = int(os.getenv("PORT", "3017"))
ROOT_PATH = os.getenv("ROOT_PATH", "/cnn-lstm-train")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/cnn-lstm")
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
CNN_LSTM_SERVICE_URL = os.getenv("CNN_LSTM_SERVICE_URL", "http://trading-cnn-lstm:3007")

# Global state
_start_time: float = 0.0
_training_in_progress: bool = False


def get_uptime() -> float:
    """Gibt die Uptime in Sekunden zurueck."""
    return time.time() - _start_time


def is_training_in_progress() -> bool:
    """Prueft ob ein Training laeuft."""
    return _training_in_progress


def set_training_in_progress(value: bool):
    """Setzt den Training-Status."""
    global _training_in_progress
    _training_in_progress = value


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _start_time

    # Startup
    _start_time = time.time()
    logger.info(f"Starting {SERVICE_NAME} service v{SERVICE_VERSION} on port {SERVICE_PORT}")

    # Erstelle Model-Verzeichnis falls nicht vorhanden
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Model directory: {MODEL_DIR}")

    # Log Data Service URL
    logger.info(f"Data Service URL: {DATA_SERVICE_URL}")
    logger.info(f"CNN-LSTM Service URL: {CNN_LSTM_SERVICE_URL}")

    # Check PyTorch und GPU
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {device_name}")
        else:
            logger.info("GPU not available - training will use CPU (slower)")
    except ImportError:
        logger.error("PyTorch not installed - training will fail!")

    # Lade Training-Historie
    try:
        from .services.training_service import training_service
        await training_service.load_history()
        logger.info("Training history loaded")
    except Exception as e:
        logger.warning(f"Could not load training history: {e}")

    logger.info(f"{SERVICE_NAME} service started successfully")

    yield

    # Shutdown
    logger.info(f"Shutting down {SERVICE_NAME} service")

    # Speichere Training-Historie
    try:
        from .services.training_service import training_service
        await training_service.save_history()
        logger.info("Training history saved")
    except Exception as e:
        logger.warning(f"Could not save training history: {e}")

    # Abbrechen falls Training laeuft
    if _training_in_progress:
        logger.warning("Training in progress during shutdown - cancelling")
        try:
            from .services.training_service import training_service
            await training_service.cancel_training()
        except Exception as e:
            logger.error(f"Error cancelling training: {e}")


# =============================================================================
# FastAPI Application
# =============================================================================

# OpenAPI Tags
openapi_tags = [
    {
        "name": "1. System",
        "description": "Health checks und Service-Informationen"
    },
    {
        "name": "2. Training",
        "description": "Training-Jobs starten, ueberwachen und verwalten"
    },
    {
        "name": "3. Models",
        "description": "Trainierte Modelle verwalten"
    },
]

app = FastAPI(
    title="CNN-LSTM Training Service",
    description="""
Training Service für das Hybrid CNN-LSTM Multi-Task Modell.

## Features

- **Multi-Task Training**: Gleichzeitiges Training für Preis, Patterns und Regime
- **Flexible Konfiguration**: Anpassbare Gewichtungen, Epochen, Batch-Groesse
- **Watchdog Integration**: Wird vom Training Orchestrator gesteuert
- **Model Checkpointing**: Automatisches Speichern der besten Modelle

## Training Tasks

| Task | Loss-Funktion | Gewichtung |
|------|---------------|------------|
| Preis-Vorhersage | MSE + Direction | 40% |
| Pattern-Klassifikation | BCE (Multi-Label) | 35% |
| Regime-Vorhersage | CrossEntropy | 25% |

## Daten-Pipeline

Daten werden ueber den Data Service (Port 3001) mit 3-Layer-Caching abgerufen.
    """,
    version=SERVICE_VERSION,
    openapi_tags=openapi_tags,
    root_path=ROOT_PATH,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Include Routers
# =============================================================================

app.include_router(system_router, prefix="/api/v1", tags=["1. System"])
app.include_router(training_router, prefix="/api/v1", tags=["2. Training"])


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": f"{ROOT_PATH}/docs",
        "health": f"{ROOT_PATH}/health"
    }


@app.get("/health", tags=["1. System"])
async def health_check():
    """Quick health check endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": SERVICE_VERSION,
        "training_in_progress": _training_in_progress,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# Export for uvicorn
# =============================================================================

# Fuer direkten Start: uvicorn src.services.cnn_lstm_train_app.main:app --host 0.0.0.0 --port 3017
