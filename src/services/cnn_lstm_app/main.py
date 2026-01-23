"""
CNN-LSTM Multi-Task Inference Service

FastAPI Application für Hybrid CNN-LSTM Inference mit Multi-Task Vorhersagen:
- Preis-Vorhersage (Regression)
- Pattern-Klassifikation (16 Chart-Patterns)
- Regime-Vorhersage (Bull/Bear/Sideways/HighVol)

Port: 3007
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import prediction_router, system_router, revalidation_router, outcome_router

# =============================================================================
# Configuration
# =============================================================================

SERVICE_NAME = os.getenv("SERVICE_NAME", "cnn-lstm")
SERVICE_VERSION = "1.0.0"
SERVICE_PORT = int(os.getenv("PORT", "3007"))
ROOT_PATH = os.getenv("ROOT_PATH", "/cnn-lstm")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/cnn-lstm")

# Global state
_start_time: float = 0.0
_model_loaded: bool = False
_model_version: str | None = None


def get_uptime() -> float:
    """Gibt die Uptime in Sekunden zurueck."""
    return time.time() - _start_time


def is_model_loaded() -> bool:
    """Prueft ob ein Modell geladen ist."""
    return _model_loaded


def get_model_version() -> str | None:
    """Gibt die aktuelle Modell-Version zurueck."""
    return _model_version


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _start_time, _model_loaded, _model_version

    # Startup
    _start_time = time.time()
    logger.info(f"Starting {SERVICE_NAME} service v{SERVICE_VERSION} on port {SERVICE_PORT}")

    # Erstelle Model-Verzeichnis falls nicht vorhanden
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Model directory: {MODEL_DIR}")

    # Versuche Modell zu laden
    try:
        from .services.inference_service import inference_service
        loaded = await inference_service.load_model()
        if loaded:
            _model_loaded = True
            _model_version = inference_service.get_model_version()
            logger.info(f"Model loaded: {_model_version}")
        else:
            logger.warning("No model available - service will return errors until model is trained")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
        _model_loaded = False

    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {device_name}")
        else:
            logger.info("GPU not available - running on CPU")
    except ImportError:
        logger.info("PyTorch not available - running in limited mode")

    # Auto-Start Backtest Scheduler wenn via Umgebungsvariable aktiviert
    try:
        from .services.backtesting_service import backtesting_service
        await backtesting_service.auto_start_if_enabled()
    except Exception as e:
        logger.warning(f"Could not auto-start backtest scheduler: {e}")

    # Start Outcome Tracker loop
    outcome_tracking_enabled = os.getenv("OUTCOME_TRACKING_ENABLED", "true").lower() == "true"
    if outcome_tracking_enabled:
        try:
            from .services.outcome_tracker_service import outcome_tracker_service
            await outcome_tracker_service.start_loop()
            logger.info("Outcome tracker loop started")
        except Exception as e:
            logger.warning(f"Could not start outcome tracker: {e}")

    logger.info(f"{SERVICE_NAME} service started successfully")

    yield

    # Shutdown
    logger.info(f"Shutting down {SERVICE_NAME} service")

    # Stop Outcome Tracker loop
    try:
        from .services.outcome_tracker_service import outcome_tracker_service
        if outcome_tracker_service.is_running():
            await outcome_tracker_service.stop_loop()
            logger.info("Outcome tracker loop stopped")
    except Exception as e:
        logger.warning(f"Could not stop outcome tracker: {e}")

    # Stoppe Auto-Backtest Scheduler
    try:
        from .services.backtesting_service import backtesting_service
        await backtesting_service.stop_auto_backtest()
    except Exception as e:
        logger.warning(f"Could not stop backtest scheduler: {e}")


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
        "name": "2. Predictions",
        "description": "Multi-Task Vorhersagen (Preis, Patterns, Regime)"
    },
    {
        "name": "3. Models",
        "description": "Modell-Verwaltung und Informationen"
    },
    {
        "name": "4. History",
        "description": "Prediction-History und Statistiken"
    },
    {
        "name": "5. Feedback",
        "description": "Nutzer-Feedback und Korrekturen"
    },
    {
        "name": "6. Backtesting",
        "description": "Historische Validierung gegen Marktdaten"
    },
    {
        "name": "7. Outcome Tracking",
        "description": "Prediction-Outcome-Tracking für Self-Learning"
    },
]

app = FastAPI(
    title="CNN-LSTM Multi-Task Service",
    description="""
Hybrid CNN-LSTM Service für Multi-Task Trading-Analysen.

## Features

- **Preis-Vorhersage**: Regression für mehrere Zeithorizonte (1h, 4h, 1d, 1w)
- **Pattern-Klassifikation**: 16 Chart-Patterns (Head & Shoulders, Triangles, etc.)
- **Regime-Vorhersage**: 4 Markt-Regime (Bull, Bear, Sideways, High Volatility)

## Architektur

- CNN-Encoder für lokale Feature-Extraktion
- Bidirectional LSTM für sequenzielle Verarbeitung
- Multi-Task Heads mit gewichteter Loss-Funktion

## Timeframes

Unterstuetzt alle Timeframes von M1 bis MN mit angepassten Sequenzlaengen.
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
app.include_router(prediction_router, prefix="/api/v1", tags=["2. Predictions"])
app.include_router(
    revalidation_router,
    prefix="/api/v1",
    tags=["4. History", "5. Feedback", "6. Backtesting"]
)
app.include_router(outcome_router, prefix="/api/v1", tags=["7. Outcome Tracking"])


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
        "status": "healthy" if _model_loaded else "degraded",
        "version": SERVICE_VERSION,
        "model_loaded": _model_loaded,
        "model_version": _model_version,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# Export for uvicorn
# =============================================================================

# Fuer direkten Start: uvicorn src.services.cnn_lstm_app.main:app --host 0.0.0.0 --port 3007
