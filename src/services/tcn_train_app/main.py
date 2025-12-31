"""
TCN Training Service - Dedicated training container for TCN pattern models.

This service runs separately from the TCN inference service to prevent
training from blocking API requests.

Features:
- Scheduled and on-demand model training
- Writes models to shared volume
- Notifies TCN service when new model is available
- Low priority resource allocation
"""

import os
import sys
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .services.training_service import training_service
from .services.training_scheduler import training_scheduler
from .routers import training_router, system_router

# Import für Test-Health-Funktionalität
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.shared.test_health_router import create_test_health_router
from src.shared.health import get_test_unhealthy_status

# Configuration
VERSION = "1.0.0"
SERVICE_NAME = "tcn-train"
SERVICE_PORT = int(os.getenv("PORT", "3013"))
ROOT_PATH = os.getenv("ROOT_PATH", "")

# Logging setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>TCN-TRAIN</cyan> | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger.add(
    "logs/tcn_train_service.log",
    rotation="10 MB",
    retention="7 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info(f"Starting TCN Training Service v{VERSION} on port {SERVICE_PORT}")

    # Start scheduler if enabled
    auto_train_enabled = os.getenv("TCN_AUTO_TRAIN", "true").lower() == "true"
    if auto_train_enabled:
        training_scheduler.start()
        logger.info("Training scheduler started")

    yield

    # Stop scheduler
    training_scheduler.stop()
    logger.info("Shutting down TCN Training Service...")


# Create FastAPI app
app = FastAPI(
    title="TCN Training Service",
    description="""
    Dedicated training service for TCN pattern recognition models.

    ## Purpose

    This service handles all training operations separately from the inference service,
    ensuring that long-running training jobs don't block pattern detection API requests.

    ## Features

    - **Scheduled Training**: Automatic retraining on configurable schedule
    - **On-Demand Training**: Manual training triggers via API
    - **Model Management**: List, cleanup, and manage trained models
    - **Hot-Reload Support**: Notifies inference service when new model is available

    ## Architecture

    ```
    TCN-Train Service (this)     TCN Inference Service
           │                            │
           │  trains model              │  loads model
           ▼                            ▼
        ┌─────────────────────────────────┐
        │     Shared Volume: /models      │
        └─────────────────────────────────┘
    ```
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
app.include_router(training_router, prefix="/api/v1", tags=["2. Training"])

# Test-Health-Router
test_health_router = create_test_health_router("tcn-train")
app.include_router(test_health_router, prefix="/api/v1", tags=["1. System & Monitoring"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "TCN Training Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "train": "/api/v1/train",
            "train_status": "/api/v1/train/status",
            "train_history": "/api/v1/train/history",
            "models": "/api/v1/models",
            "scheduler": "/api/v1/scheduler",
            "scheduler_config": "/api/v1/scheduler/config"
        }
    }


@app.get("/health")
async def health():
    """Simple health check - always responds quickly."""
    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status("tcn-train")
    is_unhealthy = test_status.get("test_unhealthy", False)

    response = {
        "status": "unhealthy" if is_unhealthy else "healthy",
        "service": SERVICE_NAME,
        "training_active": training_service.is_training()
    }

    # Test-Status hinzufügen wenn aktiv
    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.tcn_train_app.main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False
    )
