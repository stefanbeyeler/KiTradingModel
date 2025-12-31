"""
Candlestick Pattern Training Service - Model Training Microservice

Port: 3016
Handles training of TCN models for candlestick pattern recognition.

Features:
- PyTorch TCN model training
- Scheduled automatic training
- Training job management
- Model versioning
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import training_router, system_router
from .services.training_service import training_service
from .services.training_scheduler import training_scheduler

# Import für Test-Health-Funktionalität
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.shared.test_health_router import create_test_health_router
from src.shared.health import get_test_unhealthy_status

# Configuration
VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_NAME = "candlestick-train"
SERVICE_PORT = int(os.getenv("PORT", "3016"))
ROOT_PATH = os.getenv("ROOT_PATH", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Logging setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>CANDLE-TRAIN</cyan> | <level>{message}</level>",
    level=LOG_LEVEL
)
logger.add(
    "logs/candlestick_train_service.log",
    rotation="10 MB",
    retention="7 days",
    level=LOG_LEVEL
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info(f"Starting Candlestick Training Service v{VERSION} on port {SERVICE_PORT}")

    # Start scheduler if enabled
    await training_scheduler.start()

    yield

    # Stop scheduler
    await training_scheduler.stop()

    # Close HTTP client
    await training_service.close()

    logger.info("Shutting down Candlestick Training Service...")


# OpenAPI Tags
openapi_tags = [
    {
        "name": "1. System",
        "description": "Health checks, version info, and system monitoring"
    },
    {
        "name": "2. Training",
        "description": "Training-Jobs starten, überwachen und verwalten"
    },
]


# Create FastAPI app
app = FastAPI(
    title="Candlestick Training Service",
    description="""
## Candlestick Pattern Training Service

Training-Service fuer TCN-basierte Candlestick-Pattern-Erkennung.

### Features

- **PyTorch TCN Training**: Temporal Convolutional Network Training
- **Job Management**: Training-Jobs starten, stoppen, ueberwachen
- **Scheduled Training**: Automatisches Training nach Zeitplan
- **Model Versioning**: Modell-Versionierung und -Verwaltung

### Architektur

Dieser Service trainiert Modelle, die vom Candlestick Service (Port 3006)
fuer die Pattern-Erkennung verwendet werden.

**Datenbezug:** Data Service (Port 3001)
**Modell-Nutzung:** Candlestick Service (Port 3006)
""",
    version=VERSION,
    root_path=ROOT_PATH,
    lifespan=lifespan,
    openapi_tags=openapi_tags,
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
app.include_router(system_router, prefix="/api/v1", tags=["1. System"])
app.include_router(training_router, prefix="/api/v1", tags=["2. Training"])

# Test-Health-Router
test_health_router = create_test_health_router("candlestick-train")
app.include_router(test_health_router, prefix="/api/v1", tags=["1. System"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Candlestick Training Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "train": "/api/v1/train",
            "status": "/api/v1/train/status",
            "progress": "/api/v1/train/progress",
            "jobs": "/api/v1/train/jobs",
            "model": "/api/v1/model",
            "scheduler": "/api/v1/scheduler",
        }
    }


@app.get("/health")
async def health():
    """Simple health check at root level."""
    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status("candlestick-train")
    is_unhealthy = test_status.get("test_unhealthy", False)

    response = {
        "status": "unhealthy" if is_unhealthy else "healthy",
        "service": SERVICE_NAME
    }

    # Test-Status hinzufügen wenn aktiv
    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.candlestick_train_app.main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False
    )
