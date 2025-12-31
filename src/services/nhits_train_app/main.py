"""
NHITS Training Service - Dedicated Training Container

Port: 3012

This service handles NHITS model training separately from the inference service.
Training is CPU/GPU intensive and should not block forecast API requests.

Architecture:
    - NHITS Service (Port 3002): Inference only - fast forecast API responses
    - NHITS-Train Service (Port 3012): Training only - background model training
    - Shared volume: /app/data/models/nhits - trained models accessible by both
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.services.nhits_train_app.routers import training_router, system_router
from src.services.nhits_train_app.services.training_service import training_service
from src.shared.test_health_router import create_test_health_router
from src.shared.health import get_test_unhealthy_status

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>NHITS-TRAIN</cyan> | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger.add(
    "logs/nhits_train_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)

# Create FastAPI application
app = FastAPI(
    title="NHITS Training Service",
    description="Dedicated training service for NHITS models - runs separately from inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=os.getenv("ROOT_PATH", "")
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")

# Test-Health-Router
test_health_router = create_test_health_router("nhits-train")
app.include_router(test_health_router, prefix="/api/v1", tags=["System"])


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting NHITS Training Service...")
    logger.info(f"Model directory: {os.getenv('MODEL_DIR', '/app/data/models/nhits')}")
    logger.info(f"Data Service URL: {os.getenv('DATA_SERVICE_URL', 'http://trading-data:3001')}")
    logger.info(f"NHITS Service URL: {os.getenv('NHITS_SERVICE_URL', 'http://trading-nhits:3002')}")

    # Check PyTorch availability
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"PyTorch available - device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch not available - training disabled")

    logger.info("NHITS Training Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down NHITS Training Service...")

    # Cancel any running training
    if training_service.is_training():
        await training_service.cancel_training()
        logger.info("Cancelled running training job")

    await training_service.close()
    logger.info("NHITS Training Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint - must respond quickly even during training."""
    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status("nhits-train")
    is_unhealthy = test_status.get("test_unhealthy", False)

    response = {
        "service": "nhits-train",
        "status": "unhealthy" if is_unhealthy else "healthy",
        "training_in_progress": training_service.is_training()
    }

    # Test-Status hinzufügen wenn aktiv
    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "NHITS Training Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "training": "/api/v1/train"
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3012"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
