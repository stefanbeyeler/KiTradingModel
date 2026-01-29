"""
HMM Training Service - Dedicated Training Container

Port: 3014

This service handles HMM and LightGBM Signal Scorer training separately
from the inference service. Training is CPU intensive and should not
block regime detection API requests.

Architecture:
    - HMM Service (Port 3004): Inference only - fast regime detection API
    - HMM-Train Service (Port 3014): Training only - background model training
    - Shared volume: /app/data/models/hmm - trained models accessible by both

Model Types:
    - HMM: Gaussian HMM for market regime detection (per symbol)
    - Scorer: LightGBM signal scoring model (trained across symbols)
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.config.microservices import microservices_config
from src.services.hmm_train_app.routers import training_router, system_router
from src.services.hmm_train_app.routers.training_router import validation_router, scheduler_router, self_learning_router
from src.services.hmm_train_app.services.training_service import training_service
from src.services.hmm_train_app.services.scheduler_service import scheduler_service
from src.services.hmm_train_app.services.self_learning_service import self_learning_service
from src.shared.test_health_router import create_test_health_router
from src.shared.health import get_test_unhealthy_status

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>HMM-TRAIN</cyan> | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger.add(
    "logs/hmm_train_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)

# Create FastAPI application
app = FastAPI(
    title="HMM Training Service",
    description="Dedicated training service for HMM and LightGBM Scorer models",
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
app.include_router(validation_router, prefix="/api/v1")
app.include_router(scheduler_router, prefix="/api/v1")
app.include_router(self_learning_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")

# Test-Health-Router
test_health_router = create_test_health_router("hmm-train")
app.include_router(test_health_router, prefix="/api/v1", tags=["System"])


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Starting HMM Training Service...")
    logger.info(f"Model directory: {os.getenv('MODEL_DIR', '/app/data/models/hmm')}")
    logger.info(f"Data Service URL: {os.getenv('DATA_SERVICE_URL', microservices_config.data_service_url)}")
    logger.info(f"HMM Service URL: {os.getenv('HMM_SERVICE_URL', microservices_config.hmm_service_url)}")
    logger.info(f"Validation enabled: {os.getenv('HMM_ENABLE_VALIDATION', 'true')}")

    # Check library availability
    try:
        from hmmlearn import hmm
        logger.info("hmmlearn available")
    except ImportError:
        logger.warning("hmmlearn not available - HMM training disabled")

    try:
        import lightgbm
        logger.info("lightgbm available")
    except ImportError:
        logger.warning("lightgbm not available - Scorer training disabled")

    # Start the scheduler
    await scheduler_service.start()
    logger.info("HMM Scheduler started")

    # Start self-learning monitor if enabled
    if os.getenv("HMM_SELFLEARNING_ENABLED", "true").lower() == "true":
        await self_learning_service.start_monitor()
        logger.info("Self-Learning Monitor started")

    logger.info("HMM Training Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down HMM Training Service...")

    # Stop self-learning monitor
    await self_learning_service.stop_monitor()
    logger.info("Self-Learning Monitor stopped")

    # Stop the scheduler
    await scheduler_service.stop()
    logger.info("HMM Scheduler stopped")

    # Cancel any running training
    if training_service.is_training():
        await training_service.cancel_training()
        logger.info("Cancelled running training job")

    await training_service.close()
    logger.info("HMM Training Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint - must respond quickly even during training."""
    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status("hmm-train")
    is_unhealthy = test_status.get("test_unhealthy", False)

    response = {
        "service": "hmm-train",
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
        "service": "HMM Training Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "training": "/api/v1/train"
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3014"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
