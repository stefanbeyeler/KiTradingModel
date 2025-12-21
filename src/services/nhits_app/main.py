"""
NHITS Service - Training & Forecasting Microservice

Handles:
- NHITS Model Training
- Price Forecasting
- Model Performance Evaluation
- Event-Based Training
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.config import settings
from src.version import VERSION
from src.api.routes import forecast_router, training_router, system_router, backup_router
from src.services.nhits_training_service import nhits_training_service
from src.services.event_based_training_service import event_based_training_service
from src.services.model_improvement_service import model_improvement_service

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>NHITS</cyan> | <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/nhits_service_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level
)

# Create FastAPI application
app = FastAPI(
    title="NHITS Service",
    description="Neural Hierarchical Interpolation for Time Series - Training & Forecasting",
    version=VERSION,
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
# IMPORTANT: training_router must come before forecast_router because
# forecast_router has a catch-all /forecast/{symbol} route that would
# otherwise intercept routes like /forecast/performance and /forecast/evaluated
app.include_router(training_router, prefix="/api/v1", tags=["NHITS Training"])
app.include_router(forecast_router, prefix="/api/v1", tags=["NHITS Forecast"])
app.include_router(system_router, prefix="/api/v1", tags=["System & Monitoring"])
app.include_router(backup_router, prefix="/api/v1", tags=["Backup & Restore"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting NHITS Service...")
    logger.info(f"Version: {VERSION}")
    logger.info(f"GPU Enabled: {settings.nhits_use_gpu}")

    # Start auto-retraining service if configured
    if settings.nhits_auto_retrain_days > 0:
        logger.info(f"Auto-retraining enabled: every {settings.nhits_auto_retrain_days} days")

    logger.info("NHITS Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down NHITS Service...")

    # Stop event-based training monitor if running
    if event_based_training_service._running:
        await event_based_training_service.stop()

    logger.info("NHITS Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": "nhits",
        "status": "healthy",
        "version": VERSION,
        "gpu_enabled": settings.nhits_use_gpu,
        "training_in_progress": nhits_training_service.get_status().get("training_in_progress", False)
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "NHITS Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=settings.log_level.lower()
    )
