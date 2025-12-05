"""Main entry point for the KI Trading Model service."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger
import sys
import os

from .config import settings
from .version import VERSION, RELEASE_DATE
from .api import router
from .services.rag_service import RAGService
from .services.timescaledb_sync_service import TimescaleDBSyncService
from .services.nhits_training_service import nhits_training_service
from .services.model_improvement_service import model_improvement_service

# Global service instances
rag_service = RAGService()
sync_service = TimescaleDBSyncService(rag_service)


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/ki_trading_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level
)


# Create FastAPI application
app = FastAPI(
    title="KI Trading Model",
    description="Lokaler KI-Service für Handelsempfehlungen basierend auf Llama 3.1 70B und RAG",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Trading Analysis"])

# Mount static files for dashboard
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/dashboard")
async def dashboard():
    """Serve the monitoring dashboard."""
    dashboard_path = os.path.join(static_dir, "index.html")
    if os.path.exists(dashboard_path):
        # Read the file content directly to bypass any caching
        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"error": "Dashboard not found"}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting KI Trading Model Service...")
    logger.info(f"Ollama Host: {settings.ollama_host}")
    logger.info(f"Ollama Model: {settings.ollama_model}")
    logger.info(f"TimescaleDB: {settings.timescaledb_host}:{settings.timescaledb_port}/{settings.timescaledb_database}")
    logger.info(f"FAISS Directory: {settings.faiss_persist_directory}")

    # Start TimescaleDB sync service if enabled
    if settings.rag_sync_enabled:
        try:
            await sync_service.start()
            logger.info(
                f"TimescaleDB sync started - "
                f"Host: {settings.timescaledb_host}:{settings.timescaledb_port}, "
                f"Interval: {settings.rag_sync_interval_seconds}s"
            )
        except Exception as e:
            logger.warning(f"Failed to start TimescaleDB sync: {e}")
            logger.info("Service will continue without automatic RAG sync")

    # Initialize NHITS training service
    if settings.nhits_enabled:
        try:
            # Connect training service to database
            if sync_service._pool:
                await nhits_training_service.connect(sync_service._pool)
                logger.info("NHITS training service connected to database")

            # Start scheduled training if enabled
            if settings.nhits_scheduled_training_enabled:
                await nhits_training_service.start()
                logger.info(
                    f"NHITS scheduled training started - "
                    f"Interval: {settings.nhits_scheduled_training_interval_hours}h"
                )

            # Run startup training if enabled
            if settings.nhits_train_on_startup:
                logger.info("Starting NHITS startup training...")
                import asyncio
                asyncio.create_task(
                    nhits_training_service.train_all_symbols(force=False)
                )

            # Start model improvement service for automatic feedback learning
            if sync_service._pool:
                await model_improvement_service.start_evaluation_loop(
                    sync_service._pool,
                    interval_seconds=300  # Evaluate every 5 minutes
                )
                logger.info("Model improvement service started (feedback learning enabled)")

        except Exception as e:
            logger.warning(f"Failed to initialize NHITS training service: {e}")
            logger.info("Service will continue without automatic NHITS training")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down KI Trading Model Service...")

    # Stop model improvement service
    if model_improvement_service._running:
        await model_improvement_service.stop()
        logger.info("Model improvement service stopped")

    # Stop NHITS training service
    if nhits_training_service._running:
        await nhits_training_service.stop()
        logger.info("NHITS training service stopped")

    # Stop sync service
    if sync_service._running:
        await sync_service.stop()
        await sync_service.disconnect()
        logger.info("TimescaleDB sync service stopped")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "KI Trading Model",
        "version": f"v{VERSION}",
        "release_date": RELEASE_DATE,
        "description": "Lokaler KI-Service für Handelsempfehlungen",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


def main():
    """Run the service."""
    uvicorn.run(
        "src.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
