"""Main entry point for the KI Trading Model service."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
import sys
import os

from .config import settings
from .api import router
from .services.rag_service import RAGService
from .services.timescaledb_sync_service import TimescaleDBSyncService

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
    version="1.0.0",
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
        return FileResponse(dashboard_path)
    return {"error": "Dashboard not found"}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting KI Trading Model Service...")
    logger.info(f"Ollama Host: {settings.ollama_host}")
    logger.info(f"Ollama Model: {settings.ollama_model}")
    logger.info(f"EasyInsight API: {settings.easyinsight_api_url}")
    logger.info(f"ChromaDB Directory: {settings.chroma_persist_directory}")

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


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down KI Trading Model Service...")

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
        "version": "1.0.0",
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
