"""
Data Service - Symbol Management & Sync Microservice

Handles:
- Symbol Management
- Trading Strategies
- TimescaleDB Synchronization
- Query Logs
- System Monitoring
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
from src.api.routes import (
    symbol_router,
    strategy_router,
    sync_router,
    system_router,
    query_log_router,
    router as general_router
)
from src.services.timescaledb_sync_service import TimescaleDBSyncService
from src.services.rag_service import RAGService

# Global service instances
sync_service = None
rag_service = None

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>DATA</cyan> | <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/data_service_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level
)

# Create FastAPI application
app = FastAPI(
    title="Data Service",
    description="Data Management Service - Symbols, Strategies, Sync & Monitoring",
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

# Include routers
app.include_router(general_router, prefix="/api/v1", tags=["General"])
app.include_router(symbol_router, prefix="/api/v1", tags=["📈 Symbol Management"])
app.include_router(strategy_router, prefix="/api/v1", tags=["🎯 Trading Strategies"])
app.include_router(sync_router, prefix="/api/v1", tags=["🔄 TimescaleDB Sync"])
app.include_router(system_router, prefix="/api/v1", tags=["🖥️ System & Monitoring"])
app.include_router(query_log_router, prefix="/api/v1", tags=["📝 Query Logs & Analytics"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global sync_service, rag_service

    logger.info("Starting Data Service...")
    logger.info(f"Version: {VERSION}")
    logger.info(f"EasyInsight API: {settings.easyinsight_api_url}")

    # Initialize RAG Service (for sync service)
    try:
        rag_service = RAGService()
        logger.info("RAG Service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        rag_service = None

    # Initialize TimescaleDB Sync Service
    try:
        sync_service = TimescaleDBSyncService(rag_service)
        logger.info("TimescaleDB Sync Service initialized")

        # Auto-start sync if configured
        if settings.rag_sync_enabled:
            await sync_service.start()
            logger.info("TimescaleDB Sync Service started (auto-start)")

    except Exception as e:
        logger.error(f"Failed to initialize Sync Service: {e}")

    logger.info("Data Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Data Service...")

    # Stop sync service if running
    if sync_service and getattr(sync_service, '_running', False):
        await sync_service.stop()
        logger.info("TimescaleDB Sync Service stopped")

    logger.info("Data Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    sync_status = "unknown"
    if sync_service:
        sync_status = "running" if getattr(sync_service, 'is_running', False) else "stopped"

    return {
        "service": "data",
        "status": "healthy",
        "version": VERSION,
        "easyinsight_api": settings.easyinsight_api_url,
        "sync_service_status": sync_status
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Data Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3003"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=settings.log_level.lower()
    )
