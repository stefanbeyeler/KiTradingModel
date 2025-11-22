"""Main entry point for the KI Trading Model service."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from .config import settings
from .api import router


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


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting KI Trading Model Service...")
    logger.info(f"Ollama Host: {settings.ollama_host}")
    logger.info(f"Ollama Model: {settings.ollama_model}")
    logger.info(f"EasyInsight API: {settings.easyinsight_api_url}")
    logger.info(f"ChromaDB Directory: {settings.chroma_persist_directory}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down KI Trading Model Service...")


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
