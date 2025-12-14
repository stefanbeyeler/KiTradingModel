"""
LLM Service - Analysis & RAG Microservice

Handles:
- LLM-based Trading Analysis
- RAG (Retrieval Augmented Generation)
- Knowledge Base Management
- Strategy Evaluation
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
from src.api.routes import llm_router, rag_router, trading_router
from src.services import LLMService
from src.services.rag_service import RAGService

# Global service instances
llm_service = None
rag_service = None

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>LLM</cyan> | <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/llm_service_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level
)

# Create FastAPI application
app = FastAPI(
    title="LLM Service",
    description="Large Language Model Service - Trading Analysis & RAG",
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
app.include_router(llm_router, prefix="/api/v1", tags=["LLM Service"])
app.include_router(rag_router, prefix="/api/v1", tags=["RAG & Knowledge Base"])
app.include_router(trading_router, prefix="/api/v1", tags=["Trading Analysis"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global llm_service, rag_service

    logger.info("Starting LLM Service...")
    logger.info(f"Version: {VERSION}")
    logger.info(f"LLM Model: {settings.ollama_model}")
    logger.info(f"Ollama URL: {settings.ollama_host}")

    # Initialize RAG Service
    try:
        rag_service = RAGService()
        logger.info("RAG Service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")

    # Initialize LLM Service
    try:
        llm_service = LLMService()
        logger.info("LLM Service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Service: {e}")

    logger.info("LLM Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LLM Service...")

    # Cleanup services if needed
    if rag_service:
        # Persist RAG database
        try:
            rag_service.persist()
            logger.info("RAG database persisted")
        except Exception as e:
            logger.error(f"Failed to persist RAG database: {e}")

    logger.info("LLM Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    llm_status = "unknown"
    if llm_service:
        try:
            # Quick LLM check
            llm_status = "healthy"
        except Exception:
            llm_status = "unhealthy"

    return {
        "service": "llm",
        "status": "healthy",
        "version": VERSION,
        "llm_model": settings.ollama_model,
        "llm_status": llm_status,
        "rag_enabled": rag_service is not None
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3002"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=settings.log_level.lower()
    )
