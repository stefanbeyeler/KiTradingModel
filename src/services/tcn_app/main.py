"""
TCN-Pattern Service - Chart Pattern Detection using Temporal Convolutional Networks

Port: 3005
Detects chart patterns like:
- Head & Shoulders, Double Top/Bottom
- Triangles, Flags, Wedges
- Channels and more
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import detection_router, training_router, system_router, history_router
from .services.pattern_detection_service import pattern_detection_service
from .services.tcn_pattern_history_service import tcn_pattern_history_service

# Configuration
VERSION = "1.0.0"
SERVICE_NAME = "tcn-pattern"
SERVICE_PORT = int(os.getenv("PORT", "3005"))
ROOT_PATH = os.getenv("ROOT_PATH", "")

# Logging setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>TCN</cyan> | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger.add(
    "logs/tcn_service.log",
    rotation="10 MB",
    retention="7 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info(f"Starting TCN-Pattern Service v{VERSION} on port {SERVICE_PORT}")

    # Load model if available
    model_path = os.getenv("TCN_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        pattern_detection_service.load_model(model_path)
    else:
        logger.info("No pre-trained model found, using rule-based detection only")
        pattern_detection_service.load_model(None)

    # Start auto-scan for pattern history
    auto_scan_enabled = os.getenv("TCN_AUTO_SCAN", "true").lower() == "true"
    if auto_scan_enabled:
        await tcn_pattern_history_service.start_auto_scan()
        logger.info("TCN Pattern History auto-scan started")

    yield

    # Stop auto-scan on shutdown
    if tcn_pattern_history_service.is_scan_running():
        await tcn_pattern_history_service.stop_auto_scan()
        logger.info("TCN Pattern History auto-scan stopped")

    logger.info("Shutting down TCN-Pattern Service...")


# Create FastAPI app
app = FastAPI(
    title="TCN-Pattern Service",
    description="""
    Chart pattern detection service using Temporal Convolutional Networks.

    ## Features

    - **Deep Learning Detection**: TCN-based pattern recognition
    - **Rule-Based Backup**: Classical pattern detection algorithms
    - **16 Pattern Types**: Head & Shoulders, Double Top/Bottom, Triangles, etc.
    - **Multi-Timeframe**: Supports all common timeframes

    ## Supported Patterns

    ### Reversal Patterns
    - Head & Shoulders / Inverse H&S
    - Double Top / Double Bottom
    - Triple Top / Triple Bottom
    - Cup and Handle
    - Rising / Falling Wedge

    ### Continuation Patterns
    - Ascending / Descending Triangle
    - Symmetrical Triangle
    - Bull / Bear Flag

    ### Trend Patterns
    - Channel Up / Channel Down

    ## Usage

    1. Use `/api/v1/detect` to analyze a specific symbol
    2. Use `/api/v1/scan` to scan multiple symbols
    3. Train custom models with `/api/v1/train`
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
app.include_router(detection_router, prefix="/api/v1", tags=["2. Pattern Detection"])
app.include_router(history_router, prefix="/api/v1", tags=["3. Pattern History"])
app.include_router(training_router, prefix="/api/v1", tags=["4. Model Training"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "TCN-Pattern Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "detect": "/api/v1/detect",
            "scan": "/api/v1/scan",
            "patterns": "/api/v1/patterns",
            "history": "/api/v1/history",
            "history_by_symbol": "/api/v1/history/{symbol}",
            "history_statistics": "/api/v1/history/statistics",
            "train": "/api/v1/train",
            "models": "/api/v1/models"
        }
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "healthy", "service": SERVICE_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.tcn_app.main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False
    )
