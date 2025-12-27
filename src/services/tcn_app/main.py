"""
TCN-Pattern Service - Chart Pattern Detection using Temporal Convolutional Networks

Port: 3003
Detects chart patterns like:
- Head & Shoulders, Double Top/Bottom
- Triangles, Flags, Wedges
- Channels and more

Note: Training has been moved to a separate service (tcn_train_app) to prevent
training from blocking pattern detection API requests.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import detection_router, system_router, history_router, crt_router
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

    ## CRT (Candle Range Theory)

    New! ICT/SMC-style Candle Range Theory detection:
    - Session-based H4 range tracking (1/5/9 AM EST)
    - Liquidity purge detection
    - Re-entry signal generation
    - Multi-service confirmation (HMM, NHITS, TCN)

    ## Usage

    1. Use `/api/v1/detect` to analyze a specific symbol
    2. Use `/api/v1/scan` to scan multiple symbols
    3. Use `/api/v1/crt/status/{symbol}` for CRT range status
    4. Use `/api/v1/crt/signal/{symbol}` for CRT trading signals
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
# Note: Training router moved to tcn_train_app service
app.include_router(system_router, prefix="/api/v1", tags=["1. System & Monitoring"])
app.include_router(detection_router, prefix="/api/v1", tags=["2. Pattern Detection"])
app.include_router(history_router, prefix="/api/v1", tags=["3. Pattern History"])
app.include_router(crt_router, prefix="/api/v1", tags=["4. CRT (Candle Range Theory)"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "TCN-Pattern Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "note": "Training moved to tcn-train service (port 3013)",
        "endpoints": {
            "detect": "/api/v1/detect",
            "scan": "/api/v1/scan",
            "patterns": "/api/v1/patterns",
            "history": "/api/v1/history",
            "history_by_symbol": "/api/v1/history/{symbol}",
            "history_statistics": "/api/v1/history/statistics",
            "model": "/api/v1/model",
            "model_reload": "/api/v1/model/reload",
            "crt_status": "/api/v1/crt/status/{symbol}",
            "crt_signal": "/api/v1/crt/signal/{symbol}",
            "crt_session": "/api/v1/crt/session",
            "crt_scan": "/api/v1/crt/scan",
            "crt_ranges": "/api/v1/crt/ranges"
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
