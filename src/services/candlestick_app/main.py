"""
Candlestick Pattern Service - Candlestick Pattern Detection Microservice

Port: 3006
Detects candlestick patterns like:
- Reversal: Hammer, Shooting Star, Doji, Engulfing, Morning/Evening Star
- Continuation: Three White Soldiers, Three Black Crows
- Indecision: Spinning Top, Harami

Multi-Timeframe: M5, M15, H1, H4, D1

Note: Training has been moved to a separate service (candlestick_train_app)
to prevent training from blocking pattern detection API requests.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import detection_router, system_router, history_router, claude_validator_router
from .services.pattern_detection_service import candlestick_pattern_service
from .services.pattern_history_service import pattern_history_service
from .services.claude_validator_service import claude_validator_service

# Configuration
VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_NAME = "candlestick"
SERVICE_PORT = int(os.getenv("PORT", "3006"))
ROOT_PATH = os.getenv("ROOT_PATH", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Logging setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>CANDLE</cyan> | <level>{message}</level>",
    level=LOG_LEVEL
)
logger.add(
    "logs/candlestick_service.log",
    rotation="10 MB",
    retention="7 days",
    level=LOG_LEVEL
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info(f"Starting Candlestick Pattern Service v{VERSION} on port {SERVICE_PORT}")

    # Start auto-scan for pattern history
    auto_scan_enabled = os.getenv("AUTO_SCAN_ENABLED", "true").lower() == "true"
    if auto_scan_enabled:
        await pattern_history_service.start()
        logger.info("Pattern History auto-scan started")

    yield

    # Stop auto-scan on shutdown
    if pattern_history_service.is_running():
        await pattern_history_service.stop()
        logger.info("Pattern History auto-scan stopped")

    # Close HTTP client
    await candlestick_pattern_service.close()

    logger.info("Shutting down Candlestick Pattern Service...")


# OpenAPI Tags
openapi_tags = [
    {
        "name": "1. System",
        "description": "Health checks und Service-Informationen"
    },
    {
        "name": "2. Pattern Detection",
        "description": "Candlestick-Muster erkennen (21 Typen, Multi-Timeframe)"
    },
    {
        "name": "3. Pattern History",
        "description": "Pattern-Historie, Auto-Scan und Statistiken"
    },
    {
        "name": "4. Claude QA",
        "description": "Externe Pattern-Validierung mit Claude Vision AI"
    },
]


# Create FastAPI app
app = FastAPI(
    title="Candlestick Pattern Service",
    description="""
## Candlestick Pattern Detection Service

Erkennt Candlestick-Muster ueber mehrere Timeframes fuer Trading-Signale.

### Features

- **21 Pattern Types**: Umfassende Pattern-Erkennung
- **Multi-Timeframe**: M5, M15, H1, H4, D1
- **Confidence Scoring**: 0.0-1.0 Konfidenz pro Pattern
- **Auto-Scan**: Periodisches Scannen aller Symbole
- **Pattern History**: Speicherung und Abruf erkannter Patterns

### Pattern-Kategorien

**Reversal (Umkehr):**
- Hammer, Shooting Star, Doji-Varianten
- Engulfing, Morning/Evening Star

**Continuation (Fortsetzung):**
- Three White Soldiers/Black Crows
- Rising/Falling Three Methods

**Indecision (Unentschlossenheit):**
- Spinning Top, Harami-Varianten

### Architektur

Dieser Service ruft Marktdaten vom Data Service ab (Port 3001).
Training erfolgt im separaten Candlestick-Train Service (Port 3016).
""",
    version=VERSION,
    root_path=ROOT_PATH,
    lifespan=lifespan,
    openapi_tags=openapi_tags,
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
app.include_router(system_router, prefix="/api/v1", tags=["1. System"])
app.include_router(detection_router, prefix="/api/v1", tags=["2. Pattern Detection"])
app.include_router(history_router, prefix="/api/v1", tags=["3. Pattern History"])
app.include_router(claude_validator_router, prefix="/api/v1/claude", tags=["4. Claude QA"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Candlestick Pattern Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "note": "Training moved to candlestick-train service (port 3016)",
        "endpoints": {
            "scan": "/api/v1/scan",
            "scan_symbol": "/api/v1/scan/{symbol}",
            "scan_all": "/api/v1/scan-all",
            "patterns": "/api/v1/patterns",
            "chart": "/api/v1/chart/{symbol}",
            "history": "/api/v1/history",
            "history_by_symbol": "/api/v1/history/{symbol}",
            "history_statistics": "/api/v1/history/statistics",
            "scan_trigger": "/api/v1/history/scan",
        }
    }


@app.get("/health")
async def health():
    """Simple health check at root level."""
    return {"status": "healthy", "service": SERVICE_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.candlestick_app.main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False
    )
