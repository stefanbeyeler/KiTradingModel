"""
Data Service - Symbol Management, Sync & Pattern Detection Microservice

Handles:
- Symbol Management
- Trading Strategies
- RAG Synchronization
- Query Logs
- Candlestick Pattern Detection
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
    twelvedata_router,
    config_router,
    patterns_router,
    yfinance_router,
    training_router,
    router as general_router
)
from src.services.timescaledb_sync_service import TimescaleDBSyncService
from src.services.rag_service import RAGService
from src.service_registry import register_service

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

# OpenAPI Tags für Swagger UI Dokumentation
openapi_tags = [
    {
        "name": "1. System",
        "description": "Health checks, version info, system status, and general utilities"
    },
    {
        "name": "2. Symbol Management",
        "description": "Manage trading symbols from EasyInsight API"
    },
    {
        "name": "3. Trading Strategies",
        "description": "Create, update, and manage trading strategies"
    },
    {
        "name": "4. Config Export/Import",
        "description": "Export and import configuration (symbols, strategies)"
    },
    {
        "name": "5. Twelve Data API",
        "description": """Access TwelveData market data and technical indicators.

**Technical Indicators:**
- **Trend:** SMA, EMA, WMA, DEMA, TEMA, KAMA, Ichimoku, SAR, Supertrend, VWAP
- **Momentum:** RSI, MACD, Stochastic, Williams %R, CCI, ADX, Aroon, MFI, Connors RSI
- **Volatility:** Bollinger Bands, ATR, Percent B
- **Volume:** OBV, A/D
- **ML Features:** Linear Regression Slope, Hilbert Trend Mode"""
    },
    {
        "name": "6. Yahoo Finance",
        "description": """Yahoo Finance market data integration (2. Fallback nach TwelveData).

**Funktionen:**
- Historische OHLCV-Daten (kostenlos, ohne API-Key)
- Echtzeit-Quotes und Marktdaten
- Symbol-Informationen und Fundamentaldaten
- Unterstützt Aktien, ETFs, Indizes, Forex, Crypto"""
    },
    {
        "name": "7. Candlestick Patterns",
        "description": "Multi-Timeframe Candlestick Pattern Detection (M15, H1, H4, D1). Reversal: Hammer, Shooting Star, Doji, Engulfing, Morning/Evening Star. Continuation: Three White Soldiers, Three Black Crows. Indecision: Spinning Top, Harami."
    },
    {
        "name": "8. NHITS Training",
        "description": "NHITS model training, evaluation, and performance monitoring. Train forecasting models, track progress, and manage retraining workflows."
    },
    {
        "name": "9. RAG Sync",
        "description": "Synchronization of market data to RAG knowledge base (optional, disabled by default)"
    },
    {
        "name": "10. Query Logs & Analytics",
        "description": "Query logging and analytics for monitoring"
    },
]

# Create FastAPI application
app = FastAPI(
    title="Data Service",
    description="""## Data Management Service

Zentraler Service für Datenmanagement im KI Trading Model System.

### Hauptfunktionen

- **Symbol Management**: Verwaltung von Trading-Symbolen (Forex, Crypto, Indizes)
- **Trading Strategies**: Konfiguration und Verwaltung von Handelsstrategien
- **Candlestick Patterns**: Multi-Timeframe Pattern-Erkennung
- **TwelveData Integration**: Marktdaten und technische Indikatoren
- **Yahoo Finance Integration**: Historische Kursdaten als Fallback
- **RAG Sync**: Synchronisation von Marktdaten zur RAG Knowledge Base

### Architektur

Dieser Service fungiert als **Data Gateway** für alle externen Datenquellen:
- **EasyInsight API** (primär) - TimescaleDB mit Echtzeit-Daten
- **TwelveData API** (1. Fallback) - Globale Marktdaten & technische Indikatoren
- **Yahoo Finance** (2. Fallback) - Kostenlose historische Daten

Andere Services (NHITS, RAG, LLM) greifen auf Daten ausschließlich über diesen Service zu.
""",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=openapi_tags,
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

# Include routers - Reihenfolge bestimmt Swagger UI Darstellung
app.include_router(system_router, prefix="/api/v1", tags=["1. System"])
app.include_router(general_router, prefix="/api/v1", tags=["1. System"])
app.include_router(symbol_router, prefix="/api/v1", tags=["2. Symbol Management"])
app.include_router(strategy_router, prefix="/api/v1", tags=["3. Trading Strategies"])
app.include_router(config_router, prefix="/api/v1", tags=["4. Config Export/Import"])
app.include_router(twelvedata_router, prefix="/api/v1", tags=["5. Twelve Data API"])
app.include_router(yfinance_router, prefix="/api/v1", tags=["6. Yahoo Finance"])
app.include_router(patterns_router, prefix="/api/v1", tags=["7. Candlestick Patterns"])
app.include_router(training_router, prefix="/api/v1", tags=["8. NHITS Training"])
app.include_router(sync_router, prefix="/api/v1", tags=["9. RAG Sync"])
app.include_router(query_log_router, prefix="/api/v1", tags=["10. Query Logs & Analytics"])


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
        register_service('rag_service', rag_service)
        logger.info("RAG Service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        rag_service = None

    # Initialize TimescaleDB Sync Service
    try:
        sync_service = TimescaleDBSyncService(rag_service)
        register_service('sync_service', sync_service)
        logger.info("TimescaleDB Sync Service initialized")

        # Auto-start sync if configured
        if settings.rag_sync_enabled:
            await sync_service.start()
            logger.info("TimescaleDB Sync Service started (auto-start)")

    except Exception as e:
        logger.error(f"Failed to initialize Sync Service: {e}")

    # Start Pattern History Auto-Scan
    try:
        from src.services.pattern_history_service import pattern_history_service
        await pattern_history_service.start()
        logger.info("Pattern History Auto-Scan started")
    except Exception as e:
        logger.error(f"Failed to start Pattern History Auto-Scan: {e}")

    logger.info("Data Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Data Service...")

    # Stop sync service if running
    if sync_service and getattr(sync_service, '_running', False):
        await sync_service.stop()
        logger.info("TimescaleDB Sync Service stopped")

    # Stop Pattern History Auto-Scan
    try:
        from src.services.pattern_history_service import pattern_history_service
        if pattern_history_service._running:
            await pattern_history_service.stop()
            logger.info("Pattern History Auto-Scan stopped")
    except Exception as e:
        logger.error(f"Failed to stop Pattern History Auto-Scan: {e}")

    logger.info("Data Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    sync_status = "unknown"
    if sync_service:
        sync_status = "running" if getattr(sync_service, '_running', False) else "stopped"

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
