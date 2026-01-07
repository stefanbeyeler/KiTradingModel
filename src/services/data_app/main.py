"""
Data Service - Symbol Management & Data Gateway Microservice

Handles:
- Symbol Management
- Trading Strategies
- RAG Synchronization
- Query Logs
- External Data Sources Gateway
- System Monitoring

Note: Candlestick Pattern Detection has been moved to:
- Candlestick Service (Port 3006) - Pattern Detection
- Candlestick Train Service (Port 3016) - Model Training
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
    easyinsight_router,
    config_router,
    yfinance_router,
    external_sources_router,
    indicators_router,
    router as general_router
)
# Note: patterns_router has been moved to Candlestick Service (Port 3006)
# Note: training_router (NHITS Training) is NOT included here
# These endpoints belong to the NHITS Service (Port 3002)
from src.api.testing_routes import testing_router
from src.services.training_data_cache_service import training_data_cache
from src.service_registry import register_service
from src.shared.test_health_router import create_test_health_router
from src.shared.health import is_test_unhealthy, get_test_unhealthy_status
import asyncio

# Optional imports for RAG sync (requires sentence_transformers)
try:
    from src.services.timescaledb_sync_service import TimescaleDBSyncService
    from src.services.rag_service import RAGService
    _rag_available = True
except ImportError:
    TimescaleDBSyncService = None
    RAGService = None
    _rag_available = False
    logger.warning("RAG services not available - sentence_transformers not installed")

# Global service instances
sync_service = None
rag_service = None
_cache_cleanup_task = None
_prefetch_task = None

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
        "description": """TwelveData API - Primäre Datenquelle für OHLC-Daten aller Timeframes.

**Unterstützte Timeframes:** M1, M5, M15, M30, M45, H1, H2, H4, D1, W1, MN

**Technical Indicators:**
- **Trend:** SMA, EMA, WMA, DEMA, TEMA, KAMA, Ichimoku, SAR, Supertrend, VWAP
- **Momentum:** RSI, MACD, Stochastic, Williams %R, CCI, ADX, Aroon, MFI, Connors RSI
- **Volatility:** Bollinger Bands, ATR, Percent B
- **Volume:** OBV, A/D
- **ML Features:** Linear Regression Slope, Hilbert Trend Mode"""
    },
    {
        "name": "6. EasyInsight API",
        "description": """EasyInsight API - TimescaleDB mit Echtzeit-Marktdaten und vorberechneten Indikatoren.

**Funktionen:**
- OHLCV-Daten mit H1 Timeframe
- Vorberechnete technische Indikatoren (RSI, MACD, BBands, ATR, ADX, etc.)
- Symbol-Informationen und Kategorien
- Echtzeit-Snapshots

**Hinweis:** Für andere Timeframes (M1-D1) TwelveData API verwenden."""
    },
    {
        "name": "7. Yahoo Finance",
        "description": """Yahoo Finance market data integration (2. Fallback).

**Funktionen:**
- Historische OHLCV-Daten (kostenlos, ohne API-Key)
- Echtzeit-Quotes und Marktdaten
- Symbol-Informationen und Fundamentaldaten
- Unterstützt Aktien, ETFs, Indizes, Forex, Crypto"""
    },
    {
        "name": "8. RAG Sync",
        "description": "Synchronization of market data to RAG knowledge base (optional, disabled by default)"
    },
    {
        "name": "9. Query Logs & Analytics",
        "description": "Query logging and analytics for monitoring"
    },
    {
        "name": "10. External Data Sources",
        "description": """Externe Datenquellen für Trading Intelligence (Gateway für RAG Service).

**Datenquellen:**
- **Economic Calendar**: Fed, ECB, CPI, NFP, GDP und andere Wirtschaftsereignisse
- **Sentiment**: Fear & Greed Index, Social Media, Options, VIX
- **On-Chain**: Whale Alerts, Exchange Flows, Mining, DeFi TVL
- **Orderbook**: Bid/Ask Depth, Liquidations, CVD
- **Macro**: DXY, Bond Yields, Cross-Asset Korrelationen
- **Historical Patterns**: Saisonalität, Drawdowns, Events
- **Technical Levels**: S/R Zonen, Fibonacci, Pivots, VWAP
- **Regulatory**: SEC/CFTC, ETF News, Global Regulation
- **Correlations**: Cross-Asset Korrelationen, Divergenzen, Hedge-Empfehlungen
- **Volatility Regime**: VIX, ATR, Bollinger, Position Sizing
- **Institutional Flow**: COT Reports, ETF Flows, Whale Tracking"""
    },
    {
        "name": "11. Technical Indicators",
        "description": """Technische Indikatoren Registry mit API-Mappings.

**Funktionen:**
- Vollständige Liste aller verfügbaren Indikatoren
- TwelveData und EasyInsight API-Bezeichnungen
- Kategorisierung: Trend, Momentum, Volatilität, Volumen, etc.
- Stärken, Schwächen und Anwendungsgebiete je Indikator
- Aktivieren/Deaktivieren von Indikatoren"""
    },
    {
        "name": "12. Testing",
        "description": """Test Suite Execution für automatisierte Qualitätssicherung.

**Test-Kategorien:**
- **Smoke Tests**: Health Checks für alle Microservices
- **API Tests**: Endpoint-Tests für alle Service-APIs
- **Integration Tests**: Service-übergreifende Tests
- **Contract Tests**: API-Schema-Validierung mit Pydantic
- **Unit Tests**: Isolierte Tests ohne Service-Abhängigkeiten"""
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
- **TwelveData Integration**: Marktdaten und technische Indikatoren
- **Yahoo Finance Integration**: Historische Kursdaten als Fallback
- **External Data Sources**: Gateway für externe Datenquellen
- **RAG Sync**: Synchronisation von Marktdaten zur RAG Knowledge Base

**Hinweis:** Candlestick Pattern Detection wurde in eigenen Service migriert (Port 3006).

### Architektur

Dieser Service fungiert als **Data Gateway** für alle externen Datenquellen:
- **TwelveData API** (primär) - OHLC-Daten für alle Timeframes (M1-MN)
- **EasyInsight API** (1. Fallback) - Zusätzliche Indikatoren & TimescaleDB
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
app.include_router(easyinsight_router, prefix="/api/v1", tags=["6. EasyInsight API"])
app.include_router(yfinance_router, prefix="/api/v1", tags=["7. Yahoo Finance"])
# Note: patterns_router has been moved to Candlestick Service (Port 3006)
# Note: NHITS Training endpoints are in the NHITS Service (Port 3002), not here
app.include_router(sync_router, prefix="/api/v1", tags=["8. RAG Sync"])
app.include_router(query_log_router, prefix="/api/v1", tags=["9. Query Logs & Analytics"])
app.include_router(external_sources_router, prefix="/api/v1", tags=["10. External Data Sources"])
app.include_router(indicators_router, prefix="/api/v1", tags=["11. Technical Indicators"])
app.include_router(testing_router, prefix="/api/v1/testing", tags=["12. Testing"])

# Test-Health-Router für Test-Unhealthy-Endpoint
test_health_router = create_test_health_router("data")
app.include_router(test_health_router, prefix="/api/v1", tags=["12. Testing"])


async def _periodic_cache_cleanup():
    """Background task to periodically cleanup expired cache entries."""
    cleanup_interval = int(os.getenv("CACHE_CLEANUP_INTERVAL_SECONDS", "3600"))  # Default: 1 hour
    logger.info(f"Cache cleanup task started (interval: {cleanup_interval}s)")

    while True:
        try:
            await asyncio.sleep(cleanup_interval)
            removed = training_data_cache.cleanup_expired()
            if removed > 0:
                logger.info(f"Periodic cache cleanup: removed {removed} expired entries")
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")
            await asyncio.sleep(60)  # Wait before retry


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global sync_service, rag_service, _cache_cleanup_task, _prefetch_task

    logger.info("Starting Data Service...")
    logger.info(f"Version: {VERSION}")
    logger.info(f"EasyInsight API: {settings.easyinsight_api_url}")

    # Cleanup expired training data cache on startup
    try:
        removed = training_data_cache.cleanup_expired()
        stats = training_data_cache.get_stats()
        logger.info(f"Training data cache: {stats['total_entries']} entries, {removed} expired removed")

        # Start periodic cache cleanup task
        _cache_cleanup_task = asyncio.create_task(_periodic_cache_cleanup())
    except Exception as e:
        logger.error(f"Failed to cleanup training data cache: {e}")

    # Initialize RAG Service (for sync service) - optional, requires sentence_transformers
    if _rag_available:
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
    else:
        logger.info("RAG/Sync services skipped - dependencies not available")

    # Note: Pattern History Auto-Scan has been moved to Candlestick Service (Port 3006)

    # Initialize Pre-Fetching Service for cache warming
    try:
        from src.services.prefetch_service import prefetch_service
        from src.services.cache_service import cache_service

        # Connect cache service first
        await cache_service.connect()
        logger.info("Cache service connected")

        # Configure pre-fetch based on environment (only if no saved config exists)
        prefetch_enabled = os.getenv("PREFETCH_ENABLED", "true").lower() == "true"
        prefetch_interval = int(os.getenv("PREFETCH_INTERVAL", "300"))  # 5 minutes default
        prefetch_favorites_only = os.getenv("PREFETCH_FAVORITES_ONLY", "false").lower() == "true"

        # Try to load saved config from Redis first
        config_loaded = await prefetch_service.load_config()

        if not config_loaded:
            # No saved config - apply environment defaults
            await prefetch_service.configure(
                enabled=prefetch_enabled,
                refresh_interval=prefetch_interval,
                favorites_only=prefetch_favorites_only,
            )
            logger.info("Using default pre-fetch configuration from environment")
        else:
            logger.info("Loaded pre-fetch configuration from Redis")

        if prefetch_service._config.enabled:
            await prefetch_service.start()
            logger.info(
                f"Pre-fetch service started "
                f"(interval: {prefetch_interval}s, favorites_only: {prefetch_favorites_only})"
            )
        else:
            logger.info("Pre-fetch service disabled via PREFETCH_ENABLED=false")

    except Exception as e:
        logger.error(f"Failed to initialize pre-fetch service: {e}")

    logger.info("Data Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _cache_cleanup_task

    logger.info("Shutting down Data Service...")

    # Stop cache cleanup task
    if _cache_cleanup_task and not _cache_cleanup_task.done():
        _cache_cleanup_task.cancel()
        try:
            await _cache_cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Cache cleanup task stopped")

    # Stop sync service if running
    if sync_service and getattr(sync_service, '_running', False):
        await sync_service.stop()
        logger.info("TimescaleDB Sync Service stopped")

    # Note: Pattern History Auto-Scan has been moved to Candlestick Service (Port 3006)

    # Stop pre-fetch service
    try:
        from src.services.prefetch_service import prefetch_service
        await prefetch_service.stop()
        logger.info("Pre-fetch service stopped")
    except Exception as e:
        logger.warning(f"Error stopping pre-fetch service: {e}")

    logger.info("Data Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    service_name = "data"

    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status(service_name)
    is_unhealthy = test_status.get("test_unhealthy", False)

    sync_status = "unknown"
    if sync_service:
        sync_status = "running" if getattr(sync_service, '_running', False) else "stopped"

    # Get pre-fetch status
    prefetch_status = "unknown"
    try:
        from src.services.prefetch_service import prefetch_service
        prefetch_status = "running" if prefetch_service._running else "stopped"
    except Exception:
        pass

    response = {
        "service": service_name,
        "status": "unhealthy" if is_unhealthy else "healthy",
        "version": VERSION,
        "easyinsight_api": settings.easyinsight_api_url,
        "sync_service_status": sync_status,
        "prefetch_service_status": prefetch_status
    }

    # Test-Status hinzufügen wenn aktiv
    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


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
