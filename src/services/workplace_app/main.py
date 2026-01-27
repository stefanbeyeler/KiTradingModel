"""
Trading Workplace Service - Microservice für Trading-Setup-Aggregation.

Aggregiert Vorhersagen aller ML-Services mit Multi-Signal-Scoring.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.shared.logging_config import log_shutdown_info, log_startup_info, setup_logging
from src.shared.test_health_router import create_test_health_router
from src.shared.health import get_test_unhealthy_status

from .api.routes import router
from .config import settings
from .services.scanner_service import scanner_service
from .services.watchlist_service import watchlist_service
from .services.signal_aggregator import signal_aggregator
from .services.deep_analysis_service import deep_analysis_service
from .services.mt5_trade_service import mt5_trade_service

VERSION = "1.0.0"

# OpenAPI Tags (kurze Beschreibungen für Swagger UI)
openapi_tags = [
    {
        "name": "1. Setups",
        "description": "Trading-Setups mit Multi-Signal-Scoring"
    },
    {
        "name": "2. Analyse",
        "description": "Vertiefte Analyse mit RAG + LLM"
    },
    {
        "name": "3. Watchlist",
        "description": "Konfigurierbare Symbol-Watchlist"
    },
    {
        "name": "4. Scanner",
        "description": "Auto-Scanner Kontrolle"
    },
    {
        "name": "5. System",
        "description": "Health-Checks und Service-Info"
    },
    {
        "name": "6. MT5 Connector",
        "description": "MT5 Trade-Aufzeichnung und Performance-Analyse"
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management für den Trading Workplace Service."""
    setup_logging("workplace")
    log_startup_info("workplace", VERSION, settings.workplace_port, gpu_available=False)

    # Watchlist initialisieren
    await watchlist_service.initialize()
    logger.info(f"Watchlist initialisiert mit {len(await watchlist_service.get_symbols())} Symbolen")

    # Auto-Scanner starten wenn aktiviert
    if settings.auto_scan_enabled:
        await scanner_service.start()
        logger.info(f"Auto-Scanner gestartet (Intervall: {settings.scan_interval_seconds}s)")

    # Gewichte validieren
    if not settings.validate_weights():
        logger.warning("Scoring-Gewichte ergeben nicht 1.0 - bitte Konfiguration prüfen!")

    yield

    # Shutdown
    logger.info("Shutting down Trading Workplace Service...")

    # Scanner stoppen
    if scanner_service.is_running:
        await scanner_service.stop()

    # Watchlist speichern
    await watchlist_service.save()

    # HTTP-Clients schliessen
    await signal_aggregator.close()
    await deep_analysis_service.close()
    await mt5_trade_service.close()

    log_shutdown_info("workplace")


app = FastAPI(
    title="Trading Workplace Service",
    description="""## Trading-Setup-Aggregation mit Multi-Signal-Scoring

Der Trading Workplace Service aggregiert Vorhersagen aller ML-Services
und berechnet einen gewichteten Composite-Score für Trading-Setups.

### Features

- **Schnellbeurteilung**: Top Trading-Setups auf einen Blick
- **Multi-Signal-Scoring**: Gewichtete Kombination aus NHITS, HMM, TCN, Candlestick
- **Vertiefte Analyse**: RAG-Kontext + LLM-generierte Empfehlungen
- **Konfigurierbare Watchlist**: Eigene Symbole mit Alert-Schwellen
- **Auto-Scanner**: Periodisches Scanning im Hintergrund

### Signal-Gewichte

| Service | Gewicht | Signal |
|---------|---------|--------|
| NHITS | 30% | trend_probability |
| HMM | 25% | regime_confidence |
| TCN | 20% | pattern_confidence |
| Candlestick | 15% | pattern_strength |
| Technical | 10% | trend_alignment |
""",
    version=VERSION,
    lifespan=lifespan,
    root_path="/workplace",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=openapi_tags,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router einbinden
app.include_router(router, prefix="/api/v1")

# Test-Health-Router
test_health_router = create_test_health_router("workplace")
app.include_router(test_health_router, prefix="/api/v1", tags=["Testing"])


@app.get("/health", tags=["5. System"])
async def health_check():
    """Health-Check für den Workplace Service."""
    # Test-Unhealthy-Status prüfen
    test_status = get_test_unhealthy_status("workplace")
    is_unhealthy = test_status.get("test_unhealthy", False)

    # Services-Erreichbarkeit prüfen
    services_reachable = await signal_aggregator.check_services_health()

    response = {
        "service": "workplace",
        "status": "unhealthy" if is_unhealthy else "healthy",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scanner_running": scanner_service.is_running,
        "watchlist_size": len(await watchlist_service.get_symbols()),
        "setups_cached": len(scanner_service.results),
        "last_scan": scanner_service.get_status().last_scan_time.isoformat()
            if scanner_service.get_status().last_scan_time else None,
        "services_reachable": services_reachable,
    }

    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


@app.get("/", tags=["5. System"])
async def root():
    """Service-Information."""
    return {
        "service": "Trading Workplace",
        "version": VERSION,
        "description": "Trading-Setup-Aggregation mit Multi-Signal-Scoring",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "setups": "/api/v1/setups/",
            "analyze": "/api/v1/analyze/{symbol}",
            "watchlist": "/api/v1/watchlist/",
            "scanner": "/api/v1/scan/status",
        }
    }


@app.get("/api/v1/info", tags=["5. System"])
async def get_info():
    """Detaillierte Service-Informationen."""
    scan_status = scanner_service.get_status()

    return {
        "service": "Trading Workplace",
        "version": VERSION,
        "config": {
            "scan_interval_seconds": settings.scan_interval_seconds,
            "auto_scan_enabled": settings.auto_scan_enabled,
            "high_confidence_threshold": settings.high_confidence_threshold,
            "min_confidence_threshold": settings.min_confidence_threshold,
        },
        "weights": {
            "nhits": settings.nhits_weight,
            "hmm": settings.hmm_weight,
            "tcn": settings.tcn_weight,
            "candlestick": settings.candlestick_weight,
            "technical": settings.technical_weight,
        },
        "scanner": {
            "status": scan_status.status.value,
            "symbols_scanned": scan_status.symbols_scanned_total,
            "alerts_triggered": scan_status.alerts_triggered,
            "errors": scan_status.errors_count,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.workplace_app.main:app",
        host="0.0.0.0",
        port=settings.workplace_port,
        reload=False
    )
