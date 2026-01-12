"""
Setups Router - Trading-Setup Endpoints.

Schnellbeurteilung: Top Trading-Setups basierend auf Multi-Signal-Scoring.
"""

from datetime import datetime, timezone
from typing import Optional
import time

from fastapi import APIRouter, Query, HTTPException
from loguru import logger

from ..models.schemas import (
    TradingSetup,
    SetupListResponse,
    SignalDirection,
)
from ..services.scanner_service import scanner_service
from ..services.signal_aggregator import signal_aggregator
from ..services.scoring_service import scoring_service

router = APIRouter()


@router.get(
    "/",
    response_model=SetupListResponse,
    summary="Top Trading-Setups",
    description="Gibt die vielversprechendsten Trading-Setups zurück (Schnellbeurteilung)."
)
async def get_top_setups(
    limit: int = Query(default=10, ge=1, le=50, description="Anzahl Setups"),
    min_score: float = Query(default=50.0, ge=0, le=100, description="Minimum Score"),
    direction: Optional[SignalDirection] = Query(
        default=None,
        description="Filter nach Richtung (long, short, neutral)"
    ),
):
    """
    Schnellbeurteilung: Top Trading-Setups.

    Gibt die besten Trading-Setups basierend auf dem gewichteten
    Multi-Signal Composite-Score zurück.

    Die Setups werden aus dem letzten Scan-Lauf geholt und nach
    Score absteigend sortiert. Falls keine Scanner-Ergebnisse vorhanden
    sind, werden Setups aus den Watchlist-Daten generiert.
    """
    start_time = time.time()

    # Verwende Methode mit Watchlist-Fallback
    setups = await scanner_service.get_top_setups_with_watchlist_fallback(
        limit=limit,
        min_score=min_score,
        direction=direction,
    )

    # High-Confidence zählen
    high_confidence = sum(
        1 for s in setups
        if s.confidence_level.value == "high"
    )

    scan_duration = (time.time() - start_time) * 1000

    return SetupListResponse(
        timestamp=datetime.now(timezone.utc),
        setups=setups,
        total_scanned=len(scanner_service.results),
        high_confidence_count=high_confidence,
        scan_duration_ms=scan_duration,
    )


@router.get(
    "/{symbol}",
    response_model=TradingSetup,
    summary="Setup für Symbol",
    description="Gibt das aktuelle Trading-Setup für ein bestimmtes Symbol zurück."
)
async def get_setup_for_symbol(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe"),
    force_refresh: bool = Query(
        default=False,
        description="Erzwingt einen neuen Scan statt Cache"
    ),
):
    """
    Trading-Setup für ein einzelnes Symbol.

    Gibt das aktuelle Setup aus dem Cache zurück oder führt
    bei force_refresh=True einen neuen Scan durch.
    """
    symbol = symbol.upper()

    # Aus Cache holen wenn vorhanden und kein Refresh
    if not force_refresh:
        cached_setup = scanner_service.get_setup(symbol)
        if cached_setup:
            return cached_setup

    # Neuen Scan durchführen
    try:
        signals = await signal_aggregator.fetch_all_signals(symbol, timeframe)
        setup = scoring_service.create_setup(symbol, timeframe, signals)
        return setup

    except Exception as e:
        logger.error(f"Fehler beim Scannen von {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Erstellen des Setups: {str(e)}"
        )


@router.get(
    "/symbol/{symbol}/history",
    response_model=list[TradingSetup],
    summary="Setup-Historie",
    description="Gibt die letzten Setups für ein Symbol zurück (falls verfügbar)."
)
async def get_setup_history(
    symbol: str,
    limit: int = Query(default=10, ge=1, le=100, description="Anzahl Einträge"),
):
    """
    Setup-Historie für ein Symbol.

    Hinweis: Diese Funktion ist für zukünftige Erweiterungen vorgesehen.
    Aktuell wird nur das letzte Setup zurückgegeben.
    """
    symbol = symbol.upper()

    # Aktuell nur das letzte Setup (Historie noch nicht implementiert)
    setup = scanner_service.get_setup(symbol)

    if not setup:
        return []

    return [setup]


@router.post(
    "/{symbol}/refresh",
    response_model=TradingSetup,
    summary="Setup aktualisieren",
    description="Führt einen neuen Scan für das Symbol durch und gibt das aktualisierte Setup zurück."
)
async def refresh_setup(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe"),
):
    """
    Aktualisiert das Setup für ein Symbol.

    Führt einen neuen Scan durch und aktualisiert den Cache.
    """
    symbol = symbol.upper()

    try:
        signals = await signal_aggregator.fetch_all_signals(symbol, timeframe)
        setup = scoring_service.create_setup(symbol, timeframe, signals)

        # Im Scanner-Cache aktualisieren
        scanner_service._results[symbol] = setup

        logger.info(f"Setup für {symbol} aktualisiert: Score {setup.composite_score:.1f}")
        return setup

    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren von {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Aktualisieren: {str(e)}"
        )
