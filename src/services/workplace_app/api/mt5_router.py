"""MT5 Connector API Router.

Endpoints für MT5 Terminal-Verwaltung, Trade-Historie und Performance-Analyse.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.schemas import (
    MT5OverviewResponse,
    MT5TradeListResponse,
    MT5TerminalListResponse,
    MT5TradeWithSetup,
    MT5PerformanceMetrics,
    MT5TradeSetupLink,
    MT5LinkRequest,
    MT5TerminalStatus,
)
from ..services.mt5_trade_service import mt5_trade_service
from ..services.mt5_linking_service import mt5_linking_service
from ..services.mt5_performance_service import mt5_performance_service

router = APIRouter(prefix="/mt5", tags=["6. MT5 Connector"])


# =============================================================================
# Dashboard / Overview
# =============================================================================


@router.get("/overview", response_model=MT5OverviewResponse)
async def get_overview():
    """
    Dashboard-Übersicht für MT5 Connector.

    Zeigt Terminals, letzte Trades und Basis-Metriken.
    """
    # Terminals holen
    terminals = await mt5_trade_service.get_terminals()
    terminals_online = sum(1 for t in terminals if t.status == MT5TerminalStatus.ONLINE)

    # Letzte Trades
    recent_trades = await mt5_trade_service.get_recent_trades(limit=10)

    # Offene Trades zählen
    open_trades = await mt5_trade_service.get_open_trades()

    # Performance-Metriken (letzte 30 Tage)
    since = datetime.now(timezone.utc) - timedelta(days=30)
    metrics = await mt5_performance_service.calculate_metrics(since=since)

    return MT5OverviewResponse(
        terminals=terminals,
        terminals_online=terminals_online,
        terminals_total=len(terminals),
        recent_trades=recent_trades,
        open_trades=len(open_trades),
        metrics=metrics,
        last_updated=datetime.now(timezone.utc),
    )


# =============================================================================
# Terminal-Verwaltung
# =============================================================================


@router.get("/terminals", response_model=MT5TerminalListResponse)
async def list_terminals(
    active_only: bool = Query(default=True, description="Nur aktive Terminals"),
):
    """Liste aller registrierten MT5 Terminals."""
    terminals = await mt5_trade_service.get_terminals(active_only=active_only)

    return MT5TerminalListResponse(
        terminals=terminals,
        total=len(terminals),
    )


@router.get("/terminals/{terminal_id}")
async def get_terminal(terminal_id: str):
    """Details eines spezifischen Terminals."""
    terminal = await mt5_trade_service.get_terminal(terminal_id)
    if not terminal:
        raise HTTPException(status_code=404, detail="Terminal nicht gefunden")

    return terminal


# =============================================================================
# Trade-Historie
# =============================================================================


@router.get("/trades", response_model=MT5TradeListResponse)
async def list_trades(
    terminal_id: Optional[str] = Query(None, description="Filter nach Terminal"),
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    status: Optional[str] = Query(None, description="Filter nach Status (open, closed)"),
    since: Optional[datetime] = Query(None, description="Trades seit diesem Zeitpunkt"),
    until: Optional[datetime] = Query(None, description="Trades bis zu diesem Zeitpunkt"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximale Anzahl"),
    offset: int = Query(default=0, ge=0, description="Offset für Pagination"),
):
    """
    Liste der Trades mit optionaler Filterung.

    Inkludiert verknüpfte Setups für jeden Trade.
    """
    trades, total = await mt5_trade_service.get_trades(
        terminal_id=terminal_id,
        symbol=symbol,
        status=status,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
        include_links=True,
    )

    has_more = offset + len(trades) < total

    return MT5TradeListResponse(
        trades=trades,
        total=total,
        has_more=has_more,
    )


@router.get("/trades/open")
async def list_open_trades():
    """Liste aller offenen Trades."""
    trades = await mt5_trade_service.get_open_trades()

    return {
        "trades": trades,
        "total": len(trades),
    }


@router.get("/trades/{trade_id}", response_model=MT5TradeWithSetup)
async def get_trade(trade_id: str):
    """Details eines spezifischen Trades inkl. verknüpftem Setup."""
    trade = await mt5_trade_service.get_trade(trade_id, include_link=True)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade nicht gefunden")

    return trade


# =============================================================================
# Setup-Verknüpfung
# =============================================================================


@router.post("/trades/{trade_id}/link-setup", response_model=MT5TradeSetupLink)
async def link_trade_to_setup(trade_id: str, request: MT5LinkRequest):
    """
    Verknüpft einen Trade manuell mit einem Setup.

    Nützlich wenn die Auto-Verknüpfung nicht das gewünschte Setup gefunden hat.
    """
    link = await mt5_linking_service.manual_link_trade(
        trade_id=trade_id,
        setup_timestamp=request.setup_timestamp,
        setup_timeframe=request.setup_timeframe,
        notes=request.notes,
    )

    if not link:
        raise HTTPException(
            status_code=400,
            detail="Verknüpfung konnte nicht erstellt werden",
        )

    return link


@router.delete("/trades/{trade_id}/link")
async def unlink_trade(trade_id: str):
    """Entfernt die Setup-Verknüpfung von einem Trade."""
    success = await mt5_linking_service.unlink_trade(trade_id)

    if not success:
        raise HTTPException(status_code=404, detail="Keine Verknüpfung gefunden")

    return {"success": True, "message": "Verknüpfung entfernt"}


@router.post("/trades/auto-link")
async def auto_link_trades():
    """
    Verknüpft alle neuen Trades automatisch mit Setups.

    Wird für manuelle Auslösung des Auto-Linking verwendet.
    """
    linked_count = await mt5_linking_service.auto_link_new_trades()

    return {
        "success": True,
        "linked_count": linked_count,
        "message": f"{linked_count} Trades wurden mit Setups verknüpft",
    }


# =============================================================================
# Performance-Analyse
# =============================================================================


@router.get("/performance", response_model=MT5PerformanceMetrics)
async def get_performance(
    terminal_id: Optional[str] = Query(None, description="Filter nach Terminal"),
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    days: int = Query(default=30, ge=1, le=365, description="Zeitraum in Tagen"),
):
    """
    Performance-Metriken für den angegebenen Zeitraum.

    Inkludiert Setup-basierte Analyse wenn Verknüpfungen vorhanden.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    metrics = await mt5_performance_service.calculate_metrics(
        terminal_id=terminal_id,
        symbol=symbol,
        since=since,
    )

    return metrics


@router.get("/performance/by-setup")
async def get_setup_performance(
    days: int = Query(default=30, ge=1, le=365, description="Zeitraum in Tagen"),
):
    """
    Analyse der Performance nach Setup-Befolgung.

    Vergleicht Trades die Setup-Empfehlungen folgten vs. nicht folgten.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    analysis = await mt5_performance_service.calculate_setup_performance(since=since)

    return analysis


@router.get("/performance/by-symbol")
async def get_symbol_performance(
    days: int = Query(default=30, ge=1, le=365, description="Zeitraum in Tagen"),
):
    """Performance-Aufschlüsselung nach Symbol."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    breakdown = await mt5_performance_service.get_symbol_breakdown(since=since)

    return {
        "symbols": breakdown,
        "total_symbols": len(breakdown),
    }


@router.get("/performance/time-analysis")
async def get_time_analysis(
    days: int = Query(default=30, ge=1, le=365, description="Zeitraum in Tagen"),
    group_by: str = Query(default="day", description="Gruppierung: day, week, month"),
):
    """Zeitliche Analyse der Trading-Performance."""
    if group_by not in ("day", "week", "month"):
        raise HTTPException(
            status_code=400,
            detail="group_by muss 'day', 'week' oder 'month' sein",
        )

    since = datetime.now(timezone.utc) - timedelta(days=days)

    analysis = await mt5_performance_service.get_time_analysis(
        since=since,
        group_by=group_by,
    )

    return {
        "periods": analysis,
        "group_by": group_by,
        "total_periods": len(analysis),
    }


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health")
async def health_check():
    """Health-Check für den MT5 Connector."""
    data_service_health = await mt5_trade_service.health_check()

    return {
        "service": "mt5-connector",
        "status": "healthy" if data_service_health.get("status") == "healthy" else "degraded",
        "data_service": data_service_health,
    }
