"""Pattern history endpoints."""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from loguru import logger

from ..services.pattern_history_service import pattern_history_service

router = APIRouter()


@router.get("/history")
async def get_pattern_history(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    direction: Optional[str] = Query(default=None, description="Filter by direction (bullish, bearish, neutral)"),
    category: Optional[str] = Query(default=None, description="Filter by category (reversal, continuation, indecision)"),
    timeframe: Optional[str] = Query(default=None, description="Filter by timeframe (M5, M15, H1, H4, D1)"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results")
):
    """
    Get pattern history with optional filters.

    Returns detected patterns from the history, sorted by most recent first.

    Parameters:
    - **symbol**: Filter by trading symbol
    - **direction**: Filter by signal direction (bullish, bearish, neutral)
    - **category**: Filter by pattern category (reversal, continuation, indecision)
    - **timeframe**: Filter by timeframe (M5, M15, H1, H4, D1)
    - **min_confidence**: Minimum confidence threshold
    - **limit**: Maximum number of results
    """
    try:
        history = pattern_history_service.get_history(
            symbol=symbol.upper() if symbol else None,
            direction=direction.lower() if direction else None,
            category=category.lower() if category else None,
            timeframe=timeframe.upper() if timeframe else None,
            min_confidence=min_confidence,
            limit=limit,
        )

        return {
            "count": len(history),
            "patterns": history
        }

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_symbol_history(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    Get pattern history for a specific symbol.

    Returns all detected patterns for the symbol, sorted by most recent first.
    """
    try:
        history = pattern_history_service.get_history(
            symbol=symbol.upper(),
            limit=limit,
        )

        return {
            "symbol": symbol.upper(),
            "count": len(history),
            "patterns": history
        }

    except Exception as e:
        logger.error(f"Error getting symbol history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/by-symbol")
async def get_history_by_symbol():
    """
    Get latest patterns grouped by symbol.

    Returns up to 5 most recent patterns per symbol.
    """
    try:
        by_symbol = pattern_history_service.get_latest_by_symbol()

        return {
            "symbols_count": len(by_symbol),
            "by_symbol": by_symbol
        }

    except Exception as e:
        logger.error(f"Error getting history by symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/statistics")
async def get_history_statistics():
    """
    Get statistics about detected patterns.

    Returns aggregated statistics including:
    - Total patterns
    - Patterns by direction
    - Patterns by category
    - Patterns by timeframe
    - Scan status
    """
    try:
        stats = pattern_history_service.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/scan")
async def trigger_scan():
    """
    Trigger an immediate pattern scan.

    Scans all available symbols for patterns and updates the history.
    """
    try:
        new_patterns = await pattern_history_service.scan_all_symbols()

        return {
            "status": "completed",
            "new_patterns_found": new_patterns
        }

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/clear")
async def clear_history():
    """
    Clear all pattern history.

    Warning: This action cannot be undone.
    """
    try:
        pattern_history_service.clear_history()

        return {
            "status": "cleared",
            "message": "Pattern history has been cleared"
        }

    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan/status")
async def get_scan_status():
    """
    Get the current auto-scan status.

    Returns whether the periodic scan is running and its interval.
    """
    try:
        stats = pattern_history_service.get_statistics()

        return {
            "running": stats.get("scan_running", False),
            "interval_seconds": stats.get("scan_interval_seconds", 300),
            "total_patterns": stats.get("total_patterns", 0),
            "last_scan": stats.get("last_scan"),
        }

    except Exception as e:
        logger.error(f"Error getting scan status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan/start")
async def start_auto_scan():
    """
    Start the automatic pattern scanning.

    Begins periodic scanning of all symbols for patterns.
    """
    try:
        if pattern_history_service.is_running():
            return {
                "status": "already_running",
                "message": "Auto-scan is already running"
            }

        await pattern_history_service.start()

        return {
            "status": "started",
            "message": "Auto-scan has been started"
        }

    except Exception as e:
        logger.error(f"Error starting scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan/stop")
async def stop_auto_scan():
    """
    Stop the automatic pattern scanning.
    """
    try:
        if not pattern_history_service.is_running():
            return {
                "status": "not_running",
                "message": "Auto-scan is not running"
            }

        await pattern_history_service.stop()

        return {
            "status": "stopped",
            "message": "Auto-scan has been stopped"
        }

    except Exception as e:
        logger.error(f"Error stopping scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))
