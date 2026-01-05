"""TCN Pattern History endpoints."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..services.tcn_pattern_history_service import tcn_pattern_history_service

router = APIRouter()


@router.get("/history")
async def get_pattern_history(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(default=None, description="Filter by timeframe (1h, 4h, 1d)"),
    pattern_type: Optional[str] = Query(default=None, description="Filter by pattern type"),
    category: Optional[str] = Query(default=None, description="Filter by category (reversal, continuation, trend)"),
    direction: Optional[str] = Query(default=None, description="Filter by direction (bullish, bearish)"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum entries to return")
):
    """
    Get filtered TCN pattern history.

    Returns historical pattern detections sorted by detection time (newest first).

    **Example queries:**
    - `/history?symbol=BTCUSD` - All patterns for BTCUSD
    - `/history?category=reversal&min_confidence=0.7` - High-confidence reversal patterns
    - `/history?direction=bullish&timeframe=4h` - Bullish patterns on 4h timeframe
    """
    try:
        entries = tcn_pattern_history_service.get_history(
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            category=category,
            direction=direction,
            min_confidence=min_confidence,
            limit=limit
        )

        return {
            "count": len(entries),
            "filters": {
                "symbol": symbol,
                "timeframe": timeframe,
                "pattern_type": pattern_type,
                "category": category,
                "direction": direction,
                "min_confidence": min_confidence
            },
            "patterns": [entry.to_dict() for entry in entries]
        }

    except Exception as e:
        logger.error(f"Error getting pattern history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history-by-symbol")
async def get_history_by_symbol(
    limit_per_symbol: int = Query(default=10, ge=1, le=50, description="Max patterns per symbol")
):
    """
    Get pattern history grouped by symbol.

    Returns a dictionary with symbols as keys and their pattern lists as values.
    Useful for getting an overview of patterns across all symbols.
    """
    try:
        grouped = tcn_pattern_history_service.get_history_by_symbol(limit_per_symbol)

        return {
            "symbols_count": len(grouped),
            "limit_per_symbol": limit_per_symbol,
            "by_symbol": grouped
        }

    except Exception as e:
        logger.error(f"Error getting grouped history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/statistics")
async def get_history_statistics():
    """
    Get pattern history statistics.

    Returns aggregated statistics about detected patterns:
    - Total patterns count
    - Patterns by type, category, direction, timeframe
    - Auto-scan status
    """
    try:
        stats = tcn_pattern_history_service.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/scan")
async def trigger_pattern_scan(
    symbol: Optional[str] = Query(
        default=None,
        description="Symbol to scan (None = all symbols)"
    ),
    timeframes: Optional[str] = Query(
        default="1h,4h,1d",
        description="Comma-separated timeframes to scan"
    ),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum confidence")
):
    """
    Trigger a manual pattern scan.

    Scans specified symbol (or all symbols) for patterns and adds new detections to history.
    Duplicate patterns (same symbol/timeframe/type within 4 hours) are skipped.
    """
    try:
        tf_list = [tf.strip() for tf in timeframes.split(",")]

        if symbol:
            # Scan single symbol
            result = await tcn_pattern_history_service.scan_single_symbol(
                symbol=symbol,
                timeframes=tf_list,
                threshold=threshold
            )
        else:
            # Scan all symbols
            result = await tcn_pattern_history_service.scan_all_symbols(
                timeframes=tf_list,
                threshold=threshold
            )

        return result

    except Exception as e:
        logger.error(f"Error during pattern scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/start-auto-scan")
async def start_auto_scan():
    """
    Start automatic pattern scanning.

    Patterns will be scanned every 5 minutes across all symbols and timeframes.
    New patterns are automatically added to history.
    """
    try:
        started = await tcn_pattern_history_service.start_auto_scan()

        if started:
            return {
                "status": "started",
                "message": "Auto-scan started successfully",
                "interval_seconds": tcn_pattern_history_service._scan_interval
            }
        else:
            return {
                "status": "already_running",
                "message": "Auto-scan is already running"
            }

    except Exception as e:
        logger.error(f"Error starting auto-scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/stop-auto-scan")
async def stop_auto_scan():
    """
    Stop automatic pattern scanning.
    """
    try:
        stopped = await tcn_pattern_history_service.stop_auto_scan()

        if stopped:
            return {
                "status": "stopped",
                "message": "Auto-scan stopped successfully"
            }
        else:
            return {
                "status": "not_running",
                "message": "Auto-scan was not running"
            }

    except Exception as e:
        logger.error(f"Error stopping auto-scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def clear_pattern_history():
    """
    Clear all pattern history.

    **Warning:** This permanently deletes all stored pattern detections.
    """
    try:
        count = tcn_pattern_history_service.clear_history()

        return {
            "status": "cleared",
            "entries_removed": count
        }

    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Symbol-specific endpoint MUST be last (catches all /history/{anything})
@router.get("/history/{symbol}")
async def get_symbol_pattern_history(
    symbol: str,
    timeframe: Optional[str] = Query(default=None, description="Filter by timeframe"),
    pattern_type: Optional[str] = Query(default=None, description="Filter by pattern type"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum entries")
):
    """
    Get pattern history for a specific symbol.

    Returns all detected patterns for this symbol with timestamps.

    **Response includes:**
    - Pattern type (e.g., double_top, head_and_shoulders)
    - Detection timestamp
    - Pattern start/end times
    - Confidence score
    - Price targets and invalidation levels
    """
    try:
        entries = tcn_pattern_history_service.get_history(
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            min_confidence=min_confidence,
            limit=limit
        )

        return {
            "symbol": symbol,
            "count": len(entries),
            "patterns": [entry.to_dict() for entry in entries]
        }

    except Exception as e:
        logger.error(f"Error getting pattern history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
