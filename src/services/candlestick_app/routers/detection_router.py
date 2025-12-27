"""Pattern detection endpoints."""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.schemas import (
    PatternScanRequest,
    PatternScanResponse,
    PatternType,
    PatternCategory,
    Timeframe,
)
from ..services.pattern_detection_service import candlestick_pattern_service

router = APIRouter()


@router.post("/scan", response_model=PatternScanResponse)
async def scan_patterns(request: PatternScanRequest):
    """
    Scan for candlestick patterns across multiple timeframes.

    Analyzes price action to detect patterns like:
    - **Reversal**: Hammer, Shooting Star, Doji, Engulfing, Morning/Evening Star
    - **Continuation**: Three White Soldiers, Three Black Crows
    - **Indecision**: Spinning Top, Harami

    Parameters:
    - **symbol**: Trading symbol (e.g., BTCUSD, EURUSD)
    - **timeframes**: Timeframes to scan (M5, M15, H1, H4, D1)
    - **lookback_candles**: Number of candles to analyze (10-500)
    - **min_confidence**: Minimum confidence threshold (0.0-1.0)
    - **include_weak_patterns**: Include patterns with confidence < 0.5
    """
    try:
        response = await candlestick_pattern_service.scan_patterns(request)
        return response

    except Exception as e:
        logger.error(f"Pattern scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan/{symbol}", response_model=PatternScanResponse)
async def scan_patterns_get(
    symbol: str,
    timeframes: str = Query(
        default="M15,H1,H4,D1",
        description="Comma-separated timeframes (M5,M15,H1,H4,D1)"
    ),
    lookback: int = Query(default=100, ge=10, le=500, description="Lookback candles"),
    min_confidence: float = Query(default=0.5, ge=0, le=1, description="Minimum confidence")
):
    """
    Scan patterns for a symbol (GET endpoint).

    Simplified endpoint for quick pattern checks.
    """
    try:
        # Parse timeframes
        tf_list = [Timeframe(tf.strip().upper()) for tf in timeframes.split(",")]

        request = PatternScanRequest(
            symbol=symbol.upper(),
            timeframes=tf_list,
            lookback_candles=lookback,
            min_confidence=min_confidence,
        )

        response = await candlestick_pattern_service.scan_patterns(request)
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {e}")
    except Exception as e:
        logger.error(f"Pattern scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan-all")
async def scan_all_symbols(
    timeframes: str = Query(
        default="M15,H1,H4,D1",
        description="Comma-separated timeframes"
    ),
    min_confidence: float = Query(default=0.6, ge=0, le=1)
):
    """
    Scan all active symbols for patterns.

    Returns symbols with detected patterns above threshold.
    """
    try:
        # Parse timeframes
        tf_list = [Timeframe(tf.strip().upper()) for tf in timeframes.split(",")]

        results = await candlestick_pattern_service.scan_all_symbols(
            timeframes=tf_list,
            min_confidence=min_confidence
        )

        # Convert to serializable format
        serialized = {}
        for symbol, result in results.items():
            serialized[symbol] = {
                "total_patterns": result.total_patterns_found,
                "dominant_direction": result.dominant_direction.value if result.dominant_direction else None,
                "confluence_score": result.confluence_score,
                "bullish_count": result.bullish_patterns_count,
                "bearish_count": result.bearish_patterns_count,
                "neutral_count": result.neutral_patterns_count,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "timeframes": [tf.value for tf in tf_list],
            "min_confidence": min_confidence,
            "total_symbols_scanned": len(results),
            "symbols_with_patterns": len([r for r in results.values() if r.total_patterns_found > 0]),
            "results": serialized
        }

    except Exception as e:
        logger.error(f"Scan all error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart/{symbol}")
async def get_pattern_chart(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe (M5,M15,H1,H4,D1)"),
    pattern_timestamp: str = Query(..., description="Pattern timestamp (ISO 8601)"),
    candles_before: int = Query(default=15, ge=5, le=50),
    candles_after: int = Query(default=5, ge=0, le=20)
):
    """
    Get chart data for visualizing a specific pattern.

    Returns OHLCV data around the pattern timestamp with the pattern
    candle highlighted.

    Parameters:
    - **symbol**: Trading symbol
    - **timeframe**: Timeframe of the pattern
    - **pattern_timestamp**: When the pattern was detected (ISO 8601)
    - **candles_before**: Candles to show before pattern (5-50)
    - **candles_after**: Candles to show after pattern (0-20)
    """
    try:
        # Parse timestamp
        ts = datetime.fromisoformat(pattern_timestamp.replace('Z', '+00:00'))

        chart_data = await candlestick_pattern_service.get_pattern_chart_data(
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            pattern_timestamp=ts,
            candles_before=candles_before,
            candles_after=candles_after,
        )

        return chart_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {e}")
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_supported_patterns():
    """
    Get list of supported candlestick pattern types.

    Returns all patterns that can be detected, grouped by category.
    """
    patterns = candlestick_pattern_service.get_supported_patterns()

    # Group by category
    by_category = {
        "reversal": [],
        "continuation": [],
        "indecision": [],
    }

    for p in patterns:
        category = p["category"]
        if category in by_category:
            by_category[category].append(p)

    return {
        "patterns": patterns,
        "count": len(patterns),
        "categories": by_category
    }
