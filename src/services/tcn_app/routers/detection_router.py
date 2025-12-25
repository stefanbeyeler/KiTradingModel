"""Pattern detection endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from loguru import logger

from ..models.schemas import (
    PatternDetectionRequest,
    PatternDetectionResponse,
    PatternScanRequest,
    PatternScanResponse,
    SymbolPatternResult,
)
from ..services.pattern_detection_service import pattern_detection_service

router = APIRouter()


@router.post("/detect", response_model=PatternDetectionResponse)
async def detect_patterns(request: PatternDetectionRequest):
    """
    Detect chart patterns in current price action.

    Uses TCN deep learning combined with rule-based detection.

    - **symbol**: Trading symbol (e.g., BTCUSD)
    - **timeframe**: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    - **lookback**: Number of candles for analysis
    - **threshold**: Confidence threshold (0.0-1.0)
    - **patterns**: Optional filter for specific patterns
    """
    try:
        response = await pattern_detection_service.detect_patterns(
            symbol=request.symbol,
            timeframe=request.timeframe,
            lookback=request.lookback,
            threshold=request.threshold,
            pattern_filter=request.patterns
        )
        return response

    except Exception as e:
        logger.error(f"Pattern detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detect/{symbol}", response_model=PatternDetectionResponse)
async def detect_patterns_get(
    symbol: str,
    timeframe: str = Query(default="1h", description="Timeframe"),
    lookback: int = Query(default=200, description="Lookback candles"),
    threshold: float = Query(default=0.5, ge=0, le=1, description="Confidence threshold")
):
    """
    Detect patterns for a symbol (GET endpoint).

    Simplified endpoint for quick pattern checks.
    """
    try:
        response = await pattern_detection_service.detect_patterns(
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            threshold=threshold
        )
        return response

    except Exception as e:
        logger.error(f"Pattern detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan", response_model=PatternScanResponse)
async def scan_symbols(request: PatternScanRequest):
    """
    Scan multiple symbols for patterns.

    - **symbols**: List of symbols to scan (None = all active)
    - **timeframe**: Timeframe to analyze
    - **threshold**: Minimum confidence
    - **min_patterns**: Minimum patterns to report
    """
    import time
    start_time = time.time()

    try:
        # Get symbols list
        if request.symbols:
            symbols = request.symbols
        else:
            # Fetch active symbols from data service
            from src.services.data_gateway_service import data_gateway
            symbols_data = await data_gateway.get_available_symbols()
            symbols = [s.get('symbol', s) if isinstance(s, dict) else s for s in symbols_data[:20]]  # Limit

        results = await pattern_detection_service.scan_symbols(
            symbols=symbols,
            timeframe=request.timeframe,
            threshold=request.threshold,
            min_patterns=request.min_patterns
        )

        # Convert to response format
        symbol_results = [
            SymbolPatternResult(
                symbol=r["symbol"],
                patterns=[],  # Simplified
                scan_time_ms=r["scan_time_ms"]
            )
            for r in results
        ]

        return PatternScanResponse(
            timestamp=datetime.now(),
            timeframe=request.timeframe,
            threshold=request.threshold,
            results=symbol_results,
            total_symbols=len(symbols),
            symbols_with_patterns=len(results),
            total_patterns=sum(len(r.get("patterns", [])) for r in results),
            scan_duration_ms=round((time.time() - start_time) * 1000, 2)
        )

    except Exception as e:
        logger.error(f"Pattern scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan-all")
async def scan_all_symbols(
    timeframe: str = Query(default="1h"),
    threshold: float = Query(default=0.6, ge=0, le=1),
    min_patterns: int = Query(default=1, ge=0)
):
    """
    Scan all active symbols for patterns.

    Returns symbols with detected patterns above threshold.
    """
    try:
        from src.services.data_gateway_service import data_gateway

        # Get active symbols
        symbols_data = await data_gateway.get_available_symbols()
        symbols = [s.get('symbol', s) if isinstance(s, dict) else s for s in symbols_data]

        results = await pattern_detection_service.scan_symbols(
            symbols=symbols,
            timeframe=timeframe,
            threshold=threshold,
            min_patterns=min_patterns
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "threshold": threshold,
            "total_symbols_scanned": len(symbols),
            "symbols_with_patterns": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Scan all error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_supported_patterns():
    """
    Get list of supported pattern types.

    Returns all patterns that can be detected.
    """
    patterns = pattern_detection_service.get_supported_patterns()

    return {
        "patterns": patterns,
        "count": len(patterns),
        "categories": {
            "reversal": [
                "head_and_shoulders",
                "inverse_head_and_shoulders",
                "double_top",
                "double_bottom",
                "triple_top",
                "triple_bottom",
                "cup_and_handle",
                "rising_wedge",
                "falling_wedge"
            ],
            "continuation": [
                "ascending_triangle",
                "descending_triangle",
                "symmetrical_triangle",
                "bull_flag",
                "bear_flag"
            ],
            "trend": [
                "channel_up",
                "channel_down"
            ]
        }
    }
