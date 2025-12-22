"""Regime detection endpoints."""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.schemas import (
    RegimeDetectionRequest,
    RegimeDetectionResponse,
    RegimeHistoryResponse,
    MarketRegime,
)
from ..services.regime_detection_service import regime_detection_service

router = APIRouter()


@router.post("/detect", response_model=RegimeDetectionResponse)
async def detect_regime(request: RegimeDetectionRequest):
    """
    Detect the current market regime.

    Uses Hidden Markov Model to classify market into:
    - **bull_trend**: Upward trend with low volatility
    - **bear_trend**: Downward trend with elevated volatility
    - **sideways**: Low directional movement
    - **high_volatility**: High volatility, unclear direction

    Parameters:
    - **symbol**: Trading symbol
    - **timeframe**: Timeframe (1h, 4h, 1d, etc.)
    - **lookback**: Number of candles for analysis
    - **include_history**: Include historical regime changes
    """
    try:
        response = await regime_detection_service.detect_regime(
            symbol=request.symbol,
            timeframe=request.timeframe,
            lookback=request.lookback,
            include_history=request.include_history
        )
        return response

    except Exception as e:
        logger.error(f"Regime detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detect/{symbol}", response_model=RegimeDetectionResponse)
async def detect_regime_get(
    symbol: str,
    timeframe: str = Query(default="1h"),
    lookback: int = Query(default=500),
    include_history: bool = Query(default=False)
):
    """
    Detect regime for a symbol (GET endpoint).

    Simplified endpoint for quick regime checks.
    """
    try:
        response = await regime_detection_service.detect_regime(
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            include_history=include_history
        )
        return response

    except Exception as e:
        logger.error(f"Regime detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}", response_model=RegimeHistoryResponse)
async def get_regime_history(
    symbol: str,
    timeframe: str = Query(default="1h"),
    days: int = Query(default=30, ge=1, le=365)
):
    """
    Get historical regime changes for a symbol.

    Shows how the market regime changed over time.

    - **symbol**: Trading symbol
    - **timeframe**: Timeframe
    - **days**: Number of days of history
    """
    try:
        response = await regime_detection_service.get_regime_history(
            symbol=symbol,
            timeframe=timeframe,
            days=days
        )
        return response

    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regimes")
async def get_regime_types():
    """
    Get list of supported regime types with descriptions.
    """
    return {
        "regimes": [
            {
                "type": MarketRegime.BULL_TREND.value,
                "name": "Bull Trend",
                "description": "Upward price movement with relatively low volatility",
                "characteristics": ["Higher highs", "Higher lows", "Positive returns"],
                "trading_bias": "long"
            },
            {
                "type": MarketRegime.BEAR_TREND.value,
                "name": "Bear Trend",
                "description": "Downward price movement with elevated volatility",
                "characteristics": ["Lower highs", "Lower lows", "Negative returns"],
                "trading_bias": "short"
            },
            {
                "type": MarketRegime.SIDEWAYS.value,
                "name": "Sideways/Range",
                "description": "Price consolidation with low directional movement",
                "characteristics": ["Range-bound", "Low volatility", "Mean reversion"],
                "trading_bias": "neutral"
            },
            {
                "type": MarketRegime.HIGH_VOLATILITY.value,
                "name": "High Volatility",
                "description": "Elevated volatility with unclear direction",
                "characteristics": ["Large price swings", "Uncertainty", "Risk-off"],
                "trading_bias": "reduce_exposure"
            }
        ]
    }


@router.post("/scan")
async def scan_regimes(
    symbols: list[str],
    timeframe: str = "1h"
):
    """
    Scan multiple symbols for current regimes.

    Returns regime classification for each symbol.
    """
    try:
        results = []

        for symbol in symbols:
            response = await regime_detection_service.detect_regime(
                symbol=symbol,
                timeframe=timeframe,
                lookback=300,
                include_history=False
            )

            results.append({
                "symbol": symbol,
                "regime": response.current_regime.value,
                "probability": response.regime_probability,
                "duration": response.regime_duration
            })

        # Group by regime
        regime_groups = {}
        for r in results:
            regime = r["regime"]
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(r["symbol"])

        return {
            "timestamp": results[0]["symbol"] if results else None,
            "results": results,
            "by_regime": regime_groups,
            "summary": {
                regime: len(symbols)
                for regime, symbols in regime_groups.items()
            }
        }

    except Exception as e:
        logger.error(f"Scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
