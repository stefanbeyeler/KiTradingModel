"""Signal scoring endpoints."""

from typing import List
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.schemas import (
    SignalScoringRequest,
    SignalScoringResponse,
    BatchSignalRequest,
    BatchSignalResponse,
    SignalType,
)
from ..services.signal_scoring_service import signal_scoring_service

router = APIRouter()


@router.post("/score", response_model=SignalScoringResponse)
async def score_signal(request: SignalScoringRequest):
    """
    Score a trading signal based on current market regime.

    Uses LightGBM to evaluate signal quality considering:
    - Current market regime (from HMM)
    - Technical indicators
    - Price action
    - Volume profile

    Score interpretation:
    - **80-100**: Strong signal, well-aligned with regime
    - **60-79**: Moderate signal
    - **40-59**: Weak signal, use caution
    - **0-39**: Poor signal, likely contrary to regime

    Parameters:
    - **symbol**: Trading symbol
    - **signal_type**: "long" or "short"
    - **entry_price**: Optional proposed entry price
    - **timeframe**: Timeframe for analysis
    """
    try:
        response = await signal_scoring_service.score_signal(
            symbol=request.symbol,
            signal_type=request.signal_type,
            timeframe=request.timeframe,
            entry_price=request.entry_price
        )
        return response

    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/score/{symbol}")
async def score_signal_get(
    symbol: str,
    signal_type: str = Query(..., description="Signal type: long or short"),
    timeframe: str = Query(default="1h"),
    entry_price: float = Query(default=None)
):
    """
    Score a signal (GET endpoint).

    Quick way to evaluate a trading signal.
    """
    try:
        sig_type = SignalType(signal_type.lower())

        response = await signal_scoring_service.score_signal(
            symbol=symbol,
            signal_type=sig_type,
            timeframe=timeframe,
            entry_price=entry_price
        )
        return response

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid signal_type: {signal_type}. Use 'long' or 'short'"
        )
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-score", response_model=BatchSignalResponse)
async def score_batch_signals(request: BatchSignalRequest):
    """
    Score multiple signals at once.

    Useful for evaluating multiple trading opportunities.
    """
    try:
        results = await signal_scoring_service.score_batch([
            {
                "symbol": s.symbol,
                "signal_type": s.signal_type.value,
                "timeframe": s.timeframe,
                "entry_price": s.entry_price
            }
            for s in request.signals
        ])

        avg_score = sum(r.score for r in results) / len(results) if results else 0

        return BatchSignalResponse(
            results=results,
            total_scored=len(results),
            average_score=round(avg_score, 2)
        )

    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-setup")
async def evaluate_trading_setup(
    symbol: str,
    timeframe: str = "1h"
):
    """
    Evaluate both long and short setups for a symbol.

    Returns scores for both directions to help decide trade direction.
    """
    try:
        long_response = await signal_scoring_service.score_signal(
            symbol=symbol,
            signal_type=SignalType.LONG,
            timeframe=timeframe
        )

        short_response = await signal_scoring_service.score_signal(
            symbol=symbol,
            signal_type=SignalType.SHORT,
            timeframe=timeframe
        )

        # Determine preferred direction
        if long_response.score > short_response.score + 10:
            preferred = "long"
            bias = "bullish"
        elif short_response.score > long_response.score + 10:
            preferred = "short"
            bias = "bearish"
        else:
            preferred = "neutral"
            bias = "neutral"

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_regime": long_response.current_regime.value,
            "long_setup": {
                "score": long_response.score,
                "confidence": long_response.confidence,
                "alignment": long_response.regime_alignment,
                "risk": long_response.risk_assessment
            },
            "short_setup": {
                "score": short_response.score,
                "confidence": short_response.confidence,
                "alignment": short_response.regime_alignment,
                "risk": short_response.risk_assessment
            },
            "preferred_direction": preferred,
            "market_bias": bias,
            "score_difference": round(long_response.score - short_response.score, 2)
        }

    except Exception as e:
        logger.error(f"Setup evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alignment-check/{symbol}")
async def check_regime_alignment(
    symbol: str,
    signal_type: str,
    timeframe: str = "1h"
):
    """
    Quick check if a signal aligns with current regime.

    Returns simple aligned/neutral/contrary classification.
    """
    try:
        sig_type = SignalType(signal_type.lower())

        response = await signal_scoring_service.score_signal(
            symbol=symbol,
            signal_type=sig_type,
            timeframe=timeframe
        )

        return {
            "symbol": symbol,
            "signal_type": signal_type,
            "current_regime": response.current_regime.value,
            "alignment": response.regime_alignment,
            "recommendation": "proceed" if response.regime_alignment == "aligned" else "caution"
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid signal_type: {signal_type}"
        )
    except Exception as e:
        logger.error(f"Alignment check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
