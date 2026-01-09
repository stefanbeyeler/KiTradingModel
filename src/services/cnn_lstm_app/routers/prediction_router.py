"""
Prediction Router für CNN-LSTM Inference Service.

Endpoints für Multi-Task Vorhersagen.
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PatternsOnlyResponse,
    PredictionResponse,
    PriceOnlyResponse,
    RegimeOnlyResponse,
)

router = APIRouter()


# =============================================================================
# Combined Predictions
# =============================================================================

@router.get("/predict/{symbol}", response_model=PredictionResponse, tags=["2. Predictions"])
async def predict_all(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe (M1-MN)")
):
    """
    Kombinierte Multi-Task Vorhersage.

    Gibt Preis-Vorhersage, Pattern-Klassifikation und Regime-Vorhersage zurueck.

    - **symbol**: Trading-Symbol (z.B. BTCUSD, EURUSD, AAPL)
    - **timeframe**: Timeframe fuer Analyse (M1, M5, M15, M30, H1, H4, D1, W1, MN)
    """
    from ..services.inference_service import inference_service

    if not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    result = await inference_service.predict(symbol, timeframe)

    if result is None:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for {symbol}"
        )

    return result


# =============================================================================
# Task-Specific Predictions
# =============================================================================

@router.get("/price/{symbol}", response_model=PriceOnlyResponse, tags=["2. Predictions"])
async def predict_price(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe (M1-MN)")
):
    """
    Nur Preis-Vorhersage.

    Gibt Preis-Forecasts fuer verschiedene Zeithorizonte zurueck.
    """
    from ..services.inference_service import inference_service

    if not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    start_time = time.time()
    prediction = await inference_service.predict_price(symbol, timeframe)

    if prediction is None:
        raise HTTPException(
            status_code=500,
            detail=f"Price prediction failed for {symbol}"
        )

    return PriceOnlyResponse(
        symbol=symbol,
        timeframe=timeframe.upper(),
        timestamp=datetime.now(timezone.utc),
        prediction=prediction,
        model_version=inference_service.get_model_version() or "unknown",
        inference_time_ms=round((time.time() - start_time) * 1000, 2)
    )


@router.get("/patterns/{symbol}", response_model=PatternsOnlyResponse, tags=["2. Predictions"])
async def predict_patterns(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe (M1-MN)"),
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimale Konfidenz")
):
    """
    Nur Pattern-Klassifikation.

    Gibt erkannte Chart-Patterns mit Konfidenz zurueck.
    """
    from ..services.inference_service import inference_service

    if not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    start_time = time.time()
    patterns = await inference_service.predict_patterns(symbol, timeframe)

    if patterns is None:
        raise HTTPException(
            status_code=500,
            detail=f"Pattern prediction failed for {symbol}"
        )

    # Filter nach Konfidenz
    filtered_patterns = [p for p in patterns if p.confidence >= min_confidence]

    return PatternsOnlyResponse(
        symbol=symbol,
        timeframe=timeframe.upper(),
        timestamp=datetime.now(timezone.utc),
        patterns=filtered_patterns,
        total_patterns_detected=len(filtered_patterns),
        model_version=inference_service.get_model_version() or "unknown",
        inference_time_ms=round((time.time() - start_time) * 1000, 2)
    )


@router.get("/regime/{symbol}", response_model=RegimeOnlyResponse, tags=["2. Predictions"])
async def predict_regime(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe (M1-MN)")
):
    """
    Nur Regime-Vorhersage.

    Gibt aktuelles Markt-Regime und Uebergangswahrscheinlichkeiten zurueck.
    """
    from ..services.inference_service import inference_service

    if not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    start_time = time.time()
    regime = await inference_service.predict_regime(symbol, timeframe)

    if regime is None:
        raise HTTPException(
            status_code=500,
            detail=f"Regime prediction failed for {symbol}"
        )

    return RegimeOnlyResponse(
        symbol=symbol,
        timeframe=timeframe.upper(),
        timestamp=datetime.now(timezone.utc),
        regime=regime,
        model_version=inference_service.get_model_version() or "unknown",
        inference_time_ms=round((time.time() - start_time) * 1000, 2)
    )


# =============================================================================
# Batch Predictions
# =============================================================================

@router.post("/batch", response_model=BatchPredictionResponse, tags=["2. Predictions"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch-Vorhersage fuer mehrere Symbole.

    Fuehrt Vorhersagen fuer alle angegebenen Symbole parallel aus.
    """
    from ..services.inference_service import inference_service

    if not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    for symbol in request.symbols:
        try:
            prediction = await inference_service.predict(symbol, request.timeframe)

            if prediction:
                results.append(BatchPredictionItem(
                    symbol=symbol,
                    success=True,
                    prediction=prediction
                ))
                successful += 1
            else:
                results.append(BatchPredictionItem(
                    symbol=symbol,
                    success=False,
                    error="Prediction failed"
                ))
                failed += 1

        except Exception as e:
            logger.error(f"Batch prediction error for {symbol}: {e}")
            results.append(BatchPredictionItem(
                symbol=symbol,
                success=False,
                error=str(e)
            ))
            failed += 1

    return BatchPredictionResponse(
        timeframe=request.timeframe.upper(),
        timestamp=datetime.now(timezone.utc),
        total_symbols=len(request.symbols),
        successful=successful,
        failed=failed,
        results=results,
        total_inference_time_ms=round((time.time() - start_time) * 1000, 2)
    )


# =============================================================================
# Analysis Endpoints
# =============================================================================

@router.get("/analysis/{symbol}", tags=["2. Predictions"])
async def full_analysis(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe")
):
    """
    Vollstaendige Analyse mit zusaetzlichen Insights.

    Kombiniert Multi-Task Vorhersagen mit interpretierbaren Erklaerungen.
    """
    from ..services.inference_service import inference_service

    if not inference_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    result = await inference_service.predict(symbol, timeframe)

    if result is None:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed for {symbol}"
        )

    # Generiere Insights
    insights = []

    # Price Insight
    price = result.predictions.price
    if price.direction.value == "bullish":
        insights.append(f"Bullish outlook with {price.confidence*100:.0f}% confidence")
    elif price.direction.value == "bearish":
        insights.append(f"Bearish outlook with {price.confidence*100:.0f}% confidence")
    else:
        insights.append("Neutral/sideways price expectation")

    # Pattern Insights
    if result.predictions.patterns:
        top_pattern = result.predictions.patterns[0]
        insights.append(f"Detected {top_pattern.type.value} pattern ({top_pattern.confidence*100:.0f}% confidence)")

    # Regime Insight
    regime = result.predictions.regime
    regime_descriptions = {
        "bull_trend": "Market is in a bullish trend phase",
        "bear_trend": "Market is in a bearish trend phase",
        "sideways": "Market is ranging/consolidating",
        "high_volatility": "High volatility detected - exercise caution"
    }
    insights.append(regime_descriptions.get(regime.current.value, "Unknown regime"))

    return {
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions": result.predictions.model_dump(),
        "insights": insights,
        "recommendation": _generate_recommendation(result.predictions),
        "model_version": result.model_version,
        "inference_time_ms": result.inference_time_ms
    }


def _generate_recommendation(predictions) -> dict:
    """Generiert Trading-Empfehlung basierend auf Vorhersagen."""
    price = predictions.price
    regime = predictions.regime
    patterns = predictions.patterns

    # Scoring
    score = 0

    # Price Direction
    if price.direction.value == "bullish":
        score += 2
    elif price.direction.value == "bearish":
        score -= 2

    # Confidence
    score += (price.confidence - 0.5) * 2

    # Regime
    if regime.current.value == "bull_trend":
        score += 1
    elif regime.current.value == "bear_trend":
        score -= 1
    elif regime.current.value == "high_volatility":
        score *= 0.5  # Reduziere bei hoher Volatilitaet

    # Patterns
    for pattern in patterns[:2]:
        if pattern.direction.value == "bullish":
            score += pattern.confidence * 0.5
        else:
            score -= pattern.confidence * 0.5

    # Generate Recommendation
    if score > 2:
        action = "STRONG_BUY"
    elif score > 0.5:
        action = "BUY"
    elif score < -2:
        action = "STRONG_SELL"
    elif score < -0.5:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "action": action,
        "score": round(score, 2),
        "confidence": price.confidence,
        "risk_level": "HIGH" if regime.current.value == "high_volatility" else "NORMAL"
    }
