"""Feature embedding endpoints."""

from typing import List
from fastapi import APIRouter, HTTPException
import numpy as np
from loguru import logger

from ..models.schemas import (
    FeatureEmbeddingRequest,
    EmbeddingResponse,
    EmbeddingType,
)
from ..services.embedding_service import embedding_service

router = APIRouter()


@router.post("/embed", response_model=EmbeddingResponse)
async def embed_features(request: FeatureEmbeddingRequest):
    """
    Compress feature vectors using autoencoder.

    - **features**: List of feature vectors to embed

    Returns compressed embeddings (128 dimensions).
    """
    try:
        features = np.array(request.features, dtype=np.float32)
        result = await embedding_service.embed_features(features)

        return EmbeddingResponse(
            embeddings=result.embedding.tolist(),
            dimension=result.dimension,
            model=result.model_name,
            embedding_type=result.embedding_type,
            cached=result.cached,
            count=len(request.features)
        )
    except Exception as e:
        logger.error(f"Error embedding features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed-technical-indicators")
async def embed_technical_indicators(
    symbol: str,
    timeframe: str = "1h",
    lookback: int = 100
):
    """
    Generate embedding from technical indicators.

    Automatically calculates and embeds common indicators:
    - RSI, MACD, Bollinger Bands, ATR, etc.

    - **symbol**: Trading symbol
    - **timeframe**: Timeframe
    - **lookback**: Number of candles
    """
    try:
        from src.services.data_gateway_service import data_gateway

        # Fetch OHLCV data
        data = await data_gateway.get_historical_data(
            symbol=symbol,
            interval=timeframe,
            limit=lookback + 50  # Extra for indicator calculation
        )

        if not data or len(data) < lookback:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient data for {symbol}"
            )

        # Extract price arrays
        closes = np.array([d['close'] for d in data], dtype=np.float32)
        highs = np.array([d['high'] for d in data], dtype=np.float32)
        lows = np.array([d['low'] for d in data], dtype=np.float32)
        volumes = np.array([d.get('volume', 0) for d in data], dtype=np.float32)

        # Calculate technical indicators
        features = _calculate_indicators(closes, highs, lows, volumes)

        # Take last 'lookback' rows
        features = features[-lookback:]

        # Generate embedding
        result = await embedding_service.embed_features(features)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "embedding": result.embedding.tolist(),
            "dimension": result.dimension,
            "indicators_used": [
                "returns", "volatility", "rsi", "macd", "bb_position",
                "atr", "volume_ratio", "momentum"
            ],
            "cached": result.cached
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding technical indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_indicators(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray
) -> np.ndarray:
    """Calculate technical indicators for embedding."""
    n = len(closes)
    features = []

    for i in range(50, n):
        window_close = closes[i-50:i+1]
        window_high = highs[i-50:i+1]
        window_low = lows[i-50:i+1]
        window_vol = volumes[i-50:i+1]

        # Returns
        returns = (window_close[-1] - window_close[-2]) / window_close[-2] if window_close[-2] > 0 else 0

        # Volatility (20-period)
        volatility = np.std(np.diff(np.log(window_close[-21:]))) if len(window_close) > 20 else 0

        # RSI (14-period)
        deltas = np.diff(window_close[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = _ema(window_close, 12)
        ema26 = _ema(window_close, 26)
        macd = ema12 - ema26

        # Bollinger Bands position
        sma20 = np.mean(window_close[-20:])
        std20 = np.std(window_close[-20:])
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_position = (window_close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # ATR (14-period)
        tr_values = []
        for j in range(1, min(15, len(window_high))):
            tr = max(
                window_high[-j] - window_low[-j],
                abs(window_high[-j] - window_close[-j-1]),
                abs(window_low[-j] - window_close[-j-1])
            )
            tr_values.append(tr)
        atr = np.mean(tr_values) if tr_values else 0
        atr_normalized = atr / window_close[-1] if window_close[-1] > 0 else 0

        # Volume ratio
        vol_sma = np.mean(window_vol[-20:]) if len(window_vol) > 20 else 1
        vol_ratio = window_vol[-1] / vol_sma if vol_sma > 0 else 1

        # Momentum (10-period)
        momentum = (window_close[-1] - window_close[-11]) / window_close[-11] if len(window_close) > 10 and window_close[-11] > 0 else 0

        # SMA slopes
        sma20_prev = np.mean(window_close[-25:-5]) if len(window_close) > 25 else sma20
        sma_slope = (sma20 - sma20_prev) / sma20_prev if sma20_prev > 0 else 0

        # Price position relative to SMAs
        sma50 = np.mean(window_close) if len(window_close) >= 50 else sma20
        price_to_sma20 = (window_close[-1] - sma20) / sma20 if sma20 > 0 else 0
        price_to_sma50 = (window_close[-1] - sma50) / sma50 if sma50 > 0 else 0

        features.append([
            returns,
            volatility,
            rsi / 100,  # Normalize to 0-1
            macd / window_close[-1] if window_close[-1] > 0 else 0,  # Normalize
            np.clip(bb_position, 0, 1),
            atr_normalized,
            np.clip(vol_ratio, 0, 5) / 5,  # Normalize
            momentum,
            sma_slope,
            price_to_sma20,
            price_to_sma50,
        ])

    return np.array(features, dtype=np.float32)


def _ema(data: np.ndarray, period: int) -> float:
    """Calculate EMA."""
    if len(data) < period:
        return data[-1]

    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return np.convolve(data[-period:], weights, mode='valid')[-1]
