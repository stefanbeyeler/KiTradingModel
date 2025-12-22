"""Regime detection service."""

from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import os
from loguru import logger

from ..models.hmm_regime_model import HMMRegimeModel, MarketRegime, RegimeState
from ..models.schemas import RegimeDetectionResponse, RegimeHistoryResponse, RegimeHistoryEntry


class RegimeDetectionService:
    """
    Service for market regime detection using HMM.

    Detects four market regimes:
    - Bull Trend: Upward trend with low volatility
    - Bear Trend: Downward trend with elevated volatility
    - Sideways: Low directional movement
    - High Volatility: High volatility, unclear direction
    """

    MODEL_DIR = "data/models/hmm"

    def __init__(self):
        """Initialize the service."""
        self._models: Dict[str, HMMRegimeModel] = {}
        self._default_model: Optional[HMMRegimeModel] = None

        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def get_model(self, symbol: str) -> HMMRegimeModel:
        """Get or create model for symbol."""
        if symbol not in self._models:
            # Try to load saved model
            model_path = os.path.join(self.MODEL_DIR, f"{symbol}_hmm.pkl")
            if os.path.exists(model_path):
                self._models[symbol] = HMMRegimeModel.load(model_path)
            else:
                self._models[symbol] = HMMRegimeModel()

        return self._models[symbol]

    async def detect_regime(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback: int = 500,
        include_history: bool = False
    ) -> RegimeDetectionResponse:
        """
        Detect the current market regime.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            lookback: Number of candles to analyze
            include_history: Include regime history in response

        Returns:
            RegimeDetectionResponse
        """
        try:
            # Fetch data
            from src.services.data_gateway_service import data_gateway

            data = await data_gateway.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                limit=lookback
            )

            if not data or len(data) < 100:
                return self._empty_response(symbol, timeframe, "Insufficient data")

            # Extract prices
            closes = np.array([d['close'] for d in data], dtype=np.float64)
            timestamps = [d.get('timestamp', d.get('time', '')) for d in data]

            # Get model
            model = self.get_model(symbol)

            # Fit if not fitted
            if not model.is_fitted():
                logger.info(f"Fitting HMM model for {symbol}")
                model.fit(closes)

            # Predict regime
            regime_state = model.predict_regime(closes)

            # Get history if requested
            regime_history = None
            if include_history:
                history = model.get_regime_history(closes)
                regime_history = [
                    {
                        "timestamp": str(timestamps[i]) if i < len(timestamps) else "",
                        "regime": regime.value,
                        "probability": round(prob, 4)
                    }
                    for i, (regime, prob) in enumerate(history)
                ]

            # Calculate market metrics
            ohlcv = np.array([
                [d['open'], d['high'], d['low'], d['close'], d.get('volume', 0)]
                for d in data
            ], dtype=np.float64)
            metrics = self._calculate_market_metrics(ohlcv)

            return RegimeDetectionResponse(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                current_regime=regime_state.regime,
                regime_probability=round(regime_state.probability, 4),
                regime_duration=regime_state.duration,
                transition_probabilities={
                    k.value: round(v, 4)
                    for k, v in regime_state.transition_probs.items()
                },
                regime_history=regime_history,
                market_metrics=metrics
            )

        except Exception as e:
            logger.error(f"Regime detection error for {symbol}: {e}")
            return self._empty_response(symbol, timeframe, str(e))

    async def get_regime_history(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 30
    ) -> RegimeHistoryResponse:
        """
        Get historical regime changes.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            days: Number of days

        Returns:
            RegimeHistoryResponse
        """
        try:
            from src.services.data_gateway_service import data_gateway

            # Calculate candles needed
            candles_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
            limit = days * candles_per_day.get(timeframe, 24)

            data = await data_gateway.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )

            if not data or len(data) < 100:
                return RegimeHistoryResponse(
                    symbol=symbol,
                    timeframe=timeframe,
                    entries=[],
                    total_count=0,
                    regime_distribution={}
                )

            closes = np.array([d['close'] for d in data], dtype=np.float64)
            timestamps = [d.get('timestamp', d.get('time', '')) for d in data]

            model = self.get_model(symbol)
            if not model.is_fitted():
                model.fit(closes)

            history = model.get_regime_history(closes)

            # Create entries
            entries = []
            regime_counts = {r.value: 0 for r in MarketRegime}

            for i, (regime, prob) in enumerate(history):
                if i < len(timestamps):
                    entries.append(RegimeHistoryEntry(
                        timestamp=str(timestamps[i]),
                        regime=regime,
                        probability=round(prob, 4)
                    ))
                    regime_counts[regime.value] += 1

            # Calculate distribution
            total = len(history)
            distribution = {
                k: round(v / total, 4) if total > 0 else 0
                for k, v in regime_counts.items()
            }

            return RegimeHistoryResponse(
                symbol=symbol,
                timeframe=timeframe,
                entries=entries[-100:],  # Limit to last 100
                total_count=len(entries),
                regime_distribution=distribution
            )

        except Exception as e:
            logger.error(f"History error for {symbol}: {e}")
            return RegimeHistoryResponse(
                symbol=symbol,
                timeframe=timeframe,
                entries=[],
                total_count=0,
                regime_distribution={}
            )

    def _calculate_market_metrics(self, ohlcv: np.ndarray) -> Dict:
        """Calculate market metrics."""
        closes = ohlcv[:, 3]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        volumes = ohlcv[:, 4]

        # Current price
        current_price = closes[-1]

        # Returns
        returns = np.diff(np.log(closes))

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # Trend (SMA comparison)
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price

        if current_price > sma_20 > sma_50:
            trend = "bullish"
        elif current_price < sma_20 < sma_50:
            trend = "bearish"
        else:
            trend = "neutral"

        # ATR
        tr_values = []
        for i in range(1, min(15, len(highs))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]) if i < len(closes) else 0,
                abs(lows[-i] - closes[-i-1]) if i < len(closes) else 0
            )
            tr_values.append(tr)
        atr = np.mean(tr_values) if tr_values else 0

        # Volume
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]

        return {
            "price": round(float(current_price), 4),
            "trend": trend,
            "volatility_annualized": round(float(volatility), 4),
            "sma_20": round(float(sma_20), 4),
            "sma_50": round(float(sma_50), 4),
            "atr_14": round(float(atr), 4),
            "avg_volume_20": round(float(avg_volume), 2)
        }

    def _empty_response(
        self,
        symbol: str,
        timeframe: str,
        error: str
    ) -> RegimeDetectionResponse:
        """Create empty response for error cases."""
        return RegimeDetectionResponse(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            current_regime=MarketRegime.SIDEWAYS,
            regime_probability=0.0,
            regime_duration=0,
            transition_probabilities={},
            regime_history=None,
            market_metrics={"error": error}
        )

    async def train_model(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 365
    ) -> Dict:
        """
        Train HMM model for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            lookback_days: Days of training data

        Returns:
            Training result
        """
        try:
            from src.services.data_gateway_service import data_gateway

            candles_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
            limit = lookback_days * candles_per_day.get(timeframe, 24)

            data = await data_gateway.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )

            if not data or len(data) < 200:
                return {"status": "failed", "message": "Insufficient data"}

            closes = np.array([d['close'] for d in data], dtype=np.float64)

            # Create and fit model
            model = HMMRegimeModel()
            model.fit(closes)

            # Save model
            model_path = os.path.join(self.MODEL_DIR, f"{symbol}_hmm.pkl")
            model.save(model_path)

            # Store in cache
            self._models[symbol] = model

            return {
                "status": "completed",
                "symbol": symbol,
                "samples": len(closes),
                "model_path": model_path
            }

        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            return {"status": "failed", "message": str(e)}


# Singleton instance
regime_detection_service = RegimeDetectionService()
