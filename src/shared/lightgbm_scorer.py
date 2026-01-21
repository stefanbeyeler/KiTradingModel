"""Shared LightGBM Signal Scorer for trading signals.

This module is shared between:
- hmm_app (inference) - uses score_signal() for live scoring
- hmm_train_app (training) - uses fit() for model training

Both services use the same feature calculation to ensure consistency.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
from loguru import logger

# Lazy imports
lgb = None
joblib = None


def _load_lightgbm():
    global lgb, joblib
    if lgb is None:
        try:
            import lightgbm as l
            import joblib as jl
            lgb = l
            joblib = jl
            return True
        except ImportError:
            logger.warning("lightgbm not installed. Scorer will use fallback.")
            return False
    return True


class SignalType(str, Enum):
    """Trading signal types."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class MarketRegime(str, Enum):
    """Market regime types (for standalone use without hmm_regime_model)."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class SignalScore:
    """Signal scoring result."""
    signal_type: SignalType
    score: float           # 0-100
    confidence: float      # 0-1
    regime_alignment: str  # "aligned", "neutral", "contrary"
    feature_importance: Dict[str, float]


class LightGBMSignalScorer:
    """
    LightGBM-based Signal Scorer.

    Evaluates trading signals based on:
    - Current market regime (from HMM)
    - Technical indicators
    - Price action features
    - Volume profile
    """

    FEATURE_COLUMNS = [
        # Regime features (from HMM)
        "regime_bull_prob",
        "regime_bear_prob",
        "regime_sideways_prob",
        "regime_highvol_prob",
        "regime_duration",

        # Technical indicators
        "rsi_14",
        "macd_signal",
        "macd_histogram",
        "bb_position",
        "atr_normalized",

        # Trend features
        "sma_20_slope",
        "sma_50_slope",
        "ema_cross_signal",

        # Price action
        "higher_highs_count",
        "lower_lows_count",
        "last_swing_type",

        # Volume
        "volume_sma_ratio",
        "volume_trend",
    ]

    def __init__(self):
        """Initialize the scorer."""
        self._model = None
        self._is_fitted = False
        self._feature_importance: Dict[str, float] = {}

    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def calculate_features(
        self,
        ohlcv: np.ndarray,
        regime_probs: Dict[str, float],
        regime_duration: int
    ) -> np.ndarray:
        """
        Calculate all features for scoring.

        Args:
            ohlcv: OHLCV data (n, 5) - Open, High, Low, Close, Volume
            regime_probs: Regime probabilities from HMM
            regime_duration: Current regime duration

        Returns:
            Feature vector (18 features)
        """
        if len(ohlcv) < 50:
            # Return neutral features if insufficient data
            return np.zeros(len(self.FEATURE_COLUMNS))

        high = ohlcv[:, 1]
        low = ohlcv[:, 2]
        close = ohlcv[:, 3]
        volume = ohlcv[:, 4]

        features = {}

        # Regime features
        features["regime_bull_prob"] = regime_probs.get("bull_trend", 0)
        features["regime_bear_prob"] = regime_probs.get("bear_trend", 0)
        features["regime_sideways_prob"] = regime_probs.get("sideways", 0)
        features["regime_highvol_prob"] = regime_probs.get("high_volatility", 0)
        features["regime_duration"] = min(regime_duration / 100, 1.0)  # Normalize

        # RSI
        features["rsi_14"] = self._calc_rsi(close, 14) / 100  # Normalize

        # MACD
        macd, signal, hist = self._calc_macd(close)
        features["macd_signal"] = 1 if macd > signal else -1
        features["macd_histogram"] = np.clip(hist / close[-1] * 100, -1, 1)

        # Bollinger Bands position
        bb_upper, bb_lower = self._calc_bollinger(close)
        bb_range = bb_upper - bb_lower
        features["bb_position"] = np.clip(
            (close[-1] - bb_lower) / (bb_range + 1e-8), 0, 1
        )

        # ATR
        atr = self._calc_atr(high, low, close)
        features["atr_normalized"] = atr / close[-1] if close[-1] > 0 else 0

        # SMA slopes
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        sma_20_prev = np.mean(close[-25:-5]) if len(close) >= 25 else sma_20
        features["sma_20_slope"] = np.clip(
            (sma_20 - sma_20_prev) / (sma_20_prev + 1e-8), -0.1, 0.1
        ) * 10

        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
        sma_50_prev = np.mean(close[-55:-5]) if len(close) >= 55 else sma_50
        features["sma_50_slope"] = np.clip(
            (sma_50 - sma_50_prev) / (sma_50_prev + 1e-8), -0.1, 0.1
        ) * 10

        # EMA cross
        ema_12 = self._calc_ema(close, 12)
        ema_26 = self._calc_ema(close, 26)
        features["ema_cross_signal"] = 1 if ema_12 > ema_26 else -1

        # Price action
        features["higher_highs_count"] = self._count_higher_highs(high[-20:]) / 10
        features["lower_lows_count"] = self._count_lower_lows(low[-20:]) / 10
        features["last_swing_type"] = 1 if high[-1] > np.max(high[-10:-1]) else -1

        # Volume
        vol_sma = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
        features["volume_sma_ratio"] = np.clip(
            volume[-1] / (vol_sma + 1e-8), 0, 5
        ) / 5

        vol_recent = np.mean(volume[-5:]) if len(volume) >= 5 else volume[-1]
        vol_older = np.mean(volume[-20:-5]) if len(volume) >= 20 else vol_recent
        features["volume_trend"] = np.clip(
            (vol_recent - vol_older) / (vol_older + 1e-8), -1, 1
        )

        return np.array([features[col] for col in self.FEATURE_COLUMNS])

    # Alias for backward compatibility
    _calculate_features = calculate_features

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> "LightGBMSignalScorer":
        """
        Train the LightGBM scorer.

        Args:
            X: Feature matrix (n_samples, 18 features)
            y: Target values (signal quality 0-100)
            eval_set: Optional validation set for early stopping

        Returns:
            self
        """
        if not _load_lightgbm():
            logger.warning("LightGBM not available, using fallback scoring")
            self._is_fitted = False
            return self

        try:
            train_data = lgb.Dataset(
                X, label=y,
                feature_name=self.FEATURE_COLUMNS
            )

            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1
            }

            valid_sets = []
            callbacks = []

            if eval_set is not None:
                valid_data = lgb.Dataset(eval_set[0], label=eval_set[1])
                valid_sets = [valid_data]
                callbacks = [lgb.early_stopping(50)]

            self._model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=valid_sets,
                callbacks=callbacks
            )

            # Feature importance
            importance = self._model.feature_importance(importance_type='gain')
            total_importance = sum(importance) + 1e-8
            self._feature_importance = {
                col: float(imp / total_importance)
                for col, imp in zip(self.FEATURE_COLUMNS, importance)
            }

            self._is_fitted = True
            logger.info("LightGBM scorer trained successfully")

        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            self._is_fitted = False

        return self

    def score_signal(
        self,
        ohlcv: np.ndarray,
        regime_probs: Dict[str, float],
        regime_duration: int,
        signal_type: SignalType
    ) -> SignalScore:
        """
        Score a trading signal.

        Args:
            ohlcv: OHLCV data
            regime_probs: Regime probabilities from HMM
            regime_duration: Current regime duration
            signal_type: Signal type (long/short)

        Returns:
            SignalScore with score, confidence, and alignment
        """
        features = self.calculate_features(ohlcv, regime_probs, regime_duration)

        # Get raw score
        if self._is_fitted and self._model is not None:
            raw_score = self._model.predict([features])[0]
        else:
            raw_score = self._fallback_score(features, signal_type)

        score = float(np.clip(raw_score, 0, 100))

        # Calculate confidence
        confidence = self._calculate_confidence(features, regime_probs)

        # Check regime alignment
        alignment = self._check_regime_alignment(signal_type, regime_probs)

        # Adjust score based on alignment
        if alignment == "contrary":
            score *= 0.7
        elif alignment == "aligned":
            score *= 1.1
        score = float(np.clip(score, 0, 100))

        return SignalScore(
            signal_type=signal_type,
            score=round(score, 2),
            confidence=round(confidence, 4),
            regime_alignment=alignment,
            feature_importance=self._feature_importance or {}
        )

    def _fallback_score(
        self,
        features: np.ndarray,
        signal_type: SignalType
    ) -> float:
        """Fallback scoring when model not available."""
        # Simple rule-based scoring
        base_score = 50.0

        # RSI factor
        rsi = features[5] * 100  # Denormalize
        if signal_type == SignalType.LONG:
            if rsi < 30:
                base_score += 15  # Oversold, good for long
            elif rsi > 70:
                base_score -= 15  # Overbought, bad for long
        elif signal_type == SignalType.SHORT:
            if rsi > 70:
                base_score += 15  # Overbought, good for short
            elif rsi < 30:
                base_score -= 15  # Oversold, bad for short

        # Regime factor
        bull_prob = features[0]
        bear_prob = features[1]

        if signal_type == SignalType.LONG:
            base_score += (bull_prob - bear_prob) * 20
        elif signal_type == SignalType.SHORT:
            base_score += (bear_prob - bull_prob) * 20

        # EMA cross factor
        ema_cross = features[12]
        if signal_type == SignalType.LONG and ema_cross > 0:
            base_score += 10
        elif signal_type == SignalType.SHORT and ema_cross < 0:
            base_score += 10

        return np.clip(base_score, 0, 100)

    def _calculate_confidence(
        self,
        features: np.ndarray,
        regime_probs: Dict[str, float]
    ) -> float:
        """Calculate confidence based on feature quality."""
        # Higher confidence when regime is clear
        regime_max = max(regime_probs.values()) if regime_probs else 0.25

        # Higher confidence when RSI is at extremes
        rsi = features[5] * 100
        rsi_clarity = abs(rsi - 50) / 50

        # Combine factors
        confidence = (regime_max * 0.6 + rsi_clarity * 0.4)
        return np.clip(confidence, 0.1, 0.95)

    def _check_regime_alignment(
        self,
        signal_type: SignalType,
        regime_probs: Dict[str, float]
    ) -> str:
        """Check if signal aligns with current regime."""
        bull_prob = regime_probs.get("bull_trend", 0)
        bear_prob = regime_probs.get("bear_trend", 0)

        if signal_type == SignalType.LONG:
            if bull_prob > 0.5:
                return "aligned"
            elif bear_prob > 0.5:
                return "contrary"
        elif signal_type == SignalType.SHORT:
            if bear_prob > 0.5:
                return "aligned"
            elif bull_prob > 0.5:
                return "contrary"

        return "neutral"

    # Technical indicator helpers
    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_macd(
        self,
        prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate MACD."""
        ema_12 = self._calc_ema(prices, 12)
        ema_26 = self._calc_ema(prices, 26)
        macd = ema_12 - ema_26

        # Simplified signal line
        signal = ema_12 * 0.5 + ema_26 * 0.5 - prices[-1] * 0.1

        return macd, signal, macd - signal

    def _calc_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1]

        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return float(np.convolve(prices[-period:], weights, mode='valid')[-1])

    def _calc_bollinger(
        self,
        prices: np.ndarray,
        period: int = 20
    ) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return prices[-1], prices[-1]

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        return sma + 2*std, sma - 2*std

    def _calc_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate ATR."""
        if len(high) < period + 1:
            return 0.0

        tr_values = []
        for i in range(1, min(period + 1, len(high))):
            tr = max(
                high[-i] - low[-i],
                abs(high[-i] - close[-i-1]),
                abs(low[-i] - close[-i-1])
            )
            tr_values.append(tr)
        return np.mean(tr_values) if tr_values else 0.0

    def _count_higher_highs(self, highs: np.ndarray) -> int:
        """Count consecutive higher highs."""
        count = 0
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                count += 1
        return count

    def _count_lower_lows(self, lows: np.ndarray) -> int:
        """Count consecutive lower lows."""
        count = 0
        for i in range(1, len(lows)):
            if lows[i] < lows[i-1]:
                count += 1
        return count

    def save(self, path: str):
        """Save model to disk."""
        if not _load_lightgbm():
            return

        if self._model is not None:
            data = {
                'model': self._model,
                'feature_importance': self._feature_importance,
                'is_fitted': self._is_fitted
            }
            joblib.dump(data, path)
            logger.info(f"Scorer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LightGBMSignalScorer":
        """Load model from disk."""
        if not _load_lightgbm():
            return cls()

        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return cls()

        try:
            data = joblib.load(path)
            instance = cls()
            instance._model = data.get('model')
            instance._feature_importance = data.get('feature_importance', {})
            instance._is_fitted = data.get('is_fitted', False)
            logger.info(f"Scorer loaded from {path}")
            return instance
        except Exception as e:
            logger.error(f"Failed to load scorer: {e}")
            return cls()
