"""
Accuracy tracking service for HMM regime detection.

Tracks predictions vs ground truth for Self-Learning feedback loop.
"""

from typing import Dict, List, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
from loguru import logger
import json
import os


@dataclass
class PredictionRecord:
    """Single prediction record for tracking."""
    timestamp: datetime
    symbol: str
    predicted_regime: int
    predicted_regime_name: str
    confidence: float
    price_at_prediction: float
    # Ground truth (filled later when we can evaluate)
    actual_regime: Optional[int] = None
    actual_regime_name: Optional[str] = None
    is_correct: Optional[bool] = None
    evaluated_at: Optional[datetime] = None


@dataclass
class SymbolAccuracyStats:
    """Accuracy statistics for a single symbol."""
    symbol: str
    total_predictions: int = 0
    evaluated_predictions: int = 0
    correct_predictions: int = 0
    rolling_accuracy: float = 0.0
    last_updated: Optional[datetime] = None

    # Per-regime stats
    regime_counts: Dict[str, int] = field(default_factory=dict)
    regime_correct: Dict[str, int] = field(default_factory=dict)


class AccuracyTracker:
    """
    Tracks HMM prediction accuracy over time.

    Features:
    - Rolling window accuracy calculation
    - Per-symbol and aggregate statistics
    - Ground truth evaluation using price action
    - Persistence to disk
    """

    REGIME_NAMES = ["bull_trend", "bear_trend", "sideways", "high_volatility"]
    DATA_DIR = "data/accuracy"

    def __init__(self, window_size: int = 500, evaluation_delay_bars: int = 5):
        """
        Initialize accuracy tracker.

        Args:
            window_size: Number of predictions to keep for rolling accuracy
            evaluation_delay_bars: How many bars to wait before evaluating prediction
        """
        self.window_size = window_size
        self.evaluation_delay_bars = evaluation_delay_bars

        # Prediction history per symbol
        self._predictions: Dict[str, Deque[PredictionRecord]] = {}

        # Aggregated stats per symbol
        self._stats: Dict[str, SymbolAccuracyStats] = {}

        # Global stats
        self._global_stats = SymbolAccuracyStats(symbol="GLOBAL")

        # Self-learning state
        self._self_learning_enabled = True
        self._accuracy_threshold = 0.40
        self._last_retrain_trigger: Optional[datetime] = None
        self._cooldown_hours = 6

        os.makedirs(self.DATA_DIR, exist_ok=True)
        self._load_state()

    def record_prediction(
        self,
        symbol: str,
        predicted_regime: int,
        confidence: float,
        price: float
    ) -> None:
        """
        Record a new prediction.

        Args:
            symbol: Trading symbol
            predicted_regime: Predicted regime index (0-3)
            confidence: Prediction confidence (0-1)
            price: Current price at prediction time
        """
        if symbol not in self._predictions:
            self._predictions[symbol] = deque(maxlen=self.window_size)
            self._stats[symbol] = SymbolAccuracyStats(symbol=symbol)

        record = PredictionRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            predicted_regime=predicted_regime,
            predicted_regime_name=self.REGIME_NAMES[predicted_regime] if 0 <= predicted_regime < 4 else "unknown",
            confidence=confidence,
            price_at_prediction=price
        )

        self._predictions[symbol].append(record)
        self._stats[symbol].total_predictions += 1
        self._stats[symbol].last_updated = record.timestamp

        # Update regime counts
        regime_name = record.predicted_regime_name
        self._stats[symbol].regime_counts[regime_name] = \
            self._stats[symbol].regime_counts.get(regime_name, 0) + 1

        self._global_stats.total_predictions += 1
        self._global_stats.last_updated = record.timestamp

    def evaluate_predictions(
        self,
        symbol: str,
        prices: List[float],
        lookback: int = 20
    ) -> int:
        """
        Evaluate pending predictions using price action.

        Uses the same ground truth logic as the validation service:
        - BULL_TREND: Strong positive returns
        - BEAR_TREND: Strong negative returns
        - SIDEWAYS: Low returns, low volatility
        - HIGH_VOLATILITY: High volatility

        Args:
            symbol: Trading symbol
            prices: Recent price data
            lookback: Lookback for calculating regime

        Returns:
            Number of predictions evaluated
        """
        if symbol not in self._predictions or len(prices) < lookback + self.evaluation_delay_bars:
            return 0

        prices_arr = np.array(prices, dtype=np.float64)
        evaluated_count = 0

        for record in self._predictions[symbol]:
            # Skip already evaluated
            if record.is_correct is not None:
                continue

            # Check if enough time has passed
            age_seconds = (datetime.now(timezone.utc) - record.timestamp).total_seconds()
            min_age = self.evaluation_delay_bars * 3600  # Assuming H1 timeframe

            if age_seconds < min_age:
                continue

            # Calculate ground truth regime
            actual_regime = self._calculate_regime(prices_arr, lookback)

            record.actual_regime = actual_regime
            record.actual_regime_name = self.REGIME_NAMES[actual_regime] if 0 <= actual_regime < 4 else "unknown"
            record.is_correct = (record.predicted_regime == actual_regime)
            record.evaluated_at = datetime.now(timezone.utc)

            # Update stats
            self._stats[symbol].evaluated_predictions += 1
            self._global_stats.evaluated_predictions += 1

            if record.is_correct:
                self._stats[symbol].correct_predictions += 1
                self._global_stats.correct_predictions += 1

                regime_name = record.predicted_regime_name
                self._stats[symbol].regime_correct[regime_name] = \
                    self._stats[symbol].regime_correct.get(regime_name, 0) + 1

            evaluated_count += 1

        # Update rolling accuracy
        self._update_rolling_accuracy(symbol)
        self._update_global_rolling_accuracy()

        return evaluated_count

    def _calculate_regime(self, prices: np.ndarray, lookback: int = 20) -> int:
        """
        Calculate ground truth regime from price action.

        Uses percentile-based thresholds for balanced distribution.
        """
        if len(prices) < lookback + 1:
            return 2  # Default to sideways

        # Calculate returns
        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < lookback:
            return 2

        # Use last lookback period
        recent_returns = returns[-lookback:]

        mean_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)

        # Percentile-based thresholds
        vol_threshold = np.percentile(np.abs(recent_returns), 80)
        ret_upper = np.percentile(recent_returns, 70)
        ret_lower = np.percentile(recent_returns, 30)

        # Assign regime with priority
        if volatility > vol_threshold:
            return 3  # HIGH_VOLATILITY
        elif mean_return > ret_upper:
            return 0  # BULL_TREND
        elif mean_return < ret_lower:
            return 1  # BEAR_TREND
        else:
            return 2  # SIDEWAYS

    def _update_rolling_accuracy(self, symbol: str) -> None:
        """Update rolling accuracy for a symbol."""
        if symbol not in self._predictions:
            return

        evaluated = [r for r in self._predictions[symbol] if r.is_correct is not None]
        if not evaluated:
            return

        correct = sum(1 for r in evaluated if r.is_correct)
        self._stats[symbol].rolling_accuracy = correct / len(evaluated)

    def _update_global_rolling_accuracy(self) -> None:
        """Update global rolling accuracy across all symbols."""
        total_evaluated = 0
        total_correct = 0

        for symbol_preds in self._predictions.values():
            evaluated = [r for r in symbol_preds if r.is_correct is not None]
            total_evaluated += len(evaluated)
            total_correct += sum(1 for r in evaluated if r.is_correct)

        if total_evaluated > 0:
            self._global_stats.rolling_accuracy = total_correct / total_evaluated

    def get_stats(self, symbol: Optional[str] = None) -> Dict:
        """
        Get accuracy statistics.

        Args:
            symbol: Specific symbol or None for global stats

        Returns:
            Dictionary with accuracy statistics
        """
        if symbol and symbol in self._stats:
            stats = self._stats[symbol]
        else:
            stats = self._global_stats

        return {
            "symbol": stats.symbol,
            "total_predictions": stats.total_predictions,
            "evaluated_predictions": stats.evaluated_predictions,
            "correct_predictions": stats.correct_predictions,
            "rolling_accuracy": round(stats.rolling_accuracy, 4),
            "last_updated": stats.last_updated.isoformat() if stats.last_updated else None,
            "regime_distribution": stats.regime_counts,
            "regime_accuracy": {
                regime: round(stats.regime_correct.get(regime, 0) / max(stats.regime_counts.get(regime, 1), 1), 3)
                for regime in self.REGIME_NAMES
            } if stats.regime_counts else {}
        }

    def get_all_stats(self) -> Dict:
        """Get stats for all symbols plus global."""
        return {
            "global": self.get_stats(),
            "symbols": {symbol: self.get_stats(symbol) for symbol in self._stats},
            "self_learning": {
                "enabled": self._self_learning_enabled,
                "accuracy_threshold": self._accuracy_threshold,
                "cooldown_hours": self._cooldown_hours,
                "last_retrain_trigger": self._last_retrain_trigger.isoformat() if self._last_retrain_trigger else None,
                "should_retrain": self.should_trigger_retrain()
            }
        }

    def should_trigger_retrain(self) -> bool:
        """
        Check if accuracy has dropped below threshold and retrain should be triggered.

        Returns:
            True if retrain should be triggered
        """
        if not self._self_learning_enabled:
            return False

        # Need minimum evaluations
        if self._global_stats.evaluated_predictions < 50:
            return False

        # Check accuracy threshold
        if self._global_stats.rolling_accuracy >= self._accuracy_threshold:
            return False

        # Check cooldown
        if self._last_retrain_trigger:
            hours_since_last = (datetime.now(timezone.utc) - self._last_retrain_trigger).total_seconds() / 3600
            if hours_since_last < self._cooldown_hours:
                return False

        return True

    def mark_retrain_triggered(self) -> None:
        """Mark that a retrain was triggered."""
        self._last_retrain_trigger = datetime.now(timezone.utc)
        self._save_state()
        logger.info(f"Self-learning retrain triggered at {self._last_retrain_trigger}")

    def set_config(
        self,
        enabled: Optional[bool] = None,
        threshold: Optional[float] = None,
        cooldown_hours: Optional[int] = None
    ) -> Dict:
        """
        Update self-learning configuration.

        Returns:
            Updated configuration
        """
        if enabled is not None:
            self._self_learning_enabled = enabled
        if threshold is not None:
            self._accuracy_threshold = max(0.1, min(0.9, threshold))
        if cooldown_hours is not None:
            self._cooldown_hours = max(1, min(48, cooldown_hours))

        self._save_state()

        return {
            "enabled": self._self_learning_enabled,
            "accuracy_threshold": self._accuracy_threshold,
            "cooldown_hours": self._cooldown_hours
        }

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "self_learning_enabled": self._self_learning_enabled,
            "accuracy_threshold": self._accuracy_threshold,
            "cooldown_hours": self._cooldown_hours,
            "last_retrain_trigger": self._last_retrain_trigger.isoformat() if self._last_retrain_trigger else None,
            "global_stats": {
                "total_predictions": self._global_stats.total_predictions,
                "evaluated_predictions": self._global_stats.evaluated_predictions,
                "correct_predictions": self._global_stats.correct_predictions,
                "rolling_accuracy": self._global_stats.rolling_accuracy
            }
        }

        try:
            with open(os.path.join(self.DATA_DIR, "accuracy_state.json"), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save accuracy state: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = os.path.join(self.DATA_DIR, "accuracy_state.json")

        if not os.path.exists(state_file):
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            self._self_learning_enabled = state.get("self_learning_enabled", True)
            self._accuracy_threshold = state.get("accuracy_threshold", 0.40)
            self._cooldown_hours = state.get("cooldown_hours", 6)

            if state.get("last_retrain_trigger"):
                self._last_retrain_trigger = datetime.fromisoformat(state["last_retrain_trigger"])

            global_stats = state.get("global_stats", {})
            self._global_stats.total_predictions = global_stats.get("total_predictions", 0)
            self._global_stats.evaluated_predictions = global_stats.get("evaluated_predictions", 0)
            self._global_stats.correct_predictions = global_stats.get("correct_predictions", 0)
            self._global_stats.rolling_accuracy = global_stats.get("rolling_accuracy", 0.0)

            logger.info(f"Loaded accuracy tracker state: {self._global_stats.evaluated_predictions} evaluations, "
                       f"{self._global_stats.rolling_accuracy:.1%} accuracy")
        except Exception as e:
            logger.error(f"Failed to load accuracy state: {e}")


# Global singleton
accuracy_tracker = AccuracyTracker()
