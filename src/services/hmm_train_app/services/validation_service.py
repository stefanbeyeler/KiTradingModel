"""Validation Service for HMM models.

Calculates validation metrics on held-out test data for:
- HMM regime detection accuracy
- LightGBM signal scoring performance

Supports:
- Train/validation data splitting
- Validation data caching for consistent A/B comparisons
- Ground truth generation for regime and signal targets
"""

import os
import json
import asyncio
import httpx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

# Lazy imports for ML libraries
joblib = None
sklearn_metrics = None

# Import shared LightGBM scorer for consistent feature calculation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from src.shared.lightgbm_scorer import LightGBMSignalScorer as SharedScorer
    SHARED_SCORER_AVAILABLE = True
except ImportError:
    SHARED_SCORER_AVAILABLE = False


def _load_sklearn():
    global sklearn_metrics
    if sklearn_metrics is None:
        try:
            from sklearn import metrics
            sklearn_metrics = metrics
            return True
        except ImportError:
            logger.warning("sklearn not installed")
            return False
    return True


def _load_joblib():
    global joblib
    if joblib is None:
        try:
            import joblib as jl
            joblib = jl
            return True
        except ImportError:
            logger.warning("joblib not installed")
            return False
    return True


@dataclass
class HMMValidationMetrics:
    """Validation metrics for HMM regime detection."""
    # Core accuracy metrics
    regime_accuracy: float = 0.0           # Overall correct regime predictions
    regime_precision: Dict[str, float] = field(default_factory=dict)
    regime_recall: Dict[str, float] = field(default_factory=dict)
    regime_f1: Dict[str, float] = field(default_factory=dict)
    regime_f1_weighted: float = 0.0        # Weighted average F1

    # Regime stability
    avg_regime_duration: float = 0.0       # Average candles per regime
    regime_transition_accuracy: float = 0.0

    # Confidence calibration
    confidence_calibration: float = 0.0    # How well confidence matches accuracy

    # Log-likelihood
    log_likelihood: float = 0.0            # HMM log-likelihood on validation data

    # Samples info
    validation_samples: int = 0
    validation_period: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for comparisons."""
        flat = {
            "regime_accuracy": self.regime_accuracy,
            "regime_f1_weighted": self.regime_f1_weighted,
            "avg_regime_duration": self.avg_regime_duration,
            "regime_transition_accuracy": self.regime_transition_accuracy,
            "confidence_calibration": self.confidence_calibration,
            "log_likelihood": self.log_likelihood,
        }
        # Add per-regime F1
        for regime, f1 in self.regime_f1.items():
            flat[f"f1_{regime}"] = f1
        return flat


@dataclass
class ScorerValidationMetrics:
    """Validation metrics for LightGBM signal scorer."""
    # Classification metrics (for buy/sell/hold)
    accuracy: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0

    # Per-class metrics
    class_precision: Dict[str, float] = field(default_factory=dict)
    class_recall: Dict[str, float] = field(default_factory=dict)
    class_f1: Dict[str, float] = field(default_factory=dict)

    # Regression metrics (for score 0-100)
    mae: float = 0.0                       # Mean Absolute Error
    rmse: float = 0.0                      # Root Mean Squared Error

    # Trading-relevant metrics
    profitable_signals_rate: float = 0.0   # % of signals that would have been profitable
    avg_return_per_signal: float = 0.0     # Average return for predicted signals

    # Feature importance
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    # Samples info
    validation_samples: int = 0
    symbols_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for comparisons."""
        return {
            "accuracy": self.accuracy,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "mae": self.mae,
            "rmse": self.rmse,
            "profitable_signals_rate": self.profitable_signals_rate,
            "avg_return_per_signal": self.avg_return_per_signal,
        }


# Service URLs
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
VALIDATION_DATA_DIR = os.getenv("VALIDATION_DATA_DIR", "/app/data/models/hmm/validation_data")


class ValidationService:
    """
    Service for validating trained models on held-out data.

    Key Features:
    - Consistent train/validation splits
    - Cached validation data for fair A/B comparisons
    - Ground truth generation for regimes and signals
    """

    # Market regime labels
    REGIMES = ["bull_trend", "bear_trend", "sideways", "high_volatility"]

    # Signal classes
    SIGNAL_CLASSES = {-1: "sell", 0: "hold", 1: "buy"}

    def __init__(
        self,
        data_service_url: str = DATA_SERVICE_URL,
        validation_split: float = 0.2,
        validation_dir: str = VALIDATION_DATA_DIR
    ):
        self._data_url = data_service_url
        self._validation_split = validation_split
        self._validation_dir = Path(validation_dir)
        self._validation_dir.mkdir(parents=True, exist_ok=True)
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info(f"ValidationService initialized (split: {validation_split})")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # =========================================================================
    # Data Management
    # =========================================================================

    async def fetch_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 365
    ) -> List[Dict]:
        """Fetch data from Data Service."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self._data_url}/api/v1/training-data/{symbol}",
                params={"timeframe": timeframe, "days": days, "use_cache": True}
            )
            response.raise_for_status()
            result = response.json()
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return []

    def split_data(
        self,
        data: List[Dict],
        validation_split: Optional[float] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into training and validation sets.

        Uses temporal split (last N% for validation) to prevent data leakage.
        """
        split = validation_split or self._validation_split
        split_idx = int(len(data) * (1 - split))

        train_data = data[:split_idx]
        val_data = data[split_idx:]

        return train_data, val_data

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for validation data."""
        return self._validation_dir / f"{symbol}_{timeframe}_val.npz"

    def cache_validation_data(
        self,
        symbol: str,
        timeframe: str,
        prices: np.ndarray,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None
    ):
        """
        Cache validation data for consistent A/B comparisons.

        Args:
            symbol: Symbol identifier
            timeframe: Data timeframe
            prices: Close prices array
            features: Feature matrix
            targets: Optional ground truth targets
        """
        cache_path = self._get_cache_path(symbol, timeframe)

        try:
            save_dict = {
                "prices": prices,
                "features": features,
                "cached_at": datetime.now(timezone.utc).isoformat()
            }
            if targets is not None:
                save_dict["targets"] = targets

            np.savez_compressed(cache_path, **save_dict)
            logger.debug(f"Cached validation data for {symbol}/{timeframe}")

        except Exception as e:
            logger.error(f"Failed to cache validation data: {e}")

    def load_cached_validation_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load cached validation data if available."""
        cache_path = self._get_cache_path(symbol, timeframe)

        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path, allow_pickle=True) as data:
                result = {
                    "prices": data["prices"],
                    "features": data["features"]
                }
                if "targets" in data:
                    result["targets"] = data["targets"]
                return result

        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
            return None

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Clear cached validation data."""
        if symbol and timeframe:
            cache_path = self._get_cache_path(symbol, timeframe)
            if cache_path.exists():
                cache_path.unlink()
        else:
            for f in self._validation_dir.glob("*.npz"):
                f.unlink()

    # =========================================================================
    # Ground Truth Generation
    # =========================================================================

    def _extract_prices(self, data: List[Dict]) -> np.ndarray:
        """Extract close prices from data."""
        prices = []
        for row in data:
            close = row.get("close") or row.get("h1_close") or row.get("d1_close")
            if close:
                prices.append(float(close))
        return np.array(prices, dtype=np.float64)

    def generate_regime_ground_truth(
        self,
        prices: np.ndarray,
        lookback: int = 20
    ) -> np.ndarray:
        """
        Generate ground truth regime labels based on price action.

        Uses a rule-based approach:
        - BULL_TREND: Positive returns + low volatility
        - BEAR_TREND: Negative returns + high volatility
        - SIDEWAYS: Low absolute returns + low volatility
        - HIGH_VOLATILITY: High volatility regardless of direction

        Returns:
            Array of regime indices (0-3)
        """
        n = len(prices)
        if n < lookback + 1:
            return np.zeros(n, dtype=np.int32)

        # Calculate rolling metrics
        returns = np.diff(np.log(prices + 1e-10))

        # Rolling mean return
        mean_returns = np.array([
            np.mean(returns[max(0, i-lookback):i+1])
            for i in range(len(returns))
        ])

        # Rolling volatility
        volatility = np.array([
            np.std(returns[max(0, i-lookback):i+1])
            for i in range(len(returns))
        ])

        # Normalize for threshold calculation
        vol_median = np.median(volatility[lookback:])
        vol_high = np.percentile(volatility[lookback:], 80)
        ret_threshold = 0.0005  # ~0.05% daily return threshold

        # Assign regimes
        regimes = np.zeros(len(returns), dtype=np.int32)

        for i in range(len(returns)):
            vol = volatility[i]
            ret = mean_returns[i]

            if vol > vol_high:
                regimes[i] = 3  # HIGH_VOLATILITY
            elif ret > ret_threshold and vol <= vol_median:
                regimes[i] = 0  # BULL_TREND
            elif ret < -ret_threshold and vol <= vol_high:
                regimes[i] = 1  # BEAR_TREND
            else:
                regimes[i] = 2  # SIDEWAYS

        # Pad to match prices length
        regimes = np.concatenate([[regimes[0]], regimes])

        return regimes

    def generate_signal_ground_truth(
        self,
        prices: np.ndarray,
        lookahead: int = 5,
        threshold: float = 0.01
    ) -> np.ndarray:
        """
        Generate ground truth signal labels based on future returns.

        Args:
            prices: Close prices
            lookahead: Number of periods to look ahead
            threshold: Return threshold for buy/sell (default 1%)

        Returns:
            Array of signal labels (-1=sell, 0=hold, 1=buy)
        """
        n = len(prices)
        signals = np.zeros(n, dtype=np.int32)

        for i in range(n - lookahead):
            future_return = (prices[i + lookahead] - prices[i]) / prices[i]

            if future_return > threshold:
                signals[i] = 1   # Buy
            elif future_return < -threshold:
                signals[i] = -1  # Sell
            else:
                signals[i] = 0   # Hold

        return signals

    # =========================================================================
    # HMM Validation
    # =========================================================================

    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract features for HMM (same as training)."""
        from sklearn.preprocessing import StandardScaler

        # Log returns
        returns = np.diff(np.log(prices + 1e-10))

        # Rolling volatility (20 periods)
        volatility = np.array([
            np.std(returns[max(0, i-20):i+1]) if i >= 20 else np.std(returns[:i+1])
            for i in range(len(returns))
        ])

        # Trend strength
        sma_period = 20
        sma = np.convolve(prices, np.ones(sma_period)/sma_period, mode='valid')
        sma_padded = np.concatenate([np.full(sma_period - 1, sma[0]), sma])
        trend_strength = (prices - sma_padded) / (sma_padded + 1e-8)

        # Combine features
        features = np.column_stack([
            returns,
            volatility,
            trend_strength[1:]
        ])

        # Clean and standardize
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features

    async def validate_hmm_model(
        self,
        model_path: str,
        symbol: str,
        timeframe: str,
        use_cached: bool = True
    ) -> HMMValidationMetrics:
        """
        Validate an HMM model on held-out data.

        Args:
            model_path: Path to trained model file
            symbol: Symbol to validate on
            timeframe: Data timeframe
            use_cached: Use cached validation data if available

        Returns:
            HMMValidationMetrics with calculated metrics
        """
        if not _load_joblib() or not _load_sklearn():
            return HMMValidationMetrics(validation_samples=0)

        metrics = HMMValidationMetrics()

        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data.get("model")
            regime_mapping = model_data.get("regime_mapping", {})

            if model is None:
                logger.error("Model not found in file")
                return metrics

            # Get validation data
            cached = None
            if use_cached:
                cached = self.load_cached_validation_data(symbol, timeframe)

            if cached is not None:
                prices = cached["prices"]
                features = cached["features"]
                ground_truth = cached.get("targets")
            else:
                # Fetch fresh data
                data = await self.fetch_data(symbol, timeframe)
                if len(data) < 100:
                    logger.warning(f"Insufficient data for validation: {len(data)}")
                    return metrics

                # Split data - use validation portion
                _, val_data = self.split_data(data)
                prices = self._extract_prices(val_data)

                if len(prices) < 50:
                    logger.warning(f"Insufficient validation samples: {len(prices)}")
                    return metrics

                features = self._extract_features(prices)
                ground_truth = self.generate_regime_ground_truth(prices)

                # Cache for future A/B comparisons
                self.cache_validation_data(symbol, timeframe, prices, features, ground_truth)

            if ground_truth is None:
                ground_truth = self.generate_regime_ground_truth(prices)

            # Ensure lengths match
            min_len = min(len(features), len(ground_truth) - 1)
            features = features[:min_len]
            ground_truth = ground_truth[1:min_len + 1]

            metrics.validation_samples = len(features)

            # HMM predictions
            predicted_states = model.predict(features)

            # Map states to regime indices
            state_to_regime = {}
            for state, regime_name in regime_mapping.items():
                regime_idx = self.REGIMES.index(regime_name) if regime_name in self.REGIMES else 2
                state_to_regime[int(state)] = regime_idx

            predicted_regimes = np.array([
                state_to_regime.get(s, 2) for s in predicted_states
            ])

            # Calculate metrics
            metrics.regime_accuracy = sklearn_metrics.accuracy_score(ground_truth, predicted_regimes)

            # Per-regime metrics
            precision, recall, f1, support = sklearn_metrics.precision_recall_fscore_support(
                ground_truth, predicted_regimes, labels=[0, 1, 2, 3], zero_division=0
            )

            for i, regime in enumerate(self.REGIMES):
                metrics.regime_precision[regime] = float(precision[i])
                metrics.regime_recall[regime] = float(recall[i])
                metrics.regime_f1[regime] = float(f1[i])

            # Weighted F1
            metrics.regime_f1_weighted = sklearn_metrics.f1_score(
                ground_truth, predicted_regimes, average='weighted', zero_division=0
            )

            # Log-likelihood
            try:
                metrics.log_likelihood = float(model.score(features))
            except:
                metrics.log_likelihood = 0.0

            # Regime duration (average consecutive same regime)
            durations = []
            current_duration = 1
            for i in range(1, len(predicted_regimes)):
                if predicted_regimes[i] == predicted_regimes[i-1]:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_duration = 1
            durations.append(current_duration)
            metrics.avg_regime_duration = np.mean(durations) if durations else 0.0

            # Transition accuracy (did transitions happen at correct times?)
            gt_transitions = np.diff(ground_truth) != 0
            pred_transitions = np.diff(predicted_regimes) != 0

            if np.sum(gt_transitions) > 0:
                # Check if predicted transitions align with actual transitions (within 2 periods)
                correct_transitions = 0
                for i, gt_trans in enumerate(gt_transitions):
                    if gt_trans:
                        # Check if prediction had transition within ±2 periods
                        start = max(0, i - 2)
                        end = min(len(pred_transitions), i + 3)
                        if np.any(pred_transitions[start:end]):
                            correct_transitions += 1

                metrics.regime_transition_accuracy = correct_transitions / np.sum(gt_transitions)

            # Confidence calibration (using state probabilities)
            try:
                posteriors = model.predict_proba(features)
                max_probs = np.max(posteriors, axis=1)

                # Group by confidence bins and check accuracy
                bins = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
                calibration_errors = []

                for low, high in bins:
                    mask = (max_probs >= low) & (max_probs < high)
                    if np.sum(mask) > 10:
                        bin_acc = np.mean(predicted_regimes[mask] == ground_truth[mask])
                        expected_acc = (low + high) / 2
                        calibration_errors.append(abs(bin_acc - expected_acc))

                metrics.confidence_calibration = 1 - np.mean(calibration_errors) if calibration_errors else 0.5

            except Exception as e:
                logger.debug(f"Could not calculate confidence calibration: {e}")
                metrics.confidence_calibration = 0.5

            logger.info(
                f"HMM validation for {symbol}: accuracy={metrics.regime_accuracy:.3f}, "
                f"f1_weighted={metrics.regime_f1_weighted:.3f}"
            )

        except Exception as e:
            logger.error(f"HMM validation failed: {e}")
            import traceback
            traceback.print_exc()

        return metrics

    # =========================================================================
    # Scorer Validation
    # =========================================================================

    async def validate_scorer_model(
        self,
        model_path: str,
        symbols: List[str],
        timeframe: str = "1h"
    ) -> ScorerValidationMetrics:
        """
        Validate a LightGBM scorer model.

        Args:
            model_path: Path to trained scorer model
            symbols: Symbols to validate on
            timeframe: Data timeframe

        Returns:
            ScorerValidationMetrics with calculated metrics
        """
        if not _load_joblib() or not _load_sklearn():
            return ScorerValidationMetrics(validation_samples=0)

        metrics = ScorerValidationMetrics()

        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data.get("model")

            if model is None:
                logger.error("Scorer model not found in file")
                return metrics

            # Collect validation data from multiple symbols
            X_val = []
            y_val = []
            returns_for_signals = []

            # Use shared scorer for consistent 18-feature calculation
            temp_scorer = None
            if SHARED_SCORER_AVAILABLE:
                temp_scorer = SharedScorer()

            for symbol in symbols:  # Use ALL symbols for validation
                data = await self.fetch_data(symbol, timeframe)
                if len(data) < 100:
                    continue

                # Use validation split
                _, val_data = self.split_data(data)

                # Build OHLCV array for feature calculation
                ohlcv_list = []
                for row in val_data:
                    open_p = float(row.get("open") or row.get("h1_open") or 0)
                    high_p = float(row.get("high") or row.get("h1_high") or 0)
                    low_p = float(row.get("low") or row.get("h1_low") or 0)
                    close_p = float(row.get("close") or row.get("h1_close") or 0)
                    volume = float(row.get("volume") or row.get("h1_volume") or 0)
                    ohlcv_list.append([open_p, high_p, low_p, close_p, volume])

                if len(ohlcv_list) < 100:
                    continue

                ohlcv = np.array(ohlcv_list, dtype=np.float64)

                for i in range(100, len(ohlcv) - 5):
                    window = ohlcv[i-100:i]
                    close_now = ohlcv[i, 3]
                    close_future = ohlcv[i + 5, 3]

                    if close_now <= 0 or close_future <= 0:
                        continue

                    # Calculate features using shared scorer (18 features)
                    if temp_scorer is not None:
                        # Estimate regime probs from recent price action
                        recent_returns = np.diff(np.log(window[-20:, 3] + 1e-10))
                        mean_ret = np.mean(recent_returns)
                        volatility = np.std(recent_returns)
                        vol_threshold = np.percentile(np.abs(recent_returns), 80)

                        if volatility > vol_threshold:
                            regime_probs = {"bull_trend": 0.1, "bear_trend": 0.1, "sideways": 0.1, "high_volatility": 0.7}
                        elif mean_ret > 0.001:
                            regime_probs = {"bull_trend": 0.6, "bear_trend": 0.1, "sideways": 0.2, "high_volatility": 0.1}
                        elif mean_ret < -0.001:
                            regime_probs = {"bull_trend": 0.1, "bear_trend": 0.6, "sideways": 0.2, "high_volatility": 0.1}
                        else:
                            regime_probs = {"bull_trend": 0.2, "bear_trend": 0.2, "sideways": 0.5, "high_volatility": 0.1}

                        features = temp_scorer.calculate_features(window, regime_probs, 10)
                    else:
                        # Fallback to simple 5 features if shared scorer not available
                        row = val_data[i]
                        features = np.array([
                            row.get("rsi", 50) or 50,
                            row.get("macd_main", 0) or 0,
                            row.get("adx_main", 25) or 25,
                            row.get("atr_pct_d1", 1) or 1,
                            row.get("strength_1d", 0) or 0,
                        ])

                    # Calculate target
                    future_return = (close_future - close_now) / close_now

                    if future_return > 0.01:
                        y_val.append(1)
                    elif future_return < -0.01:
                        y_val.append(-1)
                    else:
                        y_val.append(0)

                    X_val.append(features)
                    returns_for_signals.append(future_return)

            if len(X_val) < 50:
                logger.warning(f"Insufficient scorer validation samples: {len(X_val)}")
                return metrics

            X = np.array(X_val)
            y = np.array(y_val)
            returns_arr = np.array(returns_for_signals)

            metrics.validation_samples = len(X)
            metrics.symbols_used = symbols  # ALL symbols used

            # Get predictions - model returns regression scores (0-100)
            y_pred_scores = model.predict(X)

            # Handle both 1D and 2D outputs
            if y_pred_scores.ndim > 1:
                y_pred_scores = y_pred_scores.flatten()

            # Convert scores to signals:
            # Score > 60 -> buy (1)
            # Score < 40 -> sell (-1)
            # 40-60 -> hold (0)
            y_pred = np.zeros_like(y_pred_scores, dtype=np.int32)
            y_pred[y_pred_scores > 60] = 1
            y_pred[y_pred_scores < 40] = -1

            # Classification metrics
            metrics.accuracy = sklearn_metrics.accuracy_score(y, y_pred)
            metrics.precision_weighted = sklearn_metrics.precision_score(
                y, y_pred, average='weighted', zero_division=0
            )
            metrics.recall_weighted = sklearn_metrics.recall_score(
                y, y_pred, average='weighted', zero_division=0
            )
            metrics.f1_weighted = sklearn_metrics.f1_score(
                y, y_pred, average='weighted', zero_division=0
            )

            # Per-class metrics
            for cls, name in self.SIGNAL_CLASSES.items():
                mask = y == cls
                if np.sum(mask) > 0:
                    cls_pred = (y_pred == cls)
                    metrics.class_precision[name] = sklearn_metrics.precision_score(
                        mask, cls_pred, zero_division=0
                    )
                    metrics.class_recall[name] = sklearn_metrics.recall_score(
                        mask, cls_pred, zero_division=0
                    )
                    metrics.class_f1[name] = sklearn_metrics.f1_score(
                        mask, cls_pred, zero_division=0
                    )

            # Trading metrics
            # Calculate profitability of predicted signals
            buy_signals = y_pred == 1
            sell_signals = y_pred == -1

            # For buy signals, positive return is profitable
            buy_profitable = np.sum((buy_signals) & (returns_arr > 0))
            # For sell signals, negative return is profitable
            sell_profitable = np.sum((sell_signals) & (returns_arr < 0))

            total_signals = np.sum(buy_signals) + np.sum(sell_signals)
            if total_signals > 0:
                metrics.profitable_signals_rate = (buy_profitable + sell_profitable) / total_signals

                # Average return per signal
                buy_returns = returns_arr[buy_signals].sum() if np.any(buy_signals) else 0
                sell_returns = -returns_arr[sell_signals].sum() if np.any(sell_signals) else 0
                metrics.avg_return_per_signal = (buy_returns + sell_returns) / total_signals

            # MAE/RMSE for score prediction (convert y to 0-100 scale)
            y_scaled = (y + 1) * 33.33  # -1->0, 0->33, 1->66
            y_pred_scaled = y_pred_scores  # Already in 0-100 range

            metrics.mae = sklearn_metrics.mean_absolute_error(y_scaled, y_pred_scaled)
            metrics.rmse = np.sqrt(sklearn_metrics.mean_squared_error(y_scaled, y_pred_scaled))

            # Feature importance
            try:
                # 18 features from shared scorer
                feature_names = [
                    "bull_trend_prob", "bear_trend_prob", "sideways_prob", "high_vol_prob",
                    "regime_duration",
                    "rsi", "rsi_slope", "macd_hist", "macd_slope", "adx",
                    "volatility_ratio", "atr_pct",
                    "close_vs_sma20", "close_vs_sma50", "bb_position",
                    "volume_ratio", "price_momentum", "recent_high_dist"
                ]
                importance = model.feature_importance()
                # Handle case where importance length differs
                names_to_use = feature_names[:len(importance)] if len(importance) <= len(feature_names) else feature_names + [f"feature_{i}" for i in range(len(feature_names), len(importance))]
                metrics.top_features = sorted(
                    zip(names_to_use, importance.tolist()),
                    key=lambda x: x[1],
                    reverse=True
                )
            except:
                pass

            logger.info(
                f"Scorer validation: accuracy={metrics.accuracy:.3f}, "
                f"profitable_rate={metrics.profitable_signals_rate:.3f}"
            )

        except Exception as e:
            logger.error(f"Scorer validation failed: {e}")
            import traceback
            traceback.print_exc()

        return metrics

    async def prepare_validation_data(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and cache validation data for a symbol.

        Returns:
            Tuple of (prices, features, ground_truth_regimes)
        """
        data = await self.fetch_data(symbol, timeframe, days)

        if len(data) < 100:
            return np.array([]), np.array([]), np.array([])

        # Split and use validation portion
        _, val_data = self.split_data(data)
        prices = self._extract_prices(val_data)

        if len(prices) < 50:
            return np.array([]), np.array([]), np.array([])

        features = self._extract_features(prices)
        ground_truth = self.generate_regime_ground_truth(prices)

        # Cache for A/B comparisons
        self.cache_validation_data(symbol, timeframe, prices, features, ground_truth)

        return prices, features, ground_truth


# Global singleton
validation_service = ValidationService()
