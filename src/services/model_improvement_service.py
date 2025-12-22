"""
Automatic Model Improvement Service for NHITS forecasting.

This service implements:
1. Automatic feedback collection (predictions vs actual prices)
2. Performance tracking and metrics
3. Adaptive retraining based on prediction errors
4. Hyperparameter optimization
5. Direction accuracy loss for better trend prediction
"""

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn

from src.config.settings import settings


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to JSON-safe float (handles NaN, Infinity)."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_round(value: Any, decimals: int = 4, default: float = 0.0) -> float:
    """Round value safely, returning default for NaN/Infinity."""
    safe_val = _safe_float(value, default)
    return round(safe_val, decimals)

logger = logging.getLogger(__name__)


@dataclass
class PredictionFeedback:
    """Stores a single prediction with its outcome."""
    symbol: str
    timestamp: datetime
    horizon: str  # "1h", "4h", "24h"
    current_price: float
    predicted_price: float
    actual_price: Optional[float] = None
    prediction_error_pct: Optional[float] = None
    direction_correct: Optional[bool] = None
    evaluated_at: Optional[datetime] = None


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a symbol's model."""
    symbol: str
    total_predictions: int = 0
    evaluated_predictions: int = 0
    avg_error_pct: float = 0.0
    direction_accuracy: float = 0.0
    last_updated: Optional[datetime] = None

    # By horizon
    metrics_1h: Dict = None
    metrics_4h: Dict = None
    metrics_24h: Dict = None

    # Improvement indicators
    needs_retraining: bool = False
    retraining_reason: Optional[str] = None

    def __post_init__(self):
        if self.metrics_1h is None:
            self.metrics_1h = {"count": 0, "avg_error": 0.0, "direction_acc": 0.0}
        if self.metrics_4h is None:
            self.metrics_4h = {"count": 0, "avg_error": 0.0, "direction_acc": 0.0}
        if self.metrics_24h is None:
            self.metrics_24h = {"count": 0, "avg_error": 0.0, "direction_acc": 0.0}


class DirectionAwareLoss(nn.Module):
    """
    Custom loss function that combines:
    1. Quantile loss for price prediction accuracy
    2. Direction classification loss for trend prediction

    This helps the model learn both accurate price levels AND correct direction.
    """

    def __init__(self, direction_weight: float = 0.3, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.direction_weight = direction_weight
        self.quantiles = torch.tensor(quantiles)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_last: torch.Tensor  # Last input price for direction calculation
    ) -> torch.Tensor:
        """
        Args:
            pred: (batch, horizon, n_quantiles) - predicted quantiles
            target: (batch, horizon) - actual future prices
            input_last: (batch,) - last price in input sequence
        """
        device = pred.device
        quantiles = self.quantiles.to(device)

        # 1. Standard quantile loss
        target_expanded = target.unsqueeze(-1)  # (batch, horizon, 1)
        errors = target_expanded - pred  # (batch, horizon, n_quantiles)
        quantile_loss = torch.max(
            (quantiles - 1) * errors,
            quantiles * errors
        ).mean()

        # 2. Direction loss
        # Get median prediction (index 1 is 50th percentile)
        pred_median = pred[:, :, 1]  # (batch, horizon)

        # Calculate predicted direction for each horizon step
        input_last_expanded = input_last.unsqueeze(1)  # (batch, 1)
        pred_direction = (pred_median > input_last_expanded).float()  # 1 if up, 0 if down
        actual_direction = (target > input_last_expanded).float()

        # Binary cross entropy for direction
        direction_loss = self.bce_loss(
            pred_median - input_last_expanded,  # Use as logits
            actual_direction
        )

        # Combine losses
        total_loss = (1 - self.direction_weight) * quantile_loss + self.direction_weight * direction_loss

        return total_loss


class ModelImprovementService:
    """
    Service for automatic model improvement through feedback learning.

    Features:
    - Collects prediction feedback automatically
    - Tracks model performance metrics
    - Triggers retraining when performance degrades
    - Implements adaptive learning rate and hyperparameters
    - Automatic background evaluation of pending predictions
    - Auto-retrain when performance thresholds are exceeded
    """

    def __init__(self):
        self.feedback_path = Path("data/model_feedback")
        self.feedback_path.mkdir(parents=True, exist_ok=True)

        self.metrics_path = Path("data/model_metrics")
        self.metrics_path.mkdir(parents=True, exist_ok=True)

        self.pending_feedback: Dict[str, List[PredictionFeedback]] = {}
        self.evaluated_feedback: Dict[str, List[PredictionFeedback]] = {}  # Store evaluated predictions
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}

        # Thresholds for retraining
        self.error_threshold_pct = 0.5  # Retrain if avg error > 0.5%
        self.direction_threshold = 0.45  # Retrain if direction accuracy < 45%
        self.min_evaluations = 10  # Minimum evaluations before checking

        self._running = False
        self._evaluation_task: Optional[asyncio.Task] = None

        # Auto-evaluation settings
        self._auto_evaluation_enabled = True
        self._auto_evaluation_interval_seconds = 300  # 5 minutes
        self._last_evaluation_time: Optional[datetime] = None
        self._evaluation_in_progress = False

        # Auto-retrain settings
        self._auto_retrain_enabled = True
        self._retrain_in_progress = False
        self._last_retrain_time: Optional[datetime] = None
        self._retrain_cooldown_hours = 4  # Minimum hours between retrains per symbol
        self._retrain_history: Dict[str, datetime] = {}  # symbol -> last retrain time

        # Statistics
        self._total_auto_evaluations = 0
        self._total_auto_retrains = 0
        self._last_auto_retrain_symbols: List[str] = []

        # Load existing data
        self._load_data()

        logger.info("ModelImprovementService initialized with auto-evaluation and auto-retrain")

    def _load_data(self):
        """Load existing feedback and metrics from disk."""
        # Load pending feedback
        feedback_file = self.feedback_path / "pending_feedback.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    for symbol, feedbacks in data.items():
                        parsed_feedbacks = []
                        for fb in feedbacks:
                            # Convert timestamp strings back to datetime objects
                            if isinstance(fb.get("timestamp"), str):
                                fb["timestamp"] = datetime.fromisoformat(fb["timestamp"])
                            if isinstance(fb.get("evaluated_at"), str):
                                fb["evaluated_at"] = datetime.fromisoformat(fb["evaluated_at"])
                            parsed_feedbacks.append(PredictionFeedback(**fb))
                        self.pending_feedback[symbol] = parsed_feedbacks
                logger.info(f"Loaded {sum(len(v) for v in self.pending_feedback.values())} pending feedbacks")
            except Exception as e:
                logger.warning(f"Failed to load pending feedback: {e}")

        # Load evaluated feedback
        evaluated_file = self.feedback_path / "evaluated_feedback.json"
        if evaluated_file.exists():
            try:
                with open(evaluated_file, "r") as f:
                    data = json.load(f)
                    for symbol, feedbacks in data.items():
                        parsed_feedbacks = []
                        for fb in feedbacks:
                            if isinstance(fb.get("timestamp"), str):
                                fb["timestamp"] = datetime.fromisoformat(fb["timestamp"])
                            if isinstance(fb.get("evaluated_at"), str):
                                fb["evaluated_at"] = datetime.fromisoformat(fb["evaluated_at"])
                            parsed_feedbacks.append(PredictionFeedback(**fb))
                        self.evaluated_feedback[symbol] = parsed_feedbacks
                logger.info(f"Loaded {sum(len(v) for v in self.evaluated_feedback.values())} evaluated feedbacks")
            except Exception as e:
                logger.warning(f"Failed to load evaluated feedback: {e}")

        # Load metrics
        metrics_file = self.metrics_path / "performance_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    for symbol, metrics in data.items():
                        self.performance_metrics[symbol] = ModelPerformanceMetrics(**metrics)
                logger.info(f"Loaded metrics for {len(self.performance_metrics)} symbols")
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")

    def _save_data(self):
        """Save feedback and metrics to disk."""
        # Save pending feedback
        feedback_file = self.feedback_path / "pending_feedback.json"
        try:
            data = {}
            for symbol, feedbacks in self.pending_feedback.items():
                data[symbol] = [
                    {k: v.isoformat() if isinstance(v, datetime) else v
                     for k, v in asdict(fb).items()}
                    for fb in feedbacks
                ]
            with open(feedback_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save pending feedback: {e}")

        # Save evaluated feedback
        evaluated_file = self.feedback_path / "evaluated_feedback.json"
        try:
            data = {}
            for symbol, feedbacks in self.evaluated_feedback.items():
                data[symbol] = [
                    {k: v.isoformat() if isinstance(v, datetime) else v
                     for k, v in asdict(fb).items()}
                    for fb in feedbacks
                ]
            with open(evaluated_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save evaluated feedback: {e}")

        # Save metrics
        metrics_file = self.metrics_path / "performance_metrics.json"
        try:
            data = {}
            for symbol, metrics in self.performance_metrics.items():
                d = asdict(metrics)
                if d.get("last_updated"):
                    d["last_updated"] = d["last_updated"].isoformat() if isinstance(d["last_updated"], datetime) else d["last_updated"]
                data[symbol] = d
            with open(metrics_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    def record_prediction(
        self,
        symbol: str,
        current_price: float,
        predicted_price_1h: Optional[float] = None,
        predicted_price_4h: Optional[float] = None,
        predicted_price_24h: Optional[float] = None,
        timeframe: str = "H1",
    ):
        """
        Record a new prediction for later evaluation.

        Called automatically when NHITS makes a prediction.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            current_price: Current price at prediction time
            predicted_price_1h: Short-term prediction (1h for H1, 30min for M15, 1d for D1)
            predicted_price_4h: Medium-term prediction (4h for H1, 1h for M15, 3d for D1)
            predicted_price_24h: Long-term prediction (24h for H1, 2h for M15, 7d for D1)
            timeframe: M15, H1, or D1
        """
        now = datetime.utcnow()

        if symbol not in self.pending_feedback:
            self.pending_feedback[symbol] = []

        # Map timeframe to actual horizons
        if timeframe.upper() == "M15":
            # M15: short=30min, mid=1h, long=2h
            horizons = [("30m", predicted_price_1h), ("1h", predicted_price_4h), ("2h", predicted_price_24h)]
        elif timeframe.upper() == "D1":
            # D1: short=1d, mid=3d, long=7d
            horizons = [("1d", predicted_price_1h), ("3d", predicted_price_4h), ("7d", predicted_price_24h)]
        else:
            # H1 (default): short=1h, mid=4h, long=24h
            horizons = [("1h", predicted_price_1h), ("4h", predicted_price_4h), ("24h", predicted_price_24h)]

        recorded_count = 0
        for horizon, pred_price in horizons:
            if pred_price is not None:
                feedback = PredictionFeedback(
                    symbol=symbol,
                    timestamp=now,
                    horizon=horizon,
                    current_price=current_price,
                    predicted_price=pred_price,
                )
                self.pending_feedback[symbol].append(feedback)
                recorded_count += 1

        # Update metrics count
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = ModelPerformanceMetrics(symbol=symbol)
        self.performance_metrics[symbol].total_predictions += 1

        # Always save immediately to ensure persistence
        self._save_data()

        logger.info(
            f"Recorded {recorded_count} predictions for {symbol}/{timeframe}: "
            f"short={predicted_price_1h}, mid={predicted_price_4h}, long={predicted_price_24h}"
        )

    def _parse_horizon_to_target_time(self, timestamp: datetime, horizon: str) -> datetime:
        """
        Parse horizon string to target time.

        Supports formats:
        - Minutes: 30m, 15m
        - Hours: 1h, 2h, 4h, 24h
        - Days: 1d, 3d, 7d
        """
        horizon = horizon.lower().strip()

        if horizon.endswith('m'):
            # Minutes (e.g., 30m, 15m)
            minutes = int(horizon[:-1])
            return timestamp + timedelta(minutes=minutes)
        elif horizon.endswith('d'):
            # Days (e.g., 1d, 3d, 7d)
            days = int(horizon[:-1])
            return timestamp + timedelta(days=days)
        elif horizon.endswith('h'):
            # Hours (e.g., 1h, 4h, 24h)
            hours = int(horizon[:-1])
            return timestamp + timedelta(hours=hours)
        else:
            # Fallback: assume hours
            try:
                hours = int(horizon)
                return timestamp + timedelta(hours=hours)
            except ValueError:
                logger.warning(f"Unknown horizon format: {horizon}, defaulting to 1h")
                return timestamp + timedelta(hours=1)

    async def evaluate_pending_predictions(self, db_pool) -> Dict[str, int]:
        """
        Evaluate pending predictions against actual prices.

        Args:
            db_pool: AsyncPG connection pool

        Returns:
            Dict mapping symbol to number of evaluated predictions
        """
        evaluated = {}
        now = datetime.utcnow()

        for symbol, feedbacks in list(self.pending_feedback.items()):
            symbol_evaluated = 0
            remaining = []

            for fb in feedbacks:
                # Check if enough time has passed
                target_time = self._parse_horizon_to_target_time(fb.timestamp, fb.horizon)

                if now < target_time + timedelta(minutes=30):
                    # Not yet time to evaluate
                    remaining.append(fb)
                    continue

                # Try to get actual price
                try:
                    actual_price = await self._get_actual_price(
                        db_pool, symbol, target_time
                    )

                    if actual_price is not None:
                        # Sanity check: skip implausible predictions
                        if not self._is_prediction_plausible(fb, symbol):
                            continue  # Discard broken prediction

                        # Calculate error
                        fb.actual_price = actual_price
                        fb.prediction_error_pct = abs(fb.predicted_price - actual_price) / actual_price * 100
                        fb.direction_correct = (
                            (fb.predicted_price > fb.current_price) ==
                            (actual_price > fb.current_price)
                        )
                        fb.evaluated_at = now

                        # Store evaluated feedback for display
                        if symbol not in self.evaluated_feedback:
                            self.evaluated_feedback[symbol] = []
                        self.evaluated_feedback[symbol].append(fb)

                        # Keep only last 100 evaluated feedbacks per symbol
                        if len(self.evaluated_feedback[symbol]) > 100:
                            self.evaluated_feedback[symbol] = self.evaluated_feedback[symbol][-100:]

                        # Update metrics
                        self._update_metrics(symbol, fb)
                        symbol_evaluated += 1

                        logger.info(
                            f"Evaluated {symbol} {fb.horizon}: "
                            f"error={fb.prediction_error_pct:.3f}%, "
                            f"direction={'correct' if fb.direction_correct else 'wrong'}"
                        )
                    else:
                        # No data yet, keep for later
                        remaining.append(fb)

                except Exception as e:
                    logger.warning(f"Failed to evaluate {symbol} prediction: {e}")
                    remaining.append(fb)

            self.pending_feedback[symbol] = remaining
            if symbol_evaluated > 0:
                evaluated[symbol] = symbol_evaluated

        if evaluated:
            self._save_data()
            logger.info(f"Evaluated predictions: {evaluated}")

        return evaluated

    async def _get_actual_price(
        self,
        db_pool,
        symbol: str,
        target_time: datetime
    ) -> Optional[float]:
        """Get actual price from database at target time.

        Uses a tiered approach to find the closest price:
        1. First try ±30 minutes window (ideal for regular trading hours)
        2. If not found, extend to ±48 hours (handles weekends/holidays)
        """
        async with db_pool.acquire() as conn:
            # First try narrow window (±30 minutes)
            time_start = target_time - timedelta(minutes=30)
            time_end = target_time + timedelta(minutes=30)

            query = """
                SELECT d1_close as close
                FROM symbol
                WHERE symbol = $1
                  AND data_timestamp >= $2
                  AND data_timestamp <= $3
                ORDER BY ABS(EXTRACT(EPOCH FROM (data_timestamp - $4)))
                LIMIT 1
            """

            row = await conn.fetchrow(query, symbol, time_start, time_end, target_time)
            if row:
                return float(row["close"])

            # If not found, try extended window for weekends/holidays (±48 hours)
            # But only look AFTER the target time (we want the next available price)
            time_end_extended = target_time + timedelta(hours=48)

            query_extended = """
                SELECT d1_close as close
                FROM symbol
                WHERE symbol = $1
                  AND data_timestamp >= $2
                  AND data_timestamp <= $3
                ORDER BY data_timestamp ASC
                LIMIT 1
            """

            row = await conn.fetchrow(query_extended, symbol, target_time, time_end_extended)
            if row:
                logger.debug(f"Used extended window to find price for {symbol} at {target_time}")
                return float(row["close"])

        return None

    async def _get_actual_price_via_api(
        self,
        symbol: str,
        target_time: datetime
    ) -> Optional[float]:
        """Get actual price via EasyInsight/Data Service API.

        Uses the Data Service as gateway to fetch historical price data.
        """
        import httpx
        from src.config import settings
        from datetime import timezone as tz

        data_service_url = getattr(settings, 'data_service_url', 'http://localhost:3001')

        # Make target_time timezone-aware (UTC) if needed
        target_aware = target_time
        if target_time.tzinfo is None:
            target_aware = target_time.replace(tzinfo=tz.utc)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use training-data endpoint which returns EasyInsight data
                # Use 3 days to ensure we have historical data for older predictions
                response = await client.get(
                    f"{data_service_url}/api/v1/training-data/{symbol}",
                    params={
                        "days_back": 3,
                        "interval": "H1"
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    data = result.get('data', [])

                    if data and len(data) > 0:
                        # Find the closest price to target_time
                        closest_price = None
                        min_diff = float('inf')

                        for candle in data:
                            # Parse snapshot_time from EasyInsight data
                            snapshot_time_str = candle.get('snapshot_time', '')
                            if not snapshot_time_str:
                                continue

                            try:
                                candle_time = datetime.fromisoformat(snapshot_time_str.replace('Z', '+00:00'))
                            except ValueError:
                                continue

                            diff = abs((candle_time - target_aware).total_seconds())
                            if diff < min_diff:
                                min_diff = diff
                                # Use h1_close for H1 timeframe
                                closest_price = float(candle.get('h1_close', 0))

                        if closest_price and closest_price > 0 and min_diff < 14400:  # Within 4 hours
                            return closest_price
                        elif min_diff >= 14400:
                            logger.debug(f"No close price for {symbol} at {target_aware}: min_diff={min_diff/3600:.1f}h")

                # Fallback: Try live-data endpoint for recent targets
                now_utc = datetime.now(tz.utc)
                if (now_utc - target_aware).total_seconds() < 300:  # Within 5 minutes
                    response = await client.get(
                        f"{data_service_url}/api/v1/managed-symbols/live-data/{symbol}"
                    )
                    if response.status_code == 200:
                        data = response.json()
                        price = data.get('h1_close') or data.get('bid')
                        return float(price) if price else None

        except Exception as e:
            logger.warning(f"Failed to get price via API for {symbol}: {e}")

        return None

    async def evaluate_pending_predictions_via_api(self) -> Dict[str, int]:
        """
        Evaluate pending predictions using EasyInsight/Data Service API.

        This method does not require database access and uses the Data Service
        as a gateway to fetch historical prices.

        Returns:
            Dict mapping symbol to number of evaluated predictions
        """
        evaluated = {}
        now = datetime.utcnow()

        for symbol, feedbacks in list(self.pending_feedback.items()):
            symbol_evaluated = 0
            remaining = []

            for fb in feedbacks:
                # Check if enough time has passed
                target_time = self._parse_horizon_to_target_time(fb.timestamp, fb.horizon)

                if now < target_time + timedelta(minutes=30):
                    # Not yet time to evaluate
                    remaining.append(fb)
                    continue

                # Try to get actual price via API
                try:
                    actual_price = await self._get_actual_price_via_api(symbol, target_time)

                    if actual_price is not None:
                        # Sanity check: skip implausible predictions
                        if not self._is_prediction_plausible(fb, symbol):
                            continue  # Discard broken prediction

                        # Calculate error
                        fb.actual_price = actual_price
                        fb.prediction_error_pct = abs(fb.predicted_price - actual_price) / actual_price * 100
                        fb.direction_correct = (
                            (fb.predicted_price > fb.current_price) ==
                            (actual_price > fb.current_price)
                        )
                        fb.evaluated_at = now

                        # Store evaluated feedback for display
                        if symbol not in self.evaluated_feedback:
                            self.evaluated_feedback[symbol] = []
                        self.evaluated_feedback[symbol].append(fb)

                        # Keep only last 100 evaluated feedbacks per symbol
                        if len(self.evaluated_feedback[symbol]) > 100:
                            self.evaluated_feedback[symbol] = self.evaluated_feedback[symbol][-100:]

                        # Update metrics
                        self._update_metrics(symbol, fb)
                        symbol_evaluated += 1

                        logger.info(
                            f"Evaluated {symbol} {fb.horizon}: "
                            f"error={fb.prediction_error_pct:.3f}%, "
                            f"direction={'correct' if fb.direction_correct else 'wrong'}"
                        )
                    else:
                        # No data yet, keep for later
                        remaining.append(fb)

                except Exception as e:
                    logger.warning(f"Failed to evaluate {symbol} prediction: {e}")
                    remaining.append(fb)

            self.pending_feedback[symbol] = remaining
            if symbol_evaluated > 0:
                evaluated[symbol] = symbol_evaluated

        if evaluated:
            self._save_data()
            logger.info(f"Evaluated predictions via API: {evaluated}")

        return evaluated

    def _is_prediction_plausible(self, fb: PredictionFeedback, symbol: str) -> bool:
        """
        Check if a prediction is plausible.

        Returns False for clearly broken predictions (negative prices,
        extreme deviations from current price, etc.)
        """
        # Check for invalid prices
        if fb.predicted_price <= 0 or fb.current_price <= 0:
            logger.warning(
                f"Discarding {symbol} {fb.horizon}: invalid price "
                f"(predicted={fb.predicted_price:.2f}, current={fb.current_price:.2f})"
            )
            return False

        # Check for extreme deviation (>50% from current price)
        # Normal price movements in crypto/forex are typically <10% per day
        deviation_pct = abs(fb.predicted_price - fb.current_price) / fb.current_price * 100
        if deviation_pct > 50:
            logger.warning(
                f"Discarding {symbol} {fb.horizon}: implausible prediction "
                f"(predicted={fb.predicted_price:.2f}, current={fb.current_price:.2f}, "
                f"deviation={deviation_pct:.1f}%)"
            )
            return False

        return True

    def _update_metrics(self, symbol: str, feedback: PredictionFeedback):
        """Update performance metrics with new feedback."""
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = ModelPerformanceMetrics(symbol=symbol)

        metrics = self.performance_metrics[symbol]
        metrics.evaluated_predictions += 1
        metrics.last_updated = datetime.utcnow()

        # Map horizon to available metrics buckets (1h, 4h, 24h)
        horizon = feedback.horizon
        horizon_map = {
            "15m": "1h", "30m": "1h", "1h": "1h", "2h": "1h",  # Short-term → 1h bucket
            "4h": "4h", "6h": "4h", "8h": "4h", "12h": "4h",   # Medium-term → 4h bucket
            "24h": "24h", "1d": "24h", "48h": "24h", "1w": "24h"  # Long-term → 24h bucket
        }
        metrics_key = f"metrics_{horizon_map.get(horizon, '1h')}"

        # Update horizon-specific metrics
        horizon_metrics = getattr(metrics, metrics_key, None)
        if horizon_metrics is None:
            # Fallback to 1h if unknown horizon
            horizon_metrics = metrics.metrics_1h

        old_count = horizon_metrics["count"]
        horizon_metrics["count"] += 1

        # Running average for error
        horizon_metrics["avg_error"] = (
            (horizon_metrics["avg_error"] * old_count + feedback.prediction_error_pct) /
            horizon_metrics["count"]
        )

        # Running average for direction accuracy
        horizon_metrics["direction_acc"] = (
            (horizon_metrics["direction_acc"] * old_count + (1.0 if feedback.direction_correct else 0.0)) /
            horizon_metrics["count"]
        )

        # Update overall metrics
        all_horizons = [metrics.metrics_1h, metrics.metrics_4h, metrics.metrics_24h]
        total_count = sum(h["count"] for h in all_horizons)
        if total_count > 0:
            metrics.avg_error_pct = sum(h["avg_error"] * h["count"] for h in all_horizons) / total_count
            metrics.direction_accuracy = sum(h["direction_acc"] * h["count"] for h in all_horizons) / total_count

        # Check if retraining is needed
        self._check_retraining_needed(symbol)

    def _check_retraining_needed(self, symbol: str):
        """Check if model needs retraining based on performance."""
        metrics = self.performance_metrics.get(symbol)
        if not metrics or metrics.evaluated_predictions < self.min_evaluations:
            return

        reasons = []

        # Check overall error
        if metrics.avg_error_pct > self.error_threshold_pct:
            reasons.append(f"high error ({metrics.avg_error_pct:.2f}% > {self.error_threshold_pct}%)")

        # Check direction accuracy
        if metrics.direction_accuracy < self.direction_threshold:
            reasons.append(f"low direction accuracy ({metrics.direction_accuracy:.1%} < {self.direction_threshold:.0%})")

        # Check 1h accuracy specifically (most important for trading)
        if metrics.metrics_1h["count"] >= 5:
            if metrics.metrics_1h["direction_acc"] < 0.4:
                reasons.append(f"poor 1h direction ({metrics.metrics_1h['direction_acc']:.1%})")

        if reasons:
            metrics.needs_retraining = True
            metrics.retraining_reason = "; ".join(reasons)
            logger.warning(f"{symbol} model needs retraining: {metrics.retraining_reason}")
        else:
            metrics.needs_retraining = False
            metrics.retraining_reason = None

    def get_symbols_needing_retraining(self) -> List[str]:
        """Get list of symbols whose models need retraining."""
        return [
            symbol for symbol, metrics in self.performance_metrics.items()
            if metrics.needs_retraining
        ]

    def get_performance_summary(self) -> Dict:
        """Get performance summary for all models."""
        summary = {
            "total_symbols": len(self.performance_metrics),
            "total_predictions": sum(m.total_predictions for m in self.performance_metrics.values()),
            "total_evaluated": sum(m.evaluated_predictions for m in self.performance_metrics.values()),
            "pending_evaluations": sum(len(v) for v in self.pending_feedback.values()),
            "symbols_needing_retraining": self.get_symbols_needing_retraining(),
            "by_symbol": {}
        }

        for symbol, metrics in self.performance_metrics.items():
            # Sanitize metrics_1h, metrics_4h, metrics_24h dicts
            def sanitize_metrics_dict(d: Dict) -> Dict:
                if not d:
                    return {}
                return {k: _safe_round(v) if isinstance(v, (int, float)) else v for k, v in d.items()}

            summary["by_symbol"][symbol] = {
                "total": metrics.total_predictions,
                "evaluated": metrics.evaluated_predictions,
                "avg_error_pct": _safe_round(metrics.avg_error_pct, 4),
                "direction_accuracy": _safe_round(metrics.direction_accuracy, 4),
                "needs_retraining": metrics.needs_retraining,
                "reason": metrics.retraining_reason,
                "1h": sanitize_metrics_dict(metrics.metrics_1h),
                "4h": sanitize_metrics_dict(metrics.metrics_4h),
                "24h": sanitize_metrics_dict(metrics.metrics_24h),
            }

        return summary

    def get_evaluated_predictions(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Get list of evaluated predictions for display.

        Args:
            symbol: Optional symbol to filter by. If None, returns all.
            limit: Maximum number of predictions to return per symbol.

        Returns:
            List of evaluated prediction dictionaries.
        """
        result = []

        if symbol:
            feedbacks = self.evaluated_feedback.get(symbol, [])
            for fb in feedbacks[-limit:]:
                result.append({
                    "symbol": fb.symbol,
                    "timestamp": fb.timestamp.isoformat() if fb.timestamp else None,
                    "horizon": fb.horizon,
                    "current_price": fb.current_price,
                    "predicted_price": fb.predicted_price,
                    "actual_price": fb.actual_price,
                    "prediction_error_pct": round(fb.prediction_error_pct, 4) if fb.prediction_error_pct else None,
                    "direction_correct": fb.direction_correct,
                    "evaluated_at": fb.evaluated_at.isoformat() if fb.evaluated_at else None,
                })
        else:
            for sym, feedbacks in self.evaluated_feedback.items():
                for fb in feedbacks[-limit:]:
                    result.append({
                        "symbol": fb.symbol,
                        "timestamp": fb.timestamp.isoformat() if fb.timestamp else None,
                        "horizon": fb.horizon,
                        "current_price": fb.current_price,
                        "predicted_price": fb.predicted_price,
                        "actual_price": fb.actual_price,
                        "prediction_error_pct": round(fb.prediction_error_pct, 4) if fb.prediction_error_pct else None,
                        "direction_correct": fb.direction_correct,
                        "evaluated_at": fb.evaluated_at.isoformat() if fb.evaluated_at else None,
                    })

        # Sort by evaluated_at descending
        result.sort(key=lambda x: x.get("evaluated_at") or "", reverse=True)
        return result[:limit]

    async def start_evaluation_loop(self, db_pool=None, interval_seconds: int = 300):
        """Start background evaluation loop."""
        if self._running:
            return

        self._running = True
        self._auto_evaluation_interval_seconds = interval_seconds
        logger.info(f"Starting model evaluation loop (interval: {interval_seconds}s)")

        async def evaluation_loop():
            while self._running:
                try:
                    await self._run_auto_evaluation_cycle()
                except Exception as e:
                    logger.error(f"Error in evaluation loop: {e}")

                await asyncio.sleep(interval_seconds)

        self._evaluation_task = asyncio.create_task(evaluation_loop())

    async def _run_auto_evaluation_cycle(self):
        """
        Run a single auto-evaluation cycle.

        This evaluates pending predictions and triggers auto-retrain if needed.
        """
        if self._evaluation_in_progress:
            logger.debug("Evaluation already in progress, skipping cycle")
            return

        self._evaluation_in_progress = True
        try:
            # Evaluate pending predictions via API (no DB required)
            evaluated = await self.evaluate_pending_predictions_via_api()

            if evaluated:
                self._total_auto_evaluations += sum(evaluated.values())
                self._last_evaluation_time = datetime.utcnow()
                logger.info(f"Auto-evaluation completed: {evaluated}")

                # Check if any symbols need retraining and trigger auto-retrain
                if self._auto_retrain_enabled:
                    await self._trigger_auto_retrain_if_needed()

        except Exception as e:
            logger.error(f"Error in auto-evaluation cycle: {e}")
        finally:
            self._evaluation_in_progress = False

    async def _trigger_auto_retrain_if_needed(self):
        """
        Check if any symbols need retraining and trigger auto-retrain.
        """
        if self._retrain_in_progress:
            logger.debug("Retrain already in progress, skipping")
            return

        symbols_to_retrain = self._get_symbols_eligible_for_retrain()

        if not symbols_to_retrain:
            return

        self._retrain_in_progress = True
        try:
            logger.info(f"Auto-retrain triggered for symbols: {symbols_to_retrain}")
            self._last_auto_retrain_symbols = symbols_to_retrain

            # Import training service here to avoid circular imports
            from .nhits_training_service import nhits_training_service

            # Check if training is already in progress
            status = nhits_training_service.get_status()
            if status.get("training_in_progress"):
                logger.info("Training already in progress, deferring auto-retrain")
                return

            # Retrain each symbol
            retrained_count = 0
            for symbol in symbols_to_retrain:
                try:
                    # Train all timeframes for the symbol
                    for timeframe in ["M15", "H1", "D1"]:
                        result = await nhits_training_service.train_symbol(
                            symbol=symbol,
                            force=True,
                            timeframe=timeframe
                        )

                        if result.success:
                            logger.info(
                                f"Auto-retrain successful: {symbol}/{timeframe} "
                                f"(samples={result.training_samples}, duration={result.training_duration_seconds:.1f}s)"
                            )
                        else:
                            logger.warning(f"Auto-retrain failed: {symbol}/{timeframe} - {result.error_message}")

                    # Update retrain history
                    self._retrain_history[symbol] = datetime.utcnow()
                    retrained_count += 1

                    # Reset metrics after successful retrain
                    if symbol in self.performance_metrics:
                        self.performance_metrics[symbol].needs_retraining = False
                        self.performance_metrics[symbol].retraining_reason = None

                except Exception as e:
                    logger.error(f"Error during auto-retrain for {symbol}: {e}")

            if retrained_count > 0:
                self._total_auto_retrains += retrained_count
                self._last_retrain_time = datetime.utcnow()
                self._save_data()
                logger.info(f"Auto-retrain completed: {retrained_count} symbols retrained")

        except Exception as e:
            logger.error(f"Error in auto-retrain: {e}")
        finally:
            self._retrain_in_progress = False

    def _get_symbols_eligible_for_retrain(self) -> List[str]:
        """
        Get list of symbols that need retraining and are not on cooldown.
        """
        now = datetime.utcnow()
        eligible = []

        for symbol, metrics in self.performance_metrics.items():
            if not metrics.needs_retraining:
                continue

            # Check cooldown
            last_retrain = self._retrain_history.get(symbol)
            if last_retrain:
                hours_since_retrain = (now - last_retrain).total_seconds() / 3600
                if hours_since_retrain < self._retrain_cooldown_hours:
                    logger.debug(
                        f"Skipping {symbol} - retrained {hours_since_retrain:.1f}h ago "
                        f"(cooldown: {self._retrain_cooldown_hours}h)"
                    )
                    continue

            eligible.append(symbol)

        return eligible

    def set_auto_evaluation_enabled(self, enabled: bool):
        """Enable or disable auto-evaluation."""
        self._auto_evaluation_enabled = enabled
        logger.info(f"Auto-evaluation {'enabled' if enabled else 'disabled'}")

    def set_auto_retrain_enabled(self, enabled: bool):
        """Enable or disable auto-retrain."""
        self._auto_retrain_enabled = enabled
        logger.info(f"Auto-retrain {'enabled' if enabled else 'disabled'}")

    def set_retrain_cooldown_hours(self, hours: int):
        """Set the cooldown period between retrains for the same symbol."""
        self._retrain_cooldown_hours = max(1, hours)
        logger.info(f"Retrain cooldown set to {self._retrain_cooldown_hours} hours")

    def get_auto_status(self) -> Dict:
        """Get status of automatic evaluation and retraining."""
        pending_count = sum(len(v) for v in self.pending_feedback.values())
        evaluated_count = sum(len(v) for v in self.evaluated_feedback.values())

        # Count predictions ready for evaluation
        now = datetime.utcnow()
        ready_for_eval = 0
        waiting_for_horizon = 0

        for symbol, feedbacks in self.pending_feedback.items():
            for fb in feedbacks:
                target_time = self._parse_horizon_to_target_time(fb.timestamp, fb.horizon)
                if now >= target_time + timedelta(minutes=30):
                    ready_for_eval += 1
                else:
                    waiting_for_horizon += 1

        return {
            "auto_evaluation": {
                "enabled": self._auto_evaluation_enabled,
                "running": self._running,
                "interval_seconds": self._auto_evaluation_interval_seconds,
                "in_progress": self._evaluation_in_progress,
                "last_run": self._last_evaluation_time.isoformat() if self._last_evaluation_time else None,
                "total_evaluations": self._total_auto_evaluations,
            },
            "auto_retrain": {
                "enabled": self._auto_retrain_enabled,
                "in_progress": self._retrain_in_progress,
                "last_run": self._last_retrain_time.isoformat() if self._last_retrain_time else None,
                "total_retrains": self._total_auto_retrains,
                "cooldown_hours": self._retrain_cooldown_hours,
                "last_retrained_symbols": self._last_auto_retrain_symbols,
            },
            "predictions": {
                "pending_count": pending_count,
                "evaluated_count": evaluated_count,
                "ready_for_evaluation": ready_for_eval,
                "waiting_for_horizon": waiting_for_horizon,
            },
            "symbols_needing_retrain": self.get_symbols_needing_retraining(),
            "symbols_eligible_for_retrain": self._get_symbols_eligible_for_retrain(),
        }

    async def trigger_manual_evaluation(self) -> Dict:
        """
        Manually trigger an evaluation cycle.

        Returns evaluation results.
        """
        logger.info("Manual evaluation triggered")
        evaluated = await self.evaluate_pending_predictions_via_api()

        result = {
            "success": True,
            "evaluated": evaluated,
            "total_evaluated": sum(evaluated.values()) if evaluated else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check for symbols needing retrain
        symbols_need_retrain = self.get_symbols_needing_retraining()
        if symbols_need_retrain:
            result["symbols_needing_retrain"] = symbols_need_retrain

        return result

    async def trigger_manual_retrain(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Manually trigger retraining for specified symbols or all symbols needing it.

        Args:
            symbols: List of symbols to retrain. If None, retrain all symbols needing it.

        Returns:
            Retrain results.
        """
        if symbols is None:
            symbols = self.get_symbols_needing_retraining()

        if not symbols:
            return {
                "success": True,
                "message": "No symbols need retraining",
                "retrained": [],
            }

        logger.info(f"Manual retrain triggered for: {symbols}")

        # Import training service
        from .nhits_training_service import nhits_training_service

        status = nhits_training_service.get_status()
        if status.get("training_in_progress"):
            return {
                "success": False,
                "message": "Training already in progress",
                "retrained": [],
            }

        retrained = []
        failed = []

        for symbol in symbols:
            try:
                for timeframe in ["M15", "H1", "D1"]:
                    result = await nhits_training_service.train_symbol(
                        symbol=symbol,
                        force=True,
                        timeframe=timeframe
                    )

                    if result.success:
                        retrained.append(f"{symbol}/{timeframe}")
                    else:
                        failed.append(f"{symbol}/{timeframe}: {result.error_message}")

                # Reset metrics
                if symbol in self.performance_metrics:
                    self.performance_metrics[symbol].needs_retraining = False
                    self.performance_metrics[symbol].retraining_reason = None

                self._retrain_history[symbol] = datetime.utcnow()

            except Exception as e:
                failed.append(f"{symbol}: {str(e)}")

        self._save_data()

        return {
            "success": len(retrained) > 0,
            "retrained": retrained,
            "failed": failed,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def stop(self):
        """Stop the evaluation loop."""
        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
        self._save_data()
        logger.info("ModelImprovementService stopped")

    async def start(self, interval_seconds: int = 300):
        """
        Start the automatic evaluation and retrain service.

        This is the main entry point for the auto-improvement loop.
        """
        if self._running:
            logger.warning("ModelImprovementService already running")
            return

        await self.start_evaluation_loop(interval_seconds=interval_seconds)
        logger.info(
            f"ModelImprovementService started "
            f"(eval_interval={interval_seconds}s, "
            f"auto_retrain={'enabled' if self._auto_retrain_enabled else 'disabled'})"
        )


# Singleton instance
model_improvement_service = ModelImprovementService()
