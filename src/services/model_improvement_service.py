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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn

from src.config.settings import settings

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
    """

    def __init__(self):
        self.feedback_path = Path("data/model_feedback")
        self.feedback_path.mkdir(parents=True, exist_ok=True)

        self.metrics_path = Path("data/model_metrics")
        self.metrics_path.mkdir(parents=True, exist_ok=True)

        self.pending_feedback: Dict[str, List[PredictionFeedback]] = {}
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}

        # Thresholds for retraining
        self.error_threshold_pct = 0.5  # Retrain if avg error > 0.5%
        self.direction_threshold = 0.45  # Retrain if direction accuracy < 45%
        self.min_evaluations = 10  # Minimum evaluations before checking

        self._running = False
        self._evaluation_task: Optional[asyncio.Task] = None

        # Load existing data
        self._load_data()

        logger.info("ModelImprovementService initialized")

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
    ):
        """
        Record a new prediction for later evaluation.

        Called automatically when NHITS makes a prediction.
        """
        now = datetime.utcnow()

        if symbol not in self.pending_feedback:
            self.pending_feedback[symbol] = []

        # Record each horizon prediction separately
        for horizon, pred_price in [("1h", predicted_price_1h), ("4h", predicted_price_4h), ("24h", predicted_price_24h)]:
            if pred_price is not None:
                feedback = PredictionFeedback(
                    symbol=symbol,
                    timestamp=now,
                    horizon=horizon,
                    current_price=current_price,
                    predicted_price=pred_price,
                )
                self.pending_feedback[symbol].append(feedback)

        # Update metrics count
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = ModelPerformanceMetrics(symbol=symbol)
        self.performance_metrics[symbol].total_predictions += 1

        # Save periodically
        if self.performance_metrics[symbol].total_predictions % 5 == 0:
            self._save_data()

        logger.debug(f"Recorded predictions for {symbol}: 1h={predicted_price_1h}, 4h={predicted_price_4h}, 24h={predicted_price_24h}")

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
                hours = int(fb.horizon.replace("h", ""))
                target_time = fb.timestamp + timedelta(hours=hours)

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
                        # Calculate error
                        fb.actual_price = actual_price
                        fb.prediction_error_pct = abs(fb.predicted_price - actual_price) / actual_price * 100
                        fb.direction_correct = (
                            (fb.predicted_price > fb.current_price) ==
                            (actual_price > fb.current_price)
                        )
                        fb.evaluated_at = now

                        # Update metrics
                        self._update_metrics(symbol, fb)
                        symbol_evaluated += 1

                        logger.debug(
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
        """Get actual price from database at target time."""
        query = """
            SELECT d1_close as close
            FROM symbol
            WHERE symbol = $1
              AND data_timestamp >= $2 - interval '30 minutes'
              AND data_timestamp <= $2 + interval '30 minutes'
            ORDER BY ABS(EXTRACT(EPOCH FROM (data_timestamp - $2)))
            LIMIT 1
        """

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(query, symbol, target_time)
            if row:
                return float(row["close"])
        return None

    def _update_metrics(self, symbol: str, feedback: PredictionFeedback):
        """Update performance metrics with new feedback."""
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = ModelPerformanceMetrics(symbol=symbol)

        metrics = self.performance_metrics[symbol]
        metrics.evaluated_predictions += 1
        metrics.last_updated = datetime.utcnow()

        # Update horizon-specific metrics
        horizon_metrics = getattr(metrics, f"metrics_{feedback.horizon}")
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
            summary["by_symbol"][symbol] = {
                "total": metrics.total_predictions,
                "evaluated": metrics.evaluated_predictions,
                "avg_error_pct": round(metrics.avg_error_pct, 4),
                "direction_accuracy": round(metrics.direction_accuracy, 4),
                "needs_retraining": metrics.needs_retraining,
                "reason": metrics.retraining_reason,
                "1h": metrics.metrics_1h,
                "4h": metrics.metrics_4h,
                "24h": metrics.metrics_24h,
            }

        return summary

    async def start_evaluation_loop(self, db_pool, interval_seconds: int = 300):
        """Start background evaluation loop."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting model evaluation loop (interval: {interval_seconds}s)")

        async def evaluation_loop():
            while self._running:
                try:
                    await self.evaluate_pending_predictions(db_pool)
                except Exception as e:
                    logger.error(f"Error in evaluation loop: {e}")

                await asyncio.sleep(interval_seconds)

        self._evaluation_task = asyncio.create_task(evaluation_loop())

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


# Singleton instance
model_improvement_service = ModelImprovementService()
