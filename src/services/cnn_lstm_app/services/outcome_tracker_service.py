"""
Outcome Tracker Service for CNN-LSTM Predictions.

Tracks price movements and market conditions after multi-task predictions
to evaluate prediction accuracy for self-learning feedback.
"""

import asyncio
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger


DATA_DIR = os.getenv("DATA_DIR", "/app/data")


class OutcomeStatus(Enum):
    """Status of outcome tracking."""
    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class TaskOutcome:
    """Outcome for a single prediction task."""
    task_type: str  # price, patterns, regime
    predicted: Any  # Original prediction
    actual: Optional[Any] = None
    correct: Optional[bool] = None
    accuracy_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionOutcome:
    """Tracks the outcome of a CNN-LSTM multi-task prediction."""
    # Identification
    prediction_id: str
    symbol: str
    timeframe: str

    # Timing
    prediction_timestamp: str
    tracking_started: str
    tracking_ends: str

    # Price at detection
    price_at_prediction: float

    # Current tracking state
    status: OutcomeStatus = OutcomeStatus.PENDING

    # Task outcomes
    price_outcome: Optional[Dict[str, Any]] = None
    pattern_outcome: Optional[Dict[str, Any]] = None
    regime_outcome: Optional[Dict[str, Any]] = None

    # Original predictions (for feedback)
    original_predictions: Dict[str, Any] = field(default_factory=dict)

    # Price tracking
    current_price: Optional[float] = None
    high_since: Optional[float] = None
    low_since: Optional[float] = None
    max_favorable_move: float = 0.0
    max_adverse_move: float = 0.0

    # OHLCV context at prediction time
    ohlcv_context: Optional[List[Dict]] = None

    # Final evaluation
    final_evaluation: Optional[str] = None
    overall_accuracy: float = 0.0
    task_accuracies: Dict[str, float] = field(default_factory=dict)

    # Metadata
    last_updated: Optional[str] = None
    update_count: int = 0
    sent_to_feedback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


# Evaluation periods based on timeframe
EVALUATION_PERIODS = {
    "M1": timedelta(hours=1),
    "M5": timedelta(hours=4),
    "M15": timedelta(hours=12),
    "M30": timedelta(hours=24),
    "H1": timedelta(hours=24),
    "H4": timedelta(hours=72),
    "D1": timedelta(days=14),
    "W1": timedelta(days=28),
}


class OutcomeTrackerService:
    """
    Service for tracking CNN-LSTM prediction outcomes.

    Monitors price movements and market conditions after predictions
    to evaluate accuracy and generate feedback for self-learning.
    """

    def __init__(self, outcomes_file: str = None):
        if outcomes_file is None:
            outcomes_file = os.path.join(DATA_DIR, "cnn_lstm_outcomes.json")

        self._outcomes_file = Path(outcomes_file)
        self._outcomes: Dict[str, PredictionOutcome] = {}

        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._update_interval = 15 * 60  # 15 minutes

        # Service URLs
        self._data_service_url = os.getenv(
            "DATA_SERVICE_URL", "http://trading-data:3001"
        )
        self._train_service_url = os.getenv(
            "CNN_LSTM_TRAIN_SERVICE_URL", "http://trading-cnn-lstm-train:3017"
        )

        self._load_outcomes()
        logger.info(f"OutcomeTrackerService initialized - {len(self._outcomes)} outcomes loaded")

    def _load_outcomes(self):
        """Load outcomes from file."""
        try:
            if self._outcomes_file.exists():
                with open(self._outcomes_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item["status"] = OutcomeStatus(item["status"])
                        outcome = PredictionOutcome(**item)
                        self._outcomes[outcome.prediction_id] = outcome
                logger.info(f"Loaded {len(self._outcomes)} prediction outcomes")
        except Exception as e:
            logger.error(f"Failed to load outcomes: {e}")
            self._outcomes = {}

    def _save_outcomes(self):
        """Save outcomes to file."""
        try:
            self._outcomes_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._outcomes_file, 'w') as f:
                json.dump([o.to_dict() for o in self._outcomes.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save outcomes: {e}")

    async def track_prediction(self, prediction_data: Dict[str, Any]) -> Optional[PredictionOutcome]:
        """
        Start tracking a prediction for outcome evaluation.

        Args:
            prediction_data: Prediction data including id, symbol, timeframe, predictions

        Returns:
            PredictionOutcome if tracking started, None if failed
        """
        try:
            prediction_id = prediction_data.get("id") or prediction_data.get("prediction_id")
            symbol = prediction_data.get("symbol", "").upper()
            timeframe = prediction_data.get("timeframe", "H1").upper()

            if not prediction_id or not symbol:
                logger.warning("Missing prediction_id or symbol")
                return None

            # Check if already tracking
            if prediction_id in self._outcomes:
                logger.debug(f"Already tracking prediction {prediction_id}")
                return self._outcomes[prediction_id]

            # Determine evaluation period
            eval_period = EVALUATION_PERIODS.get(timeframe, timedelta(hours=24))
            now = datetime.now(timezone.utc)

            # Extract original predictions
            original_predictions = {
                "price": prediction_data.get("price_prediction"),
                "patterns": prediction_data.get("pattern_prediction"),
                "regime": prediction_data.get("regime_prediction"),
            }

            outcome = PredictionOutcome(
                prediction_id=prediction_id,
                symbol=symbol,
                timeframe=timeframe,
                prediction_timestamp=prediction_data.get("timestamp", now.isoformat()),
                tracking_started=now.isoformat(),
                tracking_ends=(now + eval_period).isoformat(),
                price_at_prediction=prediction_data.get("price_at_prediction", 0.0),
                original_predictions=original_predictions,
                ohlcv_context=prediction_data.get("ohlcv_context"),
            )

            self._outcomes[prediction_id] = outcome
            self._save_outcomes()

            logger.info(f"Started tracking prediction {prediction_id} for {symbol}/{timeframe}")
            return outcome

        except Exception as e:
            logger.error(f"Failed to track prediction: {e}")
            return None

    async def update_outcomes(self):
        """Update all pending outcomes with current market data."""
        pending = [o for o in self._outcomes.values() if o.status == OutcomeStatus.PENDING]

        if not pending:
            return

        logger.info(f"Updating {len(pending)} pending outcomes")

        for outcome in pending:
            try:
                await self._update_single_outcome(outcome)
            except Exception as e:
                logger.error(f"Error updating outcome {outcome.prediction_id}: {e}")

        self._save_outcomes()

    async def _update_single_outcome(self, outcome: PredictionOutcome):
        """Update a single outcome with current market data."""
        now = datetime.now(timezone.utc)
        tracking_ends = datetime.fromisoformat(outcome.tracking_ends.replace('Z', '+00:00'))

        # Fetch current price
        current_price = await self._fetch_current_price(outcome.symbol)
        if current_price is None:
            return

        outcome.current_price = current_price
        outcome.update_count += 1
        outcome.last_updated = now.isoformat()

        # Update price extremes
        if outcome.high_since is None or current_price > outcome.high_since:
            outcome.high_since = current_price
        if outcome.low_since is None or current_price < outcome.low_since:
            outcome.low_since = current_price

        # Calculate moves
        if outcome.price_at_prediction > 0:
            price_change = (current_price - outcome.price_at_prediction) / outcome.price_at_prediction

            # Determine favorable/adverse based on price prediction direction
            price_pred = outcome.original_predictions.get("price", {})
            predicted_direction = price_pred.get("direction", "neutral") if price_pred else "neutral"

            if predicted_direction == "bullish":
                outcome.max_favorable_move = max(
                    outcome.max_favorable_move,
                    (outcome.high_since - outcome.price_at_prediction) / outcome.price_at_prediction
                )
                outcome.max_adverse_move = min(
                    outcome.max_adverse_move,
                    (outcome.low_since - outcome.price_at_prediction) / outcome.price_at_prediction
                )
            elif predicted_direction == "bearish":
                outcome.max_favorable_move = max(
                    outcome.max_favorable_move,
                    (outcome.price_at_prediction - outcome.low_since) / outcome.price_at_prediction
                )
                outcome.max_adverse_move = min(
                    outcome.max_adverse_move,
                    (outcome.price_at_prediction - outcome.high_since) / outcome.price_at_prediction
                )

        # Check if evaluation period has ended
        if now >= tracking_ends:
            await self._evaluate_outcome(outcome)

    async def _evaluate_outcome(self, outcome: PredictionOutcome):
        """Evaluate final outcome and determine accuracy."""
        task_accuracies = {}

        # Evaluate price prediction
        price_accuracy = await self._evaluate_price_task(outcome)
        if price_accuracy is not None:
            task_accuracies["price"] = price_accuracy

        # Evaluate pattern prediction
        pattern_accuracy = await self._evaluate_pattern_task(outcome)
        if pattern_accuracy is not None:
            task_accuracies["patterns"] = pattern_accuracy

        # Evaluate regime prediction
        regime_accuracy = await self._evaluate_regime_task(outcome)
        if regime_accuracy is not None:
            task_accuracies["regime"] = regime_accuracy

        # Calculate overall accuracy (weighted)
        weights = {"price": 0.4, "patterns": 0.35, "regime": 0.25}
        total_weight = sum(weights.get(k, 0) for k in task_accuracies.keys())

        if total_weight > 0:
            overall_accuracy = sum(
                task_accuracies.get(k, 0) * weights.get(k, 0)
                for k in task_accuracies.keys()
            ) / total_weight
        else:
            overall_accuracy = 0.0

        outcome.task_accuracies = task_accuracies
        outcome.overall_accuracy = overall_accuracy

        # Determine status based on overall accuracy
        if overall_accuracy >= 0.7:
            outcome.status = OutcomeStatus.SUCCESS
            outcome.final_evaluation = "accurate"
        elif overall_accuracy >= 0.4:
            outcome.status = OutcomeStatus.PARTIAL
            outcome.final_evaluation = "partially_accurate"
        else:
            outcome.status = OutcomeStatus.FAILED
            outcome.final_evaluation = "inaccurate"

        logger.info(
            f"Evaluated {outcome.prediction_id}: {outcome.status.value} "
            f"(overall: {overall_accuracy:.2f}, tasks: {task_accuracies})"
        )

        # Send observation to drift detection service
        await self._send_to_drift_detection(outcome)

        # Send to feedback buffer
        await self._send_to_feedback_buffer(outcome)

    async def _evaluate_price_task(self, outcome: PredictionOutcome) -> Optional[float]:
        """Evaluate price prediction accuracy."""
        price_pred = outcome.original_predictions.get("price")
        if not price_pred:
            return None

        predicted_direction = price_pred.get("direction", "neutral")
        predicted_change = price_pred.get("change_percent", 0.0)

        if outcome.price_at_prediction <= 0 or outcome.current_price is None:
            return None

        actual_change = (outcome.current_price - outcome.price_at_prediction) / outcome.price_at_prediction
        actual_direction = "bullish" if actual_change > 0.001 else ("bearish" if actual_change < -0.001 else "neutral")

        # Direction accuracy (0 or 1)
        direction_correct = (
            (predicted_direction == "bullish" and actual_direction == "bullish") or
            (predicted_direction == "bearish" and actual_direction == "bearish") or
            (predicted_direction == "neutral" and abs(actual_change) < 0.005)
        )

        # Magnitude accuracy (how close was the predicted change)
        if predicted_change != 0:
            magnitude_error = abs(actual_change - predicted_change) / abs(predicted_change)
            magnitude_accuracy = max(0, 1 - magnitude_error)
        else:
            magnitude_accuracy = 0.5 if abs(actual_change) < 0.01 else 0.0

        # Combined score: 60% direction, 40% magnitude
        accuracy = (0.6 * (1.0 if direction_correct else 0.0)) + (0.4 * magnitude_accuracy)

        outcome.price_outcome = {
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "direction_correct": direction_correct,
            "predicted_change": predicted_change,
            "actual_change": actual_change,
            "magnitude_accuracy": magnitude_accuracy,
            "accuracy": accuracy
        }

        return accuracy

    async def _evaluate_pattern_task(self, outcome: PredictionOutcome) -> Optional[float]:
        """Evaluate pattern prediction accuracy."""
        pattern_pred = outcome.original_predictions.get("patterns")
        if not pattern_pred:
            return None

        # Pattern evaluation would require comparing predicted patterns
        # with what actually developed - this is complex and may need
        # additional chart analysis

        # For now, use a simplified evaluation based on confidence
        predicted_patterns = pattern_pred.get("detected_patterns", [])
        confidence = pattern_pred.get("confidence", 0.5)

        # Higher confidence predictions should be held to higher standards
        # This is a placeholder - real evaluation would compare with chart developments
        accuracy = confidence if predicted_patterns else 0.5

        outcome.pattern_outcome = {
            "predicted_patterns": predicted_patterns,
            "confidence": confidence,
            "accuracy": accuracy,
            "note": "Simplified evaluation based on prediction confidence"
        }

        return accuracy

    async def _evaluate_regime_task(self, outcome: PredictionOutcome) -> Optional[float]:
        """Evaluate regime prediction accuracy."""
        regime_pred = outcome.original_predictions.get("regime")
        if not regime_pred:
            return None

        predicted_regime = regime_pred.get("regime", "unknown")
        confidence = regime_pred.get("confidence", 0.5)

        # Determine actual regime based on price movement
        if outcome.price_at_prediction <= 0 or outcome.current_price is None:
            return None

        price_change = (outcome.current_price - outcome.price_at_prediction) / outcome.price_at_prediction
        volatility = abs(outcome.max_favorable_move) + abs(outcome.max_adverse_move)

        # Simplified regime determination
        if volatility > 0.05:  # > 5% total movement
            actual_regime = "high_volatility"
        elif price_change > 0.02:  # > 2% gain
            actual_regime = "bull_trend"
        elif price_change < -0.02:  # > 2% loss
            actual_regime = "bear_trend"
        else:
            actual_regime = "sideways"

        # Check if prediction was correct
        regime_correct = (predicted_regime == actual_regime)

        # Partial credit for related regimes
        related_regimes = {
            "bull_trend": ["high_volatility"],
            "bear_trend": ["high_volatility"],
            "sideways": [],
            "high_volatility": ["bull_trend", "bear_trend"]
        }

        if regime_correct:
            accuracy = 1.0
        elif actual_regime in related_regimes.get(predicted_regime, []):
            accuracy = 0.5
        else:
            accuracy = 0.0

        outcome.regime_outcome = {
            "predicted_regime": predicted_regime,
            "actual_regime": actual_regime,
            "regime_correct": regime_correct,
            "confidence": confidence,
            "price_change": price_change,
            "volatility": volatility,
            "accuracy": accuracy
        }

        return accuracy

    async def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price from data service."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._data_service_url}/api/v1/ohlcv/{symbol}",
                    params={"limit": 1}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data and "data" in data and len(data["data"]) > 0:
                        return float(data["data"][-1].get("close", 0))
        except Exception as e:
            logger.debug(f"Failed to fetch price for {symbol}: {e}")
        return None

    async def _send_to_drift_detection(self, outcome: PredictionOutcome):
        """Send observation to drift detection service."""
        try:
            from .drift_detection_service import drift_detection_service

            # Extract task-specific info for drift detection
            price_outcome = outcome.price_outcome or {}
            pattern_outcome = outcome.pattern_outcome or {}
            regime_outcome = outcome.regime_outcome or {}

            price_pred = outcome.original_predictions.get("price", {})
            pattern_pred = outcome.original_predictions.get("patterns", {})
            regime_pred = outcome.original_predictions.get("regime", {})

            drift_detection_service.add_observation(
                prediction_id=outcome.prediction_id,
                symbol=outcome.symbol,
                timeframe=outcome.timeframe,
                # Price task
                price_direction_correct=price_outcome.get("direction_correct"),
                price_confidence=price_pred.get("confidence") if price_pred else None,
                price_magnitude_error=price_outcome.get("magnitude_accuracy"),
                # Pattern task
                pattern_correct=pattern_outcome.get("accuracy", 0) >= 0.5 if pattern_outcome else None,
                pattern_confidence=pattern_pred.get("confidence") if pattern_pred else None,
                patterns_predicted=pattern_pred.get("detected_patterns") if pattern_pred else None,
                # Regime task
                regime_correct=regime_outcome.get("regime_correct"),
                regime_confidence=regime_pred.get("confidence") if regime_pred else None,
                regime_predicted=regime_pred.get("regime") if regime_pred else None,
                # Overall
                overall_accuracy=outcome.overall_accuracy,
            )
            logger.debug(f"Sent outcome {outcome.prediction_id} to drift detection")
        except Exception as e:
            logger.debug(f"Could not send to drift detection: {e}")

    async def _send_to_feedback_buffer(self, outcome: PredictionOutcome):
        """Send completed outcome to training service feedback buffer."""
        if outcome.sent_to_feedback:
            return

        try:
            feedback_data = {
                "prediction_id": outcome.prediction_id,
                "symbol": outcome.symbol,
                "timeframe": outcome.timeframe,
                "status": outcome.status.value,
                "overall_accuracy": outcome.overall_accuracy,
                "task_accuracies": outcome.task_accuracies,
                "original_predictions": outcome.original_predictions,
                "price_outcome": outcome.price_outcome,
                "pattern_outcome": outcome.pattern_outcome,
                "regime_outcome": outcome.regime_outcome,
                "ohlcv_context": outcome.ohlcv_context,
                "price_at_prediction": outcome.price_at_prediction,
                "final_price": outcome.current_price,
                "max_favorable_move": outcome.max_favorable_move,
                "max_adverse_move": outcome.max_adverse_move,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._train_service_url}/api/v1/feedback-buffer/outcome",
                    json=feedback_data
                )

                if response.status_code == 200:
                    outcome.sent_to_feedback = True
                    logger.info(f"Sent outcome {outcome.prediction_id} to feedback buffer")
                else:
                    logger.warning(
                        f"Failed to send to feedback buffer: {response.status_code}"
                    )
        except Exception as e:
            logger.debug(f"Could not send to feedback buffer: {e}")

    async def start_loop(self) -> Dict[str, Any]:
        """Start the background update loop."""
        if self._running:
            return {"status": "already_running"}

        self._running = True
        self._loop_task = asyncio.create_task(self._update_loop())
        logger.info("Outcome tracker loop started")

        return {"status": "started"}

    async def stop_loop(self) -> Dict[str, Any]:
        """Stop the background update loop."""
        if not self._running:
            return {"status": "not_running"}

        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        logger.info("Outcome tracker loop stopped")
        return {"status": "stopped"}

    async def _update_loop(self):
        """Background loop to update outcomes."""
        while self._running:
            try:
                await self.update_outcomes()
            except Exception as e:
                logger.error(f"Error in outcome update loop: {e}")

            await asyncio.sleep(self._update_interval)

    def is_running(self) -> bool:
        """Check if loop is running."""
        return self._running

    def get_outcomes(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get outcomes with optional filters."""
        results = []

        for outcome in sorted(
            self._outcomes.values(),
            key=lambda x: x.tracking_started,
            reverse=True
        ):
            if status and outcome.status.value != status:
                continue
            if symbol and outcome.symbol != symbol.upper():
                continue

            results.append(outcome.to_dict())

            if len(results) >= limit:
                break

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get outcome statistics."""
        if not self._outcomes:
            return {
                "total_tracked": 0,
                "pending": 0,
                "completed": 0,
                "by_status": {},
                "average_accuracy": 0.0,
                "task_averages": {}
            }

        by_status = {}
        completed = []
        task_sums = {"price": 0.0, "patterns": 0.0, "regime": 0.0}
        task_counts = {"price": 0, "patterns": 0, "regime": 0}

        for outcome in self._outcomes.values():
            status = outcome.status.value
            by_status[status] = by_status.get(status, 0) + 1

            if outcome.status != OutcomeStatus.PENDING:
                completed.append(outcome)

                for task, acc in outcome.task_accuracies.items():
                    if task in task_sums:
                        task_sums[task] += acc
                        task_counts[task] += 1

        avg_accuracy = (
            sum(o.overall_accuracy for o in completed) / len(completed)
            if completed else 0.0
        )

        task_averages = {
            task: task_sums[task] / task_counts[task] if task_counts[task] > 0 else 0.0
            for task in task_sums
        }

        return {
            "total_tracked": len(self._outcomes),
            "pending": by_status.get("pending", 0),
            "completed": len(completed),
            "by_status": by_status,
            "average_accuracy": round(avg_accuracy, 3),
            "task_averages": {k: round(v, 3) for k, v in task_averages.items()},
            "success_rate": round(
                by_status.get("success", 0) / len(completed) * 100, 1
            ) if completed else 0.0
        }

    def clear_completed(self):
        """Clear completed outcomes."""
        before = len(self._outcomes)
        self._outcomes = {
            k: v for k, v in self._outcomes.items()
            if v.status == OutcomeStatus.PENDING
        }
        after = len(self._outcomes)
        self._save_outcomes()
        logger.info(f"Cleared {before - after} completed outcomes")


# Global singleton
outcome_tracker_service = OutcomeTrackerService()
