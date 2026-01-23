"""
Feedback Buffer Service for CNN-LSTM Self-Learning.

Collects and manages evaluated prediction outcomes for incremental training.
Supports multi-task feedback (price, patterns, regime).
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
from loguru import logger


DATA_DIR = os.getenv("DATA_DIR", "/app/data")


class FeedbackType(Enum):
    """Types of feedback sources."""
    # Outcome-based (automatic)
    OUTCOME_SUCCESS = "outcome_success"
    OUTCOME_PARTIAL = "outcome_partial"
    OUTCOME_FAILED = "outcome_failed"
    OUTCOME_EXPIRED = "outcome_expired"

    # User feedback (manual)
    USER_CONFIRMED = "user_confirmed"
    USER_CORRECTED = "user_corrected"
    USER_REJECTED = "user_rejected"


# Weights for different feedback types
# Failed predictions have higher weight for learning from mistakes
FEEDBACK_WEIGHTS = {
    FeedbackType.OUTCOME_SUCCESS: 1.0,
    FeedbackType.OUTCOME_PARTIAL: 1.2,
    FeedbackType.OUTCOME_FAILED: 1.5,  # Learn more from failures
    FeedbackType.OUTCOME_EXPIRED: 0.8,
    FeedbackType.USER_CONFIRMED: 1.3,  # User feedback is valuable
    FeedbackType.USER_CORRECTED: 1.8,  # Corrections are highly valuable
    FeedbackType.USER_REJECTED: 1.6,
}


# Multi-task label mappings
REGIME_LABELS = {
    "bull_trend": 0,
    "bear_trend": 1,
    "sideways": 2,
    "high_volatility": 3,
}

PATTERN_LABELS = {
    "head_and_shoulders": 0,
    "inverse_head_and_shoulders": 1,
    "double_top": 2,
    "double_bottom": 3,
    "ascending_triangle": 4,
    "descending_triangle": 5,
    "symmetric_triangle": 6,
    "rising_wedge": 7,
    "falling_wedge": 8,
    "bull_flag": 9,
    "bear_flag": 10,
    "cup_and_handle": 11,
    "rounding_bottom": 12,
    "rounding_top": 13,
    "channel_up": 14,
    "channel_down": 15,
}


@dataclass
class FeedbackSample:
    """A single feedback sample for training."""
    sample_id: str
    prediction_id: str
    symbol: str
    timeframe: str

    # Feedback type and timing
    feedback_type: FeedbackType
    timestamp: str

    # Multi-task labels
    price_label: Optional[Dict[str, Any]] = None  # direction, change_percent
    pattern_labels: Optional[List[int]] = None  # Multi-label pattern indices
    regime_label: Optional[int] = None  # Single regime class

    # Accuracy scores per task
    price_accuracy: float = 0.0
    pattern_accuracy: float = 0.0
    regime_accuracy: float = 0.0
    overall_accuracy: float = 0.0

    # OHLCV context (for input reconstruction)
    ohlcv_context: Optional[List[Dict]] = None

    # Price info
    price_at_prediction: float = 0.0
    final_price: float = 0.0

    # Weight for training
    weight: float = 1.0

    # Status
    used_for_training: bool = False
    training_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["feedback_type"] = self.feedback_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackSample":
        """Create from dictionary."""
        data = data.copy()
        if "feedback_type" in data:
            data["feedback_type"] = FeedbackType(data["feedback_type"])
        return cls(**data)


@dataclass
class BufferStatistics:
    """Statistics about the feedback buffer."""
    total_samples: int = 0
    unused_samples: int = 0
    ready_for_training: bool = False

    by_feedback_type: Dict[str, int] = field(default_factory=dict)
    by_symbol: Dict[str, int] = field(default_factory=dict)
    by_timeframe: Dict[str, int] = field(default_factory=dict)

    # Task-specific stats
    samples_with_price_label: int = 0
    samples_with_pattern_label: int = 0
    samples_with_regime_label: int = 0

    average_accuracy: float = 0.0
    task_averages: Dict[str, float] = field(default_factory=dict)

    oldest_sample: Optional[str] = None
    newest_sample: Optional[str] = None


class FeedbackBufferService:
    """
    Service for managing the feedback buffer for CNN-LSTM self-learning.

    Collects evaluated prediction outcomes and prepares them for incremental training.
    """

    def __init__(self, buffer_file: str = None):
        if buffer_file is None:
            buffer_file = os.path.join(DATA_DIR, "cnn_lstm_feedback_buffer.json")

        self._buffer_file = Path(buffer_file)
        self._samples: List[FeedbackSample] = []

        # Configuration
        self._max_buffer_size = int(os.getenv("FEEDBACK_BUFFER_MAX_SIZE", "10000"))
        self._max_age_hours = int(os.getenv("FEEDBACK_BUFFER_MAX_AGE_HOURS", "168"))  # 7 days
        self._min_samples_for_training = int(os.getenv("MIN_SAMPLES_FOR_TRAINING", "100"))

        self._load_buffer()
        logger.info(f"FeedbackBufferService initialized - {len(self._samples)} samples loaded")

    def _load_buffer(self):
        """Load buffer from file."""
        try:
            if self._buffer_file.exists():
                with open(self._buffer_file, 'r') as f:
                    data = json.load(f)
                    self._samples = [FeedbackSample.from_dict(item) for item in data]
                logger.info(f"Loaded {len(self._samples)} feedback samples")
        except Exception as e:
            logger.error(f"Failed to load feedback buffer: {e}")
            self._samples = []

    def _save_buffer(self):
        """Save buffer to file."""
        try:
            self._buffer_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._buffer_file, 'w') as f:
                json.dump([s.to_dict() for s in self._samples], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback buffer: {e}")

    def add_outcome_feedback(self, outcome_data: Dict[str, Any]) -> Optional[FeedbackSample]:
        """
        Add feedback from an evaluated prediction outcome.

        Args:
            outcome_data: Outcome data from the outcome tracker

        Returns:
            Created FeedbackSample or None if failed
        """
        try:
            prediction_id = outcome_data.get("prediction_id")
            status = outcome_data.get("status", "")

            # Map status to feedback type
            feedback_type_map = {
                "success": FeedbackType.OUTCOME_SUCCESS,
                "partial": FeedbackType.OUTCOME_PARTIAL,
                "failed": FeedbackType.OUTCOME_FAILED,
                "expired": FeedbackType.OUTCOME_EXPIRED,
            }

            feedback_type = feedback_type_map.get(status, FeedbackType.OUTCOME_PARTIAL)

            # Extract task-specific outcomes
            price_outcome = outcome_data.get("price_outcome", {})
            pattern_outcome = outcome_data.get("pattern_outcome", {})
            regime_outcome = outcome_data.get("regime_outcome", {})

            # Build price label
            price_label = None
            if price_outcome:
                price_label = {
                    "direction": price_outcome.get("actual_direction"),
                    "change_percent": price_outcome.get("actual_change", 0.0),
                }

            # Build pattern labels (multi-label)
            pattern_labels = None
            if pattern_outcome and pattern_outcome.get("predicted_patterns"):
                # For now, use predicted patterns as labels
                # In production, this would be the actual patterns that developed
                pattern_labels = []
                for p in pattern_outcome.get("predicted_patterns", []):
                    if p in PATTERN_LABELS:
                        pattern_labels.append(PATTERN_LABELS[p])

            # Build regime label
            regime_label = None
            if regime_outcome:
                actual_regime = regime_outcome.get("actual_regime")
                if actual_regime in REGIME_LABELS:
                    regime_label = REGIME_LABELS[actual_regime]

            # Calculate weight
            weight = FEEDBACK_WEIGHTS.get(feedback_type, 1.0)

            # Get task accuracies
            task_accuracies = outcome_data.get("task_accuracies", {})

            sample = FeedbackSample(
                sample_id=f"fb_{prediction_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                prediction_id=prediction_id,
                symbol=outcome_data.get("symbol", ""),
                timeframe=outcome_data.get("timeframe", ""),
                feedback_type=feedback_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                price_label=price_label,
                pattern_labels=pattern_labels,
                regime_label=regime_label,
                price_accuracy=task_accuracies.get("price", 0.0),
                pattern_accuracy=task_accuracies.get("patterns", 0.0),
                regime_accuracy=task_accuracies.get("regime", 0.0),
                overall_accuracy=outcome_data.get("overall_accuracy", 0.0),
                ohlcv_context=outcome_data.get("ohlcv_context"),
                price_at_prediction=outcome_data.get("price_at_prediction", 0.0),
                final_price=outcome_data.get("final_price", 0.0),
                weight=weight,
            )

            self._samples.append(sample)
            self._cleanup_old_samples()
            self._save_buffer()

            logger.info(f"Added feedback sample {sample.sample_id} (type: {feedback_type.value})")
            return sample

        except Exception as e:
            logger.error(f"Failed to add outcome feedback: {e}")
            return None

    def add_user_feedback(
        self,
        prediction_id: str,
        symbol: str,
        timeframe: str,
        feedback_type: str,  # confirmed, corrected, rejected
        price_correction: Optional[Dict] = None,
        pattern_correction: Optional[List[str]] = None,
        regime_correction: Optional[str] = None,
        ohlcv_context: Optional[List[Dict]] = None,
    ) -> Optional[FeedbackSample]:
        """
        Add feedback from user input.

        Args:
            prediction_id: ID of the prediction
            symbol: Trading symbol
            timeframe: Timeframe
            feedback_type: confirmed, corrected, or rejected
            price_correction: Corrected price info
            pattern_correction: List of actual pattern names
            regime_correction: Actual regime name
            ohlcv_context: OHLCV context data

        Returns:
            Created FeedbackSample or None if failed
        """
        try:
            # Map to feedback type
            type_map = {
                "confirmed": FeedbackType.USER_CONFIRMED,
                "corrected": FeedbackType.USER_CORRECTED,
                "rejected": FeedbackType.USER_REJECTED,
            }

            fb_type = type_map.get(feedback_type, FeedbackType.USER_CORRECTED)

            # Build labels from corrections
            price_label = price_correction

            pattern_labels = None
            if pattern_correction:
                pattern_labels = [
                    PATTERN_LABELS[p] for p in pattern_correction
                    if p in PATTERN_LABELS
                ]

            regime_label = None
            if regime_correction and regime_correction in REGIME_LABELS:
                regime_label = REGIME_LABELS[regime_correction]

            weight = FEEDBACK_WEIGHTS.get(fb_type, 1.0)

            sample = FeedbackSample(
                sample_id=f"fb_user_{prediction_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                prediction_id=prediction_id,
                symbol=symbol.upper(),
                timeframe=timeframe.upper(),
                feedback_type=fb_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                price_label=price_label,
                pattern_labels=pattern_labels,
                regime_label=regime_label,
                ohlcv_context=ohlcv_context,
                weight=weight,
            )

            self._samples.append(sample)
            self._cleanup_old_samples()
            self._save_buffer()

            logger.info(f"Added user feedback sample {sample.sample_id}")
            return sample

        except Exception as e:
            logger.error(f"Failed to add user feedback: {e}")
            return None

    def _cleanup_old_samples(self):
        """Remove old samples and enforce buffer size limit."""
        now = datetime.now(timezone.utc)
        max_age = timedelta(hours=self._max_age_hours)

        # Remove old samples
        self._samples = [
            s for s in self._samples
            if (now - datetime.fromisoformat(s.timestamp.replace('Z', '+00:00'))) < max_age
        ]

        # Enforce size limit (keep most recent)
        if len(self._samples) > self._max_buffer_size:
            # Sort by timestamp and keep newest
            self._samples.sort(key=lambda x: x.timestamp, reverse=True)
            self._samples = self._samples[:self._max_buffer_size]

    def get_training_batch(
        self,
        batch_size: int = 100,
        task_filter: Optional[str] = None,
    ) -> List[FeedbackSample]:
        """
        Get a batch of samples for training.

        Uses stratified sampling to ensure balanced representation.

        Args:
            batch_size: Maximum number of samples
            task_filter: Optional filter for specific task (price, patterns, regime)

        Returns:
            List of FeedbackSample
        """
        # Get unused samples
        unused = [s for s in self._samples if not s.used_for_training]

        # Apply task filter
        if task_filter == "price":
            unused = [s for s in unused if s.price_label is not None]
        elif task_filter == "patterns":
            unused = [s for s in unused if s.pattern_labels is not None]
        elif task_filter == "regime":
            unused = [s for s in unused if s.regime_label is not None]

        if len(unused) <= batch_size:
            return unused

        # Stratified sampling by feedback type
        by_type = {}
        for s in unused:
            t = s.feedback_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(s)

        # Calculate samples per type
        num_types = len(by_type)
        samples_per_type = batch_size // num_types if num_types > 0 else batch_size

        result = []
        for type_samples in by_type.values():
            random.shuffle(type_samples)
            result.extend(type_samples[:samples_per_type])

        # Fill remaining slots if needed
        remaining = batch_size - len(result)
        if remaining > 0:
            all_remaining = [s for s in unused if s not in result]
            random.shuffle(all_remaining)
            result.extend(all_remaining[:remaining])

        return result[:batch_size]

    def mark_as_used(self, sample_ids: List[str]):
        """Mark samples as used for training."""
        now = datetime.now(timezone.utc).isoformat()

        for sample in self._samples:
            if sample.sample_id in sample_ids:
                sample.used_for_training = True
                sample.training_timestamp = now

        self._save_buffer()
        logger.info(f"Marked {len(sample_ids)} samples as used")

    def clear_used_samples(self):
        """Clear all samples that have been used for training."""
        before = len(self._samples)
        self._samples = [s for s in self._samples if not s.used_for_training]
        after = len(self._samples)
        self._save_buffer()
        logger.info(f"Cleared {before - after} used samples")

    def get_statistics(self) -> BufferStatistics:
        """Get buffer statistics."""
        if not self._samples:
            return BufferStatistics()

        unused = [s for s in self._samples if not s.used_for_training]

        by_type = {}
        by_symbol = {}
        by_timeframe = {}

        samples_price = 0
        samples_pattern = 0
        samples_regime = 0
        total_accuracy = 0.0

        task_accuracy_sums = {"price": 0.0, "patterns": 0.0, "regime": 0.0}
        task_counts = {"price": 0, "patterns": 0, "regime": 0}

        for s in self._samples:
            # By type
            t = s.feedback_type.value
            by_type[t] = by_type.get(t, 0) + 1

            # By symbol
            by_symbol[s.symbol] = by_symbol.get(s.symbol, 0) + 1

            # By timeframe
            by_timeframe[s.timeframe] = by_timeframe.get(s.timeframe, 0) + 1

            # Task labels
            if s.price_label:
                samples_price += 1
                task_accuracy_sums["price"] += s.price_accuracy
                task_counts["price"] += 1
            if s.pattern_labels:
                samples_pattern += 1
                task_accuracy_sums["patterns"] += s.pattern_accuracy
                task_counts["patterns"] += 1
            if s.regime_label is not None:
                samples_regime += 1
                task_accuracy_sums["regime"] += s.regime_accuracy
                task_counts["regime"] += 1

            total_accuracy += s.overall_accuracy

        task_averages = {
            task: task_accuracy_sums[task] / task_counts[task] if task_counts[task] > 0 else 0.0
            for task in task_accuracy_sums
        }

        timestamps = [s.timestamp for s in self._samples]

        return BufferStatistics(
            total_samples=len(self._samples),
            unused_samples=len(unused),
            ready_for_training=len(unused) >= self._min_samples_for_training,
            by_feedback_type=by_type,
            by_symbol=by_symbol,
            by_timeframe=by_timeframe,
            samples_with_price_label=samples_price,
            samples_with_pattern_label=samples_pattern,
            samples_with_regime_label=samples_regime,
            average_accuracy=total_accuracy / len(self._samples) if self._samples else 0.0,
            task_averages=task_averages,
            oldest_sample=min(timestamps) if timestamps else None,
            newest_sample=max(timestamps) if timestamps else None,
        )

    def get_samples(
        self,
        unused_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get samples from buffer."""
        samples = self._samples
        if unused_only:
            samples = [s for s in samples if not s.used_for_training]

        return [s.to_dict() for s in samples[-limit:]]


# Global singleton
feedback_buffer_service = FeedbackBufferService()
