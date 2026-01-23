"""
Feedback Buffer Service for Candlestick Pattern Self-Learning.

Collects and manages feedback samples from pattern outcome tracking
for incremental model training.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
from loguru import logger


class FeedbackType(Enum):
    """Types of feedback for candlestick patterns."""
    OUTCOME_SUCCESS = "outcome_success"
    OUTCOME_PARTIAL = "outcome_partial"
    OUTCOME_FAILED = "outcome_failed"
    OUTCOME_EXPIRED = "outcome_expired"
    CLAUDE_CONFIRMED = "claude_confirmed"
    CLAUDE_REJECTED = "claude_rejected"
    MANUAL_POSITIVE = "manual_positive"
    MANUAL_NEGATIVE = "manual_negative"


# Weights for different feedback types (failures are weighted higher to learn from mistakes)
FEEDBACK_WEIGHTS = {
    FeedbackType.OUTCOME_SUCCESS: 1.0,
    FeedbackType.OUTCOME_PARTIAL: 0.7,
    FeedbackType.OUTCOME_FAILED: 1.5,  # Learn more from failures
    FeedbackType.OUTCOME_EXPIRED: 0.3,
    FeedbackType.CLAUDE_CONFIRMED: 1.2,
    FeedbackType.CLAUDE_REJECTED: 1.5,  # High weight for AI rejections
    FeedbackType.MANUAL_POSITIVE: 1.0,
    FeedbackType.MANUAL_NEGATIVE: 1.3,
}


# Pattern type to label index mapping
PATTERN_LABELS = {
    "hammer": 0, "inverted_hammer": 1, "shooting_star": 2, "hanging_man": 3,
    "doji": 4, "dragonfly_doji": 5, "gravestone_doji": 6, "spinning_top": 7,
    "bullish_engulfing": 8, "bearish_engulfing": 9,
    "piercing_line": 10, "dark_cloud_cover": 11,
    "bullish_harami": 12, "bearish_harami": 13, "harami_cross": 14,
    "morning_star": 15, "evening_star": 16,
    "three_white_soldiers": 17, "three_black_crows": 18,
    "three_inside_up": 19, "three_inside_down": 20,
}


@dataclass
class FeedbackSample:
    """A feedback sample for model training."""
    sample_id: str
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    feedback_type: FeedbackType

    # OHLC data (flattened for JSON)
    ohlc_data: List[Dict[str, Any]] = field(default_factory=list)

    # Labels
    pattern_label: int = 0
    direction_label: int = 0  # 0=neutral, 1=bullish, 2=bearish

    # Outcome metrics
    outcome_score: float = 0.0  # -1 to 1
    original_confidence: float = 0.5

    # Metadata
    weight: float = 1.0
    timestamp: str = ""
    used_in_training: bool = False
    training_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["feedback_type"] = self.feedback_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackSample":
        """Create from dictionary."""
        if "feedback_type" in data:
            data["feedback_type"] = FeedbackType(data["feedback_type"])
        return cls(**data)

    def to_training_data(self) -> Tuple[List[List[float]], int, float]:
        """
        Convert to training data format.

        Returns:
            Tuple of (X: OHLC sequence, y: pattern label, weight: sample weight)
        """
        # Normalize OHLC data
        X = []
        for candle in self.ohlc_data:
            X.append([
                float(candle.get("open", 0)),
                float(candle.get("high", 0)),
                float(candle.get("low", 0)),
                float(candle.get("close", 0)),
            ])

        return X, self.pattern_label, self.weight


@dataclass
class BufferStatistics:
    """Statistics about the feedback buffer."""
    total_samples: int = 0
    unused_samples: int = 0
    samples_by_type: Dict[str, int] = field(default_factory=dict)
    samples_by_pattern: Dict[str, int] = field(default_factory=dict)
    positive_samples: int = 0
    negative_samples: int = 0
    neutral_samples: int = 0
    oldest_sample_age_hours: float = 0.0
    average_weight: float = 0.0
    ready_for_training: bool = False
    min_samples_required: int = 100


class FeedbackBufferService:
    """
    Manages feedback samples for self-learning.

    Features:
    - Stores feedback from outcome tracking and Claude validation
    - Provides balanced batches for training
    - Tracks sample usage and age
    - Thread-safe operations
    """

    def __init__(self, data_file: str = None):
        data_dir = os.getenv("DATA_DIR", "/app/data")
        if data_file is None:
            data_file = os.path.join(data_dir, "candlestick_feedback_buffer.json")

        self._data_file = Path(data_file)
        self._samples: Dict[str, FeedbackSample] = {}
        self._max_samples = int(os.getenv("FEEDBACK_BUFFER_MAX_SAMPLES", "10000"))
        self._max_age_days = int(os.getenv("FEEDBACK_BUFFER_MAX_AGE_DAYS", "7"))
        self._min_samples_for_training = int(os.getenv("MIN_SAMPLES_FOR_TRAINING", "100"))

        self._load_buffer()

        logger.info(f"FeedbackBufferService initialized - {len(self._samples)} samples loaded")

    def _load_buffer(self):
        """Load buffer from file."""
        try:
            if self._data_file.exists():
                with open(self._data_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        sample = FeedbackSample.from_dict(item)
                        self._samples[sample.sample_id] = sample
                logger.info(f"Loaded {len(self._samples)} feedback samples")
        except Exception as e:
            logger.error(f"Failed to load feedback buffer: {e}")

    def _save_buffer(self):
        """Save buffer to file."""
        try:
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._data_file, 'w') as f:
                data = [s.to_dict() for s in self._samples.values()]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback buffer: {e}")

    def _cleanup_old_samples(self):
        """Remove old and excess samples."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._max_age_days)
        cutoff_str = cutoff.isoformat()

        # Remove old samples
        self._samples = {
            k: v for k, v in self._samples.items()
            if v.timestamp > cutoff_str
        }

        # Remove excess samples (keep most recent)
        if len(self._samples) > self._max_samples:
            sorted_samples = sorted(
                self._samples.items(),
                key=lambda x: x[1].timestamp,
                reverse=True
            )
            self._samples = dict(sorted_samples[:self._max_samples])

    def add_feedback_from_outcome(self, outcome_data: Dict[str, Any]) -> Optional[str]:
        """
        Add feedback from a completed pattern outcome.

        Args:
            outcome_data: Outcome data from outcome tracker

        Returns:
            Sample ID if added, None otherwise
        """
        try:
            pattern_id = outcome_data.get("pattern_id", "")
            pattern_type = outcome_data.get("pattern_type", "").lower()
            outcome_status = outcome_data.get("outcome_status", "")

            # Validate pattern type
            if pattern_type not in PATTERN_LABELS:
                logger.warning(f"Unknown pattern type: {pattern_type}")
                return None

            # Determine feedback type
            feedback_type_map = {
                "success": FeedbackType.OUTCOME_SUCCESS,
                "partial": FeedbackType.OUTCOME_PARTIAL,
                "failed": FeedbackType.OUTCOME_FAILED,
                "expired": FeedbackType.OUTCOME_EXPIRED,
            }
            feedback_type = feedback_type_map.get(outcome_status, FeedbackType.OUTCOME_EXPIRED)

            # Get OHLC data
            ohlc_data = outcome_data.get("ohlc_data", [])
            if not ohlc_data or len(ohlc_data) < 3:
                logger.warning(f"Insufficient OHLC data for {pattern_id}")
                return None

            # Calculate outcome score (-1 to 1)
            max_favorable = outcome_data.get("max_favorable_percent", 0)
            max_adverse = outcome_data.get("max_adverse_percent", 0)

            if outcome_status == "success":
                outcome_score = min(1.0, max_favorable / 2.0)  # Cap at 1.0
            elif outcome_status == "partial":
                outcome_score = max_favorable / 4.0  # Partial credit
            elif outcome_status == "failed":
                outcome_score = -min(1.0, max_adverse / 2.0)
            else:
                outcome_score = 0.0

            # Calculate weight
            base_weight = FEEDBACK_WEIGHTS[feedback_type]
            confidence = outcome_data.get("confidence", 0.5)

            if outcome_status in ["success", "partial"]:
                weight = base_weight * (0.5 + confidence * 0.5)
            elif outcome_status == "failed":
                weight = base_weight * (0.5 + confidence)  # Penalize overconfidence
            else:
                weight = base_weight

            # Direction label
            direction = outcome_data.get("direction", "neutral")
            direction_label = {"neutral": 0, "bullish": 1, "bearish": 2}.get(direction, 0)

            # Create sample
            sample_id = f"fb_{pattern_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

            sample = FeedbackSample(
                sample_id=sample_id,
                pattern_id=pattern_id,
                symbol=outcome_data.get("symbol", ""),
                timeframe=outcome_data.get("timeframe", "H1"),
                pattern_type=pattern_type,
                feedback_type=feedback_type,
                ohlc_data=ohlc_data,
                pattern_label=PATTERN_LABELS[pattern_type],
                direction_label=direction_label,
                outcome_score=outcome_score,
                original_confidence=confidence,
                weight=weight,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            self._samples[sample_id] = sample
            self._cleanup_old_samples()
            self._save_buffer()

            logger.debug(f"Added feedback sample {sample_id} ({feedback_type.value})")
            return sample_id

        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return None

    def add_claude_feedback(
        self,
        pattern_data: Dict[str, Any],
        is_confirmed: bool,
        confidence: float = 0.8
    ) -> Optional[str]:
        """Add feedback from Claude validation."""
        try:
            pattern_id = pattern_data.get("id", pattern_data.get("pattern_id", ""))
            pattern_type = pattern_data.get("pattern_type", "").lower()

            if pattern_type not in PATTERN_LABELS:
                return None

            feedback_type = FeedbackType.CLAUDE_CONFIRMED if is_confirmed else FeedbackType.CLAUDE_REJECTED
            outcome_score = 1.0 if is_confirmed else -1.0

            # Get OHLC data from pattern
            ohlc_context = pattern_data.get("ohlc_context", {})
            ohlc_data = ohlc_context.get("candles", []) if ohlc_context else []

            if not ohlc_data:
                return None

            direction = pattern_data.get("direction", "neutral")
            direction_label = {"neutral": 0, "bullish": 1, "bearish": 2}.get(direction, 0)

            sample_id = f"claude_{pattern_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

            sample = FeedbackSample(
                sample_id=sample_id,
                pattern_id=pattern_id,
                symbol=pattern_data.get("symbol", ""),
                timeframe=pattern_data.get("timeframe", "H1"),
                pattern_type=pattern_type,
                feedback_type=feedback_type,
                ohlc_data=ohlc_data,
                pattern_label=PATTERN_LABELS[pattern_type],
                direction_label=direction_label,
                outcome_score=outcome_score,
                original_confidence=confidence,
                weight=FEEDBACK_WEIGHTS[feedback_type],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            self._samples[sample_id] = sample
            self._cleanup_old_samples()
            self._save_buffer()

            return sample_id

        except Exception as e:
            logger.error(f"Failed to add Claude feedback: {e}")
            return None

    def get_training_batch(
        self,
        batch_size: int = 100,
        balanced: bool = True
    ) -> List[FeedbackSample]:
        """
        Get a batch of samples for training.

        Args:
            batch_size: Number of samples to return
            balanced: Whether to balance positive/negative samples

        Returns:
            List of FeedbackSample objects
        """
        unused = [s for s in self._samples.values() if not s.used_in_training]

        if not unused:
            return []

        if balanced:
            return self._stratified_sample(unused, batch_size)

        # Simple random sample
        return random.sample(unused, min(batch_size, len(unused)))

    def _stratified_sample(
        self,
        samples: List[FeedbackSample],
        batch_size: int
    ) -> List[FeedbackSample]:
        """Get a stratified sample balancing positive and negative outcomes."""
        positive = [s for s in samples if s.outcome_score > 0]
        negative = [s for s in samples if s.outcome_score < 0]
        neutral = [s for s in samples if s.outcome_score == 0]

        result = []
        target_per_group = batch_size // 3

        # Add samples from each group
        for group, target in [
            (positive, target_per_group),
            (negative, target_per_group),
            (neutral, batch_size - 2 * target_per_group)
        ]:
            if group:
                result.extend(random.sample(group, min(target, len(group))))

        # Fill remaining slots if needed
        remaining = batch_size - len(result)
        if remaining > 0:
            available = [s for s in samples if s not in result]
            if available:
                result.extend(random.sample(available, min(remaining, len(available))))

        return result

    def mark_as_used(self, sample_ids: List[str]):
        """Mark samples as used in training."""
        now = datetime.now(timezone.utc).isoformat()
        for sample_id in sample_ids:
            if sample_id in self._samples:
                self._samples[sample_id].used_in_training = True
                self._samples[sample_id].training_timestamp = now

        self._save_buffer()

    def clear_used_samples(self):
        """Remove samples that have been used in training."""
        self._samples = {
            k: v for k, v in self._samples.items()
            if not v.used_in_training
        }
        self._save_buffer()

    def clear_all(self):
        """Clear all samples."""
        self._samples = {}
        self._save_buffer()

    def get_statistics(self) -> BufferStatistics:
        """Get buffer statistics."""
        if not self._samples:
            return BufferStatistics(
                min_samples_required=self._min_samples_for_training
            )

        now = datetime.now(timezone.utc)
        by_type: Dict[str, int] = {}
        by_pattern: Dict[str, int] = {}
        positive = 0
        negative = 0
        neutral = 0
        weights = []
        unused = 0
        oldest_age = 0.0

        for sample in self._samples.values():
            # By type
            ft = sample.feedback_type.value
            by_type[ft] = by_type.get(ft, 0) + 1

            # By pattern
            by_pattern[sample.pattern_type] = by_pattern.get(sample.pattern_type, 0) + 1

            # Outcome category
            if sample.outcome_score > 0:
                positive += 1
            elif sample.outcome_score < 0:
                negative += 1
            else:
                neutral += 1

            weights.append(sample.weight)

            if not sample.used_in_training:
                unused += 1

            # Age
            try:
                ts = datetime.fromisoformat(sample.timestamp.replace('Z', '+00:00'))
                age = (now - ts).total_seconds() / 3600
                if age > oldest_age:
                    oldest_age = age
            except:
                pass

        return BufferStatistics(
            total_samples=len(self._samples),
            unused_samples=unused,
            samples_by_type=by_type,
            samples_by_pattern=by_pattern,
            positive_samples=positive,
            negative_samples=negative,
            neutral_samples=neutral,
            oldest_sample_age_hours=round(oldest_age, 1),
            average_weight=round(sum(weights) / len(weights), 2) if weights else 0,
            ready_for_training=unused >= self._min_samples_for_training,
            min_samples_required=self._min_samples_for_training,
        )

    def is_ready_for_training(self) -> bool:
        """Check if buffer has enough samples for training."""
        unused = sum(1 for s in self._samples.values() if not s.used_in_training)
        return unused >= self._min_samples_for_training


# Global singleton
feedback_buffer_service = FeedbackBufferService()
