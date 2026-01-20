"""
Feedback Buffer Service - Collects pattern outcomes for incremental training.

Stores evaluated pattern outcomes as training samples for the self-learning system.
The buffer maintains balanced samples between successful and failed patterns.

Features:
- Collects outcomes from TCN Inference Service
- Converts outcomes to training samples (OHLCV sequences + labels)
- Stratified sampling for balanced training batches
- Persistence to disk for recovery after restarts
"""

import os
import json
import asyncio
import random
import uuid
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from enum import Enum
from loguru import logger


class FeedbackType(str, Enum):
    """Type of feedback signal."""
    OUTCOME_SUCCESS = "outcome_success"       # Pattern was correct
    OUTCOME_PARTIAL = "outcome_partial"       # Partial success
    OUTCOME_FAILED = "outcome_failed"         # Pattern was wrong
    OUTCOME_EXPIRED = "outcome_expired"       # No clear outcome
    CLAUDE_CONFIRMED = "claude_confirmed"     # Claude agreed
    CLAUDE_REJECTED = "claude_rejected"       # Claude disagreed
    MANUAL_POSITIVE = "manual_positive"       # Manual positive feedback
    MANUAL_NEGATIVE = "manual_negative"       # Manual negative feedback


# Weights for different feedback types (higher = more important for training)
FEEDBACK_WEIGHTS = {
    FeedbackType.OUTCOME_SUCCESS: 1.0,
    FeedbackType.OUTCOME_PARTIAL: 0.7,
    FeedbackType.OUTCOME_FAILED: 1.5,         # Learn more from failures
    FeedbackType.OUTCOME_EXPIRED: 0.3,        # Low signal
    FeedbackType.CLAUDE_CONFIRMED: 1.2,
    FeedbackType.CLAUDE_REJECTED: 1.5,        # Learn from rejections
    FeedbackType.MANUAL_POSITIVE: 1.3,
    FeedbackType.MANUAL_NEGATIVE: 1.5,
}


@dataclass
class FeedbackSample:
    """A feedback sample for incremental training."""

    # Identification
    sample_id: str
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str

    # OHLCV data (flattened for JSON serialization)
    ohlcv_sequence: List[List[float]]  # [[O,H,L,C,V], ...]

    # Labels and confidence
    original_confidence: float
    pattern_index: int                  # Index in PATTERN_CLASSES (0-15)

    # Feedback
    feedback_type: str
    outcome_score: float               # -1.0 to 1.0
    sample_weight: float = 1.0

    # Metrics from outcome
    max_favorable_percent: float = 0.0
    max_adverse_percent: float = 0.0

    # External validation
    claude_validated: bool = False
    claude_agreed: Optional[bool] = None
    claude_confidence: Optional[float] = None

    # Metadata
    created_at: str = ""
    used_for_training: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackSample":
        """Create from dictionary."""
        return cls(**data)

    def to_training_data(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert to training data format.

        Returns:
            Tuple of (X, y, weight) where:
            - X: OHLCV sequence as numpy array (seq_len, 5)
            - y: Multi-label target vector (16,)
            - weight: Sample weight for loss function
        """
        # Convert OHLCV to numpy
        X = np.array(self.ohlcv_sequence, dtype=np.float32)

        # Create label vector (16 pattern classes)
        y = np.zeros(16, dtype=np.float32)

        # Set the target based on feedback type
        if self.feedback_type in [
            FeedbackType.OUTCOME_SUCCESS.value,
            FeedbackType.OUTCOME_PARTIAL.value,
            FeedbackType.CLAUDE_CONFIRMED.value,
            FeedbackType.MANUAL_POSITIVE.value
        ]:
            # Positive feedback: reinforce the pattern
            y[self.pattern_index] = 1.0
        elif self.feedback_type in [
            FeedbackType.OUTCOME_FAILED.value,
            FeedbackType.CLAUDE_REJECTED.value,
            FeedbackType.MANUAL_NEGATIVE.value
        ]:
            # Negative feedback: set to 0 (don't detect this pattern)
            y[self.pattern_index] = 0.0
        else:
            # Expired/unclear: use original confidence as soft label
            y[self.pattern_index] = self.original_confidence * 0.5

        return X, y, self.sample_weight


@dataclass
class BufferStatistics:
    """Statistics about the feedback buffer."""

    total_samples: int = 0
    unused_samples: int = 0
    used_samples: int = 0

    # By feedback type
    by_feedback_type: Dict[str, int] = field(default_factory=dict)

    # By pattern type
    by_pattern_type: Dict[str, int] = field(default_factory=dict)

    # By outcome score range
    positive_outcomes: int = 0    # score > 0
    negative_outcomes: int = 0    # score < 0
    neutral_outcomes: int = 0     # score == 0

    # Buffer health
    oldest_sample_age_hours: float = 0.0
    avg_sample_weight: float = 0.0
    ready_for_training: bool = False
    min_samples_for_training: int = 100

    last_update: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class FeedbackBufferService:
    """
    Service for collecting and managing feedback samples.

    The buffer collects outcomes from the TCN service and converts them
    into training samples for incremental learning.
    """

    # Pattern classes (must match TCN model)
    PATTERN_CLASSES = [
        "head_and_shoulders",
        "inverse_head_and_shoulders",
        "double_top",
        "double_bottom",
        "triple_top",
        "triple_bottom",
        "ascending_triangle",
        "descending_triangle",
        "symmetrical_triangle",
        "bull_flag",
        "bear_flag",
        "cup_and_handle",
        "rising_wedge",
        "falling_wedge",
        "channel_up",
        "channel_down"
    ]

    def __init__(
        self,
        buffer_file: str = "data/tcn_feedback_buffer.json",
        max_size: int = 10000,
        max_age_hours: int = 168,  # 7 days
        min_samples_for_training: int = 100,
    ):
        """
        Initialize the Feedback Buffer Service.

        Args:
            buffer_file: Path to JSON file for persistence
            max_size: Maximum number of samples to keep
            max_age_hours: Maximum age of samples in hours
            min_samples_for_training: Minimum samples needed for training
        """
        self._buffer_file = Path(buffer_file)
        self._max_size = max_size
        self._max_age_hours = max_age_hours
        self._min_samples = min_samples_for_training

        self._samples: List[FeedbackSample] = []
        self._lock = asyncio.Lock()

        # Ensure data directory exists
        self._buffer_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing buffer
        self._load_buffer()

        logger.info(
            f"Feedback Buffer initialized with {len(self._samples)} samples "
            f"(min for training: {self._min_samples})"
        )

    def _load_buffer(self) -> None:
        """Load buffer from JSON file."""
        if self._buffer_file.exists():
            try:
                with open(self._buffer_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._samples = [
                        FeedbackSample.from_dict(s) for s in data.get("samples", [])
                    ]
                logger.info(f"Loaded {len(self._samples)} samples from buffer file")
            except Exception as e:
                logger.error(f"Error loading buffer: {e}")
                self._samples = []
        else:
            self._samples = []

    def _save_buffer(self) -> None:
        """Save buffer to JSON file."""
        try:
            data = {
                "samples": [s.to_dict() for s in self._samples],
                "last_save": datetime.now(timezone.utc).isoformat()
            }
            with open(self._buffer_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving buffer: {e}")

    async def _save_buffer_async(self) -> None:
        """Save buffer asynchronously."""
        await asyncio.to_thread(self._save_buffer)

    def _get_pattern_index(self, pattern_type: str) -> int:
        """Get the index of a pattern type in PATTERN_CLASSES."""
        try:
            return self.PATTERN_CLASSES.index(pattern_type)
        except ValueError:
            logger.warning(f"Unknown pattern type: {pattern_type}")
            return -1

    def _cleanup_old_samples(self) -> int:
        """Remove old and excess samples. Returns number removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_age_hours)
        original_count = len(self._samples)

        # Remove old samples
        self._samples = [
            s for s in self._samples
            if datetime.fromisoformat(s.created_at.replace("Z", "+00:00")) > cutoff
        ]

        # Remove used samples if over max size
        if len(self._samples) > self._max_size:
            # Keep unused samples, remove oldest used samples first
            unused = [s for s in self._samples if not s.used_for_training]
            used = [s for s in self._samples if s.used_for_training]

            # Sort used by creation time (oldest first) and remove excess
            used.sort(key=lambda x: x.created_at)
            excess = len(self._samples) - self._max_size

            if excess > 0:
                used = used[excess:]

            self._samples = unused + used

        removed = original_count - len(self._samples)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old/excess samples")

        return removed

    async def add_feedback_from_outcome(
        self,
        pattern_id: str,
        symbol: str,
        timeframe: str,
        pattern_type: str,
        direction: str,
        original_confidence: float,
        outcome_status: str,
        outcome_score: float,
        max_favorable_percent: float,
        max_adverse_percent: float,
        ohlcv_data: List[dict],
        claude_validated: bool = False,
        claude_agreed: Optional[bool] = None,
        claude_confidence: Optional[float] = None,
    ) -> Optional[FeedbackSample]:
        """
        Add a feedback sample from a completed pattern outcome.

        Args:
            pattern_id: Unique pattern identifier
            symbol: Trading symbol
            timeframe: Pattern timeframe
            pattern_type: Type of pattern
            direction: Pattern direction (bullish/bearish)
            original_confidence: Original TCN confidence
            outcome_status: Final outcome status (success/partial/failed/expired)
            outcome_score: Score from -1 to 1
            max_favorable_percent: Maximum favorable movement %
            max_adverse_percent: Maximum adverse movement %
            ohlcv_data: OHLCV candles used for detection
            claude_validated: Whether Claude validated
            claude_agreed: Whether Claude agreed (if validated)
            claude_confidence: Claude's confidence (if validated)

        Returns:
            The created sample, or None if invalid
        """
        # Validate pattern type
        pattern_index = self._get_pattern_index(pattern_type)
        if pattern_index < 0:
            logger.warning(f"Cannot add feedback: unknown pattern type {pattern_type}")
            return None

        # Validate OHLCV data
        if not ohlcv_data or len(ohlcv_data) < 50:
            logger.warning(f"Cannot add feedback: insufficient OHLCV data ({len(ohlcv_data) if ohlcv_data else 0})")
            return None

        # Convert OHLCV dict list to sequence
        ohlcv_sequence = []
        for candle in ohlcv_data:
            try:
                ohlcv_sequence.append([
                    float(candle.get("open", 0)),
                    float(candle.get("high", 0)),
                    float(candle.get("low", 0)),
                    float(candle.get("close", 0)),
                    float(candle.get("volume", 0)),
                ])
            except (ValueError, TypeError):
                continue

        if len(ohlcv_sequence) < 50:
            logger.warning(f"Cannot add feedback: invalid OHLCV data")
            return None

        # Determine feedback type
        feedback_type = self._determine_feedback_type(
            outcome_status, claude_validated, claude_agreed
        )

        # Calculate sample weight
        base_weight = FEEDBACK_WEIGHTS.get(feedback_type, 1.0)
        # Adjust weight by confidence calibration
        if outcome_status in ["success", "partial"]:
            # Good outcome: weight by how close confidence was to 1.0
            base_weight *= (0.5 + original_confidence * 0.5)
        elif outcome_status == "failed":
            # Bad outcome: weight more if confidence was high (overconfident)
            base_weight *= (0.5 + original_confidence)

        # Create sample
        sample = FeedbackSample(
            sample_id=str(uuid.uuid4()),
            pattern_id=pattern_id,
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            direction=direction,
            ohlcv_sequence=ohlcv_sequence,
            original_confidence=original_confidence,
            pattern_index=pattern_index,
            feedback_type=feedback_type.value,
            outcome_score=outcome_score,
            sample_weight=round(base_weight, 4),
            max_favorable_percent=max_favorable_percent,
            max_adverse_percent=max_adverse_percent,
            claude_validated=claude_validated,
            claude_agreed=claude_agreed,
            claude_confidence=claude_confidence,
        )

        async with self._lock:
            self._samples.append(sample)
            self._cleanup_old_samples()
            await self._save_buffer_async()

        logger.debug(
            f"Added feedback sample: {pattern_type} ({feedback_type.value}) "
            f"weight={base_weight:.2f}"
        )

        return sample

    def _determine_feedback_type(
        self,
        outcome_status: str,
        claude_validated: bool,
        claude_agreed: Optional[bool]
    ) -> FeedbackType:
        """Determine the feedback type based on outcome and validation."""
        # Claude validation takes precedence if available
        if claude_validated:
            if claude_agreed:
                return FeedbackType.CLAUDE_CONFIRMED
            else:
                return FeedbackType.CLAUDE_REJECTED

        # Otherwise use outcome status
        status_map = {
            "success": FeedbackType.OUTCOME_SUCCESS,
            "partial": FeedbackType.OUTCOME_PARTIAL,
            "failed": FeedbackType.OUTCOME_FAILED,
            "invalidated": FeedbackType.OUTCOME_FAILED,
            "expired": FeedbackType.OUTCOME_EXPIRED,
        }
        return status_map.get(outcome_status, FeedbackType.OUTCOME_EXPIRED)

    async def add_manual_feedback(
        self,
        pattern_id: str,
        symbol: str,
        timeframe: str,
        pattern_type: str,
        direction: str,
        original_confidence: float,
        is_positive: bool,
        ohlcv_data: List[dict],
        reason: str = "",
    ) -> Optional[FeedbackSample]:
        """
        Add manual feedback for a pattern.

        Args:
            pattern_id: Pattern identifier
            symbol: Trading symbol
            timeframe: Pattern timeframe
            pattern_type: Type of pattern
            direction: Pattern direction
            original_confidence: Original confidence
            is_positive: Whether feedback is positive
            ohlcv_data: OHLCV data
            reason: Reason for feedback

        Returns:
            The created sample, or None if invalid
        """
        return await self.add_feedback_from_outcome(
            pattern_id=pattern_id,
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            direction=direction,
            original_confidence=original_confidence,
            outcome_status="manual",
            outcome_score=1.0 if is_positive else -1.0,
            max_favorable_percent=0.0,
            max_adverse_percent=0.0,
            ohlcv_data=ohlcv_data,
        )

    def get_training_batch(
        self,
        batch_size: Optional[int] = None,
        stratified: bool = True,
    ) -> Optional[List[FeedbackSample]]:
        """
        Get a batch of samples for training.

        Args:
            batch_size: Number of samples (default: min_samples_for_training)
            stratified: Whether to use stratified sampling

        Returns:
            List of samples, or None if not enough samples
        """
        if batch_size is None:
            batch_size = self._min_samples

        # Get unused samples
        unused = [s for s in self._samples if not s.used_for_training]

        if len(unused) < batch_size:
            logger.info(f"Not enough unused samples: {len(unused)} < {batch_size}")
            return None

        if stratified:
            batch = self._stratified_sample(unused, batch_size)
        else:
            # Random sampling
            batch = random.sample(unused, batch_size)

        return batch

    def _stratified_sample(
        self,
        samples: List[FeedbackSample],
        n: int
    ) -> List[FeedbackSample]:
        """
        Stratified sampling to maintain balance.

        Ensures roughly equal representation of:
        - Positive vs negative outcomes
        - Different pattern types
        """
        # Group by outcome (positive/negative)
        positive = [s for s in samples if s.outcome_score > 0]
        negative = [s for s in samples if s.outcome_score <= 0]

        result = []

        # Sample equally from positive and negative
        pos_count = min(len(positive), n // 2)
        neg_count = min(len(negative), n - pos_count)

        if positive:
            result.extend(random.sample(positive, pos_count))
        if negative:
            result.extend(random.sample(negative, neg_count))

        # Fill remaining if needed
        remaining = n - len(result)
        if remaining > 0:
            available = [s for s in samples if s not in result]
            if available:
                result.extend(random.sample(
                    available,
                    min(remaining, len(available))
                ))

        random.shuffle(result)
        return result

    def mark_as_used(self, sample_ids: List[str]) -> int:
        """
        Mark samples as used for training.

        Args:
            sample_ids: List of sample IDs to mark

        Returns:
            Number of samples marked
        """
        marked = 0
        for sample in self._samples:
            if sample.sample_id in sample_ids:
                sample.used_for_training = True
                marked += 1

        if marked > 0:
            self._save_buffer()
            logger.info(f"Marked {marked} samples as used for training")

        return marked

    def clear_used_samples(self) -> int:
        """
        Remove all samples that have been used for training.

        Returns:
            Number of samples removed
        """
        original_count = len(self._samples)
        self._samples = [s for s in self._samples if not s.used_for_training]
        removed = original_count - len(self._samples)

        if removed > 0:
            self._save_buffer()
            logger.info(f"Cleared {removed} used samples")

        return removed

    def clear_all(self) -> int:
        """
        Clear all samples from the buffer.

        Returns:
            Number of samples removed
        """
        count = len(self._samples)
        self._samples = []
        self._save_buffer()
        logger.info(f"Cleared all {count} samples from buffer")
        return count

    def get_statistics(self) -> BufferStatistics:
        """Get buffer statistics."""
        stats = BufferStatistics()
        stats.total_samples = len(self._samples)
        stats.unused_samples = sum(1 for s in self._samples if not s.used_for_training)
        stats.used_samples = stats.total_samples - stats.unused_samples
        stats.min_samples_for_training = self._min_samples
        stats.ready_for_training = stats.unused_samples >= self._min_samples

        if not self._samples:
            stats.last_update = datetime.now(timezone.utc).isoformat()
            return stats

        # By feedback type
        for sample in self._samples:
            ft = sample.feedback_type
            stats.by_feedback_type[ft] = stats.by_feedback_type.get(ft, 0) + 1

            pt = sample.pattern_type
            stats.by_pattern_type[pt] = stats.by_pattern_type.get(pt, 0) + 1

            if sample.outcome_score > 0:
                stats.positive_outcomes += 1
            elif sample.outcome_score < 0:
                stats.negative_outcomes += 1
            else:
                stats.neutral_outcomes += 1

        # Calculate oldest sample age
        oldest = min(self._samples, key=lambda x: x.created_at)
        oldest_time = datetime.fromisoformat(oldest.created_at.replace("Z", "+00:00"))
        stats.oldest_sample_age_hours = round(
            (datetime.now(timezone.utc) - oldest_time).total_seconds() / 3600, 2
        )

        # Average weight
        stats.avg_sample_weight = round(
            sum(s.sample_weight for s in self._samples) / len(self._samples), 4
        )

        stats.last_update = datetime.now(timezone.utc).isoformat()

        return stats

    def get_samples(
        self,
        limit: int = 100,
        unused_only: bool = False,
        pattern_type: Optional[str] = None,
    ) -> List[FeedbackSample]:
        """
        Get samples with optional filtering.

        Args:
            limit: Maximum number of samples
            unused_only: Only return unused samples
            pattern_type: Filter by pattern type

        Returns:
            List of samples
        """
        samples = self._samples.copy()

        if unused_only:
            samples = [s for s in samples if not s.used_for_training]

        if pattern_type:
            samples = [s for s in samples if s.pattern_type == pattern_type]

        # Sort by creation time (newest first)
        samples.sort(key=lambda x: x.created_at, reverse=True)

        return samples[:limit]


# Singleton instance
feedback_buffer_service = FeedbackBufferService()
