"""
Outcome Tracker Service - Tracks pattern outcomes for self-learning.

Monitors detected patterns and tracks their actual market outcomes:
- Did the price reach the target?
- Was the invalidation level hit?
- What was the actual price movement?

This data is used for:
1. Performance metrics calculation
2. Feedback generation for incremental training
3. Drift detection
"""

import os
import json
import asyncio
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Callable, Awaitable
from enum import Enum
from loguru import logger

from .tcn_pattern_history_service import TCNPatternHistoryEntry


class OutcomeStatus(str, Enum):
    """Status of a pattern outcome."""
    PENDING = "pending"           # Still being tracked
    SUCCESS = "success"           # Target reached or favorable movement
    PARTIAL = "partial"           # Partial success (30-60% of target)
    FAILED = "failed"             # Invalidation hit or wrong direction
    INVALIDATED = "invalidated"   # External invalidation (gap, news)
    EXPIRED = "expired"           # Tracking period ended without clear outcome


@dataclass
class PatternOutcome:
    """Outcome tracking for a detected pattern."""

    # Pattern identification
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str                      # bullish, bearish, neutral
    category: str                       # reversal, continuation, trend

    # Detection context
    detected_at: str                    # ISO 8601 UTC
    price_at_detection: float
    confidence: float

    # Targets
    price_target: Optional[float] = None
    invalidation_level: Optional[float] = None

    # Tracking state
    status: str = OutcomeStatus.PENDING.value
    tracking_started: Optional[str] = None
    tracking_ends: Optional[str] = None

    # Price tracking
    current_price: Optional[float] = None
    max_favorable_price: Optional[float] = None    # Best price in expected direction
    max_adverse_price: Optional[float] = None      # Worst price against direction
    last_update: Optional[str] = None

    # Calculated metrics
    max_favorable_percent: float = 0.0    # Max favorable move in %
    max_adverse_percent: float = 0.0      # Max adverse move in %
    current_move_percent: float = 0.0     # Current move from detection price

    # Outcome details
    target_reached: bool = False
    target_reached_at: Optional[str] = None
    invalidation_hit: bool = False
    invalidation_hit_at: Optional[str] = None
    outcome_reason: str = ""

    # External validation
    claude_validated: bool = False
    claude_agreed: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PatternOutcome":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_pattern_entry(cls, entry: TCNPatternHistoryEntry) -> "PatternOutcome":
        """Create outcome tracker from a pattern history entry."""
        return cls(
            pattern_id=entry.id,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            pattern_type=entry.pattern_type,
            direction=entry.direction,
            category=entry.category,
            detected_at=entry.detected_at,
            price_at_detection=entry.price_at_detection,
            confidence=entry.confidence,
            price_target=entry.price_target,
            invalidation_level=entry.invalidation_level,
            tracking_started=datetime.now(timezone.utc).isoformat()
        )


@dataclass
class OutcomeStatistics:
    """Aggregated outcome statistics."""

    total_tracked: int = 0
    total_completed: int = 0
    pending: int = 0

    # Outcome counts
    success_count: int = 0
    partial_count: int = 0
    failed_count: int = 0
    invalidated_count: int = 0
    expired_count: int = 0

    # Rates
    success_rate: float = 0.0
    partial_rate: float = 0.0
    failure_rate: float = 0.0

    # Performance metrics
    avg_favorable_percent: float = 0.0
    avg_adverse_percent: float = 0.0
    profit_factor: float = 0.0           # sum(favorable) / sum(adverse)

    # By category
    by_pattern_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_direction: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_timeframe: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Timing
    last_update: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class OutcomeTrackerService:
    """
    Service for tracking pattern outcomes.

    Features:
    - Automatic tracking of detected patterns
    - Periodic price updates via Data Service
    - Outcome evaluation based on targets and invalidation levels
    - Statistics calculation for performance monitoring
    - Callback support for new outcomes (for feedback buffer integration)
    """

    # Evaluation timeframes per pattern timeframe
    EVALUATION_PERIODS = {
        "M15": timedelta(hours=12),
        "M30": timedelta(hours=24),
        "1h": timedelta(hours=48),
        "H1": timedelta(hours=48),
        "4h": timedelta(hours=72),
        "H4": timedelta(hours=72),
        "1d": timedelta(days=14),
        "D1": timedelta(days=14),
        "W1": timedelta(days=30),
        "MN": timedelta(days=90),
    }

    # Default evaluation period
    DEFAULT_EVALUATION_PERIOD = timedelta(hours=72)

    def __init__(
        self,
        outcomes_file: str = "data/tcn/tcn_pattern_outcomes.json",
        max_entries: int = 5000,
        max_age_days: int = 30,
        update_interval_seconds: int = 900,  # 15 minutes
    ):
        """
        Initialize the Outcome Tracker Service.

        Args:
            outcomes_file: Path to JSON file for persistence
            max_entries: Maximum number of outcomes to keep
            max_age_days: Maximum age of outcomes in days
            update_interval_seconds: Interval for price updates
        """
        self._outcomes_file = Path(outcomes_file)
        self._max_entries = max_entries
        self._max_age_days = max_age_days
        self._update_interval = update_interval_seconds

        self._outcomes: Dict[str, PatternOutcome] = {}  # pattern_id -> outcome
        self._completed_outcomes: List[PatternOutcome] = []

        self._update_task: Optional[asyncio.Task] = None
        self._update_running = False
        self._last_update: Optional[datetime] = None

        # Callbacks for outcome completion (for feedback buffer integration)
        self._outcome_callbacks: List[Callable[[PatternOutcome], Awaitable[None]]] = []

        # Ensure data directory exists
        self._outcomes_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self._load_outcomes()

        logger.info(
            f"Outcome Tracker initialized: {len(self._outcomes)} active, "
            f"{len(self._completed_outcomes)} completed"
        )

    def _load_outcomes(self) -> None:
        """Load outcomes from JSON file."""
        if self._outcomes_file.exists():
            try:
                with open(self._outcomes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Load active outcomes
                    for outcome_data in data.get("active", []):
                        outcome = PatternOutcome.from_dict(outcome_data)
                        self._outcomes[outcome.pattern_id] = outcome

                    # Load completed outcomes
                    for outcome_data in data.get("completed", []):
                        self._completed_outcomes.append(
                            PatternOutcome.from_dict(outcome_data)
                        )

                logger.info(
                    f"Loaded {len(self._outcomes)} active and "
                    f"{len(self._completed_outcomes)} completed outcomes"
                )
            except Exception as e:
                logger.error(f"Error loading outcomes: {e}")
                self._outcomes = {}
                self._completed_outcomes = []
        else:
            self._outcomes = {}
            self._completed_outcomes = []

    def _save_outcomes(self) -> None:
        """Save outcomes to JSON file."""
        try:
            data = {
                "active": [o.to_dict() for o in self._outcomes.values()],
                "completed": [o.to_dict() for o in self._completed_outcomes[-self._max_entries:]],
                "last_save": datetime.now(timezone.utc).isoformat()
            }
            with open(self._outcomes_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving outcomes: {e}")

    async def _save_outcomes_async(self) -> None:
        """Save outcomes asynchronously."""
        await asyncio.to_thread(self._save_outcomes)

    def register_outcome_callback(
        self,
        callback: Callable[[PatternOutcome], Awaitable[None]]
    ) -> None:
        """
        Register a callback for completed outcomes.

        The callback is called when an outcome transitions from pending to completed.
        Used for integrating with the feedback buffer.

        Args:
            callback: Async function that receives the completed outcome
        """
        self._outcome_callbacks.append(callback)
        logger.debug(f"Registered outcome callback: {callback.__name__}")

    async def track_pattern(self, pattern: TCNPatternHistoryEntry) -> Optional[PatternOutcome]:
        """
        Start tracking a pattern's outcome.

        Args:
            pattern: The pattern history entry to track

        Returns:
            The created outcome tracker, or None if already tracking
        """
        if pattern.id in self._outcomes:
            logger.debug(f"Pattern {pattern.id} already being tracked")
            return None

        # Create outcome tracker
        outcome = PatternOutcome.from_pattern_entry(pattern)

        # Calculate tracking end time based on timeframe
        evaluation_period = self.EVALUATION_PERIODS.get(
            pattern.timeframe,
            self.DEFAULT_EVALUATION_PERIOD
        )
        tracking_ends = datetime.now(timezone.utc) + evaluation_period
        outcome.tracking_ends = tracking_ends.isoformat()

        # Initialize price tracking
        outcome.current_price = pattern.price_at_detection
        outcome.max_favorable_price = pattern.price_at_detection
        outcome.max_adverse_price = pattern.price_at_detection

        self._outcomes[pattern.id] = outcome

        await self._save_outcomes_async()

        logger.debug(
            f"Started tracking pattern {pattern.id}: {pattern.pattern_type} "
            f"({pattern.direction}) until {outcome.tracking_ends}"
        )

        return outcome

    async def update_outcomes(self) -> Dict[str, int]:
        """
        Update all active outcomes with current prices.

        Returns:
            Summary of updates: {updated, completed, errors}
        """
        from .data_service_client import data_service_client

        if not self._outcomes:
            return {"updated": 0, "completed": 0, "errors": 0}

        self._last_update = datetime.now(timezone.utc)
        updated = 0
        completed = 0
        errors = 0
        now = datetime.now(timezone.utc)

        # Group outcomes by symbol for efficient price fetching
        by_symbol: Dict[str, List[str]] = {}
        for pattern_id, outcome in self._outcomes.items():
            by_symbol.setdefault(outcome.symbol, []).append(pattern_id)

        for symbol, pattern_ids in by_symbol.items():
            try:
                # Fetch current price
                data, source = await data_service_client.get_historical_data(
                    symbol=symbol,
                    timeframe="1h",
                    limit=1
                )

                if not data:
                    errors += len(pattern_ids)
                    continue

                # Get current price from latest candle
                latest = data[0]
                current_price = float(latest.get("close", 0))

                if current_price <= 0:
                    errors += len(pattern_ids)
                    continue

                # Update each outcome for this symbol
                for pattern_id in pattern_ids:
                    outcome = self._outcomes.get(pattern_id)
                    if not outcome:
                        continue

                    # Update prices
                    outcome.current_price = current_price
                    outcome.last_update = now.isoformat()

                    # Calculate move percentage
                    if outcome.price_at_detection > 0:
                        move_pct = (
                            (current_price - outcome.price_at_detection)
                            / outcome.price_at_detection * 100
                        )
                        outcome.current_move_percent = round(move_pct, 4)

                        # Track favorable/adverse based on direction
                        if outcome.direction == "bullish":
                            # Bullish: higher is favorable
                            if current_price > (outcome.max_favorable_price or 0):
                                outcome.max_favorable_price = current_price
                                outcome.max_favorable_percent = round(
                                    max(0, move_pct), 4
                                )
                            if current_price < (outcome.max_adverse_price or float('inf')):
                                outcome.max_adverse_price = current_price
                                outcome.max_adverse_percent = round(
                                    min(0, move_pct), 4
                                )
                        else:
                            # Bearish: lower is favorable
                            if current_price < (outcome.max_favorable_price or float('inf')):
                                outcome.max_favorable_price = current_price
                                outcome.max_favorable_percent = round(
                                    max(0, -move_pct), 4
                                )
                            if current_price > (outcome.max_adverse_price or 0):
                                outcome.max_adverse_price = current_price
                                outcome.max_adverse_percent = round(
                                    min(0, -move_pct), 4
                                )

                    # Check for target/invalidation
                    self._check_targets(outcome, current_price)

                    # Check if tracking period ended
                    if outcome.tracking_ends:
                        tracking_ends = datetime.fromisoformat(
                            outcome.tracking_ends.replace("Z", "+00:00")
                        )
                        if now >= tracking_ends:
                            await self._complete_outcome(outcome)
                            completed += 1
                            continue

                    updated += 1

            except Exception as e:
                logger.error(f"Error updating outcomes for {symbol}: {e}")
                errors += len(pattern_ids)

        # Save updates
        await self._save_outcomes_async()

        logger.debug(
            f"Outcome update: {updated} updated, {completed} completed, {errors} errors"
        )

        return {"updated": updated, "completed": completed, "errors": errors}

    def _check_targets(self, outcome: PatternOutcome, current_price: float) -> None:
        """Check if target or invalidation levels have been hit."""
        now = datetime.now(timezone.utc).isoformat()

        # Check target
        if outcome.price_target and not outcome.target_reached:
            if outcome.direction == "bullish":
                if current_price >= outcome.price_target:
                    outcome.target_reached = True
                    outcome.target_reached_at = now
            else:
                if current_price <= outcome.price_target:
                    outcome.target_reached = True
                    outcome.target_reached_at = now

        # Check invalidation
        if outcome.invalidation_level and not outcome.invalidation_hit:
            if outcome.direction == "bullish":
                if current_price <= outcome.invalidation_level:
                    outcome.invalidation_hit = True
                    outcome.invalidation_hit_at = now
            else:
                if current_price >= outcome.invalidation_level:
                    outcome.invalidation_hit = True
                    outcome.invalidation_hit_at = now

    async def _complete_outcome(self, outcome: PatternOutcome) -> None:
        """Complete an outcome and move it to completed list."""
        # Determine final status
        if outcome.target_reached:
            outcome.status = OutcomeStatus.SUCCESS.value
            outcome.outcome_reason = "Target reached"
        elif outcome.invalidation_hit:
            outcome.status = OutcomeStatus.FAILED.value
            outcome.outcome_reason = "Invalidation level hit"
        else:
            # Evaluate based on favorable/adverse movement
            favorable = abs(outcome.max_favorable_percent)
            adverse = abs(outcome.max_adverse_percent)

            # Calculate target achievement ratio if target exists
            if outcome.price_target and outcome.price_at_detection:
                target_distance = abs(outcome.price_target - outcome.price_at_detection)
                if outcome.direction == "bullish":
                    achieved = max(0, (outcome.max_favorable_price or outcome.price_at_detection) - outcome.price_at_detection)
                else:
                    achieved = max(0, outcome.price_at_detection - (outcome.max_favorable_price or outcome.price_at_detection))

                achievement_ratio = achieved / target_distance if target_distance > 0 else 0

                if achievement_ratio >= 0.6:
                    outcome.status = OutcomeStatus.SUCCESS.value
                    outcome.outcome_reason = f"Achieved {achievement_ratio:.0%} of target"
                elif achievement_ratio >= 0.3:
                    outcome.status = OutcomeStatus.PARTIAL.value
                    outcome.outcome_reason = f"Partial: {achievement_ratio:.0%} of target"
                elif adverse > 2.0:  # More than 2% against
                    outcome.status = OutcomeStatus.FAILED.value
                    outcome.outcome_reason = f"Adverse move: {adverse:.1f}%"
                else:
                    outcome.status = OutcomeStatus.EXPIRED.value
                    outcome.outcome_reason = "No significant movement"
            else:
                # No target defined - use favorable vs adverse
                if favorable > adverse * 1.5:  # 1.5:1 ratio
                    outcome.status = OutcomeStatus.SUCCESS.value
                    outcome.outcome_reason = f"Favorable: {favorable:.1f}% vs {adverse:.1f}%"
                elif adverse > 2.0:
                    outcome.status = OutcomeStatus.FAILED.value
                    outcome.outcome_reason = f"Adverse move: {adverse:.1f}%"
                else:
                    outcome.status = OutcomeStatus.EXPIRED.value
                    outcome.outcome_reason = "No clear outcome"

        # Remove from active, add to completed
        if outcome.pattern_id in self._outcomes:
            del self._outcomes[outcome.pattern_id]

        self._completed_outcomes.append(outcome)

        # Cleanup old completed outcomes
        self._cleanup_completed()

        logger.info(
            f"Outcome completed: {outcome.pattern_id} -> {outcome.status} "
            f"({outcome.outcome_reason})"
        )

        # Call registered callbacks
        for callback in self._outcome_callbacks:
            try:
                await callback(outcome)
            except Exception as e:
                logger.error(f"Error in outcome callback: {e}")

        # Send feedback to TCN Training Service
        await self._send_feedback_to_training_service(outcome)

        # Add observation to drift detection
        await self._add_drift_observation(outcome)

    def _cleanup_completed(self) -> None:
        """Cleanup old completed outcomes."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)

        self._completed_outcomes = [
            o for o in self._completed_outcomes
            if datetime.fromisoformat(o.detected_at.replace("Z", "+00:00")) > cutoff
        ]

        # Enforce max entries
        if len(self._completed_outcomes) > self._max_entries:
            self._completed_outcomes = self._completed_outcomes[-self._max_entries:]

    def get_outcome(self, pattern_id: str) -> Optional[PatternOutcome]:
        """Get an outcome by pattern ID (active or completed)."""
        if pattern_id in self._outcomes:
            return self._outcomes[pattern_id]

        for outcome in self._completed_outcomes:
            if outcome.pattern_id == pattern_id:
                return outcome

        return None

    def get_active_outcomes(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[PatternOutcome]:
        """Get active (pending) outcomes."""
        outcomes = list(self._outcomes.values())

        if symbol:
            outcomes = [o for o in outcomes if o.symbol == symbol]

        # Sort by detection time (newest first)
        outcomes.sort(key=lambda x: x.detected_at, reverse=True)

        return outcomes[:limit]

    def get_completed_outcomes(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 100
    ) -> List[PatternOutcome]:
        """Get completed outcomes with optional filtering."""
        outcomes = self._completed_outcomes.copy()

        if symbol:
            outcomes = [o for o in outcomes if o.symbol == symbol]

        if status:
            outcomes = [o for o in outcomes if o.status == status]

        if pattern_type:
            outcomes = [o for o in outcomes if o.pattern_type == pattern_type]

        # Sort by detection time (newest first)
        outcomes.sort(key=lambda x: x.detected_at, reverse=True)

        return outcomes[:limit]

    def get_statistics(self) -> OutcomeStatistics:
        """Calculate and return outcome statistics."""
        stats = OutcomeStatistics()

        completed = self._completed_outcomes
        stats.total_tracked = len(self._outcomes) + len(completed)
        stats.total_completed = len(completed)
        stats.pending = len(self._outcomes)

        if not completed:
            stats.last_update = self._last_update.isoformat() if self._last_update else None
            return stats

        # Count by status
        for outcome in completed:
            if outcome.status == OutcomeStatus.SUCCESS.value:
                stats.success_count += 1
            elif outcome.status == OutcomeStatus.PARTIAL.value:
                stats.partial_count += 1
            elif outcome.status == OutcomeStatus.FAILED.value:
                stats.failed_count += 1
            elif outcome.status == OutcomeStatus.INVALIDATED.value:
                stats.invalidated_count += 1
            elif outcome.status == OutcomeStatus.EXPIRED.value:
                stats.expired_count += 1

            # By pattern type
            pt = outcome.pattern_type
            if pt not in stats.by_pattern_type:
                stats.by_pattern_type[pt] = {"total": 0, "success": 0, "failed": 0}
            stats.by_pattern_type[pt]["total"] += 1
            if outcome.status in [OutcomeStatus.SUCCESS.value, OutcomeStatus.PARTIAL.value]:
                stats.by_pattern_type[pt]["success"] += 1
            elif outcome.status == OutcomeStatus.FAILED.value:
                stats.by_pattern_type[pt]["failed"] += 1

            # By direction
            direction = outcome.direction
            if direction not in stats.by_direction:
                stats.by_direction[direction] = {"total": 0, "success": 0, "failed": 0}
            stats.by_direction[direction]["total"] += 1
            if outcome.status in [OutcomeStatus.SUCCESS.value, OutcomeStatus.PARTIAL.value]:
                stats.by_direction[direction]["success"] += 1
            elif outcome.status == OutcomeStatus.FAILED.value:
                stats.by_direction[direction]["failed"] += 1

            # By timeframe
            tf = outcome.timeframe
            if tf not in stats.by_timeframe:
                stats.by_timeframe[tf] = {"total": 0, "success": 0, "failed": 0}
            stats.by_timeframe[tf]["total"] += 1
            if outcome.status in [OutcomeStatus.SUCCESS.value, OutcomeStatus.PARTIAL.value]:
                stats.by_timeframe[tf]["success"] += 1
            elif outcome.status == OutcomeStatus.FAILED.value:
                stats.by_timeframe[tf]["failed"] += 1

        # Calculate rates
        total = stats.total_completed
        if total > 0:
            stats.success_rate = round((stats.success_count + stats.partial_count) / total, 4)
            stats.partial_rate = round(stats.partial_count / total, 4)
            stats.failure_rate = round(stats.failed_count / total, 4)

        # Calculate average movements
        favorable_sum = sum(abs(o.max_favorable_percent) for o in completed)
        adverse_sum = sum(abs(o.max_adverse_percent) for o in completed)

        stats.avg_favorable_percent = round(favorable_sum / total, 4) if total > 0 else 0
        stats.avg_adverse_percent = round(adverse_sum / total, 4) if total > 0 else 0
        stats.profit_factor = round(favorable_sum / adverse_sum, 4) if adverse_sum > 0 else 0

        stats.last_update = self._last_update.isoformat() if self._last_update else None

        return stats

    async def _update_loop(self) -> None:
        """Background loop for periodic price updates."""
        logger.info(f"Starting outcome update loop (interval: {self._update_interval}s)")

        while self._update_running:
            try:
                await self.update_outcomes()
            except Exception as e:
                logger.error(f"Error in outcome update loop: {e}")

            await asyncio.sleep(self._update_interval)

    async def start_update_loop(self) -> bool:
        """Start the background update loop."""
        if self._update_running:
            logger.warning("Update loop already running")
            return False

        self._update_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Outcome update loop started")
        return True

    async def stop_update_loop(self) -> bool:
        """Stop the background update loop."""
        if not self._update_running:
            logger.warning("Update loop not running")
            return False

        self._update_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

        logger.info("Outcome update loop stopped")
        return True

    def is_update_running(self) -> bool:
        """Check if update loop is running."""
        return self._update_running

    async def _send_feedback_to_training_service(self, outcome: PatternOutcome) -> bool:
        """
        Send completed outcome to TCN Training Service for feedback buffer.

        Args:
            outcome: The completed outcome

        Returns:
            True if feedback was sent successfully
        """
        import httpx

        tcn_train_url = os.getenv("TCN_TRAIN_SERVICE_URL", "http://trading-tcn-train:3013")

        # Get OHLCV data from pattern history
        ohlcv_data = await self._get_pattern_ohlcv(outcome.pattern_id)
        if not ohlcv_data:
            logger.debug(f"No OHLCV data for pattern {outcome.pattern_id}, skipping feedback")
            return False

        # Calculate outcome score
        outcome_score = self._calculate_outcome_score(outcome)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{tcn_train_url}/api/v1/feedback-buffer/outcome",
                    json={
                        "pattern_id": outcome.pattern_id,
                        "symbol": outcome.symbol,
                        "timeframe": outcome.timeframe,
                        "pattern_type": outcome.pattern_type,
                        "direction": outcome.direction,
                        "original_confidence": outcome.confidence,
                        "outcome_status": outcome.status,
                        "outcome_score": outcome_score,
                        "max_favorable_percent": outcome.max_favorable_percent,
                        "max_adverse_percent": outcome.max_adverse_percent,
                        "ohlcv_data": ohlcv_data,
                        "claude_validated": outcome.claude_validated,
                        "claude_agreed": outcome.claude_agreed,
                    }
                )

                if response.status_code == 200:
                    logger.debug(f"Sent feedback for {outcome.pattern_id} to training service")
                    return True
                else:
                    logger.warning(
                        f"Failed to send feedback: {response.status_code} - {response.text}"
                    )
                    return False

        except httpx.ConnectError:
            logger.debug("TCN Training Service not available for feedback")
            return False
        except Exception as e:
            logger.error(f"Error sending feedback: {e}")
            return False

    async def _get_pattern_ohlcv(self, pattern_id: str) -> Optional[List[dict]]:
        """Get OHLCV data for a pattern from history."""
        from .tcn_pattern_history_service import tcn_pattern_history_service

        # Search in pattern history
        for entry in tcn_pattern_history_service._history:
            if entry.id == pattern_id:
                return entry.ohlcv_data

        return None

    def _calculate_outcome_score(self, outcome: PatternOutcome) -> float:
        """
        Calculate outcome score from -1 to 1.

        Score based on:
        - Status (success=1, partial=0.5, failed=-1, expired=0)
        - Favorable vs adverse movement
        """
        # Base score from status
        status_scores = {
            OutcomeStatus.SUCCESS.value: 1.0,
            OutcomeStatus.PARTIAL.value: 0.5,
            OutcomeStatus.FAILED.value: -1.0,
            OutcomeStatus.INVALIDATED.value: -0.5,
            OutcomeStatus.EXPIRED.value: 0.0,
        }
        base_score = status_scores.get(outcome.status, 0.0)

        # Adjust by movement ratio
        favorable = abs(outcome.max_favorable_percent)
        adverse = abs(outcome.max_adverse_percent)

        if favorable + adverse > 0:
            movement_ratio = (favorable - adverse) / (favorable + adverse)
            # Blend base score with movement ratio
            score = base_score * 0.7 + movement_ratio * 0.3
        else:
            score = base_score

        return round(max(-1.0, min(1.0, score)), 4)


    async def _add_drift_observation(self, outcome: PatternOutcome) -> None:
        """Add completed outcome to drift detection service."""
        try:
            from .drift_detection_service import drift_detection_service

            is_success = outcome.status in [
                OutcomeStatus.SUCCESS.value,
                OutcomeStatus.PARTIAL.value
            ]

            drift_detection_service.add_observation(
                pattern_type=outcome.pattern_type,
                confidence=outcome.confidence,
                is_success=is_success,
                timestamp=datetime.fromisoformat(
                    outcome.detected_at.replace("Z", "+00:00")
                )
            )

            logger.debug(f"Added drift observation for {outcome.pattern_id}")
        except Exception as e:
            logger.debug(f"Could not add drift observation: {e}")


# Singleton instance
outcome_tracker_service = OutcomeTrackerService()
