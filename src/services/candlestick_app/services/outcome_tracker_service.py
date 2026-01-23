"""
Outcome Tracker Service for Candlestick Patterns.

Tracks the price movement after candlestick pattern detection to evaluate
pattern effectiveness and provide feedback for self-learning.
"""

import asyncio
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from loguru import logger
import httpx


class OutcomeStatus(Enum):
    """Status of a tracked pattern outcome."""
    PENDING = "pending"           # Still being tracked
    SUCCESS = "success"           # Pattern prediction correct (price moved as expected)
    PARTIAL = "partial"           # Partial success (30-60% of expected move)
    FAILED = "failed"             # Pattern prediction wrong
    EXPIRED = "expired"           # Tracking period ended without clear outcome


# Evaluation periods per timeframe (in hours)
EVALUATION_PERIODS = {
    "M1": 1,
    "M5": 2,
    "M15": 4,
    "M30": 8,
    "H1": 24,
    "H4": 72,
    "D1": 336,  # 14 days
    "W1": 672,  # 4 weeks
}

# Expected move percentages by pattern strength
EXPECTED_MOVE_PERCENT = {
    "strong": 2.0,
    "moderate": 1.0,
    "weak": 0.5,
}


@dataclass
class PatternOutcome:
    """Represents a tracked candlestick pattern outcome."""
    # Identification
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str

    # Pattern details
    direction: str  # bullish, bearish, neutral
    category: str   # reversal, continuation, indecision
    strength: str   # strong, moderate, weak
    confidence: float

    # Detection context
    price_at_detection: float
    detection_time: str
    ohlc_data: List[Dict[str, Any]] = field(default_factory=list)

    # Tracking state
    status: OutcomeStatus = OutcomeStatus.PENDING
    tracking_start: str = ""
    tracking_ends: str = ""

    # Price tracking
    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    last_update: str = ""

    # Outcome metrics
    max_favorable_percent: float = 0.0
    max_adverse_percent: float = 0.0
    final_move_percent: float = 0.0

    # Additional info
    completion_time: str = ""
    outcome_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternOutcome":
        """Create from dictionary."""
        if "status" in data:
            data["status"] = OutcomeStatus(data["status"])
        return cls(**data)


class OutcomeTrackerService:
    """
    Tracks candlestick pattern outcomes to evaluate prediction effectiveness.

    Features:
    - Automatic price tracking after pattern detection
    - Configurable evaluation periods per timeframe
    - Success/failure classification based on price movement
    - Feedback generation for self-learning system
    """

    def __init__(self, data_file: str = None):
        data_dir = os.getenv("DATA_DIR", "/app/data")
        if data_file is None:
            data_file = os.path.join(data_dir, "candlestick_pattern_outcomes.json")

        self._data_file = Path(data_file)
        self._outcomes: Dict[str, PatternOutcome] = {}
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._update_interval = int(os.getenv("OUTCOME_UPDATE_INTERVAL", "900"))  # 15 min

        # Callbacks for completed outcomes
        self._callbacks: List[Callable] = []

        # Data service URL
        self._data_service_url = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")

        # Training service URL for feedback
        self._train_service_url = os.getenv("CANDLESTICK_TRAIN_SERVICE_URL", "http://trading-candlestick-train:3016")

        # Load existing outcomes
        self._load_outcomes()

        logger.info(f"OutcomeTrackerService initialized - {len(self._outcomes)} outcomes loaded")

    def _load_outcomes(self):
        """Load outcomes from file."""
        try:
            if self._data_file.exists():
                with open(self._data_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        outcome = PatternOutcome.from_dict(item)
                        self._outcomes[outcome.pattern_id] = outcome
                logger.info(f"Loaded {len(self._outcomes)} pattern outcomes")
        except Exception as e:
            logger.error(f"Failed to load outcomes: {e}")

    def _save_outcomes(self):
        """Save outcomes to file."""
        try:
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._data_file, 'w') as f:
                data = [o.to_dict() for o in self._outcomes.values()]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save outcomes: {e}")

    def register_callback(self, callback: Callable):
        """Register a callback for completed outcomes."""
        self._callbacks.append(callback)

    async def track_pattern(self, pattern_data: Dict[str, Any]) -> Optional[PatternOutcome]:
        """
        Start tracking a detected pattern.

        Args:
            pattern_data: Pattern data from pattern_history_service

        Returns:
            PatternOutcome if tracking started, None otherwise
        """
        try:
            pattern_id = pattern_data.get("id")
            if not pattern_id:
                logger.warning("Pattern has no ID, cannot track")
                return None

            # Check if already tracking
            if pattern_id in self._outcomes:
                logger.debug(f"Pattern {pattern_id} already being tracked")
                return self._outcomes[pattern_id]

            symbol = pattern_data.get("symbol", "")
            timeframe = pattern_data.get("timeframe", "H1")

            # Calculate evaluation period
            eval_hours = EVALUATION_PERIODS.get(timeframe, 24)
            now = datetime.now(timezone.utc)
            tracking_ends = now + timedelta(hours=eval_hours)

            # Get price at detection
            price_at_detection = pattern_data.get("price_at_detection", 0.0)
            if not price_at_detection:
                # Try to get current price
                price_at_detection = await self._get_current_price(symbol)

            # Extract OHLC data if available
            ohlc_context = pattern_data.get("ohlc_context", {})
            ohlc_data = ohlc_context.get("candles", []) if ohlc_context else []

            outcome = PatternOutcome(
                pattern_id=pattern_id,
                symbol=symbol,
                timeframe=timeframe,
                pattern_type=pattern_data.get("pattern_type", "unknown"),
                direction=pattern_data.get("direction", "neutral"),
                category=pattern_data.get("category", "indecision"),
                strength=pattern_data.get("strength", "moderate"),
                confidence=pattern_data.get("confidence", 0.5),
                price_at_detection=price_at_detection,
                detection_time=pattern_data.get("timestamp", now.isoformat()),
                ohlc_data=ohlc_data,
                tracking_start=now.isoformat(),
                tracking_ends=tracking_ends.isoformat(),
                current_price=price_at_detection,
                highest_price=price_at_detection,
                lowest_price=price_at_detection,
                last_update=now.isoformat(),
            )

            self._outcomes[pattern_id] = outcome
            self._save_outcomes()

            logger.info(f"Started tracking pattern {pattern_id} ({outcome.pattern_type} on {symbol}/{timeframe})")
            return outcome

        except Exception as e:
            logger.error(f"Failed to track pattern: {e}")
            return None

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price from data service."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._data_service_url}/api/v1/twelvedata/price/{symbol}"
                )
                if response.status_code == 200:
                    data = response.json()
                    return float(data.get("price", 0))
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
        return 0.0

    async def update_outcomes(self):
        """Update all pending outcomes with current prices."""
        pending = [o for o in self._outcomes.values() if o.status == OutcomeStatus.PENDING]

        if not pending:
            return

        logger.debug(f"Updating {len(pending)} pending outcomes")

        # Group by symbol for efficient price fetching
        by_symbol: Dict[str, List[PatternOutcome]] = {}
        for outcome in pending:
            if outcome.symbol not in by_symbol:
                by_symbol[outcome.symbol] = []
            by_symbol[outcome.symbol].append(outcome)

        now = datetime.now(timezone.utc)
        completed = []

        for symbol, outcomes in by_symbol.items():
            try:
                current_price = await self._get_current_price(symbol)
                if current_price <= 0:
                    continue

                for outcome in outcomes:
                    # Update price tracking
                    outcome.current_price = current_price
                    outcome.highest_price = max(outcome.highest_price, current_price)
                    outcome.lowest_price = min(outcome.lowest_price, current_price) if outcome.lowest_price > 0 else current_price
                    outcome.last_update = now.isoformat()

                    # Calculate movement percentages
                    if outcome.price_at_detection > 0:
                        move_percent = ((current_price - outcome.price_at_detection) / outcome.price_at_detection) * 100

                        if outcome.direction == "bullish":
                            outcome.max_favorable_percent = max(
                                outcome.max_favorable_percent,
                                ((outcome.highest_price - outcome.price_at_detection) / outcome.price_at_detection) * 100
                            )
                            outcome.max_adverse_percent = max(
                                outcome.max_adverse_percent,
                                ((outcome.price_at_detection - outcome.lowest_price) / outcome.price_at_detection) * 100
                            )
                        elif outcome.direction == "bearish":
                            outcome.max_favorable_percent = max(
                                outcome.max_favorable_percent,
                                ((outcome.price_at_detection - outcome.lowest_price) / outcome.price_at_detection) * 100
                            )
                            outcome.max_adverse_percent = max(
                                outcome.max_adverse_percent,
                                ((outcome.highest_price - outcome.price_at_detection) / outcome.price_at_detection) * 100
                            )

                        outcome.final_move_percent = move_percent

                    # Check if tracking period ended
                    tracking_ends = datetime.fromisoformat(outcome.tracking_ends.replace('Z', '+00:00'))
                    if now >= tracking_ends:
                        self._evaluate_outcome(outcome)
                        completed.append(outcome)

            except Exception as e:
                logger.warning(f"Error updating outcomes for {symbol}: {e}")

        # Save and trigger callbacks for completed outcomes
        self._save_outcomes()

        for outcome in completed:
            await self._on_outcome_completed(outcome)

    def _evaluate_outcome(self, outcome: PatternOutcome):
        """Evaluate the final outcome of a pattern."""
        expected_move = EXPECTED_MOVE_PERCENT.get(outcome.strength, 1.0)

        if outcome.direction == "bullish":
            favorable = outcome.max_favorable_percent
            adverse = outcome.max_adverse_percent
        elif outcome.direction == "bearish":
            favorable = outcome.max_favorable_percent
            adverse = outcome.max_adverse_percent
        else:  # neutral
            # For neutral patterns, success is no major move
            if abs(outcome.final_move_percent) < expected_move:
                outcome.status = OutcomeStatus.SUCCESS
                outcome.outcome_reason = "Neutral pattern: price remained stable"
            else:
                outcome.status = OutcomeStatus.FAILED
                outcome.outcome_reason = f"Neutral pattern: unexpected move of {outcome.final_move_percent:.2f}%"
            outcome.completion_time = datetime.now(timezone.utc).isoformat()
            return

        # Evaluate directional patterns
        if favorable >= expected_move:
            if adverse < favorable * 0.5:  # Adverse move less than 50% of favorable
                outcome.status = OutcomeStatus.SUCCESS
                outcome.outcome_reason = f"Target reached: {favorable:.2f}% favorable move"
            else:
                outcome.status = OutcomeStatus.PARTIAL
                outcome.outcome_reason = f"Partial: {favorable:.2f}% favorable but {adverse:.2f}% adverse"
        elif favorable >= expected_move * 0.3:  # At least 30% of expected
            outcome.status = OutcomeStatus.PARTIAL
            outcome.outcome_reason = f"Partial success: {favorable:.2f}% of {expected_move:.2f}% target"
        else:
            outcome.status = OutcomeStatus.FAILED
            outcome.outcome_reason = f"Failed: only {favorable:.2f}% favorable, {adverse:.2f}% adverse"

        outcome.completion_time = datetime.now(timezone.utc).isoformat()

    async def _on_outcome_completed(self, outcome: PatternOutcome):
        """Handle completed outcome - send to feedback buffer."""
        logger.info(f"Outcome completed: {outcome.pattern_id} -> {outcome.status.value}")

        # Send to training service feedback buffer
        await self._send_feedback_to_training_service(outcome)

        # Trigger registered callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(outcome)
                else:
                    callback(outcome)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _send_feedback_to_training_service(self, outcome: PatternOutcome):
        """Send completed outcome to training service feedback buffer."""
        try:
            feedback_data = {
                "pattern_id": outcome.pattern_id,
                "symbol": outcome.symbol,
                "timeframe": outcome.timeframe,
                "pattern_type": outcome.pattern_type,
                "direction": outcome.direction,
                "category": outcome.category,
                "strength": outcome.strength,
                "confidence": outcome.confidence,
                "outcome_status": outcome.status.value,
                "max_favorable_percent": outcome.max_favorable_percent,
                "max_adverse_percent": outcome.max_adverse_percent,
                "final_move_percent": outcome.final_move_percent,
                "ohlc_data": outcome.ohlc_data,
                "detection_time": outcome.detection_time,
                "completion_time": outcome.completion_time,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._train_service_url}/api/v1/feedback-buffer/outcome",
                    json=feedback_data
                )

                if response.status_code == 200:
                    logger.debug(f"Sent feedback for {outcome.pattern_id} to training service")
                else:
                    logger.warning(f"Failed to send feedback: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send feedback to training service: {e}")

    async def _update_loop(self):
        """Background loop to update outcomes."""
        while self._running:
            try:
                await self.update_outcomes()
            except Exception as e:
                logger.error(f"Error in outcome update loop: {e}")

            await asyncio.sleep(self._update_interval)

    async def start_loop(self):
        """Start the background update loop."""
        if self._running:
            return {"status": "already_running"}

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Outcome tracker loop started")
        return {"status": "started"}

    async def stop_loop(self):
        """Stop the background update loop."""
        if not self._running:
            return {"status": "not_running"}

        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("Outcome tracker loop stopped")
        return {"status": "stopped"}

    def is_running(self) -> bool:
        """Check if update loop is running."""
        return self._running

    def get_outcomes(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get outcomes with optional filters."""
        results = []

        for outcome in sorted(
            self._outcomes.values(),
            key=lambda x: x.detection_time,
            reverse=True
        ):
            if status and outcome.status.value != status:
                continue
            if symbol and outcome.symbol != symbol:
                continue
            if pattern_type and outcome.pattern_type != pattern_type:
                continue

            results.append(outcome.to_dict())

            if len(results) >= limit:
                break

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get outcome statistics."""
        total = len(self._outcomes)
        if total == 0:
            return {
                "total": 0,
                "pending": 0,
                "completed": 0,
                "success_rate": 0,
                "by_status": {},
                "by_pattern_type": {},
                "by_direction": {},
            }

        by_status = {}
        by_pattern = {}
        by_direction = {}

        for outcome in self._outcomes.values():
            status = outcome.status.value
            by_status[status] = by_status.get(status, 0) + 1

            pt = outcome.pattern_type
            if pt not in by_pattern:
                by_pattern[pt] = {"total": 0, "success": 0, "partial": 0, "failed": 0}
            by_pattern[pt]["total"] += 1
            if outcome.status == OutcomeStatus.SUCCESS:
                by_pattern[pt]["success"] += 1
            elif outcome.status == OutcomeStatus.PARTIAL:
                by_pattern[pt]["partial"] += 1
            elif outcome.status == OutcomeStatus.FAILED:
                by_pattern[pt]["failed"] += 1

            direction = outcome.direction
            if direction not in by_direction:
                by_direction[direction] = {"total": 0, "success": 0}
            by_direction[direction]["total"] += 1
            if outcome.status in [OutcomeStatus.SUCCESS, OutcomeStatus.PARTIAL]:
                by_direction[direction]["success"] += 1

        completed = total - by_status.get("pending", 0)
        success = by_status.get("success", 0) + by_status.get("partial", 0)
        success_rate = (success / completed * 100) if completed > 0 else 0

        return {
            "total": total,
            "pending": by_status.get("pending", 0),
            "completed": completed,
            "success_rate": round(success_rate, 1),
            "by_status": by_status,
            "by_pattern_type": by_pattern,
            "by_direction": by_direction,
            "loop_running": self._running,
            "update_interval_seconds": self._update_interval,
        }

    def clear_completed(self):
        """Clear all completed outcomes."""
        self._outcomes = {
            k: v for k, v in self._outcomes.items()
            if v.status == OutcomeStatus.PENDING
        }
        self._save_outcomes()
        logger.info("Cleared completed outcomes")


# Global singleton
outcome_tracker_service = OutcomeTrackerService()
