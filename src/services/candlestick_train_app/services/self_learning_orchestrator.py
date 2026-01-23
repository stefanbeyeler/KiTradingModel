"""
Self-Learning Orchestrator for Candlestick Patterns.

Coordinates the self-learning loop:
1. Monitor feedback buffer
2. Trigger rule parameter optimization when conditions met
3. Validate changes before deployment
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

from .feedback_buffer_service import feedback_buffer_service


class OrchestratorState(Enum):
    """States of the self-learning orchestrator."""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"


@dataclass
class OrchestratorConfig:
    """Configuration for the self-learning orchestrator."""
    enabled: bool = True
    check_interval_minutes: int = 60

    # Time-based scheduling
    scheduled_optimization_enabled: bool = True
    scheduled_hour: int = 3  # UTC
    scheduled_minute: int = 0

    # Trigger conditions
    min_samples_for_optimization: int = 100
    min_hours_between_optimizations: int = 24
    max_hours_between_optimizations: int = 168  # 7 days

    # Optimization settings
    auto_apply_recommendations: bool = False  # Require manual review by default
    min_feedback_count_for_change: int = 3  # Minimum feedback instances before changing a parameter
    max_parameter_change_percent: float = 20.0  # Maximum % change per optimization

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OrchestratorStatus:
    """Status of the self-learning orchestrator."""
    state: OrchestratorState = OrchestratorState.IDLE
    loop_running: bool = False
    last_check: Optional[str] = None
    last_optimization: Optional[str] = None
    last_optimization_trigger: Optional[str] = None
    next_check: Optional[str] = None
    next_scheduled_optimization: Optional[str] = None
    optimization_count: int = 0
    successful_optimizations: int = 0
    rejected_optimizations: int = 0
    feedback_buffer_samples: int = 0
    pending_recommendations: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


class SelfLearningOrchestrator:
    """
    Orchestrates the self-learning loop for candlestick pattern detection.

    Unlike TCN which uses neural network retraining, Candlestick uses rule-based
    detection. Self-learning here means analyzing feedback to adjust rule parameters.
    """

    def __init__(self, state_file: str = None):
        data_dir = os.getenv("DATA_DIR", "/app/data")
        if state_file is None:
            state_file = os.path.join(data_dir, "candlestick_self_learning_state.json")

        self._state_file = Path(state_file)
        self._config = OrchestratorConfig()
        self._status = OrchestratorStatus()

        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

        # Service URLs
        self._candlestick_service_url = os.getenv(
            "CANDLESTICK_SERVICE_URL",
            "http://trading-candlestick:3006"
        )

        # Load saved state
        self._load_state()

        logger.info("SelfLearningOrchestrator initialized")

    def _load_state(self):
        """Load state from file."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    data = json.load(f)

                if "config" in data:
                    self._config = OrchestratorConfig.from_dict(data["config"])

                if "status" in data:
                    status_data = data["status"]
                    if "state" in status_data:
                        status_data["state"] = OrchestratorState(status_data["state"])
                    self._status = OrchestratorStatus(**{
                        k: v for k, v in status_data.items()
                        if k in OrchestratorStatus.__dataclass_fields__
                    })

                logger.info("Loaded self-learning orchestrator state")
        except Exception as e:
            logger.error(f"Failed to load orchestrator state: {e}")

    def _save_state(self):
        """Save state to file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump({
                    "config": self._config.to_dict(),
                    "status": self._status.to_dict(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save orchestrator state: {e}")

    async def start_loop(self) -> Dict[str, Any]:
        """Start the monitoring loop."""
        if self._running:
            return {"status": "already_running"}

        self._running = True
        self._status.loop_running = True
        self._status.state = OrchestratorState.MONITORING
        self._save_state()

        self._loop_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Self-learning orchestrator loop started")

        return {"status": "started"}

    async def stop_loop(self) -> Dict[str, Any]:
        """Stop the monitoring loop."""
        if not self._running:
            return {"status": "not_running"}

        self._running = False
        self._status.loop_running = False
        self._status.state = OrchestratorState.IDLE

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info("Self-learning orchestrator loop stopped")

        return {"status": "stopped"}

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_and_optimize()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._status.error_message = str(e)

            # Calculate next check time
            now = datetime.now(timezone.utc)
            next_check = now + timedelta(minutes=self._config.check_interval_minutes)
            self._status.next_check = next_check.isoformat()

            # Calculate next scheduled optimization
            if self._config.scheduled_optimization_enabled:
                next_scheduled = self._calculate_next_scheduled_time()
                self._status.next_scheduled_optimization = next_scheduled.isoformat()

            self._save_state()

            await asyncio.sleep(self._config.check_interval_minutes * 60)

    def _calculate_next_scheduled_time(self) -> datetime:
        """Calculate the next scheduled optimization time."""
        now = datetime.now(timezone.utc)
        scheduled = now.replace(
            hour=self._config.scheduled_hour,
            minute=self._config.scheduled_minute,
            second=0,
            microsecond=0
        )

        if scheduled <= now:
            scheduled += timedelta(days=1)

        return scheduled

    async def _check_and_optimize(self):
        """Check conditions and trigger optimization if needed."""
        now = datetime.now(timezone.utc)
        self._status.last_check = now.isoformat()
        self._status.state = OrchestratorState.MONITORING

        # Get buffer statistics
        buffer_stats = feedback_buffer_service.get_statistics()
        self._status.feedback_buffer_samples = buffer_stats.unused_samples

        # Check if optimization should be triggered
        should_optimize, trigger_reason = self._should_optimize(buffer_stats)

        if should_optimize:
            await self._trigger_optimization(trigger_reason)

        self._save_state()

    def _should_optimize(self, buffer_stats) -> tuple[bool, str]:
        """Determine if optimization should be triggered."""
        if not self._config.enabled:
            return False, ""

        now = datetime.now(timezone.utc)

        # Check minimum samples
        if buffer_stats.unused_samples < self._config.min_samples_for_optimization:
            return False, ""

        # Check minimum time since last optimization
        if self._status.last_optimization:
            last_opt = datetime.fromisoformat(self._status.last_optimization.replace('Z', '+00:00'))
            hours_since = (now - last_opt).total_seconds() / 3600

            if hours_since < self._config.min_hours_between_optimizations:
                return False, ""

            # Force optimization if max interval exceeded
            if hours_since >= self._config.max_hours_between_optimizations:
                return True, "max_interval_exceeded"

        # Check scheduled time
        if self._config.scheduled_optimization_enabled:
            scheduled = self._calculate_next_scheduled_time() - timedelta(days=1)
            time_diff = abs((now - scheduled).total_seconds())

            # Within 5 minute window of scheduled time
            if time_diff < 300:
                return True, "scheduled"

        # Buffer is ready and conditions met
        if buffer_stats.ready_for_training:
            return True, "buffer_ready"

        return False, ""

    async def _trigger_optimization(self, trigger_reason: str):
        """Trigger the optimization process."""
        logger.info(f"Triggering optimization (reason: {trigger_reason})")

        self._status.state = OrchestratorState.ANALYZING
        self._status.last_optimization_trigger = trigger_reason
        self._save_state()

        try:
            # Analyze feedback and generate recommendations
            recommendations = await self._analyze_feedback()

            if not recommendations:
                logger.info("No optimization recommendations generated")
                self._status.state = OrchestratorState.MONITORING
                return

            self._status.pending_recommendations = len(recommendations)
            self._status.state = OrchestratorState.OPTIMIZING

            # Apply recommendations if auto-apply is enabled
            if self._config.auto_apply_recommendations:
                applied = await self._apply_recommendations(recommendations)

                if applied:
                    self._status.successful_optimizations += 1
                    logger.info(f"Applied {len(recommendations)} parameter adjustments")
                else:
                    self._status.rejected_optimizations += 1
            else:
                logger.info(f"Generated {len(recommendations)} recommendations (manual review required)")

            self._status.optimization_count += 1
            self._status.last_optimization = datetime.now(timezone.utc).isoformat()

            # Mark samples as used
            batch = feedback_buffer_service.get_training_batch(
                batch_size=self._config.min_samples_for_optimization
            )
            feedback_buffer_service.mark_as_used([s.sample_id for s in batch])

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self._status.error_message = str(e)

        finally:
            self._status.state = OrchestratorState.MONITORING
            self._save_state()

    async def _analyze_feedback(self) -> List[Dict[str, Any]]:
        """Analyze feedback buffer and generate recommendations."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Call the feedback analysis endpoint on the inference service
                response = await client.post(
                    f"{self._candlestick_service_url}/api/v1/rule-optimizer/analyze-feedback"
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("recommendations", [])
                else:
                    logger.warning(f"Feedback analysis failed: {response.status_code}")
                    return []

        except Exception as e:
            logger.error(f"Failed to analyze feedback: {e}")
            return []

    async def _apply_recommendations(self, recommendations: List[Dict[str, Any]]) -> bool:
        """Apply optimization recommendations."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Apply each recommendation
                for rec in recommendations:
                    response = await client.post(
                        f"{self._candlestick_service_url}/api/v1/rule-optimizer/apply-recommendation",
                        json=rec
                    )

                    if response.status_code != 200:
                        logger.warning(f"Failed to apply recommendation: {rec.get('parameter')}")

                return True

        except Exception as e:
            logger.error(f"Failed to apply recommendations: {e}")
            return False

    async def trigger_manual_optimization(self) -> Dict[str, Any]:
        """Manually trigger optimization."""
        buffer_stats = feedback_buffer_service.get_statistics()

        if buffer_stats.unused_samples < self._config.min_samples_for_optimization:
            return {
                "status": "skipped",
                "reason": f"Insufficient samples ({buffer_stats.unused_samples} < {self._config.min_samples_for_optimization})"
            }

        await self._trigger_optimization("manual")

        return {
            "status": "completed",
            "recommendations": self._status.pending_recommendations
        }

    def update_config(self, updates: Dict[str, Any]) -> OrchestratorConfig:
        """Update orchestrator configuration."""
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._save_state()
        return self._config

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "status": self._status.to_dict(),
            "config": self._config.to_dict(),
        }

    def get_config(self) -> OrchestratorConfig:
        """Get current configuration."""
        return self._config


# Global singleton
self_learning_orchestrator = SelfLearningOrchestrator()
