"""
Self-Learning Orchestrator for TCN Model.

Coordinates all self-learning components:
- Monitors drift detection status
- Checks feedback buffer readiness
- Triggers incremental training
- Manages A/B comparison and deployment decisions
"""

import os
import json
import asyncio
import httpx
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from .training_service import training_service
from .feedback_buffer_service import feedback_buffer_service
from .ewc_trainer import EWCConfig
from .tcn_ab_comparison_service import ab_comparison_service, DeploymentRecommendation


class OrchestratorState(str, Enum):
    """Orchestrator states."""
    IDLE = "idle"
    MONITORING = "monitoring"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    ERROR = "error"


class TrainingTrigger(str, Enum):
    """What triggered the training."""
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    BUFFER_READY = "buffer_ready"
    MANUAL = "manual"


@dataclass
class OrchestratorConfig:
    """Configuration for the self-learning orchestrator."""
    # Loop settings
    enabled: bool = True
    check_interval_minutes: int = 60  # Check every hour

    # Time-based scheduling (daily at specific time)
    scheduled_training_enabled: bool = False  # Enable time-based scheduling
    scheduled_hour: int = 2  # Hour (0-23) for scheduled training (default: 02:00)
    scheduled_minute: int = 0  # Minute (0-59) for scheduled training

    # Training triggers
    min_samples_for_training: int = 100
    min_hours_between_training: int = 24
    max_hours_between_training: int = 168  # 1 week

    # Drift thresholds that trigger training
    drift_threshold_for_incremental: str = "high"  # high or critical
    drift_threshold_for_full_retrain: str = "critical"

    # A/B comparison settings
    auto_deploy_on_improvement: bool = True
    require_manual_review_on_regression: bool = True

    # EWC settings
    ewc_lambda: float = 1000.0
    incremental_learning_rate: float = 1e-5
    incremental_epochs: int = 10

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OrchestratorStatus:
    """Current orchestrator status."""
    state: OrchestratorState = OrchestratorState.IDLE
    loop_running: bool = False
    last_check: Optional[str] = None
    last_training: Optional[str] = None
    last_training_trigger: Optional[str] = None
    next_check: Optional[str] = None
    next_scheduled_training: Optional[str] = None  # Next scheduled training time
    training_count: int = 0
    successful_deployments: int = 0
    rejected_models: int = 0
    current_drift_severity: str = "none"
    feedback_buffer_samples: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "loop_running": self.loop_running,
            "last_check": self.last_check,
            "last_training": self.last_training,
            "last_training_trigger": self.last_training_trigger,
            "next_check": self.next_check,
            "next_scheduled_training": self.next_scheduled_training,
            "training_count": self.training_count,
            "successful_deployments": self.successful_deployments,
            "rejected_models": self.rejected_models,
            "current_drift_severity": self.current_drift_severity,
            "feedback_buffer_samples": self.feedback_buffer_samples,
            "error_message": self.error_message
        }


class SelfLearningOrchestrator:
    """
    Orchestrates the self-learning loop for TCN model.

    The orchestrator runs a background loop that:
    1. Checks drift status from TCN inference service
    2. Checks feedback buffer readiness
    3. Triggers training when conditions are met
    4. Runs A/B comparison after training
    5. Deploys or rejects based on comparison results
    """

    STATE_FILE = "data/self_learning_orchestrator_state.json"

    def __init__(self):
        """Initialize the orchestrator."""
        self.config = OrchestratorConfig()
        self.status = OrchestratorStatus()
        self._loop_task: Optional[asyncio.Task] = None
        self._tcn_service_url = os.getenv(
            "TCN_SERVICE_URL",
            "http://trading-tcn:3003"
        )
        self._load_state()

    def _load_state(self) -> None:
        """Load orchestrator state from disk."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)

                    # Load config
                    config_data = data.get("config", {})
                    self.config = OrchestratorConfig(**config_data)

                    # Load status (partial)
                    status_data = data.get("status", {})
                    self.status.last_training = status_data.get("last_training")
                    self.status.last_training_trigger = status_data.get("last_training_trigger")
                    self.status.training_count = status_data.get("training_count", 0)
                    self.status.successful_deployments = status_data.get("successful_deployments", 0)
                    self.status.rejected_models = status_data.get("rejected_models", 0)

                    logger.info("Loaded orchestrator state")
        except Exception as e:
            logger.warning(f"Could not load orchestrator state: {e}")

    def _save_state(self) -> None:
        """Save orchestrator state to disk."""
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(self.STATE_FILE, 'w') as f:
                json.dump({
                    "config": self.config.to_dict(),
                    "status": self.status.to_dict(),
                    "saved_at": datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save orchestrator state: {e}")

    async def start_loop(self) -> bool:
        """Start the self-learning loop."""
        if self.status.loop_running:
            logger.warning("Self-learning loop already running")
            return False

        if not self.config.enabled:
            logger.info("Self-learning is disabled in config")
            return False

        self.status.loop_running = True
        self.status.state = OrchestratorState.MONITORING
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Self-learning loop started")
        return True

    async def stop_loop(self) -> bool:
        """Stop the self-learning loop."""
        if not self.status.loop_running:
            logger.warning("Self-learning loop not running")
            return False

        self.status.loop_running = False
        self.status.state = OrchestratorState.IDLE

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        self._save_state()
        logger.info("Self-learning loop stopped")
        return True

    async def _run_loop(self) -> None:
        """Main self-learning loop."""
        logger.info(f"Self-learning loop running (check interval: {self.config.check_interval_minutes}min)")

        # Update next scheduled training time
        self._update_next_scheduled_training()

        while self.status.loop_running:
            try:
                await self._check_and_train()
            except Exception as e:
                logger.error(f"Error in self-learning loop: {e}")
                self.status.error_message = str(e)
                self.status.state = OrchestratorState.ERROR

            # Calculate next check time
            interval = timedelta(minutes=self.config.check_interval_minutes)
            next_check = datetime.utcnow() + interval
            self.status.next_check = next_check.isoformat()
            self._save_state()

            # Wait for next check
            await asyncio.sleep(self.config.check_interval_minutes * 60)

    def _update_next_scheduled_training(self) -> None:
        """Calculate and update the next scheduled training time."""
        if not self.config.scheduled_training_enabled:
            self.status.next_scheduled_training = None
            return

        now = datetime.utcnow()
        scheduled_time = now.replace(
            hour=self.config.scheduled_hour,
            minute=self.config.scheduled_minute,
            second=0,
            microsecond=0
        )

        # If scheduled time has passed today, schedule for tomorrow
        if scheduled_time <= now:
            scheduled_time += timedelta(days=1)

        self.status.next_scheduled_training = scheduled_time.isoformat()

    def _is_scheduled_training_time(self) -> bool:
        """Check if current time is within the scheduled training window."""
        if not self.config.scheduled_training_enabled:
            return False

        now = datetime.utcnow()

        # Check if we're within 5 minutes of the scheduled time
        scheduled_time = now.replace(
            hour=self.config.scheduled_hour,
            minute=self.config.scheduled_minute,
            second=0,
            microsecond=0
        )

        time_diff = abs((now - scheduled_time).total_seconds())

        # Window of 5 minutes (300 seconds) to catch the scheduled time
        return time_diff <= 300

    async def _check_and_train(self) -> None:
        """Check conditions and trigger training if needed."""
        now = datetime.utcnow()
        self.status.last_check = now.isoformat()
        self.status.state = OrchestratorState.MONITORING
        self.status.error_message = None

        # Check if training is already in progress
        if training_service.is_training():
            logger.debug("Training already in progress, skipping check")
            return

        # Get current conditions
        drift_status = await self._get_drift_status()
        buffer_stats = feedback_buffer_service.get_statistics()
        self.status.feedback_buffer_samples = buffer_stats.total_samples
        self.status.current_drift_severity = drift_status.get("overall_severity", "none")

        # Check minimum time between trainings
        if self.status.last_training:
            last_training_time = datetime.fromisoformat(self.status.last_training)
            hours_since_training = (now - last_training_time).total_seconds() / 3600

            if hours_since_training < self.config.min_hours_between_training:
                logger.debug(
                    f"Too soon since last training ({hours_since_training:.1f}h < {self.config.min_hours_between_training}h)"
                )
                return

        # Determine if training should be triggered
        trigger = self._should_train(drift_status, buffer_stats)

        if trigger:
            await self._trigger_training(trigger)

    async def _get_drift_status(self) -> Dict:
        """Get drift status from TCN inference service."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._tcn_service_url}/api/v1/drift/status"
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.debug(f"Could not get drift status: {e}")

        return {"overall_severity": "none"}

    def _should_train(self, drift_status: Dict, buffer_stats) -> Optional[TrainingTrigger]:
        """Determine if training should be triggered."""
        severity = drift_status.get("overall_severity", "none")

        # Check for critical drift (full retrain)
        if severity == "critical" and self.config.drift_threshold_for_full_retrain == "critical":
            logger.info("Critical drift detected, triggering training")
            return TrainingTrigger.DRIFT_DETECTED

        # Check for high drift (incremental training)
        if severity in ["high", "critical"] and self.config.drift_threshold_for_incremental == "high":
            if buffer_stats.total_samples >= self.config.min_samples_for_training:
                logger.info("High drift detected with sufficient samples, triggering training")
                return TrainingTrigger.DRIFT_DETECTED

        # Check time-based scheduling
        if self._is_scheduled_training_time():
            if buffer_stats.total_samples >= self.config.min_samples_for_training:
                logger.info(
                    f"Scheduled training time reached ({self.config.scheduled_hour:02d}:{self.config.scheduled_minute:02d} UTC)"
                )
                # Update next scheduled time after triggering
                self._update_next_scheduled_training()
                return TrainingTrigger.SCHEDULED
            else:
                logger.debug(
                    f"Scheduled time reached but insufficient samples ({buffer_stats.total_samples} < {self.config.min_samples_for_training})"
                )

        # Check buffer readiness (max interval exceeded)
        if buffer_stats.total_samples >= self.config.min_samples_for_training:
            # Check if max time between trainings exceeded
            if self.status.last_training:
                last_training_time = datetime.fromisoformat(self.status.last_training)
                hours_since = (datetime.utcnow() - last_training_time).total_seconds() / 3600

                if hours_since >= self.config.max_hours_between_training:
                    logger.info(f"Max training interval exceeded ({hours_since:.1f}h), triggering training")
                    return TrainingTrigger.SCHEDULED
            else:
                # First training
                logger.info("Sufficient samples available for first training")
                return TrainingTrigger.BUFFER_READY

        return None

    async def _trigger_training(self, trigger: TrainingTrigger) -> Dict:
        """Trigger incremental training."""
        self.status.state = OrchestratorState.TRAINING
        self.status.last_training_trigger = trigger.value
        self._save_state()

        logger.info(f"Triggering incremental training (reason: {trigger.value})")

        # Get training batch from feedback buffer
        batch = feedback_buffer_service.get_training_batch(
            batch_size=feedback_buffer_service.get_statistics().total_samples,
            stratified=True
        )

        if not batch:
            logger.warning("No samples available for training")
            self.status.state = OrchestratorState.MONITORING
            return {"status": "skipped", "reason": "No samples available"}

        # Prepare training data
        import numpy as np
        sequences = np.array([s.ohlcv_sequence for s in batch])
        labels = np.array([s.label_vector for s in batch])
        sample_ids = [s.sample_id for s in batch]

        config = EWCConfig(
            ewc_lambda=self.config.ewc_lambda,
            learning_rate=self.config.incremental_learning_rate,
            epochs=self.config.incremental_epochs,
            min_samples=self.config.min_samples_for_training
        )

        # Run incremental training
        result = await training_service.incremental_train(
            sequences=sequences,
            labels=labels,
            config=config
        )

        if result.get("status") == "completed":
            # Mark samples as used
            feedback_buffer_service.mark_as_used(sample_ids)

            self.status.training_count += 1
            self.status.last_training = datetime.utcnow().isoformat()

            # If we had a candidate path, we'd run A/B comparison here
            # For now, the model is deployed automatically via training_service

            self.status.successful_deployments += 1

            logger.info(f"Incremental training completed: {result}")
        else:
            logger.warning(f"Training did not complete successfully: {result}")

        self.status.state = OrchestratorState.MONITORING
        self._save_state()

        return result

    async def trigger_manual_training(self, full_retrain: bool = False) -> Dict:
        """
        Manually trigger training.

        Args:
            full_retrain: If True, trigger full retraining instead of incremental

        Returns:
            Training result
        """
        if training_service.is_training():
            return {"status": "failed", "message": "Training already in progress"}

        if full_retrain:
            # TODO: Implement full retraining trigger
            return {"status": "not_implemented", "message": "Full retraining not yet implemented"}

        return await self._trigger_training(TrainingTrigger.MANUAL)

    def get_status(self) -> OrchestratorStatus:
        """Get current orchestrator status."""
        # Update dynamic fields
        self.status.feedback_buffer_samples = feedback_buffer_service.get_statistics().total_samples
        return self.status

    def get_config(self) -> OrchestratorConfig:
        """Get current configuration."""
        return self.config

    def update_config(
        self,
        enabled: Optional[bool] = None,
        check_interval_minutes: Optional[int] = None,
        scheduled_training_enabled: Optional[bool] = None,
        scheduled_hour: Optional[int] = None,
        scheduled_minute: Optional[int] = None,
        min_samples_for_training: Optional[int] = None,
        min_hours_between_training: Optional[int] = None,
        max_hours_between_training: Optional[int] = None,
        auto_deploy_on_improvement: Optional[bool] = None,
        ewc_lambda: Optional[float] = None,
        incremental_learning_rate: Optional[float] = None,
        incremental_epochs: Optional[int] = None
    ) -> OrchestratorConfig:
        """Update orchestrator configuration."""
        if enabled is not None:
            self.config.enabled = enabled
        if check_interval_minutes is not None:
            self.config.check_interval_minutes = check_interval_minutes
        if scheduled_training_enabled is not None:
            self.config.scheduled_training_enabled = scheduled_training_enabled
        if scheduled_hour is not None:
            # Validate hour (0-23)
            self.config.scheduled_hour = max(0, min(23, scheduled_hour))
        if scheduled_minute is not None:
            # Validate minute (0-59)
            self.config.scheduled_minute = max(0, min(59, scheduled_minute))
        if min_samples_for_training is not None:
            self.config.min_samples_for_training = min_samples_for_training
        if min_hours_between_training is not None:
            self.config.min_hours_between_training = min_hours_between_training
        if max_hours_between_training is not None:
            self.config.max_hours_between_training = max_hours_between_training
        if auto_deploy_on_improvement is not None:
            self.config.auto_deploy_on_improvement = auto_deploy_on_improvement
        if ewc_lambda is not None:
            self.config.ewc_lambda = ewc_lambda
        if incremental_learning_rate is not None:
            self.config.incremental_learning_rate = incremental_learning_rate
        if incremental_epochs is not None:
            self.config.incremental_epochs = incremental_epochs

        # Update next scheduled training time if scheduling changed
        if scheduled_training_enabled is not None or scheduled_hour is not None or scheduled_minute is not None:
            self._update_next_scheduled_training()

        self._save_state()
        return self.config

    def get_history(self) -> List[Dict]:
        """Get training history from the orchestrator's perspective."""
        return ab_comparison_service.get_history(limit=20)


# Singleton instance
self_learning_orchestrator = SelfLearningOrchestrator()
