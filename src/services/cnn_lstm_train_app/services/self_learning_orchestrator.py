"""
Self-Learning Orchestrator for CNN-LSTM.

Coordinates the self-learning loop:
1. Monitor feedback buffer
2. Trigger incremental training when conditions met
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
from .cnn_lstm_validation_service import validation_service, ValidationRecommendation
from .cnn_lstm_rollback_service import rollback_service, RollbackReason


class OrchestratorState(Enum):
    """States of the self-learning orchestrator."""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYING = "deploying"


@dataclass
class OrchestratorConfig:
    """Configuration for the self-learning orchestrator."""
    enabled: bool = True
    check_interval_minutes: int = 60

    # Time-based scheduling
    scheduled_training_enabled: bool = True
    scheduled_hour: int = 3  # UTC
    scheduled_minute: int = 0

    # Trigger conditions
    min_samples_for_training: int = 100
    min_hours_between_trainings: int = 24
    max_hours_between_trainings: int = 168  # 7 days

    # Training settings
    incremental_epochs: int = 5
    incremental_learning_rate: float = 1e-5
    ewc_lambda: float = 1000.0  # Elastic Weight Consolidation strength

    # Validation settings
    auto_deploy_enabled: bool = False  # Require manual review by default
    min_accuracy_improvement: float = 0.0  # Allow any improvement
    max_accuracy_regression: float = 0.05  # 5% max regression allowed

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
    last_training: Optional[str] = None
    last_training_trigger: Optional[str] = None
    next_check: Optional[str] = None
    next_scheduled_training: Optional[str] = None
    training_count: int = 0
    successful_trainings: int = 0
    rejected_trainings: int = 0
    feedback_buffer_samples: int = 0
    current_model_version: Optional[str] = None
    candidate_model_version: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


class SelfLearningOrchestrator:
    """
    Orchestrates the self-learning loop for CNN-LSTM.

    Monitors the feedback buffer, triggers incremental training when conditions
    are met, validates new models, and coordinates deployment.
    """

    def __init__(self, state_file: str = None):
        data_dir = os.getenv("DATA_DIR", "/app/data")
        if state_file is None:
            state_file = os.path.join(data_dir, "cnn_lstm_self_learning_state.json")

        self._state_file = Path(state_file)
        self._config = OrchestratorConfig()
        self._status = OrchestratorStatus()

        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

        # Service URLs
        self._cnn_lstm_service_url = os.getenv(
            "CNN_LSTM_SERVICE_URL",
            "http://trading-cnn-lstm:3007"
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
                await self._check_and_train()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._status.error_message = str(e)

            # Calculate next check time
            now = datetime.now(timezone.utc)
            next_check = now + timedelta(minutes=self._config.check_interval_minutes)
            self._status.next_check = next_check.isoformat()

            # Calculate next scheduled training
            if self._config.scheduled_training_enabled:
                next_scheduled = self._calculate_next_scheduled_time()
                self._status.next_scheduled_training = next_scheduled.isoformat()

            self._save_state()

            await asyncio.sleep(self._config.check_interval_minutes * 60)

    def _calculate_next_scheduled_time(self) -> datetime:
        """Calculate the next scheduled training time."""
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

    async def _check_and_train(self):
        """Check conditions and trigger training if needed."""
        now = datetime.now(timezone.utc)
        self._status.last_check = now.isoformat()
        self._status.state = OrchestratorState.MONITORING

        # Get buffer statistics
        buffer_stats = feedback_buffer_service.get_statistics()
        self._status.feedback_buffer_samples = buffer_stats.unused_samples

        # Check if training should be triggered
        should_train, trigger_reason = self._should_train(buffer_stats)

        if should_train:
            await self._trigger_training(trigger_reason)

        self._save_state()

    def _should_train(self, buffer_stats) -> tuple[bool, str]:
        """Determine if training should be triggered."""
        if not self._config.enabled:
            return False, ""

        now = datetime.now(timezone.utc)

        # Check minimum samples
        if buffer_stats.unused_samples < self._config.min_samples_for_training:
            return False, ""

        # Check minimum time since last training
        if self._status.last_training:
            last_train = datetime.fromisoformat(self._status.last_training.replace('Z', '+00:00'))
            hours_since = (now - last_train).total_seconds() / 3600

            if hours_since < self._config.min_hours_between_trainings:
                return False, ""

            # Force training if max interval exceeded
            if hours_since >= self._config.max_hours_between_trainings:
                return True, "max_interval_exceeded"

        # Check scheduled time
        if self._config.scheduled_training_enabled:
            scheduled = self._calculate_next_scheduled_time() - timedelta(days=1)
            time_diff = abs((now - scheduled).total_seconds())

            # Within 5 minute window of scheduled time
            if time_diff < 300:
                return True, "scheduled"

        # Buffer is ready and conditions met
        if buffer_stats.ready_for_training:
            return True, "buffer_ready"

        return False, ""

    async def _trigger_training(self, trigger_reason: str):
        """Trigger the incremental training process."""
        logger.info(f"Triggering training (reason: {trigger_reason})")

        self._status.state = OrchestratorState.ANALYZING
        self._status.last_training_trigger = trigger_reason
        self._save_state()

        try:
            # Get training batch from feedback buffer
            training_batch = feedback_buffer_service.get_training_batch(
                batch_size=self._config.min_samples_for_training
            )

            if not training_batch:
                logger.info("No training samples available")
                self._status.state = OrchestratorState.MONITORING
                return

            logger.info(f"Starting incremental training with {len(training_batch)} samples")

            self._status.state = OrchestratorState.TRAINING
            self._save_state()

            # Prepare training request
            training_data = {
                "samples": [s.to_dict() for s in training_batch],
                "config": {
                    "epochs": self._config.incremental_epochs,
                    "learning_rate": self._config.incremental_learning_rate,
                    "ewc_lambda": self._config.ewc_lambda,
                },
                "trigger_reason": trigger_reason,
            }

            # Call the incremental training endpoint
            result = await self._run_incremental_training(training_data)

            if result.get("status") == "completed":
                self._status.training_count += 1
                self._status.last_training = datetime.now(timezone.utc).isoformat()
                model_version = result.get("model_version")
                model_path = result.get("model_path")
                self._status.candidate_model_version = model_version

                # Register model with rollback service
                if model_path and model_version:
                    rollback_service.register_model(
                        model_path=model_path,
                        training_type="incremental",
                        metrics=result.get("metrics", {}),
                        notes=f"Incremental training triggered by: {trigger_reason}"
                    )
                    logger.info(f"Registered model {model_version} with rollback service")

                # Mark samples as used
                sample_ids = [s.sample_id for s in training_batch]
                feedback_buffer_service.mark_as_used(sample_ids)

                # Validate the new model
                if result.get("validation"):
                    validation_passed = await self._validate_model(result["validation"])

                    if validation_passed:
                        self._status.successful_trainings += 1

                        if self._config.auto_deploy_enabled:
                            await self._deploy_model()
                        else:
                            logger.info("Model ready for manual review/deployment")
                    else:
                        self._status.rejected_trainings += 1
                        logger.warning("Model validation failed - not deploying")
                else:
                    # No validation data, consider it successful
                    self._status.successful_trainings += 1

                logger.info(f"Incremental training completed: {result}")
            else:
                logger.warning(f"Incremental training failed: {result}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._status.error_message = str(e)

        finally:
            self._status.state = OrchestratorState.MONITORING
            self._save_state()

    async def _run_incremental_training(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run incremental training via the training service."""
        try:
            # For now, this is a placeholder - actual training would be done locally
            # or via an internal method

            from .training_service import training_service

            result = await training_service.incremental_train(
                samples=training_data["samples"],
                config=training_data["config"],
            )

            return result

        except Exception as e:
            logger.error(f"Failed to run incremental training: {e}")
            return {"status": "error", "message": str(e)}

    async def _validate_model(self, validation_data: Dict[str, Any]) -> bool:
        """Validate the newly trained model using A/B comparison."""
        self._status.state = OrchestratorState.VALIDATING
        self._save_state()

        try:
            candidate_path = validation_data.get("model_path")
            if not candidate_path:
                logger.warning("No candidate model path for validation")
                return self._validate_basic(validation_data)

            # Get production model path
            current_version = rollback_service.get_current_version()
            if not current_version:
                # No production model, use basic validation
                logger.info("No production model to compare - using basic validation")
                return self._validate_basic(validation_data)

            production_path = current_version.get("model_path")

            # Prepare test data from feedback buffer
            buffer_stats = feedback_buffer_service.get_statistics()
            if buffer_stats.unused_samples < 20:
                logger.info("Insufficient samples for A/B comparison - using basic validation")
                return self._validate_basic(validation_data)

            # Get test samples
            test_batch = feedback_buffer_service.get_training_batch(
                batch_size=min(100, buffer_stats.unused_samples)
            )

            test_data = [
                {
                    "features": s.get("ohlcv_context", []),
                    "price_label": [1 if s.get("price_direction_correct", False) else 0],
                    "pattern_labels": s.get("pattern_predictions", [0] * 16),
                    "regime_label": s.get("regime_prediction", 0),
                    "outcome": {"is_success": s.get("overall_success", False)}
                }
                for s in test_batch
            ]

            # Run A/B comparison
            comparison = validation_service.compare_models(
                production_path=production_path,
                candidate_path=candidate_path,
                test_data=test_data
            )

            logger.info(
                f"A/B Comparison: {comparison.recommendation.value} "
                f"(accuracy delta: {comparison.overall_accuracy_delta:+.3f}, "
                f"F1 delta: {comparison.overall_f1_delta:+.3f})"
            )

            if comparison.recommendation == ValidationRecommendation.DEPLOY:
                logger.info(f"Validation PASSED: {comparison.recommendation_reason}")
                return True
            elif comparison.recommendation == ValidationRecommendation.REJECT:
                logger.warning(f"Validation REJECTED: {comparison.recommendation_reason}")
                return False
            else:
                # Manual review - check basic thresholds
                logger.info(f"Validation REVIEW: {comparison.recommendation_reason}")
                return comparison.overall_accuracy_delta >= -self._config.max_accuracy_regression

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return self._validate_basic(validation_data)

    def _validate_basic(self, validation_data: Dict[str, Any]) -> bool:
        """Basic validation when A/B comparison is not possible."""
        accuracy_change = validation_data.get("accuracy_change", 0.0)

        if accuracy_change >= self._config.min_accuracy_improvement:
            logger.info(f"Basic validation passed: accuracy change {accuracy_change:+.3f}")
            return True

        if accuracy_change >= -self._config.max_accuracy_regression:
            logger.info(f"Basic validation passed (minor regression): {accuracy_change:+.3f}")
            return True

        logger.warning(f"Basic validation failed: accuracy change {accuracy_change:+.3f}")
        return False

    async def _deploy_model(self):
        """Deploy the validated model using the rollback service."""
        self._status.state = OrchestratorState.DEPLOYING
        self._save_state()

        try:
            candidate_version = self._status.candidate_model_version
            if not candidate_version:
                logger.warning("No candidate model version to deploy")
                return

            # Deploy via rollback service (handles versioning, backup, and hot-reload)
            result = await rollback_service.deploy_model(
                version_id=candidate_version,
                force=False
            )

            if result.get("status") == "deployed":
                self._status.current_model_version = candidate_version
                self._status.candidate_model_version = None
                logger.info(f"Model {candidate_version} deployed successfully via rollback service")

                # Log deployment info
                if result.get("previous_version"):
                    logger.info(f"Previous version: {result['previous_version']}")
                if result.get("inference_reloaded"):
                    logger.info("Inference service reloaded")
            else:
                logger.warning(f"Model deployment failed: {result.get('message', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")

    async def trigger_manual_training(self) -> Dict[str, Any]:
        """Manually trigger training."""
        buffer_stats = feedback_buffer_service.get_statistics()

        if buffer_stats.unused_samples < self._config.min_samples_for_training:
            return {
                "status": "skipped",
                "reason": f"Insufficient samples ({buffer_stats.unused_samples} < {self._config.min_samples_for_training})"
            }

        await self._trigger_training("manual")

        return {
            "status": "completed",
            "training_count": self._status.training_count,
            "candidate_version": self._status.candidate_model_version
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

    async def handle_drift_alert(self, drift_severity: str, drift_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a drift alert from the inference service.

        If drift is critical, triggers automatic rollback.

        Args:
            drift_severity: Severity level (none, low, medium, high, critical)
            drift_details: Details about the detected drift

        Returns:
            Action taken
        """
        logger.info(f"Received drift alert: severity={drift_severity}")

        if drift_severity == "critical":
            logger.warning("Critical drift detected - triggering automatic rollback")

            result = await rollback_service.auto_rollback_on_drift(
                drift_severity=drift_severity,
                drift_details=drift_details
            )

            if result.get("status") == "deployed":
                self._status.current_model_version = result.get("version_id")
                self._save_state()
                logger.info(f"Automatic rollback successful: {result}")
            else:
                logger.warning(f"Automatic rollback skipped/failed: {result}")

            return {
                "action": "automatic_rollback",
                "result": result
            }

        elif drift_severity == "high":
            # High drift - schedule urgent retraining
            logger.warning("High drift detected - scheduling urgent retraining")

            buffer_stats = feedback_buffer_service.get_statistics()
            if buffer_stats.unused_samples >= self._config.min_samples_for_training // 2:
                # Lower threshold for urgent training
                await self._trigger_training("high_drift")
                return {"action": "urgent_retraining_triggered"}

            return {"action": "urgent_retraining_pending", "reason": "insufficient_samples"}

        elif drift_severity == "medium":
            logger.info("Medium drift detected - monitoring closely")
            return {"action": "monitoring", "note": "Will retrain if samples available"}

        return {"action": "none", "severity": drift_severity}


# Global singleton
self_learning_orchestrator = SelfLearningOrchestrator()
