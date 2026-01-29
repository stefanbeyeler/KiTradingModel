"""
Self-Learning Monitor Service for HMM Training.

Implements closed-loop feedback by monitoring accuracy from the
HMM Inference service and automatically triggering re-training
when accuracy drops below threshold.
"""

import asyncio
from typing import Optional, Dict
from datetime import datetime, timezone
from dataclasses import dataclass
from loguru import logger
import httpx

from src.config.microservices import microservices_config


@dataclass
class SelfLearningStatus:
    """Status of the self-learning monitor."""
    enabled: bool = True
    monitor_running: bool = False
    check_interval_seconds: int = 300  # 5 minutes
    last_check: Optional[datetime] = None
    last_accuracy: Optional[float] = None
    last_retrain_trigger: Optional[datetime] = None
    pending_retrain: bool = False
    inference_service_url: str = microservices_config.hmm_service_url


class SelfLearningService:
    """
    Self-Learning Monitor for HMM models.

    Features:
    - Periodic accuracy monitoring from HMM Inference service
    - Automatic retrain trigger when accuracy drops
    - Cooldown management
    - Status reporting
    """

    def __init__(self):
        """Initialize the self-learning service."""
        self._status = SelfLearningStatus()
        self._monitor_task: Optional[asyncio.Task] = None
        self._training_service = None  # Set lazily

    @property
    def status(self) -> Dict:
        """Get current status as dict."""
        return {
            "enabled": self._status.enabled,
            "monitor_running": self._status.monitor_running,
            "check_interval_seconds": self._status.check_interval_seconds,
            "last_check": self._status.last_check.isoformat() if self._status.last_check else None,
            "last_accuracy": self._status.last_accuracy,
            "last_retrain_trigger": self._status.last_retrain_trigger.isoformat() if self._status.last_retrain_trigger else None,
            "pending_retrain": self._status.pending_retrain,
            "inference_service_url": self._status.inference_service_url
        }

    async def start_monitor(self) -> None:
        """Start the background accuracy monitor."""
        if self._monitor_task and not self._monitor_task.done():
            logger.warning("Self-learning monitor already running")
            return

        self._status.enabled = True
        self._status.monitor_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Self-learning monitor started")

    async def stop_monitor(self) -> None:
        """Stop the background accuracy monitor."""
        self._status.enabled = False
        self._status.monitor_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Self-learning monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info(f"Self-learning monitor loop started, checking every {self._status.check_interval_seconds}s")

        while self._status.enabled:
            try:
                await self._check_and_trigger()
            except Exception as e:
                logger.error(f"Self-learning monitor error: {e}")

            await asyncio.sleep(self._status.check_interval_seconds)

    async def _check_and_trigger(self) -> None:
        """Check accuracy and trigger retrain if needed."""
        self._status.last_check = datetime.now(timezone.utc)

        try:
            # Check if retrain is needed
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._status.inference_service_url}/api/v1/accuracy/should-retrain"
                )

                if response.status_code != 200:
                    logger.warning(f"Failed to check accuracy: {response.status_code}")
                    return

                data = response.json()

            self._status.last_accuracy = data.get("current_accuracy")

            if data.get("should_retrain"):
                logger.info(f"Self-learning trigger: {data.get('reason')}")
                await self._trigger_retrain()
            else:
                logger.debug(f"Self-learning check OK: accuracy={self._status.last_accuracy:.1%}")

        except httpx.RequestError as e:
            logger.warning(f"Could not reach HMM Inference service: {e}")
        except Exception as e:
            logger.error(f"Self-learning check failed: {e}")

    async def _trigger_retrain(self) -> None:
        """Trigger a re-training job."""
        if self._status.pending_retrain:
            logger.info("Retrain already pending, skipping")
            return

        self._status.pending_retrain = True

        try:
            # Import training service lazily to avoid circular imports
            if self._training_service is None:
                from .training_service import training_service
                self._training_service = training_service

            # Start training job
            logger.info("Self-learning: Starting automatic retrain job")

            result = await self._training_service.start_training(
                symbols=[],  # All symbols
                train_hmm=True,
                train_scorer=True,
                timeframe="1h",
                lookback_days=365
            )

            if result.get("status") == "started":
                self._status.last_retrain_trigger = datetime.now(timezone.utc)

                # Mark retrain on inference side
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(
                            f"{self._status.inference_service_url}/api/v1/accuracy/mark-retrain"
                        )
                except Exception as e:
                    logger.warning(f"Could not mark retrain on inference service: {e}")

                logger.info(f"Self-learning: Training job started: {result.get('job_id')}")
            else:
                logger.warning(f"Self-learning: Failed to start training: {result}")

        except Exception as e:
            logger.error(f"Self-learning retrain trigger failed: {e}")
        finally:
            self._status.pending_retrain = False

    async def manual_check(self) -> Dict:
        """
        Manually trigger an accuracy check.

        Returns:
            Check result including whether retrain is needed
        """
        self._status.last_check = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._status.inference_service_url}/api/v1/accuracy/should-retrain"
                )

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"Failed to check accuracy: {response.status_code}"
                    }

                data = response.json()

            self._status.last_accuracy = data.get("current_accuracy")

            return {
                "status": "ok",
                "should_retrain": data.get("should_retrain"),
                "reason": data.get("reason"),
                "current_accuracy": data.get("current_accuracy"),
                "threshold": data.get("threshold"),
                "evaluated_predictions": data.get("evaluated_predictions")
            }

        except httpx.RequestError as e:
            return {
                "status": "error",
                "message": f"Could not reach HMM Inference service: {e}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def force_retrain(self) -> Dict:
        """
        Force a retrain regardless of accuracy.

        Returns:
            Training job result
        """
        logger.info("Self-learning: Force retrain requested")
        await self._trigger_retrain()

        return {
            "status": "ok",
            "message": "Retrain triggered",
            "last_retrain_trigger": self._status.last_retrain_trigger.isoformat() if self._status.last_retrain_trigger else None
        }

    def set_config(
        self,
        enabled: Optional[bool] = None,
        check_interval_seconds: Optional[int] = None,
        inference_service_url: Optional[str] = None
    ) -> Dict:
        """
        Update self-learning configuration.

        Args:
            enabled: Enable/disable self-learning
            check_interval_seconds: Interval between checks
            inference_service_url: URL of HMM Inference service

        Returns:
            Updated configuration
        """
        if enabled is not None:
            self._status.enabled = enabled
            if not enabled and self._monitor_task:
                # Will stop on next loop iteration
                pass

        if check_interval_seconds is not None:
            self._status.check_interval_seconds = max(60, min(3600, check_interval_seconds))

        if inference_service_url is not None:
            self._status.inference_service_url = inference_service_url

        return {
            "enabled": self._status.enabled,
            "check_interval_seconds": self._status.check_interval_seconds,
            "inference_service_url": self._status.inference_service_url
        }


# Global singleton
self_learning_service = SelfLearningService()
