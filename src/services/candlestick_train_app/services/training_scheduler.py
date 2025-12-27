"""Training Scheduler - Scheduled model training for candlestick patterns."""

import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

from loguru import logger

from .training_service import training_service


@dataclass
class ScheduleConfig:
    """Schedule configuration."""
    enabled: bool = False
    interval_hours: int = 24
    training_hour: int = 2  # UTC hour to run training
    min_symbols: int = 5
    epochs: int = 50
    last_run: Optional[str] = None
    next_run: Optional[str] = None


class TrainingScheduler:
    """
    Scheduler for automatic model training.

    Can be configured to run training on a schedule (e.g., daily at 2 AM UTC).
    """

    def __init__(self):
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._config = ScheduleConfig(
            enabled=os.getenv("SCHEDULED_TRAINING_ENABLED", "false").lower() == "true",
            interval_hours=int(os.getenv("TRAINING_INTERVAL_HOURS", "24")),
            training_hour=int(os.getenv("TRAINING_HOUR_UTC", "2")),
            epochs=int(os.getenv("SCHEDULED_TRAINING_EPOCHS", "50")),
        )

        logger.info(
            f"TrainingScheduler initialized (enabled: {self._config.enabled}, "
            f"interval: {self._config.interval_hours}h)"
        )

    def _calculate_next_run(self) -> datetime:
        """Calculate next scheduled run time."""
        now = datetime.now(timezone.utc)

        # Calculate next run at the specified hour
        next_run = now.replace(
            hour=self._config.training_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        # If we've passed today's scheduled time, schedule for tomorrow
        if now >= next_run:
            next_run += timedelta(days=1)

        return next_run

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                next_run = self._calculate_next_run()
                self._config.next_run = next_run.isoformat()

                # Calculate wait time
                wait_seconds = (next_run - now).total_seconds()

                if wait_seconds > 0:
                    logger.info(f"Next scheduled training at {next_run.isoformat()}")
                    await asyncio.sleep(min(wait_seconds, 3600))  # Check every hour max
                    continue

                # Time to run training
                if not training_service.is_training():
                    logger.info("Starting scheduled training...")
                    try:
                        await training_service.start_training(
                            epochs=self._config.epochs,
                        )
                        self._config.last_run = datetime.now(timezone.utc).isoformat()
                    except Exception as e:
                        logger.error(f"Scheduled training failed: {e}")

                # Wait until next interval
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                logger.info("Scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)

    async def start(self):
        """Start the scheduler."""
        if self._running:
            return

        if not self._config.enabled:
            logger.info("Scheduled training is disabled")
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Training scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Training scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def get_config(self) -> Dict[str, Any]:
        """Get current scheduler configuration."""
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "interval_hours": self._config.interval_hours,
            "training_hour_utc": self._config.training_hour,
            "epochs": self._config.epochs,
            "last_run": self._config.last_run,
            "next_run": self._config.next_run,
        }

    def update_config(
        self,
        enabled: Optional[bool] = None,
        interval_hours: Optional[int] = None,
        training_hour: Optional[int] = None,
        epochs: Optional[int] = None
    ):
        """Update scheduler configuration."""
        if enabled is not None:
            self._config.enabled = enabled
        if interval_hours is not None:
            self._config.interval_hours = interval_hours
        if training_hour is not None:
            self._config.training_hour = training_hour
        if epochs is not None:
            self._config.epochs = epochs

        logger.info(f"Scheduler config updated: {self.get_config()}")


# Global singleton
training_scheduler = TrainingScheduler()
