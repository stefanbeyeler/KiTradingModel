"""Scheduler Service for HMM Training.

Manages training schedules independently within the HMM-Train service.
This replaces the Watchdog-based scheduling for HMM training.
"""

import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from loguru import logger


class ScheduleInterval(str, Enum):
    """Training schedule intervals."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"


@dataclass
class TrainingSchedule:
    """A scheduled training configuration."""
    schedule_id: str
    interval: ScheduleInterval
    enabled: bool = True
    symbols: Optional[List[str]] = None  # None = all symbols
    timeframe: str = "1h"
    lookback_days: int = 365
    train_hmm: bool = True
    train_scorer: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_status: Optional[str] = None  # "success", "failed", "running"
    last_job_id: Optional[str] = None
    custom_hour: int = 3  # Hour of day for daily/weekly (UTC)
    custom_weekday: int = 0  # 0=Monday for weekly schedule

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "interval": self.interval.value
        }


class SchedulerService:
    """
    HMM Training Scheduler Service.

    Features:
    - Independent scheduling for HMM training
    - Configurable intervals: hourly, daily, weekly
    - Persistent schedule storage
    - Background scheduler worker
    """

    def __init__(self):
        self._schedules: Dict[str, TrainingSchedule] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._training_service = None  # Lazy import to avoid circular dependency

        # Persistence
        self._state_file = Path(os.getenv(
            "HMM_SCHEDULER_STATE_FILE",
            "/app/data/models/hmm/scheduler_state.json"
        ))
        self._load_state()

        logger.info("HMM SchedulerService initialized")

    def _get_training_service(self):
        """Lazy import of training service to avoid circular dependency."""
        if self._training_service is None:
            from .training_service import training_service
            self._training_service = training_service
        return self._training_service

    def _load_state(self):
        """Load persisted scheduler state."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)

                for schedule_data in state.get("schedules", []):
                    schedule_data["interval"] = ScheduleInterval(schedule_data["interval"])
                    schedule = TrainingSchedule(**schedule_data)
                    self._schedules[schedule.schedule_id] = schedule

                logger.info(f"Loaded {len(self._schedules)} schedules from state file")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")

    def _save_state(self):
        """Persist scheduler state to file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "schedules": [s.to_dict() for s in self._schedules.values()],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Saved scheduler state with {len(self._schedules)} schedules")
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")

    def _calculate_next_run(self, schedule: TrainingSchedule, from_time: Optional[datetime] = None) -> datetime:
        """Calculate next run time based on schedule configuration."""
        now = from_time or datetime.now(timezone.utc)

        if schedule.interval == ScheduleInterval.HOURLY:
            # Next hour at minute 0
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        elif schedule.interval == ScheduleInterval.DAILY:
            # Next occurrence of specified hour
            next_run = now.replace(
                hour=schedule.custom_hour,
                minute=0,
                second=0,
                microsecond=0
            )
            if next_run <= now:
                next_run += timedelta(days=1)

        elif schedule.interval == ScheduleInterval.WEEKLY:
            # Next occurrence of specified weekday and hour
            days_ahead = schedule.custom_weekday - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.hour >= schedule.custom_hour):
                days_ahead += 7

            next_run = now.replace(
                hour=schedule.custom_hour,
                minute=0,
                second=0,
                microsecond=0
            ) + timedelta(days=days_ahead)

        else:
            # Custom interval - default to daily
            next_run = now + timedelta(days=1)

        return next_run

    async def start(self):
        """Start the scheduler background worker."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_worker())
        logger.info("HMM Scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info("HMM Scheduler stopped")

    async def _scheduler_worker(self):
        """Background worker that checks and executes scheduled training."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                for schedule in list(self._schedules.values()):
                    if not schedule.enabled:
                        continue

                    # Check if training is due
                    if schedule.next_run:
                        next_run = datetime.fromisoformat(schedule.next_run)
                        if now >= next_run:
                            await self._execute_scheduled_training(schedule)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler worker: {e}")
                await asyncio.sleep(60)

    async def _execute_scheduled_training(self, schedule: TrainingSchedule):
        """Execute a scheduled training job."""
        training_service = self._get_training_service()

        # Check if training is already running
        if training_service.is_training():
            logger.warning(f"Skipping scheduled training {schedule.schedule_id} - training already in progress")
            # Don't update next_run yet, try again in next cycle
            return

        try:
            logger.info(f"Executing scheduled training: {schedule.schedule_id}")
            schedule.last_status = "running"
            self._save_state()

            # Determine model type
            from .training_service import ModelType
            if schedule.train_hmm and schedule.train_scorer:
                model_type = ModelType.BOTH
            elif schedule.train_hmm:
                model_type = ModelType.HMM
            else:
                model_type = ModelType.SCORER

            # Start training
            job = await training_service.start_training(
                model_type=model_type,
                symbols=schedule.symbols,
                timeframe=schedule.timeframe,
                lookback_days=schedule.lookback_days
            )

            schedule.last_job_id = job.job_id
            schedule.last_run = datetime.now(timezone.utc).isoformat()
            schedule.next_run = self._calculate_next_run(schedule).isoformat()
            schedule.last_status = "running"

            logger.info(
                f"Scheduled training started: {job.job_id}, "
                f"next run: {schedule.next_run}"
            )

        except Exception as e:
            logger.error(f"Failed to execute scheduled training: {e}")
            schedule.last_status = "failed"
            # Still update next_run to prevent continuous retries
            schedule.next_run = self._calculate_next_run(schedule).isoformat()

        finally:
            self._save_state()

    def create_schedule(
        self,
        interval: str = "daily",
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        lookback_days: int = 365,
        train_hmm: bool = True,
        train_scorer: bool = True,
        custom_hour: int = 3,
        custom_weekday: int = 0,
        enabled: bool = True
    ) -> TrainingSchedule:
        """Create a new training schedule."""
        # Parse interval
        try:
            schedule_interval = ScheduleInterval(interval.lower())
        except ValueError:
            schedule_interval = ScheduleInterval.DAILY

        # Generate unique ID
        schedule_id = f"hmm_schedule_{len(self._schedules)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        schedule = TrainingSchedule(
            schedule_id=schedule_id,
            interval=schedule_interval,
            enabled=enabled,
            symbols=symbols,
            timeframe=timeframe,
            lookback_days=lookback_days,
            train_hmm=train_hmm,
            train_scorer=train_scorer,
            custom_hour=custom_hour,
            custom_weekday=custom_weekday
        )

        # Calculate next run
        schedule.next_run = self._calculate_next_run(schedule).isoformat()

        self._schedules[schedule_id] = schedule
        self._save_state()

        logger.info(f"Created schedule: {schedule_id}, next run: {schedule.next_run}")
        return schedule

    def update_schedule(
        self,
        schedule_id: str,
        interval: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        lookback_days: Optional[int] = None,
        train_hmm: Optional[bool] = None,
        train_scorer: Optional[bool] = None,
        custom_hour: Optional[int] = None,
        custom_weekday: Optional[int] = None,
        enabled: Optional[bool] = None
    ) -> Optional[TrainingSchedule]:
        """Update an existing schedule."""
        if schedule_id not in self._schedules:
            return None

        schedule = self._schedules[schedule_id]

        if interval is not None:
            try:
                schedule.interval = ScheduleInterval(interval.lower())
            except ValueError:
                pass

        if symbols is not None:
            schedule.symbols = symbols if symbols else None  # Empty list = all symbols
        if timeframe is not None:
            schedule.timeframe = timeframe
        if lookback_days is not None:
            schedule.lookback_days = lookback_days
        if train_hmm is not None:
            schedule.train_hmm = train_hmm
        if train_scorer is not None:
            schedule.train_scorer = train_scorer
        if custom_hour is not None:
            schedule.custom_hour = custom_hour
        if custom_weekday is not None:
            schedule.custom_weekday = custom_weekday
        if enabled is not None:
            schedule.enabled = enabled

        # Recalculate next run
        schedule.next_run = self._calculate_next_run(schedule).isoformat()

        self._save_state()
        logger.info(f"Updated schedule: {schedule_id}")
        return schedule

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            self._save_state()
            logger.info(f"Deleted schedule: {schedule_id}")
            return True
        return False

    def get_schedule(self, schedule_id: str) -> Optional[TrainingSchedule]:
        """Get a specific schedule."""
        return self._schedules.get(schedule_id)

    def get_all_schedules(self) -> List[TrainingSchedule]:
        """Get all schedules."""
        return list(self._schedules.values())

    def toggle_schedule(self, schedule_id: str, enabled: bool) -> Optional[TrainingSchedule]:
        """Enable or disable a schedule."""
        if schedule_id not in self._schedules:
            return None

        schedule = self._schedules[schedule_id]
        schedule.enabled = enabled

        if enabled:
            # Recalculate next run when re-enabling
            schedule.next_run = self._calculate_next_run(schedule).isoformat()

        self._save_state()
        logger.info(f"{'Enabled' if enabled else 'Disabled'} schedule: {schedule_id}")
        return schedule

    def trigger_now(self, schedule_id: str) -> bool:
        """Trigger a schedule to run immediately."""
        if schedule_id not in self._schedules:
            return False

        schedule = self._schedules[schedule_id]

        # Set next_run to now to trigger in next worker cycle
        schedule.next_run = datetime.now(timezone.utc).isoformat()
        self._save_state()

        logger.info(f"Triggered immediate run for schedule: {schedule_id}")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        now = datetime.now(timezone.utc)
        active_schedules = [s for s in self._schedules.values() if s.enabled]

        # Find next scheduled run
        next_scheduled = None
        for schedule in active_schedules:
            if schedule.next_run:
                next_run = datetime.fromisoformat(schedule.next_run)
                if next_scheduled is None or next_run < datetime.fromisoformat(next_scheduled["next_run"]):
                    next_scheduled = {
                        "schedule_id": schedule.schedule_id,
                        "next_run": schedule.next_run,
                        "interval": schedule.interval.value
                    }

        return {
            "running": self._running,
            "total_schedules": len(self._schedules),
            "active_schedules": len(active_schedules),
            "next_scheduled_run": next_scheduled,
            "state_file": str(self._state_file)
        }


# Global singleton
scheduler_service = SchedulerService()
