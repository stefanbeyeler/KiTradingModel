"""Automated training scheduler for TCN models."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .training_service import training_service, TrainingConfig, TrainingStatus


class ScheduleInterval(str, Enum):
    """Training schedule intervals."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MANUAL = "manual"


@dataclass
class AutoTrainingConfig:
    """Configuration for automated training."""
    enabled: bool = False
    interval: ScheduleInterval = ScheduleInterval.WEEKLY
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    lookback_days: int = 365
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    min_symbols: int = 5
    last_run: Optional[str] = None
    next_run: Optional[str] = None


class TrainingScheduler:
    """
    Scheduler for automated TCN model training.

    Features:
    - Trains models for all available symbols
    - Supports multiple timeframes
    - Configurable schedule (daily, weekly, monthly)
    - Persists configuration
    """

    CONFIG_FILE = "data/models/tcn/auto_training_config.json"

    def __init__(self):
        self.config = AutoTrainingConfig()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self._current_training: Optional[Dict] = None

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)

        # Load existing config
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.config = AutoTrainingConfig(
                        enabled=data.get("enabled", False),
                        interval=ScheduleInterval(data.get("interval", "weekly")),
                        timeframes=data.get("timeframes", ["1h", "4h", "1d"]),
                        lookback_days=data.get("lookback_days", 365),
                        epochs=data.get("epochs", 100),
                        batch_size=data.get("batch_size", 32),
                        learning_rate=data.get("learning_rate", 1e-4),
                        min_symbols=data.get("min_symbols", 5),
                        last_run=data.get("last_run"),
                        next_run=data.get("next_run")
                    )
                logger.info(f"Loaded auto-training config: enabled={self.config.enabled}")
        except Exception as e:
            logger.warning(f"Could not load auto-training config: {e}")

    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            data = {
                "enabled": self.config.enabled,
                "interval": self.config.interval.value,
                "timeframes": self.config.timeframes,
                "lookback_days": self.config.lookback_days,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "min_symbols": self.config.min_symbols,
                "last_run": self.config.last_run,
                "next_run": self.config.next_run
            }
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved auto-training config")
        except Exception as e:
            logger.error(f"Could not save auto-training config: {e}")

    def _calculate_next_run(self) -> Optional[datetime]:
        """Calculate next scheduled run time."""
        now = datetime.now()

        if self.config.interval == ScheduleInterval.DAILY:
            # Run at 2 AM next day
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif self.config.interval == ScheduleInterval.WEEKLY:
            # Run Sunday at 2 AM
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0 and now.hour >= 2:
                days_until_sunday = 7
            next_run = (now + timedelta(days=days_until_sunday)).replace(
                hour=2, minute=0, second=0, microsecond=0
            )
        elif self.config.interval == ScheduleInterval.MONTHLY:
            # Run 1st of month at 2 AM
            if now.day == 1 and now.hour < 2:
                next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
            else:
                # First day of next month
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=1,
                                          hour=2, minute=0, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=1,
                                          hour=2, minute=0, second=0, microsecond=0)
        else:
            # Manual - no automatic scheduling
            return None

        return next_run

    async def _get_available_symbols(self) -> List[str]:
        """Fetch available symbols from Data Service."""
        try:
            import httpx
            from src.config.microservices import microservices_config

            data_service_url = os.getenv("DATA_SERVICE_URL", microservices_config.data_service_url)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{data_service_url}/api/v1/managed-symbols")
                response.raise_for_status()

                symbols_data = response.json()

                # Filter active symbols with data
                symbols = [
                    s["symbol"] for s in symbols_data
                    if s.get("has_timescaledb_data") and s.get("status") == "active"
                ]

                logger.info(f"Found {len(symbols)} available symbols for training")
                return symbols

        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []

    async def run_training_for_all(self, timeframes: Optional[List[str]] = None) -> Dict:
        """Run training for all available symbols and timeframes."""
        if training_service.is_training():
            return {
                "status": "error",
                "message": "Training already in progress"
            }

        timeframes = timeframes or self.config.timeframes
        symbols = await self._get_available_symbols()

        if len(symbols) < self.config.min_symbols:
            return {
                "status": "error",
                "message": f"Not enough symbols ({len(symbols)} < {self.config.min_symbols})"
            }

        self._current_training = {
            "started_at": datetime.now().isoformat(),
            "symbols": symbols,
            "timeframes": timeframes,
            "completed_timeframes": [],
            "results": {}
        }

        config = TrainingConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate
        )

        results = {}

        for timeframe in timeframes:
            logger.info(f"Starting auto-training for timeframe {timeframe} with {len(symbols)} symbols")

            try:
                result = await training_service.train(
                    symbols=symbols,
                    timeframe=timeframe,
                    lookback_days=self.config.lookback_days,
                    config=config
                )

                results[timeframe] = {
                    "status": result.get("status", TrainingStatus.FAILED.value),
                    "job_id": result.get("job_id"),
                    "metrics": result.get("metrics")
                }

                self._current_training["completed_timeframes"].append(timeframe)

            except Exception as e:
                logger.error(f"Training failed for {timeframe}: {e}")
                results[timeframe] = {
                    "status": "failed",
                    "error": str(e)
                }

        # Update last run
        self.config.last_run = datetime.now().isoformat()
        next_run = self._calculate_next_run()
        self.config.next_run = next_run.isoformat() if next_run else None
        self._save_config()

        self._current_training["completed_at"] = datetime.now().isoformat()
        self._current_training["results"] = results

        return {
            "status": "completed",
            "symbols_count": len(symbols),
            "timeframes": timeframes,
            "results": results
        }

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Auto-training scheduler started")

        while self._running:
            try:
                if not self.config.enabled:
                    await asyncio.sleep(60)
                    continue

                next_run = self._calculate_next_run()
                if next_run is None:
                    await asyncio.sleep(60)
                    continue

                self.config.next_run = next_run.isoformat()
                self._save_config()

                now = datetime.now()
                if now >= next_run:
                    logger.info("Starting scheduled auto-training")
                    await self.run_training_for_all()

                # Check every minute
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

        logger.info("Auto-training scheduler stopped")

    def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Auto-training scheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
        logger.info("Auto-training scheduler stopped")

    def update_config(
        self,
        enabled: Optional[bool] = None,
        interval: Optional[str] = None,
        timeframes: Optional[List[str]] = None,
        lookback_days: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        min_symbols: Optional[int] = None
    ) -> AutoTrainingConfig:
        """Update scheduler configuration."""
        if enabled is not None:
            self.config.enabled = enabled
        if interval is not None:
            self.config.interval = ScheduleInterval(interval)
        if timeframes is not None:
            self.config.timeframes = timeframes
        if lookback_days is not None:
            self.config.lookback_days = lookback_days
        if epochs is not None:
            self.config.epochs = epochs
        if batch_size is not None:
            self.config.batch_size = batch_size
        if learning_rate is not None:
            self.config.learning_rate = learning_rate
        if min_symbols is not None:
            self.config.min_symbols = min_symbols

        # Recalculate next run
        if self.config.enabled:
            next_run = self._calculate_next_run()
            self.config.next_run = next_run.isoformat() if next_run else None

        self._save_config()
        return self.config

    def get_status(self) -> Dict:
        """Get scheduler status."""
        return {
            "enabled": self.config.enabled,
            "interval": self.config.interval.value,
            "timeframes": self.config.timeframes,
            "lookback_days": self.config.lookback_days,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "min_symbols": self.config.min_symbols,
            "last_run": self.config.last_run,
            "next_run": self.config.next_run,
            "is_running": self._running,
            "current_training": self._current_training
        }


# Singleton instance
training_scheduler = TrainingScheduler()
