"""
NHITS Auto-Training Service.

Provides scheduled and batch training for NHITS models.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from ..config.settings import settings
from ..models.trading_data import TimeSeriesData
from ..models.forecast_data import ForecastTrainingResult

logger = logging.getLogger(__name__)


class NHITSTrainingService:
    """Service for automated NHITS model training."""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._db_pool = None
        self._training_in_progress = False
        self._last_training_run: Optional[datetime] = None
        self._training_results: Dict[str, ForecastTrainingResult] = {}

    async def connect(self, db_pool):
        """Set the database connection pool."""
        self._db_pool = db_pool

    async def start(self):
        """Start the scheduled training service."""
        if self._running:
            logger.warning("NHITS Training Service already running")
            return

        if not settings.nhits_scheduled_training_enabled:
            logger.info("NHITS scheduled training is disabled")
            return

        self._running = True
        self._task = asyncio.create_task(self._training_loop())
        logger.info(
            f"NHITS Training Service started - "
            f"interval: {settings.nhits_scheduled_training_interval_hours}h"
        )

    async def stop(self):
        """Stop the scheduled training service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("NHITS Training Service stopped")

    async def _training_loop(self):
        """Background loop for scheduled training."""
        interval_seconds = settings.nhits_scheduled_training_interval_hours * 3600

        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(interval_seconds)

                if not self._running:
                    break

                # Run training
                logger.info("Starting scheduled NHITS training run")
                await self.train_all_symbols()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in NHITS training loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def get_available_symbols(self) -> List[str]:
        """Get list of symbols available for training from TimescaleDB."""
        if not self._db_pool:
            logger.warning("No database connection available")
            return []

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT symbol
                    FROM market_data
                    WHERE timestamp > NOW() - INTERVAL '30 days'
                    ORDER BY symbol
                """)
                symbols = [row['symbol'] for row in rows]
                logger.info(f"Found {len(symbols)} symbols for NHITS training")
                return symbols
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    async def get_training_data(
        self,
        symbol: str,
        days: int = 30
    ) -> List[TimeSeriesData]:
        """Fetch training data for a symbol from TimescaleDB."""
        if not self._db_pool:
            logger.warning("No database connection available")
            return []

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        COALESCE(volume, 0) as volume
                    FROM market_data
                    WHERE symbol = $1
                      AND timestamp > NOW() - INTERVAL '%s days'
                    ORDER BY timestamp ASC
                """ % days, symbol)

                time_series = [
                    TimeSeriesData(
                        timestamp=row['timestamp'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume'])
                    )
                    for row in rows
                ]

                logger.debug(
                    f"Fetched {len(time_series)} data points for {symbol}"
                )
                return time_series

        except Exception as e:
            logger.error(f"Failed to get training data for {symbol}: {e}")
            return []

    async def train_symbol(
        self,
        symbol: str,
        force: bool = False
    ) -> ForecastTrainingResult:
        """Train NHITS model for a single symbol."""
        from .forecast_service import forecast_service

        try:
            # Get training data
            time_series = await self.get_training_data(symbol)

            if not time_series:
                return ForecastTrainingResult(
                    symbol=symbol,
                    trained_at=datetime.utcnow(),
                    training_samples=0,
                    training_duration_seconds=0,
                    model_path="",
                    metrics={},
                    success=False,
                    error_message="No training data available"
                )

            min_required = settings.nhits_input_size + settings.nhits_horizon
            if len(time_series) < min_required:
                return ForecastTrainingResult(
                    symbol=symbol,
                    trained_at=datetime.utcnow(),
                    training_samples=len(time_series),
                    training_duration_seconds=0,
                    model_path="",
                    metrics={},
                    success=False,
                    error_message=f"Insufficient data: {len(time_series)} < {min_required}"
                )

            # Check if retraining is needed
            if not force and not forecast_service._should_retrain(symbol):
                logger.info(f"Model for {symbol} is up to date, skipping")
                info = forecast_service.get_model_info(symbol)
                return ForecastTrainingResult(
                    symbol=symbol,
                    trained_at=info.last_trained or datetime.utcnow(),
                    training_samples=info.training_samples or 0,
                    training_duration_seconds=0,
                    model_path=str(forecast_service._get_model_path(symbol)),
                    metrics={},
                    success=True,
                    error_message="Model up to date (skipped)"
                )

            # Train model
            result = await forecast_service.train(time_series, symbol)
            self._training_results[symbol] = result

            return result

        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
            return ForecastTrainingResult(
                symbol=symbol,
                trained_at=datetime.utcnow(),
                training_samples=0,
                training_duration_seconds=0,
                model_path="",
                metrics={},
                success=False,
                error_message=str(e)
            )

    async def train_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        force: bool = False,
        max_concurrent: int = 2
    ) -> Dict[str, Any]:
        """
        Train NHITS models for all (or specified) symbols.

        Args:
            symbols: List of symbols to train (None = all available)
            force: Force retraining even if model is up to date
            max_concurrent: Maximum concurrent training tasks

        Returns:
            Summary of training results
        """
        if self._training_in_progress:
            return {
                "status": "error",
                "message": "Training already in progress",
                "results": {}
            }

        self._training_in_progress = True
        start_time = datetime.utcnow()

        try:
            # Get symbols to train
            if symbols:
                symbols_to_train = symbols
            elif settings.nhits_training_symbols:
                symbols_to_train = settings.nhits_training_symbols
            else:
                symbols_to_train = await self.get_available_symbols()

            if not symbols_to_train:
                return {
                    "status": "error",
                    "message": "No symbols to train",
                    "results": {}
                }

            logger.info(
                f"Starting batch training for {len(symbols_to_train)} symbols"
            )

            results = {}
            successful = 0
            failed = 0
            skipped = 0

            # Train in batches to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def train_with_semaphore(sym: str):
                async with semaphore:
                    return await self.train_symbol(sym, force=force)

            # Create tasks
            tasks = [
                train_with_semaphore(symbol)
                for symbol in symbols_to_train
            ]

            # Execute all tasks
            training_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for symbol, result in zip(symbols_to_train, training_results):
                if isinstance(result, Exception):
                    results[symbol] = {
                        "success": False,
                        "error": str(result)
                    }
                    failed += 1
                elif result.success:
                    if "skipped" in (result.error_message or "").lower():
                        results[symbol] = {
                            "success": True,
                            "skipped": True,
                            "samples": result.training_samples
                        }
                        skipped += 1
                    else:
                        results[symbol] = {
                            "success": True,
                            "samples": result.training_samples,
                            "duration": result.training_duration_seconds,
                            "loss": result.metrics.get("final_loss")
                        }
                        successful += 1
                else:
                    results[symbol] = {
                        "success": False,
                        "error": result.error_message
                    }
                    failed += 1

            duration = (datetime.utcnow() - start_time).total_seconds()
            self._last_training_run = datetime.utcnow()

            summary = {
                "status": "completed",
                "started_at": start_time.isoformat(),
                "duration_seconds": duration,
                "total_symbols": len(symbols_to_train),
                "successful": successful,
                "failed": failed,
                "skipped": skipped,
                "results": results
            }

            logger.info(
                f"Batch training completed: "
                f"{successful} successful, {failed} failed, {skipped} skipped "
                f"in {duration:.1f}s"
            )

            return summary

        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "results": {}
            }

        finally:
            self._training_in_progress = False

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the training service."""
        from .forecast_service import forecast_service

        models = forecast_service.list_models()

        return {
            "enabled": settings.nhits_scheduled_training_enabled,
            "running": self._running,
            "training_in_progress": self._training_in_progress,
            "last_training_run": self._last_training_run.isoformat() if self._last_training_run else None,
            "interval_hours": settings.nhits_scheduled_training_interval_hours,
            "trained_models": len([m for m in models if m.model_exists]),
            "configured_symbols": settings.nhits_training_symbols or "all",
        }


# Singleton instance
nhits_training_service = NHITSTrainingService()
