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
from ..models.forecast_data import ForecastTrainingResult, TrainingMetrics

logger = logging.getLogger(__name__)


class NHITSTrainingService:
    """Service for automated NHITS model training."""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._training_in_progress = False
        self._training_cancelled = False
        self._last_training_run: Optional[datetime] = None
        self._training_results: Dict[str, ForecastTrainingResult] = {}
        # Progress tracking
        self._current_training_start: Optional[datetime] = None
        self._training_total_symbols: int = 0
        self._training_completed_symbols: int = 0
        self._training_current_symbol: Optional[str] = None
        self._training_successful: int = 0
        self._training_failed: int = 0
        self._training_skipped: int = 0

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
        """Get list of symbols available for training from EasyInsight API."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{settings.easyinsight_api_url}/symbols")
                response.raise_for_status()

                data = response.json()
                # API returns list of dicts with 'symbol', 'category', etc.
                symbols = [item.get('symbol') for item in data if item.get('symbol')]

                logger.info(f"Found {len(symbols)} symbols for NHITS training from EasyInsight API")
                return symbols

        except Exception as e:
            logger.error(f"Failed to get symbols from EasyInsight API: {e}")
            return []

    async def get_training_data(
        self,
        symbol: str,
        days: int = 30
    ) -> List[TimeSeriesData]:
        """Fetch training data for a symbol from EasyInsight API."""
        try:
            import httpx
            from datetime import datetime

            # Request enough data points. The API returns snapshots which may be
            # more frequent than daily. Request days * 24 to ensure we get enough hourly data.
            # For daily training we'll filter to unique days later.
            limit = days * 24

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{settings.easyinsight_api_url}/symbol-data-full/{symbol}",
                    params={"limit": limit}
                )
                response.raise_for_status()

                data = response.json()
                rows = data.get('data', [])

                time_series = []
                for row in rows:
                    try:
                        # Parse timestamp
                        timestamp_str = row.get('snapshot_time')
                        if not timestamp_str:
                            continue

                        # Parse ISO format timestamp
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                        # Use h1_ (hourly) data for training with ALL available indicators
                        time_series.append(TimeSeriesData(
                            timestamp=timestamp,
                            symbol=symbol,
                            open=float(row.get('h1_open', 0)),
                            high=float(row.get('h1_high', 0)),
                            low=float(row.get('h1_low', 0)),
                            close=float(row.get('h1_close', 0)),
                            volume=0.0,  # Volume not available in API response
                            # Momentum Indicators
                            rsi=row.get('rsi'),
                            macd_main=row.get('macd_main'),
                            macd_signal=row.get('macd_signal'),
                            cci=row.get('cci'),
                            stoch_k=row.get('sto_main'),
                            stoch_d=row.get('sto_signal'),
                            # Trend Indicators
                            adx=row.get('adx_main'),
                            adx_plus_di=row.get('adx_plusdi'),
                            adx_minus_di=row.get('adx_minusdi'),
                            ma100=row.get('ma_10'),
                            # Volatility Indicators
                            atr=row.get('atr_d1'),
                            atr_pct=row.get('atr_pct_d1'),
                            bb_upper=row.get('bb_upper'),
                            bb_middle=row.get('bb_base'),
                            bb_lower=row.get('bb_lower'),
                            range_d1=row.get('range_d1'),
                            # Ichimoku Cloud (complete)
                            ichimoku_tenkan=row.get('ichimoku_tenkan'),
                            ichimoku_kijun=row.get('ichimoku_kijun'),
                            ichimoku_senkou_a=row.get('ichimoku_senkoua'),
                            ichimoku_senkou_b=row.get('ichimoku_senkoub'),
                            ichimoku_chikou=row.get('ichimoku_chikou'),
                            # Strength Indicators
                            strength_4h=row.get('strength_4h'),
                            strength_1d=row.get('strength_1d'),
                            strength_1w=row.get('strength_1w'),
                            # Support/Resistance Pivot Points
                            s1_level=row.get('s1_level_m5'),
                            r1_level=row.get('r1_level_m5'),
                            # Store additional data for LLM context
                            additional_data={
                                'bid': row.get('bid'),
                                'ask': row.get('ask'),
                                'spread': row.get('spread'),
                                'spread_pct': row.get('spread_pct'),
                                'category': row.get('category'),
                                'd1_open': row.get('d1_open'),
                                'd1_high': row.get('d1_high'),
                                'd1_low': row.get('d1_low'),
                                'd1_close': row.get('d1_close'),
                                'm15_open': row.get('m15_open'),
                                'm15_high': row.get('m15_high'),
                                'm15_low': row.get('m15_low'),
                                'm15_close': row.get('m15_close')
                            }
                        ))
                    except Exception as row_error:
                        logger.warning(f"Failed to parse row for {symbol}: {row_error}")
                        continue

                logger.info(
                    f"Fetched {len(time_series)} data points for {symbol} from EasyInsight API"
                )
                return time_series

        except Exception as e:
            logger.error(f"Failed to get training data from EasyInsight API for {symbol}: {e}")
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
                    metrics=TrainingMetrics(),
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
                    metrics=TrainingMetrics(),
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
                    metrics=TrainingMetrics(),
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
        self._training_cancelled = False
        start_time = datetime.utcnow()
        self._current_training_start = start_time

        # Reset progress counters
        self._training_completed_symbols = 0
        self._training_successful = 0
        self._training_failed = 0
        self._training_skipped = 0
        self._training_current_symbol = None

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

            self._training_total_symbols = len(symbols_to_train)

            logger.info(
                f"Starting batch training for {len(symbols_to_train)} symbols"
            )

            results = {}

            # Train sequentially to track progress properly
            for symbol in symbols_to_train:
                # Check if cancelled
                if self._training_cancelled:
                    logger.info("Training cancelled by user")
                    break

                self._training_current_symbol = symbol

                try:
                    result = await self.train_symbol(symbol, force=force)

                    if isinstance(result, Exception):
                        results[symbol] = {
                            "success": False,
                            "error": str(result)
                        }
                        self._training_failed += 1
                    elif result.success:
                        if "skipped" in (result.error_message or "").lower():
                            results[symbol] = {
                                "success": True,
                                "skipped": True,
                                "samples": result.training_samples
                            }
                            self._training_skipped += 1
                        else:
                            results[symbol] = {
                                "success": True,
                                "samples": result.training_samples,
                                "duration": result.training_duration_seconds,
                                "loss": result.metrics.final_loss if result.metrics else None
                            }
                            self._training_successful += 1
                    else:
                        results[symbol] = {
                            "success": False,
                            "error": result.error_message
                        }
                        self._training_failed += 1

                except Exception as e:
                    results[symbol] = {
                        "success": False,
                        "error": str(e)
                    }
                    self._training_failed += 1

                self._training_completed_symbols += 1

            duration = (datetime.utcnow() - start_time).total_seconds()
            self._last_training_run = datetime.utcnow()

            status = "cancelled" if self._training_cancelled else "completed"

            summary = {
                "status": status,
                "started_at": start_time.isoformat(),
                "duration_seconds": duration,
                "total_symbols": len(symbols_to_train),
                "successful": self._training_successful,
                "failed": self._training_failed,
                "skipped": self._training_skipped,
                "results": results
            }

            logger.info(
                f"Batch training {status}: "
                f"{self._training_successful} successful, {self._training_failed} failed, {self._training_skipped} skipped "
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
            self._training_cancelled = False
            self._training_current_symbol = None
            self._current_training_start = None

    def cancel_training(self) -> bool:
        """Cancel the current training run."""
        if not self._training_in_progress:
            return False
        self._training_cancelled = True
        logger.info("Training cancellation requested")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the training service."""
        from .forecast_service import forecast_service

        models = forecast_service.list_models()

        status = {
            "enabled": settings.nhits_scheduled_training_enabled,
            "running": self._running,
            "training_in_progress": self._training_in_progress,
            "last_training_run": self._last_training_run.isoformat() if self._last_training_run else None,
            "interval_hours": settings.nhits_scheduled_training_interval_hours,
            "trained_models": len([m for m in models if m.model_exists]),
            "configured_symbols": settings.nhits_training_symbols or "all",
        }

        # Add detailed progress if training is in progress
        if self._training_in_progress:
            elapsed = 0
            if self._current_training_start:
                elapsed = (datetime.utcnow() - self._current_training_start).total_seconds()

            progress_pct = 0
            if self._training_total_symbols > 0:
                progress_pct = int((self._training_completed_symbols / self._training_total_symbols) * 100)

            status["progress"] = {
                "total_symbols": self._training_total_symbols,
                "completed_symbols": self._training_completed_symbols,
                "current_symbol": self._training_current_symbol,
                "successful": self._training_successful,
                "failed": self._training_failed,
                "skipped": self._training_skipped,
                "progress_percent": progress_pct,
                "elapsed_seconds": int(elapsed),
                "cancelling": self._training_cancelled,
                "started_at": self._current_training_start.isoformat() if self._current_training_start else None,
            }

        return status


# Singleton instance
nhits_training_service = NHITSTrainingService()
