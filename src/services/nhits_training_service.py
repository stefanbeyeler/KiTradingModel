"""
NHITS Auto-Training Service.

Provides scheduled and batch training for NHITS models.
Includes persistent data caching to reduce API calls during batch training.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from ..config.settings import settings
from ..models.trading_data import TimeSeriesData
from ..models.forecast_data import ForecastTrainingResult, TrainingMetrics
from .training_data_cache_service import training_data_cache

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
        # Detailed failure tracking for debugging
        self._failed_models: Dict[str, Dict[str, Any]] = {}
        self._successful_models: Dict[str, Dict[str, Any]] = {}

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
        days: int = 30,
        timeframe: str = "H1",
        use_cache: bool = True
    ) -> List[TimeSeriesData]:
        """Fetch training data for a symbol from EasyInsight API.

        Args:
            symbol: Trading symbol
            days: Number of days of data to fetch
            timeframe: Timeframe for OHLCV data (M15, H1, D1)
            use_cache: Whether to use cached data if available
        """
        try:
            import httpx
            from datetime import datetime
            from .forecast_service import forecast_service

            tf = timeframe.upper()

            # Get timeframe-specific requirements
            tf_config = forecast_service.get_timeframe_config(tf)
            min_required = tf_config["input_size"] + tf_config["horizon"]

            # Calculate limit based on timeframe
            # M15 = 96 candles/day, H1 = 24 candles/day, D1 = 1 candle/day
            # For D1: API returns rows per snapshot, need more rows to get unique days
            if tf == "M15":
                limit = max(days * 96, min_required * 2)  # Extra buffer for missing data
            elif tf == "D1":
                # D1 requires 60 input + 7 horizon = 67 unique days
                # Since API stores ~4 snapshots per hour, we need many more rows to get 67 unique days
                # Request enough rows to ensure we get at least min_required unique days
                limit = max(min_required * 96, 5000)  # ~52 days worth of snapshots at 96/day
            else:  # H1 default
                limit = max(days * 24, min_required * 2)

            # Try to get data from cache first
            rows = None
            from_cache = False
            if use_cache:
                cached_data = training_data_cache.get_cached_data(symbol, tf, "easyinsight")
                if cached_data:
                    rows = cached_data
                    from_cache = True
                    logger.info(f"Using cached data for {symbol}/{tf}: {len(rows)} rows")

            # Fetch from API if no cache
            if rows is None:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{settings.easyinsight_api_url}/symbol-data-full/{symbol}",
                        params={"limit": limit}
                    )
                    response.raise_for_status()

                    data = response.json()
                    rows = data.get('data', [])

                    # Cache the data for future use
                    if rows and use_cache:
                        training_data_cache.cache_data(symbol, tf, rows, "easyinsight")
                        logger.debug(f"Cached {len(rows)} rows for {symbol}/{tf}")

            # Define OHLCV field prefixes based on timeframe
            if tf == "M15":
                ohlc_prefix = "m15_"
            elif tf == "D1":
                ohlc_prefix = "d1_"
            else:  # H1 default
                ohlc_prefix = "h1_"

            time_series = []
            seen_timestamps = set()  # For D1, deduplicate by date

            for row in rows:
                try:
                    # Parse timestamp
                    timestamp_str = row.get('snapshot_time')
                    if not timestamp_str:
                        continue

                    # Parse ISO format timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                    # For D1, deduplicate by date (only keep one row per day)
                    if tf == "D1":
                        date_key = timestamp.date()
                        if date_key in seen_timestamps:
                            continue
                        seen_timestamps.add(date_key)

                    # Get OHLCV based on timeframe
                    open_val = row.get(f'{ohlc_prefix}open', 0)
                    high_val = row.get(f'{ohlc_prefix}high', 0)
                    low_val = row.get(f'{ohlc_prefix}low', 0)
                    close_val = row.get(f'{ohlc_prefix}close', 0)

                    # Skip rows with missing OHLCV data
                    if not all([open_val, high_val, low_val, close_val]):
                        continue

                    time_series.append(TimeSeriesData(
                        timestamp=timestamp,
                        symbol=symbol,
                        open=float(open_val),
                        high=float(high_val),
                        low=float(low_val),
                        close=float(close_val),
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
                            'timeframe': tf
                        }
                    ))
                except Exception as row_error:
                    logger.warning(f"Failed to parse row for {symbol}/{tf}: {row_error}")
                    continue

            logger.info(
                f"Fetched {len(time_series)} data points for {symbol}/{tf} "
                f"{'(from cache)' if from_cache else '(from API)'}"
            )
            return time_series

        except Exception as e:
            logger.error(f"Failed to get training data from EasyInsight API for {symbol}/{tf}: {e}")
            return []

    async def train_symbol(
        self,
        symbol: str,
        force: bool = False,
        timeframe: str = "H1"
    ) -> ForecastTrainingResult:
        """Train NHITS model for a single symbol and timeframe."""
        from .forecast_service import forecast_service

        tf = timeframe.upper()
        model_key = f"{symbol}_{tf}" if tf != "H1" else symbol

        try:
            # Get training data for specific timeframe
            time_series = await self.get_training_data(symbol, timeframe=tf)

            if not time_series:
                return ForecastTrainingResult(
                    symbol=model_key,
                    trained_at=datetime.utcnow(),
                    training_samples=0,
                    training_duration_seconds=0,
                    model_path="",
                    metrics=TrainingMetrics(),
                    success=False,
                    error_message=f"No training data available for {tf}"
                )

            # Get timeframe-specific requirements
            tf_config = forecast_service.get_timeframe_config(tf)
            min_required = tf_config["input_size"] + tf_config["horizon"]

            if len(time_series) < min_required:
                return ForecastTrainingResult(
                    symbol=model_key,
                    trained_at=datetime.utcnow(),
                    training_samples=len(time_series),
                    training_duration_seconds=0,
                    model_path="",
                    metrics=TrainingMetrics(),
                    success=False,
                    error_message=f"Insufficient data for {tf}: {len(time_series)} < {min_required}"
                )

            # Check if retraining is needed
            if not force and not forecast_service._should_retrain(symbol, timeframe=tf):
                logger.info(f"Model for {symbol}/{tf} is up to date, skipping")
                info = forecast_service.get_model_info(symbol, timeframe=tf)
                return ForecastTrainingResult(
                    symbol=model_key,
                    trained_at=info.last_trained or datetime.utcnow(),
                    training_samples=info.training_samples or 0,
                    training_duration_seconds=0,
                    model_path=str(forecast_service._get_model_path(symbol, tf)),
                    metrics=TrainingMetrics(),
                    success=True,
                    error_message="Model up to date (skipped)"
                )

            # Train model with timeframe
            result = await forecast_service.train(time_series, symbol, timeframe=tf)
            self._training_results[model_key] = result

            return result

        except Exception as e:
            logger.error(f"Failed to train {symbol}/{tf}: {e}")
            return ForecastTrainingResult(
                symbol=model_key,
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
        max_concurrent: int = 2,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train NHITS models for all (or specified) symbols across multiple timeframes.

        Args:
            symbols: List of symbols to train (None = all available)
            force: Force retraining even if model is up to date
            max_concurrent: Maximum concurrent training tasks
            timeframes: List of timeframes to train (default: ["M15", "H1", "D1"])

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

        # Default timeframes for multi-timeframe training
        if timeframes is None:
            timeframes = ["M15", "H1", "D1"]

        # Reset progress counters
        self._training_completed_symbols = 0
        self._training_successful = 0
        self._training_failed = 0
        self._training_skipped = 0
        self._training_current_symbol = None
        # Reset failure/success tracking for new training run
        self._failed_models = {}
        self._successful_models = {}

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

            # Total training tasks = symbols × timeframes
            total_tasks = len(symbols_to_train) * len(timeframes)
            self._training_total_symbols = total_tasks

            logger.info(
                f"Starting multi-timeframe batch training: "
                f"{len(symbols_to_train)} symbols × {len(timeframes)} timeframes = {total_tasks} models"
            )

            results = {}

            # Train sequentially: each symbol across all timeframes
            for symbol in symbols_to_train:
                # Check if cancelled
                if self._training_cancelled:
                    logger.info("Training cancelled by user")
                    break

                for tf in timeframes:
                    if self._training_cancelled:
                        break

                    model_key = f"{symbol}_{tf}"
                    self._training_current_symbol = model_key

                    try:
                        result = await self.train_symbol(symbol, force=force, timeframe=tf)

                        if isinstance(result, Exception):
                            error_info = {
                                "success": False,
                                "error": str(result),
                                "timeframe": tf,
                                "symbol": symbol,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            results[model_key] = error_info
                            self._failed_models[model_key] = error_info
                            self._training_failed += 1
                            logger.warning(f"Training failed for {model_key}: {result}")
                        elif result.success:
                            if "skipped" in (result.error_message or "").lower():
                                results[model_key] = {
                                    "success": True,
                                    "skipped": True,
                                    "samples": result.training_samples,
                                    "timeframe": tf
                                }
                                self._training_skipped += 1
                            else:
                                success_info = {
                                    "success": True,
                                    "samples": result.training_samples,
                                    "duration": result.training_duration_seconds,
                                    "loss": result.metrics.final_loss if result.metrics else None,
                                    "timeframe": tf,
                                    "symbol": symbol,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                results[model_key] = success_info
                                self._successful_models[model_key] = success_info
                                self._training_successful += 1
                        else:
                            error_info = {
                                "success": False,
                                "error": result.error_message,
                                "timeframe": tf,
                                "symbol": symbol,
                                "samples_found": result.training_samples,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            results[model_key] = error_info
                            self._failed_models[model_key] = error_info
                            self._training_failed += 1
                            logger.warning(f"Training failed for {model_key}: {result.error_message}")

                    except Exception as e:
                        error_info = {
                            "success": False,
                            "error": str(e),
                            "timeframe": tf,
                            "symbol": symbol,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        results[model_key] = error_info
                        self._failed_models[model_key] = error_info
                        self._training_failed += 1
                        logger.error(f"Exception during training {model_key}: {e}")

                    self._training_completed_symbols += 1

            duration = (datetime.utcnow() - start_time).total_seconds()
            self._last_training_run = datetime.utcnow()

            status = "cancelled" if self._training_cancelled else "completed"

            # Count by timeframe
            by_timeframe = {}
            for tf in timeframes:
                tf_results = {k: v for k, v in results.items() if v.get("timeframe") == tf}
                by_timeframe[tf] = {
                    "successful": sum(1 for v in tf_results.values() if v.get("success") and not v.get("skipped")),
                    "failed": sum(1 for v in tf_results.values() if not v.get("success")),
                    "skipped": sum(1 for v in tf_results.values() if v.get("skipped"))
                }

            summary = {
                "status": status,
                "started_at": start_time.isoformat(),
                "duration_seconds": duration,
                "total_symbols": len(symbols_to_train),
                "timeframes": timeframes,
                "total_models": total_tasks,
                "successful": self._training_successful,
                "failed": self._training_failed,
                "skipped": self._training_skipped,
                "by_timeframe": by_timeframe,
                "results": results
            }

            logger.info(
                f"Multi-timeframe batch training {status}: "
                f"{self._training_successful} successful, {self._training_failed} failed, {self._training_skipped} skipped "
                f"in {duration:.1f}s"
            )

            # Keep cache for future re-training - data expires based on TTL per timeframe
            # This reduces API calls especially for Twelve Data which has daily limits
            cache_stats = training_data_cache.get_stats()
            summary["cache_stats"] = {
                "entries_cached": cache_stats.get("total_entries", 0),
                "cache_size_mb": cache_stats.get("total_size_mb", 0),
                "hit_rate_during_training": cache_stats.get("hit_rate", 0),
                "bytes_saved": cache_stats.get("bytes_saved", 0)
            }
            logger.info(
                f"Training cache retained for future re-training: "
                f"{cache_stats.get('total_entries', 0)} entries, "
                f"{cache_stats.get('total_size_mb', 0):.2f} MB, "
                f"hit rate: {cache_stats.get('hit_rate', 0)}%"
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
            # Always cleanup expired cache entries
            training_data_cache.cleanup_expired()
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

    def get_failed_models(self) -> Dict[str, Any]:
        """Get details about failed model trainings from the current/last run."""
        return {
            "count": len(self._failed_models),
            "models": self._failed_models,
            "training_in_progress": self._training_in_progress,
        }

    def get_successful_models(self) -> Dict[str, Any]:
        """Get details about successful model trainings from the current/last run."""
        return {
            "count": len(self._successful_models),
            "models": self._successful_models,
            "training_in_progress": self._training_in_progress,
        }

    def get_training_results(self) -> Dict[str, Any]:
        """Get comprehensive results from the current/last training run."""
        return {
            "training_in_progress": self._training_in_progress,
            "summary": {
                "total": self._training_total_symbols,
                "completed": self._training_completed_symbols,
                "successful": self._training_successful,
                "failed": self._training_failed,
                "skipped": self._training_skipped,
            },
            "failed_models": self._failed_models,
            "successful_models": self._successful_models,
            "last_training_run": self._last_training_run.isoformat() if self._last_training_run else None,
        }


# Singleton instance
nhits_training_service = NHITSTrainingService()
