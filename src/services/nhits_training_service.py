"""
NHITS Auto-Training Service.

Provides scheduled and batch training for NHITS models.
Fetches training data from Data-Service which handles caching.

WICHTIG: Dieser Service verwendet den DataGatewayService für alle externen
Datenzugriffe. Direkte API-Aufrufe zu EasyInsight sind NICHT erlaubt.

Siehe: DEVELOPMENT_GUIDELINES.md - Datenzugriff-Architektur
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from ..config.settings import settings
from ..models.trading_data import TimeSeriesData
from ..models.forecast_data import ForecastTrainingResult, TrainingMetrics
from .data_gateway_service import data_gateway

logger = logging.getLogger(__name__)

# Data Service URL for fetching training data (with caching)
# Uses central configuration from settings - can be overridden via DATA_SERVICE_URL env var
DATA_SERVICE_URL = settings.data_service_url


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
        """
        Get list of symbols available for training via Data Gateway.

        Verwendet: DataGatewayService (siehe DEVELOPMENT_GUIDELINES.md)
        """
        try:
            symbols = await data_gateway.get_symbol_names()
            logger.info(f"Found {len(symbols)} symbols for NHITS training via Data Gateway")
            return symbols
        except Exception as e:
            logger.error(f"Failed to get symbols via Data Gateway: {e}")
            return []

    async def get_training_data(
        self,
        symbol: str,
        days: int = 30,
        timeframe: str = "H1",
        use_cache: bool = True
    ) -> List[TimeSeriesData]:
        """Fetch training data for a symbol via Data-Service API.

        The Data-Service handles caching and fallback to Twelve Data.

        Args:
            symbol: Trading symbol
            days: Number of days of data to fetch
            timeframe: Timeframe for OHLCV data (M15, H1, D1)
            use_cache: Whether to use cached data if available
        """
        try:
            import httpx
            from datetime import datetime

            tf = timeframe.upper()

            # Fetch from Data-Service (which handles caching)
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    f"{DATA_SERVICE_URL}/api/v1/training-data/{symbol}",
                    params={
                        "timeframe": tf,
                        "days": days,
                        "use_cache": use_cache
                    }
                )
                response.raise_for_status()

                result = response.json()
                rows = result.get('data', [])
                source = result.get('source', 'unknown')
                from_cache = result.get('from_cache', False)

            # Define OHLCV field prefixes based on timeframe (for EasyInsight data)
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
                    # Handle different data sources
                    if source == "twelvedata":
                        # Twelve Data format
                        timestamp_str = row.get('datetime')
                        if not timestamp_str:
                            continue
                        timestamp = datetime.fromisoformat(timestamp_str)

                        open_val = float(row.get('open', 0))
                        high_val = float(row.get('high', 0))
                        low_val = float(row.get('low', 0))
                        close_val = float(row.get('close', 0))
                        volume_val = float(row.get('volume', 0))

                        if not all([open_val, high_val, low_val, close_val]):
                            continue

                        time_series.append(TimeSeriesData(
                            timestamp=timestamp,
                            symbol=symbol,
                            open=open_val,
                            high=high_val,
                            low=low_val,
                            close=close_val,
                            volume=volume_val,
                            additional_data={'source': 'twelvedata', 'timeframe': tf}
                        ))
                    else:
                        # EasyInsight format (Data Service already extracts OHLC to standard fields)
                        timestamp_str = row.get('snapshot_time') or row.get('timestamp')
                        if not timestamp_str:
                            continue

                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                        # Deduplicate based on timeframe interval
                        # EasyInsight returns rolling snapshots, we need unique candles
                        if tf == "D1":
                            # Deduplicate by date
                            dedup_key = timestamp.date()
                        elif tf in ("H1", "H4"):
                            # Deduplicate by hour
                            dedup_key = (timestamp.date(), timestamp.hour)
                        elif tf in ("M15", "M30"):
                            # Deduplicate by 15/30 min interval
                            interval = 15 if tf == "M15" else 30
                            minute_bucket = (timestamp.minute // interval) * interval
                            dedup_key = (timestamp.date(), timestamp.hour, minute_bucket)
                        elif tf == "M5":
                            # Deduplicate by 5 min interval
                            minute_bucket = (timestamp.minute // 5) * 5
                            dedup_key = (timestamp.date(), timestamp.hour, minute_bucket)
                        else:
                            dedup_key = timestamp  # No dedup for other timeframes

                        if dedup_key in seen_timestamps:
                            continue
                        seen_timestamps.add(dedup_key)

                        # Data Service already extracts OHLC to standard field names
                        # Try standard names first, then prefixed names for backward compatibility
                        open_val = row.get('open') or row.get(f'{ohlc_prefix}open', 0)
                        high_val = row.get('high') or row.get(f'{ohlc_prefix}high', 0)
                        low_val = row.get('low') or row.get(f'{ohlc_prefix}low', 0)
                        close_val = row.get('close') or row.get(f'{ohlc_prefix}close', 0)

                        if not all([open_val, high_val, low_val, close_val]):
                            continue

                        time_series.append(TimeSeriesData(
                            timestamp=timestamp,
                            symbol=symbol,
                            open=float(open_val),
                            high=float(high_val),
                            low=float(low_val),
                            close=float(close_val),
                            volume=0.0,
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
                            # Ichimoku Cloud
                            ichimoku_tenkan=row.get('ichimoku_tenkan'),
                            ichimoku_kijun=row.get('ichimoku_kijun'),
                            ichimoku_senkou_a=row.get('ichimoku_senkoua'),
                            ichimoku_senkou_b=row.get('ichimoku_senkoub'),
                            ichimoku_chikou=row.get('ichimoku_chikou'),
                            # Strength Indicators
                            strength_4h=row.get('strength_4h'),
                            strength_1d=row.get('strength_1d'),
                            strength_1w=row.get('strength_1w'),
                            # Support/Resistance
                            s1_level=row.get('s1_level_m5'),
                            r1_level=row.get('r1_level_m5'),
                            additional_data={
                                'bid': row.get('bid'),
                                'ask': row.get('ask'),
                                'spread': row.get('spread'),
                                'spread_pct': row.get('spread_pct'),
                                'category': row.get('category'),
                                'timeframe': tf,
                                'source': source
                            }
                        ))
                except Exception as row_error:
                    logger.warning(f"Failed to parse row for {symbol}/{tf}: {row_error}")
                    continue

            cache_info = "(from cache)" if from_cache else f"(from {source})"
            logger.info(f"Fetched {len(time_series)} data points for {symbol}/{tf} {cache_info}")
            return time_series

        except Exception as e:
            logger.error(f"Failed to get training data from Data-Service for {symbol}/{timeframe}: {e}")
            return []

    async def get_training_data_twelvedata(
        self,
        symbol: str,
        timeframe: str = "H1",
        use_cache: bool = True
    ) -> List[TimeSeriesData]:
        """Fetch training data for a symbol via Data Service (TwelveData endpoint).

        ARCHITEKTUR-KONFORM: Alle Datenzugriffe erfolgen über den Data Service.
        Der Data Service (Port 3001) ist das einzige Gateway für TwelveData.

        This is used as fallback when EasyInsight doesn't have enough data.
        The Data Service handles symbol format conversion automatically.

        Args:
            symbol: Trading symbol (will be converted to Twelve Data format by Data Service)
            timeframe: Timeframe for OHLCV data (M15, H1, D1)
            use_cache: Whether to use cached data if available
        """
        import httpx
        from .forecast_service import forecast_service
        from .training_data_cache_service import training_data_cache

        try:
            tf = timeframe.upper()

            # Get timeframe-specific requirements
            tf_config = forecast_service.get_timeframe_config(tf)
            min_required = tf_config["input_size"] + tf_config["horizon"]

            # Map timeframe to Twelve Data interval
            interval_map = {
                "M15": "15min",
                "H1": "1h",
                "D1": "1day"
            }
            interval = interval_map.get(tf, "1h")

            # Calculate outputsize based on timeframe
            if tf == "M15":
                outputsize = min(5000, min_required * 3)  # Extra buffer
            elif tf == "D1":
                outputsize = min(5000, min_required * 2)
            else:  # H1
                outputsize = min(5000, min_required * 2)

            # Try to get data from cache first
            if use_cache:
                cached_data = training_data_cache.get_cached_data(symbol, tf, "twelvedata")
                if cached_data:
                    logger.info(f"Using cached Twelve Data for {symbol}/{tf}: {len(cached_data)} rows")
                    return self._parse_twelvedata_response(cached_data, symbol, tf)

            # Fetch via Data Service HTTP API (architekturkonform)
            logger.info(f"Fetching {symbol}/{tf} via Data Service TwelveData endpoint (interval: {interval})")

            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{DATA_SERVICE_URL}/api/v1/twelvedata/time_series/{symbol}"
                params = {
                    "interval": interval,
                    "outputsize": outputsize
                }

                response = await client.get(url, params=params)

                if response.status_code != 200:
                    logger.warning(f"Data Service returned {response.status_code} for {symbol}/{tf}")
                    return []

                data = response.json()

                # Check for errors or unsupported symbols
                if data.get("unsupported") or data.get("error"):
                    logger.warning(f"Data Service: {symbol} not supported by TwelveData")
                    return []

                values = data.get("values", [])
                if values:
                    logger.info(f"Data Service returned {len(values)} rows for {symbol}/{tf}")
                    # Cache the data
                    if use_cache:
                        training_data_cache.cache_data(symbol, tf, values, "twelvedata")
                        logger.debug(f"Cached {len(values)} Twelve Data rows for {symbol}/{tf}")
                    return self._parse_twelvedata_response(values, symbol, tf)

            logger.warning(f"No data from Data Service for {symbol}/{tf}")
            return []

        except Exception as e:
            logger.error(f"Failed to get training data via Data Service for {symbol}/{tf}: {e}")
            return []

    def _parse_twelvedata_response(
        self,
        values: List[dict],
        symbol: str,
        timeframe: str
    ) -> List[TimeSeriesData]:
        """Parse Twelve Data time series response into TimeSeriesData objects."""
        from datetime import datetime

        time_series = []
        for row in values:
            try:
                # Parse timestamp
                timestamp_str = row.get("datetime")
                if not timestamp_str:
                    continue

                # Twelve Data returns format: "2024-01-15 14:30:00" or "2024-01-15"
                try:
                    if " " in timestamp_str:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    else:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
                except ValueError:
                    timestamp = datetime.fromisoformat(timestamp_str)

                # Get OHLCV
                open_val = row.get("open")
                high_val = row.get("high")
                low_val = row.get("low")
                close_val = row.get("close")
                volume = row.get("volume", 0)

                if not all([open_val, high_val, low_val, close_val]):
                    continue

                time_series.append(TimeSeriesData(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=float(open_val),
                    high=float(high_val),
                    low=float(low_val),
                    close=float(close_val),
                    volume=float(volume) if volume else 0.0,
                    # Twelve Data doesn't provide indicators in time series
                    # These will be None but NHITS can still train on OHLCV
                    additional_data={
                        'source': 'twelvedata',
                        'timeframe': timeframe
                    }
                ))
            except Exception as e:
                logger.warning(f"Failed to parse Twelve Data row: {e}")
                continue

        # Sort by timestamp (oldest first for training)
        time_series.sort(key=lambda x: x.timestamp)

        logger.info(f"Parsed {len(time_series)} data points from Twelve Data for {symbol}/{timeframe}")
        return time_series

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
            # Get timeframe-specific requirements first
            tf_config = forecast_service.get_timeframe_config(tf)
            min_required = tf_config["input_size"] + tf_config["horizon"]

            # Calculate required days based on timeframe
            # For D1: min_required is in days, add buffer
            # For H1: min_required is in hours, convert to days
            # For M15: min_required is in 15-min intervals, convert to days
            step_minutes = tf_config["step_minutes"]
            if step_minutes == 10080:  # W1 (weekly)
                required_days = (min_required * 7) + 30  # Weeks to days + buffer
            elif step_minutes == 1440:  # D1
                required_days = min_required + 14  # Add 2 week buffer for D1
            elif step_minutes == 240:  # H4
                required_days = (min_required // 6) + 7  # 6 H4 candles per day + buffer
            elif step_minutes == 60:  # H1
                required_days = (min_required // 24) + 7  # Convert hours to days + buffer
            elif step_minutes == 15:  # M15
                required_days = (min_required // 96) + 3  # 96 = 24*4 (15min intervals per day)
            else:  # M5
                required_days = (min_required // 288) + 2  # 288 = 24*12 (5min intervals per day)

            logger.info(
                f"Training {symbol}/{tf}: min_required={min_required} samples, "
                f"requesting {required_days} days of data"
            )

            # Track data from both sources for detailed error messages
            td_count = 0
            ei_count = 0

            # Get training data from TwelveData first (primary source)
            time_series = await self.get_training_data_twelvedata(symbol, timeframe=tf)
            td_count = len(time_series)
            data_source = "TwelveData"

            # If TwelveData doesn't have enough data, try EasyInsight as fallback
            if td_count < min_required:
                logger.info(
                    f"TwelveData has insufficient data for {symbol}/{tf} "
                    f"({td_count} < {min_required}), trying EasyInsight..."
                )
                ei_data = await self.get_training_data(symbol, days=required_days, timeframe=tf)
                ei_count = len(ei_data)

                if ei_count >= min_required:
                    time_series = ei_data
                    data_source = "EasyInsight"
                    logger.info(f"Using EasyInsight for {symbol}/{tf}: {ei_count} samples")
                elif ei_count > td_count:
                    # Use EasyInsight if it has more data, even if still insufficient
                    time_series = ei_data
                    data_source = "EasyInsight"

            if not time_series:
                return ForecastTrainingResult(
                    symbol=model_key,
                    trained_at=datetime.utcnow(),
                    training_samples=0,
                    training_duration_seconds=0,
                    model_path="",
                    metrics=TrainingMetrics(),
                    success=False,
                    error_message=f"Keine Daten für {tf}: TwelveData={td_count}, EasyInsight={ei_count} (benötigt: {min_required})"
                )

            if len(time_series) < min_required:
                return ForecastTrainingResult(
                    symbol=model_key,
                    trained_at=datetime.utcnow(),
                    training_samples=len(time_series),
                    training_duration_seconds=0,
                    model_path="",
                    metrics=TrainingMetrics(),
                    success=False,
                    error_message=f"Zu wenig Daten für {tf}: TwelveData={td_count}, EasyInsight={ei_count}, verwendet={len(time_series)} (benötigt: {min_required})"
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
            timeframes: List of timeframes to train (default: all supported)

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
            timeframes = ["M5", "M15", "H1", "H4", "D1", "W1"]

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

            # Parallel training configuration
            # Limit concurrent training to avoid GPU memory issues
            max_concurrent = 4  # Train up to 4 models in parallel

            async def train_single_model(symbol: str, tf: str) -> tuple:
                """Train a single model and return result tuple."""
                model_key = f"{symbol}_{tf}"
                self._training_current_symbol = model_key

                try:
                    result = await self.train_symbol(symbol, force=force, timeframe=tf)

                    if isinstance(result, Exception):
                        result_info = {
                            "success": False,
                            "error": str(result),
                            "timeframe": tf,
                            "symbol": symbol,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        # Update counters immediately for live progress
                        self._training_completed_symbols += 1
                        self._training_failed += 1
                        self._failed_models[model_key] = result_info
                        return (model_key, result_info, "failed")
                    elif result.success:
                        if "skipped" in (result.error_message or "").lower():
                            result_info = {
                                "success": True,
                                "skipped": True,
                                "samples": result.training_samples,
                                "timeframe": tf
                            }
                            # Update counters immediately for live progress
                            self._training_completed_symbols += 1
                            self._training_skipped += 1
                            return (model_key, result_info, "skipped")
                        else:
                            result_info = {
                                "success": True,
                                "samples": result.training_samples,
                                "duration": result.training_duration_seconds,
                                "loss": result.metrics.final_loss if result.metrics else None,
                                "timeframe": tf,
                                "symbol": symbol,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            # Update counters immediately for live progress
                            self._training_completed_symbols += 1
                            self._training_successful += 1
                            self._successful_models[model_key] = result_info
                            return (model_key, result_info, "success")
                    else:
                        result_info = {
                            "success": False,
                            "error": result.error_message,
                            "timeframe": tf,
                            "symbol": symbol,
                            "samples_found": result.training_samples,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        # Update counters immediately for live progress
                        self._training_completed_symbols += 1
                        self._training_failed += 1
                        self._failed_models[model_key] = result_info
                        return (model_key, result_info, "failed")

                except Exception as e:
                    logger.error(f"Exception during training {model_key}: {e}")
                    result_info = {
                        "success": False,
                        "error": str(e),
                        "timeframe": tf,
                        "symbol": symbol,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    # Update counters immediately for live progress
                    self._training_completed_symbols += 1
                    self._training_failed += 1
                    self._failed_models[model_key] = result_info
                    return (model_key, result_info, "failed")

            # Create all training tasks
            all_tasks = [
                (symbol, tf)
                for symbol in symbols_to_train
                for tf in timeframes
            ]

            logger.info(
                f"Starting PARALLEL training with max_concurrent={max_concurrent} "
                f"for {len(all_tasks)} models"
            )

            # Process in batches to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def train_with_semaphore(symbol: str, tf: str) -> tuple:
                """Train with semaphore to limit concurrency."""
                if self._training_cancelled:
                    return (f"{symbol}_{tf}", {"success": False, "error": "Cancelled"}, "failed")
                async with semaphore:
                    return await train_single_model(symbol, tf)

            # Run all training tasks in parallel (limited by semaphore)
            training_results = await asyncio.gather(
                *[train_with_semaphore(s, tf) for s, tf in all_tasks],
                return_exceptions=True
            )

            # Collect results (counters already updated in train_single_model)
            for result in training_results:
                if isinstance(result, Exception):
                    logger.error(f"Unexpected training error: {result}")
                    continue

                model_key, result_info, status = result
                results[model_key] = result_info

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

            # Note: Cache is managed by Data-Service, no local cache stats available
            summary["cache_stats"] = {
                "note": "Cache is managed by Data-Service"
            }

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
