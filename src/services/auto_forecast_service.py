"""
Auto Forecast Service - Automatically generates forecasts for favorite and non-favorite symbols.

This service provides two modes:
1. **Favorites Mode**: Timeframe-based auto-forecasts
   - M15: Every 15 minutes when new data is available
   - H1: Every hour when new data is available
   - D1: Once daily when new data is available

2. **Non-Favorites Mode**: Daily scheduled forecasts
   - Runs once daily at a configurable time (default: 05:00 local time)
   - Generates H1 forecasts for all non-favorite symbols with trained models

Both modes can run independently or together.
"""

import asyncio
from datetime import datetime, timezone, time, timedelta
from typing import Optional, Dict, List
import httpx
from loguru import logger
import pytz

from ..config import settings


class AutoForecastService:
    """Service for automatically generating forecasts for favorite and non-favorite symbols."""

    def __init__(self):
        # Favorites auto-forecast state
        self._favorites_running = False
        self._favorites_tasks: Dict[str, asyncio.Task] = {}  # timeframe -> task
        self._favorites_last_run: Dict[str, datetime] = {}  # timeframe -> last run time
        self._favorites_enabled_timeframes: List[str] = []  # Which timeframes are active

        # Non-favorites (daily) auto-forecast state
        self._daily_running = False
        self._daily_task: Optional[asyncio.Task] = None
        self._daily_last_run: Optional[datetime] = None
        self._daily_scheduled_time: time = time(5, 0)  # Default: 05:00 local time
        self._daily_timezone: str = settings.display_timezone  # From central config

        # Legacy compatibility
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None

        # Shared forecast results
        self._forecast_results: dict = {}  # symbol_timeframe -> last forecast info
        self._favorites_forecast_count: int = 0
        self._daily_forecast_count: int = 0

        # Timeframe intervals for favorites (how often to generate forecasts)
        self._timeframe_intervals = {
            "M15": 900,    # 15 minutes
            "H1": 3600,    # 1 hour
            "D1": 86400,   # 24 hours
        }

        # Statistics
        self._favorites_total_forecasts: int = 0
        self._daily_total_forecasts: int = 0
        self._favorites_failed_forecasts: int = 0
        self._daily_failed_forecasts: int = 0

        logger.info("AutoForecastService initialized with favorites and daily mode support")

    async def get_favorites(self) -> list[dict]:
        """Fetch favorite symbols from Data Service."""
        data_service_url = getattr(settings, 'data_service_url', 'http://localhost:3001')
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{data_service_url}/api/v1/managed-symbols",
                    params={"favorites_only": "true"}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch favorites: {e}")
        return []

    async def get_available_models(self) -> dict:
        """Get available NHITS models grouped by base symbol."""
        from .forecast_service import forecast_service

        models = forecast_service.list_models()
        models_by_symbol = {}

        for model in models:
            if not model.model_exists:
                continue

            symbol = model.symbol
            base_symbol = symbol
            timeframe = "H1"  # default

            for tf in ["M15", "H1", "D1"]:
                if symbol.endswith(f"_{tf}"):
                    base_symbol = symbol[:-len(f"_{tf}")]
                    timeframe = tf
                    break

            if base_symbol not in models_by_symbol:
                models_by_symbol[base_symbol] = {}
            models_by_symbol[base_symbol][timeframe] = {
                "model_symbol": symbol,
                "last_trained": model.last_trained,
            }

        return models_by_symbol

    async def generate_forecast(self, symbol: str, timeframe: str = "H1") -> dict:
        """Generate a forecast for a specific symbol and timeframe."""
        from datetime import timedelta
        from .forecast_service import forecast_service
        from . import AnalysisService
        from ..models.forecast_data import ForecastConfig

        analysis_service = AnalysisService()

        # Determine data requirements based on timeframe
        if timeframe == "M15":
            days_needed = 2
            interval = "m15"
        elif timeframe == "D1":
            days_needed = 60
            interval = "d1"
        else:  # H1
            days_needed = 30
            interval = "h1"

        try:
            # Fetch time series data
            time_series = await analysis_service._fetch_time_series(
                symbol=symbol,
                start_date=datetime.now(timezone.utc) - timedelta(days=days_needed),
                end_date=datetime.now(timezone.utc),
                interval=interval
            )

            if not time_series:
                return {
                    "success": False,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "error": f"No data available for {symbol}",
                }

            # Create config
            config = ForecastConfig(
                symbol=symbol,
                timeframe=timeframe
            )

            # Generate forecast
            result = await forecast_service.forecast(
                time_series=time_series,
                symbol=symbol,
                config=config,
                timeframe=timeframe
            )

            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "predicted_prices": result.predicted_prices[:5] if result.predicted_prices else [],
                "trend": "up" if result.predicted_prices and result.predicted_prices[-1] > result.predicted_prices[0] else "down",
            }
        except Exception as e:
            logger.warning(f"Failed to generate forecast for {symbol}/{timeframe}: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(e),
            }

    async def run_forecasts_for_favorites(self, timeframe: Optional[str] = None) -> dict:
        """
        Run forecasts for all favorite symbols.

        Args:
            timeframe: Optional specific timeframe (M15, H1, D1).
                       If None, runs all available timeframes.

        Returns:
            Dict with forecast results summary.
        """
        favorites = await self.get_favorites()
        models_by_symbol = await self.get_available_models()

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "favorites_count": len(favorites),
            "forecasts_generated": 0,
            "forecasts_failed": 0,
            "details": [],
        }

        for favorite in favorites:
            symbol = favorite.get("symbol")
            if not symbol:
                continue

            available_models = models_by_symbol.get(symbol, {})
            if not available_models:
                logger.debug(f"No models available for favorite {symbol}")
                continue

            # Determine which timeframes to forecast
            timeframes_to_run = [timeframe] if timeframe else list(available_models.keys())

            for tf in timeframes_to_run:
                if tf not in available_models:
                    continue

                forecast_result = await self.generate_forecast(symbol, tf)
                results["details"].append(forecast_result)

                if forecast_result.get("success"):
                    results["forecasts_generated"] += 1
                    self._forecast_results[f"{symbol}_{tf}"] = forecast_result
                else:
                    results["forecasts_failed"] += 1

        self._last_run = datetime.now(timezone.utc)
        logger.info(
            f"Auto-forecast completed: {results['forecasts_generated']} generated, "
            f"{results['forecasts_failed']} failed"
        )

        return results

    async def get_non_favorites(self) -> list[dict]:
        """Fetch non-favorite symbols that have trained models from Data Service."""
        data_service_url = getattr(settings, 'data_service_url', 'http://localhost:3001')
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{data_service_url}/api/v1/managed-symbols",
                    params={"favorites_only": "false"}
                )
                if response.status_code == 200:
                    all_symbols = response.json()
                    # Filter to non-favorites only
                    return [s for s in all_symbols if not s.get("is_favorite", False)]
        except Exception as e:
            logger.error(f"Failed to fetch non-favorites: {e}")
        return []

    # ==================== FAVORITES AUTO-FORECAST ====================

    async def _favorites_forecast_loop(self, timeframe: str):
        """
        Background loop for generating forecasts for favorites at timeframe-specific intervals.

        - M15: Every 15 minutes
        - H1: Every hour
        - D1: Once daily
        """
        interval = self._timeframe_intervals.get(timeframe, 3600)
        logger.info(f"Starting favorites auto-forecast loop for {timeframe} (interval: {interval}s)")

        while self._favorites_running and timeframe in self._favorites_enabled_timeframes:
            try:
                result = await self.run_forecasts_for_favorites(timeframe)
                self._favorites_last_run[timeframe] = datetime.now(timezone.utc)
                self._favorites_total_forecasts += result.get("forecasts_generated", 0)
                self._favorites_failed_forecasts += result.get("forecasts_failed", 0)
            except Exception as e:
                logger.error(f"Error in favorites auto-forecast loop ({timeframe}): {e}")

            await asyncio.sleep(interval)

        logger.info(f"Favorites auto-forecast loop for {timeframe} stopped")

    async def start_favorites_auto_forecast(self, timeframes: Optional[List[str]] = None):
        """
        Start auto-forecasting for favorite symbols.

        Args:
            timeframes: List of timeframes to enable (M15, H1, D1).
                        If None, enables all timeframes.
        """
        if timeframes is None:
            timeframes = ["M15", "H1", "D1"]

        # Validate timeframes
        valid_timeframes = [tf for tf in timeframes if tf in self._timeframe_intervals]
        if not valid_timeframes:
            logger.error("No valid timeframes specified for favorites auto-forecast")
            return

        self._favorites_running = True
        self._favorites_enabled_timeframes = valid_timeframes

        # Start a loop for each timeframe
        for tf in valid_timeframes:
            if tf not in self._favorites_tasks or self._favorites_tasks[tf].done():
                self._favorites_tasks[tf] = asyncio.create_task(
                    self._favorites_forecast_loop(tf)
                )
                logger.info(f"Started favorites auto-forecast for {tf}")

        logger.info(f"Favorites auto-forecast started for timeframes: {valid_timeframes}")

    async def stop_favorites_auto_forecast(self, timeframes: Optional[List[str]] = None):
        """
        Stop auto-forecasting for favorite symbols.

        Args:
            timeframes: List of timeframes to stop. If None, stops all.
        """
        if timeframes is None:
            timeframes = list(self._favorites_tasks.keys())

        for tf in timeframes:
            if tf in self._favorites_enabled_timeframes:
                self._favorites_enabled_timeframes.remove(tf)

            if tf in self._favorites_tasks:
                task = self._favorites_tasks[tf]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self._favorites_tasks[tf]
                logger.info(f"Stopped favorites auto-forecast for {tf}")

        if not self._favorites_enabled_timeframes:
            self._favorites_running = False
            logger.info("All favorites auto-forecast loops stopped")

    # ==================== DAILY (NON-FAVORITES) AUTO-FORECAST ====================

    async def _calculate_seconds_until_scheduled_time(self) -> float:
        """Calculate seconds until the next scheduled daily forecast time."""
        try:
            tz = pytz.timezone(self._daily_timezone)
        except Exception:
            tz = pytz.timezone(settings.display_timezone)

        now = datetime.now(tz)
        scheduled_today = now.replace(
            hour=self._daily_scheduled_time.hour,
            minute=self._daily_scheduled_time.minute,
            second=0,
            microsecond=0
        )

        if now >= scheduled_today:
            # Already past today's scheduled time, wait until tomorrow
            scheduled_tomorrow = scheduled_today + timedelta(days=1)
            seconds_until = (scheduled_tomorrow - now).total_seconds()
        else:
            seconds_until = (scheduled_today - now).total_seconds()

        return max(0, seconds_until)

    async def _daily_forecast_loop(self):
        """
        Background loop for generating daily forecasts for non-favorite symbols.

        Runs once daily at the configured time.
        """
        logger.info(
            f"Starting daily auto-forecast loop "
            f"(scheduled at {self._daily_scheduled_time.strftime('%H:%M')} {self._daily_timezone})"
        )

        while self._daily_running:
            try:
                # Wait until scheduled time
                seconds_until = await self._calculate_seconds_until_scheduled_time()
                logger.info(f"Daily forecast scheduled in {seconds_until / 3600:.1f} hours")

                # Sleep in chunks to allow for cancellation
                while seconds_until > 0 and self._daily_running:
                    sleep_time = min(60, seconds_until)  # Sleep max 60s at a time
                    await asyncio.sleep(sleep_time)
                    seconds_until -= sleep_time

                if not self._daily_running:
                    break

                # Run forecasts for non-favorites
                logger.info("Running scheduled daily forecasts for non-favorite symbols")
                result = await self.run_forecasts_for_non_favorites()
                self._daily_last_run = datetime.now(timezone.utc)
                self._daily_total_forecasts += result.get("forecasts_generated", 0)
                self._daily_failed_forecasts += result.get("forecasts_failed", 0)

                # Wait at least 1 hour before checking schedule again
                # (prevents multiple runs if loop restarts quickly)
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in daily auto-forecast loop: {e}")
                await asyncio.sleep(300)  # Wait 5 min before retry

        logger.info("Daily auto-forecast loop stopped")

    async def run_forecasts_for_non_favorites(self, timeframe: str = "H1") -> dict:
        """
        Run forecasts for all non-favorite symbols that have trained models.

        Args:
            timeframe: Timeframe to forecast (default: H1)

        Returns:
            Dict with forecast results summary.
        """
        non_favorites = await self.get_non_favorites()
        models_by_symbol = await self.get_available_models()

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "daily",
            "timeframe": timeframe,
            "symbols_count": len(non_favorites),
            "forecasts_generated": 0,
            "forecasts_failed": 0,
            "skipped_no_model": 0,
            "details": [],
        }

        for symbol_info in non_favorites:
            symbol = symbol_info.get("symbol")
            if not symbol:
                continue

            available_models = models_by_symbol.get(symbol, {})
            if timeframe not in available_models:
                results["skipped_no_model"] += 1
                continue

            forecast_result = await self.generate_forecast(symbol, timeframe)
            results["details"].append(forecast_result)

            if forecast_result.get("success"):
                results["forecasts_generated"] += 1
                self._forecast_results[f"{symbol}_{timeframe}"] = forecast_result
            else:
                results["forecasts_failed"] += 1

        logger.info(
            f"Daily auto-forecast completed: {results['forecasts_generated']} generated, "
            f"{results['forecasts_failed']} failed, {results['skipped_no_model']} skipped (no model)"
        )

        return results

    async def start_daily_auto_forecast(
        self,
        scheduled_time: Optional[str] = None,
        timezone_str: Optional[str] = None
    ):
        """
        Start daily auto-forecasting for non-favorite symbols.

        Args:
            scheduled_time: Time in HH:MM format (default: "05:00")
            timezone_str: Timezone string (default: from settings.display_timezone)
        """
        if self._daily_running:
            logger.warning("Daily auto-forecast already running")
            return

        # Parse scheduled time
        if scheduled_time:
            try:
                parts = scheduled_time.split(":")
                self._daily_scheduled_time = time(int(parts[0]), int(parts[1]))
            except Exception as e:
                logger.error(f"Invalid scheduled time format: {e}")
                self._daily_scheduled_time = time(5, 0)

        if timezone_str:
            try:
                pytz.timezone(timezone_str)  # Validate timezone
                self._daily_timezone = timezone_str
            except Exception as e:
                logger.error(f"Invalid timezone: {e}")
                self._daily_timezone = settings.display_timezone

        self._daily_running = True
        self._daily_task = asyncio.create_task(self._daily_forecast_loop())

        logger.info(
            f"Daily auto-forecast started "
            f"(scheduled at {self._daily_scheduled_time.strftime('%H:%M')} {self._daily_timezone})"
        )

    async def stop_daily_auto_forecast(self):
        """Stop daily auto-forecasting for non-favorite symbols."""
        if not self._daily_running:
            return

        self._daily_running = False
        if self._daily_task:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except asyncio.CancelledError:
                pass
            self._daily_task = None

        logger.info("Daily auto-forecast stopped")

    def set_daily_schedule(self, scheduled_time: str, timezone_str: Optional[str] = None):
        """
        Update the daily forecast schedule without restarting.

        Args:
            scheduled_time: Time in HH:MM format
            timezone_str: Optional timezone string
        """
        try:
            parts = scheduled_time.split(":")
            self._daily_scheduled_time = time(int(parts[0]), int(parts[1]))
            logger.info(f"Daily schedule updated to {scheduled_time}")
        except Exception as e:
            logger.error(f"Invalid scheduled time format: {e}")

        if timezone_str:
            try:
                pytz.timezone(timezone_str)
                self._daily_timezone = timezone_str
                logger.info(f"Daily timezone updated to {timezone_str}")
            except Exception as e:
                logger.error(f"Invalid timezone: {e}")

    # ==================== LEGACY COMPATIBILITY ====================

    async def _forecast_loop(self, timeframe: str = "H1"):
        """Legacy: Background loop for generating forecasts at regular intervals."""
        interval = self._timeframe_intervals.get(timeframe, 3600)
        logger.info(f"Starting legacy auto-forecast loop for {timeframe} (interval: {interval}s)")

        while self._running:
            try:
                await self.run_forecasts_for_favorites(timeframe)
            except Exception as e:
                logger.error(f"Error in auto-forecast loop: {e}")

            await asyncio.sleep(interval)

    async def start(self, timeframe: str = "H1"):
        """Legacy: Start the auto-forecast background service."""
        if self._running:
            logger.warning("AutoForecastService already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._forecast_loop(timeframe))
        logger.info(f"AutoForecastService started for {timeframe}")

    async def stop(self):
        """Stop all auto-forecast services."""
        # Stop legacy loop
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None

        # Stop favorites auto-forecast
        await self.stop_favorites_auto_forecast()

        # Stop daily auto-forecast
        await self.stop_daily_auto_forecast()

        logger.info("All AutoForecastService loops stopped")

    # ==================== STATUS & INFO ====================

    def get_status(self) -> dict:
        """Get comprehensive status of the auto-forecast service."""
        # Build favorites status per timeframe
        favorites_timeframe_status = {}
        for tf in ["M15", "H1", "D1"]:
            task = self._favorites_tasks.get(tf)
            last_run = self._favorites_last_run.get(tf)
            favorites_timeframe_status[tf] = {
                "running": tf in self._favorites_enabled_timeframes and task and not task.done(),
                "last_run": last_run.isoformat() if last_run else None,
                "interval_seconds": self._timeframe_intervals[tf],
            }

        # Calculate next daily run
        try:
            seconds_until = asyncio.get_event_loop().run_until_complete(
                self._calculate_seconds_until_scheduled_time()
            )
            next_daily_run = (datetime.now(timezone.utc) + timedelta(seconds=seconds_until)).isoformat()
        except Exception:
            next_daily_run = None

        return {
            # Favorites status
            "favorites": {
                "running": self._favorites_running,
                "enabled_timeframes": self._favorites_enabled_timeframes,
                "timeframes": favorites_timeframe_status,
                "total_forecasts": self._favorites_total_forecasts,
                "failed_forecasts": self._favorites_failed_forecasts,
            },
            # Daily (non-favorites) status
            "daily": {
                "running": self._daily_running,
                "scheduled_time": self._daily_scheduled_time.strftime("%H:%M"),
                "timezone": self._daily_timezone,
                "last_run": self._daily_last_run.isoformat() if self._daily_last_run else None,
                "next_run": next_daily_run,
                "total_forecasts": self._daily_total_forecasts,
                "failed_forecasts": self._daily_failed_forecasts,
            },
            # Legacy compatibility
            "running": self._running or self._favorites_running or self._daily_running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "forecast_count": len(self._forecast_results),
        }

    def get_latest_forecasts(self, favorites_only: bool = True) -> list[dict]:
        """Get latest forecast results."""
        return list(self._forecast_results.values())

    def get_favorites_forecast_history(self) -> Dict[str, list]:
        """Get forecast history grouped by timeframe."""
        history = {"M15": [], "H1": [], "D1": []}
        for key, result in self._forecast_results.items():
            for tf in ["M15", "H1", "D1"]:
                if key.endswith(f"_{tf}"):
                    history[tf].append(result)
                    break
        return history


# Singleton instance
auto_forecast_service = AutoForecastService()
