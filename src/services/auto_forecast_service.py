"""
Auto Forecast Service - Automatically generates forecasts for favorite symbols.

This service:
- Fetches favorite symbols from Data Service
- Generates NHITS forecasts for each favorite based on available timeframes
- Runs periodically in the background
- Stores forecasts for performance tracking
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
import httpx
from loguru import logger

from ..config import settings


class AutoForecastService:
    """Service for automatically generating forecasts for favorite symbols."""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None
        self._forecast_results: dict = {}  # symbol -> last forecast info
        self._interval_seconds = 3600  # Default: 1 hour for H1, adjusted per timeframe

        # Timeframe intervals (how often to generate forecasts)
        self._timeframe_intervals = {
            "M15": 900,    # 15 minutes
            "H1": 3600,    # 1 hour
            "D1": 86400,   # 24 hours
        }

        logger.info("AutoForecastService initialized")

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

    async def _forecast_loop(self, timeframe: str = "H1"):
        """Background loop for generating forecasts at regular intervals."""
        interval = self._timeframe_intervals.get(timeframe, 3600)
        logger.info(f"Starting auto-forecast loop for {timeframe} (interval: {interval}s)")

        while self._running:
            try:
                await self.run_forecasts_for_favorites(timeframe)
            except Exception as e:
                logger.error(f"Error in auto-forecast loop: {e}")

            await asyncio.sleep(interval)

    async def start(self, timeframe: str = "H1"):
        """Start the auto-forecast background service."""
        if self._running:
            logger.warning("AutoForecastService already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._forecast_loop(timeframe))
        logger.info(f"AutoForecastService started for {timeframe}")

    async def stop(self):
        """Stop the auto-forecast service."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("AutoForecastService stopped")

    def get_status(self) -> dict:
        """Get current status of the auto-forecast service."""
        return {
            "running": self._running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "forecast_count": len(self._forecast_results),
            "forecasts": self._forecast_results,
        }

    def get_latest_forecasts(self, favorites_only: bool = True) -> list[dict]:
        """Get latest forecast results."""
        return list(self._forecast_results.values())


# Singleton instance
auto_forecast_service = AutoForecastService()
