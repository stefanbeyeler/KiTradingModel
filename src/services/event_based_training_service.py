"""
Event-Based Training Service for NHITS.

Uses trading events from EasyInsight logs API to trigger and enhance training.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter

import httpx

from ..config.settings import settings
from .nhits_training_service import nhits_training_service

logger = logging.getLogger(__name__)


class EventBasedTrainingService:
    """
    Service that monitors trading events and triggers NHITS training accordingly.

    Uses the EasyInsight /api/logs endpoint to detect:
    - High volatility events (ATR spikes)
    - Support/Resistance breaks (FXL events)
    - Custom trading signals

    When significant events are detected for a symbol, triggers model retraining.
    """

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._event_threshold = 10  # Number of events to trigger retraining
        self._check_interval_minutes = 15  # How often to check for events
        self._monitored_indicators = ["ATR", "FXL"]  # Indicators that trigger retraining

    async def start(self):
        """Start the event-based training monitor."""
        if self._running:
            logger.warning("Event-Based Training Service already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info(
            f"Event-Based Training Service started - "
            f"check interval: {self._check_interval_minutes}min"
        )

    async def stop(self):
        """Stop the event-based training monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Event-Based Training Service stopped")

    async def _monitoring_loop(self):
        """Background loop for monitoring events."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval_minutes * 60)

                if not self._running:
                    break

                # Analyze recent events
                await self._check_and_trigger_training()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_and_trigger_training(self):
        """Check recent events and trigger training if needed."""
        try:
            # Get events from last check interval
            lookback_minutes = self._check_interval_minutes * 2
            symbols_to_train = await self._analyze_recent_events(lookback_minutes)

            if symbols_to_train:
                logger.info(
                    f"Detected significant events for {len(symbols_to_train)} symbols: "
                    f"{', '.join(symbols_to_train)}"
                )

                # Trigger training for symbols with significant events
                # Use force=True to retrain even if recently trained
                asyncio.create_task(
                    nhits_training_service.train_all_symbols(
                        symbols=symbols_to_train,
                        force=True
                    )
                )

        except Exception as e:
            logger.error(f"Failed to check and trigger training: {e}")

    async def _analyze_recent_events(self, lookback_minutes: int) -> List[str]:
        """
        Analyze recent events and return symbols that need retraining.

        Args:
            lookback_minutes: How far back to look for events

        Returns:
            List of symbols that have significant events
        """
        try:
            since = datetime.utcnow() - timedelta(minutes=lookback_minutes)

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get recent events
                response = await client.get(
                    f"{settings.easyinsight_api_url}/logs",
                    params={
                        "limit": 1000,
                        "since": since.isoformat()
                    }
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("success") or not data.get("data"):
                    return []

                events = data["data"]

                # Analyze events by symbol
                symbol_events = {}
                for event in events:
                    symbol = event.get("symbol")
                    indicator = event.get("indicator")

                    if not symbol or indicator not in self._monitored_indicators:
                        continue

                    if symbol not in symbol_events:
                        symbol_events[symbol] = []

                    symbol_events[symbol].append(event)

                # Determine which symbols need retraining
                symbols_to_train = []
                for symbol, symbol_event_list in symbol_events.items():
                    if self._should_retrain_based_on_events(symbol_event_list):
                        symbols_to_train.append(symbol)

                return symbols_to_train

        except Exception as e:
            logger.error(f"Failed to analyze recent events: {e}")
            return []

    def _should_retrain_based_on_events(self, events: List[Dict[str, Any]]) -> bool:
        """
        Determine if events are significant enough to trigger retraining.

        Args:
            events: List of events for a symbol

        Returns:
            True if retraining should be triggered
        """
        if len(events) < self._event_threshold:
            return False

        # Count event types
        indicator_counts = Counter([e.get("indicator") for e in events])

        # Check for ATR spikes (volatility)
        atr_events = [e for e in events if e.get("indicator") == "ATR"]
        if len(atr_events) >= 5:
            # Parse ATR values to detect spikes
            try:
                atr_values = []
                for event in atr_events:
                    content = event.get("content", "")
                    # Parse "SYMBOL ATR: XX%"
                    if "ATR:" in content:
                        value_str = content.split("ATR:")[-1].strip().rstrip("%")
                        atr_values.append(int(value_str))

                if atr_values:
                    avg_atr = sum(atr_values) / len(atr_values)
                    # High volatility if ATR > 100%
                    if avg_atr > 100:
                        logger.info(f"High volatility detected: avg ATR = {avg_atr}%")
                        return True

            except Exception as e:
                logger.debug(f"Could not parse ATR values: {e}")

        # Check for FXL events (support/resistance)
        fxl_events = [e for e in events if e.get("indicator") == "FXL"]
        if len(fxl_events) >= 3:
            logger.info(f"Multiple FXL events detected: {len(fxl_events)}")
            return True

        return False

    async def get_event_summary(self, symbol: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of recent events.

        Args:
            symbol: Optional symbol to filter by
            hours: Number of hours to look back

        Returns:
            Dictionary with event statistics
        """
        try:
            params = {"limit": 1000}
            if symbol:
                params["symbol"] = symbol

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{settings.easyinsight_api_url}/logs",
                    params=params
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("success"):
                    return {"error": "API request failed"}

                events = data.get("data", [])

                # Filter by time
                from datetime import timezone
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
                recent_events = []
                for e in events:
                    try:
                        timestamp_str = e.get("timestamp", "")
                        if timestamp_str:
                            # Parse ISO format with timezone
                            event_time = datetime.fromisoformat(timestamp_str.replace("+01:00", "+00:00"))
                            # Make timezone aware if needed
                            if event_time.tzinfo is None:
                                event_time = event_time.replace(tzinfo=timezone.utc)
                            if event_time > cutoff:
                                recent_events.append(e)
                    except Exception:
                        continue

                # Analyze
                indicator_counts = Counter([e.get("indicator") for e in recent_events])
                symbol_counts = Counter([e.get("symbol") for e in recent_events])

                return {
                    "total_events": len(recent_events),
                    "timeframe_hours": hours,
                    "symbol_filter": symbol,
                    "indicators": dict(indicator_counts),
                    "top_symbols": dict(symbol_counts.most_common(10)),
                    "events": recent_events[:20]  # Latest 20
                }

        except Exception as e:
            logger.error(f"Failed to get event summary: {e}")
            return {"error": str(e)}


# Singleton instance
event_based_training_service = EventBasedTrainingService()
