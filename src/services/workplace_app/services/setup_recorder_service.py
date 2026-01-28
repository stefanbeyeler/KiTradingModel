"""
Setup Recorder Service.

Records trading setups to prediction_history for later evaluation.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from loguru import logger

from ..config import settings
from ..models.schemas import TradingSetup, SignalDirection


# Horizon mapping: timeframe -> evaluation delay
HORIZON_DELAYS = {
    "M1": timedelta(minutes=5),
    "M5": timedelta(minutes=15),
    "M15": timedelta(minutes=45),
    "M30": timedelta(hours=1, minutes=30),
    "H1": timedelta(hours=4),
    "H4": timedelta(hours=12),
    "D1": timedelta(days=1),
    "W1": timedelta(weeks=1),
    "MN": timedelta(days=30),
}


class SetupRecorderService:
    """Records trading setups to prediction history for evaluation."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._data_service_url = settings.data_service_url

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0)
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def record_setup(self, setup: TradingSetup) -> Optional[str]:
        """
        Record a trading setup to prediction history.

        Args:
            setup: The TradingSetup to record

        Returns:
            prediction_id if successful, None otherwise
        """
        try:
            client = await self._get_client()

            # Calculate target time based on timeframe
            horizon = setup.timeframe or "H1"
            delay = HORIZON_DELAYS.get(horizon, timedelta(hours=4))
            target_time = datetime.now(timezone.utc) + delay

            # Fetch current price as entry_price for later evaluation
            entry_price = getattr(setup, 'entry_price', None) or getattr(setup, 'current_price', None)
            if not entry_price:
                try:
                    # Try market-snapshot first (cached data)
                    price_response = await client.get(
                        f"{self._data_service_url}/api/v1/db/market-snapshot/{setup.symbol}",
                        params={"timeframe": setup.timeframe or "H1"}
                    )
                    if price_response.status_code == 200:
                        price_data = price_response.json()
                        entry_price = price_data.get("price", {}).get("last")

                    # Fallback: Try TwelveData OHLCV (fresh data)
                    if not entry_price:
                        # Map timeframe to TwelveData interval
                        tf_map = {
                            "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
                            "H1": "1h", "H4": "4h", "D1": "1day", "W1": "1week", "MN": "1month"
                        }
                        td_interval = tf_map.get(setup.timeframe or "H1", "1h")

                        ohlcv_response = await client.get(
                            f"{self._data_service_url}/api/v1/twelvedata/time_series/{setup.symbol}",
                            params={"interval": td_interval, "outputsize": 1}
                        )
                        if ohlcv_response.status_code == 200:
                            ohlcv_data = ohlcv_response.json()
                            values = ohlcv_data.get("values", [])
                            if values:
                                entry_price = float(values[0].get("close", 0))

                    if not entry_price:
                        logger.warning(f"Could not fetch entry price for {setup.symbol} - setup will not be evaluable")

                except Exception as e:
                    logger.debug(f"Could not fetch entry price for {setup.symbol}: {e}")

            # Build prediction data
            confidence_str = "low"
            if setup.confidence_level:
                confidence_str = setup.confidence_level.value

            prediction_data = {
                "service": "workplace",
                "symbol": setup.symbol,
                "timeframe": setup.timeframe or "H1",
                "prediction_type": "signal",
                "prediction": {
                    "direction": setup.direction.value if setup.direction else "neutral",
                    "composite_score": setup.composite_score,
                    "confidence_level": confidence_str,
                    "signals": {
                        "nhits": _signal_to_dict(setup.nhits_signal) if hasattr(setup, 'nhits_signal') and setup.nhits_signal else None,
                        "hmm": _signal_to_dict(setup.hmm_signal) if hasattr(setup, 'hmm_signal') and setup.hmm_signal else None,
                        "tcn": _signal_to_dict(setup.tcn_signal) if hasattr(setup, 'tcn_signal') and setup.tcn_signal else None,
                        "candlestick": _signal_to_dict(setup.candlestick_signal) if hasattr(setup, 'candlestick_signal') and setup.candlestick_signal else None,
                        "technical": _signal_to_dict(setup.technical_signal) if hasattr(setup, 'technical_signal') and setup.technical_signal else None,
                    },
                    "entry_price": entry_price,
                    "stop_loss": getattr(setup, 'stop_loss', None),
                    "take_profit": getattr(setup, 'take_profit', None),
                },
                "confidence": setup.composite_score / 100.0 if setup.composite_score else None,
                "target_time": target_time.isoformat(),
                "horizon": horizon,
                "triggered_by": "scanner",
                "tags": ["auto-scan", f"direction:{setup.direction.value}" if setup.direction else "direction:neutral"],
            }

            # Send to Data Service prediction history API
            response = await client.post(
                f"{self._data_service_url}/api/v1/predictions/",
                json=prediction_data
            )

            if response.status_code == 200:
                result = response.json()
                prediction_id = result.get("prediction_id")
                logger.debug(f"Recorded setup {setup.symbol}: {prediction_id}")
                return prediction_id
            else:
                logger.warning(f"Failed to record setup {setup.symbol}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error recording setup {setup.symbol}: {e}")
            return None

    async def evaluate_pending_setups(self) -> dict:
        """
        Fetch and evaluate setups that are due for evaluation.

        Returns:
            Statistics about evaluated setups
        """
        stats = {
            "evaluated": 0,
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
        }

        try:
            client = await self._get_client()

            # Get predictions due for evaluation
            response = await client.get(
                f"{self._data_service_url}/api/v1/predictions/due-for-evaluation",
                params={"service": "workplace", "limit": 50}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to fetch pending predictions: {response.status_code}")
                return stats

            predictions = response.json()

            for pred in predictions:
                try:
                    result = await self._evaluate_single_prediction(client, pred)
                    if result is None:
                        stats["errors"] += 1
                    elif result:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1
                    stats["evaluated"] += 1
                except Exception as e:
                    logger.error(f"Error evaluating prediction {pred.get('prediction_id')}: {e}")
                    stats["errors"] += 1

            if stats["evaluated"] > 0:
                logger.info(
                    f"Evaluated {stats['evaluated']} setups: "
                    f"{stats['correct']} correct, {stats['incorrect']} incorrect"
                )

        except Exception as e:
            logger.error(f"Error in evaluate_pending_setups: {e}")

        return stats

    async def _evaluate_single_prediction(
        self,
        client: httpx.AsyncClient,
        prediction: dict
    ) -> Optional[bool]:
        """
        Evaluate a single prediction against actual market data.

        Returns:
            True if correct, False if incorrect, None if error
        """
        try:
            prediction_id = prediction.get("prediction_id")
            symbol = prediction.get("symbol")
            timeframe = prediction.get("timeframe", "H1")

            # Get the original prediction details
            detail_response = await client.get(
                f"{self._data_service_url}/api/v1/predictions/{prediction_id}"
            )

            if detail_response.status_code != 200:
                return None

            pred_detail = detail_response.json()
            pred_data = pred_detail.get("prediction", {})
            direction = pred_data.get("direction", "neutral")
            entry_price = pred_data.get("entry_price")
            predicted_at = pred_detail.get("predicted_at")

            if not entry_price or direction == "neutral":
                # Can't evaluate without entry price or neutral direction
                return None

            # Get current price from Data Service
            price_response = await client.get(
                f"{self._data_service_url}/api/v1/db/market-snapshot/{symbol}",
                params={"timeframe": timeframe}
            )

            if price_response.status_code != 200:
                return None

            price_data = price_response.json()
            current_price = price_data.get("price", {}).get("last")

            if not current_price:
                return None

            # Calculate price change
            price_change = (current_price - entry_price) / entry_price * 100

            # Determine if prediction was correct
            is_correct = False
            if direction == "long" and price_change > 0:
                is_correct = True
            elif direction == "short" and price_change < 0:
                is_correct = True

            # Calculate accuracy score (how much it moved in predicted direction)
            accuracy_score = abs(price_change) / 10.0  # Normalize to 0-1 range (10% = 1.0)
            if not is_correct:
                accuracy_score = -accuracy_score

            # Send evaluation to Data Service
            eval_response = await client.put(
                f"{self._data_service_url}/api/v1/predictions/{prediction_id}/evaluate",
                json={
                    "actual_outcome": {
                        "current_price": current_price,
                        "price_change_percent": price_change,
                        "direction_correct": is_correct,
                    },
                    "is_correct": is_correct,
                    "accuracy_score": min(1.0, max(-1.0, accuracy_score)),
                    "error_amount": abs(price_change) if not is_correct else None,
                }
            )

            return is_correct if eval_response.status_code == 200 else None

        except Exception as e:
            logger.error(f"Error evaluating prediction: {e}")
            return None

    async def get_accuracy_stats(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> dict:
        """
        Get accuracy statistics for workplace setups.

        Args:
            symbol: Optional symbol filter
            days: Number of days to look back

        Returns:
            Statistics dictionary
        """
        try:
            client = await self._get_client()

            params = {"service": "workplace"}
            if symbol:
                params["symbol"] = symbol

            response = await client.get(
                f"{self._data_service_url}/api/v1/predictions/stats",
                params=params
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting accuracy stats: {e}")
            return {}

    async def cleanup_invalid_predictions(self) -> dict:
        """
        Delete predictions that cannot be evaluated (no entry_price).

        Uses the Data Service's optimized cleanup endpoint for fast execution.

        Returns:
            Statistics about cleanup operation
        """
        stats = {
            "checked": 0,
            "deleted": 0,
            "errors": 0,
        }

        try:
            client = await self._get_client()

            # First, count total predictions to report "checked"
            count_response = await client.get(
                f"{self._data_service_url}/api/v1/predictions/",
                params={"service": "workplace", "limit": 1}
            )

            # Use Data Service's optimized cleanup endpoint (runs directly in DB)
            response = await client.post(
                f"{self._data_service_url}/api/v1/predictions/cleanup-invalid",
                params={"service": "workplace"},
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                stats["deleted"] = result.get("deleted_count", 0)

                # Get updated count
                new_count_response = await client.get(
                    f"{self._data_service_url}/api/v1/predictions/stats",
                    params={"service": "workplace"}
                )
                if new_count_response.status_code == 200:
                    new_stats = new_count_response.json()
                    stats["checked"] = new_stats.get("total_predictions", 0) + stats["deleted"]

                logger.info(
                    f"Cleanup: {stats['checked']} geprüft, "
                    f"{stats['deleted']} gelöscht"
                )
            else:
                logger.warning(f"Cleanup failed: {response.status_code}")
                stats["errors"] = 1

        except Exception as e:
            logger.error(f"Error in cleanup_invalid_predictions: {e}")
            stats["errors"] = 1

        return stats


def _signal_to_dict(signal) -> dict:
    """Convert a signal object to a serializable dict."""
    if signal is None:
        return None

    try:
        # Handle Pydantic models
        if hasattr(signal, "model_dump"):
            return signal.model_dump()
        elif hasattr(signal, "dict"):
            return signal.dict()
        else:
            return dict(signal)
    except Exception:
        return {"available": False}


# Singleton instance
setup_recorder = SetupRecorderService()
