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


# Horizon mapping: timeframe -> evaluation delay (längere Horizonte für realistische Evaluation)
HORIZON_DELAYS = {
    "M1": timedelta(minutes=10),
    "M5": timedelta(minutes=30),
    "M15": timedelta(hours=1),
    "M30": timedelta(hours=2),
    "H1": timedelta(hours=8),       # 8x Timeframe für Trendbestätigung
    "H4": timedelta(days=1),        # Nach Overnight-Session
    "D1": timedelta(days=3),        # 3 Tage für Daily-Trends
    "W1": timedelta(weeks=2),       # 2 Wochen
    "MN": timedelta(days=60),       # 2 Monate
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
            # Priority: 1) setup data, 2) TwelveData (fresh), 3) market-snapshot (cached)
            entry_price = getattr(setup, 'entry_price', None) or getattr(setup, 'current_price', None)
            if not entry_price:
                try:
                    # Map timeframe to TwelveData interval
                    tf_map = {
                        "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
                        "H1": "1h", "H4": "4h", "D1": "1day", "W1": "1week", "MN": "1month"
                    }
                    td_interval = tf_map.get(setup.timeframe or "H1", "1h")

                    # Try TwelveData first (fresh data)
                    ohlcv_response = await client.get(
                        f"{self._data_service_url}/api/v1/twelvedata/time_series/{setup.symbol}",
                        params={"interval": td_interval, "outputsize": 1}
                    )
                    if ohlcv_response.status_code == 200:
                        ohlcv_data = ohlcv_response.json()
                        values = ohlcv_data.get("values", [])
                        if values:
                            entry_price = float(values[0].get("close", 0))

                    # Fallback: Try market-snapshot (cached data)
                    if not entry_price:
                        price_response = await client.get(
                            f"{self._data_service_url}/api/v1/db/market-snapshot/{setup.symbol}",
                            params={"timeframe": setup.timeframe or "H1"}
                        )
                        if price_response.status_code == 200:
                            price_data = price_response.json()
                            entry_price = price_data.get("price", {}).get("last")

                    if not entry_price:
                        logger.warning(f"Could not fetch entry price for {setup.symbol} - setup will not be evaluable")

                except Exception as e:
                    logger.debug(f"Could not fetch entry price for {setup.symbol}: {e}")

            # Build prediction data
            confidence_str = "low"
            if setup.confidence_level:
                confidence_str = setup.confidence_level.value

            # Build complete signals dict including CNN-LSTM
            signals_dict = {
                "nhits": _signal_to_dict(setup.nhits_signal) if hasattr(setup, 'nhits_signal') and setup.nhits_signal else None,
                "hmm": _signal_to_dict(setup.hmm_signal) if hasattr(setup, 'hmm_signal') and setup.hmm_signal else None,
                "tcn": _signal_to_dict(setup.tcn_signal) if hasattr(setup, 'tcn_signal') and setup.tcn_signal else None,
                "candlestick": _signal_to_dict(setup.candlestick_signal) if hasattr(setup, 'candlestick_signal') and setup.candlestick_signal else None,
                "technical": _signal_to_dict(setup.technical_signal) if hasattr(setup, 'technical_signal') and setup.technical_signal else None,
                "cnn_lstm": _signal_to_dict(setup.cnn_lstm_signal) if hasattr(setup, 'cnn_lstm_signal') and setup.cnn_lstm_signal else None,
            }

            prediction_data = {
                "service": "workplace",
                "symbol": setup.symbol,
                "timeframe": setup.timeframe or "H1",
                "prediction_type": "signal",
                "prediction": {
                    "direction": setup.direction.value if setup.direction else "neutral",
                    "composite_score": setup.composite_score,
                    "confidence_level": confidence_str,
                    "signals": signals_dict,
                    # Aggregation metadata
                    "signal_alignment": setup.signal_alignment.value if hasattr(setup, 'signal_alignment') and setup.signal_alignment else "mixed",
                    "key_drivers": setup.key_drivers if hasattr(setup, 'key_drivers') else [],
                    "signals_available": setup.signals_available if hasattr(setup, 'signals_available') else 0,
                    # Multi-Timeframe scores
                    "timeframe_scores": setup.timeframe_scores if hasattr(setup, 'timeframe_scores') else None,
                    # Price data
                    "current_price": setup.current_price if hasattr(setup, 'current_price') else None,
                    "entry_price": entry_price,
                    # Entry/Exit Levels für TP/SL-basierte Evaluation
                    "stop_loss": setup.entry_exit_levels.stop_loss if hasattr(setup, 'entry_exit_levels') and setup.entry_exit_levels else None,
                    "take_profit_1": setup.entry_exit_levels.take_profit_1 if hasattr(setup, 'entry_exit_levels') and setup.entry_exit_levels else None,
                    "take_profit_2": setup.entry_exit_levels.take_profit_2 if hasattr(setup, 'entry_exit_levels') and setup.entry_exit_levels else None,
                    "take_profit_3": setup.entry_exit_levels.take_profit_3 if hasattr(setup, 'entry_exit_levels') and setup.entry_exit_levels else None,
                    "risk_reward_ratio": setup.entry_exit_levels.risk_reward_ratio if hasattr(setup, 'entry_exit_levels') and setup.entry_exit_levels else None,
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
            stop_loss = pred_data.get("stop_loss")
            take_profit_1 = pred_data.get("take_profit_1")
            predicted_at = pred_detail.get("predicted_at")

            if not entry_price or direction == "neutral":
                # Can't evaluate without entry price or neutral direction
                return None

            # Get current price from TwelveData (fresh data, not cached)
            # Map timeframe to TwelveData interval
            tf_map = {
                "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
                "H1": "1h", "H4": "4h", "D1": "1day", "W1": "1week", "MN": "1month"
            }
            td_interval = tf_map.get(timeframe, "1h")

            price_response = await client.get(
                f"{self._data_service_url}/api/v1/twelvedata/time_series/{symbol}",
                params={"interval": td_interval, "outputsize": 1}
            )

            current_price = None
            if price_response.status_code == 200:
                price_data = price_response.json()
                values = price_data.get("values", [])
                if values:
                    current_price = float(values[0].get("close", 0))

            # Fallback to market-snapshot if TwelveData fails
            if not current_price:
                snapshot_response = await client.get(
                    f"{self._data_service_url}/api/v1/db/market-snapshot/{symbol}",
                    params={"timeframe": timeframe}
                )
                if snapshot_response.status_code == 200:
                    snapshot_data = snapshot_response.json()
                    current_price = snapshot_data.get("price", {}).get("last")

            if not current_price:
                logger.warning(f"Could not fetch current price for {symbol} evaluation")
                return None

            # Calculate price change
            price_change = (current_price - entry_price) / entry_price * 100

            # TP/SL-basierte Evaluation: Erfolg = TP1 erreicht, Fehlschlag = SL erreicht
            is_correct = None  # None = noch nicht entschieden (Timeout ohne TP/SL)
            outcome_reason = "timeout"

            if direction == "long":
                if take_profit_1 and current_price >= take_profit_1:
                    is_correct = True
                    outcome_reason = "tp1_reached"
                elif stop_loss and current_price <= stop_loss:
                    is_correct = False
                    outcome_reason = "sl_hit"
                else:
                    # Timeout: Prüfe ob mindestens in richtige Richtung
                    is_correct = price_change > 0.3  # Min 0.3% Bewegung
                    outcome_reason = "timeout_direction"
            elif direction == "short":
                if take_profit_1 and current_price <= take_profit_1:
                    is_correct = True
                    outcome_reason = "tp1_reached"
                elif stop_loss and current_price >= stop_loss:
                    is_correct = False
                    outcome_reason = "sl_hit"
                else:
                    # Timeout: Prüfe ob mindestens in richtige Richtung
                    is_correct = price_change < -0.3  # Min 0.3% Bewegung
                    outcome_reason = "timeout_direction"

            # Calculate accuracy score
            if take_profit_1 and stop_loss:
                # Berechne wie weit Richtung TP vs SL
                if direction == "long":
                    tp_distance = take_profit_1 - entry_price
                    sl_distance = entry_price - stop_loss
                    price_move = current_price - entry_price
                    if tp_distance > 0:
                        accuracy_score = price_move / tp_distance  # 1.0 = TP erreicht
                    else:
                        accuracy_score = 0
                else:  # short
                    tp_distance = entry_price - take_profit_1
                    sl_distance = stop_loss - entry_price
                    price_move = entry_price - current_price
                    if tp_distance > 0:
                        accuracy_score = price_move / tp_distance
                    else:
                        accuracy_score = 0
            else:
                # Fallback: Prozent-basiert
                accuracy_score = abs(price_change) / 10.0
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
                        "outcome_reason": outcome_reason,
                        "stop_loss": stop_loss,
                        "take_profit_1": take_profit_1,
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
