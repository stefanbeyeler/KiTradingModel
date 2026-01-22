"""
Prediction History Service - Speichert und verwaltet CNN-LSTM Vorhersagen.

Dieser Service speichert alle Predictions fuer spaetere Validierung und Backtesting.
Analog zum PatternHistoryService des Candlestick Service.
"""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

from loguru import logger


# Data directory
DATA_DIR = os.getenv("DATA_DIR", "/app/data")


@dataclass
class PredictionHistoryEntry:
    """Einzelner Eintrag in der Prediction-History."""
    id: str
    timestamp: str  # Vorhersage-Zeitpunkt
    symbol: str
    timeframe: str
    model_version: str

    # Price Prediction
    price_current: float
    price_forecast_1h: Optional[float] = None
    price_forecast_4h: Optional[float] = None
    price_forecast_1d: Optional[float] = None
    price_forecast_1w: Optional[float] = None
    price_direction: str = "neutral"
    price_confidence: float = 0.0
    price_change_percent_1h: Optional[float] = None
    price_change_percent_1d: Optional[float] = None

    # Pattern Predictions (Top 3)
    patterns: list = field(default_factory=list)

    # Regime Prediction
    regime_current: str = "sideways"
    regime_probability: float = 0.0
    regime_transition_probs: dict = field(default_factory=dict)

    # Metadata
    inference_time_ms: float = 0.0
    ohlcv_snapshot: Optional[list] = None  # OHLCV-Daten zum Zeitpunkt der Vorhersage

    # Feedback Status
    feedback_status: str = "pending"  # pending, confirmed, corrected, rejected

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary."""
        return asdict(self)


class PredictionHistoryService:
    """
    Service fuer die Verwaltung der Prediction-History.

    Speichert alle CNN-LSTM Predictions in einer JSON-Datei fuer:
    - Revalidierung nach Model-Training
    - Backtesting gegen echte Marktdaten
    - Statistiken und Performance-Analyse
    """

    def __init__(self, history_file: str = None):
        if history_file is None:
            history_file = os.path.join(DATA_DIR, "cnn_lstm_prediction_history.json")
        self._history_file = Path(history_file)
        self._history: list[PredictionHistoryEntry] = []
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._scan_interval = int(os.getenv("PREDICTION_SCAN_INTERVAL_SECONDS", "300"))  # 5 Minuten
        self._max_history_entries = int(os.getenv("MAX_PREDICTION_HISTORY", "5000"))
        self._max_history_age_days = int(os.getenv("MAX_PREDICTION_AGE_DAYS", "30"))

        # Default Symbole fuer Auto-Scan
        self._scan_symbols = os.getenv(
            "PREDICTION_SCAN_SYMBOLS",
            "BTCUSD,ETHUSD,EURUSD,GBPUSD,USDJPY"
        ).split(",")
        self._scan_timeframes = ["H1", "H4", "D1"]

        # Scan Progress Tracking
        self._scan_in_progress = False
        self._scan_total_symbols = 0
        self._scan_processed_symbols = 0
        self._scan_current_symbol = None
        self._scan_new_predictions = 0
        self._scan_last_completed = None
        self._scan_total_new_predictions = 0

        # Lade bestehende History
        self._load_history()

        logger.info(f"PredictionHistoryService initialized - {len(self._history)} entries loaded")

    def _load_history(self):
        """Lade History aus Datei."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._history = []
                    for entry in data:
                        self._history.append(PredictionHistoryEntry(**entry))
                logger.info(f"Loaded {len(self._history)} prediction history entries")
        except Exception as e:
            logger.error(f"Failed to load prediction history: {e}")
            self._history = []

    def _save_history(self):
        """Speichere History in Datei."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self._history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prediction history: {e}")

    def _cleanup_old_entries(self):
        """Entferne alte Eintraege."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_history_age_days)
        cutoff_str = cutoff.isoformat()

        original_count = len(self._history)
        self._history = [
            entry for entry in self._history
            if entry.timestamp > cutoff_str
        ]

        # Begrenze auf max Eintraege
        if len(self._history) > self._max_history_entries:
            self._history = self._history[-self._max_history_entries:]

        removed = original_count - len(self._history)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old prediction entries")

    def _generate_id(self, symbol: str, timeframe: str, timestamp: str) -> str:
        """Generiert eindeutige ID fuer einen Eintrag."""
        return f"{symbol}_{timeframe}_{timestamp}"

    def _is_duplicate(self, symbol: str, timeframe: str, timestamp: str) -> bool:
        """Prueft ob eine Prediction bereits existiert."""
        entry_id = self._generate_id(symbol, timeframe, timestamp)
        for entry in self._history:
            if entry.id == entry_id:
                return True
        return False

    async def add_prediction(
        self,
        symbol: str,
        timeframe: str,
        prediction_response: dict,
        ohlcv_snapshot: Optional[list] = None
    ) -> Optional[PredictionHistoryEntry]:
        """
        Fuegt eine neue Prediction zur History hinzu.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            prediction_response: Response vom Inference Service
            ohlcv_snapshot: OHLCV-Daten zum Zeitpunkt (fuer Reproduktion)

        Returns:
            PredictionHistoryEntry oder None wenn Duplikat
        """
        try:
            timestamp = prediction_response.get("timestamp", datetime.now(timezone.utc).isoformat())

            # Pruefe auf Duplikat
            if self._is_duplicate(symbol, timeframe, timestamp):
                logger.debug(f"Duplicate prediction for {symbol}/{timeframe} at {timestamp}")
                return None

            predictions = prediction_response.get("predictions", {})
            price = predictions.get("price", {})
            patterns = predictions.get("patterns", [])
            regime = predictions.get("regime", {})

            entry = PredictionHistoryEntry(
                id=self._generate_id(symbol, timeframe, timestamp),
                timestamp=timestamp,
                symbol=symbol.upper(),
                timeframe=timeframe.upper(),
                model_version=prediction_response.get("model_version", "unknown"),

                # Price
                price_current=price.get("current", 0.0),
                price_forecast_1h=price.get("forecast_1h"),
                price_forecast_4h=price.get("forecast_4h"),
                price_forecast_1d=price.get("forecast_1d"),
                price_forecast_1w=price.get("forecast_1w"),
                price_direction=price.get("direction", "neutral"),
                price_confidence=price.get("confidence", 0.0),
                price_change_percent_1h=price.get("change_percent_1h"),
                price_change_percent_1d=price.get("change_percent_1d"),

                # Patterns (Top 3)
                patterns=[
                    {
                        "type": p.get("type"),
                        "confidence": p.get("confidence"),
                        "direction": p.get("direction")
                    }
                    for p in patterns[:3]
                ],

                # Regime
                regime_current=regime.get("current", "sideways"),
                regime_probability=regime.get("probability", 0.0),
                regime_transition_probs=regime.get("transition_probs", {}),

                # Metadata
                inference_time_ms=prediction_response.get("inference_time_ms", 0.0),
                ohlcv_snapshot=ohlcv_snapshot,
            )

            self._history.append(entry)
            self._cleanup_old_entries()
            self._save_history()

            logger.debug(f"Added prediction for {symbol}/{timeframe}")

            # Trigger continuous optimization hook
            try:
                from .backtesting_service import backtesting_service
                await backtesting_service.on_prediction_completed(entry.to_dict())
            except Exception as hook_error:
                logger.debug(f"Continuous optimization hook error: {hook_error}")

            return entry

        except Exception as e:
            logger.error(f"Error adding prediction: {e}")
            return None

    async def scan_and_store_predictions(self) -> int:
        """
        Fuehrt Predictions fuer alle konfigurierten Symbole durch und speichert sie.

        Returns:
            Anzahl der neuen Predictions
        """
        try:
            import httpx

            # Initialize progress tracking
            self._scan_in_progress = True
            self._scan_total_symbols = len(self._scan_symbols)
            self._scan_processed_symbols = 0
            self._scan_current_symbol = None
            self._scan_new_predictions = 0

            new_predictions = 0
            cnn_lstm_url = os.getenv("CNN_LSTM_SERVICE_URL", "http://trading-cnn-lstm:3007")
            data_service_url = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")

            async with httpx.AsyncClient(timeout=60.0) as client:
                for symbol in self._scan_symbols:
                    # Update progress
                    self._scan_current_symbol = symbol

                    for timeframe in self._scan_timeframes:
                        try:
                            # Hole Prediction
                            response = await client.get(
                                f"{cnn_lstm_url}/api/v1/predict/{symbol}",
                                params={"timeframe": timeframe}
                            )

                            if response.status_code != 200:
                                continue

                            prediction_data = response.json()

                            # Hole OHLCV-Snapshot fuer Kontext
                            ohlcv_snapshot = None
                            try:
                                ohlcv_response = await client.get(
                                    f"{data_service_url}/api/v1/db/ohlcv/{symbol}",
                                    params={"timeframe": timeframe, "limit": 50}
                                )
                                if ohlcv_response.status_code == 200:
                                    ohlcv_data = ohlcv_response.json()
                                    ohlcv_snapshot = ohlcv_data.get("data", [])[-20:]  # Letzte 20 Kerzen
                            except Exception:
                                pass

                            # Speichere Prediction
                            entry = await self.add_prediction(
                                symbol=symbol,
                                timeframe=timeframe,
                                prediction_response=prediction_data,
                                ohlcv_snapshot=ohlcv_snapshot
                            )

                            if entry:
                                new_predictions += 1
                                self._scan_new_predictions = new_predictions

                        except Exception as e:
                            logger.warning(f"Error scanning {symbol}/{timeframe}: {e}")
                            continue

                    # Update processed count after each symbol
                    self._scan_processed_symbols += 1

            # Mark scan as completed
            self._scan_in_progress = False
            self._scan_last_completed = datetime.now(timezone.utc).isoformat()
            self._scan_total_new_predictions = new_predictions
            self._scan_current_symbol = None

            logger.info(f"Prediction scan complete: {new_predictions} new predictions")
            return new_predictions

        except Exception as e:
            # Reset progress on error
            self._scan_in_progress = False
            self._scan_current_symbol = None
            logger.error(f"Error in prediction scan: {e}")
            return 0

    async def _scan_loop(self):
        """Hintergrund-Scan-Loop."""
        while self._running:
            try:
                await self.scan_and_store_predictions()
            except Exception as e:
                logger.error(f"Error in prediction scan loop: {e}")

            await asyncio.sleep(self._scan_interval)

    async def start(self):
        """Starte den periodischen Scan."""
        if self._running:
            return

        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info(f"Prediction scan started (interval: {self._scan_interval}s)")

    async def stop(self):
        """Stoppe den periodischen Scan."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Prediction scan stopped")

    def is_running(self) -> bool:
        """Check if scan is running."""
        return self._running

    def get_history(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        feedback_status: Optional[str] = None,
        price_direction: Optional[str] = None,
        regime: Optional[str] = None,
        min_confidence: float = 0.0,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Hole Prediction-History mit optionalen Filtern.

        Args:
            symbol: Filter nach Symbol
            timeframe: Filter nach Timeframe
            feedback_status: Filter nach Feedback-Status
            price_direction: Filter nach Preis-Richtung
            regime: Filter nach Regime
            min_confidence: Minimale Konfidenz
            date_from: Ab Datum (ISO format)
            date_to: Bis Datum (ISO format)
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Liste der Prediction-Eintraege
        """
        results = []

        for entry in reversed(self._history):  # Neueste zuerst
            # Filter anwenden
            if symbol and entry.symbol != symbol.upper():
                continue
            if timeframe and entry.timeframe != timeframe.upper():
                continue
            if feedback_status and entry.feedback_status != feedback_status.lower():
                continue
            if price_direction and entry.price_direction != price_direction.lower():
                continue
            if regime and entry.regime_current != regime.lower():
                continue
            if entry.price_confidence < min_confidence:
                continue

            # Datum-Filter
            if date_from:
                if entry.timestamp < date_from:
                    continue
            if date_to:
                if entry.timestamp > date_to:
                    continue

            results.append(entry.to_dict())

            if len(results) >= limit:
                break

        return results

    def get_by_id(self, prediction_id: str) -> Optional[dict]:
        """Hole einzelne Prediction nach ID."""
        for entry in self._history:
            if entry.id == prediction_id:
                return entry.to_dict()
        return None

    def get_statistics(self) -> dict:
        """Hole Statistiken ueber die Prediction-History."""
        if not self._history:
            return {
                "total_predictions": 0,
                "symbols_count": 0,
                "by_direction": {},
                "by_regime": {},
                "by_timeframe": {},
                "by_feedback_status": {},
                "avg_confidence": 0.0,
                "scan_running": self._running,
                "scan_interval_seconds": self._scan_interval,
            }

        by_direction = {"bullish": 0, "bearish": 0, "neutral": 0}
        by_regime = {"bull_trend": 0, "bear_trend": 0, "sideways": 0, "high_volatility": 0}
        by_timeframe = {}
        by_feedback = {"pending": 0, "confirmed": 0, "corrected": 0, "rejected": 0}
        symbols = set()
        total_confidence = 0.0

        for entry in self._history:
            symbols.add(entry.symbol)
            by_direction[entry.price_direction] = by_direction.get(entry.price_direction, 0) + 1
            by_regime[entry.regime_current] = by_regime.get(entry.regime_current, 0) + 1
            by_timeframe[entry.timeframe] = by_timeframe.get(entry.timeframe, 0) + 1
            by_feedback[entry.feedback_status] = by_feedback.get(entry.feedback_status, 0) + 1
            total_confidence += entry.price_confidence

        return {
            "total_predictions": len(self._history),
            "symbols_count": len(symbols),
            "symbols": list(symbols),
            "by_direction": by_direction,
            "by_regime": by_regime,
            "by_timeframe": by_timeframe,
            "by_feedback_status": by_feedback,
            "avg_confidence": round(total_confidence / len(self._history), 3) if self._history else 0.0,
            "last_prediction": self._history[-1].timestamp if self._history else None,
            "scan_running": self._running,
            "scan_interval_seconds": self._scan_interval,
        }

    def get_scan_progress(self) -> dict:
        """
        Hole detaillierten Scan-Fortschritt.

        Returns:
            Dict mit scan_running, total_symbols, processed_symbols,
            current_symbol, new_predictions, last_scan_completed, total_new_predictions
        """
        return {
            "scan_running": self._scan_in_progress,
            "total_symbols": self._scan_total_symbols,
            "processed_symbols": self._scan_processed_symbols,
            "current_symbol": self._scan_current_symbol,
            "new_predictions": self._scan_new_predictions,
            "last_scan_completed": self._scan_last_completed,
            "total_new_predictions": self._scan_total_new_predictions,
        }

    def update_feedback_status(self, prediction_id: str, status: str) -> bool:
        """
        Aktualisiert den Feedback-Status einer Prediction.

        Args:
            prediction_id: ID der Prediction
            status: Neuer Status (pending, confirmed, corrected, rejected)

        Returns:
            True wenn erfolgreich
        """
        for entry in self._history:
            if entry.id == prediction_id:
                entry.feedback_status = status
                self._save_history()
                return True
        return False

    def clear_history(self):
        """Loesche die gesamte History."""
        self._history = []
        self._save_history()
        logger.info("Prediction history cleared")

    def clear_memory(self):
        """Loesche alle In-Memory Daten fuer Factory Reset."""
        was_running = self._running
        if self._running:
            self._running = False
            if self._scan_task:
                self._scan_task.cancel()

        self._history = []

        logger.info(f"Prediction history memory cleared (was running: {was_running})")

        return {
            "cleared": True,
            "was_running": was_running,
        }


# Global singleton
prediction_history_service = PredictionHistoryService()
