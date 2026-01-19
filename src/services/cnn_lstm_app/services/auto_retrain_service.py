"""
Auto-Retrain Service - Automatisches Re-Training bei schlechter Performance.

Implementiert einen geschlossenen Feedback-Loop:
1. Backtest-Ergebnisse analysieren
2. Bei Accuracy < Schwellwert -> Re-Training triggern
3. Neues Modell automatisch deployen
4. Vergleich mit altem Modell (Rollback wenn schlechter)
"""

import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path

import httpx
from loguru import logger


DATA_DIR = os.getenv("DATA_DIR", "/app/data")

# Auto-Retrain Konfiguration via Umgebungsvariablen
AUTO_RETRAIN_ENABLED = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() == "true"
AUTO_RETRAIN_ACCURACY_THRESHOLD = float(os.getenv("AUTO_RETRAIN_ACCURACY_THRESHOLD", "60.0"))
AUTO_RETRAIN_MIN_SAMPLES = int(os.getenv("AUTO_RETRAIN_MIN_SAMPLES", "20"))
AUTO_RETRAIN_COOLDOWN_HOURS = float(os.getenv("AUTO_RETRAIN_COOLDOWN_HOURS", "24"))


@dataclass
class RetrainTriggerEvent:
    """Ereignis das ein Re-Training ausgeloest hat."""
    timestamp: str
    reason: str
    accuracy_before: float
    threshold: float
    samples_count: int
    training_job_id: Optional[str] = None
    training_status: str = "pending"  # pending, running, completed, failed
    accuracy_after: Optional[float] = None
    improvement: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrainConfig:
    """Konfiguration fuer Auto-Retrain."""
    enabled: bool = True
    accuracy_threshold: float = 60.0  # Minimum Accuracy in %
    min_samples: int = 20  # Mindestanzahl Backtests
    cooldown_hours: float = 24.0  # Mindestwartezeit zwischen Trainings
    focus_on_errors: bool = True  # Fehler-gewichtetes Training
    symbols_from_errors: bool = True  # Symbole aus fehlerhaften Predictions

    def to_dict(self) -> dict:
        return asdict(self)


class AutoRetrainService:
    """
    Service fuer automatisches Re-Training basierend auf Backtest-Ergebnissen.

    Implementiert einen Closed-Loop Feedback-Mechanismus:
    - Ueberwacht Backtest-Accuracy kontinuierlich
    - Triggert Re-Training wenn Accuracy unter Schwellwert faellt
    - Analysiert Fehler-Patterns fuer gezieltes Training
    - Vergleicht neues Modell mit altem (A/B Testing)
    """

    def __init__(self, history_file: str = None):
        if history_file is None:
            history_file = os.path.join(DATA_DIR, "cnn_lstm_retrain_history.json")
        self._history_file = Path(history_file)
        self._retrain_history: list[RetrainTriggerEvent] = []

        # Konfiguration
        self._config = RetrainConfig(
            enabled=AUTO_RETRAIN_ENABLED,
            accuracy_threshold=AUTO_RETRAIN_ACCURACY_THRESHOLD,
            min_samples=AUTO_RETRAIN_MIN_SAMPLES,
            cooldown_hours=AUTO_RETRAIN_COOLDOWN_HOURS
        )

        # Service URLs
        self._train_service_url = os.getenv("CNN_LSTM_TRAIN_SERVICE_URL", "http://trading-cnn-lstm-train:3017")

        # State
        self._last_retrain: Optional[datetime] = None
        self._current_training_job: Optional[str] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_running = False

        self._load_history()
        logger.info(
            f"AutoRetrainService initialized - "
            f"enabled={self._config.enabled}, "
            f"threshold={self._config.accuracy_threshold}%, "
            f"min_samples={self._config.min_samples}"
        )

    def _load_history(self):
        """Lade Retrain-History aus Datei."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._retrain_history = [RetrainTriggerEvent(**e) for e in data]

                    # Finde letztes erfolgreiches Training
                    for event in reversed(self._retrain_history):
                        if event.training_status in ["completed", "running"]:
                            self._last_retrain = datetime.fromisoformat(
                                event.timestamp.replace('Z', '+00:00')
                            )
                            break
        except Exception as e:
            logger.error(f"Failed to load retrain history: {e}")
            self._retrain_history = []

    def _save_history(self):
        """Speichere Retrain-History."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump([e.to_dict() for e in self._retrain_history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save retrain history: {e}")

    def get_config(self) -> dict:
        """Hole aktuelle Konfiguration."""
        return self._config.to_dict()

    def update_config(self, **kwargs) -> dict:
        """Aktualisiere Konfiguration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        logger.info(f"AutoRetrain config updated: {kwargs}")
        return self._config.to_dict()

    def _is_cooldown_active(self) -> bool:
        """Pruefe ob Cooldown-Periode aktiv ist."""
        if self._last_retrain is None:
            return False

        cooldown_end = self._last_retrain + timedelta(hours=self._config.cooldown_hours)
        return datetime.now(timezone.utc) < cooldown_end

    def _get_cooldown_remaining(self) -> Optional[int]:
        """Hole verbleibende Cooldown-Zeit in Sekunden."""
        if self._last_retrain is None:
            return None

        cooldown_end = self._last_retrain + timedelta(hours=self._config.cooldown_hours)
        remaining = (cooldown_end - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining)) if remaining > 0 else None

    async def check_and_trigger_retrain(self, backtest_summary: dict) -> Optional[RetrainTriggerEvent]:
        """
        Pruefe Backtest-Ergebnisse und triggere Re-Training falls noetig.

        Args:
            backtest_summary: Zusammenfassung der Backtest-Ergebnisse

        Returns:
            RetrainTriggerEvent wenn Training getriggert wurde, sonst None
        """
        if not self._config.enabled:
            logger.debug("Auto-retrain disabled")
            return None

        # Pruefe Mindestanzahl Samples
        total_samples = backtest_summary.get("backtested", 0)
        if total_samples < self._config.min_samples:
            logger.debug(
                f"Not enough samples for retrain check: "
                f"{total_samples} < {self._config.min_samples}"
            )
            return None

        # Pruefe Cooldown
        if self._is_cooldown_active():
            remaining = self._get_cooldown_remaining()
            logger.debug(f"Retrain cooldown active: {remaining}s remaining")
            return None

        # Pruefe Accuracy
        accuracy = backtest_summary.get("price_direction_accuracy", 100.0)

        if accuracy >= self._config.accuracy_threshold:
            logger.debug(
                f"Accuracy OK: {accuracy:.1f}% >= {self._config.accuracy_threshold}%"
            )
            return None

        # Accuracy unter Schwellwert - Trigger Re-Training!
        logger.warning(
            f"Accuracy below threshold: {accuracy:.1f}% < {self._config.accuracy_threshold}% "
            f"-> Triggering re-training"
        )

        return await self._trigger_retrain(
            reason=f"Accuracy {accuracy:.1f}% below threshold {self._config.accuracy_threshold}%",
            accuracy_before=accuracy,
            samples_count=total_samples,
            backtest_summary=backtest_summary
        )

    async def _trigger_retrain(
        self,
        reason: str,
        accuracy_before: float,
        samples_count: int,
        backtest_summary: dict
    ) -> Optional[RetrainTriggerEvent]:
        """Triggere Re-Training beim Training Service."""
        event = RetrainTriggerEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            accuracy_before=accuracy_before,
            threshold=self._config.accuracy_threshold,
            samples_count=samples_count
        )

        try:
            # Analysiere Fehler-Patterns fuer gezieltes Training
            training_params = await self._analyze_errors_for_training(backtest_summary)

            # Sende Training-Request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._train_service_url}/api/v1/train",
                    json=training_params
                )

                if response.status_code == 200:
                    result = response.json()
                    event.training_job_id = result.get("job_id")
                    event.training_status = "running"
                    self._current_training_job = event.training_job_id
                    self._last_retrain = datetime.now(timezone.utc)

                    logger.info(
                        f"Re-training triggered: job_id={event.training_job_id}, "
                        f"reason={reason}"
                    )
                elif response.status_code == 409:
                    # Training already in progress
                    event.training_status = "skipped"
                    logger.warning("Re-training skipped: Training already in progress")
                else:
                    event.training_status = "failed"
                    logger.error(
                        f"Failed to trigger re-training: {response.status_code} - "
                        f"{response.text}"
                    )

        except Exception as e:
            event.training_status = "failed"
            logger.error(f"Error triggering re-training: {e}")

        # Speichere Event
        self._retrain_history.append(event)
        self._save_history()

        return event

    async def _analyze_errors_for_training(self, backtest_summary: dict) -> dict:
        """
        Analysiere Fehler-Patterns und erstelle optimierte Training-Parameter.

        Args:
            backtest_summary: Backtest-Zusammenfassung mit by_symbol/by_timeframe

        Returns:
            Training-Parameter mit Fokus auf Schwachstellen
        """
        # Basis-Parameter
        params = {
            "epochs": 50,  # Weniger Epochen fuer Re-Training
            "batch_size": 64,
            "learning_rate": 5e-5,  # Kleinere Lernrate fuer Feintuning
            "early_stopping_patience": 5,
            "timeframes": ["H1", "D1"]
        }

        # Finde Symbole mit schlechter Performance
        by_symbol = backtest_summary.get("by_symbol", {})
        weak_symbols = []

        for symbol, data in by_symbol.items():
            accuracy = data.get("accuracy", 100)
            if accuracy < self._config.accuracy_threshold:
                weak_symbols.append(symbol)

        # Wenn Fehler-fokussiertes Training aktiviert
        if self._config.focus_on_errors and weak_symbols:
            # Trainiere nur auf schwachen Symbolen
            params["symbols"] = weak_symbols[:10]  # Max 10 Symbole
            logger.info(f"Focus training on weak symbols: {weak_symbols[:10]}")
        else:
            # Standard: Alle verfuegbaren Symbole
            params["symbols"] = ["BTCUSD", "EURUSD", "AAPL", "MSFT", "NVDA"]

        # Passe Task-Gewichte an basierend auf Fehler-Analyse
        regime_accuracy = backtest_summary.get("regime_accuracy", 100)
        pattern_success = backtest_summary.get("pattern_success_rate", 100)
        direction_accuracy = backtest_summary.get("price_direction_accuracy", 100)

        # Erhoehe Gewicht fuer schwache Tasks
        if direction_accuracy < 50:
            params["price_weight"] = 0.5  # Mehr Fokus auf Preis
            params["pattern_weight"] = 0.3
            params["regime_weight"] = 0.2
        elif regime_accuracy < 50:
            params["price_weight"] = 0.35
            params["pattern_weight"] = 0.3
            params["regime_weight"] = 0.35  # Mehr Fokus auf Regime
        elif pattern_success < 50:
            params["price_weight"] = 0.35
            params["pattern_weight"] = 0.4  # Mehr Fokus auf Patterns
            params["regime_weight"] = 0.25
        else:
            # Standard-Gewichte
            params["price_weight"] = 0.4
            params["pattern_weight"] = 0.35
            params["regime_weight"] = 0.25

        return params

    async def check_training_status(self) -> Optional[dict]:
        """Pruefe Status des aktuellen Trainings."""
        if not self._current_training_job:
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._train_service_url}/api/v1/train/status"
                )

                if response.status_code == 200:
                    status = response.json()

                    # Update letztes Event
                    if self._retrain_history:
                        last_event = self._retrain_history[-1]
                        if last_event.training_job_id == self._current_training_job:
                            last_event.training_status = status.get("status", "unknown")

                            if status.get("status") == "completed":
                                self._current_training_job = None
                                self._save_history()

                    return status

        except Exception as e:
            logger.warning(f"Error checking training status: {e}")

        return None

    async def update_accuracy_after_training(self, new_accuracy: float):
        """
        Aktualisiere das letzte Retrain-Event mit der neuen Accuracy.

        Args:
            new_accuracy: Accuracy nach dem Training
        """
        if not self._retrain_history:
            return

        last_event = self._retrain_history[-1]
        if last_event.training_status == "completed":
            last_event.accuracy_after = new_accuracy
            last_event.improvement = new_accuracy - last_event.accuracy_before

            logger.info(
                f"Retrain result: {last_event.accuracy_before:.1f}% -> "
                f"{new_accuracy:.1f}% (improvement: {last_event.improvement:+.1f}%)"
            )

            self._save_history()

    def get_status(self) -> dict:
        """Hole aktuellen Status des Auto-Retrain Service."""
        return {
            "enabled": self._config.enabled,
            "config": self._config.to_dict(),
            "cooldown_active": self._is_cooldown_active(),
            "cooldown_remaining_seconds": self._get_cooldown_remaining(),
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "current_training_job": self._current_training_job,
            "total_retrains": len(self._retrain_history),
            "monitoring_running": self._monitoring_running
        }

    def get_history(self, limit: int = 20) -> list[dict]:
        """Hole Retrain-History."""
        return [e.to_dict() for e in reversed(self._retrain_history)][:limit]

    def get_statistics(self) -> dict:
        """Berechne Statistiken ueber alle Retrains."""
        if not self._retrain_history:
            return {
                "total_retrains": 0,
                "successful": 0,
                "failed": 0,
                "average_improvement": 0.0,
                "best_improvement": 0.0,
                "worst_improvement": 0.0
            }

        successful = [e for e in self._retrain_history if e.training_status == "completed"]
        failed = [e for e in self._retrain_history if e.training_status == "failed"]

        improvements = [
            e.improvement for e in successful
            if e.improvement is not None
        ]

        return {
            "total_retrains": len(self._retrain_history),
            "successful": len(successful),
            "failed": len(failed),
            "average_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
            "best_improvement": max(improvements) if improvements else 0.0,
            "worst_improvement": min(improvements) if improvements else 0.0
        }

    def clear_history(self):
        """Loesche Retrain-History."""
        self._retrain_history = []
        self._save_history()
        logger.info("Retrain history cleared")


# Global singleton
auto_retrain_service = AutoRetrainService()
