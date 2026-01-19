"""
Prediction Feedback Service - Verwaltet Feedback und Revalidierung fuer CNN-LSTM Predictions.

Ermoeglicht:
- Nutzer-Feedback zu Predictions (Bestaetigung, Korrektur, Ablehnung)
- Revalidierung nach Model-Training
- Statistiken zur Model-Verbesserung
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

from loguru import logger


DATA_DIR = os.getenv("DATA_DIR", "/app/data")


# Vordefinierte Begruendungskategorien fuer Feedback
REASON_CATEGORIES = {
    # Price-bezogene Gruende
    "wrong_direction": {
        "label": "Falsche Richtung",
        "description": "Die vorhergesagte Richtung war falsch",
        "task": "price",
        "patterns": ["all"]
    },
    "magnitude_too_high": {
        "label": "Aenderung ueberschaetzt",
        "description": "Die vorhergesagte Preisaenderung war zu gross",
        "task": "price",
        "patterns": ["all"]
    },
    "magnitude_too_low": {
        "label": "Aenderung unterschaetzt",
        "description": "Die vorhergesagte Preisaenderung war zu klein",
        "task": "price",
        "patterns": ["all"]
    },
    "timing_off": {
        "label": "Timing falsch",
        "description": "Die Bewegung kam frueher/spaeter als vorhergesagt",
        "task": "price",
        "patterns": ["all"]
    },

    # Pattern-bezogene Gruende
    "false_positive_pattern": {
        "label": "Pattern nicht vorhanden",
        "description": "Das erkannte Pattern existiert nicht",
        "task": "patterns",
        "patterns": ["all"]
    },
    "missed_pattern": {
        "label": "Pattern verpasst",
        "description": "Ein wichtiges Pattern wurde nicht erkannt",
        "task": "patterns",
        "patterns": ["all"]
    },
    "wrong_pattern_type": {
        "label": "Falscher Pattern-Typ",
        "description": "Es ist ein anderes Pattern als erkannt",
        "task": "patterns",
        "patterns": ["all"]
    },

    # Regime-bezogene Gruende
    "wrong_regime": {
        "label": "Falsches Regime",
        "description": "Das erkannte Markt-Regime war falsch",
        "task": "regime",
        "patterns": ["all"]
    },
    "regime_transition_missed": {
        "label": "Regime-Wechsel verpasst",
        "description": "Ein Regime-Wechsel wurde nicht erkannt",
        "task": "regime",
        "patterns": ["all"]
    },
    "premature_regime_change": {
        "label": "Verfrühter Regime-Wechsel",
        "description": "Ein Regime-Wechsel wurde zu frueh erkannt",
        "task": "regime",
        "patterns": ["all"]
    },

    # Allgemeine Gruende
    "low_confidence_correct": {
        "label": "Trotz niedriger Konfidenz korrekt",
        "description": "Die Vorhersage war trotz niedriger Konfidenz richtig",
        "task": "all",
        "patterns": ["all"]
    },
    "high_confidence_wrong": {
        "label": "Trotz hoher Konfidenz falsch",
        "description": "Die Vorhersage war trotz hoher Konfidenz falsch",
        "task": "all",
        "patterns": ["all"]
    },
    "other": {
        "label": "Anderer Grund",
        "description": "Ein anderer, nicht aufgelisteter Grund",
        "task": "all",
        "patterns": ["all"]
    }
}


@dataclass
class PriceFeedback:
    """Feedback fuer Preis-Vorhersage."""
    actual_direction: Optional[str] = None  # bullish, bearish, neutral
    actual_change_percent: Optional[float] = None
    was_correct: bool = False
    reason: Optional[str] = None


@dataclass
class PatternFeedback:
    """Feedback fuer Pattern-Klassifikation."""
    confirmed_patterns: list = field(default_factory=list)  # Korrekt erkannte Patterns
    missed_patterns: list = field(default_factory=list)  # Verpasste Patterns
    false_patterns: list = field(default_factory=list)  # Falsch erkannte Patterns
    reason: Optional[str] = None


@dataclass
class RegimeFeedback:
    """Feedback fuer Regime-Vorhersage."""
    actual_regime: Optional[str] = None  # bull_trend, bear_trend, sideways, high_volatility
    was_correct: bool = False
    reason: Optional[str] = None


@dataclass
class PredictionFeedback:
    """Vollstaendiges Feedback fuer eine CNN-LSTM Prediction."""
    id: str  # Feedback-ID
    prediction_id: str  # Referenz auf die Prediction
    symbol: str
    timeframe: str
    prediction_timestamp: str

    # Feedback-Typ
    feedback_type: str  # confirmed, corrected, rejected

    # Task-spezifisches Feedback (optional)
    price_feedback: Optional[dict] = None
    pattern_feedback: Optional[dict] = None
    regime_feedback: Optional[dict] = None

    # Begruendung
    reason_category: Optional[str] = None
    reason_text: Optional[str] = None

    # Metadata
    feedback_timestamp: str = ""
    ohlcv_data: Optional[list] = None  # OHLCV-Daten zum Zeitpunkt der Prediction

    # Revalidierung
    revalidated: bool = False
    revalidation_result: Optional[str] = None  # correct, still_wrong, now_correct
    revalidation_timestamp: Optional[str] = None
    revalidation_notes: Optional[str] = None
    revalidation_history: list = field(default_factory=list)

    # Update History
    update_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary."""
        return asdict(self)


class PredictionFeedbackService:
    """
    Service fuer die Verwaltung von Prediction-Feedback.

    Analog zum history_router.py des Candlestick Service, aber angepasst
    fuer die 3-Task Architektur (Price, Patterns, Regime).
    """

    def __init__(self, feedback_file: str = None):
        if feedback_file is None:
            feedback_file = os.path.join(DATA_DIR, "cnn_lstm_prediction_feedback.json")
        self._feedback_file = Path(feedback_file)
        self._feedback: list[PredictionFeedback] = []

        self._load_feedback()
        logger.info(f"PredictionFeedbackService initialized - {len(self._feedback)} entries loaded")

    def _load_feedback(self):
        """Lade Feedback aus Datei."""
        try:
            if self._feedback_file.exists():
                with open(self._feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._feedback = []
                    for entry in data:
                        self._feedback.append(PredictionFeedback(**entry))
                logger.info(f"Loaded {len(self._feedback)} prediction feedback entries")
        except Exception as e:
            logger.error(f"Failed to load prediction feedback: {e}")
            self._feedback = []

    def _save_feedback(self):
        """Speichere Feedback in Datei."""
        try:
            self._feedback_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._feedback_file, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self._feedback], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save prediction feedback: {e}")

    def submit_feedback(
        self,
        prediction_id: str,
        feedback_type: str,  # confirmed, corrected, rejected
        symbol: str,
        timeframe: str,
        prediction_timestamp: str,
        price_feedback: Optional[dict] = None,
        pattern_feedback: Optional[dict] = None,
        regime_feedback: Optional[dict] = None,
        reason_category: Optional[str] = None,
        reason_text: Optional[str] = None,
        ohlcv_data: Optional[list] = None,
    ) -> dict:
        """
        Speichert Nutzer-Feedback zu einer Prediction.

        Args:
            prediction_id: ID der Prediction
            feedback_type: confirmed, corrected, rejected
            symbol: Symbol
            timeframe: Timeframe
            prediction_timestamp: Zeitpunkt der Prediction
            price_feedback: Feedback zur Preis-Vorhersage
            pattern_feedback: Feedback zur Pattern-Klassifikation
            regime_feedback: Feedback zur Regime-Vorhersage
            reason_category: Begruendungskategorie
            reason_text: Freitext-Begruendung
            ohlcv_data: OHLCV-Daten fuer Reproduktion

        Returns:
            Status-Dict
        """
        try:
            feedback_id = f"fb_{prediction_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            # Pruefe ob bereits Feedback existiert - update statt neu anlegen
            existing_idx = None
            for i, fb in enumerate(self._feedback):
                if fb.prediction_id == prediction_id:
                    existing_idx = i
                    break

            entry = PredictionFeedback(
                id=feedback_id,
                prediction_id=prediction_id,
                symbol=symbol.upper(),
                timeframe=timeframe.upper(),
                prediction_timestamp=prediction_timestamp,
                feedback_type=feedback_type,
                price_feedback=price_feedback,
                pattern_feedback=pattern_feedback,
                regime_feedback=regime_feedback,
                reason_category=reason_category,
                reason_text=reason_text,
                feedback_timestamp=datetime.utcnow().isoformat(),
                ohlcv_data=ohlcv_data,
            )

            if existing_idx is not None:
                # Update existierenden Eintrag
                old_entry = self._feedback[existing_idx]
                entry.id = old_entry.id  # Behalte alte ID
                entry.update_history = old_entry.update_history or []
                entry.update_history.append({
                    "old_feedback_type": old_entry.feedback_type,
                    "new_feedback_type": feedback_type,
                    "updated_at": datetime.utcnow().isoformat(),
                })
                entry.revalidation_history = old_entry.revalidation_history
                self._feedback[existing_idx] = entry
                logger.info(f"Updated feedback for {prediction_id}")
            else:
                self._feedback.append(entry)
                logger.info(f"New feedback saved for {prediction_id}")

            self._save_feedback()

            return {
                "status": "saved",
                "feedback_id": entry.id,
                "prediction_id": prediction_id,
                "total_feedback_count": len(self._feedback)
            }

        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return {"status": "error", "message": str(e)}

    def get_feedback(
        self,
        prediction_id: Optional[str] = None,
        feedback_type: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Hole Feedback mit optionalen Filtern."""
        results = []

        for fb in reversed(self._feedback):
            if prediction_id and fb.prediction_id != prediction_id:
                continue
            if feedback_type and fb.feedback_type != feedback_type:
                continue
            if symbol and fb.symbol != symbol.upper():
                continue

            results.append(fb.to_dict())

            if len(results) >= limit:
                break

        return results

    def get_pending_revalidation(self) -> list[dict]:
        """
        Hole Predictions die nach Training revalidiert werden muessen.

        Returns Feedback-Eintraege die:
        - corrected oder rejected sind
        - noch nicht revalidiert wurden
        """
        pending = []

        for fb in self._feedback:
            if fb.feedback_type in ('corrected', 'rejected') and not fb.revalidated:
                pending.append(fb.to_dict())

        return sorted(pending, key=lambda x: x.get('feedback_timestamp', ''), reverse=True)

    def mark_revalidated(
        self,
        feedback_id: str,
        validation_result: str,  # correct, still_wrong, now_correct
        notes: Optional[str] = None,
        corrected_data: Optional[dict] = None
    ) -> dict:
        """
        Markiert ein Feedback als revalidiert.

        Args:
            feedback_id: Feedback-ID
            validation_result: correct, still_wrong, now_correct
            notes: Optionale Notizen
            corrected_data: Neue Korrekturdaten wenn still_wrong

        Returns:
            Status-Dict
        """
        for fb in self._feedback:
            if fb.id == feedback_id:
                fb.revalidated = True
                fb.revalidation_result = validation_result
                fb.revalidation_timestamp = datetime.utcnow().isoformat()
                fb.revalidation_notes = notes

                # Speichere in History
                if fb.revalidation_history is None:
                    fb.revalidation_history = []
                fb.revalidation_history.append({
                    "result": validation_result,
                    "timestamp": fb.revalidation_timestamp,
                    "notes": notes
                })

                # Update Korrektur wenn still_wrong
                if validation_result == "still_wrong" and corrected_data:
                    if corrected_data.get("price_feedback"):
                        fb.price_feedback = corrected_data["price_feedback"]
                    if corrected_data.get("pattern_feedback"):
                        fb.pattern_feedback = corrected_data["pattern_feedback"]
                    if corrected_data.get("regime_feedback"):
                        fb.regime_feedback = corrected_data["regime_feedback"]

                self._save_feedback()
                logger.info(f"Revalidation marked: {feedback_id} -> {validation_result}")

                return {
                    "status": "success",
                    "feedback_id": feedback_id,
                    "validation_result": validation_result
                }

        return {"status": "not_found", "feedback_id": feedback_id}

    def reset_revalidation(self) -> dict:
        """
        Setzt alle Revalidierungs-Status zurueck.

        Wird nach neuem Training aufgerufen um alle korrigierten/abgelehnten
        Predictions erneut zur Pruefung freizugeben.
        """
        reset_count = 0

        for fb in self._feedback:
            if fb.revalidated:
                fb.revalidated = False
                # Speichere vorherige Revalidierung in History
                if fb.revalidation_result:
                    if fb.revalidation_history is None:
                        fb.revalidation_history = []
                    fb.revalidation_history.append({
                        "result": fb.revalidation_result,
                        "timestamp": fb.revalidation_timestamp,
                        "notes": fb.revalidation_notes,
                        "reset_at": datetime.utcnow().isoformat()
                    })
                fb.revalidation_result = None
                fb.revalidation_timestamp = None
                fb.revalidation_notes = None
                reset_count += 1

        self._save_feedback()
        logger.info(f"Reset revalidation status for {reset_count} entries")

        return {
            "status": "success",
            "reset_count": reset_count
        }

    def get_statistics(self) -> dict:
        """Hole Feedback-Statistiken."""
        if not self._feedback:
            return {
                "total_feedback": 0,
                "by_type": {},
                "by_task": {},
                "pending_revalidation": 0,
                "revalidated": 0,
                "improvement_rate": 0.0
            }

        by_type = {"confirmed": 0, "corrected": 0, "rejected": 0}
        by_task = {"price": 0, "patterns": 0, "regime": 0}
        pending = 0
        revalidated = 0
        improved = 0

        for fb in self._feedback:
            by_type[fb.feedback_type] = by_type.get(fb.feedback_type, 0) + 1

            # Zaehle nach Task
            if fb.price_feedback:
                by_task["price"] += 1
            if fb.pattern_feedback:
                by_task["patterns"] += 1
            if fb.regime_feedback:
                by_task["regime"] += 1

            # Revalidierungs-Status
            if fb.feedback_type in ('corrected', 'rejected'):
                if fb.revalidated:
                    revalidated += 1
                    if fb.revalidation_result in ('correct', 'now_correct'):
                        improved += 1
                else:
                    pending += 1

        improvement_rate = (improved / revalidated * 100) if revalidated > 0 else 0.0

        return {
            "total_feedback": len(self._feedback),
            "by_type": by_type,
            "by_task": by_task,
            "pending_revalidation": pending,
            "revalidated": revalidated,
            "improved": improved,
            "improvement_rate": round(improvement_rate, 1),
            "correction_rate": round(
                (by_type.get("corrected", 0) + by_type.get("rejected", 0)) / len(self._feedback) * 100, 1
            ) if self._feedback else 0.0
        }

    def get_revalidation_statistics(self) -> dict:
        """Hole detaillierte Revalidierungs-Statistiken."""
        if not self._feedback:
            return {
                "total_corrections": 0,
                "pending": 0,
                "revalidated": 0,
                "results": {},
                "improvement_rate": 0.0,
                "by_task": {}
            }

        total_corrections = 0
        pending = 0
        revalidated = 0
        results = {"correct": 0, "still_wrong": 0, "now_correct": 0}
        by_task = {
            "price": {"total": 0, "improved": 0},
            "patterns": {"total": 0, "improved": 0},
            "regime": {"total": 0, "improved": 0}
        }

        for fb in self._feedback:
            if fb.feedback_type in ('corrected', 'rejected'):
                total_corrections += 1

                if fb.revalidated:
                    revalidated += 1
                    result = fb.revalidation_result or "unknown"
                    results[result] = results.get(result, 0) + 1

                    # Nach Task
                    is_improved = result in ('correct', 'now_correct')
                    if fb.price_feedback:
                        by_task["price"]["total"] += 1
                        if is_improved:
                            by_task["price"]["improved"] += 1
                    if fb.pattern_feedback:
                        by_task["patterns"]["total"] += 1
                        if is_improved:
                            by_task["patterns"]["improved"] += 1
                    if fb.regime_feedback:
                        by_task["regime"]["total"] += 1
                        if is_improved:
                            by_task["regime"]["improved"] += 1
                else:
                    pending += 1

        improved = results.get("correct", 0) + results.get("now_correct", 0)
        improvement_rate = (improved / revalidated * 100) if revalidated > 0 else 0.0

        return {
            "total_corrections": total_corrections,
            "pending": pending,
            "revalidated": revalidated,
            "results": results,
            "improved": improved,
            "improvement_rate": round(improvement_rate, 1),
            "by_task": by_task,
            "message": f"Verbesserungsrate: {improvement_rate:.1f}% ({improved}/{revalidated})"
        }

    def get_reason_categories(self, task: Optional[str] = None) -> dict:
        """
        Hole verfuegbare Begruendungskategorien.

        Args:
            task: Optional Filter fuer Task (price, patterns, regime)
        """
        if task:
            filtered = {}
            for key, info in REASON_CATEGORIES.items():
                if info.get("task") in (task, "all"):
                    filtered[key] = {
                        "label": info["label"],
                        "description": info["description"]
                    }
            return {"task": task, "categories": filtered, "count": len(filtered)}

        return {
            "categories": {
                k: {"label": v["label"], "description": v["description"], "task": v["task"]}
                for k, v in REASON_CATEGORIES.items()
            },
            "count": len(REASON_CATEGORIES)
        }

    def delete_feedback(self, feedback_id: str) -> dict:
        """Loesche ein Feedback."""
        for i, fb in enumerate(self._feedback):
            if fb.id == feedback_id:
                del self._feedback[i]
                self._save_feedback()
                logger.info(f"Deleted feedback: {feedback_id}")
                return {"status": "deleted", "feedback_id": feedback_id}

        return {"status": "not_found", "feedback_id": feedback_id}

    def clear_memory(self):
        """Loesche alle In-Memory Daten."""
        count = len(self._feedback)
        self._feedback = []
        logger.info(f"Prediction feedback memory cleared: {count} entries")
        return {"cleared": True, "entries_cleared": count}


# Global singleton
prediction_feedback_service = PredictionFeedbackService()
