"""Alert History Service für persistente Speicherung der Alert-Historie.

Speichert die letzten 100 Alerts in einer JSON-Datei.
Das Volume /app/data ist persistent gemountet.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from threading import Lock

from loguru import logger


class AlertHistoryService:
    """Verwaltet die persistente Speicherung der Alert-Historie."""

    def __init__(self, data_dir: str = "/app/data", max_entries: int = 100):
        """
        Initialisiert den Alert History Service.

        Args:
            data_dir: Verzeichnis für persistente Daten
            max_entries: Maximale Anzahl gespeicherter Alerts (Standard: 100)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / "alert_history.json"
        self._lock = Lock()
        self._history: List[Dict] = []
        self._max_entries = max_entries

        # Lade existierende Historie
        self._load_history()
        logger.info(f"Alert History Service initialisiert. {len(self._history)} Einträge geladen. Max: {self._max_entries}")

    def _load_history(self):
        """Lädt die Historie aus der Datei."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._history = data.get("alerts", [])
                    # Begrenze auf max_entries
                    if len(self._history) > self._max_entries:
                        self._history = self._history[-self._max_entries:]
                    logger.debug(f"Alert-Historie geladen: {len(self._history)} Einträge")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Alert-Historie: {e}")
            self._history = []

    def _save_history(self):
        """Speichert die Historie in die Datei."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "max_entries": self._max_entries,
                    "alerts": self._history
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Alert-Historie: {e}")

    def add_alert(self, alert: Dict) -> Dict:
        """
        Fügt einen neuen Alert zur Historie hinzu.

        Args:
            alert: Alert-Daten als Dictionary

        Returns:
            Der hinzugefügte Alert
        """
        with self._lock:
            self._history.append(alert)

            # Begrenze auf max_entries
            if len(self._history) > self._max_entries:
                self._history = self._history[-self._max_entries:]

            self._save_history()
            return alert

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Gibt die Alert-Historie zurück.

        Args:
            limit: Optionale Begrenzung der Anzahl (neueste zuerst)

        Returns:
            Liste der Alerts (neueste zuerst)
        """
        with self._lock:
            # Kopie der Liste in umgekehrter Reihenfolge (neueste zuerst)
            result = list(reversed(self._history))
            if limit:
                result = result[:limit]
            return result

    def clear_history(self):
        """Löscht die gesamte Alert-Historie."""
        with self._lock:
            self._history = []
            self._save_history()
            logger.info("Alert-Historie gelöscht")

    def get_statistics(self) -> Dict:
        """
        Gibt Statistiken über die Alert-Historie zurück.

        Returns:
            Dict mit Statistiken
        """
        with self._lock:
            total = len(self._history)

            if not self._history:
                return {
                    "total_alerts": 0,
                    "alerts_by_type": {},
                    "alerts_by_service": {},
                    "telegram_sent_count": 0,
                    "suppressed_count": 0,
                    "last_alert": None
                }

            # Zähle nach Typ
            by_type = {}
            by_service = {}
            telegram_sent = 0
            suppressed = 0

            for alert in self._history:
                # Nach Typ
                alert_type = alert.get("type", "UNKNOWN")
                by_type[alert_type] = by_type.get(alert_type, 0) + 1

                # Nach Service
                service = alert.get("service", "unknown")
                by_service[service] = by_service.get(service, 0) + 1

                # Telegram
                if alert.get("telegram_sent"):
                    telegram_sent += 1

                # Unterdrückt
                if alert.get("suppressed"):
                    suppressed += 1

            return {
                "total_alerts": total,
                "max_entries": self._max_entries,
                "alerts_by_type": by_type,
                "alerts_by_service": by_service,
                "telegram_sent_count": telegram_sent,
                "suppressed_count": suppressed,
                "last_alert": self._history[-1] if self._history else None
            }


# Singleton-Instanz
alert_history = AlertHistoryService()
