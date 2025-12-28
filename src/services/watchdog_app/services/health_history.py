"""Health History Service für persistente Speicherung der Service-Health-Historie.

Speichert Health-Checks der letzten 24 Stunden in einer JSON-Datei.
Das Volume /app/data ist persistent gemountet.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from threading import Lock

from loguru import logger


class HealthHistoryService:
    """Verwaltet die persistente Speicherung der Health-Historie."""

    def __init__(self, data_dir: str = "/app/data"):
        """
        Initialisiert den Health History Service.

        Args:
            data_dir: Verzeichnis für persistente Daten
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / "health_history.json"
        self._lock = Lock()
        self._history: List[Dict] = []
        self._max_age_hours = 24
        self._max_entries = 2880  # 24h * 60min / 0.5min = max 2880 bei 30s Intervall

        # Lade existierende Historie
        self._load_history()
        logger.info(f"Health History Service initialisiert. {len(self._history)} Einträge geladen.")

    def _load_history(self):
        """Lädt die Historie aus der Datei."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._history = data.get("history", [])
                    # Bereinige alte Einträge
                    self._cleanup_old_entries()
                    logger.debug(f"Historie geladen: {len(self._history)} Einträge")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Historie: {e}")
            self._history = []

    def _save_history(self):
        """Speichert die Historie in die Datei."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "history": self._history
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Historie: {e}")

    def _cleanup_old_entries(self):
        """Entfernt Einträge älter als max_age_hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_age_hours)
        cutoff_iso = cutoff.isoformat()

        self._history = [
            entry for entry in self._history
            if entry.get("timestamp", "") >= cutoff_iso
        ]

        # Begrenze auf max_entries
        if len(self._history) > self._max_entries:
            self._history = self._history[-self._max_entries:]

    def add_check_result(self, services_status: Dict) -> Dict:
        """
        Fügt ein neues Health-Check-Ergebnis hinzu.

        Args:
            services_status: Dict mit Service-Namen und deren Status (ServiceStatus Objekte oder Dicts)

        Returns:
            Das hinzugefügte Historien-Entry
        """
        with self._lock:
            # Zähle healthy/unhealthy
            healthy_count = 0
            total = len(services_status)

            service_states = {}
            for name, status in services_status.items():
                # Handle both ServiceStatus objects and dicts
                if hasattr(status, 'state'):
                    # ServiceStatus Pydantic object
                    state = status.state.value if hasattr(status.state, 'value') else str(status.state)
                    response_time = status.response_time_ms
                    error = status.error
                elif isinstance(status, dict):
                    # Dict
                    state = str(status.get('state', 'UNKNOWN'))
                    response_time = status.get('response_time_ms')
                    error = status.get('error')
                else:
                    state = "UNKNOWN"
                    response_time = None
                    error = None

                is_healthy = state == "HEALTHY"
                if is_healthy:
                    healthy_count += 1
                service_states[name] = {
                    "state": state,
                    "response_time_ms": response_time,
                    "error": error
                }

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "healthy_count": healthy_count,
                "total": total,
                "all_healthy": healthy_count == total,
                "services": service_states
            }

            self._history.append(entry)

            # Bereinige und speichere
            self._cleanup_old_entries()
            self._save_history()

            return entry

    def get_history(
        self,
        hours: int = 24,
        limit: Optional[int] = None,
        aggregation: str = "raw"
    ) -> List[Dict]:
        """
        Gibt die Health-Historie zurück.

        Args:
            hours: Zeitraum in Stunden (max 24)
            limit: Maximale Anzahl Einträge
            aggregation: "raw" für alle Einträge, "hourly" für stündliche Aggregation

        Returns:
            Liste der Historien-Einträge
        """
        hours = min(hours, self._max_age_hours)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()

        with self._lock:
            filtered = [
                entry for entry in self._history
                if entry.get("timestamp", "") >= cutoff_iso
            ]

        if aggregation == "hourly":
            filtered = self._aggregate_hourly(filtered)

        if limit:
            filtered = filtered[-limit:]

        return filtered

    def _aggregate_hourly(self, entries: List[Dict]) -> List[Dict]:
        """Aggregiert Einträge auf Stundenbasis."""
        hourly = {}

        for entry in entries:
            try:
                ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                hour_key = ts.strftime("%Y-%m-%dT%H:00:00+00:00")

                if hour_key not in hourly:
                    hourly[hour_key] = {
                        "checks": 0,
                        "healthy_sum": 0,
                        "total_sum": 0,
                        "all_healthy_count": 0
                    }

                hourly[hour_key]["checks"] += 1
                hourly[hour_key]["healthy_sum"] += entry.get("healthy_count", 0)
                hourly[hour_key]["total_sum"] += entry.get("total", 0)
                if entry.get("all_healthy"):
                    hourly[hour_key]["all_healthy_count"] += 1
            except Exception as e:
                logger.warning(f"Fehler bei Aggregation: {e}")

        # Konvertiere zu Liste
        result = []
        for hour_key, data in sorted(hourly.items()):
            avg_healthy = data["healthy_sum"] / data["checks"] if data["checks"] > 0 else 0
            avg_total = data["total_sum"] / data["checks"] if data["checks"] > 0 else 0
            uptime_percent = (data["all_healthy_count"] / data["checks"] * 100) if data["checks"] > 0 else 0

            result.append({
                "timestamp": hour_key,
                "checks": data["checks"],
                "avg_healthy": round(avg_healthy, 1),
                "avg_total": round(avg_total, 1),
                "uptime_percent": round(uptime_percent, 1)
            })

        return result

    def get_statistics(self, hours: int = 24) -> Dict:
        """
        Berechnet Statistiken über die Historie.

        Args:
            hours: Zeitraum für Statistiken

        Returns:
            Dict mit Statistiken
        """
        history = self.get_history(hours=hours)

        if not history:
            return {
                "period_hours": hours,
                "total_checks": 0,
                "uptime_percent": 0,
                "avg_healthy_services": 0,
                "avg_response_time_ms": 0
            }

        total_checks = len(history)
        all_healthy_checks = sum(1 for h in history if h.get("all_healthy"))
        uptime_percent = (all_healthy_checks / total_checks * 100) if total_checks > 0 else 0

        # Durchschnittliche healthy Services
        avg_healthy = sum(h.get("healthy_count", 0) for h in history) / total_checks

        # Durchschnittliche Response-Zeit
        response_times = []
        for h in history:
            for svc in h.get("services", {}).values():
                rt = svc.get("response_time_ms")
                if rt is not None:
                    response_times.append(rt)
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            "period_hours": hours,
            "total_checks": total_checks,
            "uptime_percent": round(uptime_percent, 2),
            "avg_healthy_services": round(avg_healthy, 1),
            "avg_response_time_ms": round(avg_response_time, 1),
            "first_check": history[0].get("timestamp") if history else None,
            "last_check": history[-1].get("timestamp") if history else None
        }

    def get_service_history(self, service_name: str, hours: int = 24) -> List[Dict]:
        """
        Gibt die Historie für einen bestimmten Service zurück.

        Args:
            service_name: Name des Services
            hours: Zeitraum in Stunden

        Returns:
            Liste mit Service-spezifischer Historie
        """
        history = self.get_history(hours=hours)

        result = []
        for entry in history:
            service_data = entry.get("services", {}).get(service_name)
            if service_data:
                result.append({
                    "timestamp": entry["timestamp"],
                    **service_data
                })

        return result


# Singleton-Instanz
health_history = HealthHistoryService()
