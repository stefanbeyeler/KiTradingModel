"""Persistente Konfiguration für den Watchdog Service.

Speichert alle konfigurierbaren Einstellungen in einer JSON-Datei.
Das Volume /app/data ist persistent gemountet.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

from loguru import logger


class ConfigService:
    """Verwaltet die persistente Speicherung der Watchdog-Konfiguration."""

    def __init__(self, data_dir: str = "/app/data"):
        """
        Initialisiert den Config Service.

        Args:
            data_dir: Verzeichnis für persistente Daten
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.data_dir / "watchdog_config.json"
        self._lock = Lock()
        self._config: Dict[str, Any] = {}

        # Lade existierende Konfiguration
        self._load_config()
        logger.info(f"Config Service initialisiert. Pfad: {self.config_file}")

    def _get_defaults(self) -> Dict[str, Any]:
        """Gibt die Standard-Konfiguration zurück."""
        try:
            from ..config import settings
            return {
                # Monitoring
                "check_interval_seconds": settings.check_interval_seconds,
                "timeout_seconds": settings.timeout_seconds,
                "max_retries": settings.max_retries,

                # Telegram
                "telegram_enabled": settings.telegram_enabled,
                "telegram_bot_token": settings.telegram_bot_token,
                "telegram_chat_ids": settings.telegram_chat_ids,

                # Alert-Konfiguration
                "alert_cooldown_minutes": settings.alert_cooldown_minutes,
                "alert_on_recovery": settings.alert_on_recovery,
                "alert_on_critical": settings.alert_on_critical,
                "alert_on_high": settings.alert_on_high,
                "alert_on_medium": settings.alert_on_medium,

                # Tägliche Zusammenfassung
                "daily_summary_enabled": settings.daily_summary_enabled,
                "daily_summary_hour": settings.daily_summary_hour,
            }
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Default-Settings: {e}")
            return {
                "check_interval_seconds": 30,
                "timeout_seconds": 10,
                "max_retries": 3,
                "telegram_enabled": True,
                "telegram_bot_token": "",
                "telegram_chat_ids": "",
                "alert_cooldown_minutes": 15,
                "alert_on_recovery": True,
                "alert_on_critical": True,
                "alert_on_high": True,
                "alert_on_medium": False,
                "daily_summary_enabled": True,
                "daily_summary_hour": 8,
            }

    def _load_config(self):
        """Lädt die Konfiguration aus der Datei."""
        defaults = self._get_defaults()

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    # Merge mit Defaults (gespeicherte Werte überschreiben Defaults)
                    self._config = {**defaults, **saved_config}
                    logger.debug(f"Konfiguration geladen: {len(self._config)} Einträge")
            else:
                self._config = defaults
                # Speichere initiale Konfiguration
                self._save_config()
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            self._config = defaults

    def _save_config(self):
        """Speichert die Konfiguration in die Datei."""
        try:
            with self._lock:
                save_data = {
                    **self._config,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2)
                logger.debug("Konfiguration gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Gibt einen Konfigurationswert zurück."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Setzt einen Konfigurationswert und speichert."""
        with self._lock:
            self._config[key] = value
        self._save_config()

    def update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aktualisiert mehrere Konfigurationswerte.

        Args:
            updates: Dict mit zu aktualisierenden Werten

        Returns:
            Die aktualisierte Konfiguration
        """
        with self._lock:
            for key, value in updates.items():
                self._config[key] = value
        self._save_config()
        return self.get_config()

    def get_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Gibt die vollständige Konfiguration zurück.

        Args:
            include_sensitive: Ob sensible Daten (Token) inkludiert werden sollen

        Returns:
            Dict mit Konfiguration
        """
        config = dict(self._config)

        if not include_sensitive:
            # Maskiere sensible Daten
            if config.get("telegram_bot_token"):
                token = config["telegram_bot_token"]
                if len(token) > 10:
                    config["telegram_bot_token"] = f"{token[:4]}...{token[-4:]}"
                else:
                    config["telegram_bot_token"] = "***"

        return config

    def get_telegram_config(self) -> Dict[str, Any]:
        """Gibt die Telegram-Konfiguration zurück (mit Token)."""
        return {
            "enabled": self._config.get("telegram_enabled", True),
            "bot_token": self._config.get("telegram_bot_token", ""),
            "chat_ids": self._config.get("telegram_chat_ids", ""),
            "configured": bool(self._config.get("telegram_bot_token"))
        }

    def set_telegram_config(
        self,
        enabled: Optional[bool] = None,
        bot_token: Optional[str] = None,
        chat_ids: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aktualisiert die Telegram-Konfiguration.

        Args:
            enabled: Telegram aktiviert/deaktiviert
            bot_token: Bot-Token
            chat_ids: Kommagetrennte Chat-IDs

        Returns:
            Die aktualisierte Telegram-Konfiguration
        """
        updates = {}

        if enabled is not None:
            updates["telegram_enabled"] = enabled
        if bot_token is not None:
            updates["telegram_bot_token"] = bot_token
        if chat_ids is not None:
            updates["telegram_chat_ids"] = chat_ids

        if updates:
            self.update(updates)

        return self.get_telegram_config()

    def get_alert_config(self) -> Dict[str, Any]:
        """Gibt die Alert-Konfiguration zurück."""
        return {
            "cooldown_minutes": self._config.get("alert_cooldown_minutes", 15),
            "on_recovery": self._config.get("alert_on_recovery", True),
            "on_critical": self._config.get("alert_on_critical", True),
            "on_high": self._config.get("alert_on_high", True),
            "on_medium": self._config.get("alert_on_medium", False),
        }

    def set_alert_config(
        self,
        cooldown_minutes: Optional[int] = None,
        on_recovery: Optional[bool] = None,
        on_critical: Optional[bool] = None,
        on_high: Optional[bool] = None,
        on_medium: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Aktualisiert die Alert-Konfiguration.

        Returns:
            Die aktualisierte Alert-Konfiguration
        """
        updates = {}

        if cooldown_minutes is not None:
            updates["alert_cooldown_minutes"] = max(1, min(1440, cooldown_minutes))
        if on_recovery is not None:
            updates["alert_on_recovery"] = on_recovery
        if on_critical is not None:
            updates["alert_on_critical"] = on_critical
        if on_high is not None:
            updates["alert_on_high"] = on_high
        if on_medium is not None:
            updates["alert_on_medium"] = on_medium

        if updates:
            self.update(updates)

        return self.get_alert_config()

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Gibt die Monitoring-Konfiguration zurück."""
        return {
            "check_interval_seconds": self._config.get("check_interval_seconds", 30),
            "timeout_seconds": self._config.get("timeout_seconds", 10),
            "max_retries": self._config.get("max_retries", 3),
        }

    def set_monitoring_config(
        self,
        check_interval_seconds: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Aktualisiert die Monitoring-Konfiguration.

        Returns:
            Die aktualisierte Monitoring-Konfiguration
        """
        updates = {}

        if check_interval_seconds is not None:
            updates["check_interval_seconds"] = max(10, min(300, check_interval_seconds))
        if timeout_seconds is not None:
            updates["timeout_seconds"] = max(5, min(60, timeout_seconds))
        if max_retries is not None:
            updates["max_retries"] = max(1, min(10, max_retries))

        if updates:
            self.update(updates)

        return self.get_monitoring_config()

    def get_daily_summary_config(self) -> Dict[str, Any]:
        """Gibt die Konfiguration für die tägliche Zusammenfassung zurück."""
        return {
            "enabled": self._config.get("daily_summary_enabled", True),
            "hour": self._config.get("daily_summary_hour", 8),
        }

    def set_daily_summary_config(
        self,
        enabled: Optional[bool] = None,
        hour: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Aktualisiert die Konfiguration für die tägliche Zusammenfassung.

        Returns:
            Die aktualisierte Konfiguration
        """
        updates = {}

        if enabled is not None:
            updates["daily_summary_enabled"] = enabled
        if hour is not None:
            updates["daily_summary_hour"] = max(0, min(23, hour))

        if updates:
            self.update(updates)

        return self.get_daily_summary_config()

    def _get_default_service_alerts(self) -> Dict[str, bool]:
        """Gibt die Standard-Alarm-Einstellungen pro Service zurück."""
        return {
            "frontend": True,
            "data": True,
            "nhits": True,
            "tcn": True,
            "hmm": True,
            "embedder": True,
            "candlestick": True,
            "redis": True,
            "nhits-train": False,
            "tcn-train": False,
            "hmm-train": False,
            "candlestick-train": False,
            "rag": True,
            "llm": True,
            "easyinsight": True,
        }

    def get_service_alert_config(self) -> Dict[str, Any]:
        """
        Gibt die Alarm-Konfiguration pro Service zurück.

        Returns:
            Dict mit Service-Namen und deren Alarm-Status
        """
        defaults = self._get_default_service_alerts()
        saved = self._config.get("service_alerts", {})

        # Merge: gespeicherte Werte überschreiben Defaults
        service_alerts = {**defaults, **saved}

        return {
            "services": service_alerts,
            "enabled_count": sum(1 for v in service_alerts.values() if v),
            "disabled_count": sum(1 for v in service_alerts.values() if not v),
            "total_count": len(service_alerts)
        }

    def set_service_alert(self, service_name: str, enabled: bool) -> Dict[str, Any]:
        """
        Aktiviert oder deaktiviert Alarme für einen bestimmten Service.

        Args:
            service_name: Name des Services
            enabled: True = Alarme aktiviert, False = deaktiviert

        Returns:
            Die aktualisierte Service-Alarm-Konfiguration
        """
        service_alerts = self._config.get("service_alerts", {})
        service_alerts[service_name] = enabled

        with self._lock:
            self._config["service_alerts"] = service_alerts
        self._save_config()

        return self.get_service_alert_config()

    def set_service_alerts_bulk(self, updates: Dict[str, bool]) -> Dict[str, Any]:
        """
        Aktualisiert Alarme für mehrere Services gleichzeitig.

        Args:
            updates: Dict mit Service-Namen und deren neuen Alarm-Status

        Returns:
            Die aktualisierte Service-Alarm-Konfiguration
        """
        service_alerts = self._config.get("service_alerts", {})
        service_alerts.update(updates)

        with self._lock:
            self._config["service_alerts"] = service_alerts
        self._save_config()

        return self.get_service_alert_config()

    def is_service_alert_enabled(self, service_name: str) -> bool:
        """
        Prüft, ob Alarme für einen bestimmten Service aktiviert sind.

        Args:
            service_name: Name des Services

        Returns:
            True wenn Alarme aktiviert sind
        """
        service_alerts = self._config.get("service_alerts", {})
        defaults = self._get_default_service_alerts()

        return service_alerts.get(service_name, defaults.get(service_name, True))


# Singleton-Instanz
config_service = ConfigService()
