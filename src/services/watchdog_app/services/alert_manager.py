"""Alert Manager für Watchdog Service."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from loguru import logger

from ..config import WatchdogSettings
from ..models.service_status import HealthState, ServiceStatus
from .telegram_notifier import TelegramNotifier


class AlertManager:
    """Verwaltet Alerts mit Deduplizierung und Cooldown."""

    def __init__(
        self,
        settings: WatchdogSettings,
        telegram_notifier: Optional[TelegramNotifier] = None
    ):
        """
        Initialisiert den Alert Manager.

        Args:
            settings: Watchdog-Konfiguration
            telegram_notifier: Telegram Notifier Instanz
        """
        self.settings = settings
        self.telegram = telegram_notifier
        self.last_alerts: Dict[str, datetime] = {}
        self.last_states: Dict[str, HealthState] = {}
        self.alert_history: list = []

    def _should_alert(self, service_name: str, criticality: str) -> bool:
        """Prüft ob ein Alert basierend auf Kritikalität gesendet werden soll."""
        if criticality == "critical" and self.settings.alert_on_critical:
            return True
        if criticality == "high" and self.settings.alert_on_high:
            return True
        if criticality == "medium" and self.settings.alert_on_medium:
            return True
        return False

    def _is_in_cooldown(self, service_name: str) -> bool:
        """Prüft ob der Service im Alert-Cooldown ist."""
        last_alert = self.last_alerts.get(service_name)
        if not last_alert:
            return False

        cooldown = timedelta(minutes=self.settings.alert_cooldown_minutes)
        return datetime.now(timezone.utc) - last_alert < cooldown

    async def process_status_change(
        self,
        service_name: str,
        new_status: ServiceStatus,
        criticality: str
    ):
        """
        Verarbeitet Statusänderungen und sendet ggf. Alerts.

        Args:
            service_name: Name des Services
            new_status: Neuer Status
            criticality: Kritikalitätsstufe des Services
        """
        old_state = self.last_states.get(service_name, HealthState.UNKNOWN)
        new_state = new_status.state

        # Keine Änderung - kein Alert
        if old_state == new_state:
            return

        # Status aktualisieren
        self.last_states[service_name] = new_state

        # Logging
        logger.info(f"Service {service_name}: {old_state.value} -> {new_state.value}")

        # Recovery-Alert?
        if new_state == HealthState.HEALTHY and old_state in [
            HealthState.UNHEALTHY, HealthState.DEGRADED
        ]:
            if self.settings.alert_on_recovery:
                await self._send_alert(
                    service_name=service_name,
                    state=new_state.value,
                    recovery=True
                )
                self._record_alert(service_name, "RECOVERY")
            return

        # Failure-Alert?
        if new_state in [HealthState.UNHEALTHY, HealthState.DEGRADED]:
            # Kritikalität prüfen
            if not self._should_alert(service_name, criticality):
                logger.debug(
                    f"Alert for {service_name} suppressed (criticality: {criticality})"
                )
                return

            # Cooldown prüfen
            if self._is_in_cooldown(service_name):
                logger.debug(f"Alert for {service_name} suppressed (cooldown)")
                return

            # Alert senden
            await self._send_alert(
                service_name=service_name,
                state=new_state.value,
                error=new_status.error,
                recovery=False
            )
            self._record_alert(service_name, new_state.value)

    async def _send_alert(
        self,
        service_name: str,
        state: str,
        error: Optional[str] = None,
        recovery: bool = False
    ):
        """Sendet Alert über Telegram."""
        if self.telegram:
            await self.telegram.send_alert(
                service_name=service_name,
                state=state,
                error=error,
                recovery=recovery
            )

    def _record_alert(self, service_name: str, alert_type: str):
        """Zeichnet einen Alert auf."""
        now = datetime.now(timezone.utc)
        self.last_alerts[service_name] = now
        self.alert_history.append({
            "service": service_name,
            "type": alert_type,
            "timestamp": now.isoformat()
        })

        # History auf 1000 Einträge begrenzen
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    def get_statistics(self) -> dict:
        """Gibt Alert-Statistiken zurück."""
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)

        alerts_24h = [
            a for a in self.alert_history
            if datetime.fromisoformat(a["timestamp"].replace("Z", "+00:00")) > last_24h
        ]

        return {
            "total_alerts": len(self.alert_history),
            "alerts_24h": len(alerts_24h),
            "last_alert": self.alert_history[-1] if self.alert_history else None,
            "services_in_cooldown": [
                name for name in self.last_alerts
                if self._is_in_cooldown(name)
            ]
        }

    def get_alert_history(self, limit: int = 50) -> list:
        """Gibt die letzten Alerts zurück."""
        return list(reversed(self.alert_history[-limit:]))
