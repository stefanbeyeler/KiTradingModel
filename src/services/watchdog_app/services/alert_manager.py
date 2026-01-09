"""Alert Manager für Watchdog Service."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from loguru import logger

from ..config import WatchdogSettings
from ..models.service_status import HealthState, ServiceStatus
from .telegram_notifier import TelegramNotifier
from .alert_history import alert_history


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
        self._alert_history_service = alert_history

    def _should_alert(self, service_name: str, criticality: str) -> bool:
        """Prüft ob ein Alert basierend auf Service-Konfiguration und Kritikalität gesendet werden soll."""
        # Zuerst prüfen: Ist der Service-Alert überhaupt aktiviert?
        try:
            from .config_service import config_service
            if not config_service.is_service_alert_enabled(service_name):
                logger.debug(f"Alert for {service_name} disabled via service config")
                return False
        except Exception as e:
            logger.warning(f"Could not check service alert config: {e}")
            # Bei Fehler: Fallback auf Kritikalitäts-Prüfung

        # Dann Kritikalitätsstufe prüfen
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
            # Prüfe ob Service-Alarmierung aktiviert ist
            service_alert_enabled = True
            try:
                from .config_service import config_service
                service_alert_enabled = config_service.is_service_alert_enabled(service_name)
            except Exception as e:
                logger.warning(f"Could not check service alert config for recovery: {e}")

            if not service_alert_enabled:
                logger.debug(f"Recovery alert for {service_name} suppressed (service alerts disabled)")
                self._record_alert(
                    service_name=service_name,
                    alert_type="RECOVERY",
                    criticality=criticality,
                    old_state=old_state.value,
                    new_state=new_state.value,
                    error=None,
                    telegram_sent=False,
                    message=f"Recovery-Alert unterdrückt (Service-Alarmierung deaktiviert)",
                    suppressed=True,
                    suppressed_reason="service_disabled"
                )
                return

            if self.settings.alert_on_recovery:
                sent = await self._send_alert(
                    service_name=service_name,
                    state=new_state.value,
                    recovery=True
                )
                self._record_alert(
                    service_name=service_name,
                    alert_type="RECOVERY",
                    criticality=criticality,
                    old_state=old_state.value,
                    new_state=new_state.value,
                    error=None,
                    telegram_sent=sent,
                    message=f"Service {service_name} ist wieder verfügbar"
                )
            return

        # Failure-Alert?
        if new_state in [HealthState.UNHEALTHY, HealthState.DEGRADED]:
            # Service-spezifische und Kritikalitäts-Prüfung
            if not self._should_alert(service_name, criticality):
                # Ermittle den genauen Grund für die Unterdrückung
                suppressed_reason = "criticality"
                suppressed_message = f"Alert unterdrückt (Kritikalität {criticality} nicht aktiviert)"

                try:
                    from .config_service import config_service
                    if not config_service.is_service_alert_enabled(service_name):
                        suppressed_reason = "service_disabled"
                        suppressed_message = f"Alert unterdrückt (Service-Alarmierung deaktiviert)"
                except Exception:
                    pass

                logger.debug(
                    f"Alert for {service_name} suppressed ({suppressed_reason})"
                )
                # Trotzdem aufzeichnen, aber als suppressed
                self._record_alert(
                    service_name=service_name,
                    alert_type=new_state.value,
                    criticality=criticality,
                    old_state=old_state.value,
                    new_state=new_state.value,
                    error=new_status.error,
                    telegram_sent=False,
                    message=suppressed_message,
                    suppressed=True,
                    suppressed_reason=suppressed_reason
                )
                return

            # Cooldown prüfen
            if self._is_in_cooldown(service_name):
                logger.debug(f"Alert for {service_name} suppressed (cooldown)")
                self._record_alert(
                    service_name=service_name,
                    alert_type=new_state.value,
                    criticality=criticality,
                    old_state=old_state.value,
                    new_state=new_state.value,
                    error=new_status.error,
                    telegram_sent=False,
                    message=f"Alert unterdrückt (Cooldown aktiv)",
                    suppressed=True,
                    suppressed_reason="cooldown"
                )
                return

            # Alert senden
            sent = await self._send_alert(
                service_name=service_name,
                state=new_state.value,
                error=new_status.error,
                recovery=False
            )
            self._record_alert(
                service_name=service_name,
                alert_type=new_state.value,
                criticality=criticality,
                old_state=old_state.value,
                new_state=new_state.value,
                error=new_status.error,
                telegram_sent=sent,
                message=f"Service {service_name} ist {new_state.value}: {new_status.error or 'Keine Details'}"
            )

    async def _send_alert(
        self,
        service_name: str,
        state: str,
        error: Optional[str] = None,
        recovery: bool = False
    ) -> bool:
        """
        Sendet Alert über Telegram.

        Returns:
            True wenn Alert erfolgreich gesendet wurde
        """
        if self.telegram and self.telegram.enabled:
            return await self.telegram.send_alert(
                service_name=service_name,
                state=state,
                error=error,
                recovery=recovery
            )
        return False

    def _record_alert(
        self,
        service_name: str,
        alert_type: str,
        criticality: str = "unknown",
        old_state: str = "UNKNOWN",
        new_state: str = "UNKNOWN",
        error: Optional[str] = None,
        telegram_sent: bool = False,
        message: str = "",
        suppressed: bool = False,
        suppressed_reason: Optional[str] = None
    ):
        """
        Zeichnet einen Alert mit allen Details auf (persistent).

        Args:
            service_name: Name des betroffenen Services
            alert_type: Art des Alerts (UNHEALTHY, DEGRADED, RECOVERY)
            criticality: Kritikalitätsstufe (critical, high, medium)
            old_state: Vorheriger Status
            new_state: Neuer Status
            error: Fehlermeldung falls vorhanden
            telegram_sent: Ob Telegram-Nachricht gesendet wurde
            message: Zusammenfassende Nachricht
            suppressed: Ob Alert unterdrückt wurde
            suppressed_reason: Grund für Unterdrückung (cooldown, criticality)
        """
        now = datetime.now(timezone.utc)

        # Cooldown nur bei tatsächlich gesendeten Alerts aktualisieren
        if not suppressed and telegram_sent:
            self.last_alerts[service_name] = now

        alert_entry = {
            "service": service_name,
            "type": alert_type,
            "timestamp": now.isoformat(),
            "criticality": criticality,
            "state_change": {
                "from": old_state,
                "to": new_state
            },
            "error": error,
            "telegram_sent": telegram_sent,
            "message": message,
            "suppressed": suppressed,
            "suppressed_reason": suppressed_reason
        }

        # Persistente Speicherung über AlertHistoryService
        self._alert_history_service.add_alert(alert_entry)

    def get_statistics(self) -> dict:
        """Gibt Alert-Statistiken zurück."""
        stats = self._alert_history_service.get_statistics()

        # Füge Cooldown-Informationen hinzu
        stats["services_in_cooldown"] = [
            name for name in self.last_alerts
            if self._is_in_cooldown(name)
        ]

        return stats

    def get_alert_history(self, limit: int = 50) -> list:
        """Gibt die letzten Alerts zurück (persistent)."""
        return self._alert_history_service.get_history(limit=limit)

    def clear_alert_history(self):
        """Löscht die Alert-Historie."""
        self._alert_history_service.clear_history()
