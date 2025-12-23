# Watchdog Service - Vorschlag

## Übersicht

Ein zentraler Watchdog-Service zur Überwachung aller Microservices mit **Telegram** und optional **WhatsApp** Alarmierung.

```
┌─────────────────────────────────────────────────────────────────┐
│                    WATCHDOG SERVICE (Port 3010)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Health     │    │  Metrics    │    │  Alert      │        │
│   │  Checker    │───▶│  Collector  │───▶│  Manager    │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                                      │                │
│          ▼                                      ▼                │
│   ┌─────────────┐              ┌────────────────────────────┐   │
│   │  Service    │              │     Notification Router    │   │
│   │  Registry   │              ├──────────────┬─────────────┤   │
│   └─────────────┘              │   Telegram   │  WhatsApp   │   │
│                                │   Notifier   │  Notifier   │   │
│                                └──────────────┴─────────────┘   │
│                                       │              │          │
└───────────────────────────────────────│──────────────│──────────┘
                                        │              │
                                        ▼              ▼
                                 ┌───────────┐  ┌─────────────┐
                                 │ Telegram  │  │  Twilio API │
                                 │ Bot API   │  │  (WhatsApp) │
                                 │ (GRATIS)  │  │  (Kostenpfl)│
                                 └───────────┘  └─────────────┘
```

## Vergleich: Telegram vs. WhatsApp

| Feature                      | Telegram           | WhatsApp (Twilio)        |
|------------------------------|--------------------| -------------------------|
| **Kosten**                   | Kostenlos          | ~$0.005-0.05/Nachricht   |
| **Setup**                    | 2 Minuten          | 10-15 Minuten            |
| **Bot erstellen**            | @BotFather         | Twilio Console           |
| **Empfänger-Registrierung**  | `/start` an Bot    | Join Sandbox             |
| **Gruppenchats**             | Ja                 | Nein (Sandbox)           |
| **Rich Formatting**          | Markdown, HTML     | Nur Basic                |
| **Bilder/Dateien**           | Ja                 | Ja                       |
| **API Rate Limits**          | 30 msg/sec         | Abhängig vom Plan        |
| **Empfehlung**               | **Primär**         | Backup/Optional          |

## Zu überwachende Services

| Service | Port | Docker-Host | Kritikalität | Startup-Zeit |
|---------|------|-------------|--------------|--------------|
| Frontend | 3000 | trading-frontend | Medium | 10s |
| Data Service | 3001 | trading-data | **Kritisch** | 20s |
| NHITS | 3002 | trading-nhits | Hoch | 40s |
| TCN-Pattern | 3003 | trading-tcn | Hoch | 40s |
| HMM-Regime | 3004 | trading-hmm | Hoch | 30s |
| Embedder | 3005 | trading-embedder | Hoch | 120s |
| RAG | 3008 | trading-rag | Hoch | 60s |
| LLM | 3009 | trading-llm | Medium | 60s |

## Architektur

### 1. Service-Struktur

```
src/services/watchdog_app/
├── __init__.py
├── main.py                    # FastAPI App mit Lifespan
├── config.py                  # Watchdog-Konfiguration
├── models/
│   ├── __init__.py
│   ├── service_status.py      # Status-Modelle
│   └── alert_models.py        # Alert-Datenmodelle
├── services/
│   ├── __init__.py
│   ├── health_checker.py      # Health-Check-Logik
│   ├── metrics_collector.py   # Metriken-Sammlung
│   ├── alert_manager.py       # Alert-Logik & Deduplizierung
│   └── whatsapp_notifier.py   # Twilio WhatsApp Integration
└── api/
    ├── __init__.py
    └── routes.py              # API-Endpoints
```

### 2. Konfiguration

```python
# src/services/watchdog_app/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class WatchdogSettings(BaseSettings):
    # Service-Konfiguration
    watchdog_port: int = 3010
    check_interval_seconds: int = 30
    timeout_seconds: int = 10
    max_retries: int = 3

    # ============================================
    # TELEGRAM (Empfohlen - Kostenlos)
    # ============================================
    telegram_enabled: bool = True
    telegram_bot_token: str = ""           # Von @BotFather
    telegram_chat_ids: str = ""            # Kommagetrennt: "123456789,-100987654321"

    # ============================================
    # WHATSAPP (Optional - Kostenpflichtig)
    # ============================================
    whatsapp_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = "whatsapp:+14155238886"  # Twilio Sandbox
    whatsapp_recipients: str = ""          # Kommagetrennt: "+41791234567,+41799876543"

    # ============================================
    # Alert-Konfiguration
    # ============================================
    alert_cooldown_minutes: int = 15       # Keine Wiederholung innerhalb 15 Min
    alert_on_recovery: bool = True         # Auch bei Wiederherstellung alarmieren

    # Kritikalitätsstufen für Alarmierung
    alert_on_critical: bool = True
    alert_on_high: bool = True
    alert_on_medium: bool = False

    # Tägliche Zusammenfassung
    daily_summary_enabled: bool = True
    daily_summary_hour: int = 8            # 08:00 Uhr

    model_config = {
        "env_file": ".env.watchdog",
        "env_prefix": "WATCHDOG_"
    }
```

### 3. Health-Checker Service

```python
# src/services/watchdog_app/services/health_checker.py
import asyncio
import httpx
from datetime import datetime, timezone
from typing import Dict, Optional
from loguru import logger

from ..models.service_status import ServiceStatus, HealthState

class HealthChecker:
    """Prüft regelmässig den Health-Status aller Services."""

    def __init__(self, settings):
        self.settings = settings
        self.services = self._init_services()
        self.status: Dict[str, ServiceStatus] = {}
        self._running = False

    def _init_services(self) -> Dict[str, dict]:
        """Service-Registry mit Konfiguration."""
        return {
            "frontend": {
                "url": "http://trading-frontend:3000/health",
                "criticality": "medium",
                "startup_grace": 10,
                "dependencies": []
            },
            "data": {
                "url": "http://trading-data:3001/health",
                "criticality": "critical",
                "startup_grace": 20,
                "dependencies": []
            },
            "nhits": {
                "url": "http://trading-nhits:3002/health",
                "criticality": "high",
                "startup_grace": 40,
                "dependencies": ["data"]
            },
            "tcn": {
                "url": "http://trading-tcn:3003/health",
                "criticality": "high",
                "startup_grace": 40,
                "dependencies": ["data", "embedder"]
            },
            "hmm": {
                "url": "http://trading-hmm:3004/health",
                "criticality": "high",
                "startup_grace": 30,
                "dependencies": ["data"]
            },
            "embedder": {
                "url": "http://trading-embedder:3005/health",
                "criticality": "high",
                "startup_grace": 120,
                "dependencies": ["data"]
            },
            "rag": {
                "url": "http://trading-rag:3008/health",
                "criticality": "high",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "llm": {
                "url": "http://trading-llm:3009/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["rag"]
            }
        }

    async def check_service(self, name: str, config: dict) -> ServiceStatus:
        """Prüft einen einzelnen Service."""
        start_time = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                response = await client.get(config["url"])
                response_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "healthy")

                    return ServiceStatus(
                        name=name,
                        state=HealthState.HEALTHY if status == "healthy" else HealthState.DEGRADED,
                        response_time_ms=response_time_ms,
                        last_check=start_time,
                        details=data,
                        consecutive_failures=0
                    )
                else:
                    return self._create_failure_status(name, start_time, f"HTTP {response.status_code}")

        except httpx.TimeoutException:
            return self._create_failure_status(name, start_time, "Timeout")
        except httpx.ConnectError:
            return self._create_failure_status(name, start_time, "Connection refused")
        except Exception as e:
            return self._create_failure_status(name, start_time, str(e))

    def _create_failure_status(self, name: str, check_time: datetime, error: str) -> ServiceStatus:
        """Erstellt einen Fehler-Status."""
        prev_status = self.status.get(name)
        consecutive = (prev_status.consecutive_failures + 1) if prev_status else 1

        return ServiceStatus(
            name=name,
            state=HealthState.UNHEALTHY,
            response_time_ms=None,
            last_check=check_time,
            error=error,
            consecutive_failures=consecutive
        )

    async def check_all_services(self) -> Dict[str, ServiceStatus]:
        """Prüft alle Services parallel."""
        tasks = [
            self.check_service(name, config)
            for name, config in self.services.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, _) in enumerate(self.services.items()):
            if isinstance(results[i], Exception):
                self.status[name] = self._create_failure_status(
                    name, datetime.now(timezone.utc), str(results[i])
                )
            else:
                self.status[name] = results[i]

        return self.status

    async def run_monitoring_loop(self):
        """Hauptschleife für kontinuierliches Monitoring."""
        self._running = True
        logger.info("Watchdog monitoring loop started")

        while self._running:
            try:
                await self.check_all_services()
                logger.debug(f"Health check completed: {len(self.status)} services checked")
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            await asyncio.sleep(self.settings.check_interval_seconds)

    def stop(self):
        """Stoppt die Monitoring-Schleife."""
        self._running = False
        logger.info("Watchdog monitoring loop stopped")
```

### 4. Telegram-Notifier (Empfohlen - Kostenlos)

```python
# src/services/watchdog_app/services/telegram_notifier.py
import httpx
from datetime import datetime, timezone
from typing import List, Optional
from loguru import logger

class TelegramNotifier:
    """Sendet Telegram-Nachrichten über die Bot API (kostenlos)."""

    def __init__(self, settings):
        self.settings = settings
        self.bot_token = settings.telegram_bot_token
        self.chat_ids = self._parse_chat_ids(settings.telegram_chat_ids)
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.enabled = settings.telegram_enabled and bool(self.bot_token)

    def _parse_chat_ids(self, chat_ids_str: str) -> List[str]:
        """Parst kommagetrennte Chat-IDs (User oder Gruppen)."""
        if not chat_ids_str:
            return []
        return [cid.strip() for cid in chat_ids_str.split(",") if cid.strip()]

    async def send_alert(
        self,
        service_name: str,
        state: str,
        error: Optional[str] = None,
        recovery: bool = False
    ) -> bool:
        """
        Sendet einen Alert an alle konfigurierten Chats.

        Args:
            service_name: Name des betroffenen Services
            state: Aktueller Status (UNHEALTHY, DEGRADED, etc.)
            error: Optionale Fehlermeldung
            recovery: True wenn es eine Wiederherstellungsmeldung ist

        Returns:
            True wenn mindestens eine Nachricht erfolgreich gesendet wurde
        """
        if not self.enabled:
            logger.debug("Telegram notifications disabled")
            return False

        if not self.chat_ids:
            logger.warning("No Telegram chat IDs configured")
            return False

        # Nachricht formatieren (Telegram MarkdownV2)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if recovery:
            message = f"""
✅ *SERVICE RECOVERED*

📦 Service: `{service_name}`
🕐 Zeit: {timestamp}
📊 Status: {state}

Der Service ist wieder verfügbar\\.
"""
        else:
            emoji = "🚨" if state == "UNHEALTHY" else "⚠️"
            # Escape special chars for MarkdownV2
            error_escaped = self._escape_markdown(error or "Keine Details")
            message = f"""
{emoji} *SERVICE ALERT*

📦 Service: `{service_name}`
🕐 Zeit: {timestamp}
📊 Status: *{state}*
❌ Fehler: {error_escaped}

Bitte umgehend prüfen\\!
"""

        success_count = 0
        async with httpx.AsyncClient(timeout=10) as client:
            for chat_id in self.chat_ids:
                try:
                    response = await client.post(
                        f"{self.api_url}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": message.strip(),
                            "parse_mode": "MarkdownV2"
                        }
                    )
                    if response.status_code == 200:
                        logger.info(f"Telegram alert sent to {chat_id}")
                        success_count += 1
                    else:
                        logger.error(f"Telegram API error: {response.text}")
                except Exception as e:
                    logger.error(f"Failed to send Telegram to {chat_id}: {e}")

        return success_count > 0

    def _escape_markdown(self, text: str) -> str:
        """Escaped Sonderzeichen für Telegram MarkdownV2."""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    async def send_daily_summary(self, summary: dict) -> bool:
        """Sendet eine tägliche Zusammenfassung."""
        if not self.enabled or not self.chat_ids:
            return False

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        healthy = summary.get("healthy", 0)
        degraded = summary.get("degraded", 0)
        unhealthy = summary.get("unhealthy", 0)
        total = healthy + degraded + unhealthy

        status_emoji = "✅" if unhealthy == 0 else "⚠️"

        message = f"""
📊 *DAILY SYSTEM REPORT*

🕐 Zeit: {timestamp}

{status_emoji} Service Status:
• Healthy: {healthy}/{total}
• Degraded: {degraded}/{total}
• Unhealthy: {unhealthy}/{total}

📈 24h Statistiken:
• Alerts: {summary.get('alerts_24h', 0)}
• Avg Response: {summary.get('avg_response_ms', 0):.0f}ms
• Uptime: {summary.get('uptime_percent', 0):.1f}%
"""

        async with httpx.AsyncClient(timeout=10) as client:
            for chat_id in self.chat_ids:
                try:
                    await client.post(
                        f"{self.api_url}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": message.strip(),
                            "parse_mode": "Markdown"
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to send daily summary to {chat_id}: {e}")

        return True

    async def send_test_message(self) -> dict:
        """Sendet eine Test-Nachricht und gibt Ergebnis zurück."""
        results = {"success": [], "failed": []}

        message = """
🧪 *WATCHDOG TEST*

Dies ist eine Test\\-Nachricht vom KI Trading Watchdog\\.

✅ Telegram\\-Integration funktioniert\\!
"""

        async with httpx.AsyncClient(timeout=10) as client:
            for chat_id in self.chat_ids:
                try:
                    response = await client.post(
                        f"{self.api_url}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": message.strip(),
                            "parse_mode": "MarkdownV2"
                        }
                    )
                    if response.status_code == 200:
                        results["success"].append(chat_id)
                    else:
                        results["failed"].append({
                            "chat_id": chat_id,
                            "error": response.json().get("description", "Unknown error")
                        })
                except Exception as e:
                    results["failed"].append({"chat_id": chat_id, "error": str(e)})

        return results
```

### 5. WhatsApp-Notifier (Optional - Twilio)

```python
# src/services/watchdog_app/services/whatsapp_notifier.py
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from datetime import datetime, timezone
from typing import List, Optional
from loguru import logger

class WhatsAppNotifier:
    """Sendet WhatsApp-Nachrichten über Twilio."""

    def __init__(self, settings):
        self.settings = settings
        self.client = Client(
            settings.twilio_account_sid,
            settings.twilio_auth_token
        )
        self.from_number = settings.twilio_whatsapp_from
        self.recipients = self._parse_recipients(settings.alert_recipients)

    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parst kommagetrennte Empfängerliste."""
        if not recipients_str:
            return []
        return [f"whatsapp:{r.strip()}" for r in recipients_str.split(",") if r.strip()]

    async def send_alert(
        self,
        service_name: str,
        state: str,
        error: Optional[str] = None,
        recovery: bool = False
    ) -> bool:
        """
        Sendet einen Alert an alle Empfänger.

        Args:
            service_name: Name des betroffenen Services
            state: Aktueller Status (UNHEALTHY, DEGRADED, etc.)
            error: Optionale Fehlermeldung
            recovery: True wenn es eine Wiederherstellungsmeldung ist

        Returns:
            True wenn mindestens eine Nachricht erfolgreich gesendet wurde
        """
        if not self.recipients:
            logger.warning("No WhatsApp recipients configured")
            return False

        # Nachricht formatieren
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if recovery:
            emoji = "✅"
            title = "SERVICE RECOVERED"
            message = f"""
{emoji} *{title}*

📦 Service: *{service_name}*
🕐 Zeit: {timestamp}
📊 Status: {state}

Der Service ist wieder verfügbar.
"""
        else:
            emoji = "🚨" if state == "UNHEALTHY" else "⚠️"
            title = "SERVICE ALERT"
            message = f"""
{emoji} *{title}*

📦 Service: *{service_name}*
🕐 Zeit: {timestamp}
📊 Status: *{state}*
❌ Fehler: {error or 'Keine Details'}

Bitte umgehend prüfen!
"""

        success_count = 0
        for recipient in self.recipients:
            try:
                self.client.messages.create(
                    body=message.strip(),
                    from_=self.from_number,
                    to=recipient
                )
                logger.info(f"WhatsApp alert sent to {recipient}")
                success_count += 1
            except TwilioRestException as e:
                logger.error(f"Failed to send WhatsApp to {recipient}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending WhatsApp: {e}")

        return success_count > 0

    async def send_daily_summary(self, summary: dict) -> bool:
        """Sendet eine tägliche Zusammenfassung."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        healthy = summary.get("healthy", 0)
        degraded = summary.get("degraded", 0)
        unhealthy = summary.get("unhealthy", 0)
        total = healthy + degraded + unhealthy

        status_emoji = "✅" if unhealthy == 0 else "⚠️"

        message = f"""
📊 *DAILY SYSTEM REPORT*

🕐 Zeit: {timestamp}

{status_emoji} Service Status:
• Healthy: {healthy}/{total}
• Degraded: {degraded}/{total}
• Unhealthy: {unhealthy}/{total}

📈 24h Statistiken:
• Alerts: {summary.get('alerts_24h', 0)}
• Avg Response: {summary.get('avg_response_ms', 0):.0f}ms
• Uptime: {summary.get('uptime_percent', 0):.1f}%
"""

        for recipient in self.recipients:
            try:
                self.client.messages.create(
                    body=message.strip(),
                    from_=self.from_number,
                    to=recipient
                )
            except Exception as e:
                logger.error(f"Failed to send daily summary to {recipient}: {e}")

        return True
```

### 6. Alert-Manager (Multi-Channel)

```python
# src/services/watchdog_app/services/alert_manager.py
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from loguru import logger

from ..models.service_status import ServiceStatus, HealthState
from .telegram_notifier import TelegramNotifier
from .whatsapp_notifier import WhatsAppNotifier

class AlertManager:
    """Verwaltet Alerts mit Deduplizierung, Cooldown und Multi-Channel Support."""

    def __init__(
        self,
        settings,
        telegram_notifier: Optional[TelegramNotifier] = None,
        whatsapp_notifier: Optional[WhatsAppNotifier] = None
    ):
        self.settings = settings
        self.telegram = telegram_notifier
        self.whatsapp = whatsapp_notifier
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

        Alert-Regeln:
        1. Nur bei Statusänderung (nicht bei jedem Check)
        2. Cooldown beachten
        3. Kritikalität prüfen
        4. Optional: Recovery-Alerts
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
        if new_state == HealthState.HEALTHY and old_state in [HealthState.UNHEALTHY, HealthState.DEGRADED]:
            if self.settings.alert_on_recovery:
                await self._send_to_all_channels(
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
                logger.debug(f"Alert for {service_name} suppressed (criticality: {criticality})")
                return

            # Cooldown prüfen
            if self._is_in_cooldown(service_name):
                logger.debug(f"Alert for {service_name} suppressed (cooldown)")
                return

            # Alert über alle Kanäle senden
            await self._send_to_all_channels(
                service_name=service_name,
                state=new_state.value,
                error=new_status.error,
                recovery=False
            )
            self._record_alert(service_name, new_state.value)

    async def _send_to_all_channels(
        self,
        service_name: str,
        state: str,
        error: Optional[str] = None,
        recovery: bool = False
    ):
        """Sendet Alert an alle konfigurierten Kanäle."""
        # Telegram (primär)
        if self.telegram:
            await self.telegram.send_alert(
                service_name=service_name,
                state=state,
                error=error,
                recovery=recovery
            )

        # WhatsApp (sekundär/backup)
        if self.whatsapp:
            await self.whatsapp.send_alert(
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
            if datetime.fromisoformat(a["timestamp"]) > last_24h
        ]

        return {
            "total_alerts": len(self.alert_history),
            "alerts_24h": len(alerts_24h),
            "last_alert": self.alert_history[-1] if self.alert_history else None,
            "services_in_cooldown": [
                name for name, _ in self.last_alerts.items()
                if self._is_in_cooldown(name)
            ]
        }
```

### 6. Datenmodelle

```python
# src/services/watchdog_app/models/service_status.py
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

class HealthState(str, Enum):
    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    STARTING = "STARTING"

class ServiceStatus(BaseModel):
    name: str
    state: HealthState
    response_time_ms: Optional[float] = None
    last_check: datetime
    error: Optional[str] = None
    consecutive_failures: int = 0
    details: Optional[Dict[str, Any]] = None

class SystemHealth(BaseModel):
    timestamp: datetime
    overall_state: HealthState
    services: Dict[str, ServiceStatus]
    healthy_count: int
    degraded_count: int
    unhealthy_count: int

class AlertConfig(BaseModel):
    service_name: str
    enabled: bool = True
    criticality: str = "medium"
    custom_recipients: Optional[list] = None
```

### 7. Main Application

```python
# src/services/watchdog_app/main.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from src.shared.logging_config import setup_logging, log_startup_info, log_shutdown_info
from .config import WatchdogSettings
from .services.health_checker import HealthChecker
from .services.whatsapp_notifier import WhatsAppNotifier
from .services.alert_manager import AlertManager
from .api.routes import router

VERSION = "1.0.0"
settings = WatchdogSettings()

# Services
health_checker: HealthChecker = None
whatsapp_notifier: WhatsAppNotifier = None
alert_manager: AlertManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management für den Watchdog Service."""
    global health_checker, whatsapp_notifier, alert_manager

    setup_logging("watchdog")
    log_startup_info("watchdog", VERSION, settings.watchdog_port, gpu_enabled=False)

    # Services initialisieren
    health_checker = HealthChecker(settings)
    whatsapp_notifier = WhatsAppNotifier(settings)
    alert_manager = AlertManager(settings, whatsapp_notifier)

    # Monitoring-Loop starten
    monitoring_task = asyncio.create_task(run_monitoring_with_alerts())

    logger.info(f"Watchdog monitoring {len(health_checker.services)} services")
    logger.info(f"WhatsApp recipients: {len(whatsapp_notifier.recipients)}")

    yield

    # Shutdown
    health_checker.stop()
    monitoring_task.cancel()
    log_shutdown_info("watchdog")

async def run_monitoring_with_alerts():
    """Kombiniert Health-Checks mit Alert-Verarbeitung."""
    while True:
        try:
            # Alle Services prüfen
            await health_checker.check_all_services()

            # Status-Änderungen verarbeiten
            for name, status in health_checker.status.items():
                config = health_checker.services.get(name, {})
                await alert_manager.process_status_change(
                    service_name=name,
                    new_status=status,
                    criticality=config.get("criticality", "medium")
                )

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

        await asyncio.sleep(settings.check_interval_seconds)

app = FastAPI(
    title="Watchdog Service",
    description="Überwacht alle Microservices und alarmiert per WhatsApp",
    version=VERSION,
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health-Check für den Watchdog selbst."""
    return {
        "service": "watchdog",
        "status": "healthy",
        "version": VERSION,
        "monitoring_active": health_checker._running if health_checker else False,
        "services_monitored": len(health_checker.services) if health_checker else 0
    }
```

### 8. API-Endpoints

```python
# src/services/watchdog_app/api/routes.py
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone

router = APIRouter(tags=["Watchdog"])

@router.get("/status")
async def get_system_status():
    """Gibt den aktuellen System-Status zurück."""
    from ..main import health_checker, alert_manager

    if not health_checker:
        raise HTTPException(status_code=503, detail="Watchdog not initialized")

    services = health_checker.status

    healthy = sum(1 for s in services.values() if s.state.value == "HEALTHY")
    degraded = sum(1 for s in services.values() if s.state.value == "DEGRADED")
    unhealthy = sum(1 for s in services.values() if s.state.value == "UNHEALTHY")

    overall = "HEALTHY"
    if unhealthy > 0:
        overall = "UNHEALTHY"
    elif degraded > 0:
        overall = "DEGRADED"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "services": {
            name: {
                "state": s.state.value,
                "response_time_ms": s.response_time_ms,
                "last_check": s.last_check.isoformat(),
                "error": s.error,
                "consecutive_failures": s.consecutive_failures
            }
            for name, s in services.items()
        },
        "summary": {
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "total": len(services)
        }
    }

@router.get("/alerts/history")
async def get_alert_history(limit: int = 50):
    """Gibt die Alert-Historie zurück."""
    from ..main import alert_manager

    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    history = alert_manager.alert_history[-limit:]
    return {
        "alerts": list(reversed(history)),
        "statistics": alert_manager.get_statistics()
    }

@router.post("/alerts/test")
async def send_test_alert():
    """Sendet einen Test-Alert."""
    from ..main import whatsapp_notifier

    if not whatsapp_notifier:
        raise HTTPException(status_code=503, detail="Notifier not initialized")

    success = await whatsapp_notifier.send_alert(
        service_name="watchdog-test",
        state="TEST",
        error="Dies ist ein Test-Alert"
    )

    return {"success": success, "message": "Test alert sent" if success else "Failed to send"}

@router.get("/services")
async def list_monitored_services():
    """Liste aller überwachten Services."""
    from ..main import health_checker

    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")

    return {
        "services": [
            {
                "name": name,
                "url": config["url"],
                "criticality": config["criticality"],
                "dependencies": config["dependencies"]
            }
            for name, config in health_checker.services.items()
        ]
    }
```

## Docker-Integration

### docker-compose.watchdog.yml

```yaml
version: '3.8'

services:
  trading-watchdog:
    build:
      context: .
      dockerfile: docker/Dockerfile.watchdog
    container_name: trading-watchdog
    ports:
      - "3010:3010"
    environment:
      - WATCHDOG_PORT=3010
      - WATCHDOG_CHECK_INTERVAL_SECONDS=30
      - WATCHDOG_TIMEOUT_SECONDS=10
      - WATCHDOG_TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID}
      - WATCHDOG_TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN}
      - WATCHDOG_TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
      - WATCHDOG_ALERT_RECIPIENTS=${WHATSAPP_RECIPIENTS}
      - WATCHDOG_ALERT_COOLDOWN_MINUTES=15
      - WATCHDOG_ALERT_ON_RECOVERY=true
    networks:
      - trading-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3010/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

networks:
  trading-net:
    external: true
```

### Dockerfile.watchdog

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.watchdog.txt .
RUN pip install --no-cache-dir -r requirements.watchdog.txt

# Application code
COPY src/ ./src/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 3010

CMD ["python", "-m", "uvicorn", "src.services.watchdog_app.main:app", "--host", "0.0.0.0", "--port", "3010"]
```

### requirements.watchdog.txt

```
fastapi>=0.109.0
uvicorn>=0.27.0
httpx>=0.26.0
twilio>=8.10.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
loguru>=0.7.2
python-dotenv>=1.0.0
```

## Umgebungsvariablen

### .env.watchdog

```bash
# ============================================
# TELEGRAM (Empfohlen - Kostenlos)
# ============================================
WATCHDOG_TELEGRAM_ENABLED=true
WATCHDOG_TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WATCHDOG_TELEGRAM_CHAT_IDS=123456789,-100987654321

# ============================================
# WHATSAPP (Optional - Kostenpflichtig)
# ============================================
WATCHDOG_WHATSAPP_ENABLED=false
WATCHDOG_TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WATCHDOG_TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WATCHDOG_WHATSAPP_RECIPIENTS=+41791234567,+41799876543

# ============================================
# Monitoring-Konfiguration
# ============================================
WATCHDOG_CHECK_INTERVAL_SECONDS=30
WATCHDOG_TIMEOUT_SECONDS=10
WATCHDOG_ALERT_COOLDOWN_MINUTES=15
WATCHDOG_ALERT_ON_RECOVERY=true
WATCHDOG_ALERT_ON_CRITICAL=true
WATCHDOG_ALERT_ON_HIGH=true
WATCHDOG_ALERT_ON_MEDIUM=false

# Tägliche Zusammenfassung
WATCHDOG_DAILY_SUMMARY_ENABLED=true
WATCHDOG_DAILY_SUMMARY_HOUR=8
```

## Telegram Bot Setup (Empfohlen)

### Schritt 1: Bot erstellen (2 Minuten)

1. **Öffne Telegram** und suche nach `@BotFather`
2. **Starte den Bot** mit `/start`
3. **Erstelle neuen Bot** mit `/newbot`
4. **Wähle einen Namen**: z.B. `KI Trading Watchdog`
5. **Wähle einen Username**: z.B. `ki_trading_watchdog_bot` (muss auf `_bot` enden)
6. **Kopiere den Token**: Sieht aus wie `7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

```bash
# Token in .env.watchdog eintragen
WATCHDOG_TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Schritt 2: Chat-ID ermitteln

#### Option A: Persönlicher Chat

1. Starte deinen Bot in Telegram (suche nach dem Bot-Namen und drücke `/start`)
2. Öffne im Browser: `https://api.telegram.org/bot<DEIN_TOKEN>/getUpdates`
3. Finde deine `chat.id` in der JSON-Antwort (z.B. `123456789`)

#### Option B: Gruppen-Chat

1. Erstelle eine Telegram-Gruppe
2. Füge den Bot zur Gruppe hinzu
3. Sende eine Nachricht in der Gruppe
4. Öffne: `https://api.telegram.org/bot<DEIN_TOKEN>/getUpdates`
5. Die Gruppen-ID beginnt mit `-` (z.B. `-100987654321`)

```bash
# Chat-IDs in .env.watchdog eintragen (kommagetrennt)
WATCHDOG_TELEGRAM_CHAT_IDS=123456789,-100987654321
```

### Schritt 3: Testen

```bash
# Test-Nachricht senden
curl -X POST "https://api.telegram.org/bot<TOKEN>/sendMessage" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "<CHAT_ID>", "text": "🧪 Watchdog Test!", "parse_mode": "Markdown"}'
```

### Bot-Befehle (Optional)

Für bessere UX kannst du Befehle beim BotFather registrieren:

```text
/setcommands
```

Dann die Befehle eingeben:

```text
status - Zeigt aktuellen System-Status
alerts - Zeigt letzte Alerts
test - Sendet Test-Alert
help - Zeigt Hilfe
```

---

## WhatsApp Setup (Optional - Twilio)

### 1. Twilio Sandbox (Entwicklung)

1. Twilio-Konto erstellen: https://www.twilio.com/try-twilio
2. WhatsApp Sandbox aktivieren: https://www.twilio.com/console/sms/whatsapp/sandbox
3. Empfänger registrieren: Nachricht "join <sandbox-code>" an die Sandbox-Nummer senden
4. Account SID und Auth Token aus der Console kopieren

### 2. Twilio Business (Produktion)

1. WhatsApp Business API beantragen
2. Telefonnummer registrieren und verifizieren
3. Message Templates für Alerts erstellen
4. Sender-Nummer in Konfiguration anpassen

## Erweiterte Features (Optional)

### 1. Telegram als Alternative

```python
# src/services/watchdog_app/services/telegram_notifier.py
import httpx

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_ids: list):
        self.bot_token = bot_token
        self.chat_ids = chat_ids
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    async def send_alert(self, message: str):
        async with httpx.AsyncClient() as client:
            for chat_id in self.chat_ids:
                await client.post(
                    f"{self.api_url}/sendMessage",
                    json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
                )
```

### 2. Metriken-Export (Prometheus)

```python
from prometheus_client import Counter, Gauge, Histogram

# Metriken definieren
health_check_duration = Histogram(
    'watchdog_health_check_duration_seconds',
    'Duration of health checks',
    ['service']
)
service_status = Gauge(
    'watchdog_service_status',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service']
)
alerts_total = Counter(
    'watchdog_alerts_total',
    'Total number of alerts sent',
    ['service', 'type']
)
```

### 3. Dashboard-Integration

Der Watchdog kann ins bestehende Frontend integriert werden:

```javascript
// System Health Widget
async function loadWatchdogStatus() {
    const response = await fetch('http://10.1.19.101:3010/api/v1/status');
    const data = await response.json();

    // Status-Übersicht rendern
    renderHealthGrid(data.services);
    renderAlertHistory(data.alerts);
}
```

## Deployment-Schritte

1. **Twilio einrichten**
   ```bash
   # Sandbox aktivieren und testen
   curl -X POST https://api.twilio.com/2010-04-01/Accounts/{SID}/Messages.json \
     -u "{SID}:{TOKEN}" \
     -d "From=whatsapp:+14155238886" \
     -d "To=whatsapp:+41791234567" \
     -d "Body=Test from Watchdog"
   ```

2. **Environment konfigurieren**
   ```bash
   cp .env.watchdog.example .env.watchdog
   # Twilio Credentials eintragen
   ```

3. **Container bauen und starten**
   ```bash
   docker-compose -f docker-compose.watchdog.yml build
   docker-compose -f docker-compose.watchdog.yml up -d
   ```

4. **Test-Alert senden**
   ```bash
   curl -X POST http://localhost:3010/api/v1/alerts/test
   ```

5. **Status prüfen**
   ```bash
   curl http://localhost:3010/api/v1/status
   ```

## Zusammenfassung

| Feature                  | Beschreibung                              |
|--------------------------|-------------------------------------------|
| **Health Monitoring**    | Alle 8 Services alle 30 Sekunden          |
| **Telegram Alerts**      | Kostenlos via Bot API (empfohlen)         |
| **WhatsApp Alerts**      | Optional via Twilio (kostenpflichtig)     |
| **Multi-Channel**        | Telegram + WhatsApp gleichzeitig möglich  |
| **Alert-Deduplizierung** | 15 Min Cooldown pro Service               |
| **Kritikalitätsstufen**  | Critical, High, Medium                    |
| **Recovery-Alerts**      | Benachrichtigung bei Wiederherstellung    |
| **Tägliche Summary**     | Automatischer Report um 08:00             |
| **API-Endpoints**        | Status, Historie, Test-Alert              |
| **Docker-Integration**   | Im trading-net Netzwerk                   |

## Quick Start

```bash
# 1. Telegram Bot erstellen bei @BotFather
# 2. .env.watchdog konfigurieren
cp .env.watchdog.example .env.watchdog

# 3. Token und Chat-ID eintragen
WATCHDOG_TELEGRAM_BOT_TOKEN=dein_token
WATCHDOG_TELEGRAM_CHAT_IDS=deine_chat_id

# 4. Container starten
docker-compose -f docker-compose.watchdog.yml up -d

# 5. Test-Alert senden
curl -X POST http://localhost:3010/api/v1/alerts/test/telegram
```
