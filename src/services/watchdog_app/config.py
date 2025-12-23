"""Watchdog Service Konfiguration."""

from pydantic_settings import BaseSettings


class WatchdogSettings(BaseSettings):
    """Konfiguration für den Watchdog Service."""

    # Service-Konfiguration
    watchdog_port: int = 3010
    check_interval_seconds: int = 30
    timeout_seconds: int = 10
    max_retries: int = 3

    # Telegram Konfiguration
    telegram_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_ids: str = ""  # Kommagetrennt: "123456789,-100987654321"

    # Alert-Konfiguration
    alert_cooldown_minutes: int = 15
    alert_on_recovery: bool = True

    # Kritikalitätsstufen für Alarmierung
    alert_on_critical: bool = True
    alert_on_high: bool = True
    alert_on_medium: bool = False

    # Tägliche Zusammenfassung
    daily_summary_enabled: bool = True
    daily_summary_hour: int = 8  # 08:00 Uhr

    model_config = {
        "env_file": ".env.watchdog",
        "env_prefix": "WATCHDOG_",
        "extra": "ignore"
    }


settings = WatchdogSettings()
