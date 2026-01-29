"""Watchdog Service Konfiguration."""

from pydantic_settings import BaseSettings

# Import zentrale Microservices-Konfiguration
from src.config.microservices import microservices_config


class WatchdogSettings(BaseSettings):
    """Konfiguration für den Watchdog Service."""

    # Service-Konfiguration
    watchdog_port: int = microservices_config.watchdog_service_port
    check_interval_seconds: int = 30
    timeout_seconds: int = 10
    max_retries: int = 3

    # Data Service Konfiguration (für Service Runs API)
    data_host: str = "trading-data"
    data_port: int = microservices_config.data_service_port

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

    # Health History Konfiguration
    history_retention_hours: int = 24  # Wie lange Historie gespeichert wird (1-168h / 1 Woche max)
    history_max_entries: int = 2880  # Max Einträge (24h * 60min / 0.5min bei 30s Intervall)

    # Resource Protection Konfiguration
    resource_cpu_warning: float = 75.0  # CPU-Warnschwelle (%) - Training wird verzögert
    resource_cpu_critical: float = 90.0  # CPU-Kritisch (%) - Training wird pausiert
    resource_memory_warning: float = 80.0  # Memory-Warnschwelle (%)
    resource_memory_critical: float = 90.0  # Memory-Kritisch (%)
    resource_poll_interval: float = 5.0  # Sekunden zwischen Resource-Checks
    max_concurrent_training: int = 2  # Max gleichzeitige Training-Jobs

    # GPU Monitoring via Remote Service (wenn kein lokaler GPU-Zugang)
    gpu_metrics_service_url: str = microservices_config.cnn_lstm_service_url  # Service mit GPU-Zugang
    gpu_metrics_timeout: float = 5.0  # Timeout für GPU-Metriken-Abfrage

    model_config = {
        "env_file": ".env.watchdog",
        "env_prefix": "WATCHDOG_",
        "extra": "ignore"
    }


settings = WatchdogSettings()
