"""Watchdog Service - Microservice Monitoring mit Telegram Alerts."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from loguru import logger

from src.shared.logging_config import log_shutdown_info, log_startup_info, setup_logging

from .api.routes import router
from .config import settings
from .services.alert_manager import AlertManager
from .services.health_checker import HealthChecker
from .services.telegram_notifier import TelegramNotifier

VERSION = "1.0.0"

# Global services
health_checker: HealthChecker | None = None
telegram_notifier: TelegramNotifier | None = None
alert_manager: AlertManager | None = None
monitoring_task: asyncio.Task | None = None


async def run_monitoring_with_alerts():
    """Kombiniert Health-Checks mit Alert-Verarbeitung."""
    global health_checker, alert_manager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management für den Watchdog Service."""
    global health_checker, telegram_notifier, alert_manager, monitoring_task

    setup_logging("watchdog")
    log_startup_info("watchdog", VERSION, settings.watchdog_port, gpu_available=False)

    # Services initialisieren
    health_checker = HealthChecker(settings)
    telegram_notifier = TelegramNotifier(
        bot_token=settings.telegram_bot_token,
        chat_ids=settings.telegram_chat_ids
    )
    alert_manager = AlertManager(settings, telegram_notifier)

    # Monitoring-Loop starten
    monitoring_task = asyncio.create_task(run_monitoring_with_alerts())

    logger.info(f"Monitoring {len(health_checker.services)} services")
    logger.info(
        f"Telegram: {'enabled' if telegram_notifier.enabled else 'disabled'} "
        f"({len(telegram_notifier.chat_ids)} recipients)"
    )
    logger.info(f"Check interval: {settings.check_interval_seconds}s")
    logger.info(f"Alert cooldown: {settings.alert_cooldown_minutes}min")

    yield

    # Shutdown
    if monitoring_task:
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

    if health_checker:
        health_checker.stop()

    log_shutdown_info("watchdog")


app = FastAPI(
    title="Watchdog Service",
    description="Überwacht alle Microservices und alarmiert per Telegram",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health-Check für den Watchdog selbst."""
    return {
        "service": "watchdog",
        "status": "healthy",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitoring_active": health_checker._running if health_checker else False,
        "services_monitored": len(health_checker.services) if health_checker else 0,
        "telegram_enabled": telegram_notifier.enabled if telegram_notifier else False
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.watchdog_app.main:app",
        host="0.0.0.0",
        port=settings.watchdog_port,
        reload=False
    )
