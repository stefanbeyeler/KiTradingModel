"""Watchdog Service - Microservice Monitoring mit Telegram Alerts.

Extended with Training Orchestrator for centralized ML model training coordination.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from loguru import logger

from src.shared.logging_config import log_shutdown_info, log_startup_info, setup_logging

from .api.routes import router
from .api.training_routes import router as training_router
from .config import settings
from .services.alert_manager import AlertManager
from .services.health_checker import HealthChecker
from .services.telegram_notifier import TelegramNotifier
from .services.training_orchestrator import training_orchestrator

VERSION = "1.1.0"

# Global services
health_checker: HealthChecker | None = None
telegram_notifier: TelegramNotifier | None = None
alert_manager: AlertManager | None = None
monitoring_task: asyncio.Task | None = None


async def run_monitoring_with_alerts():
    """Kombiniert Health-Checks mit Alert-Verarbeitung."""
    global health_checker, alert_manager

    # Health History Service initialisieren
    health_history = None
    try:
        from .services.health_history import health_history as hh_service
        health_history = hh_service
        logger.info("Health History Service für Monitoring aktiviert")
    except Exception as e:
        logger.warning(f"Health History Service nicht verfügbar: {e}")

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

            # Health-Check in Historie speichern
            if health_history and health_checker.status:
                try:
                    health_history.add_check_result(health_checker.status)
                except Exception as e:
                    logger.warning(f"Fehler beim Speichern der Health-Historie: {e}")

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

        await asyncio.sleep(settings.check_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management für den Watchdog Service."""
    global health_checker, telegram_notifier, alert_manager, monitoring_task

    setup_logging("watchdog")
    log_startup_info("watchdog", VERSION, settings.watchdog_port, gpu_available=False)

    # Lade persistente Konfiguration
    try:
        from .services.config_service import config_service
        telegram_token = config_service.get("telegram_bot_token") or settings.telegram_bot_token
        telegram_chat_ids = config_service.get("telegram_chat_ids") or settings.telegram_chat_ids
        logger.info("Persistente Konfiguration geladen")
    except Exception as e:
        logger.warning(f"Konnte persistente Konfiguration nicht laden: {e}")
        telegram_token = settings.telegram_bot_token
        telegram_chat_ids = settings.telegram_chat_ids

    # Services initialisieren
    health_checker = HealthChecker(settings)
    telegram_notifier = TelegramNotifier(
        bot_token=telegram_token,
        chat_ids=telegram_chat_ids
    )
    alert_manager = AlertManager(settings, telegram_notifier)

    # Monitoring-Loop starten
    monitoring_task = asyncio.create_task(run_monitoring_with_alerts())

    # Training Orchestrator starten
    await training_orchestrator.start()

    logger.info(f"Monitoring {len(health_checker.services)} services")
    logger.info(
        f"Telegram: {'enabled' if telegram_notifier.enabled else 'disabled'} "
        f"({len(telegram_notifier.chat_ids)} recipients)"
    )
    logger.info(f"Check interval: {settings.check_interval_seconds}s")
    logger.info(f"Alert cooldown: {settings.alert_cooldown_minutes}min")
    logger.info("Training Orchestrator started")

    yield

    # Shutdown
    if monitoring_task:
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

    # Stop Training Orchestrator
    await training_orchestrator.stop()
    logger.info("Training Orchestrator stopped")

    if health_checker:
        health_checker.stop()

    log_shutdown_info("watchdog")


app = FastAPI(
    title="Watchdog Service",
    description="Überwacht alle Microservices und alarmiert per Telegram. Inkludiert Training Orchestrator für ML-Modell-Training.",
    version=VERSION,
    lifespan=lifespan,
    root_path="/watchdog",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(router, prefix="/api/v1")
app.include_router(training_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health-Check für den Watchdog selbst."""
    summary = health_checker.get_summary() if health_checker else {}
    orchestrator_status = training_orchestrator.get_status()
    return {
        "service": "watchdog",
        "status": "healthy",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitoring_active": health_checker._running if health_checker else False,
        "services_monitored": len(health_checker.services) if health_checker else 0,
        "healthy_count": summary.get("healthy", 0),
        "unhealthy_count": summary.get("unhealthy", 0),
        "telegram_enabled": telegram_notifier.enabled if telegram_notifier else False,
        "training_orchestrator": {
            "running": orchestrator_status.get("running", False),
            "queued_jobs": orchestrator_status.get("queued_jobs", 0),
            "running_jobs": orchestrator_status.get("running_jobs", 0)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.watchdog_app.main:app",
        host="0.0.0.0",
        port=settings.watchdog_port,
        reload=False
    )
