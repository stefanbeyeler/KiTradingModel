"""Watchdog Service - Microservice Monitoring mit Telegram Alerts.

Extended with Training Orchestrator for centralized ML model training coordination.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from loguru import logger

from src.shared.logging_config import log_shutdown_info, log_startup_info, setup_logging
from src.shared.test_health_router import create_test_health_router
from src.shared.health import get_test_unhealthy_status

from .api.routes import router
from .api.training_routes import router as training_router
from .api.service_runs_routes import router as service_runs_router
from .config import settings
from .services.alert_manager import AlertManager
from .services.health_checker import HealthChecker
from .services.telegram_notifier import TelegramNotifier
from .services.training_orchestrator import training_orchestrator
from .services.resource_monitor import resource_monitor, ResourceMetrics

VERSION = "1.2.0"  # Added resource monitoring

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

    # Resource Monitor starten mit auto-pause Callback
    async def on_resource_alert(metrics: ResourceMetrics, level: str):
        """Callback bei kritischen Ressourcen - pausiert Training automatisch."""
        if level == "critical":
            if not training_orchestrator._paused:
                training_orchestrator.pause()
                logger.warning(
                    f"CRITICAL RESOURCES - Training paused automatically. "
                    f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%"
                )
                # Telegram-Alert senden
                if telegram_notifier and telegram_notifier.enabled:
                    await telegram_notifier.send_alert(
                        f"RESOURCE CRITICAL - Training paused\n"
                        f"CPU: {metrics.cpu_percent:.1f}%\n"
                        f"Memory: {metrics.memory_percent:.1f}%\n"
                        f"Queued jobs: {len(training_orchestrator._queue)}"
                    )

    # Resource Monitor konfigurieren und starten
    resource_monitor.cpu_warning = settings.resource_cpu_warning
    resource_monitor.cpu_critical = settings.resource_cpu_critical
    resource_monitor.memory_warning = settings.resource_memory_warning
    resource_monitor.memory_critical = settings.resource_memory_critical
    resource_monitor.poll_interval = settings.resource_poll_interval
    resource_monitor.gpu_metrics_service_url = settings.gpu_metrics_service_url
    resource_monitor.gpu_metrics_timeout = settings.gpu_metrics_timeout
    resource_monitor.register_callback(on_resource_alert)
    await resource_monitor.start()

    # Log GPU monitoring source
    if resource_monitor._nvml_initialized:
        logger.info("GPU monitoring: local (pynvml)")
    elif settings.gpu_metrics_service_url:
        logger.info(f"GPU monitoring: remote ({settings.gpu_metrics_service_url})")

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

    # Stop Resource Monitor
    await resource_monitor.stop()
    logger.info("Resource Monitor stopped")

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
app.include_router(service_runs_router, prefix="/api/v1")

# Test-Health-Router für alle Services (vom Watchdog aus steuerbar)
test_health_router = create_test_health_router("watchdog")
app.include_router(test_health_router, prefix="/api/v1", tags=["Testing"])


@app.get("/health")
async def health_check():
    """Health-Check für den Watchdog selbst."""
    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status("watchdog")
    is_unhealthy = test_status.get("test_unhealthy", False)

    summary = health_checker.get_summary() if health_checker else {}
    orchestrator_status = training_orchestrator.get_status()
    resource_status = resource_monitor.to_dict()

    response = {
        "service": "watchdog",
        "status": "unhealthy" if is_unhealthy else "healthy",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitoring_active": health_checker._running if health_checker else False,
        "services_monitored": len(health_checker.services) if health_checker else 0,
        "healthy_count": summary.get("healthy", 0),
        "unhealthy_count": summary.get("unhealthy", 0),
        "telegram_enabled": telegram_notifier.enabled if telegram_notifier else False,
        "resources": {
            "status": resource_status.get("status", "unknown"),
            "cpu_percent": resource_status.get("cpu_percent", 0),
            "memory_percent": resource_status.get("memory_percent", 0),
            "can_start_training": resource_status.get("can_start_training", True),
        },
        "training_orchestrator": {
            "running": orchestrator_status.get("running", False),
            "paused": orchestrator_status.get("paused", False),
            "queued_jobs": orchestrator_status.get("queued_jobs", 0),
            "running_jobs": orchestrator_status.get("running_jobs", 0)
        }
    }

    # Test-Status hinzufügen wenn aktiv
    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.watchdog_app.main:app",
        host="0.0.0.0",
        port=settings.watchdog_port,
        reload=False
    )
