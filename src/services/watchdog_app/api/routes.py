"""API Routes für den Watchdog Service."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Watchdog"])


def _get_services():
    """Lazy import um zirkuläre Imports zu vermeiden."""
    from ..main import alert_manager, health_checker, telegram_notifier
    return health_checker, alert_manager, telegram_notifier


@router.get("/status")
async def get_system_status():
    """Gibt den aktuellen System-Status zurück."""
    health_checker, alert_manager, _ = _get_services()

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


@router.get("/services")
async def list_monitored_services():
    """Liste aller überwachten Services."""
    health_checker, _, _ = _get_services()

    if not health_checker:
        raise HTTPException(status_code=503, detail="Watchdog not initialized")

    return {
        "services": [
            {
                "name": name,
                "url": config["url"],
                "criticality": config["criticality"],
                "startup_grace_seconds": config["startup_grace"],
                "dependencies": config["dependencies"]
            }
            for name, config in health_checker.services.items()
        ]
    }


@router.get("/alerts/history")
async def get_alert_history(limit: int = 50):
    """Gibt die Alert-Historie zurück."""
    _, alert_manager, _ = _get_services()

    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    return {
        "alerts": alert_manager.get_alert_history(limit),
        "statistics": alert_manager.get_statistics()
    }


@router.post("/alerts/test")
async def send_test_alert():
    """Sendet einen Test-Alert über Telegram."""
    _, _, telegram_notifier = _get_services()

    if not telegram_notifier:
        raise HTTPException(status_code=503, detail="Telegram notifier not initialized")

    result = await telegram_notifier.send_test_message()

    return {
        "success": len(result.get("success", [])) > 0,
        "telegram": result
    }


@router.get("/config")
async def get_config():
    """Gibt die aktuelle Konfiguration zurück (ohne sensible Daten)."""
    from ..config import settings

    return {
        "check_interval_seconds": settings.check_interval_seconds,
        "timeout_seconds": settings.timeout_seconds,
        "alert_cooldown_minutes": settings.alert_cooldown_minutes,
        "alert_on_recovery": settings.alert_on_recovery,
        "alert_on_critical": settings.alert_on_critical,
        "alert_on_high": settings.alert_on_high,
        "alert_on_medium": settings.alert_on_medium,
        "telegram_enabled": settings.telegram_enabled,
        "telegram_configured": bool(settings.telegram_bot_token),
        "daily_summary_enabled": settings.daily_summary_enabled,
        "daily_summary_hour": settings.daily_summary_hour
    }


@router.post("/check")
async def trigger_check():
    """Triggert einen manuellen Health-Check aller Services."""
    health_checker, _, _ = _get_services()

    if not health_checker:
        raise HTTPException(status_code=503, detail="Watchdog not initialized")

    await health_checker.check_all_services()

    return {
        "message": "Health check completed",
        "summary": health_checker.get_summary()
    }
