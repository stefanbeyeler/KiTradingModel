"""API Routes für den Watchdog Service."""

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

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


# ============ Test Runner API ============

def _get_test_runner():
    """Lazy import für Test Runner."""
    from ..services.test_runner import test_runner
    return test_runner


@router.post("/tests/run", tags=["Tests"])
async def start_test_run(background_tasks: BackgroundTasks):
    """
    Startet einen neuen API-Test-Lauf.

    Führt alle definierten Tests gegen die Microservices aus.
    Der Test läuft im Hintergrund - Status kann via /tests/status abgefragt werden.
    """
    test_runner = _get_test_runner()

    if test_runner._running:
        raise HTTPException(
            status_code=409,
            detail="A test run is already in progress"
        )

    # Start test run in background
    background_tasks.add_task(test_runner.start_test_run)

    return {
        "message": "Test run started",
        "status_url": "/api/v1/tests/status"
    }


@router.get("/tests/status", tags=["Tests"])
async def get_test_status():
    """
    Gibt den aktuellen Status des Test-Laufs zurück.

    Enthält:
    - Service Health Status
    - Bisherige Test-Ergebnisse
    - Aktuell laufender Test
    - Zusammenfassung
    """
    test_runner = _get_test_runner()
    status = test_runner.get_current_status()

    if not status:
        return {
            "status": "idle",
            "message": "No test run active or completed"
        }

    return status


@router.post("/tests/abort", tags=["Tests"])
async def abort_test_run():
    """Bricht den aktuellen Test-Lauf ab."""
    test_runner = _get_test_runner()

    if not test_runner._running:
        raise HTTPException(
            status_code=400,
            detail="No test run in progress"
        )

    test_runner.abort_test_run()

    return {
        "message": "Test run abort requested"
    }


@router.get("/tests/history", tags=["Tests"])
async def get_test_history(limit: int = 10):
    """Gibt die Historie vergangener Test-Läufe zurück."""
    test_runner = _get_test_runner()

    return {
        "runs": test_runner.get_history(limit)
    }


@router.get("/tests/stream", tags=["Tests"])
async def stream_test_status():
    """
    Server-Sent Events Stream für Live-Test-Updates.

    Sendet kontinuierlich den aktuellen Test-Status als SSE.
    Verbindung wird automatisch geschlossen wenn der Test abgeschlossen ist.
    """
    test_runner = _get_test_runner()

    async def event_generator():
        import json

        last_result_count = 0

        while True:
            status = test_runner.get_current_status()

            if status:
                # Send update
                yield f"data: {json.dumps(status)}\n\n"

                # Check if completed
                if status["status"] in ["completed", "error", "aborted"]:
                    yield f"event: complete\ndata: {json.dumps(status)}\n\n"
                    break

                # Track new results
                current_result_count = len(status.get("results", []))
                if current_result_count > last_result_count:
                    last_result_count = current_result_count

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
