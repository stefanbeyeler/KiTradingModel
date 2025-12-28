"""API Routes für den Watchdog Service.

Erweiterte Test-API mit verschiedenen Modi und Filtern.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["Watchdog"])


def _get_services():
    """Lazy import um zirkuläre Imports zu vermeiden."""
    from ..main import alert_manager, health_checker, telegram_notifier
    return health_checker, alert_manager, telegram_notifier


# ============ System Status API ============

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


# ============ Health History API ============

def _get_health_history():
    """Lazy import für Health History Service."""
    from ..services.health_history import health_history
    return health_history


@router.get("/health/history", tags=["Health History"])
async def get_health_history(
    hours: int = Query(default=24, ge=1, le=24, description="Zeitraum in Stunden (max 24)"),
    limit: Optional[int] = Query(default=100, ge=1, le=500, description="Max. Einträge"),
    aggregation: str = Query(default="raw", description="raw oder hourly")
):
    """
    Gibt die Health-Check-Historie der letzten Stunden zurück.

    Die Historie wird persistent gespeichert und überlebt Container-Neustarts.

    ## Aggregation
    - **raw**: Alle einzelnen Health-Checks
    - **hourly**: Stündlich aggregierte Daten mit Uptime-Prozent
    """
    history_service = _get_health_history()

    history = history_service.get_history(
        hours=hours,
        limit=limit,
        aggregation=aggregation
    )

    return {
        "period_hours": hours,
        "aggregation": aggregation,
        "count": len(history),
        "history": history
    }


@router.get("/health/history/statistics", tags=["Health History"])
async def get_health_statistics(
    hours: int = Query(default=24, ge=1, le=24, description="Zeitraum in Stunden")
):
    """
    Gibt Statistiken über die Health-Historie zurück.

    Enthält:
    - Uptime-Prozent
    - Durchschnittliche Anzahl healthy Services
    - Durchschnittliche Response-Zeit
    """
    history_service = _get_health_history()
    return history_service.get_statistics(hours=hours)


@router.get("/health/history/service/{service_name}", tags=["Health History"])
async def get_service_health_history(
    service_name: str,
    hours: int = Query(default=24, ge=1, le=24, description="Zeitraum in Stunden")
):
    """
    Gibt die Health-Historie für einen bestimmten Service zurück.
    """
    history_service = _get_health_history()
    history = history_service.get_service_history(service_name, hours=hours)

    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"Keine Historie für Service '{service_name}' gefunden"
        )

    return {
        "service": service_name,
        "period_hours": hours,
        "count": len(history),
        "history": history
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
    from ..services.test_runner import test_runner, RunMode
    return test_runner, RunMode


class TestRunRequest(BaseModel):
    """Request für Test-Lauf."""
    mode: str = Field(
        default="full",
        description="Test-Modus: smoke, api, contract, integration, critical, full"
    )
    services: Optional[List[str]] = Field(
        default=None,
        description="Filter auf bestimmte Services (z.B. ['data', 'nhits'])"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter auf Kategorien (smoke, api, contract, integration)"
    )
    priorities: Optional[List[str]] = Field(
        default=None,
        description="Filter auf Prioritäten (critical, high, medium, low)"
    )


@router.get("/tests/modes", tags=["Tests"])
async def get_test_modes():
    """
    Gibt alle verfügbaren Test-Modi zurück.

    Jeder Modus enthält:
    - mode: Modus-Bezeichnung
    - description: Beschreibung
    - tests: Anzahl der Tests in diesem Modus
    """
    test_runner, _ = _get_test_runner()
    return {
        "modes": test_runner.get_available_modes()
    }


@router.get("/tests/definitions", tags=["Tests"])
async def get_test_definitions(
    category: Optional[str] = Query(None, description="Filter nach Kategorie"),
    service: Optional[str] = Query(None, description="Filter nach Service")
):
    """
    Gibt alle Test-Definitionen zurück.

    Optional filterbar nach Kategorie und/oder Service.
    """
    test_runner, _ = _get_test_runner()

    tests = test_runner.get_test_definitions(category=category, service=service)

    return {
        "total": len(tests),
        "filters": {
            "category": category,
            "service": service
        },
        "tests": tests
    }


@router.post("/tests/run", tags=["Tests"])
async def start_test_run(
    request: TestRunRequest,
    background_tasks: BackgroundTasks
):
    """
    Startet einen neuen API-Test-Lauf.

    ## Modi

    - **smoke**: Schnelle Health-Checks (~10 Tests)
    - **api**: Alle API-Endpoint Tests (~60 Tests)
    - **contract**: Schema-Validierung (~10 Tests)
    - **integration**: Service-übergreifende Tests (~5 Tests)
    - **critical**: Nur kritische Tests (~20 Tests)
    - **full**: Alle Tests (~80 Tests)

    ## Filter

    - **services**: Nur Tests für bestimmte Services ausführen
    - **categories**: Nur Tests bestimmter Kategorien
    - **priorities**: Nur Tests mit bestimmten Prioritäten

    Der Test läuft im Hintergrund. Status via `/tests/status` abrufbar.
    """
    test_runner, RunMode = _get_test_runner()

    if test_runner.is_running():
        raise HTTPException(
            status_code=409,
            detail="A test run is already in progress"
        )

    # Modus validieren
    try:
        mode = RunMode(request.mode)
    except ValueError:
        valid_modes = [m.value for m in RunMode]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode '{request.mode}'. Valid modes: {valid_modes}"
        )

    # Test im Hintergrund starten
    background_tasks.add_task(
        test_runner.start_test_run,
        mode=mode,
        services=request.services,
        categories=request.categories,
        priorities=request.priorities
    )

    return {
        "message": "Test run started",
        "mode": request.mode,
        "filters": {
            "services": request.services,
            "categories": request.categories,
            "priorities": request.priorities
        },
        "status_url": "/api/v1/tests/status",
        "stream_url": "/api/v1/tests/stream"
    }


@router.post("/tests/run/smoke", tags=["Tests"])
async def start_smoke_tests(background_tasks: BackgroundTasks):
    """
    Shortcut: Startet Smoke-Tests (nur Health-Checks).

    Schnellster Test-Modus für grundlegende Verfügbarkeitsprüfung.
    """
    test_runner, RunMode = _get_test_runner()

    if test_runner.is_running():
        raise HTTPException(
            status_code=409,
            detail="A test run is already in progress"
        )

    background_tasks.add_task(test_runner.start_test_run, mode=RunMode.SMOKE)

    return {
        "message": "Smoke test run started",
        "mode": "smoke",
        "status_url": "/api/v1/tests/status"
    }


@router.post("/tests/run/critical", tags=["Tests"])
async def start_critical_tests(background_tasks: BackgroundTasks):
    """
    Shortcut: Startet nur kritische Tests.

    Prüft alle als kritisch markierten Endpoints.
    """
    test_runner, RunMode = _get_test_runner()

    if test_runner.is_running():
        raise HTTPException(
            status_code=409,
            detail="A test run is already in progress"
        )

    background_tasks.add_task(test_runner.start_test_run, mode=RunMode.CRITICAL)

    return {
        "message": "Critical test run started",
        "mode": "critical",
        "status_url": "/api/v1/tests/status"
    }


@router.post("/tests/run/full", tags=["Tests"])
async def start_full_tests(background_tasks: BackgroundTasks):
    """
    Shortcut: Startet alle verfügbaren Tests.

    Umfassende Test-Suite mit allen Kategorien.
    """
    test_runner, RunMode = _get_test_runner()

    if test_runner.is_running():
        raise HTTPException(
            status_code=409,
            detail="A test run is already in progress"
        )

    background_tasks.add_task(test_runner.start_test_run, mode=RunMode.FULL)

    return {
        "message": "Full test run started",
        "mode": "full",
        "status_url": "/api/v1/tests/status"
    }


@router.post("/tests/run/service/{service_name}", tags=["Tests"])
async def start_service_tests(
    service_name: str,
    background_tasks: BackgroundTasks
):
    """
    Startet Tests nur für einen bestimmten Service.

    Führt alle Tests (smoke, api, contract) für den angegebenen Service aus.
    """
    test_runner, RunMode = _get_test_runner()

    if test_runner.is_running():
        raise HTTPException(
            status_code=409,
            detail="A test run is already in progress"
        )

    # Prüfe ob Service existiert
    valid_services = list(test_runner.SERVICES.keys())
    if service_name not in valid_services:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown service '{service_name}'. Valid services: {valid_services}"
        )

    background_tasks.add_task(
        test_runner.start_test_run,
        mode=RunMode.FULL,
        services=[service_name]
    )

    return {
        "message": f"Test run for service '{service_name}' started",
        "service": service_name,
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
    - Fortschritt und Zusammenfassung
    """
    test_runner, _ = _get_test_runner()
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
    test_runner, _ = _get_test_runner()

    if not test_runner.is_running():
        raise HTTPException(
            status_code=400,
            detail="No test run in progress"
        )

    test_runner.abort_test_run()

    return {
        "message": "Test run abort requested"
    }


@router.get("/tests/history", tags=["Tests"])
async def get_test_history(
    limit: int = Query(default=10, ge=1, le=50, description="Anzahl der Einträge")
):
    """
    Gibt die Historie vergangener Test-Läufe zurück.

    Enthält die letzten n Test-Läufe mit vollständigen Ergebnissen.
    """
    test_runner, _ = _get_test_runner()

    return {
        "runs": test_runner.get_history(limit)
    }


@router.get("/tests/history/{run_id}", tags=["Tests"])
async def get_test_run_details(run_id: str):
    """
    Gibt Details eines spezifischen Test-Laufs zurück.
    """
    test_runner, _ = _get_test_runner()

    # Suche in Historie
    for run in test_runner.history:
        if run.id == run_id:
            return {
                "id": run.id,
                "mode": run.mode,
                "status": run.status,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "filters": run.filters,
                "summary": run.summary,
                "services": [
                    {
                        "name": s.name,
                        "display_name": s.display_name,
                        "healthy": s.healthy,
                        "response_time_ms": s.response_time_ms,
                        "error": s.error,
                        "version": s.version
                    }
                    for s in run.services
                ],
                "results": [
                    {
                        "name": r.name,
                        "status": r.status.value,
                        "service": r.service,
                        "category": r.category,
                        "priority": r.priority,
                        "endpoint": r.endpoint,
                        "method": r.method,
                        "duration_ms": r.duration_ms,
                        "response_status": r.response_status,
                        "message": r.message,
                        "error_type": r.error_type
                    }
                    for r in run.results
                ]
            }

    # Prüfe aktuellen Lauf
    if test_runner.current_run and test_runner.current_run.id == run_id:
        return test_runner.get_current_status()

    raise HTTPException(
        status_code=404,
        detail=f"Test run '{run_id}' not found"
    )


@router.get("/tests/summary", tags=["Tests"])
async def get_tests_summary():
    """
    Gibt eine Zusammenfassung der verfügbaren Tests zurück.

    Enthält Statistiken nach:
    - Kategorie
    - Service
    - Priorität
    """
    test_runner, _ = _get_test_runner()

    tests = test_runner.get_test_definitions()

    # Nach Kategorie
    by_category = {}
    for test in tests:
        cat = test["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    # Nach Service
    by_service = {}
    for test in tests:
        svc = test["service"]
        by_service[svc] = by_service.get(svc, 0) + 1

    # Nach Priorität
    by_priority = {}
    for test in tests:
        prio = test["priority"]
        by_priority[prio] = by_priority.get(prio, 0) + 1

    return {
        "total_tests": len(tests),
        "by_category": by_category,
        "by_service": by_service,
        "by_priority": by_priority,
        "available_modes": test_runner.get_available_modes()
    }


@router.get("/tests/stream", tags=["Tests"])
async def stream_test_status():
    """
    Server-Sent Events Stream für Live-Test-Updates.

    Sendet kontinuierlich den aktuellen Test-Status als SSE.
    Verbindung wird automatisch geschlossen wenn der Test abgeschlossen ist.

    ## Event-Typen

    - **data**: Regelmäßige Status-Updates
    - **complete**: Finales Update bei Test-Ende
    """
    test_runner, _ = _get_test_runner()

    async def event_generator():
        import json

        last_result_count = 0
        idle_count = 0

        while True:
            status = test_runner.get_current_status()

            if status:
                idle_count = 0

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

            else:
                # Keine aktiven Tests
                idle_count += 1
                if idle_count > 60:  # Nach 30 Sekunden abbrechen
                    yield f"event: timeout\ndata: {{\"message\": \"No active test run\"}}\n\n"
                    break

                yield f"data: {{\"status\": \"idle\", \"message\": \"Waiting for test run...\"}}\n\n"

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


# ============ Service Health Endpoints ============

@router.get("/tests/services", tags=["Tests"])
async def list_testable_services():
    """
    Gibt alle Services zurück, die getestet werden können.

    Enthält die Service-Konfiguration und Verfügbarkeitsstatus.
    """
    test_runner, _ = _get_test_runner()

    services = []
    for key, config in test_runner.SERVICES.items():
        test_count = len(test_runner.get_test_definitions(service=key))
        services.append({
            "name": key,
            "display_name": config["name"],
            "url": config["url"],
            "health_endpoint": config["health"],
            "test_count": test_count
        })

    return {
        "services": services,
        "total": len(services)
    }


@router.get("/tests/services/{service_name}/tests", tags=["Tests"])
async def get_service_tests(service_name: str):
    """
    Gibt alle Tests für einen bestimmten Service zurück.
    """
    test_runner, _ = _get_test_runner()

    # Prüfe ob Service existiert
    if service_name not in test_runner.SERVICES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown service '{service_name}'"
        )

    tests = test_runner.get_test_definitions(service=service_name)

    return {
        "service": service_name,
        "display_name": test_runner.SERVICES[service_name]["name"],
        "total": len(tests),
        "tests": tests
    }
