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


@router.post("/services/{service_name}/simulate-failure")
async def simulate_service_failure(
    service_name: str,
    error_message: str = Query(default="Simulierter Ausfall (Test)", description="Fehlermeldung für den simulierten Ausfall")
):
    """
    Simuliert einen Service-Ausfall für Testzwecke.

    Dies löst einen UNHEALTHY-Status für den angegebenen Service aus,
    was einen Alert-Zyklus triggert (sofern Alerts für diesen Service aktiviert sind).

    Der simulierte Ausfall wird beim nächsten Health-Check automatisch korrigiert,
    sofern der echte Service gesund ist.
    """
    health_checker, alert_manager, _ = _get_services()

    if not health_checker:
        raise HTTPException(status_code=503, detail="Watchdog not initialized")

    if service_name not in health_checker.services:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' nicht gefunden")

    # Simuliere den Ausfall
    success = health_checker.simulate_failure(service_name, error_message)

    if not success:
        raise HTTPException(status_code=500, detail="Simulation fehlgeschlagen")

    # Trigger Alert-Verarbeitung
    if alert_manager:
        from ..models.service_status import ServiceStatus, HealthState
        status = health_checker.status.get(service_name)
        if status:
            criticality = health_checker.services[service_name]["criticality"]
            await alert_manager.process_status_change(service_name, status, criticality)

    return {
        "success": True,
        "service": service_name,
        "message": f"Ausfall für '{service_name}' simuliert",
        "error_message": error_message,
        "note": "Der Status wird beim nächsten Health-Check automatisch korrigiert"
    }


@router.get("/alerts/history")
async def get_alert_history(
    limit: int = Query(default=50, ge=1, le=100, description="Anzahl der zurückgegebenen Alerts (max 100)")
):
    """
    Gibt die Alert-Historie zurück (persistent gespeichert).

    Die letzten 100 Alerts werden persistent auf dem Volume gespeichert
    und überleben Container-Neustarts.
    """
    _, alert_manager, _ = _get_services()

    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    return {
        "alerts": alert_manager.get_alert_history(limit),
        "statistics": alert_manager.get_statistics()
    }


@router.delete("/alerts/history")
async def clear_alert_history():
    """
    Löscht die gesamte Alert-Historie.

    **Achtung**: Diese Aktion kann nicht rückgängig gemacht werden!
    """
    _, alert_manager, _ = _get_services()

    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")

    alert_manager.clear_alert_history()

    return {
        "success": True,
        "message": "Alert-Historie gelöscht"
    }


# ============ Health History API ============

def _get_health_history():
    """Lazy import für Health History Service."""
    from ..services.health_history import health_history
    return health_history


@router.get("/health/history", tags=["Health History"])
async def get_health_history(
    hours: int = Query(default=24, ge=1, le=168, description="Zeitraum in Stunden (max 168 / 1 Woche)"),
    limit: Optional[int] = Query(default=100, ge=1, le=5000, description="Max. Einträge"),
    aggregation: str = Query(default="raw", description="raw oder hourly")
):
    """
    Gibt die Health-Check-Historie der letzten Stunden zurück.

    Die Historie wird persistent gespeichert und überlebt Container-Neustarts.
    Der maximale Zeitraum entspricht der konfigurierten Retention-Zeit.

    ## Aggregation
    - **raw**: Alle einzelnen Health-Checks
    - **hourly**: Stündlich aggregierte Daten mit Uptime-Prozent
    """
    history_service = _get_health_history()

    # Begrenze auf konfigurierte Retention
    max_hours = history_service.get_retention_hours()
    hours = min(hours, max_hours)

    history = history_service.get_history(
        hours=hours,
        limit=limit,
        aggregation=aggregation
    )

    return {
        "period_hours": hours,
        "max_retention_hours": max_hours,
        "aggregation": aggregation,
        "count": len(history),
        "history": history
    }


@router.get("/health/history/statistics", tags=["Health History"])
async def get_health_statistics(
    hours: int = Query(default=24, ge=1, le=168, description="Zeitraum in Stunden")
):
    """
    Gibt Statistiken über die Health-Historie zurück.

    Enthält:
    - Uptime-Prozent
    - Durchschnittliche Anzahl healthy Services
    - Durchschnittliche Response-Zeit
    """
    history_service = _get_health_history()

    # Begrenze auf konfigurierte Retention
    max_hours = history_service.get_retention_hours()
    hours = min(hours, max_hours)

    return history_service.get_statistics(hours=hours)


@router.get("/health/history/service/{service_name}", tags=["Health History"])
async def get_service_health_history(
    service_name: str,
    hours: int = Query(default=24, ge=1, le=168, description="Zeitraum in Stunden")
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


@router.get("/health/history/config", tags=["Health History"])
async def get_health_history_config():
    """
    Gibt die aktuelle Konfiguration der Health-Historie zurück.

    Enthält:
    - retention_hours: Wie lange Daten gespeichert werden
    - max_entries: Maximale Anzahl Einträge
    - current_entries: Aktuelle Anzahl Einträge
    """
    history_service = _get_health_history()
    return history_service.get_config()


@router.put("/health/history/config", tags=["Health History"])
async def update_health_history_config(
    retention_hours: int = Query(..., ge=1, le=168, description="Retention-Zeit in Stunden (1-168, max 1 Woche)")
):
    """
    Aktualisiert die Retention-Zeit der Health-Historie.

    Die Retention-Zeit bestimmt, wie lange Health-Checks gespeichert werden.
    Gültige Werte: 1-168 Stunden (1 Stunde bis 1 Woche).

    **Hinweis**: Bei Reduzierung der Retention werden ältere Einträge sofort gelöscht.
    """
    history_service = _get_health_history()
    result = history_service.set_retention_hours(retention_hours)
    return {
        "success": True,
        "message": f"Retention auf {retention_hours} Stunden gesetzt",
        **result
    }


@router.delete("/health/history", tags=["Health History"])
async def clear_health_history():
    """
    Löscht die gesamte Health-Historie.

    **Achtung**: Diese Aktion kann nicht rückgängig gemacht werden!
    """
    history_service = _get_health_history()

    # Wir müssen eine clear-Methode hinzufügen
    with history_service._lock:
        old_count = len(history_service._history)
        history_service._history = []
        history_service._save_history()

    return {
        "success": True,
        "message": f"{old_count} Einträge gelöscht"
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


def _get_config_service():
    """Lazy import für Config Service."""
    from ..services.config_service import config_service
    return config_service


@router.get("/config", tags=["Konfiguration"])
async def get_config():
    """Gibt die aktuelle Konfiguration zurück (ohne sensible Daten)."""
    config_service = _get_config_service()
    config = config_service.get_config(include_sensitive=False)

    # Füge telegram_configured hinzu
    config["telegram_configured"] = bool(config_service.get("telegram_bot_token"))

    return config


@router.get("/config/telegram", tags=["Konfiguration"])
async def get_telegram_config():
    """
    Gibt die Telegram-Konfiguration zurück.

    Das Bot-Token wird maskiert angezeigt.
    """
    config_service = _get_config_service()
    telegram_config = config_service.get_telegram_config()

    # Maskiere Token für Anzeige
    if telegram_config.get("bot_token"):
        token = telegram_config["bot_token"]
        if len(token) > 10:
            telegram_config["bot_token_display"] = f"{token[:4]}...{token[-4:]}"
        else:
            telegram_config["bot_token_display"] = "***"
        del telegram_config["bot_token"]
    else:
        telegram_config["bot_token_display"] = ""

    return telegram_config


@router.put("/config/telegram", tags=["Konfiguration"])
async def update_telegram_config(
    enabled: Optional[bool] = Query(None, description="Telegram aktivieren/deaktivieren"),
    bot_token: Optional[str] = Query(None, description="Bot-Token von @BotFather"),
    chat_ids: Optional[str] = Query(None, description="Kommagetrennte Chat-IDs")
):
    """
    Aktualisiert die Telegram-Konfiguration.

    **Hinweis**: Nach Änderung des Tokens muss der Service neu gestartet werden,
    damit die Änderungen wirksam werden.

    ## Bot einrichten

    1. Erstelle einen Bot bei @BotFather auf Telegram
    2. Kopiere den Bot-Token
    3. Starte eine Konversation mit dem Bot
    4. Finde deine Chat-ID über @userinfobot

    ## Chat-IDs Format

    Mehrere Chat-IDs können kommagetrennt angegeben werden:
    - Einzelner User: `123456789`
    - Gruppe: `-100987654321`
    - Mehrere: `123456789,-100987654321`
    """
    config_service = _get_config_service()

    result = config_service.set_telegram_config(
        enabled=enabled,
        bot_token=bot_token,
        chat_ids=chat_ids
    )

    # Aktualisiere TelegramNotifier falls möglich
    if bot_token is not None or chat_ids is not None:
        try:
            from ..main import telegram_notifier
            if telegram_notifier:
                # Notifier muss neu initialisiert werden
                result["restart_required"] = True
                result["message"] = "Konfiguration gespeichert. Bitte Service neu starten für volle Aktivierung."
        except Exception:
            pass

    result["success"] = True
    return result


@router.get("/config/alerts", tags=["Konfiguration"])
async def get_alert_config():
    """Gibt die Alert-Konfiguration zurück."""
    config_service = _get_config_service()
    return config_service.get_alert_config()


@router.put("/config/alerts", tags=["Konfiguration"])
async def update_alert_config(
    cooldown_minutes: Optional[int] = Query(None, ge=1, le=1440, description="Cooldown zwischen Alerts in Minuten (1-1440)"),
    on_recovery: Optional[bool] = Query(None, description="Alert bei Service-Wiederherstellung"),
    on_critical: Optional[bool] = Query(None, description="Alert bei kritischen Services"),
    on_high: Optional[bool] = Query(None, description="Alert bei High-Priority Services"),
    on_medium: Optional[bool] = Query(None, description="Alert bei Medium-Priority Services")
):
    """
    Aktualisiert die Alert-Konfiguration.

    ## Kritikalitätsstufen

    - **critical**: Kern-Services (Data, Redis)
    - **high**: ML-Services (NHITS, TCN, HMM)
    - **medium**: Unterstützende Services (RAG, LLM, Embedder)

    Die Änderungen werden sofort wirksam und persistent gespeichert.
    """
    config_service = _get_config_service()

    result = config_service.set_alert_config(
        cooldown_minutes=cooldown_minutes,
        on_recovery=on_recovery,
        on_critical=on_critical,
        on_high=on_high,
        on_medium=on_medium
    )

    return {
        "success": True,
        "message": "Alert-Konfiguration aktualisiert",
        **result
    }


@router.get("/config/monitoring", tags=["Konfiguration"])
async def get_monitoring_config():
    """Gibt die Monitoring-Konfiguration zurück."""
    config_service = _get_config_service()
    return config_service.get_monitoring_config()


@router.put("/config/monitoring", tags=["Konfiguration"])
async def update_monitoring_config(
    check_interval_seconds: Optional[int] = Query(None, ge=10, le=300, description="Health-Check Intervall in Sekunden (10-300)"),
    timeout_seconds: Optional[int] = Query(None, ge=5, le=60, description="Timeout für Health-Checks in Sekunden (5-60)"),
    max_retries: Optional[int] = Query(None, ge=1, le=10, description="Max. aufeinanderfolgende Fehler (1-10)")
):
    """
    Aktualisiert die Monitoring-Konfiguration.

    **Hinweis**: Änderungen am Intervall werden erst nach Neustart wirksam.
    """
    config_service = _get_config_service()

    result = config_service.set_monitoring_config(
        check_interval_seconds=check_interval_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries
    )

    return {
        "success": True,
        "message": "Monitoring-Konfiguration aktualisiert",
        "restart_required": check_interval_seconds is not None,
        **result
    }


@router.get("/config/daily-summary", tags=["Konfiguration"])
async def get_daily_summary_config():
    """Gibt die Konfiguration für die tägliche Zusammenfassung zurück."""
    config_service = _get_config_service()
    return config_service.get_daily_summary_config()


@router.put("/config/daily-summary", tags=["Konfiguration"])
async def update_daily_summary_config(
    enabled: Optional[bool] = Query(None, description="Tägliche Zusammenfassung aktivieren/deaktivieren"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Stunde für Zusammenfassung (0-23, UTC)")
):
    """
    Aktualisiert die Konfiguration für die tägliche Zusammenfassung.
    """
    config_service = _get_config_service()

    result = config_service.set_daily_summary_config(
        enabled=enabled,
        hour=hour
    )

    return {
        "success": True,
        "message": "Daily Summary Konfiguration aktualisiert",
        **result
    }


@router.get("/config/service-alerts", tags=["Konfiguration"])
async def get_service_alert_config():
    """
    Gibt die Alarm-Konfiguration pro Service zurück.

    Zeigt für jeden überwachten Service, ob Telegram-Alerts aktiviert sind.
    Training-Services sind standardmässig deaktiviert.
    """
    config_service = _get_config_service()
    return config_service.get_service_alert_config()


@router.put("/config/service-alerts/{service_name}", tags=["Konfiguration"])
async def update_service_alert(
    service_name: str,
    enabled: bool = Query(..., description="True = Alarme aktiviert, False = deaktiviert")
):
    """
    Aktiviert oder deaktiviert Telegram-Alerts für einen bestimmten Service.

    ## Verfügbare Services

    **Inference Services (Standard: aktiviert):**
    - frontend, data, nhits, tcn, hmm, embedder, candlestick, redis, rag, llm

    **Training Services (Standard: deaktiviert):**
    - nhits-train, tcn-train, hmm-train, candlestick-train

    **External Services (Standard: aktiviert):**
    - easyinsight, twelvedata, yahoo
    """
    config_service = _get_config_service()

    # Prüfe ob Service existiert
    all_services = config_service._get_default_service_alerts().keys()
    if service_name not in all_services:
        raise HTTPException(
            status_code=404,
            detail=f"Unbekannter Service: {service_name}. Verfügbar: {list(all_services)}"
        )

    result = config_service.set_service_alert(service_name, enabled)

    return {
        "success": True,
        "message": f"Alerts für '{service_name}' {'aktiviert' if enabled else 'deaktiviert'}",
        **result
    }


class ServiceAlertsBulkRequest(BaseModel):
    """Request für Bulk-Update der Service-Alerts."""
    services: dict = Field(
        ...,
        description="Dict mit Service-Namen und deren Alarm-Status (True/False)",
        json_schema_extra={"example": {"nhits": True, "tcn": False, "nhits-train": False}}
    )


@router.put("/config/service-alerts", tags=["Konfiguration"])
async def update_service_alerts_bulk(request: ServiceAlertsBulkRequest):
    """
    Aktualisiert Telegram-Alerts für mehrere Services gleichzeitig.

    Nützlich für das Speichern aller Änderungen auf einmal aus dem Config-UI.
    """
    config_service = _get_config_service()

    # Validiere alle Service-Namen
    all_services = config_service._get_default_service_alerts().keys()
    invalid_services = [s for s in request.services.keys() if s not in all_services]
    if invalid_services:
        raise HTTPException(
            status_code=400,
            detail=f"Unbekannte Services: {invalid_services}. Verfügbar: {list(all_services)}"
        )

    result = config_service.set_service_alerts_bulk(request.services)

    return {
        "success": True,
        "message": f"{len(request.services)} Service-Alerts aktualisiert",
        **result
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
