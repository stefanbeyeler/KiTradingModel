"""
Test Health Router
==================

Gemeinsamer Router für Test-Health-Endpoints.
Ermöglicht das temporäre Markieren eines Services als unhealthy zu Testzwecken.

Verwendung:
    from src.shared.test_health_router import create_test_health_router

    # In main.py oder routers/__init__.py
    test_health_router = create_test_health_router("service-name")
    app.include_router(test_health_router, prefix="/api/v1", tags=["Testing"])
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from .health import (
    set_test_unhealthy,
    clear_test_unhealthy,
    get_test_unhealthy_status,
    is_test_unhealthy,
    DEFAULT_TEST_UNHEALTHY_DURATION_SECONDS,
)


class TestUnhealthyResponse(BaseModel):
    """Response für Test-Unhealthy-Endpoint."""
    service: str = Field(description="Service-Name")
    test_unhealthy: bool = Field(description="Ob Service als unhealthy markiert ist")
    remaining_seconds: Optional[float] = Field(None, description="Verbleibende Sekunden")
    unhealthy_until: Optional[str] = Field(None, description="Zeitpunkt bis unhealthy")
    message: str = Field(description="Status-Nachricht")


class TestUnhealthyRequest(BaseModel):
    """Request für Test-Unhealthy-Aktivierung."""
    duration_seconds: int = Field(
        default=DEFAULT_TEST_UNHEALTHY_DURATION_SECONDS,
        ge=10,
        le=3600,
        description="Dauer in Sekunden (10-3600, Standard: 300 = 5 Minuten)"
    )


def create_test_health_router(service_name: str) -> APIRouter:
    """
    Erstellt einen Router mit Test-Health-Endpoints für den angegebenen Service.

    Args:
        service_name: Name des Services (z.B. "data", "nhits", "tcn")

    Returns:
        APIRouter mit Test-Health-Endpoints
    """
    router = APIRouter()

    @router.post(
        "/test-unhealthy",
        response_model=TestUnhealthyResponse,
        summary="Service als unhealthy markieren (Test)",
        description=f"""
Markiert den {service_name}-Service zu Testzwecken als unhealthy.

**Zweck:**
- Testen von Alerting und Monitoring
- Überprüfen der Watchdog-Integration
- Validieren von Telegram-Benachrichtigungen

**Hinweise:**
- Standard-Dauer: 5 Minuten (300 Sekunden)
- Maximale Dauer: 1 Stunde (3600 Sekunden)
- Der Service bleibt funktionsfähig, wird aber als unhealthy gemeldet
- Kann vorzeitig mit DELETE /test-unhealthy zurückgesetzt werden
        """
    )
    async def activate_test_unhealthy(
        request: TestUnhealthyRequest = TestUnhealthyRequest()
    ) -> TestUnhealthyResponse:
        """Aktiviert den Test-Unhealthy-Status."""
        unhealthy_until = await set_test_unhealthy(service_name, request.duration_seconds)
        status = get_test_unhealthy_status(service_name)

        logger.warning(
            f"Test-Unhealthy aktiviert für {service_name}: "
            f"{request.duration_seconds}s bis {unhealthy_until.isoformat()}"
        )

        return TestUnhealthyResponse(
            service=service_name,
            test_unhealthy=True,
            remaining_seconds=status.get("remaining_seconds"),
            unhealthy_until=status.get("unhealthy_until"),
            message=f"Service '{service_name}' für {request.duration_seconds} Sekunden als unhealthy markiert"
        )

    @router.delete(
        "/test-unhealthy",
        response_model=TestUnhealthyResponse,
        summary="Test-Unhealthy-Status zurücksetzen",
        description=f"Entfernt die Test-Unhealthy-Markierung des {service_name}-Services vorzeitig."
    )
    async def deactivate_test_unhealthy() -> TestUnhealthyResponse:
        """Deaktiviert den Test-Unhealthy-Status."""
        was_active = await clear_test_unhealthy(service_name)

        if was_active:
            logger.info(f"Test-Unhealthy deaktiviert für {service_name}")
            message = f"Test-Unhealthy-Status für '{service_name}' zurückgesetzt"
        else:
            message = f"Kein aktiver Test-Unhealthy-Status für '{service_name}'"

        return TestUnhealthyResponse(
            service=service_name,
            test_unhealthy=False,
            remaining_seconds=None,
            unhealthy_until=None,
            message=message
        )

    @router.get(
        "/test-unhealthy",
        response_model=TestUnhealthyResponse,
        summary="Test-Unhealthy-Status abfragen",
        description=f"Gibt den aktuellen Test-Unhealthy-Status des {service_name}-Services zurück."
    )
    async def get_test_unhealthy_endpoint() -> TestUnhealthyResponse:
        """Gibt den aktuellen Test-Unhealthy-Status zurück."""
        status = get_test_unhealthy_status(service_name)

        return TestUnhealthyResponse(
            service=service_name,
            test_unhealthy=status.get("test_unhealthy", False),
            remaining_seconds=status.get("remaining_seconds"),
            unhealthy_until=status.get("unhealthy_until"),
            message=(
                f"Test-Unhealthy aktiv (noch {status.get('remaining_seconds')}s)"
                if status.get("test_unhealthy")
                else "Kein aktiver Test-Unhealthy-Status"
            )
        )

    return router
