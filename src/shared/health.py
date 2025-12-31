"""
Standardized Health Check Module
=================================

Einheitliches Health-Check Schema für alle Microservices.
Ermöglicht konsistente Überwachung und Dashboard-Integration.

Enthält auch Test-Health-Funktionalität für temporäres Unhealthy-Markieren.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio

# Globale Startzeit für Uptime-Berechnung
_service_start_time: Optional[datetime] = None

# Test-Health-Status: Service -> Unhealthy-Ablaufzeit
_test_unhealthy_until: Dict[str, datetime] = {}
_test_unhealthy_lock = asyncio.Lock()

# Standard-Dauer für Test-Unhealthy (5 Minuten)
DEFAULT_TEST_UNHEALTHY_DURATION_SECONDS = 300


def set_start_time() -> None:
    """Setzt die Service-Startzeit (beim Startup aufrufen)."""
    global _service_start_time
    _service_start_time = datetime.utcnow()


def get_uptime_seconds() -> float:
    """Berechnet die Uptime in Sekunden."""
    if _service_start_time is None:
        return 0.0
    return (datetime.utcnow() - _service_start_time).total_seconds()


async def set_test_unhealthy(service_name: str, duration_seconds: int = DEFAULT_TEST_UNHEALTHY_DURATION_SECONDS) -> datetime:
    """
    Markiert einen Service zu Testzwecken als unhealthy.

    Args:
        service_name: Name des Services
        duration_seconds: Dauer in Sekunden (Standard: 300 = 5 Minuten)

    Returns:
        Zeitpunkt, bis wann der Service als unhealthy markiert ist
    """
    async with _test_unhealthy_lock:
        unhealthy_until = datetime.utcnow() + timedelta(seconds=duration_seconds)
        _test_unhealthy_until[service_name] = unhealthy_until
        return unhealthy_until


async def clear_test_unhealthy(service_name: str) -> bool:
    """
    Entfernt die Test-Unhealthy-Markierung eines Services.

    Args:
        service_name: Name des Services

    Returns:
        True wenn Markierung entfernt wurde, False wenn keine vorhanden war
    """
    async with _test_unhealthy_lock:
        if service_name in _test_unhealthy_until:
            del _test_unhealthy_until[service_name]
            return True
        return False


def is_test_unhealthy(service_name: str) -> bool:
    """
    Prüft ob ein Service zu Testzwecken als unhealthy markiert ist.

    Args:
        service_name: Name des Services

    Returns:
        True wenn der Service noch als unhealthy markiert ist
    """
    if service_name not in _test_unhealthy_until:
        return False

    unhealthy_until = _test_unhealthy_until[service_name]
    if datetime.utcnow() >= unhealthy_until:
        # Markierung ist abgelaufen, entfernen
        try:
            del _test_unhealthy_until[service_name]
        except KeyError:
            pass
        return False

    return True


def get_test_unhealthy_remaining_seconds(service_name: str) -> Optional[float]:
    """
    Gibt die verbleibende Zeit der Test-Unhealthy-Markierung zurück.

    Args:
        service_name: Name des Services

    Returns:
        Verbleibende Sekunden oder None wenn nicht markiert
    """
    if service_name not in _test_unhealthy_until:
        return None

    unhealthy_until = _test_unhealthy_until[service_name]
    remaining = (unhealthy_until - datetime.utcnow()).total_seconds()

    if remaining <= 0:
        # Markierung ist abgelaufen
        try:
            del _test_unhealthy_until[service_name]
        except KeyError:
            pass
        return None

    return remaining


def get_test_unhealthy_status(service_name: str) -> Dict[str, Any]:
    """
    Gibt den vollständigen Test-Unhealthy-Status eines Services zurück.

    Args:
        service_name: Name des Services

    Returns:
        Dict mit Status-Informationen
    """
    if service_name not in _test_unhealthy_until:
        return {
            "test_unhealthy": False,
            "remaining_seconds": None,
            "unhealthy_until": None
        }

    unhealthy_until = _test_unhealthy_until[service_name]
    remaining = (unhealthy_until - datetime.utcnow()).total_seconds()

    if remaining <= 0:
        # Markierung ist abgelaufen
        try:
            del _test_unhealthy_until[service_name]
        except KeyError:
            pass
        return {
            "test_unhealthy": False,
            "remaining_seconds": None,
            "unhealthy_until": None
        }

    return {
        "test_unhealthy": True,
        "remaining_seconds": round(remaining, 1),
        "unhealthy_until": unhealthy_until.isoformat() + "Z"
    }


class HealthStatus(str, Enum):
    """Mögliche Health-Status-Werte."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"


class HealthResponse(BaseModel):
    """
    Standardisiertes Health-Response Schema.

    Alle Services sollten dieses Schema für /health verwenden.
    """

    # Pflichtfelder
    status: HealthStatus = Field(
        description="Service-Status: healthy, degraded, unhealthy, starting"
    )
    service: str = Field(
        description="Service-Name (z.B. 'nhits', 'embedder')"
    )
    version: str = Field(
        default="1.0.0",
        description="Service-Version"
    )

    # Zeitstempel
    timestamp: str = Field(
        description="ISO 8601 Zeitstempel der Antwort"
    )
    uptime_seconds: float = Field(
        default=0.0,
        description="Sekunden seit Service-Start"
    )

    # Readiness
    is_ready: bool = Field(
        default=True,
        description="True wenn Service Anfragen verarbeiten kann"
    )

    # Fehler
    error: Optional[str] = Field(
        default=None,
        description="Fehlermeldung falls vorhanden"
    )

    # Hardware
    gpu_available: Optional[bool] = Field(
        default=None,
        description="GPU verfügbar und nutzbar"
    )
    gpu_name: Optional[str] = Field(
        default=None,
        description="GPU-Name falls verfügbar"
    )

    # Model Status
    model_loaded: Optional[bool] = Field(
        default=None,
        description="ML-Modell geladen"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name des geladenen Modells"
    )

    # Dependencies
    dependencies: Optional[Dict[str, str]] = Field(
        default=None,
        description="Status der Abhängigkeiten (Service -> Status)"
    )

    # Service-spezifische Extras
    extras: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Zusätzliche service-spezifische Informationen"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "embedder",
                "version": "2.0.0",
                "timestamp": "2024-12-22T15:30:00Z",
                "uptime_seconds": 3600.5,
                "is_ready": True,
                "gpu_available": True,
                "gpu_name": "NVIDIA Orin",
                "model_loaded": True,
                "model_name": "all-MiniLM-L6-v2",
            }
        }


def create_health_response(
    service: str,
    version: str = "1.0.0",
    is_ready: bool = True,
    error: Optional[str] = None,
    gpu_available: Optional[bool] = None,
    gpu_name: Optional[str] = None,
    model_loaded: Optional[bool] = None,
    model_name: Optional[str] = None,
    dependencies: Optional[Dict[str, str]] = None,
    extras: Optional[Dict[str, Any]] = None,
    check_test_unhealthy: bool = True,
) -> HealthResponse:
    """
    Erstellt eine standardisierte Health-Response.

    Args:
        service: Service-Name
        version: Service-Version
        is_ready: True wenn Service bereit ist
        error: Fehlermeldung falls vorhanden
        gpu_available: GPU-Verfügbarkeit
        gpu_name: GPU-Name
        model_loaded: Model-Status
        model_name: Model-Name
        dependencies: Status der Abhängigkeiten
        extras: Zusätzliche Informationen
        check_test_unhealthy: Test-Unhealthy-Status prüfen (Standard: True)

    Returns:
        HealthResponse mit korrektem Status
    """
    # Test-Unhealthy-Status prüfen
    test_status = get_test_unhealthy_status(service) if check_test_unhealthy else None
    is_test_unhealthy_active = test_status and test_status.get("test_unhealthy", False)

    # Status bestimmen
    if is_test_unhealthy_active:
        status = HealthStatus.UNHEALTHY
        error = f"Test-Unhealthy aktiv (noch {test_status['remaining_seconds']}s)"
    elif error:
        status = HealthStatus.UNHEALTHY
    elif not is_ready:
        if get_uptime_seconds() < 60:
            status = HealthStatus.STARTING
        else:
            status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.HEALTHY

    # Test-Status zu extras hinzufügen wenn aktiv
    if is_test_unhealthy_active:
        if extras is None:
            extras = {}
        extras["test_unhealthy"] = test_status

    return HealthResponse(
        status=status,
        service=service,
        version=version,
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=round(get_uptime_seconds(), 2),
        is_ready=is_ready and not is_test_unhealthy_active,
        error=error,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        model_loaded=model_loaded,
        model_name=model_name,
        dependencies=dependencies,
        extras=extras,
    )


def check_gpu_status() -> tuple[bool, Optional[str]]:
    """
    Prüft GPU-Verfügbarkeit.

    Returns:
        Tuple (gpu_available, gpu_name)
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, None
    except ImportError:
        return False, None
    except Exception:
        return False, None
