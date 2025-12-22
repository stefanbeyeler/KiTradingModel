"""
Standardized Health Check Module
=================================

Einheitliches Health-Check Schema für alle Microservices.
Ermöglicht konsistente Überwachung und Dashboard-Integration.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Globale Startzeit für Uptime-Berechnung
_service_start_time: Optional[datetime] = None


def set_start_time() -> None:
    """Setzt die Service-Startzeit (beim Startup aufrufen)."""
    global _service_start_time
    _service_start_time = datetime.utcnow()


def get_uptime_seconds() -> float:
    """Berechnet die Uptime in Sekunden."""
    if _service_start_time is None:
        return 0.0
    return (datetime.utcnow() - _service_start_time).total_seconds()


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

    Returns:
        HealthResponse mit korrektem Status
    """
    # Status bestimmen
    if error:
        status = HealthStatus.UNHEALTHY
    elif not is_ready:
        if get_uptime_seconds() < 60:
            status = HealthStatus.STARTING
        else:
            status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.HEALTHY

    return HealthResponse(
        status=status,
        service=service,
        version=version,
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=round(get_uptime_seconds(), 2),
        is_ready=is_ready,
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
