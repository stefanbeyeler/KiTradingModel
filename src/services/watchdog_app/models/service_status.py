"""Datenmodelle für Service-Status."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class HealthState(str, Enum):
    """Health-Status eines Services."""

    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    STARTING = "STARTING"


class ServiceStatus(BaseModel):
    """Status eines einzelnen Services."""

    name: str
    state: HealthState
    response_time_ms: Optional[float] = None
    last_check: datetime
    error: Optional[str] = None
    consecutive_failures: int = 0
    details: Optional[Dict[str, Any]] = None


class SystemHealth(BaseModel):
    """Gesamtstatus des Systems."""

    timestamp: datetime
    overall_state: HealthState
    services: Dict[str, ServiceStatus]
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
