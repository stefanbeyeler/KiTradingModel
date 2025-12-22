"""
Shared Components for Microservices
====================================

Standardisierte Basis-Komponenten für alle Microservices:
- ServiceBase: Basis-Klasse für Service-Apps
- HealthCheck: Standardisierte Health-Responses
- ModelLoader: Basis für Model-Loading mit Timeout
- Logging: Einheitliche Logging-Konfiguration
"""

from .service_base import ServiceBase, create_service_app
from .health import HealthResponse, HealthStatus, create_health_response
from .model_loader import ModelLoaderBase
from .logging_config import setup_logging

__all__ = [
    "ServiceBase",
    "create_service_app",
    "HealthResponse",
    "HealthStatus",
    "create_health_response",
    "ModelLoaderBase",
    "setup_logging",
]
