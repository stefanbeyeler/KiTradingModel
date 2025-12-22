"""
Standardized Service Base
==========================

Basis-Komponenten für die Erstellung standardisierter Microservices.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Callable, Awaitable, Any, Dict
from fastapi import FastAPI
from loguru import logger

from .health import (
    HealthResponse,
    create_health_response,
    check_gpu_status,
    set_start_time,
)
from .logging_config import setup_logging, log_startup_info, log_shutdown_info
from .model_loader import ModelLoaderBase


class ServiceState:
    """
    Zentrale State-Verwaltung für einen Service.

    Speichert Model Loader, Readiness und Fehler.
    """

    def __init__(self):
        self.model_loader: Optional[ModelLoaderBase] = None
        self.is_ready: bool = False
        self.startup_error: Optional[str] = None
        self.extra_state: Dict[str, Any] = {}

    def set_ready(self) -> None:
        """Markiert den Service als bereit."""
        self.is_ready = True
        self.startup_error = None

    def set_error(self, error: str) -> None:
        """Setzt einen Startup-Fehler."""
        self.startup_error = error
        self.is_ready = False


class ServiceBase:
    """
    Basis-Klasse für Microservices.

    Kapselt gemeinsame Funktionalität:
    - Lifespan Management
    - Health Checks
    - Logging
    - Model Loading

    Beispiel:
        service = ServiceBase(
            name="embedder",
            display_name="Embedder Service",
            description="Central Embedding Service",
            version="2.0.0",
        )

        @service.on_startup
        async def startup():
            service.state.model_loader = EmbedderModelLoader()
            await service.state.model_loader.load()

        app = service.create_app()
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        version: str = "1.0.0",
        port: Optional[int] = None,
        root_path: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """
        Initialisiert die Service-Basis.

        Args:
            name: Interner Service-Name (z.B. 'embedder')
            display_name: Anzeigename (z.B. 'Embedder Service')
            description: Kurzbeschreibung
            version: Service-Version
            port: Port (aus Umgebung oder Default)
            root_path: FastAPI root_path für Proxy
            log_level: Log-Level
        """
        self.name = name
        self.display_name = display_name
        self.description = description
        self.version = version

        # Port aus Umgebung oder Parameter
        self.port = port or int(os.getenv("PORT", "3000"))
        self.root_path = root_path or os.getenv("ROOT_PATH", "")
        self.log_level = log_level or os.getenv("LOG_LEVEL", "INFO")

        # State
        self.state = ServiceState()

        # Callbacks
        self._startup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._shutdown_callbacks: list[Callable[[], Awaitable[None]]] = []

        # Logging konfigurieren
        setup_logging(name, self.log_level)

    def on_startup(self, func: Callable[[], Awaitable[None]]) -> Callable:
        """Decorator für Startup-Callbacks."""
        self._startup_callbacks.append(func)
        return func

    def on_shutdown(self, func: Callable[[], Awaitable[None]]) -> Callable:
        """Decorator für Shutdown-Callbacks."""
        self._shutdown_callbacks.append(func)
        return func

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """
        Lifespan Context Manager für FastAPI.

        Führt Startup und Shutdown mit korrektem Error-Handling durch.
        """
        # === STARTUP ===
        set_start_time()
        gpu_available, gpu_name = check_gpu_status()

        log_startup_info(
            service_name=self.name,
            version=self.version,
            port=self.port,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
        )

        try:
            # Startup Callbacks ausführen
            for callback in self._startup_callbacks:
                await callback()

            self.state.set_ready()
            logger.success(f"{self.display_name} is ready!")

        except asyncio.TimeoutError as e:
            error_msg = f"Startup timeout: {e}"
            self.state.set_error(error_msg)
            logger.error(error_msg)
            # Service startet trotzdem, aber nicht ready

        except Exception as e:
            error_msg = f"Startup failed: {e}"
            self.state.set_error(error_msg)
            logger.exception(error_msg)
            # Service startet trotzdem, aber nicht ready

        yield  # === SERVICE LÄUFT ===

        # === SHUTDOWN ===
        log_shutdown_info(self.name)

        try:
            # Model Loader cleanup
            if self.state.model_loader:
                await self.state.model_loader.cleanup()

            # Shutdown Callbacks ausführen
            for callback in self._shutdown_callbacks:
                await callback()

            logger.info("Shutdown completed successfully")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    def create_health_endpoint(self) -> Callable:
        """Erstellt den Health-Endpoint."""

        async def health_check() -> HealthResponse:
            """Standardisierter Health-Check."""
            gpu_available, gpu_name = check_gpu_status()

            # Model-Status
            model_loaded = False
            model_name = None
            if self.state.model_loader:
                status = self.state.model_loader.get_status()
                model_loaded = status.get("model_loaded", False)
                model_name = status.get("model_name")

            return create_health_response(
                service=self.name,
                version=self.version,
                is_ready=self.state.is_ready,
                error=self.state.startup_error,
                gpu_available=gpu_available,
                gpu_name=gpu_name,
                model_loaded=model_loaded,
                model_name=model_name,
                extras=self.state.extra_state,
            )

        return health_check

    def create_app(self, **fastapi_kwargs) -> FastAPI:
        """
        Erstellt die FastAPI-App mit Lifespan und Health-Endpoint.

        Args:
            **fastapi_kwargs: Zusätzliche FastAPI-Argumente

        Returns:
            Konfigurierte FastAPI-App
        """
        app = FastAPI(
            title=self.display_name,
            description=self.description,
            version=self.version,
            root_path=self.root_path,
            lifespan=self._lifespan,
            **fastapi_kwargs,
        )

        # Health Endpoint registrieren
        app.get("/health", response_model=HealthResponse, tags=["System"])(
            self.create_health_endpoint()
        )

        return app


def create_service_app(
    name: str,
    display_name: str,
    description: str,
    version: str = "1.0.0",
    startup_callback: Optional[Callable[[], Awaitable[None]]] = None,
    shutdown_callback: Optional[Callable[[], Awaitable[None]]] = None,
    **fastapi_kwargs,
) -> tuple[FastAPI, ServiceState]:
    """
    Factory-Funktion für schnelle Service-Erstellung.

    Args:
        name: Service-Name
        display_name: Anzeigename
        description: Beschreibung
        version: Version
        startup_callback: Optional startup callback
        shutdown_callback: Optional shutdown callback
        **fastapi_kwargs: Zusätzliche FastAPI-Argumente

    Returns:
        Tuple (FastAPI App, ServiceState)

    Beispiel:
        app, state = create_service_app(
            name="embedder",
            display_name="Embedder Service",
            description="Central Embedding Service",
            startup_callback=load_models,
        )
    """
    service = ServiceBase(
        name=name,
        display_name=display_name,
        description=description,
        version=version,
    )

    if startup_callback:
        service.on_startup(startup_callback)

    if shutdown_callback:
        service.on_shutdown(shutdown_callback)

    app = service.create_app(**fastapi_kwargs)

    return app, service.state
