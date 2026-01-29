"""Health Checker Service für Microservice-Überwachung mit GPU-Support."""

import asyncio
import os
import socket
from datetime import datetime, timezone
from typing import Dict, Optional

import httpx
from loguru import logger

from ..config import WatchdogSettings
from ..models.service_status import HealthState, ServiceStatus

# Import zentrale Microservices-Konfiguration
from src.config.microservices import microservices_config

# Data Service URL für externe Service-Checks (Gateway-Pattern)
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", microservices_config.data_service_url)

# GPU-basierte Services die erweiterte Health-Checks benötigen
GPU_SERVICES = {"nhits", "tcn", "embedder", "cnn-lstm", "rag", "llm"}
GPU_TRAIN_SERVICES = {"nhits-train", "tcn-train", "cnn-lstm-train"}


# Mapping von technischen Fehlermeldungen zu benutzerfreundlichen Texten
ERROR_MESSAGES = {
    # DNS/Name Resolution Errors
    "Temporary failure in name resolution": "Container nicht gestartet",
    "Name or service not known": "Container nicht gestartet",
    "nodename nor servname provided": "Container nicht gestartet",
    "getaddrinfo failed": "Container nicht gestartet",
    # Connection Errors
    "Connection refused": "Verbindung abgelehnt - Service nicht bereit",
    "Connection reset": "Verbindung unterbrochen",
    "Connection timed out": "Zeitüberschreitung bei Verbindungsaufbau",
    "No route to host": "Netzwerk nicht erreichbar",
    "Network is unreachable": "Netzwerk nicht erreichbar",
    # Timeout Errors
    "Timeout": "Zeitüberschreitung - Service antwortet nicht",
    "Read timed out": "Zeitüberschreitung beim Lesen",
    "timed out": "Zeitüberschreitung",
    # TCP Errors
    "not responding": "Port nicht erreichbar",
    # HTTP Errors
    "HTTP 500": "Interner Server-Fehler",
    "HTTP 502": "Bad Gateway - Backend nicht erreichbar",
    "HTTP 503": "Service nicht verfügbar",
    "HTTP 504": "Gateway Timeout",
}


def _translate_error(error: str) -> str:
    """
    Übersetzt technische Fehlermeldungen in benutzerfreundliche Texte.

    Args:
        error: Technische Fehlermeldung

    Returns:
        Benutzerfreundliche Fehlermeldung
    """
    if not error:
        return "Unbekannter Fehler"

    # Prüfe auf bekannte Fehlermuster
    error_lower = error.lower()
    for pattern, friendly_msg in ERROR_MESSAGES.items():
        if pattern.lower() in error_lower:
            return friendly_msg

    # Errno-Codes extrahieren und übersetzen
    if "[errno" in error_lower:
        if "errno -2]" in error_lower or "errno -3]" in error_lower:
            return "Container nicht gestartet"
        if "errno 111]" in error_lower:
            return "Verbindung abgelehnt - Service nicht bereit"
        if "errno 110]" in error_lower:
            return "Zeitüberschreitung bei Verbindungsaufbau"
        if "errno 113]" in error_lower:
            return "Netzwerk nicht erreichbar"

    # Fallback: Originalmeldung zurückgeben
    return error


class HealthChecker:
    """Prüft regelmässig den Health-Status aller Services."""

    def __init__(self, settings: WatchdogSettings):
        """
        Initialisiert den Health Checker.

        Args:
            settings: Watchdog-Konfiguration
        """
        self.settings = settings
        self.services = self._init_services()
        self.status: Dict[str, ServiceStatus] = {}
        self._running = False

    def _init_services(self) -> Dict[str, dict]:
        """Service-Registry mit Konfiguration (URLs aus zentraler Microservices-Config)."""
        return {
            "frontend": {
                "url": f"{microservices_config.get_service_url('frontend')}/health",
                "criticality": "medium",
                "startup_grace": 10,
                "dependencies": []
            },
            "data": {
                "url": f"{microservices_config.data_service_url}/health",
                "criticality": "critical",
                "startup_grace": 20,
                "dependencies": []
            },
            "nhits": {
                "url": f"{microservices_config.nhits_service_url}/health",
                "criticality": "high",
                "startup_grace": 40,
                "dependencies": ["data"],
                "gpu_service": True
            },
            "tcn": {
                "url": f"{microservices_config.tcn_service_url}/health",
                "criticality": "high",
                "startup_grace": 150,  # Erhöht: TCN hängt von Embedder ab (120s)
                "dependencies": ["data", "embedder"],
                "timeout": 20,  # Erhöht: GPU-Inference kann während hoher Last 14+ Sekunden dauern
                "gpu_service": True
            },
            "hmm": {
                "url": f"{microservices_config.hmm_service_url}/health",
                "criticality": "high",
                "startup_grace": 30,
                "dependencies": ["data"]
            },
            "embedder": {
                "url": f"{microservices_config.embedder_service_url}/health",
                "criticality": "high",
                "startup_grace": 120,
                "dependencies": ["data"]
            },
            "candlestick": {
                "url": f"{microservices_config.candlestick_service_url}/health",
                "criticality": "high",
                "startup_grace": 30,
                "dependencies": ["data"]
            },
            "cnn-lstm": {
                "url": f"{microservices_config.cnn_lstm_service_url}/health",
                "criticality": "high",
                "startup_grace": 40,
                "dependencies": ["data"],
                "timeout": 20,  # Erhöht: GPU-Inference kann während hoher Last länger dauern
                "gpu_service": True  # Flag für erweiterte GPU-Health-Checks
            },
            "redis": {
                "url": "http://trading-redis:6379",
                "criticality": "critical",
                "startup_grace": 10,
                "dependencies": [],
                "check_type": "tcp"
            },
            "nhits-train": {
                "url": f"{microservices_config.nhits_train_url}/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "tcn-train": {
                "url": f"{microservices_config.tcn_train_url}/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["data", "embedder"]
            },
            "hmm-train": {
                "url": f"{microservices_config.hmm_train_url}/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "candlestick-train": {
                "url": f"{microservices_config.candlestick_train_url}/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "cnn-lstm-train": {
                "url": f"{microservices_config.cnn_lstm_train_url}/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "rag": {
                "url": f"{microservices_config.rag_service_url}/health",
                "criticality": "high",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "llm": {
                "url": f"{microservices_config.llm_service_url}/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["rag"]
            },
            "workplace": {
                "url": f"{microservices_config.workplace_service_url}/health",
                "criticality": "medium",
                "startup_grace": 30,
                "dependencies": ["data", "nhits", "hmm", "candlestick"]
            },
            # Externe Datenquellen werden über den Data Service abgefragt (Gateway-Pattern)
            "easyinsight": {
                "url": f"{microservices_config.easyinsight_api_url.replace('/api', '')}/api/components/status",
                "criticality": "critical",
                "startup_grace": 10,
                "dependencies": [],
                "check_type": "easyinsight_components",
                "is_external": True
            },
            "twelvedata": {
                "url": "/api/v1/twelvedata/status",
                "criticality": "high",
                "startup_grace": 5,
                "dependencies": ["data"],
                "check_type": "twelvedata",
                "is_external": True
            },
            "yahoo": {
                "url": "/api/v1/yfinance/status",
                "criticality": "medium",
                "startup_grace": 5,
                "dependencies": ["data"],
                "check_type": "yahoo",
                "is_external": True
            }
        }

    async def check_service(self, name: str, config: dict) -> ServiceStatus:
        """
        Prüft einen einzelnen Service mit erweiterter GPU-Unterstützung.

        Args:
            name: Service-Name
            config: Service-Konfiguration

        Returns:
            ServiceStatus mit aktuellem Status
        """
        start_time = datetime.now(timezone.utc)

        # TCP-Check für Redis und andere non-HTTP Services
        if config.get("check_type") == "tcp":
            return await self._check_tcp_service(name, config, start_time)

        # EasyInsight Components Check (direkt über EasyInsight API)
        if config.get("check_type") == "easyinsight_components":
            return await self._check_easyinsight_components(name, config, start_time)

        # EasyInsight API Check (über Data Service) - Legacy
        if config.get("check_type") == "easyinsight":
            return await self._check_easyinsight(name, config, start_time)

        # TwelveData API Check (über Data Service)
        if config.get("check_type") == "twelvedata":
            return await self._check_twelvedata(name, config, start_time)

        # Yahoo Finance API Check (über Data Service)
        if config.get("check_type") == "yahoo":
            return await self._check_yahoo(name, config, start_time)

        try:
            # Service-spezifischer Timeout oder global setting
            timeout = config.get("timeout", self.settings.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(config["url"])
                response_time_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                if response.status_code == 200:
                    # Try to parse JSON, fall back to plain text
                    try:
                        data = response.json()
                        status = data.get("status", "healthy")
                    except Exception:
                        # Plain text response (e.g., nginx health endpoint)
                        text = response.text.strip().lower()
                        status = "healthy" if text == "healthy" else "unhealthy"
                        data = {"status": status, "raw": response.text.strip()}

                    # Check for GPU errors in response for GPU services
                    gpu_status = self._check_gpu_status_in_response(name, config, data)

                    # If GPU error detected, mark as degraded/unhealthy
                    if gpu_status:
                        return ServiceStatus(
                            name=name,
                            state=gpu_status["state"],
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={**data, "gpu_issue": gpu_status["message"]},
                            error=gpu_status["message"] if gpu_status["state"] == HealthState.UNHEALTHY else None,
                            consecutive_failures=0 if gpu_status["state"] != HealthState.UNHEALTHY else 1
                        )

                    return ServiceStatus(
                        name=name,
                        state=HealthState.HEALTHY if status == "healthy" else HealthState.DEGRADED,
                        response_time_ms=response_time_ms,
                        last_check=start_time,
                        details=data,
                        consecutive_failures=0
                    )
                else:
                    return self._create_failure_status(
                        name, start_time, f"HTTP {response.status_code}"
                    )

        except httpx.TimeoutException:
            return self._create_failure_status(name, start_time, "Timeout")
        except httpx.ConnectError:
            return self._create_failure_status(name, start_time, "Connection refused")
        except Exception as e:
            return self._create_failure_status(name, start_time, str(e))

    def _check_gpu_status_in_response(
        self, name: str, config: dict, response_data: dict
    ) -> Optional[dict]:
        """
        Prüft GPU-Status in der Health-Response eines GPU-Services.

        Args:
            name: Service-Name
            config: Service-Konfiguration
            response_data: JSON-Response vom Health-Endpoint

        Returns:
            Dict mit state und message bei GPU-Problemen, sonst None
        """
        # Nur für GPU-Services prüfen
        if not config.get("gpu_service") and name not in GPU_SERVICES:
            return None

        # Check for explicit GPU/CUDA errors in response
        gpu_info = response_data.get("gpu", {})
        cuda_info = response_data.get("cuda", {})

        # Check for CUDA error indicators
        cuda_error_patterns = [
            "illegal memory access",
            "cuda error",
            "out of memory",
            "device-side assert",
            "cublas",
            "cudnn error",
        ]

        # Check in error field
        error_field = str(response_data.get("error", "")).lower()
        for pattern in cuda_error_patterns:
            if pattern in error_field:
                logger.warning(f"CUDA error detected in {name}: {response_data.get('error')}")
                return {
                    "state": HealthState.UNHEALTHY,
                    "message": f"CUDA Error: {response_data.get('error')}"
                }

        # Check GPU health status if provided
        if gpu_info:
            if gpu_info.get("cuda_healthy") is False:
                cuda_error = gpu_info.get("cuda_error", "Unknown CUDA error")
                logger.warning(f"GPU unhealthy in {name}: {cuda_error}")
                return {
                    "state": HealthState.UNHEALTHY,
                    "message": f"GPU unhealthy: {cuda_error}"
                }

            # Check GPU memory usage (warning if > 90%)
            memory_percent = gpu_info.get("memory_percent", 0)
            if memory_percent > 95:
                logger.warning(f"Critical GPU memory usage in {name}: {memory_percent}%")
                return {
                    "state": HealthState.DEGRADED,
                    "message": f"GPU memory critical: {memory_percent}%"
                }
            elif memory_percent > 90:
                return {
                    "state": HealthState.DEGRADED,
                    "message": f"GPU memory high: {memory_percent}%"
                }

        # Check CUDA availability mismatch (expected GPU but running on CPU)
        device = response_data.get("device", "").lower()
        cuda_available = response_data.get("cuda_available", True)
        if cuda_available and device == "cpu" and name in GPU_SERVICES:
            return {
                "state": HealthState.DEGRADED,
                "message": "Running on CPU instead of GPU"
            }

        return None

    async def _check_tcp_service(
        self, name: str, config: dict, start_time: datetime
    ) -> ServiceStatus:
        """
        Prüft einen TCP-Service (z.B. Redis) durch Socket-Verbindung.

        Args:
            name: Service-Name
            config: Service-Konfiguration
            start_time: Zeitpunkt des Check-Starts

        Returns:
            ServiceStatus mit aktuellem Status
        """
        try:
            # Parse host and port from URL
            url = config["url"]
            if url.startswith("http://"):
                url = url[7:]
            elif url.startswith("https://"):
                url = url[8:]

            host, port_str = url.split(":")
            port = int(port_str.split("/")[0])  # Handle paths after port

            # Non-blocking socket check
            loop = asyncio.get_event_loop()

            def check_socket():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.settings.timeout_seconds)
                try:
                    result = sock.connect_ex((host, port))
                    return result == 0
                finally:
                    sock.close()

            is_up = await loop.run_in_executor(None, check_socket)
            response_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            if is_up:
                return ServiceStatus(
                    name=name,
                    state=HealthState.HEALTHY,
                    response_time_ms=response_time_ms,
                    last_check=start_time,
                    details={"type": "tcp", "port": port},
                    consecutive_failures=0
                )
            else:
                return self._create_failure_status(
                    name, start_time, f"TCP port {port} not responding"
                )

        except Exception as e:
            return self._create_failure_status(name, start_time, str(e))

    async def _check_easyinsight(
        self, name: str, config: dict, start_time: datetime
    ) -> ServiceStatus:
        """
        Prüft EasyInsight API-Verfügbarkeit über den Data Service.

        Args:
            name: Service-Name
            config: Service-Konfiguration
            start_time: Zeitpunkt des Check-Starts

        Returns:
            ServiceStatus mit aktuellem Status
        """
        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                response = await client.get(f"{DATA_SERVICE_URL}{config['url']}")
                response_time_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "disconnected")
                    is_available = data.get("available", False)

                    # Accept both "connected" and "healthy" as healthy states
                    if status in ("connected", "healthy") or is_available:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.HEALTHY,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "healthy",
                                "url": data.get("url", "N/A"),
                                "latency_ms": data.get("latency_ms", 0),
                                "symbols_available": data.get("symbols_available", 0)
                            },
                            consecutive_failures=0
                        )
                    elif status == "error":
                        return ServiceStatus(
                            name=name,
                            state=HealthState.DEGRADED,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "degraded",
                                "reason": data.get("error", "Unknown error"),
                                "url": data.get("url", "N/A")
                            },
                            consecutive_failures=0
                        )
                    else:  # disconnected
                        return self._create_failure_status(
                            name, start_time, data.get("error", "EasyInsight disconnected")
                        )
                else:
                    return self._create_failure_status(
                        name, start_time, f"Data Service returned HTTP {response.status_code}"
                    )

        except httpx.TimeoutException:
            return self._create_failure_status(name, start_time, "Timeout")
        except Exception as e:
            return self._create_failure_status(name, start_time, str(e))

    async def _check_twelvedata(
        self, name: str, config: dict, start_time: datetime
    ) -> ServiceStatus:
        """
        Prüft TwelveData API-Verfügbarkeit über den Data Service.

        Args:
            name: Service-Name
            config: Service-Konfiguration
            start_time: Zeitpunkt des Check-Starts

        Returns:
            ServiceStatus mit aktuellem Status
        """
        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                response = await client.get(f"{DATA_SERVICE_URL}{config['url']}")
                response_time_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                if response.status_code == 200:
                    data = response.json()
                    available = data.get("available", False)
                    api_key_configured = data.get("api_key_configured", False)

                    if not api_key_configured:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.DEGRADED,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "degraded",
                                "reason": "API key not configured in Data Service"
                            },
                            consecutive_failures=0
                        )

                    if available:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.HEALTHY,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "healthy",
                                "daily_used": data.get("daily_usage", "N/A"),
                                "plan_limit": data.get("plan_limit", "N/A"),
                                "credits_remaining": data.get("credits_remaining", "N/A")
                            },
                            consecutive_failures=0
                        )
                    else:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.DEGRADED,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "degraded",
                                "reason": data.get("error", "TwelveData not available")
                            },
                            consecutive_failures=0
                        )
                else:
                    return self._create_failure_status(
                        name, start_time, f"Data Service returned HTTP {response.status_code}"
                    )

        except httpx.TimeoutException:
            return self._create_failure_status(name, start_time, "Timeout")
        except Exception as e:
            return self._create_failure_status(name, start_time, str(e))

    async def _check_yahoo(
        self, name: str, config: dict, start_time: datetime
    ) -> ServiceStatus:
        """
        Prüft Yahoo Finance API-Verfügbarkeit über den Data Service.

        Args:
            name: Service-Name
            config: Service-Konfiguration
            start_time: Zeitpunkt des Check-Starts

        Returns:
            ServiceStatus mit aktuellem Status
        """
        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                response = await client.get(f"{DATA_SERVICE_URL}{config['url']}")
                response_time_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                if response.status_code == 200:
                    data = response.json()
                    available = data.get("available", False)

                    if available:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.HEALTHY,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "healthy",
                                "version": data.get("version", "N/A"),
                                "supported_symbols": data.get("supported_symbols", 0),
                                "categories": data.get("categories", [])
                            },
                            consecutive_failures=0
                        )
                    else:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.DEGRADED,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "degraded",
                                "reason": "yfinance not installed or unavailable"
                            },
                            consecutive_failures=0
                        )
                else:
                    return self._create_failure_status(
                        name, start_time, f"Data Service returned HTTP {response.status_code}"
                    )

        except httpx.TimeoutException:
            return self._create_failure_status(name, start_time, "Timeout")
        except Exception as e:
            return self._create_failure_status(name, start_time, str(e))

    def _create_failure_status(
        self, name: str, check_time: datetime, error: str
    ) -> ServiceStatus:
        """Erstellt einen Fehler-Status mit benutzerfreundlicher Fehlermeldung."""
        prev_status = self.status.get(name)
        consecutive = (prev_status.consecutive_failures + 1) if prev_status else 1

        # Übersetze technische Fehlermeldung
        friendly_error = _translate_error(error)

        return ServiceStatus(
            name=name,
            state=HealthState.UNHEALTHY,
            response_time_ms=None,
            last_check=check_time,
            error=friendly_error,
            consecutive_failures=consecutive
        )

    async def check_all_services(self) -> Dict[str, ServiceStatus]:
        """
        Prüft alle Services parallel.

        Returns:
            Dict mit Service-Namen und deren Status
        """
        tasks = [
            self.check_service(name, config)
            for name, config in self.services.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, _) in enumerate(self.services.items()):
            if isinstance(results[i], Exception):
                self.status[name] = self._create_failure_status(
                    name, datetime.now(timezone.utc), str(results[i])
                )
            else:
                self.status[name] = results[i]

        return self.status

    async def run_monitoring_loop(self):
        """Hauptschleife für kontinuierliches Monitoring."""
        self._running = True
        logger.info("Health monitoring loop started")

        # Import Health History Service
        try:
            from .health_history import health_history
            self._health_history = health_history
            logger.info("Health History Service aktiviert")
        except Exception as e:
            logger.warning(f"Health History Service nicht verfügbar: {e}")
            self._health_history = None

        while self._running:
            try:
                await self.check_all_services()
                healthy = sum(
                    1 for s in self.status.values()
                    if s.state == HealthState.HEALTHY
                )
                logger.debug(
                    f"Health check completed: {healthy}/{len(self.status)} healthy"
                )

                # Speichere in Historie
                if self._health_history and self.status:
                    try:
                        self._health_history.add_check_result(self.status)
                    except Exception as e:
                        logger.warning(f"Fehler beim Speichern der Historie: {e}")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            await asyncio.sleep(self.settings.check_interval_seconds)

    def stop(self):
        """Stoppt die Monitoring-Schleife."""
        self._running = False
        logger.info("Health monitoring loop stopped")

    def simulate_failure(self, service_name: str, error_message: str = "Simulierter Ausfall (Test)") -> bool:
        """
        Simuliert einen Service-Ausfall für Testzwecke.

        Args:
            service_name: Name des Services
            error_message: Fehlermeldung für den simulierten Ausfall

        Returns:
            True wenn erfolgreich, False wenn Service nicht gefunden
        """
        if service_name not in self.services:
            logger.warning(f"Service {service_name} nicht gefunden für Simulation")
            return False

        # Erstelle einen Failure-Status für den Service
        self.status[service_name] = ServiceStatus(
            name=service_name,
            state=HealthState.UNHEALTHY,
            response_time_ms=None,
            last_check=datetime.now(timezone.utc),
            error=error_message,
            consecutive_failures=1
        )

        logger.info(f"Simulierter Ausfall für Service {service_name}: {error_message}")
        return True

    def get_summary(self) -> dict:
        """Gibt eine Zusammenfassung des aktuellen Status zurück inkl. GPU-Status."""
        healthy = sum(
            1 for s in self.status.values() if s.state == HealthState.HEALTHY
        )
        degraded = sum(
            1 for s in self.status.values() if s.state == HealthState.DEGRADED
        )
        unhealthy = sum(
            1 for s in self.status.values() if s.state == HealthState.UNHEALTHY
        )

        # Durchschnittliche Response-Zeit berechnen
        response_times = [
            s.response_time_ms for s in self.status.values()
            if s.response_time_ms is not None
        ]
        avg_response_ms = sum(response_times) / len(response_times) if response_times else 0

        # GPU-Services Status
        gpu_services_status = {}
        for name in GPU_SERVICES | GPU_TRAIN_SERVICES:
            if name in self.status:
                status = self.status[name]
                details = status.details or {}
                gpu_services_status[name] = {
                    "state": status.state.value,
                    "gpu_issue": details.get("gpu_issue"),
                    "device": details.get("device"),
                    "cuda_healthy": details.get("gpu", {}).get("cuda_healthy") if details.get("gpu") else None,
                }

        # Count GPU-specific issues
        gpu_issues = sum(
            1 for name, info in gpu_services_status.items()
            if info.get("gpu_issue") is not None
        )

        return {
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "total": len(self.status),
            "avg_response_ms": avg_response_ms,
            "gpu_services": gpu_services_status,
            "gpu_issues_count": gpu_issues,
        }

    async def _check_easyinsight_components(
        self, name: str, config: dict, start_time: datetime
    ) -> ServiceStatus:
        """
        Prüft EasyInsight-Komponenten direkt über /api/components/status.

        Args:
            name: Service-Name
            config: Service-Konfiguration
            start_time: Zeitpunkt des Check-Starts

        Returns:
            ServiceStatus mit aktuellem Status
        """
        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                response = await client.get(config['url'])
                response_time_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                if response.status_code == 200:
                    components = response.json()

                    # Prüfe alle Komponenten
                    all_healthy = all(
                        comp.get("health") == "healthy" and comp.get("running", False)
                        for comp in components
                    )

                    unhealthy_components = [
                        comp.get("display_name", comp.get("name", "Unknown"))
                        for comp in components
                        if comp.get("health") != "healthy" or not comp.get("running", False)
                    ]

                    if all_healthy:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.HEALTHY,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "healthy",
                                "components": len(components),
                                "all_healthy": True
                            },
                            consecutive_failures=0
                        )
                    else:
                        return ServiceStatus(
                            name=name,
                            state=HealthState.DEGRADED,
                            response_time_ms=response_time_ms,
                            last_check=start_time,
                            details={
                                "status": "degraded",
                                "components": len(components),
                                "unhealthy": unhealthy_components
                            },
                            consecutive_failures=0
                        )
                else:
                    return ServiceStatus(
                        name=name,
                        state=HealthState.UNHEALTHY,
                        response_time_ms=response_time_ms,
                        last_check=start_time,
                        error=f"HTTP {response.status_code}",
                        consecutive_failures=self.status.get(name, ServiceStatus(
                            name=name, state=HealthState.UNHEALTHY, last_check=start_time
                        )).consecutive_failures + 1
                    )
        except Exception as e:
            error_msg = str(e)
            return ServiceStatus(
                name=name,
                state=HealthState.UNHEALTHY,
                response_time_ms=0,
                last_check=start_time,
                error=_translate_error(error_msg),
                consecutive_failures=self.status.get(name, ServiceStatus(
                    name=name, state=HealthState.UNHEALTHY, last_check=start_time
                )).consecutive_failures + 1
            )
