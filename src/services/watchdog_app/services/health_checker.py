"""Health Checker Service für Microservice-Überwachung."""

import asyncio
from datetime import datetime, timezone
from typing import Dict

import httpx
from loguru import logger

from ..config import WatchdogSettings
from ..models.service_status import HealthState, ServiceStatus


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
        """Service-Registry mit Konfiguration."""
        return {
            "frontend": {
                "url": "http://trading-frontend:80/health",
                "criticality": "medium",
                "startup_grace": 10,
                "dependencies": []
            },
            "data": {
                "url": "http://trading-data:3001/health",
                "criticality": "critical",
                "startup_grace": 20,
                "dependencies": []
            },
            "nhits": {
                "url": "http://trading-nhits:3002/health",
                "criticality": "high",
                "startup_grace": 40,
                "dependencies": ["data"]
            },
            "tcn": {
                "url": "http://trading-tcn:3003/health",
                "criticality": "high",
                "startup_grace": 40,
                "dependencies": ["data", "embedder"]
            },
            "hmm": {
                "url": "http://trading-hmm:3004/health",
                "criticality": "high",
                "startup_grace": 30,
                "dependencies": ["data"]
            },
            "embedder": {
                "url": "http://trading-embedder:3005/health",
                "criticality": "high",
                "startup_grace": 120,
                "dependencies": ["data"]
            },
            "candlestick": {
                "url": "http://trading-candlestick:3006/health",
                "criticality": "high",
                "startup_grace": 30,
                "dependencies": ["data"]
            },
            "candlestick-train": {
                "url": "http://trading-candlestick-train:3016/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "rag": {
                "url": "http://trading-rag:3008/health",
                "criticality": "high",
                "startup_grace": 60,
                "dependencies": ["data"]
            },
            "llm": {
                "url": "http://trading-llm:3009/health",
                "criticality": "medium",
                "startup_grace": 60,
                "dependencies": ["rag"]
            },
            "easyinsight": {
                "url": "http://10.1.19.102:3000/api/health",
                "criticality": "critical",
                "startup_grace": 10,
                "dependencies": []
            }
        }

    async def check_service(self, name: str, config: dict) -> ServiceStatus:
        """
        Prüft einen einzelnen Service.

        Args:
            name: Service-Name
            config: Service-Konfiguration

        Returns:
            ServiceStatus mit aktuellem Status
        """
        start_time = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
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

    def _create_failure_status(
        self, name: str, check_time: datetime, error: str
    ) -> ServiceStatus:
        """Erstellt einen Fehler-Status."""
        prev_status = self.status.get(name)
        consecutive = (prev_status.consecutive_failures + 1) if prev_status else 1

        return ServiceStatus(
            name=name,
            state=HealthState.UNHEALTHY,
            response_time_ms=None,
            last_check=check_time,
            error=error,
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

    def get_summary(self) -> dict:
        """Gibt eine Zusammenfassung des aktuellen Status zurück."""
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

        return {
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "total": len(self.status),
            "avg_response_ms": avg_response_ms
        }
