"""Test Runner Service für API-Tests via Frontend."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable, Optional

import httpx
from loguru import logger


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Einzelnes Testergebnis."""
    name: str
    status: TestStatus
    service: str
    endpoint: str = ""
    duration_ms: float = 0
    message: str = ""
    error_type: str = ""
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ServiceHealth:
    """Service Health Status."""
    name: str
    display_name: str
    url: str
    healthy: bool
    response_time_ms: float = 0
    status_code: Optional[int] = None
    error: str = ""
    version: str = ""


@dataclass
class TestRun:
    """Kompletter Test-Lauf."""
    id: str
    status: str  # running, completed, aborted
    started_at: str
    completed_at: Optional[str] = None
    services: list[ServiceHealth] = field(default_factory=list)
    results: list[TestResult] = field(default_factory=list)
    current_test: Optional[str] = None
    summary: dict = field(default_factory=dict)


class TestRunnerService:
    """Service zum Ausführen von API-Tests."""

    SERVICES = {
        "frontend": {"url": "http://trading-frontend:80", "health": "/health", "name": "Frontend Dashboard"},
        "data": {"url": "http://trading-data:3001", "health": "/health", "name": "Data Service"},
        "nhits": {"url": "http://trading-nhits:3002", "health": "/health", "name": "NHITS Service"},
        "tcn": {"url": "http://trading-tcn:3003", "health": "/health", "name": "TCN-Pattern Service"},
        "hmm": {"url": "http://trading-hmm:3004", "health": "/health", "name": "HMM-Regime Service"},
        "embedder": {"url": "http://trading-embedder:3005", "health": "/health", "name": "Embedder Service"},
        "rag": {"url": "http://trading-rag:3008", "health": "/health", "name": "RAG Service"},
        "llm": {"url": "http://trading-llm:3009", "health": "/health", "name": "LLM Service"},
        "watchdog": {"url": "http://localhost:3010", "health": "/health", "name": "Watchdog Service"},
    }

    def __init__(self):
        self.current_run: Optional[TestRun] = None
        self.history: list[TestRun] = []
        self._running = False
        self._abort_requested = False

    async def check_service_health(self, key: str, config: dict, client: httpx.AsyncClient) -> ServiceHealth:
        """Prüft den Health-Status eines einzelnen Services."""
        url = f"{config['url']}{config['health']}"
        start = time.time()

        try:
            response = await client.get(url, timeout=10.0)
            elapsed = (time.time() - start) * 1000

            healthy = response.status_code == 200
            version = ""

            if healthy:
                try:
                    data = response.json()
                    version = data.get("version", "")
                except Exception:
                    pass

            return ServiceHealth(
                name=key,
                display_name=config["name"],
                url=url,
                healthy=healthy,
                response_time_ms=round(elapsed, 1),
                status_code=response.status_code,
                version=version
            )
        except httpx.ConnectError:
            elapsed = (time.time() - start) * 1000
            return ServiceHealth(
                name=key,
                display_name=config["name"],
                url=url,
                healthy=False,
                response_time_ms=round(elapsed, 1),
                error="Connection refused - Service not running"
            )
        except httpx.TimeoutException:
            elapsed = (time.time() - start) * 1000
            return ServiceHealth(
                name=key,
                display_name=config["name"],
                url=url,
                healthy=False,
                response_time_ms=round(elapsed, 1),
                error="Timeout - Service not responding"
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ServiceHealth(
                name=key,
                display_name=config["name"],
                url=url,
                healthy=False,
                response_time_ms=round(elapsed, 1),
                error=str(e)
            )

    async def run_single_test(
        self,
        name: str,
        service: str,
        endpoint: str,
        test_func: Callable[[httpx.AsyncClient], Awaitable[Any]],
        client: httpx.AsyncClient,
        service_health: dict[str, ServiceHealth]
    ) -> TestResult:
        """Führt einen einzelnen Test aus."""

        # Check if service is available
        if service in service_health and not service_health[service].healthy:
            return TestResult(
                name=name,
                status=TestStatus.SKIPPED,
                service=service,
                endpoint=endpoint,
                message=f"Service '{service}' not available"
            )

        if self._abort_requested:
            return TestResult(
                name=name,
                status=TestStatus.SKIPPED,
                service=service,
                endpoint=endpoint,
                message="Test run aborted"
            )

        start = time.time()

        try:
            await test_func(client)
            elapsed = (time.time() - start) * 1000

            return TestResult(
                name=name,
                status=TestStatus.PASSED,
                service=service,
                endpoint=endpoint,
                duration_ms=round(elapsed, 1)
            )

        except AssertionError as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name=name,
                status=TestStatus.FAILED,
                service=service,
                endpoint=endpoint,
                duration_ms=round(elapsed, 1),
                message=str(e),
                error_type="AssertionError",
                details={"assertion": str(e)}
            )

        except httpx.ConnectError:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name=name,
                status=TestStatus.FAILED,
                service=service,
                endpoint=endpoint,
                duration_ms=round(elapsed, 1),
                message="Connection refused",
                error_type="ConnectionError",
                details={"reason": "Service not running or not reachable"}
            )

        except httpx.TimeoutException:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name=name,
                status=TestStatus.FAILED,
                service=service,
                endpoint=endpoint,
                duration_ms=round(elapsed, 1),
                message="Request timeout",
                error_type="TimeoutError",
                details={"reason": "Service took too long to respond"}
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name=name,
                status=TestStatus.FAILED,
                service=service,
                endpoint=endpoint,
                duration_ms=round(elapsed, 1),
                message=str(e),
                error_type=type(e).__name__,
                details={"error": str(e)}
            )

    def _define_tests(self) -> list[dict]:
        """Definiert alle verfügbaren Tests."""
        tests = []

        # Data Service Tests
        data_url = self.SERVICES["data"]["url"]
        tests.extend([
            {
                "name": "Data Service Health",
                "service": "data",
                "endpoint": f"{data_url}/health",
                "func": lambda c, u=data_url: self._test_health(c, u)
            },
            {
                "name": "Get Managed Symbols",
                "service": "data",
                "endpoint": f"{data_url}/api/v1/managed-symbols",
                "func": lambda c, u=data_url: self._test_get_200(c, f"{u}/api/v1/managed-symbols")
            },
            {
                "name": "Get Training Data",
                "service": "data",
                "endpoint": f"{data_url}/api/v1/training-data/BTCUSD",
                "func": lambda c, u=data_url: self._test_get_200_or_404(c, f"{u}/api/v1/training-data/BTCUSD")
            },
            {
                "name": "Get Strategies",
                "service": "data",
                "endpoint": f"{data_url}/api/v1/strategies",
                "func": lambda c, u=data_url: self._test_get_200(c, f"{u}/api/v1/strategies")
            },
        ])

        # NHITS Service Tests
        nhits_url = self.SERVICES["nhits"]["url"]
        tests.extend([
            {
                "name": "NHITS Service Health",
                "service": "nhits",
                "endpoint": f"{nhits_url}/health",
                "func": lambda c, u=nhits_url: self._test_health(c, u)
            },
            {
                "name": "NHITS Forecast Status",
                "service": "nhits",
                "endpoint": f"{nhits_url}/api/v1/forecast/status",
                "func": lambda c, u=nhits_url: self._test_get_200(c, f"{u}/api/v1/forecast/status")
            },
            {
                "name": "NHITS Models List",
                "service": "nhits",
                "endpoint": f"{nhits_url}/api/v1/forecast/models",
                "func": lambda c, u=nhits_url: self._test_get_200(c, f"{u}/api/v1/forecast/models")
            },
            {
                "name": "Generate Forecast (BTCUSD)",
                "service": "nhits",
                "endpoint": f"{nhits_url}/api/v1/forecast/BTCUSD",
                "func": lambda c, u=nhits_url: self._test_get_200_or_404(c, f"{u}/api/v1/forecast/BTCUSD")
            },
        ])

        # TCN Service Tests
        tcn_url = self.SERVICES["tcn"]["url"]
        tests.extend([
            {
                "name": "TCN Service Health",
                "service": "tcn",
                "endpoint": f"{tcn_url}/health",
                "func": lambda c, u=tcn_url: self._test_health(c, u)
            },
            {
                "name": "Get Pattern Types",
                "service": "tcn",
                "endpoint": f"{tcn_url}/api/v1/patterns",
                "func": lambda c, u=tcn_url: self._test_get_200(c, f"{u}/api/v1/patterns")
            },
            {
                "name": "Get TCN Models",
                "service": "tcn",
                "endpoint": f"{tcn_url}/api/v1/models",
                "func": lambda c, u=tcn_url: self._test_get_200(c, f"{u}/api/v1/models")
            },
        ])

        # HMM Service Tests
        hmm_url = self.SERVICES["hmm"]["url"]
        tests.extend([
            {
                "name": "HMM Service Health",
                "service": "hmm",
                "endpoint": f"{hmm_url}/health",
                "func": lambda c, u=hmm_url: self._test_health(c, u)
            },
            {
                "name": "Get Market Regime",
                "service": "hmm",
                "endpoint": f"{hmm_url}/api/v1/regime/BTCUSD",
                "func": lambda c, u=hmm_url: self._test_get_200_or_404(c, f"{u}/api/v1/regime/BTCUSD")
            },
        ])

        # Embedder Service Tests
        embedder_url = self.SERVICES["embedder"]["url"]
        tests.extend([
            {
                "name": "Embedder Service Health",
                "service": "embedder",
                "endpoint": f"{embedder_url}/health",
                "func": lambda c, u=embedder_url: self._test_health(c, u)
            },
            {
                "name": "Get Embedding Models",
                "service": "embedder",
                "endpoint": f"{embedder_url}/api/v1/models",
                "func": lambda c, u=embedder_url: self._test_get_200(c, f"{u}/api/v1/models")
            },
        ])

        # RAG Service Tests
        rag_url = self.SERVICES["rag"]["url"]
        tests.extend([
            {
                "name": "RAG Service Health",
                "service": "rag",
                "endpoint": f"{rag_url}/health",
                "func": lambda c, u=rag_url: self._test_health(c, u)
            },
            {
                "name": "Get RAG Index Stats",
                "service": "rag",
                "endpoint": f"{rag_url}/api/v1/rag/stats",
                "func": lambda c, u=rag_url: self._test_get_200(c, f"{u}/api/v1/rag/stats")
            },
        ])

        # LLM Service Tests
        llm_url = self.SERVICES["llm"]["url"]
        tests.extend([
            {
                "name": "LLM Service Health",
                "service": "llm",
                "endpoint": f"{llm_url}/health",
                "func": lambda c, u=llm_url: self._test_health(c, u)
            },
            {
                "name": "LLM Model Status",
                "service": "llm",
                "endpoint": f"{llm_url}/api/v1/llm/status",
                "func": lambda c, u=llm_url: self._test_get_200(c, f"{u}/api/v1/llm/status")
            },
        ])

        # Watchdog Service Tests
        watchdog_url = self.SERVICES["watchdog"]["url"]
        tests.extend([
            {
                "name": "Watchdog Service Health",
                "service": "watchdog",
                "endpoint": f"{watchdog_url}/health",
                "func": lambda c, u=watchdog_url: self._test_health(c, u)
            },
            {
                "name": "Watchdog All Services Status",
                "service": "watchdog",
                "endpoint": f"{watchdog_url}/api/v1/status",
                "func": lambda c, u=watchdog_url: self._test_get_200(c, f"{u}/api/v1/status")
            },
        ])

        return tests

    async def _test_health(self, client: httpx.AsyncClient, base_url: str):
        """Test health endpoint."""
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("status") in ["healthy", "ok", None] or "status" not in data, f"Unexpected status: {data}"

    async def _test_get_200(self, client: httpx.AsyncClient, url: str):
        """Test GET endpoint returns 200."""
        response = await client.get(url)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    async def _test_get_200_or_404(self, client: httpx.AsyncClient, url: str):
        """Test GET endpoint returns 200 or 404."""
        response = await client.get(url)
        assert response.status_code in [200, 404], f"Expected 200 or 404, got {response.status_code}"

    async def start_test_run(self) -> TestRun:
        """Startet einen neuen Test-Lauf."""
        if self._running:
            raise RuntimeError("A test run is already in progress")

        self._running = True
        self._abort_requested = False

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.current_run = TestRun(
            id=run_id,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat()
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. Check all service health
                logger.info("Checking service health...")
                service_health = {}
                for key, config in self.SERVICES.items():
                    health = await self.check_service_health(key, config, client)
                    service_health[key] = health
                    self.current_run.services.append(health)

                healthy_count = sum(1 for s in service_health.values() if s.healthy)
                logger.info(f"Services: {healthy_count}/{len(service_health)} healthy")

                # 2. Run all tests
                tests = self._define_tests()
                logger.info(f"Running {len(tests)} tests...")

                for test_def in tests:
                    if self._abort_requested:
                        break

                    self.current_run.current_test = test_def["name"]

                    result = await self.run_single_test(
                        name=test_def["name"],
                        service=test_def["service"],
                        endpoint=test_def["endpoint"],
                        test_func=test_def["func"],
                        client=client,
                        service_health=service_health
                    )
                    self.current_run.results.append(result)

                    status_icon = "✓" if result.status == TestStatus.PASSED else "✗" if result.status == TestStatus.FAILED else "○"
                    logger.info(f"  {status_icon} {test_def['name']}: {result.status.value}")

                # 3. Calculate summary
                passed = sum(1 for r in self.current_run.results if r.status == TestStatus.PASSED)
                failed = sum(1 for r in self.current_run.results if r.status == TestStatus.FAILED)
                skipped = sum(1 for r in self.current_run.results if r.status == TestStatus.SKIPPED)

                self.current_run.summary = {
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "total": len(self.current_run.results)
                }

                self.current_run.status = "aborted" if self._abort_requested else "completed"
                self.current_run.completed_at = datetime.now(timezone.utc).isoformat()
                self.current_run.current_test = None

                logger.info(f"Test run completed: {passed} passed, {failed} failed, {skipped} skipped")

        except Exception as e:
            logger.error(f"Test run failed: {e}")
            self.current_run.status = "error"
            self.current_run.summary["error"] = str(e)

        finally:
            self._running = False
            # Store in history
            self.history.append(self.current_run)
            if len(self.history) > 20:
                self.history = self.history[-20:]

        return self.current_run

    def abort_test_run(self):
        """Bricht den aktuellen Test-Lauf ab."""
        if self._running:
            self._abort_requested = True
            logger.info("Test run abort requested")

    def get_current_status(self) -> Optional[dict]:
        """Gibt den aktuellen Test-Status zurück."""
        if not self.current_run:
            return None

        return {
            "id": self.current_run.id,
            "status": self.current_run.status,
            "started_at": self.current_run.started_at,
            "completed_at": self.current_run.completed_at,
            "current_test": self.current_run.current_test,
            "services": [
                {
                    "name": s.name,
                    "display_name": s.display_name,
                    "healthy": s.healthy,
                    "response_time_ms": s.response_time_ms,
                    "error": s.error,
                    "version": s.version
                }
                for s in self.current_run.services
            ],
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "service": r.service,
                    "endpoint": r.endpoint,
                    "duration_ms": r.duration_ms,
                    "message": r.message,
                    "error_type": r.error_type
                }
                for r in self.current_run.results
            ],
            "summary": self.current_run.summary
        }

    def get_history(self, limit: int = 10) -> list[dict]:
        """Gibt die Test-Historie zurück."""
        return [
            {
                "id": run.id,
                "status": run.status,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "summary": run.summary
            }
            for run in reversed(self.history[-limit:])
        ]


# Singleton instance
test_runner = TestRunnerService()
