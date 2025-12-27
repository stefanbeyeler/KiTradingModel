"""Test Runner Service für umfassende API-Tests.

Führt alle definierten Tests aus und bietet verschiedene Test-Modi:
- smoke: Nur kritische Health-Checks
- api: Alle API-Endpoint Tests
- contract: Schema-Validierung
- integration: Service-übergreifende Tests
- full: Alle Tests
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import httpx
from loguru import logger

from .test_definitions import (
    TestDefinition,
    TestCategory,
    TestPriority,
    get_all_tests,
    get_tests_by_category,
    get_tests_by_service,
    get_critical_tests,
)


class TestStatus(Enum):
    """Status eines einzelnen Tests."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class RunMode(str, Enum):
    """Test-Ausführungs-Modi."""
    SMOKE = "smoke"          # Nur Health-Checks (schnell)
    API = "api"              # Alle API-Tests
    CONTRACT = "contract"    # Schema-Validierung
    INTEGRATION = "integration"  # Service-übergreifend
    FULL = "full"            # Alle Tests
    CRITICAL = "critical"    # Nur kritische Tests


@dataclass
class TestResult:
    """Ergebnis eines einzelnen Tests."""
    name: str
    status: TestStatus
    service: str
    category: str
    priority: str
    endpoint: str = ""
    method: str = "GET"
    duration_ms: float = 0
    message: str = ""
    error_type: str = ""
    response_status: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
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
    mode: str
    status: str  # running, completed, aborted, error
    started_at: str
    completed_at: Optional[str] = None
    services: List[ServiceHealth] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)
    current_test: Optional[str] = None
    progress: Dict[str, int] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)


class TestRunnerService:
    """Service zum Ausführen von API-Tests."""

    # Service-Konfiguration für Health-Checks
    SERVICES = {
        "frontend": {
            "url": "http://trading-frontend:80",
            "health": "/health",
            "name": "Frontend Dashboard"
        },
        "data": {
            "url": "http://trading-data:3001",
            "health": "/health",
            "name": "Data Service"
        },
        "nhits": {
            "url": "http://trading-nhits:3002",
            "health": "/health",
            "name": "NHITS Service"
        },
        "tcn": {
            "url": "http://trading-tcn:3003",
            "health": "/health",
            "name": "TCN-Pattern Service"
        },
        "tcn_train": {
            "url": "http://trading-tcn-train:3013",
            "health": "/health",
            "name": "TCN-Train Service"
        },
        "hmm": {
            "url": "http://trading-hmm:3004",
            "health": "/health",
            "name": "HMM-Regime Service"
        },
        "embedder": {
            "url": "http://trading-embedder:3005",
            "health": "/health",
            "name": "Embedder Service"
        },
        "rag": {
            "url": "http://trading-rag:3008",
            "health": "/health",
            "name": "RAG Service"
        },
        "llm": {
            "url": "http://trading-llm:3009",
            "health": "/health",
            "name": "LLM Service"
        },
        "watchdog": {
            "url": "http://localhost:3010",
            "health": "/health",
            "name": "Watchdog Service"
        },
    }

    def __init__(self):
        self.current_run: Optional[TestRun] = None
        self.history: List[TestRun] = []
        self._running = False
        self._abort_requested = False
        self._healthy_services: Set[str] = set()

    async def check_service_health(
        self,
        key: str,
        config: Dict,
        client: httpx.AsyncClient
    ) -> ServiceHealth:
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
        test_def: TestDefinition,
        client: httpx.AsyncClient,
        passed_tests: Set[str]
    ) -> TestResult:
        """Führt einen einzelnen Test aus."""

        # Prüfe Abhängigkeiten
        for dep in test_def.depends_on:
            if dep not in passed_tests:
                return TestResult(
                    name=test_def.name,
                    status=TestStatus.SKIPPED,
                    service=test_def.service,
                    category=test_def.category.value,
                    priority=test_def.priority.value,
                    endpoint=test_def.endpoint,
                    method=test_def.method,
                    message=f"Abhängigkeit '{dep}' nicht erfüllt"
                )

        # Prüfe ob Service verfügbar (außer für Integration-Tests)
        if test_def.service != "integration" and test_def.service not in self._healthy_services:
            return TestResult(
                name=test_def.name,
                status=TestStatus.SKIPPED,
                service=test_def.service,
                category=test_def.category.value,
                priority=test_def.priority.value,
                endpoint=test_def.endpoint,
                method=test_def.method,
                message=f"Service '{test_def.service}' nicht verfügbar"
            )

        if self._abort_requested:
            return TestResult(
                name=test_def.name,
                status=TestStatus.SKIPPED,
                service=test_def.service,
                category=test_def.category.value,
                priority=test_def.priority.value,
                endpoint=test_def.endpoint,
                method=test_def.method,
                message="Test-Lauf abgebrochen"
            )

        start = time.time()

        try:
            # Request ausführen
            if test_def.method.upper() == "GET":
                response = await client.get(
                    test_def.endpoint,
                    params=test_def.params,
                    timeout=test_def.timeout
                )
            elif test_def.method.upper() == "POST":
                response = await client.post(
                    test_def.endpoint,
                    json=test_def.body,
                    params=test_def.params,
                    timeout=test_def.timeout
                )
            else:
                response = await client.request(
                    test_def.method.upper(),
                    test_def.endpoint,
                    json=test_def.body,
                    params=test_def.params,
                    timeout=test_def.timeout
                )

            elapsed = (time.time() - start) * 1000

            # Status-Code prüfen
            if response.status_code not in test_def.expected_status:
                return TestResult(
                    name=test_def.name,
                    status=TestStatus.FAILED,
                    service=test_def.service,
                    category=test_def.category.value,
                    priority=test_def.priority.value,
                    endpoint=test_def.endpoint,
                    method=test_def.method,
                    duration_ms=round(elapsed, 1),
                    response_status=response.status_code,
                    message=f"Unerwarteter Status: {response.status_code} (erwartet: {test_def.expected_status})",
                    error_type="UnexpectedStatusCode"
                )

            # JSON-Validierung
            data = None
            if test_def.validate_json and response.status_code == 200:
                try:
                    data = response.json()
                except Exception as e:
                    return TestResult(
                        name=test_def.name,
                        status=TestStatus.FAILED,
                        service=test_def.service,
                        category=test_def.category.value,
                        priority=test_def.priority.value,
                        endpoint=test_def.endpoint,
                        method=test_def.method,
                        duration_ms=round(elapsed, 1),
                        response_status=response.status_code,
                        message=f"Invalid JSON: {e}",
                        error_type="JSONDecodeError"
                    )

            # Schema-Validierung
            if test_def.schema_validator and data is not None:
                try:
                    if not test_def.schema_validator(data):
                        return TestResult(
                            name=test_def.name,
                            status=TestStatus.FAILED,
                            service=test_def.service,
                            category=test_def.category.value,
                            priority=test_def.priority.value,
                            endpoint=test_def.endpoint,
                            method=test_def.method,
                            duration_ms=round(elapsed, 1),
                            response_status=response.status_code,
                            message="Schema-Validierung fehlgeschlagen",
                            error_type="SchemaValidationError"
                        )
                except Exception as e:
                    return TestResult(
                        name=test_def.name,
                        status=TestStatus.FAILED,
                        service=test_def.service,
                        category=test_def.category.value,
                        priority=test_def.priority.value,
                        endpoint=test_def.endpoint,
                        method=test_def.method,
                        duration_ms=round(elapsed, 1),
                        response_status=response.status_code,
                        message=f"Schema-Validator Error: {e}",
                        error_type="SchemaValidatorError"
                    )

            # Pflichtfelder prüfen
            if test_def.required_fields and data is not None:
                missing = [f for f in test_def.required_fields if f not in data]
                if missing:
                    return TestResult(
                        name=test_def.name,
                        status=TestStatus.FAILED,
                        service=test_def.service,
                        category=test_def.category.value,
                        priority=test_def.priority.value,
                        endpoint=test_def.endpoint,
                        method=test_def.method,
                        duration_ms=round(elapsed, 1),
                        response_status=response.status_code,
                        message=f"Fehlende Felder: {missing}",
                        error_type="MissingRequiredFields"
                    )

            # Erfolg
            return TestResult(
                name=test_def.name,
                status=TestStatus.PASSED,
                service=test_def.service,
                category=test_def.category.value,
                priority=test_def.priority.value,
                endpoint=test_def.endpoint,
                method=test_def.method,
                duration_ms=round(elapsed, 1),
                response_status=response.status_code
            )

        except httpx.ConnectError:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name=test_def.name,
                status=TestStatus.FAILED,
                service=test_def.service,
                category=test_def.category.value,
                priority=test_def.priority.value,
                endpoint=test_def.endpoint,
                method=test_def.method,
                duration_ms=round(elapsed, 1),
                message="Verbindung abgelehnt",
                error_type="ConnectionError"
            )

        except httpx.TimeoutException:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name=test_def.name,
                status=TestStatus.FAILED,
                service=test_def.service,
                category=test_def.category.value,
                priority=test_def.priority.value,
                endpoint=test_def.endpoint,
                method=test_def.method,
                duration_ms=round(elapsed, 1),
                message=f"Timeout nach {test_def.timeout}s",
                error_type="TimeoutError"
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name=test_def.name,
                status=TestStatus.FAILED,
                service=test_def.service,
                category=test_def.category.value,
                priority=test_def.priority.value,
                endpoint=test_def.endpoint,
                method=test_def.method,
                duration_ms=round(elapsed, 1),
                message=str(e),
                error_type=type(e).__name__
            )

    def _get_tests_for_mode(
        self,
        mode: RunMode,
        services: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> List[TestDefinition]:
        """Gibt Tests basierend auf Modus und Filtern zurück."""

        if mode == RunMode.SMOKE:
            tests = get_tests_by_category(TestCategory.SMOKE)
        elif mode == RunMode.API:
            tests = get_tests_by_category(TestCategory.API)
        elif mode == RunMode.CONTRACT:
            tests = get_tests_by_category(TestCategory.CONTRACT)
        elif mode == RunMode.INTEGRATION:
            tests = get_tests_by_category(TestCategory.INTEGRATION)
        elif mode == RunMode.CRITICAL:
            tests = get_critical_tests()
        else:  # FULL
            tests = get_all_tests()

        # Service-Filter
        if services:
            tests = [t for t in tests if t.service in services]

        # Kategorie-Filter (zusätzlich zum Modus)
        if categories:
            category_enums = [TestCategory(c) for c in categories if c in [e.value for e in TestCategory]]
            if category_enums:
                tests = [t for t in tests if t.category in category_enums]

        return tests

    async def start_test_run(
        self,
        mode: RunMode = RunMode.FULL,
        services: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        priorities: Optional[List[str]] = None
    ) -> TestRun:
        """Startet einen neuen Test-Lauf."""

        if self._running:
            raise RuntimeError("Ein Test-Lauf ist bereits aktiv")

        self._running = True
        self._abort_requested = False
        self._healthy_services = set()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.current_run = TestRun(
            id=run_id,
            mode=mode.value,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            filters={
                "mode": mode.value,
                "services": services,
                "categories": categories,
                "priorities": priorities
            }
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. Health-Check aller Services
                logger.info("Checking service health...")
                for key, config in self.SERVICES.items():
                    health = await self.check_service_health(key, config, client)
                    self.current_run.services.append(health)
                    if health.healthy:
                        self._healthy_services.add(key)

                healthy_count = len(self._healthy_services)
                total_services = len(self.SERVICES)
                logger.info(f"Services: {healthy_count}/{total_services} healthy")

                # 2. Tests ermitteln
                tests = self._get_tests_for_mode(mode, services, categories)

                # Priority-Filter
                if priorities:
                    priority_enums = [TestPriority(p) for p in priorities if p in [e.value for e in TestPriority]]
                    if priority_enums:
                        tests = [t for t in tests if t.priority in priority_enums]

                total_tests = len(tests)
                logger.info(f"Running {total_tests} tests in {mode.value} mode...")

                self.current_run.progress = {
                    "total": total_tests,
                    "completed": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0
                }

                # 3. Tests ausführen
                passed_tests: Set[str] = set()

                for i, test_def in enumerate(tests):
                    if self._abort_requested:
                        break

                    self.current_run.current_test = test_def.name
                    self.current_run.progress["completed"] = i

                    result = await self.run_single_test(test_def, client, passed_tests)
                    self.current_run.results.append(result)

                    # Tracking
                    if result.status == TestStatus.PASSED:
                        passed_tests.add(result.name)
                        self.current_run.progress["passed"] += 1
                    elif result.status == TestStatus.FAILED:
                        self.current_run.progress["failed"] += 1
                    else:
                        self.current_run.progress["skipped"] += 1

                    # Logging
                    icon = "✓" if result.status == TestStatus.PASSED else \
                           "✗" if result.status == TestStatus.FAILED else "○"
                    logger.info(f"  {icon} [{result.category}] {result.name}: {result.status.value}")

                # 4. Zusammenfassung
                self.current_run.progress["completed"] = total_tests

                passed = sum(1 for r in self.current_run.results if r.status == TestStatus.PASSED)
                failed = sum(1 for r in self.current_run.results if r.status == TestStatus.FAILED)
                skipped = sum(1 for r in self.current_run.results if r.status == TestStatus.SKIPPED)
                errors = sum(1 for r in self.current_run.results if r.status == TestStatus.ERROR)

                # Statistiken nach Kategorie
                by_category = {}
                for cat in TestCategory:
                    cat_results = [r for r in self.current_run.results if r.category == cat.value]
                    if cat_results:
                        by_category[cat.value] = {
                            "total": len(cat_results),
                            "passed": sum(1 for r in cat_results if r.status == TestStatus.PASSED),
                            "failed": sum(1 for r in cat_results if r.status == TestStatus.FAILED),
                            "skipped": sum(1 for r in cat_results if r.status == TestStatus.SKIPPED)
                        }

                # Statistiken nach Service
                by_service = {}
                services_in_results = set(r.service for r in self.current_run.results)
                for svc in services_in_results:
                    svc_results = [r for r in self.current_run.results if r.service == svc]
                    by_service[svc] = {
                        "total": len(svc_results),
                        "passed": sum(1 for r in svc_results if r.status == TestStatus.PASSED),
                        "failed": sum(1 for r in svc_results if r.status == TestStatus.FAILED),
                        "skipped": sum(1 for r in svc_results if r.status == TestStatus.SKIPPED)
                    }

                # Durchschnittliche Response-Zeit
                durations = [r.duration_ms for r in self.current_run.results if r.duration_ms > 0]
                avg_duration = sum(durations) / len(durations) if durations else 0

                # Fehlgeschlagene Tests
                failed_tests = [
                    {"name": r.name, "service": r.service, "message": r.message, "error_type": r.error_type}
                    for r in self.current_run.results
                    if r.status == TestStatus.FAILED
                ]

                self.current_run.summary = {
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "errors": errors,
                    "total": len(self.current_run.results),
                    "success_rate": round(passed / len(self.current_run.results) * 100, 1) if self.current_run.results else 0,
                    "avg_duration_ms": round(avg_duration, 1),
                    "by_category": by_category,
                    "by_service": by_service,
                    "failed_tests": failed_tests[:20]  # Limitiere auf 20
                }

                self.current_run.status = "aborted" if self._abort_requested else "completed"
                self.current_run.completed_at = datetime.now(timezone.utc).isoformat()
                self.current_run.current_test = None

                logger.info(
                    f"Test run completed: {passed} passed, {failed} failed, "
                    f"{skipped} skipped ({self.current_run.summary['success_rate']}% success)"
                )

        except Exception as e:
            logger.error(f"Test run failed: {e}")
            self.current_run.status = "error"
            self.current_run.summary["error"] = str(e)
            self.current_run.completed_at = datetime.now(timezone.utc).isoformat()

        finally:
            self._running = False
            # In Historie speichern
            self.history.append(self.current_run)
            if len(self.history) > 50:  # Mehr Historie behalten
                self.history = self.history[-50:]

        return self.current_run

    def abort_test_run(self):
        """Bricht den aktuellen Test-Lauf ab."""
        if self._running:
            self._abort_requested = True
            logger.info("Test run abort requested")

    def is_running(self) -> bool:
        """Gibt zurück ob ein Test läuft."""
        return self._running

    def get_current_status(self) -> Optional[Dict]:
        """Gibt den aktuellen Test-Status zurück."""
        if not self.current_run:
            return None

        return {
            "id": self.current_run.id,
            "mode": self.current_run.mode,
            "status": self.current_run.status,
            "started_at": self.current_run.started_at,
            "completed_at": self.current_run.completed_at,
            "current_test": self.current_run.current_test,
            "progress": self.current_run.progress,
            "filters": self.current_run.filters,
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
                    "category": r.category,
                    "priority": r.priority,
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "duration_ms": r.duration_ms,
                    "response_status": r.response_status,
                    "message": r.message,
                    "error_type": r.error_type
                }
                for r in self.current_run.results
            ],
            "summary": self.current_run.summary
        }

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Gibt die Test-Historie zurück."""
        return [
            {
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
            for run in reversed(self.history[-limit:])
        ]

    def get_available_modes(self) -> List[Dict]:
        """Gibt verfügbare Test-Modi zurück."""
        return [
            {"mode": "smoke", "description": "Schnelle Health-Checks", "tests": len(get_tests_by_category(TestCategory.SMOKE))},
            {"mode": "api", "description": "Alle API-Endpoint Tests", "tests": len(get_tests_by_category(TestCategory.API))},
            {"mode": "contract", "description": "Schema-Validierung", "tests": len(get_tests_by_category(TestCategory.CONTRACT))},
            {"mode": "integration", "description": "Service-übergreifende Tests", "tests": len(get_tests_by_category(TestCategory.INTEGRATION))},
            {"mode": "critical", "description": "Nur kritische Tests", "tests": len(get_critical_tests())},
            {"mode": "full", "description": "Alle Tests", "tests": len(get_all_tests())},
        ]

    def get_test_definitions(
        self,
        category: Optional[str] = None,
        service: Optional[str] = None
    ) -> List[Dict]:
        """Gibt Test-Definitionen zurück."""
        tests = get_all_tests()

        if category:
            try:
                cat_enum = TestCategory(category)
                tests = [t for t in tests if t.category == cat_enum]
            except ValueError:
                pass

        if service:
            tests = [t for t in tests if t.service == service]

        return [
            {
                "name": t.name,
                "service": t.service,
                "category": t.category.value,
                "priority": t.priority.value,
                "endpoint": t.endpoint,
                "method": t.method,
                "description": t.description,
                "timeout": t.timeout,
                "expected_status": t.expected_status,
                "depends_on": t.depends_on
            }
            for t in tests
        ]


# Singleton instance
test_runner = TestRunnerService()
