#!/usr/bin/env python3
"""
Interactive Test Runner for KI Trading Model

Provides real-time test execution feedback with user interaction on failures.
Shows detailed API endpoint status and allows skipping failed tests.
"""

import asyncio
import httpx
import sys
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ========== ANSI Colors ==========

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Status colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class Icons:
    RUNNING = "⏳"
    PASSED = "✅"
    FAILED = "❌"
    SKIPPED = "⏭️ "
    WARNING = "⚠️ "
    INFO = "ℹ️ "
    ARROW = "→"
    CHECK = "✓"
    CROSS = "✗"
    CLOCK = "🕐"
    SERVER = "🖥️ "
    API = "🔌"
    NETWORK = "🌐"


# ========== Test Result Types ==========

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration_ms: float = 0
    message: str = ""
    details: dict = field(default_factory=dict)
    endpoint: str = ""
    error_type: str = ""


@dataclass
class ServiceStatus:
    name: str
    url: str
    healthy: bool
    response_time_ms: float = 0
    status_code: Optional[int] = None
    error: str = ""
    version: str = ""


# ========== Console Output ==========

class Console:
    """Rich console output with colors and formatting."""

    @staticmethod
    def clear_line():
        print("\r" + " " * 80 + "\r", end="", flush=True)

    @staticmethod
    def print_header(title: str):
        width = 70
        print()
        print(f"{Colors.CYAN}{Colors.BOLD}{'═' * width}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}  {title}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'═' * width}{Colors.RESET}")
        print()

    @staticmethod
    def print_section(title: str):
        print()
        print(f"{Colors.BLUE}{Colors.BOLD}── {title} ──{Colors.RESET}")
        print()

    @staticmethod
    def print_test_start(name: str, endpoint: str = ""):
        endpoint_info = f" {Colors.DIM}({endpoint}){Colors.RESET}" if endpoint else ""
        print(f"  {Icons.RUNNING} {Colors.YELLOW}Running:{Colors.RESET} {name}{endpoint_info}", end="", flush=True)

    @staticmethod
    def print_test_result(result: TestResult):
        Console.clear_line()

        if result.status == TestStatus.PASSED:
            icon = Icons.PASSED
            color = Colors.GREEN
            status_text = "PASSED"
        elif result.status == TestStatus.FAILED:
            icon = Icons.FAILED
            color = Colors.RED
            status_text = "FAILED"
        elif result.status == TestStatus.SKIPPED:
            icon = Icons.SKIPPED
            color = Colors.YELLOW
            status_text = "SKIPPED"
        else:
            icon = Icons.WARNING
            color = Colors.YELLOW
            status_text = result.status.value.upper()

        duration = f"{Colors.DIM}({result.duration_ms:.0f}ms){Colors.RESET}"
        print(f"  {icon} {color}{status_text}{Colors.RESET} {result.name} {duration}")

        if result.message:
            print(f"      {Colors.DIM}{result.message}{Colors.RESET}")

        if result.status == TestStatus.FAILED and result.details:
            Console.print_error_details(result)

    @staticmethod
    def print_error_details(result: TestResult):
        print()
        print(f"      {Colors.RED}{Colors.BOLD}Error Details:{Colors.RESET}")

        if result.endpoint:
            print(f"      {Icons.API} Endpoint: {Colors.CYAN}{result.endpoint}{Colors.RESET}")

        if result.error_type:
            print(f"      {Icons.CROSS} Type: {Colors.RED}{result.error_type}{Colors.RESET}")

        for key, value in result.details.items():
            if key not in ["endpoint", "error_type"]:
                print(f"      {Icons.ARROW} {key}: {value}")
        print()

    @staticmethod
    def print_service_status(service: ServiceStatus):
        if service.healthy:
            icon = Icons.CHECK
            color = Colors.GREEN
            status = "HEALTHY"
        else:
            icon = Icons.CROSS
            color = Colors.RED
            status = "UNHEALTHY"

        timing = f"{Colors.DIM}({service.response_time_ms:.0f}ms){Colors.RESET}" if service.response_time_ms > 0 else ""
        version = f" v{service.version}" if service.version else ""

        print(f"  {icon} {color}{status:10}{Colors.RESET} {service.name:15} {service.url:35} {timing}{version}")

        if service.error:
            print(f"    {Colors.DIM}{Icons.ARROW} {service.error}{Colors.RESET}")

    @staticmethod
    def print_summary(passed: int, failed: int, skipped: int, duration: float):
        print()
        print(f"{Colors.BOLD}{'─' * 50}{Colors.RESET}")

        total = passed + failed + skipped

        if failed == 0:
            result_color = Colors.GREEN
            result_icon = Icons.PASSED
        else:
            result_color = Colors.RED
            result_icon = Icons.FAILED

        print(f"  {result_icon} {result_color}{Colors.BOLD}Test Summary{Colors.RESET}")
        print()
        print(f"      {Colors.GREEN}{Icons.PASSED} Passed:  {passed}{Colors.RESET}")
        print(f"      {Colors.RED}{Icons.FAILED} Failed:  {failed}{Colors.RESET}")
        print(f"      {Colors.YELLOW}{Icons.SKIPPED} Skipped: {skipped}{Colors.RESET}")
        print(f"      {Colors.DIM}─────────────{Colors.RESET}")
        print(f"      Total:    {total}")
        print()
        print(f"      {Icons.CLOCK} Duration: {duration:.2f}s")
        print()

    @staticmethod
    def ask_user_decision(test_name: str, error_details: str) -> str:
        """Ask user what to do on test failure."""
        print()
        print(f"  {Colors.BG_YELLOW}{Colors.BOLD} TEST FAILED {Colors.RESET}")
        print(f"  {Colors.YELLOW}Test: {test_name}{Colors.RESET}")
        print()
        print(f"  {Colors.DIM}{error_details}{Colors.RESET}")
        print()
        print(f"  What would you like to do?")
        print(f"    [{Colors.GREEN}s{Colors.RESET}] Skip this test and continue")
        print(f"    [{Colors.YELLOW}r{Colors.RESET}] Retry this test")
        print(f"    [{Colors.RED}a{Colors.RESET}] Abort all tests")
        print(f"    [{Colors.BLUE}c{Colors.RESET}] Continue (mark as failed)")
        print()

        while True:
            try:
                choice = input(f"  Your choice [{Colors.GREEN}s{Colors.RESET}/r/a/c]: ").strip().lower()
                if choice in ["s", "r", "a", "c", ""]:
                    return choice if choice else "s"
            except (KeyboardInterrupt, EOFError):
                return "a"


# ========== Service Configuration ==========

SERVICE_HOST = os.getenv("TEST_SERVICE_HOST", "localhost")

SERVICES = {
    "frontend": {"url": f"http://{SERVICE_HOST}:3000", "health": "/health", "name": "Frontend Dashboard"},
    "data": {"url": f"http://{SERVICE_HOST}:3001", "health": "/health", "name": "Data Service"},
    "nhits": {"url": f"http://{SERVICE_HOST}:3002", "health": "/health", "name": "NHITS Service"},
    "tcn": {"url": f"http://{SERVICE_HOST}:3003", "health": "/health", "name": "TCN-Pattern Service"},
    "hmm": {"url": f"http://{SERVICE_HOST}:3004", "health": "/health", "name": "HMM-Regime Service"},
    "embedder": {"url": f"http://{SERVICE_HOST}:3005", "health": "/health", "name": "Embedder Service"},
    "rag": {"url": f"http://{SERVICE_HOST}:3008", "health": "/health", "name": "RAG Service"},
    "llm": {"url": f"http://{SERVICE_HOST}:3009", "health": "/health", "name": "LLM Service"},
    "watchdog": {"url": f"http://{SERVICE_HOST}:3010", "health": "/health", "name": "Watchdog Service"},
}


# ========== Test Runner ==========

class InteractiveTestRunner:
    """Interactive test runner with real-time feedback."""

    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        self.results: list[TestResult] = []
        self.service_status: dict[str, ServiceStatus] = {}
        self.aborted = False
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def check_service(self, key: str, config: dict) -> ServiceStatus:
        """Check a single service health."""
        url = f"{config['url']}{config['health']}"
        start = time.time()

        try:
            response = await self.client.get(url)
            elapsed = (time.time() - start) * 1000

            healthy = response.status_code == 200
            version = ""

            if healthy:
                try:
                    data = response.json()
                    version = data.get("version", "")
                except:
                    pass

            return ServiceStatus(
                name=config["name"],
                url=url,
                healthy=healthy,
                response_time_ms=elapsed,
                status_code=response.status_code,
                version=version
            )
        except httpx.ConnectError as e:
            elapsed = (time.time() - start) * 1000
            return ServiceStatus(
                name=config["name"],
                url=url,
                healthy=False,
                response_time_ms=elapsed,
                error=f"Connection refused - Service not running"
            )
        except httpx.TimeoutException:
            elapsed = (time.time() - start) * 1000
            return ServiceStatus(
                name=config["name"],
                url=url,
                healthy=False,
                response_time_ms=elapsed,
                error="Timeout - Service not responding"
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ServiceStatus(
                name=config["name"],
                url=url,
                healthy=False,
                response_time_ms=elapsed,
                error=str(e)
            )

    async def check_all_services(self) -> dict[str, ServiceStatus]:
        """Check all services in parallel."""
        Console.print_section("Service Health Check")

        tasks = {
            key: self.check_service(key, config)
            for key, config in SERVICES.items()
        }

        results = await asyncio.gather(*tasks.values())

        for key, status in zip(tasks.keys(), results):
            self.service_status[key] = status
            Console.print_service_status(status)

        healthy_count = sum(1 for s in self.service_status.values() if s.healthy)
        total_count = len(self.service_status)

        print()
        print(f"  {Icons.SERVER} Services: {Colors.GREEN}{healthy_count}{Colors.RESET}/{total_count} healthy")

        return self.service_status

    async def run_test(
        self,
        name: str,
        test_func: Callable[[], Awaitable[Any]],
        endpoint: str = "",
        required_services: list[str] = None
    ) -> TestResult:
        """Run a single test with live feedback."""

        if self.aborted:
            return TestResult(name=name, status=TestStatus.SKIPPED, message="Test run aborted")

        # Check required services
        if required_services:
            for service in required_services:
                if service in self.service_status and not self.service_status[service].healthy:
                    result = TestResult(
                        name=name,
                        status=TestStatus.SKIPPED,
                        message=f"Required service '{service}' not available",
                        endpoint=endpoint
                    )
                    Console.print_test_result(result)
                    self.results.append(result)
                    return result

        Console.print_test_start(name, endpoint)

        start = time.time()

        try:
            await test_func()
            elapsed = (time.time() - start) * 1000

            result = TestResult(
                name=name,
                status=TestStatus.PASSED,
                duration_ms=elapsed,
                endpoint=endpoint
            )

        except AssertionError as e:
            elapsed = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                status=TestStatus.FAILED,
                duration_ms=elapsed,
                message=str(e),
                endpoint=endpoint,
                error_type="AssertionError",
                details={"assertion": str(e)}
            )

        except httpx.ConnectError as e:
            elapsed = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                status=TestStatus.FAILED,
                duration_ms=elapsed,
                message="Connection refused",
                endpoint=endpoint,
                error_type="ConnectionError",
                details={
                    "reason": "Service not running or not reachable",
                    "endpoint": endpoint or "N/A"
                }
            )

        except httpx.TimeoutException as e:
            elapsed = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                status=TestStatus.FAILED,
                duration_ms=elapsed,
                message="Request timeout",
                endpoint=endpoint,
                error_type="TimeoutError",
                details={
                    "reason": "Service took too long to respond",
                    "timeout": "30s"
                }
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                status=TestStatus.FAILED,
                duration_ms=elapsed,
                message=str(e),
                endpoint=endpoint,
                error_type=type(e).__name__,
                details={"error": str(e)}
            )

        Console.print_test_result(result)

        # Handle failure interactively
        if result.status == TestStatus.FAILED and self.interactive:
            error_details = f"Endpoint: {endpoint}\nError: {result.message}"

            while True:
                decision = Console.ask_user_decision(name, error_details)

                if decision == "s":
                    result.status = TestStatus.SKIPPED
                    result.message = "Skipped by user"
                    break
                elif decision == "r":
                    # Retry the test
                    Console.print_test_start(name, endpoint)
                    return await self.run_test(name, test_func, endpoint, required_services)
                elif decision == "a":
                    self.aborted = True
                    break
                elif decision == "c":
                    break

        self.results.append(result)
        return result

    def get_summary(self) -> tuple[int, int, int]:
        """Get test summary counts."""
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        return passed, failed, skipped


# ========== Test Definitions ==========

async def create_test_suite(runner: InteractiveTestRunner):
    """Define and run all tests."""

    # ---- Data Service Tests ----
    Console.print_section("Data Service Tests")

    data_url = SERVICES["data"]["url"]

    async def test_data_health():
        response = await runner.client.get(f"{data_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("status") in ["healthy", "ok"], f"Unexpected status: {data}"

    await runner.run_test(
        "Data Service Health Check",
        test_data_health,
        endpoint=f"{data_url}/health",
        required_services=["data"]
    )

    async def test_data_symbols():
        response = await runner.client.get(f"{data_url}/api/v1/managed-symbols")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert isinstance(data, (list, dict)), "Expected list or dict response"

    await runner.run_test(
        "Get Managed Symbols",
        test_data_symbols,
        endpoint=f"{data_url}/api/v1/managed-symbols",
        required_services=["data"]
    )

    async def test_data_training_data():
        response = await runner.client.get(
            f"{data_url}/api/v1/training-data/BTCUSD",
            params={"interval": "1h", "limit": 10}
        )
        # 200 for success, 404 if symbol not configured
        assert response.status_code in [200, 404], f"Expected 200 or 404, got {response.status_code}"

    await runner.run_test(
        "Get Training Data (BTCUSD)",
        test_data_training_data,
        endpoint=f"{data_url}/api/v1/training-data/BTCUSD",
        required_services=["data"]
    )

    async def test_data_strategies():
        response = await runner.client.get(f"{data_url}/api/v1/strategies")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Get Trading Strategies",
        test_data_strategies,
        endpoint=f"{data_url}/api/v1/strategies",
        required_services=["data"]
    )

    # ---- NHITS Service Tests ----
    Console.print_section("NHITS Service Tests")

    nhits_url = SERVICES["nhits"]["url"]

    async def test_nhits_health():
        response = await runner.client.get(f"{nhits_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "NHITS Service Health Check",
        test_nhits_health,
        endpoint=f"{nhits_url}/health",
        required_services=["nhits"]
    )

    async def test_nhits_status():
        response = await runner.client.get(f"{nhits_url}/api/v1/forecast/status")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "NHITS Forecast Status",
        test_nhits_status,
        endpoint=f"{nhits_url}/api/v1/forecast/status",
        required_services=["nhits"]
    )

    async def test_nhits_models():
        response = await runner.client.get(f"{nhits_url}/api/v1/forecast/models")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "NHITS Trained Models List",
        test_nhits_models,
        endpoint=f"{nhits_url}/api/v1/forecast/models",
        required_services=["nhits"]
    )

    async def test_nhits_forecast():
        response = await runner.client.get(
            f"{nhits_url}/api/v1/forecast/BTCUSD",
            params={"horizon": 24}
        )
        # Accept 200 (success) or 404 (no model trained yet)
        assert response.status_code in [200, 404], f"Expected 200 or 404, got {response.status_code}"

    await runner.run_test(
        "Generate Forecast (BTCUSD)",
        test_nhits_forecast,
        endpoint=f"{nhits_url}/api/v1/forecast/BTCUSD",
        required_services=["nhits"]
    )

    # ---- TCN Service Tests ----
    Console.print_section("TCN-Pattern Service Tests")

    tcn_url = SERVICES["tcn"]["url"]

    async def test_tcn_health():
        response = await runner.client.get(f"{tcn_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "TCN Service Health Check",
        test_tcn_health,
        endpoint=f"{tcn_url}/health",
        required_services=["tcn"]
    )

    async def test_tcn_patterns():
        response = await runner.client.get(f"{tcn_url}/api/v1/patterns")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Get Pattern Types",
        test_tcn_patterns,
        endpoint=f"{tcn_url}/api/v1/patterns",
        required_services=["tcn"]
    )

    async def test_tcn_models():
        response = await runner.client.get(f"{tcn_url}/api/v1/models")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Get TCN Models",
        test_tcn_models,
        endpoint=f"{tcn_url}/api/v1/models",
        required_services=["tcn"]
    )

    # ---- HMM Service Tests ----
    Console.print_section("HMM-Regime Service Tests")

    hmm_url = SERVICES["hmm"]["url"]

    async def test_hmm_health():
        response = await runner.client.get(f"{hmm_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "HMM Service Health Check",
        test_hmm_health,
        endpoint=f"{hmm_url}/health",
        required_services=["hmm"]
    )

    async def test_hmm_regime():
        response = await runner.client.get(f"{hmm_url}/api/v1/regime/BTCUSD")
        # Accept 200 or 404 (no model)
        assert response.status_code in [200, 404], f"Expected 200 or 404, got {response.status_code}"

    await runner.run_test(
        "Get Market Regime (BTCUSD)",
        test_hmm_regime,
        endpoint=f"{hmm_url}/api/v1/regime/BTCUSD",
        required_services=["hmm"]
    )

    # ---- Embedder Service Tests ----
    Console.print_section("Embedder Service Tests")

    embedder_url = SERVICES["embedder"]["url"]

    async def test_embedder_health():
        response = await runner.client.get(f"{embedder_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Embedder Service Health Check",
        test_embedder_health,
        endpoint=f"{embedder_url}/health",
        required_services=["embedder"]
    )

    async def test_embedder_models():
        response = await runner.client.get(f"{embedder_url}/api/v1/models")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Get Embedding Models",
        test_embedder_models,
        endpoint=f"{embedder_url}/api/v1/models",
        required_services=["embedder"]
    )

    # ---- RAG Service Tests ----
    Console.print_section("RAG Service Tests")

    rag_url = SERVICES["rag"]["url"]

    async def test_rag_health():
        response = await runner.client.get(f"{rag_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "RAG Service Health Check",
        test_rag_health,
        endpoint=f"{rag_url}/health",
        required_services=["rag"]
    )

    async def test_rag_stats():
        response = await runner.client.get(f"{rag_url}/api/v1/rag/stats")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Get RAG Index Stats",
        test_rag_stats,
        endpoint=f"{rag_url}/api/v1/rag/stats",
        required_services=["rag"]
    )

    # ---- LLM Service Tests ----
    Console.print_section("LLM Service Tests")

    llm_url = SERVICES["llm"]["url"]

    async def test_llm_health():
        response = await runner.client.get(f"{llm_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "LLM Service Health Check",
        test_llm_health,
        endpoint=f"{llm_url}/health",
        required_services=["llm"]
    )

    async def test_llm_status():
        response = await runner.client.get(f"{llm_url}/api/v1/llm/status")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "LLM Model Status",
        test_llm_status,
        endpoint=f"{llm_url}/api/v1/llm/status",
        required_services=["llm"]
    )

    # ---- Watchdog Service Tests ----
    Console.print_section("Watchdog Service Tests")

    watchdog_url = SERVICES["watchdog"]["url"]

    async def test_watchdog_health():
        response = await runner.client.get(f"{watchdog_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Watchdog Service Health Check",
        test_watchdog_health,
        endpoint=f"{watchdog_url}/health",
        required_services=["watchdog"]
    )

    async def test_watchdog_status():
        response = await runner.client.get(f"{watchdog_url}/api/v1/status")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    await runner.run_test(
        "Watchdog All Services Status",
        test_watchdog_status,
        endpoint=f"{watchdog_url}/api/v1/status",
        required_services=["watchdog"]
    )


# ========== Main ==========

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Test Runner for KI Trading Model")
    parser.add_argument("--no-interactive", "-n", action="store_true", help="Run without user interaction")
    parser.add_argument("--host", "-H", default=None, help="Service host (default: localhost)")
    args = parser.parse_args()

    if args.host:
        global SERVICES
        for key in SERVICES:
            SERVICES[key]["url"] = SERVICES[key]["url"].replace("localhost", args.host)

    Console.print_header("KI Trading Model - Interactive Test Suite")

    print(f"  {Icons.INFO} Host: {Colors.CYAN}{args.host or 'localhost'}{Colors.RESET}")
    print(f"  {Icons.INFO} Mode: {Colors.CYAN}{'Batch' if args.no_interactive else 'Interactive'}{Colors.RESET}")
    print(f"  {Icons.INFO} Time: {Colors.CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")

    start_time = time.time()

    async with InteractiveTestRunner(interactive=not args.no_interactive) as runner:
        # Check all services first
        await runner.check_all_services()

        # Run test suite
        await create_test_suite(runner)

        # Print summary
        duration = time.time() - start_time
        passed, failed, skipped = runner.get_summary()
        Console.print_summary(passed, failed, skipped, duration)

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
