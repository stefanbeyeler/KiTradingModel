"""Testing API routes for running pytest tests via the dashboard."""

import asyncio
import json
import os
import subprocess
import signal
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

testing_router = APIRouter()


class TestRunRequest(BaseModel):
    """Request model for test execution."""
    categories: List[str] = ["smoke", "api", "integration", "contract"]
    verbose: bool = True
    timeout: int = 300  # 5 minutes default


class TestResult(BaseModel):
    """Individual test result."""
    nodeid: str
    name: str
    outcome: str  # passed, failed, skipped
    duration: Optional[float] = None
    message: Optional[str] = None
    category: Optional[str] = None


class TestRunResponse(BaseModel):
    """Response model for test execution."""
    status: str  # running, completed, failed, stopped
    started_at: str
    completed_at: Optional[str] = None
    duration: Optional[float] = None
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    tests: List[TestResult] = []
    error: Optional[str] = None


class TestStatus(BaseModel):
    """Current test status."""
    running: bool
    current_category: Optional[str] = None
    progress: int = 0
    last_run: Optional[str] = None


# Global state for tracking test runs
_test_state = {
    "running": False,
    "process": None,
    "current_category": None,
    "last_run": None,
    "last_result": None
}


def parse_pytest_json_output(output: str) -> List[TestResult]:
    """Parse pytest JSON output to extract test results."""
    results = []

    try:
        # Try to parse as JSON (if using --json flag)
        lines = output.strip().split('\n')
        for line in lines:
            if line.startswith('{'):
                try:
                    data = json.loads(line)
                    if 'nodeid' in data:
                        category = None
                        nodeid = data.get('nodeid', '')
                        if 'smoke' in nodeid:
                            category = 'smoke'
                        elif 'api' in nodeid:
                            category = 'api'
                        elif 'integration' in nodeid:
                            category = 'integration'
                        elif 'contract' in nodeid:
                            category = 'contract'
                        elif 'unit' in nodeid:
                            category = 'unit'

                        results.append(TestResult(
                            nodeid=nodeid,
                            name=data.get('name', nodeid.split('::')[-1]),
                            outcome=data.get('outcome', 'unknown'),
                            duration=data.get('duration'),
                            message=data.get('longrepr') or data.get('message'),
                            category=category
                        ))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Failed to parse pytest JSON: {e}")

    return results


def parse_pytest_verbose_output(output: str) -> List[TestResult]:
    """Parse pytest verbose output to extract test results."""
    results = []
    lines = output.strip().split('\n')

    for line in lines:
        # Parse lines like: tests/smoke/test_health.py::test_data_service_health PASSED
        if '::' in line and any(status in line for status in ['PASSED', 'FAILED', 'SKIPPED', 'ERROR']):
            parts = line.strip().split()
            if len(parts) >= 2:
                nodeid = parts[0]
                outcome = parts[-1].lower()

                # Map ERROR to failed
                if outcome == 'error':
                    outcome = 'failed'

                # Determine category from path
                category = None
                if 'smoke' in nodeid:
                    category = 'smoke'
                elif 'api/' in nodeid or '/api/' in nodeid:
                    category = 'api'
                elif 'integration' in nodeid:
                    category = 'integration'
                elif 'contract' in nodeid:
                    category = 'contract'
                elif 'unit' in nodeid:
                    category = 'unit'

                # Extract test name
                name = nodeid.split('::')[-1] if '::' in nodeid else nodeid

                results.append(TestResult(
                    nodeid=nodeid,
                    name=name,
                    outcome=outcome,
                    category=category
                ))

    return results


async def run_pytest_async(categories: List[str], timeout: int = 300) -> TestRunResponse:
    """Run pytest asynchronously and return results."""
    global _test_state

    _test_state["running"] = True
    _test_state["last_run"] = datetime.now().isoformat()
    started_at = datetime.now()

    # Find project root (where pytest.ini is located)
    project_root = os.getenv("PROJECT_ROOT", "/home/sbeyeler/KiTradingModel")
    if not os.path.exists(project_root):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Build pytest command
    markers = " or ".join(categories)
    cmd = [
        "python", "-m", "pytest",
        "-v",
        "--tb=short",
        "-m", markers,
        f"--timeout={timeout}",
        "--no-header",
        "-q"
    ]

    logger.info(f"Running tests: {' '.join(cmd)} in {project_root}")

    all_results = []
    error = None

    try:
        for category in categories:
            _test_state["current_category"] = category

            cat_cmd = [
                "python", "-m", "pytest",
                "-v",
                "--tb=short",
                "-m", category,
                f"--timeout={min(timeout // len(categories), 120)}",
                "--no-header"
            ]

            process = await asyncio.create_subprocess_exec(
                *cat_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root
            )

            _test_state["process"] = process

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout // len(categories) + 30
                )

                output = stdout.decode('utf-8', errors='replace')
                logger.debug(f"pytest output for {category}:\n{output[:1000]}")

                # Parse results
                results = parse_pytest_verbose_output(output)

                # If no results parsed, check for errors
                if not results and process.returncode != 0:
                    error_output = stderr.decode('utf-8', errors='replace')
                    if error_output:
                        logger.warning(f"pytest stderr: {error_output[:500]}")

                all_results.extend(results)

            except asyncio.TimeoutError:
                process.kill()
                logger.warning(f"Test category {category} timed out")
                all_results.append(TestResult(
                    nodeid=f"tests/{category}",
                    name=f"{category}_timeout",
                    outcome="failed",
                    message=f"Test category {category} timed out after {timeout // len(categories)}s",
                    category=category
                ))

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        error = str(e)

    finally:
        _test_state["running"] = False
        _test_state["process"] = None
        _test_state["current_category"] = None

    completed_at = datetime.now()
    duration = (completed_at - started_at).total_seconds()

    # Calculate summary
    passed = len([t for t in all_results if t.outcome == 'passed'])
    failed = len([t for t in all_results if t.outcome == 'failed'])
    skipped = len([t for t in all_results if t.outcome == 'skipped'])

    response = TestRunResponse(
        status="completed" if not error else "failed",
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration=duration,
        total=len(all_results),
        passed=passed,
        failed=failed,
        skipped=skipped,
        tests=all_results,
        error=error
    )

    _test_state["last_result"] = response

    return response


@testing_router.post("/run", response_model=TestRunResponse)
async def run_tests(request: TestRunRequest):
    """
    Run pytest tests for specified categories.

    Categories:
    - smoke: Health checks for all services
    - api: API endpoint tests
    - integration: Service integration tests
    - contract: API schema validation tests
    - unit: Unit tests (mocked, no services needed)
    """
    if _test_state["running"]:
        raise HTTPException(
            status_code=409,
            detail="Tests are already running. Wait for completion or stop them."
        )

    # Validate categories
    valid_categories = {"smoke", "api", "integration", "contract", "unit", "e2e"}
    invalid = set(request.categories) - valid_categories
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid test categories: {invalid}. Valid: {valid_categories}"
        )

    logger.info(f"Starting test run: categories={request.categories}")

    result = await run_pytest_async(request.categories, request.timeout)

    return result


@testing_router.get("/status", response_model=TestStatus)
async def get_test_status():
    """Get current test execution status."""
    return TestStatus(
        running=_test_state["running"],
        current_category=_test_state["current_category"],
        progress=0,  # Could calculate based on completed categories
        last_run=_test_state["last_run"]
    )


@testing_router.post("/stop")
async def stop_tests():
    """Stop currently running tests."""
    if not _test_state["running"]:
        return {"status": "no_tests_running"}

    process = _test_state.get("process")
    if process:
        try:
            process.terminate()
            await asyncio.sleep(1)
            if process.returncode is None:
                process.kill()
            logger.info("Tests stopped by user request")
        except Exception as e:
            logger.warning(f"Error stopping tests: {e}")

    _test_state["running"] = False
    _test_state["process"] = None
    _test_state["current_category"] = None

    return {"status": "stopped"}


@testing_router.get("/last-result", response_model=Optional[TestRunResponse])
async def get_last_result():
    """Get the result of the last test run."""
    return _test_state.get("last_result")


@testing_router.get("/categories")
async def list_categories():
    """List available test categories with descriptions."""
    return {
        "categories": [
            {
                "name": "smoke",
                "description": "Health checks for all 8 microservices",
                "requires_services": True
            },
            {
                "name": "api",
                "description": "API endpoint tests for all services",
                "requires_services": True
            },
            {
                "name": "integration",
                "description": "Service integration and data flow tests",
                "requires_services": True
            },
            {
                "name": "contract",
                "description": "API schema validation with Pydantic",
                "requires_services": True
            },
            {
                "name": "unit",
                "description": "Unit tests (mocked, no services required)",
                "requires_services": False
            },
            {
                "name": "e2e",
                "description": "End-to-end trading workflow tests",
                "requires_services": True
            }
        ]
    }
