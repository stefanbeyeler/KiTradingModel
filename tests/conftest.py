"""
Global pytest fixtures for KI Trading Model test suite.
"""
import pytest
import httpx
import asyncio
import os
from typing import AsyncGenerator, Dict, Any
from datetime import datetime, timezone

# ========== Service URLs ==========

# Allow override via environment variables for different environments
SERVICE_HOST = os.getenv("TEST_SERVICE_HOST", "localhost")

SERVICE_URLS = {
    "frontend": f"http://{SERVICE_HOST}:3000",
    "data": f"http://{SERVICE_HOST}:3001",
    "nhits": f"http://{SERVICE_HOST}:3002",
    "tcn": f"http://{SERVICE_HOST}:3003",
    "hmm": f"http://{SERVICE_HOST}:3004",
    "embedder": f"http://{SERVICE_HOST}:3005",
    "rag": f"http://{SERVICE_HOST}:3008",
    "llm": f"http://{SERVICE_HOST}:3009",
}

SERVICE_CONFIGS = {
    "frontend": {"url": SERVICE_URLS["frontend"], "health": "/health"},
    "data": {"url": SERVICE_URLS["data"], "health": "/health"},
    "nhits": {"url": SERVICE_URLS["nhits"], "health": "/health"},
    "tcn": {"url": SERVICE_URLS["tcn"], "health": "/health"},
    "hmm": {"url": SERVICE_URLS["hmm"], "health": "/health"},
    "embedder": {"url": SERVICE_URLS["embedder"], "health": "/health"},
    "rag": {"url": SERVICE_URLS["rag"], "health": "/health"},
    "llm": {"url": SERVICE_URLS["llm"], "health": "/health"},
}


# ========== Fixtures ==========

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Shared async HTTP client."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest.fixture
async def data_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for Data Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["data"],
        timeout=30.0
    ) as client:
        yield client


@pytest.fixture
async def nhits_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for NHITS Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["nhits"],
        timeout=60.0
    ) as client:
        yield client


@pytest.fixture
async def tcn_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for TCN Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["tcn"],
        timeout=60.0
    ) as client:
        yield client


@pytest.fixture
async def hmm_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for HMM Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["hmm"],
        timeout=60.0
    ) as client:
        yield client


@pytest.fixture
async def embedder_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for Embedder Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["embedder"],
        timeout=60.0
    ) as client:
        yield client


@pytest.fixture
async def rag_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for RAG Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["rag"],
        timeout=60.0
    ) as client:
        yield client


@pytest.fixture
async def llm_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for LLM Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["llm"],
        timeout=120.0
    ) as client:
        yield client


# ========== Service Health Checks ==========

async def check_service_health(url: str, timeout: float = 10.0) -> bool:
    """Check if a service is healthy."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
async def ensure_services_running():
    """Ensure all required services are running before tests."""
    required = ["data"]  # Minimum required services

    for service in required:
        url = f"{SERVICE_URLS[service]}/health"
        is_healthy = await check_service_health(url)
        if not is_healthy:
            pytest.skip(f"{service} service not running or not healthy")


@pytest.fixture
async def ensure_data_service():
    """Ensure data service is running."""
    url = f"{SERVICE_URLS['data']}/health"
    is_healthy = await check_service_health(url)
    if not is_healthy:
        pytest.skip("Data service not running")


@pytest.fixture
async def ensure_nhits_service():
    """Ensure NHITS service is running."""
    url = f"{SERVICE_URLS['nhits']}/health"
    is_healthy = await check_service_health(url)
    if not is_healthy:
        pytest.skip("NHITS service not running")


@pytest.fixture
async def ensure_rag_service():
    """Ensure RAG service is running."""
    url = f"{SERVICE_URLS['rag']}/health"
    is_healthy = await check_service_health(url)
    if not is_healthy:
        pytest.skip("RAG service not running")


@pytest.fixture
async def ensure_llm_service():
    """Ensure LLM service is running."""
    url = f"{SERVICE_URLS['llm']}/health"
    is_healthy = await check_service_health(url)
    if not is_healthy:
        pytest.skip("LLM service not running")


# ========== Test Data Fixtures ==========

@pytest.fixture
def test_symbol() -> str:
    """Standard test symbol."""
    return "BTCUSD"


@pytest.fixture
def test_symbols() -> list:
    """List of test symbols."""
    return ["BTCUSD", "ETHUSD", "EURUSD", "XAUUSD", "GER40"]


@pytest.fixture
def sample_ohlcv_data() -> list:
    """Sample OHLCV data for testing."""
    return [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 102.0,
            "volume": 1000.0
        },
        {
            "timestamp": "2024-01-01T01:00:00Z",
            "open": 102.0,
            "high": 108.0,
            "low": 101.0,
            "close": 107.0,
            "volume": 1200.0
        },
        {
            "timestamp": "2024-01-01T02:00:00Z",
            "open": 107.0,
            "high": 110.0,
            "low": 105.0,
            "close": 109.0,
            "volume": 900.0
        },
    ]


@pytest.fixture
def valid_intervals() -> list:
    """Valid interval values."""
    return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]


@pytest.fixture
def service_urls() -> Dict[str, str]:
    """Get all service URLs."""
    return SERVICE_URLS.copy()


@pytest.fixture
def service_configs() -> Dict[str, Dict[str, str]]:
    """Get all service configurations."""
    return SERVICE_CONFIGS.copy()


# ========== Helper Functions ==========

def get_docker_logs(container_name: str, lines: int = 50) -> str:
    """Get logs from a Docker container."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Failed to get logs: {e}"
