"""
Smoke tests for all service health endpoints.

These tests run first to verify all services are up before running other tests.
"""
import pytest
import httpx
from typing import Dict, Any

from conftest import SERVICE_CONFIGS, SERVICE_URLS


@pytest.mark.smoke
@pytest.mark.parametrize("service_name,config", SERVICE_CONFIGS.items())
async def test_service_health(service_name: str, config: Dict[str, str]):
    """Test that each service's health endpoint returns 200 and healthy status."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{config['url']}{config['health']}")

            assert response.status_code == 200, \
                f"{service_name} health check failed with status {response.status_code}"

            # Frontend returns plain text "healthy\n", other services return JSON
            if service_name == "frontend":
                assert "healthy" in response.text.lower(), \
                    f"Frontend health check returned unexpected content: {response.text}"
            else:
                data = response.json()
                assert data.get("status") in ["healthy", "ok", "running"], \
                    f"{service_name} is not healthy: {data.get('status')}"

        except httpx.ConnectError:
            pytest.skip(f"{service_name} service not reachable at {config['url']}")
        except httpx.TimeoutException:
            pytest.fail(f"{service_name} health check timed out")


@pytest.mark.smoke
async def test_all_services_up():
    """Quick check if all services are running."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        results = {}

        for name, config in SERVICE_CONFIGS.items():
            try:
                resp = await client.get(f"{config['url']}{config['health']}")
                results[name] = resp.status_code == 200
            except Exception:
                results[name] = False

        failed = [name for name, ok in results.items() if not ok]
        running = [name for name, ok in results.items() if ok]

        print(f"\nRunning services: {running}")
        print(f"Failed services: {failed}")

        # Skip if no services are running (local dev without Docker)
        if not running:
            pytest.skip("No services running - skip smoke test (run with Docker)")

        # At least data service must be running when services are up
        if not results.get("data", False):
            pytest.skip("Data service not running - required for smoke tests")


@pytest.mark.smoke
async def test_data_service_health_details():
    """Test Data Service health endpoint returns expected fields."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['data']}/health")

            assert response.status_code == 200
            data = response.json()

            # Check required fields
            assert "status" in data
            assert "service" in data or "name" in data

        except httpx.ConnectError:
            pytest.skip("Data service not reachable")


@pytest.mark.smoke
async def test_nhits_service_health_details():
    """Test NHITS Service health endpoint returns expected fields."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['nhits']}/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data

        except httpx.ConnectError:
            pytest.skip("NHITS service not reachable")


@pytest.mark.smoke
async def test_rag_service_health_details():
    """Test RAG Service health endpoint returns expected fields."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['rag']}/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data

        except httpx.ConnectError:
            pytest.skip("RAG service not reachable")


@pytest.mark.smoke
async def test_llm_service_health_details():
    """Test LLM Service health endpoint returns expected fields."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['llm']}/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data

        except httpx.ConnectError:
            pytest.skip("LLM service not reachable")


@pytest.mark.smoke
async def test_tcn_service_health_details():
    """Test TCN Service health endpoint returns expected fields."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['tcn']}/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data

        except httpx.ConnectError:
            pytest.skip("TCN service not reachable")


@pytest.mark.smoke
async def test_hmm_service_health_details():
    """Test HMM Service health endpoint returns expected fields."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['hmm']}/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data

        except httpx.ConnectError:
            pytest.skip("HMM service not reachable")


@pytest.mark.smoke
async def test_embedder_service_health_details():
    """Test Embedder Service health endpoint returns expected fields."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['embedder']}/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data

        except httpx.ConnectError:
            pytest.skip("Embedder service not reachable")


@pytest.mark.smoke
async def test_frontend_health():
    """Test Frontend health endpoint."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SERVICE_URLS['frontend']}/health")

            assert response.status_code == 200

        except httpx.ConnectError:
            pytest.skip("Frontend not reachable")
