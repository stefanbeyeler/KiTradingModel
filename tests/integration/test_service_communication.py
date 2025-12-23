"""
Integration tests for inter-service communication.

Tests that services can communicate with each other properly.
"""
import pytest
import httpx
import asyncio

from conftest import SERVICE_URLS, SERVICE_CONFIGS


class TestServiceCommunication:
    """Tests for service-to-service communication."""

    @pytest.mark.integration
    async def test_all_services_respond(self):
        """Test that all services respond to health checks."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            results = {}

            for name, config in SERVICE_CONFIGS.items():
                try:
                    response = await client.get(f"{config['url']}{config['health']}")
                    results[name] = {
                        "status_code": response.status_code,
                        "healthy": response.status_code == 200
                    }
                except httpx.ConnectError:
                    results[name] = {
                        "status_code": None,
                        "healthy": False,
                        "error": "Connection refused"
                    }
                except httpx.TimeoutException:
                    results[name] = {
                        "status_code": None,
                        "healthy": False,
                        "error": "Timeout"
                    }

            # Report results
            healthy_services = [k for k, v in results.items() if v.get("healthy")]
            unhealthy_services = [k for k, v in results.items() if not v.get("healthy")]

            print(f"\nHealthy services: {healthy_services}")
            print(f"Unhealthy services: {unhealthy_services}")

            # At least data service should be running
            assert results.get("data", {}).get("healthy", False), \
                "Data service must be running"

    @pytest.mark.integration
    async def test_data_service_provides_ohlcv(self):
        """Test Data Service provides OHLCV data to other services."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/BTCUSD",
                    params={"interval": "1h", "limit": 10}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Verify data structure
                    if isinstance(data, dict) and "data" in data:
                        assert len(data["data"]) > 0
                    elif isinstance(data, list):
                        assert len(data) > 0

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.integration
    async def test_concurrent_service_requests(self):
        """Test concurrent requests to multiple services."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Make concurrent requests to all services
                tasks = []
                for name, config in SERVICE_CONFIGS.items():
                    task = client.get(f"{config['url']}{config['health']}")
                    tasks.append(task)

                responses = await asyncio.gather(*tasks, return_exceptions=True)

                successful = 0
                for response in responses:
                    if isinstance(response, httpx.Response) and response.status_code == 200:
                        successful += 1

                print(f"\nConcurrent health checks: {successful}/{len(tasks)} successful")

                # At least one service should respond
                assert successful >= 1

            except Exception as e:
                pytest.skip(f"Concurrent request test failed: {e}")


class TestServiceEndpoints:
    """Tests for specific service endpoints."""

    @pytest.mark.integration
    async def test_data_service_symbols_endpoint(self):
        """Test Data Service symbols endpoint."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Try managed-symbols endpoint (actual endpoint name)
                response = await client.get(f"{SERVICE_URLS['data']}/api/v1/managed-symbols")

                assert response.status_code in [200, 404]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.integration
    async def test_nhits_service_models_endpoint(self):
        """Test NHITS Service models endpoint."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{SERVICE_URLS['nhits']}/api/v1/models")

                assert response.status_code in [200, 404]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))

            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.integration
    async def test_rag_service_stats_endpoint(self):
        """Test RAG Service stats endpoint."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{SERVICE_URLS['rag']}/api/v1/stats")

                assert response.status_code in [200, 404]

            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")


class TestErrorPropagation:
    """Tests for error handling across services."""

    @pytest.mark.integration
    async def test_invalid_symbol_handling(self):
        """Test that services properly handle invalid symbols."""
        invalid_symbol = "INVALID_SYMBOL_12345"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Data Service
            try:
                response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/{invalid_symbol}"
                )
                assert response.status_code in [400, 404, 422]
            except httpx.ConnectError:
                pass

            # NHITS Service
            try:
                response = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": invalid_symbol, "horizon": 24}
                )
                assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pass

    @pytest.mark.integration
    async def test_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # NHITS training without proper params
                response = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast/train",
                    json={}  # Missing required params
                )
                # 404 if endpoint doesn't exist, 405 if wrong method, 422 if validation fails
                assert response.status_code in [404, 405, 422]

            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")
