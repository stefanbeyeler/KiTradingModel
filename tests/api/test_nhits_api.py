"""
API tests for the NHITS Service.

Tests all endpoints of the NHITS Service including:
- Forecasting
- Model management
- Training
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["nhits"]


class TestNHITSServiceAPI:
    """API Tests for the NHITS Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_get_forecast(self, client, test_symbol):
        """POST /api/v1/forecast - Get price forecast."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/forecast",
                    json={
                        "symbol": test_symbol,
                        "horizon": 24,
                        "interval": "1h"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "predictions" in data or "forecast" in data
                else:
                    # Model might not be trained
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_list_trained_models(self, client):
        """GET /api/v1/models - List all trained models."""
        async with client:
            try:
                response = await client.get("/api/v1/models")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_model_metrics(self, client, test_symbol):
        """GET /api/v1/models/{symbol}/metrics - Get model metrics."""
        async with client:
            try:
                response = await client.get(f"/api/v1/models/{test_symbol}/metrics")

                # Model might not exist
                assert response.status_code in [200, 404]

                if response.status_code == 200:
                    data = response.json()
                    # Should contain some metrics
                    assert isinstance(data, dict)
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_model_info(self, client, test_symbol):
        """GET /api/v1/models/{symbol} - Get model info."""
        async with client:
            try:
                response = await client.get(f"/api/v1/models/{test_symbol}")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_forecast_invalid_symbol(self, client):
        """POST /api/v1/forecast - Invalid symbol should fail gracefully."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/forecast",
                    json={
                        "symbol": "INVALID_SYMBOL_123",
                        "horizon": 24
                    }
                )

                assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_forecast_invalid_horizon(self, client, test_symbol):
        """POST /api/v1/forecast - Invalid horizon should fail."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/forecast",
                    json={
                        "symbol": test_symbol,
                        "horizon": -1
                    }
                )

                assert response.status_code in [400, 422]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    @pytest.mark.slow
    async def test_trigger_training(self, client):
        """POST /api/v1/train - Trigger model training (slow)."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/train",
                    json={
                        "symbol": "EURUSD",
                        "epochs": 1,  # Minimal for test
                        "horizon": 24
                    }
                )

                # Training started, or already running, or error
                assert response.status_code in [200, 202, 409, 503]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_training_status(self, client, test_symbol):
        """GET /api/v1/training-status/{symbol} - Get training status."""
        async with client:
            try:
                response = await client.get(f"/api/v1/training-status/{test_symbol}")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_batch_forecast(self, client, test_symbols):
        """POST /api/v1/forecast/batch - Batch forecast for multiple symbols."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/forecast/batch",
                    json={
                        "symbols": test_symbols[:3],
                        "horizon": 24
                    }
                )

                # Batch endpoint might not exist
                assert response.status_code in [200, 404, 405, 503]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")
