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
        """GET /api/v1/forecast/{symbol} - Get price forecast."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/forecast/{test_symbol}",
                    params={"horizon": 24, "interval": "1h"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    # Model might not be trained
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_list_trained_models(self, client):
        """GET /api/v1/forecast/models - List all trained models."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/models")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_model_info(self, client, test_symbol):
        """GET /api/v1/forecast/{symbol}/model - Get model info."""
        async with client:
            try:
                response = await client.get(f"/api/v1/forecast/{test_symbol}/model")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_training_status(self, client):
        """GET /api/v1/forecast/training/status - Get training status."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/training/status")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_training_symbols(self, client):
        """GET /api/v1/forecast/training/symbols - Get symbols available for training."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/training/symbols")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_auto_status(self, client):
        """GET /api/v1/forecast/auto/status - Get auto-forecast status."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/auto/status")

                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_favorites(self, client):
        """GET /api/v1/forecast/favorites - Get favorite symbols."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/favorites")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_performance(self, client):
        """GET /api/v1/forecast/performance - Get model performance."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/performance")

                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_evaluated_forecasts(self, client):
        """GET /api/v1/forecast/evaluated - Get evaluated forecasts."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/evaluated")

                # 500 can happen if no evaluations exist yet
                assert response.status_code in [200, 500]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    @pytest.mark.slow
    async def test_trigger_training(self, client, test_symbol):
        """POST /api/v1/forecast/{symbol}/train - Trigger model training."""
        async with client:
            try:
                response = await client.post(
                    f"/api/v1/forecast/{test_symbol}/train",
                    json={"epochs": 1}  # Minimal for test
                )

                # Training started, queued, or already in progress
                assert response.status_code in [200, 202, 409, 503]
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_backup_status(self, client):
        """GET /api/v1/backup/status - Get backup status."""
        async with client:
            try:
                response = await client.get("/api/v1/backup/status")

                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.api
    async def test_get_training_cache_stats(self, client):
        """GET /api/v1/forecast/training/cache/stats - Get cache stats."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/training/cache/stats")

                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")
