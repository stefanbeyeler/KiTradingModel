"""
API tests for the TCN Training Service.

Tests all endpoints of the TCN-Train Service including:
- Training status and history
- Auto-training configuration
- Model management
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["tcn-train"]


class TestTCNTrainServiceAPI:
    """API Tests for the TCN Training Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    # ========== Training Status ==========

    @pytest.mark.api
    async def test_get_training_status(self, client):
        """GET /api/v1/train/status - Get current training status."""
        async with client:
            try:
                response = await client.get("/api/v1/train/status")

                assert response.status_code == 200
                data = response.json()

                # Should have status field
                assert "status" in data
                assert data["status"] in ["idle", "training", "preparing", "completed", "failed"]

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    @pytest.mark.api
    async def test_get_training_history(self, client):
        """GET /api/v1/train/history - Get training history."""
        async with client:
            try:
                response = await client.get("/api/v1/train/history")

                assert response.status_code == 200
                data = response.json()

                # Should have history array
                assert "history" in data
                assert isinstance(data["history"], list)

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    # ========== Model Management ==========

    @pytest.mark.api
    async def test_list_models(self, client):
        """GET /api/v1/models - List available trained models."""
        async with client:
            try:
                response = await client.get("/api/v1/models")

                assert response.status_code == 200
                data = response.json()

                # Should have models array and count
                assert "models" in data
                assert "count" in data
                assert isinstance(data["models"], list)
                assert isinstance(data["count"], int)

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    # ========== Auto-Training ==========

    @pytest.mark.api
    async def test_get_auto_training_status(self, client):
        """GET /api/v1/auto-training/status - Get auto-training scheduler status."""
        async with client:
            try:
                response = await client.get("/api/v1/auto-training/status")

                assert response.status_code == 200
                data = response.json()

                # Should have configuration fields
                assert "enabled" in data
                assert "interval" in data
                assert "timeframes" in data
                assert isinstance(data["enabled"], bool)
                assert isinstance(data["timeframes"], list)

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    @pytest.mark.api
    async def test_get_scheduler_status(self, client):
        """GET /api/v1/scheduler - Get scheduler status (alias)."""
        async with client:
            try:
                response = await client.get("/api/v1/scheduler")

                assert response.status_code == 200
                data = response.json()

                # Should have same fields as auto-training/status
                assert "enabled" in data
                assert "interval" in data

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    # ========== Service Info ==========

    @pytest.mark.api
    async def test_health_endpoint(self, client):
        """GET /health - Health check endpoint."""
        async with client:
            try:
                response = await client.get("/health")

                assert response.status_code == 200
                data = response.json()

                assert "status" in data
                assert data["status"] == "healthy"
                assert "training_active" in data

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    @pytest.mark.api
    async def test_root_endpoint(self, client):
        """GET / - Root endpoint with service info."""
        async with client:
            try:
                response = await client.get("/")

                assert response.status_code == 200
                data = response.json()

                assert "service" in data
                assert "version" in data
                assert "endpoints" in data

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    # ========== Training Start (Read-Only Test) ==========

    @pytest.mark.api
    async def test_train_endpoint_validation(self, client):
        """POST /api/v1/train - Test parameter validation (no actual training)."""
        async with client:
            try:
                # Send request with minimal parameters
                response = await client.post(
                    "/api/v1/train",
                    json={
                        "epochs": 1,
                        "batch_size": 32
                    }
                )

                # Either starts training (201/200) or already running (409) or no symbols (400)
                assert response.status_code in [200, 201, 400, 409]

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")

    @pytest.mark.api
    async def test_stop_training_when_idle(self, client):
        """POST /api/v1/train/stop - Stop training when idle."""
        async with client:
            try:
                response = await client.post("/api/v1/train/stop")

                assert response.status_code == 200
                data = response.json()

                # When no training is running, should return not_running or stopping
                assert "status" in data
                assert data["status"] in ["not_running", "stopping"]

            except httpx.ConnectError:
                pytest.skip("TCN-Train service not reachable")
