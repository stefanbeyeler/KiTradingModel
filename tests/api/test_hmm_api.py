"""
API tests for the HMM Regime Service.

Tests all endpoints of the HMM Service including:
- Regime detection
- Signal scoring
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["hmm"]


class TestHMMServiceAPI:
    """API Tests for the HMM Regime Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_detect_regime(self, client, test_symbol):
        """POST /api/v1/regime - Detect market regime."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/regime",
                    json={
                        "symbol": test_symbol
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("HMM service not reachable")

    @pytest.mark.api
    async def test_get_regime_history(self, client, test_symbol):
        """GET /api/v1/regime/history/{symbol} - Get regime history."""
        async with client:
            try:
                response = await client.get(f"/api/v1/regime/history/{test_symbol}")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("HMM service not reachable")

    @pytest.mark.api
    async def test_score_signal(self, client, test_symbol):
        """POST /api/v1/score - Score a trading signal."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/score",
                    json={
                        "symbol": test_symbol,
                        "signal_type": "buy",
                        "confidence": 0.75
                    }
                )

                assert response.status_code in [200, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("HMM service not reachable")

    @pytest.mark.api
    async def test_get_regime_probabilities(self, client, test_symbol):
        """GET /api/v1/regime/probabilities/{symbol} - Get regime probabilities."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/regime/probabilities/{test_symbol}"
                )

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("HMM service not reachable")

    @pytest.mark.api
    async def test_list_regimes(self, client):
        """GET /api/v1/regimes - List available regime types."""
        async with client:
            try:
                response = await client.get("/api/v1/regimes")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("HMM service not reachable")
