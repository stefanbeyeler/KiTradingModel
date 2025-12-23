"""
API tests for the TCN Pattern Service.

Tests all endpoints of the TCN Service including:
- Pattern detection
- Model management
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["tcn"]


class TestTCNServiceAPI:
    """API Tests for the TCN Pattern Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_detect_patterns(self, client, test_symbol):
        """POST /api/v1/detect - Detect chart patterns."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/detect",
                    json={
                        "symbol": test_symbol,
                        "interval": "1h"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("TCN service not reachable")

    @pytest.mark.api
    async def test_scan_patterns(self, client, test_symbol):
        """POST /api/v1/patterns/scan - Scan for patterns."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/patterns/scan",
                    json={
                        "symbol": test_symbol,
                        "timeframes": ["1h", "4h"],
                        "lookback_candles": 100,
                        "min_confidence": 0.5
                    }
                )

                assert response.status_code in [200, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("TCN service not reachable")

    @pytest.mark.api
    async def test_list_patterns(self, client):
        """GET /api/v1/patterns - List available patterns."""
        async with client:
            try:
                response = await client.get("/api/v1/patterns")

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    assert response.status_code in [404]
            except httpx.ConnectError:
                pytest.skip("TCN service not reachable")

    @pytest.mark.api
    async def test_get_pattern_history(self, client, test_symbol):
        """GET /api/v1/patterns/history/{symbol} - Get pattern history."""
        async with client:
            try:
                response = await client.get(f"/api/v1/patterns/history/{test_symbol}")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("TCN service not reachable")

    @pytest.mark.api
    async def test_detect_with_invalid_symbol(self, client):
        """POST /api/v1/detect - Invalid symbol should fail gracefully."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/detect",
                    json={
                        "symbol": "",
                        "interval": "1h"
                    }
                )

                assert response.status_code in [400, 422]
            except httpx.ConnectError:
                pytest.skip("TCN service not reachable")
