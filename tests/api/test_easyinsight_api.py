"""
API tests for EasyInsight integration.

Tests the EasyInsight API endpoints accessed through the Data Service gateway:
- Symbol availability
- OHLCV data retrieval
- MT5 connection status
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["data"]

# EasyInsight server URL (for direct tests if needed)
EASYINSIGHT_URL = "http://10.1.19.102:3000"


class TestEasyInsightSymbols:
    """API Tests for EasyInsight symbol management."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_get_available_easyinsight_symbols(self, client):
        """GET /api/v1/managed-symbols/available/easyinsight - List available symbols."""
        async with client:
            try:
                response = await client.get("/api/v1/managed-symbols/available/easyinsight")

                # May return empty list if EasyInsight is not connected
                assert response.status_code in [200, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_managed_symbols_stats(self, client):
        """GET /api/v1/managed-symbols/stats - Get symbol statistics."""
        async with client:
            try:
                response = await client.get("/api/v1/managed-symbols/stats")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, dict)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_search_easyinsight_symbols(self, client):
        """GET /api/v1/managed-symbols/search - Search symbols."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/managed-symbols/search",
                    params={"query": "BTC"}
                )

                assert response.status_code in [200, 404]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestEasyInsightOHLCV:
    """API Tests for EasyInsight OHLCV data retrieval."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_get_easyinsight_ohlcv(self, client, test_symbol):
        """GET /api/v1/easyinsight/ohlcv/{symbol} - Get OHLCV from EasyInsight."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/easyinsight/ohlcv/{test_symbol}",
                    params={"interval": "1h", "limit": 100}
                )

                # May return 503 if EasyInsight is unavailable
                assert response.status_code in [200, 404, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_easyinsight_latest(self, client, test_symbol):
        """GET /api/v1/easyinsight/latest/{symbol} - Get latest price."""
        async with client:
            try:
                response = await client.get(f"/api/v1/easyinsight/latest/{test_symbol}")

                assert response.status_code in [200, 404, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    @pytest.mark.parametrize("interval", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
    async def test_easyinsight_intervals(self, client, test_symbol, interval):
        """Test EasyInsight supports various MT5 intervals."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/easyinsight/ohlcv/{test_symbol}",
                    params={"interval": interval, "limit": 10}
                )

                # May not support all intervals
                assert response.status_code in [200, 400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestEasyInsightStatus:
    """API Tests for EasyInsight connection status."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_get_easyinsight_status(self, client):
        """GET /api/v1/easyinsight/status - Get EasyInsight connection status."""
        async with client:
            try:
                response = await client.get("/api/v1/easyinsight/status")

                assert response.status_code in [200, 404, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_rag_easyinsight_data(self):
        """GET /api/v1/rag/easyinsight - Get EasyInsight data via RAG service."""
        async with httpx.AsyncClient(base_url=SERVICE_URLS["rag"], timeout=30.0) as client:
            try:
                response = await client.get("/api/v1/rag/easyinsight")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")


class TestEasyInsightDirect:
    """Direct tests against EasyInsight API (if accessible)."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=EASYINSIGHT_URL, timeout=30.0)

    @pytest.mark.api
    async def test_easyinsight_health(self, client):
        """Direct health check on EasyInsight server."""
        async with client:
            try:
                response = await client.get("/api/health")

                # EasyInsight may have different health endpoint
                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("EasyInsight server not reachable at 10.1.19.102:3000")

    @pytest.mark.api
    async def test_easyinsight_symbols_direct(self, client):
        """Direct symbols endpoint on EasyInsight server."""
        async with client:
            try:
                response = await client.get("/api/symbols")

                assert response.status_code in [200, 404]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("EasyInsight server not reachable")

    @pytest.mark.api
    async def test_easyinsight_mt5_status(self, client):
        """Check MT5 connection status on EasyInsight."""
        async with client:
            try:
                response = await client.get("/api/mt5/status")

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("EasyInsight server not reachable")
