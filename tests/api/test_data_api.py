"""
API tests for the Data Service.

Tests all endpoints of the Data Service including:
- Symbol management
- OHLCV data retrieval
- Technical indicators
- External sources
"""
import pytest
import httpx
from datetime import datetime, timedelta

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["data"]


class TestDataServiceSymbols:
    """API Tests for symbol management endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_get_symbols_list(self, client):
        """GET /api/v1/symbols - Get list of all symbols."""
        async with client:
            try:
                response = await client.get("/api/v1/symbols")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, list)

                # Check schema if data exists
                if data:
                    first_symbol = data[0]
                    assert "symbol" in first_symbol
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_symbol_detail(self, client, test_symbol):
        """GET /api/v1/symbols/{symbol} - Get symbol details."""
        async with client:
            try:
                response = await client.get(f"/api/v1/symbols/{test_symbol}")

                if response.status_code == 200:
                    data = response.json()
                    assert data["symbol"] == test_symbol
                else:
                    # Symbol might not exist
                    assert response.status_code in [404, 422]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_managed_symbols(self, client):
        """GET /api/v1/managed-symbols - Get managed symbols list."""
        async with client:
            try:
                response = await client.get("/api/v1/managed-symbols")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, list)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestDataServiceOHLCV:
    """API Tests for OHLCV data endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_get_ohlcv_data(self, client, test_symbol):
        """GET /api/v1/ohlcv/{symbol} - Get historical OHLCV data."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/ohlcv/{test_symbol}",
                    params={"interval": "1h", "limit": 100}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Can be list or dict with data key
                    assert "data" in data or isinstance(data, list)
                else:
                    # Symbol might not have data
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    @pytest.mark.parametrize("interval", ["1m", "5m", "15m", "1h", "4h", "1d"])
    async def test_get_ohlcv_different_intervals(self, client, test_symbol, interval):
        """GET /api/v1/ohlcv/{symbol} - Test different intervals."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/ohlcv/{test_symbol}",
                    params={"interval": interval, "limit": 10}
                )

                # Either success or data not available
                assert response.status_code in [200, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_ohlcv_invalid_interval(self, client, test_symbol):
        """GET /api/v1/ohlcv - Invalid interval should fail."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/ohlcv/{test_symbol}",
                    params={"interval": "invalid"}
                )

                assert response.status_code in [400, 422]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_ohlcv_with_date_range(self, client, test_symbol):
        """GET /api/v1/ohlcv/{symbol} - With date range."""
        async with client:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)

                response = await client.get(
                    f"/api/v1/ohlcv/{test_symbol}",
                    params={
                        "interval": "1h",
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat()
                    }
                )

                # Either success or parameters not supported
                assert response.status_code in [200, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestDataServiceTechnicalIndicators:
    """API Tests for TwelveData technical indicators."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    @pytest.mark.parametrize("indicator", ["rsi", "macd", "sma", "ema", "bbands"])
    async def test_technical_indicators(self, client, test_symbol, indicator):
        """GET /api/v1/twelvedata/{indicator}/{symbol} - Technical indicators."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/twelvedata/{indicator}/{test_symbol}",
                    params={"interval": "1h"}
                )

                # Success, not found, rate limit, or service unavailable
                assert response.status_code in [200, 404, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_rsi_indicator(self, client, test_symbol):
        """GET /api/v1/twelvedata/rsi/{symbol} - RSI indicator."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/twelvedata/rsi/{test_symbol}",
                    params={"interval": "1h", "time_period": 14}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "values" in data or isinstance(data, list)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_macd_indicator(self, client, test_symbol):
        """GET /api/v1/twelvedata/macd/{symbol} - MACD indicator."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/twelvedata/macd/{test_symbol}",
                    params={"interval": "1h"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "values" in data or isinstance(data, list)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestDataServiceExternalSources:
    """API Tests for external data sources endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_get_economic_calendar(self, client):
        """GET /api/v1/external-sources/economic-calendar."""
        async with client:
            try:
                response = await client.get("/api/v1/external-sources/economic-calendar")

                assert response.status_code in [200, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_sentiment_data(self, client):
        """GET /api/v1/external-sources/sentiment."""
        async with client:
            try:
                response = await client.get("/api/v1/external-sources/sentiment")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_onchain_data(self, client, test_symbol):
        """GET /api/v1/external-sources/onchain/{symbol}."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/external-sources/onchain/{test_symbol}"
                )

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_technical_levels(self, client, test_symbol):
        """GET /api/v1/external-sources/technical-levels/{symbol}."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/external-sources/technical-levels/{test_symbol}"
                )

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_fetch_all_sources(self, client, test_symbol):
        """POST /api/v1/external-sources/fetch-all."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/external-sources/fetch-all",
                    json={"symbol": test_symbol}
                )

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_trading_context(self, client, test_symbol):
        """POST /api/v1/external-sources/trading-context/{symbol}."""
        async with client:
            try:
                response = await client.post(
                    f"/api/v1/external-sources/trading-context/{test_symbol}"
                )

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")
