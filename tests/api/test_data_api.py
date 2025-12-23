"""
API tests for the Data Service.

Tests all endpoints of the Data Service including:
- Symbol management (managed-symbols)
- OHLCV data retrieval (twelvedata time_series)
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

    @pytest.mark.api
    async def test_get_managed_symbols_stats(self, client):
        """GET /api/v1/managed-symbols/stats - Get managed symbols stats."""
        async with client:
            try:
                response = await client.get("/api/v1/managed-symbols/stats")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, dict)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_search_symbols(self, client):
        """GET /api/v1/managed-symbols/search - Search symbols."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/managed-symbols/search",
                    params={"q": "BTC"}
                )

                # Might be 200 or 422 depending on implementation
                assert response.status_code in [200, 422]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_available_easyinsight_symbols(self, client):
        """GET /api/v1/managed-symbols/available/easyinsight - Available symbols from EasyInsight."""
        async with client:
            try:
                response = await client.get("/api/v1/managed-symbols/available/easyinsight")

                # Might return empty list or connection error
                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_yfinance_symbols(self, client):
        """GET /api/v1/yfinance/symbols - Get Yahoo Finance symbols."""
        async with client:
            try:
                response = await client.get("/api/v1/yfinance/symbols")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestDataServiceOHLCV:
    """API Tests for OHLCV data endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_get_time_series_data(self, client, test_symbol):
        """GET /api/v1/twelvedata/time_series/{symbol} - Get historical data."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/twelvedata/time_series/{test_symbol}",
                    params={"interval": "1h", "outputsize": 100}
                )

                # Success, symbol not found, rate limit, or service unavailable
                assert response.status_code in [200, 404, 422, 429, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_yfinance_time_series(self, client):
        """GET /api/v1/yfinance/time-series/{symbol} - Yahoo Finance time series."""
        async with client:
            try:
                # Use a common Yahoo Finance symbol
                response = await client.get(
                    "/api/v1/yfinance/time-series/AAPL",
                    params={"interval": "1h", "period": "5d"}
                )

                assert response.status_code in [200, 400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_training_data(self, client, test_symbol):
        """GET /api/v1/training-data/{symbol} - Get cached training data."""
        async with client:
            try:
                response = await client.get(f"/api/v1/training-data/{test_symbol}")

                # Data might not be cached
                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_live_data(self, client, test_symbol):
        """GET /api/v1/managed-symbols/live-data/{symbol} - Get live data."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/managed-symbols/live-data/{test_symbol}"
                )

                # Might not have live data
                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestDataServiceTechnicalIndicators:
    """API Tests for TwelveData technical indicators."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    @pytest.mark.parametrize("indicator", ["rsi", "macd", "sma", "ema", "bbands"])
    async def test_technical_indicators(self, client, indicator):
        """GET /api/v1/twelvedata/{indicator}/{symbol} - Technical indicators."""
        async with client:
            try:
                # Use EUR/USD which TwelveData recognizes
                response = await client.get(
                    f"/api/v1/twelvedata/{indicator}/EURUSD",
                    params={"interval": "1h"}
                )

                # Success, not found, rate limit, or service unavailable
                assert response.status_code in [200, 404, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_rsi_indicator(self, client):
        """GET /api/v1/twelvedata/rsi/{symbol} - RSI indicator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/rsi/EURUSD",
                    params={"interval": "1h", "time_period": 14}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Response should have indicator data or error info
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_macd_indicator(self, client):
        """GET /api/v1/twelvedata/macd/{symbol} - MACD indicator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/macd/EURUSD",
                    params={"interval": "1h"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_indicator_generic_endpoint(self, client):
        """GET /api/v1/twelvedata/indicator/{symbol}/{indicator} - Generic indicator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/indicator/EURUSD/rsi",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 404, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestDataServiceExternalSources:
    """API Tests for external data sources endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_list_external_sources(self, client):
        """GET /api/v1/external-sources - List available sources."""
        async with client:
            try:
                response = await client.get("/api/v1/external-sources")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

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
    async def test_get_macro_data(self, client):
        """GET /api/v1/external-sources/macro."""
        async with client:
            try:
                response = await client.get("/api/v1/external-sources/macro")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_regulatory_data(self, client):
        """GET /api/v1/external-sources/regulatory."""
        async with client:
            try:
                response = await client.get("/api/v1/external-sources/regulatory")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_correlations(self, client):
        """GET /api/v1/external-sources/correlations."""
        async with client:
            try:
                response = await client.get("/api/v1/external-sources/correlations")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_fetch_all_sources(self, client, test_symbol):
        """POST /api/v1/external-sources/fetch-all."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/external-sources/fetch-all",
                    json={"symbol": test_symbol, "sources": ["sentiment", "macro"]}
                )

                # 200 success or 422 validation error or 503 service unavailable
                assert response.status_code in [200, 422, 503]
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


class TestDataServicePatterns:
    """API Tests for candlestick pattern endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_scan_patterns(self, client, test_symbol):
        """GET /api/v1/patterns/scan/{symbol} - Scan for patterns."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/patterns/scan/{test_symbol}",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_pattern_summary(self, client, test_symbol):
        """GET /api/v1/patterns/summary/{symbol} - Get pattern summary."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/patterns/summary/{test_symbol}"
                )

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestDataServiceConfig:
    """API Tests for configuration endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_get_timezone(self, client):
        """GET /api/v1/config/timezone - Get current timezone."""
        async with client:
            try:
                response = await client.get("/api/v1/config/timezone")

                assert response.status_code == 200
                data = response.json()
                assert "timezone" in data or "display_timezone" in data
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_list_timezones(self, client):
        """GET /api/v1/config/timezones - List available timezones."""
        async with client:
            try:
                response = await client.get("/api/v1/config/timezones")

                assert response.status_code == 200
                data = response.json()
                # Response can be list or dict with timezones grouped
                assert isinstance(data, (list, dict))
                if isinstance(data, dict):
                    assert "timezones" in data or "current" in data
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_list_exports(self, client):
        """GET /api/v1/config/exports - List config exports."""
        async with client:
            try:
                response = await client.get("/api/v1/config/exports")

                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")
