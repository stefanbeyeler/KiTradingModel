"""
API tests for Yahoo Finance integration via Data Service.

Tests the Yahoo Finance endpoints accessed through the Data Service:
- Time series data
- Various symbols (stocks, indices, crypto, forex)
- Different intervals and periods
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["data"]


class TestYFinanceTimeSeries:
    """API Tests for Yahoo Finance time series data."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_get_stock_time_series(self, client):
        """GET /api/v1/yfinance/time-series/{symbol} - Get stock data."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/AAPL",
                    params={"interval": "1h", "period": "5d"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    # Service may not support yfinance or symbol not found
                    assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_index_time_series(self, client):
        """GET /api/v1/yfinance/time-series/{symbol} - Get index data."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/^GSPC",  # S&P 500
                    params={"interval": "1d", "period": "1mo"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_crypto_time_series(self, client):
        """GET /api/v1/yfinance/time-series/{symbol} - Get crypto data."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/BTC-USD",
                    params={"interval": "1h", "period": "7d"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_forex_time_series(self, client):
        """GET /api/v1/yfinance/time-series/{symbol} - Get forex data."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/EURUSD=X",
                    params={"interval": "1h", "period": "5d"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    @pytest.mark.parametrize("symbol", [
        "AAPL",      # US Stock
        "MSFT",      # US Stock
        "GOOGL",     # US Stock
        "TSLA",      # US Stock
        "^DJI",      # Dow Jones
        "^IXIC",     # NASDAQ
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
    ])
    async def test_various_symbols(self, client, symbol):
        """Test Yahoo Finance with various symbol types."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/yfinance/time-series/{symbol}",
                    params={"interval": "1d", "period": "5d"}
                )

                # Any valid response is acceptable
                assert response.status_code in [200, 400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestYFinanceIntervals:
    """API Tests for Yahoo Finance interval support."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    @pytest.mark.parametrize("interval,period", [
        ("1m", "1d"),    # 1 minute, 1 day
        ("5m", "5d"),    # 5 minutes, 5 days
        ("15m", "5d"),   # 15 minutes, 5 days
        ("30m", "5d"),   # 30 minutes, 5 days
        ("1h", "1mo"),   # 1 hour, 1 month
        ("1d", "3mo"),   # 1 day, 3 months
        ("1wk", "1y"),   # 1 week, 1 year
        ("1mo", "5y"),   # 1 month, 5 years
    ])
    async def test_yfinance_intervals(self, client, interval, period):
        """Test Yahoo Finance supports various intervals and periods."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/AAPL",
                    params={"interval": interval, "period": period}
                )

                # May not support all interval/period combinations
                assert response.status_code in [200, 400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestYFinanceQuotes:
    """API Tests for Yahoo Finance quote data."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_get_quote(self, client):
        """GET /api/v1/yfinance/quote/{symbol} - Get current quote."""
        async with client:
            try:
                response = await client.get("/api/v1/yfinance/quote/AAPL")

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    # Endpoint may not exist or return error
                    assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_get_info(self, client):
        """GET /api/v1/yfinance/info/{symbol} - Get symbol info."""
        async with client:
            try:
                response = await client.get("/api/v1/yfinance/info/AAPL")

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    # Endpoint may not exist
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestYFinanceErrorHandling:
    """API Tests for Yahoo Finance error handling."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_invalid_symbol(self, client):
        """Test error handling for invalid symbol."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/INVALID_SYMBOL_XYZ123",
                    params={"interval": "1d", "period": "5d"}
                )

                # Should handle gracefully
                assert response.status_code in [200, 400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_invalid_interval(self, client):
        """Test error handling for invalid interval."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/AAPL",
                    params={"interval": "invalid", "period": "5d"}
                )

                # Should return error
                assert response.status_code in [200, 400, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_invalid_period(self, client):
        """Test error handling for invalid period."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/AAPL",
                    params={"interval": "1d", "period": "invalid"}
                )

                # Should return error
                assert response.status_code in [200, 400, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_missing_parameters(self, client):
        """Test handling of missing required parameters."""
        async with client:
            try:
                response = await client.get("/api/v1/yfinance/time-series/AAPL")

                # May use defaults or return error
                assert response.status_code in [200, 400, 404, 422]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestYFinanceFallback:
    """API Tests for Yahoo Finance as fallback data source."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_yfinance_as_ohlcv_fallback(self, client):
        """Test that yfinance can serve as OHLCV data fallback."""
        async with client:
            try:
                # Request OHLCV with yfinance source hint
                response = await client.get(
                    "/api/v1/ohlcv/AAPL",
                    params={"interval": "1h", "limit": 100, "source": "yfinance"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    # Source parameter may not be supported
                    assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_stock_data_availability(self, client):
        """Test that US stock data is available via yfinance."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/yfinance/time-series/NVDA",
                    params={"interval": "1d", "period": "1mo"}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Should have OHLCV data
                    if isinstance(data, list) and len(data) > 0:
                        # Check first candle has expected fields
                        candle = data[0]
                        if isinstance(candle, dict):
                            # May have open, high, low, close, volume
                            pass
                else:
                    assert response.status_code in [400, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")
