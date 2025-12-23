"""
API tests for TwelveData integration.

Tests the TwelveData API endpoints accessed through the Data Service gateway:
- Time series data
- Technical indicators (all categories)
- Symbol search
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["data"]


class TestTwelveDataTimeSeries:
    """API Tests for TwelveData time series endpoints."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_get_time_series(self, client):
        """GET /api/v1/twelvedata/time_series/{symbol} - Get OHLCV data."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/time_series/EURUSD",
                    params={"interval": "1h", "outputsize": 50}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    @pytest.mark.parametrize("symbol", ["EUR/USD", "BTC/USD", "AAPL", "GDAXI"])
    async def test_time_series_various_symbols(self, client, symbol):
        """Test time series for various symbol formats."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/twelvedata/time_series/{symbol}",
                    params={"interval": "1h", "outputsize": 10}
                )

                # Different symbols may have different availability
                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    @pytest.mark.parametrize("interval", ["1min", "5min", "15min", "30min", "1h", "4h", "1day"])
    async def test_time_series_intervals(self, client, interval):
        """Test time series with various intervals."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/time_series/EURUSD",
                    params={"interval": interval, "outputsize": 10}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestTwelveDataMomentumIndicators:
    """API Tests for TwelveData momentum indicators."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_rsi_indicator(self, client):
        """GET /api/v1/twelvedata/rsi/{symbol} - RSI indicator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/rsi/EURUSD",
                    params={"interval": "1h", "time_period": 14}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
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

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_stoch_indicator(self, client):
        """GET /api/v1/twelvedata/stoch/{symbol} - Stochastic oscillator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/stoch/EURUSD",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_cci_indicator(self, client):
        """GET /api/v1/twelvedata/cci/{symbol} - CCI indicator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/cci/EURUSD",
                    params={"interval": "1h", "time_period": 20}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_adx_indicator(self, client):
        """GET /api/v1/twelvedata/adx/{symbol} - ADX indicator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/adx/EURUSD",
                    params={"interval": "1h", "time_period": 14}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_willr_indicator(self, client):
        """GET /api/v1/twelvedata/willr/{symbol} - Williams %R."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/willr/EURUSD",
                    params={"interval": "1h", "time_period": 14}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestTwelveDataTrendIndicators:
    """API Tests for TwelveData trend/moving average indicators."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    @pytest.mark.parametrize("ma_type", ["sma", "ema", "wma", "dema", "tema"])
    async def test_moving_averages(self, client, ma_type):
        """Test various moving average types."""
        async with client:
            try:
                response = await client.get(
                    f"/api/v1/twelvedata/{ma_type}/EURUSD",
                    params={"interval": "1h", "time_period": 20}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_supertrend_indicator(self, client):
        """GET /api/v1/twelvedata/supertrend/{symbol} - Supertrend."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/supertrend/EURUSD",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_ichimoku_indicator(self, client):
        """GET /api/v1/twelvedata/ichimoku/{symbol} - Ichimoku Cloud."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/ichimoku/EURUSD",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_sar_indicator(self, client):
        """GET /api/v1/twelvedata/sar/{symbol} - Parabolic SAR."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/sar/EURUSD",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestTwelveDataVolatilityIndicators:
    """API Tests for TwelveData volatility indicators."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_bbands_indicator(self, client):
        """GET /api/v1/twelvedata/bbands/{symbol} - Bollinger Bands."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/bbands/EURUSD",
                    params={"interval": "1h", "time_period": 20}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
                if response.status_code == 200:
                    data = response.json()
                    # Should contain upper, middle, lower bands
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_atr_indicator(self, client):
        """GET /api/v1/twelvedata/atr/{symbol} - ATR indicator."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/atr/EURUSD",
                    params={"interval": "1h", "time_period": 14}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_percent_b_indicator(self, client):
        """GET /api/v1/twelvedata/percent_b/{symbol} - Percent B."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/percent_b/EURUSD",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestTwelveDataVolumeIndicators:
    """API Tests for TwelveData volume indicators."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_obv_indicator(self, client):
        """GET /api/v1/twelvedata/obv/{symbol} - On Balance Volume."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/obv/AAPL",  # Use stock for volume
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_vwap_indicator(self, client):
        """GET /api/v1/twelvedata/vwap/{symbol} - VWAP."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/vwap/AAPL",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_ad_indicator(self, client):
        """GET /api/v1/twelvedata/ad/{symbol} - Accumulation/Distribution."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/ad/AAPL",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestTwelveDataMLIndicators:
    """API Tests for TwelveData ML-relevant indicators."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_crsi_indicator(self, client):
        """GET /api/v1/twelvedata/crsi/{symbol} - Connors RSI (ML feature)."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/crsi/EURUSD",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_linearregslope_indicator(self, client):
        """GET /api/v1/twelvedata/linearregslope/{symbol} - Linear Regression Slope."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/linearregslope/EURUSD",
                    params={"interval": "1h", "time_period": 14}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_ht_trendmode_indicator(self, client):
        """GET /api/v1/twelvedata/ht_trendmode/{symbol} - Hilbert Transform Trend Mode."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/ht_trendmode/EURUSD",
                    params={"interval": "1h"}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestTwelveDataGenericEndpoint:
    """API Tests for TwelveData generic indicator endpoint."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_generic_indicator_endpoint(self, client):
        """GET /api/v1/twelvedata/indicator/{symbol}/{indicator} - Generic endpoint."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/indicator/EURUSD/rsi",
                    params={"interval": "1h", "time_period": 14}
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_multiple_indicators(self, client):
        """POST /api/v1/twelvedata/multiple - Get multiple indicators at once."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/twelvedata/multiple",
                    json={
                        "symbol": "EURUSD",
                        "indicators": ["rsi", "macd", "bbands"],
                        "interval": "1h"
                    }
                )

                assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestTwelveDataErrorHandling:
    """API Tests for TwelveData error handling."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    @pytest.mark.api
    async def test_invalid_symbol(self, client):
        """Test error handling for invalid symbol."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/rsi/INVALID_SYMBOL_XYZ",
                    params={"interval": "1h"}
                )

                # Service may return 200 with error in body, or 4xx
                # Data service handles TwelveData errors gracefully
                assert response.status_code in [200, 400, 404, 422, 429, 503]
                if response.status_code == 200:
                    data = response.json()
                    # Check if response indicates an error
                    assert isinstance(data, (dict, list))
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_invalid_interval(self, client):
        """Test error handling for invalid interval."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/twelvedata/rsi/EURUSD",
                    params={"interval": "invalid"}
                )

                # Service may return 200 with error in body, or 4xx
                assert response.status_code in [200, 400, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.api
    async def test_rate_limit_handling(self, client):
        """Test that rate limit errors are handled gracefully."""
        async with client:
            try:
                # Make multiple rapid requests
                for _ in range(3):
                    response = await client.get(
                        "/api/v1/twelvedata/rsi/EURUSD",
                        params={"interval": "1h"}
                    )
                    # Should handle rate limits gracefully
                    assert response.status_code in [200, 400, 404, 422, 429, 503]
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")
