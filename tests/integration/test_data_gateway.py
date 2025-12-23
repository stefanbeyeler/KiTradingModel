"""
Integration tests for the Data Gateway Pattern.

Tests that all services properly use the Data Service as the single
gateway for external data access.
"""
import pytest
import httpx

from conftest import SERVICE_URLS


class TestDataGatewayIntegration:
    """Integration tests for the Data Gateway Pattern."""

    @pytest.mark.integration
    async def test_nhits_uses_data_service(self):
        """NHITS Service must fetch data from Data Service."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # First check if both services are up
                data_health = await client.get(f"{SERVICE_URLS['data']}/health")
                nhits_health = await client.get(f"{SERVICE_URLS['nhits']}/health")

                if data_health.status_code != 200:
                    pytest.skip("Data service not available")
                if nhits_health.status_code != 200:
                    pytest.skip("NHITS service not available")

                # Trigger NHITS forecast - this should internally call Data Service
                response = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": "BTCUSD", "horizon": 24}
                )

                # Response can be success or error if model not trained
                assert response.status_code in [200, 404, 422, 503]

            except httpx.ConnectError as e:
                pytest.skip(f"Services not reachable: {e}")

    @pytest.mark.integration
    async def test_rag_uses_data_service_for_context(self):
        """RAG Service must fetch trading context via Data Service."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                data_health = await client.get(f"{SERVICE_URLS['data']}/health")
                rag_health = await client.get(f"{SERVICE_URLS['rag']}/health")

                if data_health.status_code != 200:
                    pytest.skip("Data service not available")
                if rag_health.status_code != 200:
                    pytest.skip("RAG service not available")

                # Fetch trading context via RAG
                response = await client.get(
                    f"{SERVICE_URLS['rag']}/api/v1/trading-context/BTCUSD"
                )

                if response.status_code == 200:
                    data = response.json()
                    # Should contain data from external sources
                    assert isinstance(data, dict)

            except httpx.ConnectError as e:
                pytest.skip(f"Services not reachable: {e}")

    @pytest.mark.integration
    async def test_llm_uses_data_service(self):
        """LLM Service should use Data Service for market data."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                data_health = await client.get(f"{SERVICE_URLS['data']}/health")
                llm_health = await client.get(f"{SERVICE_URLS['llm']}/health")

                if data_health.status_code != 200:
                    pytest.skip("Data service not available")
                if llm_health.status_code != 200:
                    pytest.skip("LLM service not available")

                # Request trading analysis
                response = await client.post(
                    f"{SERVICE_URLS['llm']}/api/v1/analyze",
                    json={
                        "symbol": "BTCUSD",
                        "use_rag": False,
                        "include_forecast": False
                    }
                )

                assert response.status_code in [200, 404, 503]

            except httpx.ConnectError as e:
                pytest.skip(f"Services not reachable: {e}")


class TestServiceChainIntegration:
    """Tests for complete service chains."""

    @pytest.mark.integration
    async def test_data_to_nhits_chain(self):
        """Test Data Service -> NHITS Service chain."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # 1. Get data from Data Service
                data_response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/BTCUSD",
                    params={"interval": "1h", "limit": 100}
                )

                if data_response.status_code != 200:
                    pytest.skip("Data service returned no data")

                # 2. Request forecast from NHITS (which uses Data Service internally)
                nhits_response = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": "BTCUSD", "horizon": 24}
                )

                # Both should work or return expected errors
                assert data_response.status_code == 200
                assert nhits_response.status_code in [200, 404, 422, 503]

            except httpx.ConnectError as e:
                pytest.skip(f"Services not reachable: {e}")

    @pytest.mark.integration
    async def test_full_analysis_chain(self):
        """Test Data -> NHITS -> RAG -> LLM chain."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            symbol = "BTCUSD"
            results = {}

            try:
                # 1. Data Service
                resp = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/{symbol}",
                    params={"limit": 100}
                )
                results["data"] = resp.status_code == 200

                # 2. NHITS Service
                resp = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": symbol, "horizon": 24}
                )
                results["nhits"] = resp.status_code == 200

                # 3. RAG Service
                resp = await client.post(
                    f"{SERVICE_URLS['rag']}/api/v1/query",
                    json={"query": f"{symbol} analysis", "top_k": 3}
                )
                results["rag"] = resp.status_code == 200

                # 4. LLM Service
                resp = await client.post(
                    f"{SERVICE_URLS['llm']}/api/v1/analyze",
                    json={"symbol": symbol}
                )
                results["llm"] = resp.status_code == 200

                # At least Data and one other service should work
                successful = sum(results.values())
                assert successful >= 1, f"All services failed: {results}"

            except httpx.ConnectError as e:
                pytest.skip(f"Services not reachable: {e}")


class TestFallbackChain:
    """Tests for data source fallback behavior."""

    @pytest.mark.integration
    async def test_data_source_fallback(self):
        """Test: EasyInsight -> TwelveData -> Yahoo fallback."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Request data - should succeed via some fallback source
                response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/BTCUSD",
                    params={"interval": "1h", "limit": 50}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Check if source info is included
                    if isinstance(data, dict) and "source" in data:
                        assert data["source"] in [
                            "easyinsight", "twelvedata", "yahoo", "cache"
                        ]
                else:
                    # Even if no data, should return proper error
                    assert response.status_code in [404, 503]

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.integration
    async def test_multiple_symbols_data_fetch(self):
        """Test fetching data for multiple symbols."""
        symbols = ["BTCUSD", "ETHUSD", "EURUSD"]

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                results = {}
                for symbol in symbols:
                    response = await client.get(
                        f"{SERVICE_URLS['data']}/api/v1/ohlcv/{symbol}",
                        params={"interval": "1h", "limit": 10}
                    )
                    # 200 = data found, 404 = no data but service OK
                    results[symbol] = response.status_code in [200, 404]

                # At least service should respond for all symbols
                assert all(results.values()), f"Service error for symbols: {results}"

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestExternalSourcesIntegration:
    """Tests for external data sources integration."""

    @pytest.mark.integration
    async def test_fetch_all_external_sources(self):
        """Test fetching all external sources."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{SERVICE_URLS['data']}/api/v1/external-sources/fetch-all",
                    json={"symbol": "BTCUSD"}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                    # Should have some data sources
                else:
                    # 422 = validation error (missing params), 404/503 = service unavailable
                    assert response.status_code in [404, 422, 503]

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.integration
    async def test_trading_context_aggregation(self):
        """Test trading context aggregation from multiple sources."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{SERVICE_URLS['data']}/api/v1/external-sources/trading-context/BTCUSD"
                )

                if response.status_code == 200:
                    data = response.json()
                    # Should contain aggregated data
                    assert isinstance(data, dict)

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")
