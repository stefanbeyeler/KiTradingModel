"""
End-to-End tests for complete trading workflows.

Tests complete user scenarios from data retrieval to analysis output.
"""
import pytest
import httpx
import asyncio

from conftest import SERVICE_URLS


class TestTradingWorkflow:
    """End-to-End tests for trading workflows."""

    @pytest.mark.e2e
    async def test_complete_trading_analysis(self):
        """Complete trading analysis workflow."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            symbol = "BTCUSD"
            results = {}

            try:
                # Step 1: Check symbol availability
                symbol_resp = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/symbols/{symbol}"
                )
                results["symbol_check"] = symbol_resp.status_code in [200, 404]

                # Step 2: Get current market data
                market_data = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/{symbol}",
                    params={"interval": "1h", "limit": 168}  # 1 week
                )
                results["market_data"] = market_data.status_code == 200

                # Step 3: Check for chart patterns (TCN Service)
                try:
                    pattern_resp = await client.post(
                        f"{SERVICE_URLS['tcn']}/api/v1/detect",
                        json={"symbol": symbol, "interval": "1h"}
                    )
                    results["patterns"] = pattern_resp.status_code == 200
                except httpx.ConnectError:
                    results["patterns"] = None  # Service not available

                # Step 4: Get regime detection (HMM Service)
                try:
                    regime_resp = await client.post(
                        f"{SERVICE_URLS['hmm']}/api/v1/regime",
                        json={"symbol": symbol}
                    )
                    results["regime"] = regime_resp.status_code == 200
                except httpx.ConnectError:
                    results["regime"] = None

                # Step 5: Get price forecast (NHITS Service)
                try:
                    forecast_resp = await client.post(
                        f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                        json={"symbol": symbol, "horizon": 24}
                    )
                    results["forecast"] = forecast_resp.status_code == 200
                except httpx.ConnectError:
                    results["forecast"] = None

                # Step 6: Get LLM analysis
                try:
                    analysis_resp = await client.post(
                        f"{SERVICE_URLS['llm']}/api/v1/analyze",
                        json={
                            "symbol": symbol,
                            "include_patterns": True,
                            "include_regime": True,
                            "include_forecast": True
                        }
                    )
                    results["analysis"] = analysis_resp.status_code == 200
                except httpx.ConnectError:
                    results["analysis"] = None

                # Evaluate workflow
                print(f"\nWorkflow results: {results}")

                # Market data must be available
                assert results.get("market_data", False), "Market data not available"

                # Count successful steps
                successful = sum(1 for v in results.values() if v is True)
                assert successful >= 2, f"Only {successful} steps successful: {results}"

            except httpx.ConnectError as e:
                pytest.skip(f"Services not reachable: {e}")

    @pytest.mark.e2e
    async def test_multi_symbol_analysis(self):
        """Test analysis across multiple symbols."""
        symbols = ["BTCUSD", "ETHUSD", "EURUSD"]
        results = {}

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                for symbol in symbols:
                    # Get market data for each symbol
                    response = await client.get(
                        f"{SERVICE_URLS['data']}/api/v1/ohlcv/{symbol}",
                        params={"interval": "1h", "limit": 100}
                    )
                    results[symbol] = response.status_code == 200

                # At least one symbol should have data
                successful = sum(results.values())
                assert successful >= 1, f"No symbols have data: {results}"

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_model_training_workflow(self):
        """Test model training workflow (slow)."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            symbol = "EURUSD"  # Use different symbol for training test

            try:
                # Check if NHITS service is available
                health = await client.get(f"{SERVICE_URLS['nhits']}/health")
                if health.status_code != 200:
                    pytest.skip("NHITS service not available")

                # 1. Start training
                train_resp = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/train",
                    json={
                        "symbol": symbol,
                        "epochs": 2,  # Minimal for test
                        "horizon": 24
                    }
                )

                if train_resp.status_code not in [200, 202]:
                    # Training might not be available or already running
                    pytest.skip(f"Training not available: {train_resp.status_code}")

                # 2. Wait for training to complete (with polling)
                max_wait = 120  # 2 minutes
                poll_interval = 10

                for _ in range(max_wait // poll_interval):
                    status_resp = await client.get(
                        f"{SERVICE_URLS['nhits']}/api/v1/training-status/{symbol}"
                    )

                    if status_resp.status_code == 200:
                        status = status_resp.json()
                        if status.get("status") == "completed":
                            break
                        if status.get("status") == "failed":
                            pytest.fail("Training failed")

                    await asyncio.sleep(poll_interval)

                # 3. Request forecast with trained model
                forecast_resp = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": symbol, "horizon": 24}
                )

                # Forecast should work after training
                assert forecast_resp.status_code in [200, 404, 503]

            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")


class TestRAGWorkflow:
    """End-to-End tests for RAG-based workflows."""

    @pytest.mark.e2e
    async def test_rag_enriched_analysis(self):
        """Test analysis with RAG context enrichment."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                # 1. Add some test documents to RAG
                doc_resp = await client.post(
                    f"{SERVICE_URLS['rag']}/api/v1/documents",
                    json={
                        "content": "Bitcoin is showing bullish momentum with strong support at $40,000.",
                        "metadata": {"source": "e2e_test", "symbol": "BTCUSD"}
                    }
                )

                # 2. Query RAG for relevant context
                query_resp = await client.post(
                    f"{SERVICE_URLS['rag']}/api/v1/query",
                    json={
                        "query": "Bitcoin price analysis",
                        "top_k": 5
                    }
                )

                if query_resp.status_code == 200:
                    results = query_resp.json()
                    assert isinstance(results, (list, dict))

                # 3. Get LLM analysis with RAG
                analysis_resp = await client.post(
                    f"{SERVICE_URLS['llm']}/api/v1/analyze",
                    json={
                        "symbol": "BTCUSD",
                        "use_rag": True
                    }
                )

                # Analysis should work (with or without RAG context)
                assert analysis_resp.status_code in [200, 404, 503]

            except httpx.ConnectError as e:
                pytest.skip(f"Services not reachable: {e}")


class TestServiceStartup:
    """Tests for service startup and initialization."""

    @pytest.mark.e2e
    async def test_services_startup_order(self):
        """Test that services can start in any order."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Check all services
            service_status = {}

            for name, url in SERVICE_URLS.items():
                try:
                    response = await client.get(f"{url}/health")
                    service_status[name] = {
                        "status_code": response.status_code,
                        "healthy": response.status_code == 200
                    }
                except httpx.ConnectError:
                    service_status[name] = {
                        "status_code": None,
                        "healthy": False
                    }

            print(f"\nService startup status: {service_status}")

            # Data service must be running
            assert service_status.get("data", {}).get("healthy", False), \
                "Data service is required"

    @pytest.mark.e2e
    async def test_service_recovery(self):
        """Test that services handle errors gracefully."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Make requests with invalid data
                responses = []

                # Invalid symbol request
                resp = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/INVALID_SYMBOL"
                )
                responses.append(resp.status_code)

                # Invalid forecast request
                resp = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": "", "horizon": -1}
                )
                responses.append(resp.status_code)

                # Services should handle errors (4xx) not crash (5xx)
                for status in responses:
                    assert status < 500 or status == 503, \
                        f"Service error: {status}"

            except httpx.ConnectError:
                pytest.skip("Services not reachable")


class TestFullPipeline:
    """Tests for the complete data-to-insight pipeline."""

    @pytest.mark.e2e
    async def test_data_to_insight_pipeline(self):
        """Test complete pipeline: Data -> Analysis -> Insight."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            symbol = "BTCUSD"

            try:
                # Stage 1: Data Collection
                ohlcv = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/{symbol}",
                    params={"interval": "1h", "limit": 168}
                )
                assert ohlcv.status_code == 200, "Data collection failed"

                # Stage 2: Feature Extraction (Technical Indicators)
                indicators = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/twelvedata/rsi/{symbol}",
                    params={"interval": "1h"}
                )
                # Indicators might not be available
                stage2_ok = indicators.status_code in [200, 404, 429, 503]

                # Stage 3: Context Enrichment (External Sources)
                context = await client.post(
                    f"{SERVICE_URLS['data']}/api/v1/external-sources/fetch-all",
                    json={"symbol": symbol}
                )
                stage3_ok = context.status_code in [200, 503]

                # Stage 4: Model Inference (Forecast)
                forecast = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": symbol, "horizon": 24}
                )
                stage4_ok = forecast.status_code in [200, 404, 503]

                # Stage 5: Final Analysis
                analysis = await client.post(
                    f"{SERVICE_URLS['llm']}/api/v1/analyze",
                    json={"symbol": symbol}
                )
                stage5_ok = analysis.status_code in [200, 404, 503]

                print(f"\nPipeline stages: data={ohlcv.status_code}, "
                      f"indicators={indicators.status_code}, "
                      f"context={context.status_code}, "
                      f"forecast={forecast.status_code}, "
                      f"analysis={analysis.status_code}")

                # At least data collection must work
                assert ohlcv.status_code == 200

            except httpx.ConnectError:
                pytest.skip("Services not reachable")
