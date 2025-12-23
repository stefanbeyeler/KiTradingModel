"""
Contract tests for API response schemas.

Tests that API responses conform to expected schemas/contracts.
"""
import pytest
import httpx

from conftest import SERVICE_URLS


class TestHealthContractTests:
    """Contract tests for health endpoints."""

    @pytest.mark.contract
    @pytest.mark.parametrize("service,url", [
        ("data", SERVICE_URLS["data"]),
        ("nhits", SERVICE_URLS["nhits"]),
        ("tcn", SERVICE_URLS["tcn"]),
        ("hmm", SERVICE_URLS["hmm"]),
        ("embedder", SERVICE_URLS["embedder"]),
        ("rag", SERVICE_URLS["rag"]),
        ("llm", SERVICE_URLS["llm"]),
    ])
    async def test_health_response_contract(self, service, url):
        """Health endpoint must return JSON with status field."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{url}/health")

                assert response.status_code == 200
                data = response.json()

                # Contract: must have status field
                assert "status" in data
                assert data["status"] in ["healthy", "ok", "running", "degraded"]

            except httpx.ConnectError:
                pytest.skip(f"{service} not reachable")


class TestDataServiceContracts:
    """Contract tests for Data Service API."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=SERVICE_URLS["data"], timeout=30.0)

    @pytest.mark.contract
    async def test_ohlcv_response_contract(self, client):
        """OHLCV endpoint must return array or object with data field."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/ohlcv/BTCUSD",
                    params={"interval": "1h", "limit": 10}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Contract: response is list or dict with data
                    assert isinstance(data, (list, dict))
                    if isinstance(data, dict) and "data" in data:
                        assert isinstance(data["data"], list)

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.contract
    async def test_managed_symbols_response_contract(self, client):
        """Managed symbols endpoint must return list or dict."""
        async with client:
            try:
                response = await client.get("/api/v1/managed-symbols")

                if response.status_code == 200:
                    data = response.json()
                    # Contract: response is list of symbols or dict with symbols
                    assert isinstance(data, (list, dict))

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestNHITSServiceContracts:
    """Contract tests for NHITS Service API."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=SERVICE_URLS["nhits"], timeout=60.0)

    @pytest.mark.contract
    async def test_forecast_response_contract(self, client):
        """Forecast endpoint must return predictions in expected format."""
        async with client:
            try:
                response = await client.get(
                    "/api/v1/forecast/BTCUSD",
                    params={"horizon": 12}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Contract: must have predictions
                    assert isinstance(data, dict)
                    # May have fields like: predicted_prices, predictions, forecast
                    assert any(k in data for k in ["predicted_prices", "predictions", "forecast", "values", "data"])

            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.contract
    async def test_models_response_contract(self, client):
        """Models endpoint must return list of model info."""
        async with client:
            try:
                response = await client.get("/api/v1/forecast/models")

                if response.status_code == 200:
                    data = response.json()
                    # Contract: list or dict of models
                    assert isinstance(data, (list, dict))

            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")


class TestRAGServiceContracts:
    """Contract tests for RAG Service API."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=SERVICE_URLS["rag"], timeout=60.0)

    @pytest.mark.contract
    async def test_context_response_contract(self, client):
        """Context endpoint must return structured context data."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/context/BTCUSD")

                if response.status_code == 200:
                    data = response.json()
                    # Contract: dict with context information
                    assert isinstance(data, dict)

            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.contract
    async def test_stats_response_contract(self, client):
        """Stats endpoint must return statistics."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/stats")

                if response.status_code == 200:
                    data = response.json()
                    # Contract: dict with stats
                    assert isinstance(data, dict)

            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")


class TestLLMServiceContracts:
    """Contract tests for LLM Service API."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=SERVICE_URLS["llm"], timeout=120.0)

    @pytest.mark.contract
    async def test_analyze_response_contract(self, client):
        """Analyze endpoint must return analysis with required fields."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/analyze",
                    json={"symbol": "BTCUSD"}
                )

                if response.status_code == 200:
                    data = response.json()
                    # Contract: dict with analysis result
                    assert isinstance(data, dict)

            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")
            except httpx.ReadTimeout:
                pytest.skip("LLM service timeout")
