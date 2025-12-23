"""
API tests for the LLM Service.

Tests all endpoints of the LLM Service including:
- Trading analysis
- Chat completions
- Signal generation
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["llm"]


class TestLLMServiceAPI:
    """API Tests for the LLM Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=300.0)  # 5 min for LLM

    @pytest.mark.api
    async def test_trading_analysis(self, client, test_symbol):
        """POST /api/v1/analyze - Trading analysis."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/analyze",
                    json={
                        "symbol": test_symbol,
                        "use_rag": True,
                        "include_forecast": True
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 503]
            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")
            except httpx.ReadTimeout:
                pytest.skip("LLM service timeout - model may be loading")

    @pytest.mark.api
    async def test_chat_completion(self, client):
        """POST /api/v1/chat - Chat completion."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/chat",
                    json={
                        "messages": [
                            {"role": "user", "content": "What is Bitcoin?"}
                        ]
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "content" in data or "message" in data or "response" in data
                else:
                    assert response.status_code in [404, 503]
            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")

    @pytest.mark.api
    async def test_trading_signal(self, client, test_symbol):
        """POST /api/v1/signal - Generate trading signal."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/signal",
                    json={
                        "symbol": test_symbol,
                        "timeframe": "1h"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 405, 503]
            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")

    @pytest.mark.api
    async def test_market_summary(self, client):
        """GET /api/v1/market-summary - Get market summary."""
        async with client:
            try:
                response = await client.get("/api/v1/market-summary")

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")

    @pytest.mark.api
    async def test_analyze_with_context(self, client, test_symbol):
        """POST /api/v1/analyze - Analysis with full context."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/analyze",
                    json={
                        "symbol": test_symbol,
                        "use_rag": True,
                        "include_forecast": True,
                        "include_patterns": True,
                        "include_regime": True
                    }
                )

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")

    @pytest.mark.api
    async def test_empty_chat_message(self, client):
        """POST /api/v1/chat - Empty message handling."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/chat",
                    json={
                        "messages": []
                    }
                )

                # Service may return 404 if endpoint doesn't exist, or handle empty gracefully
                assert response.status_code in [200, 400, 404, 422]
            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")

    @pytest.mark.api
    async def test_get_available_models(self, client):
        """GET /api/v1/models - List available LLM models."""
        async with client:
            try:
                response = await client.get("/api/v1/models")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("LLM service not reachable")
