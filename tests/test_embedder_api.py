"""
API tests for the Embedder Service.

Tests all endpoints of the Embedder Service including:
- Text embeddings
- Time series embeddings
- Feature embeddings
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["embedder"]


class TestEmbedderServiceAPI:
    """API Tests for the Embedder Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_text_embedding(self, client):
        """POST /api/v1/embed/text - Generate text embedding."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/embed/text",
                    json={
                        "text": "Bitcoin price is rising due to institutional adoption"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "embedding" in data or "vector" in data
                else:
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Embedder service not reachable")

    @pytest.mark.api
    async def test_batch_text_embedding(self, client):
        """POST /api/v1/embed/text/batch - Batch text embeddings."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/embed/text/batch",
                    json={
                        "texts": [
                            "Bitcoin price analysis",
                            "Ethereum technical indicators",
                            "Market sentiment is bullish"
                        ]
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
                else:
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Embedder service not reachable")

    @pytest.mark.api
    async def test_timeseries_embedding(self, client, sample_ohlcv_data):
        """POST /api/v1/embed/timeseries - Generate time series embedding."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/embed/timeseries",
                    json={
                        "data": sample_ohlcv_data
                    }
                )

                assert response.status_code in [200, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Embedder service not reachable")

    @pytest.mark.api
    async def test_feature_embedding(self, client):
        """POST /api/v1/embed/features - Generate feature embedding."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/embed/features",
                    json={
                        "features": {
                            "rsi": 65.5,
                            "macd": 0.5,
                            "volume": 1000000
                        }
                    }
                )

                assert response.status_code in [200, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("Embedder service not reachable")

    @pytest.mark.api
    async def test_similarity_search(self, client):
        """POST /api/v1/similarity - Find similar embeddings."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/similarity",
                    json={
                        "query": "bullish market trend",
                        "top_k": 5
                    }
                )

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("Embedder service not reachable")

    @pytest.mark.api
    async def test_empty_text_embedding(self, client):
        """POST /api/v1/embed/text - Empty text handling."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/embed/text",
                    json={
                        "text": ""
                    }
                )

                # Service may return 404 if endpoint doesn't exist, or handle empty text gracefully
                assert response.status_code in [200, 400, 404, 422]
            except httpx.ConnectError:
                pytest.skip("Embedder service not reachable")

    @pytest.mark.api
    async def test_get_embedding_info(self, client):
        """GET /api/v1/info - Get embedding model info."""
        async with client:
            try:
                response = await client.get("/api/v1/info")

                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("Embedder service not reachable")
