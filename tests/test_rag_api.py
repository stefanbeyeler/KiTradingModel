"""
API tests for the RAG Service.

Tests all endpoints of the RAG Service including:
- Semantic search (query)
- Document management
- External sources integration
- Scheduler
"""
import pytest
import httpx

from conftest import SERVICE_URLS

BASE_URL = SERVICE_URLS["rag"]


class TestRAGServiceAPI:
    """API Tests for the RAG Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_semantic_search(self, client):
        """POST /api/v1/rag/query - Semantic search."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/rag/query",
                    json={
                        "query": "Bitcoin price prediction",
                        "top_k": 5
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (dict, list))
                else:
                    assert response.status_code in [404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_detailed_query(self, client):
        """POST /api/v1/rag/query/detailed - Detailed query with context."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/rag/query/detailed",
                    json={
                        "query": "market analysis",
                        "top_k": 5
                    }
                )

                assert response.status_code in [200, 404, 422, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_stats(self, client):
        """GET /api/v1/rag/stats - Get RAG statistics."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/stats")

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_list_sources(self, client):
        """GET /api/v1/rag/sources - List available data sources."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/sources")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_documents(self, client):
        """GET /api/v1/rag/documents - List documents."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/documents")

                assert response.status_code in [200, 404]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_sentiment(self, client):
        """GET /api/v1/rag/sentiment - Get sentiment data."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/sentiment")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_economic_calendar(self, client):
        """GET /api/v1/rag/economic-calendar - Get economic calendar."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/economic-calendar")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_macro(self, client):
        """GET /api/v1/rag/macro - Get macro data."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/macro")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_regulatory(self, client):
        """GET /api/v1/rag/regulatory - Get regulatory data."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/regulatory")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_historical_patterns(self, client):
        """GET /api/v1/rag/historical-patterns - Get historical patterns."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/historical-patterns")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_scheduler_status(self, client):
        """GET /api/v1/rag/scheduler/status - Get scheduler status."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/scheduler/status")

                assert response.status_code == 200
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_candlestick_types(self, client):
        """GET /api/v1/rag/candlestick-patterns/types - Get candlestick pattern types."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/candlestick-patterns/types")

                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_onchain_data(self, client, test_symbol):
        """GET /api/v1/rag/onchain/{symbol} - Get on-chain data."""
        async with client:
            try:
                response = await client.get(f"/api/v1/rag/onchain/{test_symbol}")

                assert response.status_code in [200, 404, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_easyinsight(self, client):
        """GET /api/v1/rag/easyinsight - Get EasyInsight data."""
        async with client:
            try:
                response = await client.get("/api/v1/rag/easyinsight")

                assert response.status_code in [200, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")
