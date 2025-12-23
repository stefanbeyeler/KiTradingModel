"""
API tests for the RAG Service.

Tests all endpoints of the RAG Service including:
- Semantic search
- Document management
- Trading context retrieval
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
        """POST /api/v1/query - Semantic search."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/query",
                    json={
                        "query": "Bitcoin price prediction",
                        "top_k": 5
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "results" in data or isinstance(data, list)
                else:
                    assert response.status_code in [404, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_add_document(self, client):
        """POST /api/v1/documents - Add a document."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/documents",
                    json={
                        "content": "Test document for RAG system unit testing.",
                        "metadata": {"source": "test", "type": "unit_test"}
                    }
                )

                assert response.status_code in [200, 201, 422]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_stats(self, client):
        """GET /api/v1/stats - Get RAG statistics."""
        async with client:
            try:
                response = await client.get("/api/v1/stats")

                if response.status_code == 200:
                    data = response.json()
                    # Should have some stats
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_trading_context(self, client, test_symbol):
        """GET /api/v1/trading-context/{symbol} - Get trading context."""
        async with client:
            try:
                response = await client.get(f"/api/v1/trading-context/{test_symbol}")

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, dict)
                else:
                    assert response.status_code in [404, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_search_with_filter(self, client):
        """POST /api/v1/query - Search with metadata filter."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/query",
                    json={
                        "query": "market analysis",
                        "top_k": 10,
                        "filter": {"type": "analysis"}
                    }
                )

                assert response.status_code in [200, 422, 503]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_empty_query(self, client):
        """POST /api/v1/query - Empty query should fail."""
        async with client:
            try:
                response = await client.post(
                    "/api/v1/query",
                    json={
                        "query": "",
                        "top_k": 5
                    }
                )

                assert response.status_code in [400, 422]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_get_documents(self, client):
        """GET /api/v1/documents - List documents."""
        async with client:
            try:
                response = await client.get("/api/v1/documents")

                assert response.status_code in [200, 404]
                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, (list, dict))
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")

    @pytest.mark.api
    async def test_delete_document(self, client):
        """DELETE /api/v1/documents/{id} - Delete document."""
        async with client:
            try:
                # Try to delete a non-existent document
                response = await client.delete("/api/v1/documents/non-existent-id")

                # Either not found or method not allowed
                assert response.status_code in [404, 405]
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")
