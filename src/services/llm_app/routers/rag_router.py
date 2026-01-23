"""RAG Router - RAG proxy endpoints for the LLM Service.

All RAG operations are proxied to the RAG Service (Port 3008) via HTTP.
This avoids dependency conflicts with sentence-transformers.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from loguru import logger
import httpx

rag_router = APIRouter()

# RAG Service URL (accessed via HTTP, not direct import)
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://trading-rag:3008")


async def _call_rag_service(method: str, endpoint: str, params: dict = None, json: dict = None):
    """Make HTTP call to RAG Service."""
    url = f"{RAG_SERVICE_URL}/api/v1{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                response = await client.get(url, params=params)
            elif method == "POST":
                response = await client.post(url, json=json)
            elif method == "DELETE":
                response = await client.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"RAG Service request error: {e}")
        raise HTTPException(status_code=503, detail=f"RAG service not available: {e}")


@rag_router.get("/rag/health")
async def rag_health():
    """Check RAG Service health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{RAG_SERVICE_URL}/health")
            if response.status_code == 200:
                return {"status": "healthy", "rag_url": RAG_SERVICE_URL}
            return {"status": "unhealthy", "code": response.status_code}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


@rag_router.post("/rag/document")
async def add_rag_document(
    content: str,
    document_type: str,
    symbol: Optional[str] = None,
    metadata: Optional[dict] = None
):
    """Add a custom document to the RAG system (proxied to RAG Service)."""
    return await _call_rag_service("POST", "/rag/document", json={
        "content": content,
        "document_type": document_type,
        "symbol": symbol,
        "metadata": metadata
    })


@rag_router.get("/rag/query")
async def query_rag(
    query: str,
    symbol: Optional[str] = None,
    n_results: int = 5
):
    """Query the RAG system for relevant context (proxied to RAG Service)."""
    params = {"query": query, "n_results": n_results}
    if symbol:
        params["symbol"] = symbol
    return await _call_rag_service("GET", "/rag/query", params=params)


@rag_router.get("/rag/stats")
async def get_rag_stats():
    """Get statistics about the RAG collection (proxied to RAG Service)."""
    return await _call_rag_service("GET", "/rag/stats")


@rag_router.delete("/rag/documents")
async def delete_rag_documents(symbol: Optional[str] = None):
    """Delete documents from the RAG system (proxied to RAG Service)."""
    params = {}
    if symbol:
        params["symbol"] = symbol
    return await _call_rag_service("DELETE", "/rag/documents", params=params)


@rag_router.post("/rag/persist")
async def persist_rag():
    """Persist RAG database to disk (proxied to RAG Service)."""
    return await _call_rag_service("POST", "/rag/persist")
