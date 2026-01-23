"""LLM Router - LLM-specific endpoints for the LLM Service."""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from loguru import logger
import httpx

from src.config import settings

llm_router = APIRouter()

# RAG Service URL (accessed via HTTP, not direct import)
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://trading-rag:3008")

# Global LLM service reference (set by main.py)
_llm_service = None


def set_llm_service(service):
    """Set the global LLM service instance."""
    global _llm_service
    _llm_service = service


def get_llm_service():
    """Get the global LLM service instance."""
    if _llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    return _llm_service


@llm_router.get("/llm/status")
async def get_llm_status():
    """Check LLM model status."""
    llm_service = get_llm_service()
    try:
        is_available = await llm_service.check_model_available()
        model_info = await llm_service.get_model_info()
        return {
            "model": llm_service.model,
            "host": llm_service.host,
            "available": is_available,
            "options": model_info.get("options", {}),
            "details": model_info.get("details", {})
        }
    except Exception as e:
        logger.error(f"LLM status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@llm_router.post("/llm/pull")
async def pull_llm_model():
    """Pull the configured LLM model."""
    llm_service = get_llm_service()
    try:
        success = await llm_service.pull_model()
        return {
            "model": llm_service.model,
            "pulled": success,
            "success": success
        }
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@llm_router.post("/llm/chat")
async def chat_with_llm(query: str, symbol: Optional[str] = None, use_rag: bool = True):
    """
    Simple chat endpoint for general trading questions.

    This endpoint allows free-form conversations without requiring a symbol.
    If a symbol is mentioned in the query, it will be extracted automatically.

    Args:
        query: The user's question or message
        symbol: Optional trading symbol for context
        use_rag: Whether to use RAG for enhanced context (default: True)

    Returns:
        LLM response with optional RAG context
    """
    llm_service = get_llm_service()
    try:
        # Try to extract symbol from query if not provided
        detected_symbol = symbol
        if not detected_symbol:
            # Known trading symbols
            known_crypto = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD', 'BNBUSD', 'DOTUSD', 'LTCUSD',
                           'BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'BNB', 'DOT', 'LTC', 'DOGE', 'SHIB']
            known_forex = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
                          'EURGBP', 'EURJPY', 'GBPJPY']
            known_commodities = ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']
            known_indices = ['US30', 'US500', 'NAS100', 'GER40', 'UK100']

            all_known = known_crypto + known_forex + known_commodities + known_indices

            # Find known symbols in the query (case insensitive)
            query_upper = query.upper()
            for sym in all_known:
                if sym in query_upper:
                    detected_symbol = sym
                    # Normalize crypto symbols
                    if detected_symbol in ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'BNB', 'DOT', 'LTC', 'DOGE', 'SHIB']:
                        detected_symbol = detected_symbol + 'USD'
                    break

        # Get RAG context if enabled (via HTTP API)
        rag_context = []
        if use_rag:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    params = {"query": query, "n_results": 3}
                    if detected_symbol:
                        params["symbol"] = detected_symbol
                    response = await client.get(f"{RAG_SERVICE_URL}/api/v1/rag/query", params=params)
                    if response.status_code == 200:
                        rag_results = response.json()
                        rag_context = [doc.get("content", "") for doc in rag_results.get("results", [])]
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Build prompt with context
        context_text = ""
        if rag_context:
            context_text = "\n\nRelevanter Kontext:\n" + "\n---\n".join(rag_context)

        system_prompt = f"""Du bist ein erfahrener Trading-Assistent. Beantworte Fragen zu Märkten,
Trading-Strategien und Finanzanalysen auf Deutsch. Sei präzise und hilfreich.
{context_text}"""

        # Call LLM
        response = await llm_service.generate(
            prompt=query,
            system=system_prompt,
            max_tokens=1000
        )

        return {
            "response": response,
            "symbol_detected": detected_symbol,
            "rag_context_used": len(rag_context) > 0,
            "model": llm_service.model
        }

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
