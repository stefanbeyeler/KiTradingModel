"""API routes for the KI Trading Model service."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from ..models.trading_data import (
    AnalysisRequest,
    AnalysisResponse,
    TradingRecommendation,
)
from ..services import AnalysisService, RAGService, LLMService, EasyInsightClient


router = APIRouter()

# Service instances
analysis_service = AnalysisService()
rag_service = RAGService()
llm_service = LLMService()


@router.get("/health")
async def health_check():
    """Check health of all services."""
    try:
        health_status = await analysis_service.health_check()
        return {
            "status": "healthy" if health_status["all_healthy"] else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbol(request: AnalysisRequest):
    """
    Generate a trading analysis and recommendation for a symbol.

    This endpoint:
    1. Fetches time series data from EasyInsight
    2. Calculates technical indicators
    3. Queries relevant historical context from RAG
    4. Generates a recommendation using Llama 3.1 70B
    5. Stores the analysis for future reference
    """
    try:
        response = await analysis_service.analyze(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/recommendation/{symbol}", response_model=TradingRecommendation)
async def get_quick_recommendation(
    symbol: str,
    lookback_days: int = 30
):
    """Get a quick trading recommendation for a symbol."""
    try:
        request = AnalysisRequest(
            symbol=symbol,
            lookback_days=lookback_days,
            include_technical=True
        )
        response = await analysis_service.analyze(request)
        return response.recommendation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quick recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_available_symbols():
    """Get list of available trading symbols from EasyInsight."""
    try:
        async with EasyInsightClient() as client:
            symbols = await client.get_available_symbols()
            return {"symbols": symbols}
    except Exception as e:
        logger.error(f"Failed to fetch symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/document")
async def add_rag_document(
    content: str,
    document_type: str,
    symbol: Optional[str] = None,
    metadata: Optional[dict] = None
):
    """Add a custom document to the RAG system."""
    try:
        doc_id = await rag_service.add_custom_document(
            content=content,
            document_type=document_type,
            symbol=symbol,
            metadata=metadata
        )
        return {"document_id": doc_id, "status": "stored"}
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/query")
async def query_rag(
    query: str,
    symbol: Optional[str] = None,
    n_results: int = 5
):
    """Query the RAG system for relevant context."""
    try:
        results = await rag_service.query_relevant_context(
            query=query,
            symbol=symbol,
            n_results=n_results
        )
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/stats")
async def get_rag_stats():
    """Get statistics about the RAG collection."""
    try:
        stats = await rag_service.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/documents")
async def delete_rag_documents(
    symbol: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Delete documents from the RAG system."""
    try:
        deleted = await rag_service.delete_documents(symbol=symbol)
        return {
            "deleted_count": deleted,
            "symbol": symbol
        }
    except Exception as e:
        logger.error(f"Failed to delete documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/persist")
async def persist_rag():
    """Persist RAG database to disk."""
    try:
        await rag_service.persist()
        return {"status": "persisted"}
    except Exception as e:
        logger.error(f"Failed to persist RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/status")
async def get_llm_status():
    """Check LLM model status."""
    try:
        is_available = await llm_service.check_model_available()
        return {
            "model": llm_service.model,
            "host": llm_service.host,
            "available": is_available
        }
    except Exception as e:
        logger.error(f"LLM status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/pull")
async def pull_llm_model():
    """Pull the configured LLM model."""
    try:
        success = await llm_service.pull_model()
        return {
            "model": llm_service.model,
            "pulled": success
        }
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
