"""API routes for the KI Trading Model service."""

from datetime import datetime
from typing import Optional
import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from ..config import settings
from ..models.trading_data import (
    AnalysisRequest,
    AnalysisResponse,
    TradingRecommendation,
)
from ..services import AnalysisService, LLMService


router = APIRouter()

# Service instances
analysis_service = AnalysisService()
llm_service = LLMService()


def get_rag_service():
    """Get the global RAG service instance from main."""
    from ..main import rag_service
    return rag_service


@router.get("/health")
async def health_check():
    """Check health of all services."""
    try:
        # Check LLM service directly
        llm_healthy = await llm_service.check_model_available()

        # Check RAG service
        rag_healthy = False
        try:
            rag_svc = get_rag_service()
            stats = await rag_svc.get_collection_stats()
            rag_healthy = True
        except Exception:
            pass

        services = {
            "llm_service": llm_healthy,
            "rag_service": rag_healthy
        }

        all_healthy = llm_healthy and rag_healthy

        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services
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
    lookback_days: int = 30,
    use_llm: bool = False
):
    """
    Get a quick trading recommendation for a symbol.

    By default uses fast rule-based analysis (~100ms).
    Set use_llm=true for detailed LLM analysis (~30-60s).
    """
    try:
        recommendation = await analysis_service.quick_recommendation(
            symbol=symbol,
            lookback_days=lookback_days,
            use_llm=use_llm
        )
        return recommendation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quick recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_available_symbols():
    """Get list of available trading symbols from TimescaleDB."""
    try:
        symbols = await analysis_service.get_available_symbols()
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
        rag_service = get_rag_service()
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
        rag_service = get_rag_service()
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
        rag_service = get_rag_service()
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
        rag_service = get_rag_service()
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
        rag_service = get_rag_service()
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


# Sync service endpoints - import from main
@router.get("/sync/status")
async def get_sync_status():
    """Get the status of the TimescaleDB sync service."""
    from ..main import sync_service
    return sync_service.get_status()


@router.post("/sync/start")
async def start_sync():
    """Start the TimescaleDB sync service."""
    from ..main import sync_service
    try:
        await sync_service.start()
        return {
            "status": "started",
            "message": "TimescaleDB sync service started"
        }
    except Exception as e:
        logger.error(f"Failed to start sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/stop")
async def stop_sync():
    """Stop the TimescaleDB sync service."""
    from ..main import sync_service
    try:
        await sync_service.stop()
        return {
            "status": "stopped",
            "message": "TimescaleDB sync service stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/manual")
async def manual_sync(days_back: int = 7):
    """Manually trigger a sync for the specified number of days."""
    from ..main import sync_service
    try:
        synced_count = await sync_service.manual_sync(days_back=days_back)
        return {
            "status": "completed",
            "documents_synced": synced_count,
            "days_back": days_back
        }
    except Exception as e:
        logger.error(f"Manual sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/info")
async def get_system_info():
    """Get system and GPU information."""
    try:
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
                "cuda_version": torch.version.cuda,
            }
        else:
            gpu_info = {"available": False}

        rag_service = get_rag_service()
        rag_stats = await rag_service.get_collection_stats()

        return {
            "system": {
                "device": settings.device,
                "gpu_available": settings.gpu_available,
                "gpu_name": settings.gpu_name,
                "gpu_memory_gb": round(settings.gpu_memory_gb, 2),
                "pytorch_version": torch.__version__,
            },
            "gpu_runtime": gpu_info,
            "configuration": {
                "ollama_model": settings.ollama_model,
                "ollama_num_ctx": settings.ollama_num_ctx,
                "ollama_num_gpu": settings.ollama_num_gpu,
                "ollama_num_thread": settings.ollama_num_thread,
                "embedding_model": settings.embedding_model,
                "embedding_device": settings.device,
                "embedding_batch_size": settings.embedding_batch_size,
                "use_half_precision": settings.use_half_precision,
                "faiss_use_gpu": settings.faiss_use_gpu,
            },
            "rag": rag_stats
        }
    except Exception as e:
        logger.error(f"System info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
