"""API routes for the KI Trading Model service."""

from datetime import datetime
from typing import Optional
import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import PlainTextResponse
from loguru import logger

from ..config import settings
from ..version import get_version_info, VERSION, RELEASE_DATE
from ..models.trading_data import (
    AnalysisRequest,
    AnalysisResponse,
    TradingRecommendation,
    TradingStrategy,
    StrategyCreateRequest,
    StrategyUpdateRequest,
)
from ..services import AnalysisService, LLMService, StrategyService
from ..services.query_log_service import query_log_service, QueryLogEntry


router = APIRouter()

# Service instances
analysis_service = AnalysisService()
llm_service = LLMService()
strategy_service = StrategyService()


def get_rag_service():
    """Get the global RAG service instance from main."""
    from ..main import rag_service
    return rag_service


@router.get("/version")
async def get_version():
    """Get application version and release information."""
    return get_version_info()


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
    use_llm: bool = False,
    strategy_id: Optional[str] = None
):
    """
    Get a quick trading recommendation for a symbol.

    By default uses fast rule-based analysis (~100ms).
    Set use_llm=true for detailed LLM analysis (~30-60s).
    Optionally specify a strategy_id to use a specific trading strategy.
    """
    try:
        # Get strategy if specified, otherwise use default
        strategy = None
        if strategy_id:
            strategy = await strategy_service.get_strategy(strategy_id)
            if not strategy:
                raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
        else:
            strategy = await strategy_service.get_default_strategy()

        recommendation = await analysis_service.quick_recommendation(
            symbol=symbol,
            lookback_days=lookback_days,
            use_llm=use_llm,
            strategy=strategy
        )
        return recommendation
    except HTTPException:
        raise
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


@router.get("/symbol-info/{symbol:path}")
async def get_symbol_info(symbol: str):
    """Get detailed information about a symbol from TimescaleDB."""
    try:
        info = await analysis_service.get_symbol_info(symbol)
        return info
    except Exception as e:
        logger.error(f"Failed to fetch symbol info for {symbol}: {e}")
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


# ============================================
# Trading Strategy Endpoints
# ============================================

@router.get("/strategies", response_model=list[TradingStrategy])
async def get_strategies(include_inactive: bool = False):
    """Get all trading strategies."""
    try:
        strategies = await strategy_service.get_all_strategies(include_inactive=include_inactive)
        return strategies
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/default", response_model=TradingStrategy)
async def get_default_strategy():
    """Get the default trading strategy."""
    try:
        strategy = await strategy_service.get_default_strategy()
        if not strategy:
            raise HTTPException(status_code=404, detail="No default strategy found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get default strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}/export", response_class=PlainTextResponse)
async def export_strategy(strategy_id: str):
    """Export a strategy as Markdown file."""
    try:
        strategy = await strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")

        markdown = strategy_service.export_strategy_to_markdown(strategy)

        # Set filename header for download
        filename = f"strategy_{strategy.name.replace(' ', '_').lower()}.md"
        return PlainTextResponse(
            content=markdown,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}", response_model=TradingStrategy)
async def get_strategy(strategy_id: str):
    """Get a specific trading strategy by ID."""
    try:
        strategy = await strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies", response_model=TradingStrategy)
async def create_strategy(request: StrategyCreateRequest):
    """Create a new trading strategy."""
    try:
        strategy = await strategy_service.create_strategy(request)
        return strategy
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/strategies/{strategy_id}", response_model=TradingStrategy)
async def update_strategy(strategy_id: str, request: StrategyUpdateRequest):
    """Update an existing trading strategy."""
    try:
        strategy = await strategy_service.update_strategy(strategy_id, request)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a trading strategy (only custom strategies can be deleted)."""
    try:
        deleted = await strategy_service.delete_strategy(strategy_id)
        if not deleted:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete strategy '{strategy_id}'. Either it doesn't exist or it's a default strategy."
            )
        return {"status": "deleted", "strategy_id": strategy_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{strategy_id}/set-default", response_model=TradingStrategy)
async def set_default_strategy(strategy_id: str):
    """Set a strategy as the default."""
    try:
        strategy = await strategy_service.set_default_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set default strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/import", response_model=TradingStrategy)
async def import_strategy(file: UploadFile = File(...)):
    """Import a strategy from a Markdown file."""
    try:
        # Validate file type
        if not file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Only Markdown (.md) files are accepted")

        # Read file content
        content = await file.read()
        markdown_content = content.decode('utf-8')

        # Import strategy
        strategy = await strategy_service.import_and_save_strategy(markdown_content)
        if not strategy:
            raise HTTPException(status_code=400, detail="Failed to parse strategy from Markdown. Check file format.")

        return strategy
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        logger.error(f"Failed to import strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Query Logs ====================

@router.get("/query-logs")
async def get_query_logs(
    limit: int = 50,
    offset: int = 0,
    query_type: Optional[str] = None,
    symbol: Optional[str] = None,
    success_only: bool = False,
):
    """
    Get query log history with optional filtering.

    Parameters:
    - limit: Maximum number of logs to return (default: 50)
    - offset: Number of logs to skip (default: 0)
    - query_type: Filter by type (analysis, quick_recommendation, rag_query)
    - symbol: Filter by symbol (partial match)
    - success_only: Only return successful queries
    """
    try:
        logs = query_log_service.get_logs(
            limit=limit,
            offset=offset,
            query_type=query_type,
            symbol=symbol,
            success_only=success_only,
        )
        return {
            "logs": [log.model_dump() for log in logs],
            "count": len(logs),
            "offset": offset,
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"Failed to get query logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-logs/stats")
async def get_query_log_stats():
    """Get statistics about query logs."""
    try:
        return query_log_service.get_stats()
    except Exception as e:
        logger.error(f"Failed to get query log stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-logs/{log_id}")
async def get_query_log_by_id(log_id: str):
    """Get a specific query log entry by ID."""
    try:
        log = query_log_service.get_log_by_id(log_id)
        if not log:
            raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found")
        return log.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get query log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/query-logs")
async def clear_query_logs():
    """Clear all query logs."""
    try:
        count = query_log_service.clear_logs()
        return {"status": "cleared", "deleted_count": count}
    except Exception as e:
        logger.error(f"Failed to clear query logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
