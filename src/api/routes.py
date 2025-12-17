"""API routes for the KI Trading Model service."""
# Health check now includes NHITS status

import json
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
from ..models.symbol_data import (
    ManagedSymbol,
    SymbolCategory,
    SymbolSubcategory,
    SymbolStatus,
    SymbolCreateRequest,
    SymbolUpdateRequest,
    SymbolImportResult,
    SymbolStats,
)
from ..models.forecast_data import (
    ForecastResult,
    ForecastConfig,
    ForecastTrainingResult,
    ForecastModelInfo,
    TrainingProgressResponse,
    TrainingProgressResults,
    TrainingProgressTiming,
)
from ..services import AnalysisService, LLMService, StrategyService
from ..services.query_log_service import query_log_service, QueryLogEntry
from ..services.symbol_service import symbol_service
from ..services.event_based_training_service import event_based_training_service
from ..services.twelvedata_service import twelvedata_service


# Thematisch gruppierte Router für bessere API-Organisation
router = APIRouter()  # Hauptrouter für allgemeine Endpoints
trading_router = APIRouter()  # Trading-Analyse und Empfehlungen
forecast_router = APIRouter()  # NHITS Forecasting (Predictions)
training_router = APIRouter()  # NHITS Training
symbol_router = APIRouter()  # Symbol-Management
strategy_router = APIRouter()  # Trading-Strategien
rag_router = APIRouter()  # RAG & Wissensbasis
llm_router = APIRouter()  # LLM Service
sync_router = APIRouter()  # TimescaleDB Sync
system_router = APIRouter()  # System & Monitoring
query_log_router = APIRouter()  # Query Logs & Analytics
twelvedata_router = APIRouter()  # Twelve Data API

# Service instances
analysis_service = AnalysisService()
llm_service = LLMService()
strategy_service = StrategyService()


def get_rag_service():
    """Get the global RAG service instance from service registry."""
    from ..service_registry import get_rag_service as _get_rag
    service = _get_rag()
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return service


def get_sync_service():
    """Get the global sync service instance from service registry."""
    from ..service_registry import get_sync_service as _get_sync
    service = _get_sync()
    if service is None:
        raise HTTPException(status_code=503, detail="Sync service not initialized")
    return service


# ==================== System & Health ====================

@system_router.get("/version")
async def get_version():
    """Get application version and release information."""
    return get_version_info()


@system_router.get("/health")
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

        # Check NHITS forecast service
        nhits_healthy = False
        nhits_status = {}
        try:
            from ..services.forecast_service import forecast_service
            nhits_healthy = forecast_service.device is not None

            # Get model count and device info
            models = list(forecast_service.model_path.glob("*_model.pt"))
            nhits_status = {
                "enabled": settings.nhits_enabled,
                "device": str(forecast_service.device) if forecast_service.device else "none",
                "models_loaded": len(models),
                "model_symbols": [p.stem.replace("_model", "") for p in models],
                "cuda_available": torch.cuda.is_available(),
            }
            if torch.cuda.is_available():
                nhits_status["gpu_name"] = torch.cuda.get_device_name(0)
                nhits_status["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        except Exception as e:
            nhits_status = {"error": str(e)}

        services = {
            "llm_service": llm_healthy,
            "rag_service": rag_healthy,
            "nhits_service": nhits_healthy
        }

        all_healthy = llm_healthy and rag_healthy and nhits_healthy

        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services,
            "nhits": nhits_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# ==================== Trading Analysis ====================

@trading_router.post("/analyze", response_model=AnalysisResponse)
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


@trading_router.get("/recommendation/{symbol}", response_model=TradingRecommendation)
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


@trading_router.get("/symbols")
async def get_available_symbols():
    """Get list of available trading symbols from TimescaleDB."""
    try:
        symbols = await analysis_service.get_available_symbols()
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"Failed to fetch symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@trading_router.get("/symbol-info/{symbol:path}")
async def get_symbol_info(symbol: str):
    """Get detailed information about a symbol from TimescaleDB."""
    try:
        info = await analysis_service.get_symbol_info(symbol)
        return info
    except Exception as e:
        logger.error(f"Failed to fetch symbol info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/rag/document")
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


@rag_router.get("/rag/query")
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


@rag_router.get("/rag/stats")
async def get_rag_stats():
    """Get statistics about the RAG collection."""
    try:
        rag_service = get_rag_service()
        stats = await rag_service.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.delete("/rag/documents")
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


@rag_router.post("/rag/persist")
async def persist_rag():
    """Persist RAG database to disk."""
    try:
        rag_service = get_rag_service()
        await rag_service.persist()
        return {"status": "persisted"}
    except Exception as e:
        logger.error(f"Failed to persist RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@llm_router.get("/llm/status")
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


@llm_router.post("/llm/pull")
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


# ==================== TimescaleDB Sync Endpoints ====================

@sync_router.get("/sync/status")
async def get_sync_status():
    """
    Get the status of the TimescaleDB sync service.

    Returns information about:
    - Whether the sync service is running
    - Database connection status
    - Last sync time and sync count
    - Configured sync interval
    """
    sync_service = get_sync_service()
    return sync_service.get_status()


@sync_router.post("/sync/start")
async def start_sync():
    """
    Start the TimescaleDB sync service.

    Begins automatic synchronization of market data from EasyInsight API
    to the RAG knowledge base. Sync runs at configured intervals.
    """
    sync_service = get_sync_service()
    try:
        await sync_service.start()
        return {
            "status": "started",
            "message": "TimescaleDB sync service started",
            "interval_seconds": sync_service.get_status()["sync_interval_seconds"]
        }
    except Exception as e:
        logger.error(f"Failed to start sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sync_router.post("/sync/stop")
async def stop_sync():
    """
    Stop the TimescaleDB sync service.

    Stops automatic synchronization. Can be restarted later with /sync/start.
    """
    sync_service = get_sync_service()
    try:
        await sync_service.stop()
        return {
            "status": "stopped",
            "message": "TimescaleDB sync service stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@sync_router.post("/sync/manual")
async def manual_sync(days_back: int = 7):
    """
    Manually trigger a one-time sync for the specified number of days.

    Parameters:
    - days_back: Number of days of historical data to sync (default: 7)

    This fetches market data from EasyInsight API and stores it in the
    RAG knowledge base for LLM analysis. Useful for backfilling data
    or syncing after adding new symbols.
    """
    sync_service = get_sync_service()
    try:
        synced_count = await sync_service.manual_sync(days_back=days_back)
        return {
            "status": "completed",
            "documents_synced": synced_count,
            "days_back": days_back,
            "message": f"Successfully synced {synced_count} documents from the last {days_back} days"
        }
    except Exception as e:
        logger.error(f"Manual sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/system/metrics")
async def get_system_metrics():
    """
    Get real-time CPU and GPU utilization metrics.

    Returns current usage percentages for monitoring dashboards.
    Updates on each request - designed for polling (e.g., every 2-5 seconds).
    """
    import psutil

    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()

        # Memory metrics
        memory = psutil.virtual_memory()

        metrics = {
            "cpu": {
                "percent": cpu_percent,
                "cores_physical": cpu_count,
                "cores_logical": cpu_count_logical,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
            },
            "memory": {
                "percent": memory.percent,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
            },
            "gpu": None,
            "timestamp": datetime.utcnow().isoformat()
        }

        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_allocated = torch.cuda.memory_allocated(0)
                gpu_memory_reserved = torch.cuda.memory_reserved(0)

                # Try to get GPU utilization via nvidia-smi
                gpu_utilization = None
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        gpu_utilization = float(result.stdout.strip().split('\n')[0])
                except Exception:
                    pass

                metrics["gpu"] = {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "utilization_percent": gpu_utilization,
                    "memory_percent": round((gpu_memory_allocated / gpu_memory_total) * 100, 1) if gpu_memory_total > 0 else 0,
                    "memory_total_gb": round(gpu_memory_total / (1024**3), 2),
                    "memory_allocated_gb": round(gpu_memory_allocated / (1024**3), 2),
                    "memory_reserved_gb": round(gpu_memory_reserved / (1024**3), 2),
                }
            except Exception as e:
                metrics["gpu"] = {"available": True, "error": str(e)}
        else:
            metrics["gpu"] = {"available": False}

        return metrics

    except Exception as e:
        logger.error(f"System metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/system/info")
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

@strategy_router.get("/strategies", response_model=list[TradingStrategy])
async def get_strategies(include_inactive: bool = False):
    """Get all trading strategies."""
    try:
        strategies = await strategy_service.get_all_strategies(include_inactive=include_inactive)
        return strategies
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@strategy_router.get("/strategies/default", response_model=TradingStrategy)
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


@strategy_router.get("/strategies/{strategy_id}/export", response_class=PlainTextResponse)
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


@strategy_router.get("/strategies/{strategy_id}", response_model=TradingStrategy)
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


@strategy_router.post("/strategies", response_model=TradingStrategy)
async def create_strategy(request: StrategyCreateRequest):
    """Create a new trading strategy."""
    try:
        strategy = await strategy_service.create_strategy(request)
        return strategy
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@strategy_router.put("/strategies/{strategy_id}", response_model=TradingStrategy)
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


@strategy_router.delete("/strategies/{strategy_id}")
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


@strategy_router.post("/strategies/{strategy_id}/set-default", response_model=TradingStrategy)
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


@strategy_router.post("/strategies/import", response_model=TradingStrategy)
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

@query_log_router.get("/query-logs")
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


@query_log_router.get("/query-logs/stats")
async def get_query_log_stats():
    """Get statistics about query logs."""
    try:
        return query_log_service.get_stats()
    except Exception as e:
        logger.error(f"Failed to get query log stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@query_log_router.get("/query-logs/{log_id}")
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


@query_log_router.delete("/query-logs")
async def clear_query_logs():
    """Clear all query logs."""
    try:
        count = query_log_service.clear_logs()
        return {"status": "cleared", "deleted_count": count}
    except Exception as e:
        logger.error(f"Failed to clear query logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NHITS Forecast Endpoints ====================

def get_forecast_service():
    """Get the forecast service instance."""
    from ..services.forecast_service import forecast_service
    return forecast_service


def get_training_service():
    """Get the NHITS training service instance."""
    from ..services.nhits_training_service import nhits_training_service
    return nhits_training_service


# IMPORTANT: Static routes MUST come before parameterized routes
# to avoid FastAPI interpreting "status" or "models" as {symbol}

@forecast_router.get("/forecast/status")
async def get_forecast_status():
    """
    Get the status of the NHITS forecasting service.

    Returns whether NHITS is enabled and configuration details.
    """
    try:
        forecast_service = get_forecast_service()
        models = forecast_service.list_models()

        return {
            "enabled": settings.nhits_enabled,
            "configuration": {
                "horizon": settings.nhits_horizon,
                "input_size": settings.nhits_input_size,
                "hidden_size": settings.nhits_hidden_size,
                "batch_size": settings.nhits_batch_size,
                "max_steps": settings.nhits_max_steps,
                "use_gpu": settings.nhits_use_gpu,
                "auto_retrain_days": settings.nhits_auto_retrain_days,
                "model_path": settings.nhits_model_path,
            },
            "trained_models": len(models),
            "models": [m.symbol for m in models if m.model_exists],
        }
    except Exception as e:
        logger.error(f"Failed to get forecast status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@forecast_router.get("/forecast/models", response_model=list[ForecastModelInfo])
async def list_forecast_models():
    """
    List all trained NHITS models.

    Returns information about all models that have been trained.
    """
    try:
        forecast_service = get_forecast_service()
        models = forecast_service.list_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@forecast_router.get("/forecast/models/by-timeframe")
async def list_models_by_timeframe():
    """
    List all trained NHITS models grouped by timeframe.

    Returns models organized by their timeframe suffix (H1, M15, D1, or default).
    """
    try:
        forecast_service = get_forecast_service()
        models = forecast_service.list_models()

        # Group models by timeframe
        timeframe_groups = {
            "default": [],  # No suffix (standard hourly)
            "H1": [],       # Hourly
            "M15": [],      # 15 minutes
            "D1": [],       # Daily
        }

        for model in models:
            if not model.model_exists:
                continue
            symbol = model.symbol

            # Check for timeframe suffix
            if symbol.endswith("_H1"):
                timeframe_groups["H1"].append(symbol.replace("_H1", ""))
            elif symbol.endswith("_M15"):
                timeframe_groups["M15"].append(symbol.replace("_M15", ""))
            elif symbol.endswith("_D1"):
                timeframe_groups["D1"].append(symbol.replace("_D1", ""))
            else:
                timeframe_groups["default"].append(symbol)

        # Sort symbols in each group
        for key in timeframe_groups:
            timeframe_groups[key].sort()

        return {
            "total_models": len([m for m in models if m.model_exists]),
            "by_timeframe": {
                "default": {
                    "count": len(timeframe_groups["default"]),
                    "label": "Standard (1H)",
                    "symbols": timeframe_groups["default"]
                },
                "H1": {
                    "count": len(timeframe_groups["H1"]),
                    "label": "Stündlich (H1)",
                    "symbols": timeframe_groups["H1"]
                },
                "M15": {
                    "count": len(timeframe_groups["M15"]),
                    "label": "15 Minuten (M15)",
                    "symbols": timeframe_groups["M15"]
                },
                "D1": {
                    "count": len(timeframe_groups["D1"]),
                    "label": "Täglich (D1)",
                    "symbols": timeframe_groups["D1"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to list models by timeframe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NHITS Batch Training Endpoints ====================
# NOTE: These must be defined BEFORE parameterized routes like /forecast/{symbol}

@training_router.get("/forecast/training/status")
async def get_training_status():
    """
    Get the status of the NHITS training service.

    Returns information about scheduled training and recent training runs.
    """
    try:
        training_service = get_training_service()
        return training_service.get_status()
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.get("/forecast/training/progress", response_model=TrainingProgressResponse)
async def get_training_progress():
    """
    Get real-time progress of the current NHITS training session.

    Returns detailed progress information including:
    - Current symbol being trained
    - Number of symbols completed vs total
    - Success/failure/skipped counts
    - Progress percentage
    - Elapsed time
    - Estimated time remaining (if available)

    Useful for monitoring long-running batch training jobs.
    """
    try:
        training_service = get_training_service()
        status = training_service.get_status()

        # If no training in progress, return minimal info
        if not status.get("training_in_progress", False):
            return TrainingProgressResponse(
                training_in_progress=False,
                message="No training currently running",
                last_training_run=status.get("last_training_run")
            )

        # Extract detailed progress information
        progress = status.get("progress", {})

        total = progress.get("total_symbols", 0)
        completed = progress.get("completed_symbols", 0)
        successful = progress.get("successful", 0)
        failed = progress.get("failed", 0)
        skipped = progress.get("skipped", 0)
        current_symbol = progress.get("current_symbol")
        progress_pct = progress.get("progress_percent", 0)
        elapsed = progress.get("elapsed_seconds", 0)
        cancelling = progress.get("cancelling", False)
        started_at = progress.get("started_at")

        # Calculate estimated time remaining
        eta_seconds = None
        if completed > 0 and total > 0 and not cancelling:
            avg_time_per_symbol = elapsed / completed
            remaining_symbols = total - completed
            eta_seconds = int(avg_time_per_symbol * remaining_symbols)

        return TrainingProgressResponse(
            training_in_progress=True,
            current_symbol=current_symbol,
            total_symbols=total,
            completed_symbols=completed,
            remaining_symbols=total - completed,
            progress_percent=progress_pct,
            results=TrainingProgressResults(
                successful=successful,
                failed=failed,
                skipped=skipped
            ),
            timing=TrainingProgressTiming(
                elapsed_seconds=elapsed,
                eta_seconds=eta_seconds,
                elapsed_formatted=f"{elapsed // 60}m {elapsed % 60}s" if elapsed else "0s",
                eta_formatted=f"{eta_seconds // 60}m {eta_seconds % 60}s" if eta_seconds else None
            ),
            cancelling=cancelling,
            started_at=started_at
        )

    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.get("/forecast/training/symbols")
async def get_trainable_symbols():
    """
    Get list of symbols available for NHITS training.

    Returns all symbols that have sufficient data in TimescaleDB.
    """
    try:
        training_service = get_training_service()
        symbols = await training_service.get_available_symbols()

        return {
            "count": len(symbols),
            "symbols": symbols
        }

    except Exception as e:
        logger.error(f"Failed to get trainable symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.get("/forecast/training/failed")
async def get_failed_trainings():
    """
    Get details about failed model trainings from the current/last training run.

    Returns:
    - count: Number of failed models
    - models: Dictionary with model_key -> error details
    - training_in_progress: Whether training is still running

    Use this endpoint to debug why certain models failed to train.
    Common failure reasons:
    - Insufficient data: Not enough historical data points for the timeframe
    - No OHLCV data: Symbol doesn't have price data for the requested timeframe
    - API errors: Failed to fetch data from EasyInsight API
    """
    try:
        training_service = get_training_service()
        return training_service.get_failed_models()
    except Exception as e:
        logger.error(f"Failed to get failed trainings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.get("/forecast/training/results")
async def get_training_results():
    """
    Get comprehensive results from the current/last training run.

    Returns:
    - summary: Counts of total, completed, successful, failed, skipped
    - failed_models: Details of all failed trainings with error messages
    - successful_models: Details of all successful trainings

    This endpoint provides full visibility into the training process.
    """
    try:
        training_service = get_training_service()
        return training_service.get_training_results()
    except Exception as e:
        logger.error(f"Failed to get training results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/train-all")
async def train_all_models(
    symbols: list[str] | None = None,
    force: bool = False,
    background: bool = True,
    timeframes: list[str] | None = None
):
    """
    Train NHITS models for all (or specified) symbols across multiple timeframes.

    Parameters:
    - symbols: Optional list of specific symbols to train (default: all available)
    - force: Force retraining even if models are up to date
    - background: Run training in background (default: True)
    - timeframes: List of timeframes to train (default: ["M15", "H1", "D1"])

    Returns training summary or task status if running in background.

    **Multi-Timeframe Training:**
    With 64 symbols and 3 timeframes, this will train 192 models total.
    Each timeframe uses different input/output configurations:
    - M15: 15-minute candles, 2-hour forecast
    - H1: Hourly candles, 24-hour forecast
    - D1: Daily candles, 7-day forecast
    """
    import asyncio
    try:
        if not settings.nhits_enabled:
            raise HTTPException(
                status_code=503,
                detail="NHITS forecasting is disabled. Enable it in settings."
            )

        training_service = get_training_service()

        # Validate timeframes if provided
        if timeframes:
            valid_timeframes = ["M15", "H1", "D1"]
            invalid = [tf for tf in timeframes if tf.upper() not in valid_timeframes]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid timeframes: {invalid}. Valid options: {valid_timeframes}"
                )
            timeframes = [tf.upper() for tf in timeframes]

        if background:
            # Start training in background
            asyncio.create_task(
                training_service.train_all_symbols(
                    symbols=symbols,
                    force=force,
                    timeframes=timeframes
                )
            )
            return {
                "status": "started",
                "message": "Multi-timeframe training started in background",
                "symbols": symbols or "all available",
                "timeframes": timeframes or ["M15", "H1", "D1"]
            }
        else:
            # Run training synchronously
            result = await training_service.train_all_symbols(
                symbols=symbols,
                force=force,
                timeframes=timeframes
            )
            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/training/cancel")
async def cancel_training():
    """
    Cancel the current training run.

    Stops the batch training process after the current symbol completes.
    """
    try:
        training_service = get_training_service()
        cancelled = training_service.cancel_training()

        if cancelled:
            return {
                "success": True,
                "message": "Training cancellation requested. Training will stop after current symbol."
            }
        else:
            return {
                "success": False,
                "message": "No training is currently in progress."
            }
    except Exception as e:
        logger.error(f"Failed to cancel training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Event-Based Training Endpoints ====================

@training_router.get("/forecast/training/events/status")
async def get_event_training_status():
    """
    Get status of the event-based training monitor.

    Returns whether the service is running and monitoring trading events
    to trigger automatic model retraining.
    """
    try:
        return {
            "running": event_based_training_service._running,
            "check_interval_minutes": event_based_training_service._check_interval_minutes,
            "event_threshold": event_based_training_service._event_threshold,
            "monitored_indicators": event_based_training_service._monitored_indicators
        }
    except Exception as e:
        logger.error(f"Failed to get event training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.get("/forecast/training/events/summary")
async def get_event_summary(symbol: Optional[str] = None, hours: int = 24):
    """
    Get summary of recent trading events.

    Shows event statistics from the EasyInsight logs API to understand
    market activity and potential retraining triggers.

    Args:
        symbol: Optional symbol to filter events
        hours: Number of hours to look back (default: 24)
    """
    try:
        summary = await event_based_training_service.get_event_summary(
            symbol=symbol,
            hours=hours
        )
        return summary
    except Exception as e:
        logger.error(f"Failed to get event summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/training/events/start")
async def start_event_training():
    """
    Start the event-based training monitor.

    Begins monitoring trading events from EasyInsight and automatically
    triggers model retraining when significant events are detected.
    """
    try:
        await event_based_training_service.start()
        return {
            "success": True,
            "message": "Event-based training monitor started",
            "check_interval_minutes": event_based_training_service._check_interval_minutes
        }
    except Exception as e:
        logger.error(f"Failed to start event training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/training/events/stop")
async def stop_event_training():
    """
    Stop the event-based training monitor.

    Stops monitoring trading events. Scheduled and manual training
    will continue to work normally.
    """
    try:
        await event_based_training_service.stop()
        return {
            "success": True,
            "message": "Event-based training monitor stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop event training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Model Improvement & Performance Endpoints (MUST be before {symbol} routes!)
# =============================================================================

@training_router.get("/forecast/performance")
async def get_model_performance():
    """
    Get performance metrics for all NHITS models.

    Returns prediction accuracy, direction accuracy, and identifies
    models that need retraining based on performance degradation.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        return model_improvement_service.get_performance_summary()
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.get("/forecast/evaluated")
async def get_evaluated_predictions(symbol: Optional[str] = None, limit: int = 50):
    """
    Get list of evaluated predictions with their results.

    Shows past predictions compared to actual prices, including
    error percentages and direction accuracy.

    Args:
        symbol: Optional symbol to filter by
        limit: Maximum number of predictions to return (default 50)
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        predictions = model_improvement_service.get_evaluated_predictions(symbol=symbol, limit=limit)
        return {
            "count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Failed to get evaluated predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.get("/forecast/retraining-needed")
async def get_symbols_needing_retraining():
    """
    Get list of symbols whose models need retraining due to poor performance.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service

        symbols = model_improvement_service.get_symbols_needing_retraining()
        details = {}

        for symbol in symbols:
            metrics = model_improvement_service.performance_metrics.get(symbol)
            if metrics:
                details[symbol] = {
                    "reason": metrics.retraining_reason,
                    "avg_error_pct": round(metrics.avg_error_pct, 4),
                    "direction_accuracy": round(metrics.direction_accuracy, 4),
                    "evaluated_predictions": metrics.evaluated_predictions,
                }

        return {
            "symbols": symbols,
            "count": len(symbols),
            "details": details
        }
    except Exception as e:
        logger.error(f"Failed to get retraining list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/evaluate")
async def evaluate_predictions():
    """
    Manually trigger evaluation of pending predictions.

    This compares past predictions with actual prices and updates
    model performance metrics.

    NOTE: This endpoint requires direct database access which is currently
    disabled. Evaluation is done via the EasyInsight API instead.
    """
    # This endpoint requires direct database access for historical price lookups
    # Since we're using API-only mode, this functionality is not available
    raise HTTPException(
        status_code=501,
        detail="Prediction evaluation requires direct database access. "
               "This feature is not available in API-only mode. "
               "Use the EasyInsight API for historical data queries."
    )


@training_router.post("/forecast/retrain-poor-performers")
async def retrain_poor_performers():
    """
    Trigger retraining for all models that are performing poorly.

    Uses the automatic improvement system to identify and retrain
    models with low direction accuracy or high prediction error.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        from ..services.nhits_training_service import nhits_training_service

        symbols = model_improvement_service.get_symbols_needing_retraining()

        if not symbols:
            return {
                "success": True,
                "message": "No models need retraining",
                "symbols": []
            }

        # Retraining function to run in background
        async def retrain_symbols():
            logger.info(f"Starting retraining for poor performers: {symbols}")
            for symbol in symbols:
                try:
                    logger.info(f"Retraining model for {symbol}...")
                    result = await nhits_training_service.train_symbol(symbol, force=True)
                    if result.success:
                        logger.info(f"Successfully retrained {symbol}")
                        # Reset metrics after successful retraining
                        if symbol in model_improvement_service.performance_metrics:
                            model_improvement_service.performance_metrics[symbol].needs_retraining = False
                            model_improvement_service.performance_metrics[symbol].retraining_reason = None
                            model_improvement_service._save_data()
                    else:
                        logger.error(f"Retraining failed for {symbol}: {result.error_message}")
                except Exception as e:
                    logger.error(f"Failed to retrain {symbol}: {e}", exc_info=True)
            logger.info("Retraining for poor performers completed")

        # Use asyncio.create_task to properly run async function in background
        import asyncio
        asyncio.create_task(retrain_symbols())

        return {
            "success": True,
            "message": f"Queued retraining for {len(symbols)} models",
            "symbols": symbols,
            "symbols_queued": symbols  # For frontend compatibility
        }
    except Exception as e:
        logger.error(f"Failed to retrain poor performers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NHITS Symbol-specific Endpoints ====================

@forecast_router.get("/forecast/{symbol}", response_model=ForecastResult)
async def get_forecast(
    symbol: str,
    horizon: int = 24,
    retrain: bool = False,
    timeframe: str = "H1"
):
    """
    Generate NHITS price forecast for a symbol.

    Parameters:
    - symbol: Trading symbol (e.g., EURUSD)
    - horizon: Forecast horizon in hours (default: 24, max: 168) - NOTE: overridden by timeframe config
    - retrain: Force model retraining before forecast (default: false)
    - timeframe: Data granularity - M15 (15-min, 2h horizon), H1 (hourly, 24h horizon), D1 (daily, 7d horizon)

    Returns predicted prices with confidence intervals for the specified horizon.

    **Timeframe configurations:**
    - **M15**: 15-minute candles, 8-step forecast (2 hours), 96-candle lookback (24 hours)
    - **H1** (default): Hourly candles, 24-step forecast (24 hours), 168-candle lookback (7 days)
    - **D1**: Daily candles, 7-step forecast (7 days), 30-candle lookback (30 days)
    """
    try:
        # Check if NHITS is enabled
        if not settings.nhits_enabled:
            raise HTTPException(
                status_code=503,
                detail="NHITS forecasting is disabled. Enable it in settings."
            )

        # Validate timeframe
        timeframe = timeframe.upper()
        if timeframe not in ["M15", "H1", "D1"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe '{timeframe}'. Use M15, H1, or D1."
            )

        # Determine data requirements based on timeframe
        from datetime import timedelta
        if timeframe == "M15":
            # M15 needs ~24h of data for 96 candles input
            days_needed = 2
            interval = "m15"
        elif timeframe == "D1":
            # D1 needs ~30 days of data for 30 candles input + 7 horizon + buffer for sequences
            days_needed = 60
            interval = "d1"
        else:  # H1
            # H1 needs ~7 days of data for 168 candles input
            days_needed = 30
            interval = "h1"

        # Fetch time series data with appropriate interval
        time_series = await analysis_service._fetch_time_series(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=days_needed),
            end_date=datetime.now(),
            interval=interval
        )

        if not time_series:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol} with timeframe {timeframe}")

        # Create config
        config = ForecastConfig(
            symbol=symbol,
            horizon=min(horizon, 168),  # Max 7 days (note: actual horizon depends on timeframe)
            retrain=retrain,
            timeframe=timeframe
        )

        # Generate forecast
        forecast_service = get_forecast_service()
        result = await forecast_service.forecast(
            time_series=time_series,
            symbol=symbol,
            config=config,
            timeframe=timeframe
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast failed for {symbol}/{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/{symbol}/train", response_model=ForecastTrainingResult)
async def train_forecast_model(
    symbol: str,
    days: int = 90,
    force: bool = False,
    timeframe: str = "H1"
):
    """
    Train or retrain the NHITS model for a symbol.

    Parameters:
    - symbol: Trading symbol
    - days: Number of days of historical data to use (default: 90)
    - force: Force retraining even if model is up to date
    - timeframe: Timeframe for the model (M15, H1, D1) - default: H1

    This will train a new model or replace the existing one.
    The model will be saved with a timeframe suffix (e.g., EURUSD_M15, EURUSD_H1).
    """
    try:
        if not settings.nhits_enabled:
            raise HTTPException(
                status_code=503,
                detail="NHITS forecasting is disabled. Enable it in settings."
            )

        # Validate timeframe
        timeframe = timeframe.upper()
        if timeframe not in ["M15", "H1", "D1"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe '{timeframe}'. Use M15, H1, or D1."
            )

        # Use nhits_training_service which supports EasyInsight API
        from ..services.nhits_training_service import nhits_training_service
        result = await nhits_training_service.train_symbol(
            symbol=symbol,
            force=force,
            timeframe=timeframe
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed for {symbol}/{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@forecast_router.get("/forecast/{symbol}/model", response_model=ForecastModelInfo)
async def get_forecast_model_info(symbol: str):
    """
    Get information about the trained NHITS model for a symbol.

    Returns model metadata including last training date, samples used, and metrics.
    """
    try:
        forecast_service = get_forecast_service()
        info = forecast_service.get_model_info(symbol)
        return info
    except Exception as e:
        logger.error(f"Failed to get model info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Symbol Management Endpoints ====================

@symbol_router.get("/managed-symbols", response_model=list[ManagedSymbol])
async def get_managed_symbols(
    category: Optional[SymbolCategory] = None,
    status: Optional[SymbolStatus] = None,
    favorites_only: bool = False,
    with_data_only: bool = False,
):
    """
    Get all managed symbols with optional filtering.

    Parameters:
    - category: Filter by category (forex, crypto, stock, etc.)
    - status: Filter by status (active, inactive, suspended)
    - favorites_only: Only return favorites
    - with_data_only: Only return symbols with TimescaleDB data
    """
    try:
        symbols = await symbol_service.get_all_symbols(
            category=category,
            status=status,
            favorites_only=favorites_only,
            with_data_only=with_data_only,
        )
        return symbols
    except Exception as e:
        logger.error(f"Failed to get managed symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.get("/managed-symbols/stats", response_model=SymbolStats)
async def get_symbol_stats():
    """Get statistics about managed symbols."""
    try:
        stats = await symbol_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get symbol stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.get("/managed-symbols/search")
async def search_managed_symbols(query: str, limit: int = 20):
    """
    Search managed symbols by name, description, or tags.

    Parameters:
    - query: Search query string
    - limit: Maximum number of results (default: 20)
    """
    try:
        symbols = await symbol_service.search_symbols(query=query, limit=limit)
        return {
            "query": query,
            "count": len(symbols),
            "symbols": symbols,
        }
    except Exception as e:
        logger.error(f"Failed to search symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.post("/managed-symbols/import", response_model=SymbolImportResult)
async def import_symbols_from_timescaledb():
    """
    Import all symbols from TimescaleDB.

    This will:
    1. Query all distinct symbols from TimescaleDB
    2. Create new managed symbols for those not yet tracked
    3. Update existing symbols with latest data availability info
    4. Check NHITS model availability for each symbol
    """
    try:
        result = await symbol_service.import_from_timescaledb()
        return result
    except Exception as e:
        logger.error(f"Failed to import symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.get("/managed-symbols/available/easyinsight")
async def get_available_easyinsight_symbols():
    """
    Get list of available symbols from EasyInsight API for import selection.

    Returns symbols with their data availability info, marking which ones
    are already imported into the symbol management.
    """
    import httpx

    try:
        # Fetch symbols from EasyInsight API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{settings.easyinsight_api_url}/symbols")
            response.raise_for_status()
            easyinsight_symbols = response.json()

        # Get already imported symbols
        existing_symbols = {s.symbol for s in await symbol_service.get_all_symbols()}

        # Prepare result with import status
        result = []
        for s in easyinsight_symbols:
            symbol_id = s.get("symbol", "")
            result.append({
                "symbol": symbol_id,
                "category": s.get("category", "Other"),
                "count": s.get("count", 0),
                "earliest": s.get("earliest"),
                "latest": s.get("latest"),
                "already_imported": symbol_id in existing_symbols
            })

        return {
            "source": "easyinsight",
            "total": len(result),
            "already_imported": sum(1 for r in result if r["already_imported"]),
            "symbols": sorted(result, key=lambda x: (x["already_imported"], x["symbol"]))
        }
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch EasyInsight symbols: {e}")
        raise HTTPException(status_code=502, detail=f"EasyInsight API error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to get available symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.get("/managed-symbols/available/twelvedata")
async def get_available_twelvedata_symbols(
    category: str = "crypto",
    exchange: Optional[str] = None,
    country: Optional[str] = None,
    search: Optional[str] = None,
):
    """
    Get list of available symbols from Twelve Data API for import selection.

    Args:
        category: Asset category ('crypto', 'forex', 'stocks', 'etf', 'indices')
        exchange: Filter by exchange (e.g., 'NYSE', 'NASDAQ')
        country: Filter by country (e.g., 'United States')
        search: Search query for symbol name

    Returns symbols with their details, marking which ones are already imported.
    """
    try:
        if not twelvedata_service.is_available():
            raise HTTPException(status_code=503, detail="Twelve Data service not available")

        # Fetch symbols based on category
        symbols_data = []
        if category == "crypto":
            symbols_data = await twelvedata_service.get_cryptocurrencies()
        elif category == "forex":
            symbols_data = await twelvedata_service.get_forex_pairs()
        elif category == "stocks":
            symbols_data = await twelvedata_service.get_stock_list(exchange=exchange, country=country)
        elif category == "etf":
            symbols_data = await twelvedata_service.get_etf_list()
        elif category == "indices":
            symbols_data = await twelvedata_service.get_indices()

        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            symbols_data = [
                s for s in symbols_data
                if search_lower in s.get("symbol", "").lower()
                or search_lower in s.get("name", "").lower()
                or search_lower in s.get("currency_base", "").lower()
            ]

        # Get already imported symbols
        existing_symbols = {s.symbol for s in await symbol_service.get_all_symbols()}

        # Prepare result with import status
        result = []
        for s in symbols_data[:500]:  # Limit to 500 symbols for performance
            symbol_id = s.get("symbol", "")
            result.append({
                "symbol": symbol_id,
                "name": s.get("name") or s.get("currency_base", ""),
                "exchange": s.get("exchange", ""),
                "currency": s.get("currency") or s.get("currency_quote", ""),
                "country": s.get("country", ""),
                "type": s.get("type", category),
                "already_imported": symbol_id in existing_symbols
            })

        return {
            "source": "twelvedata",
            "category": category,
            "total": len(result),
            "already_imported": sum(1 for r in result if r["already_imported"]),
            "symbols": sorted(result, key=lambda x: (x["already_imported"], x["symbol"]))
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Twelve Data symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.post("/managed-symbols/import/selected")
async def import_selected_symbols(
    symbols: list[str],
    source: str = "easyinsight",
):
    """
    Import selected symbols into the symbol management.

    Args:
        symbols: List of symbol IDs to import
        source: Data source ('easyinsight' or 'twelvedata')

    Returns import result with counts.
    """
    import httpx

    result = {
        "imported": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
        "symbols": []
    }

    try:
        existing_symbols = {s.symbol: s for s in await symbol_service.get_all_symbols()}

        if source == "easyinsight":
            # Fetch full data from EasyInsight
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{settings.easyinsight_api_url}/symbols")
                response.raise_for_status()
                easyinsight_data = {s["symbol"]: s for s in response.json()}

            for symbol_id in symbols:
                try:
                    if symbol_id in existing_symbols:
                        result["skipped"] += 1
                        continue

                    symbol_info = easyinsight_data.get(symbol_id, {})
                    category = symbol_service._map_api_category_to_enum(symbol_info.get("category", "Other"))

                    request = SymbolCreateRequest(
                        symbol=symbol_id,
                        display_name=symbol_id,
                        category=category,
                    )
                    await symbol_service.create_symbol(request)
                    result["imported"] += 1
                    result["symbols"].append(symbol_id)
                except Exception as e:
                    result["errors"].append(f"{symbol_id}: {str(e)}")

        elif source == "twelvedata":
            for symbol_id in symbols:
                try:
                    if symbol_id in existing_symbols:
                        result["skipped"] += 1
                        continue

                    # Auto-detect category
                    category = symbol_service._detect_category(symbol_id)

                    request = SymbolCreateRequest(
                        symbol=symbol_id,
                        display_name=symbol_id,
                        category=category,
                    )
                    await symbol_service.create_symbol(request)
                    result["imported"] += 1
                    result["symbols"].append(symbol_id)
                except Exception as e:
                    result["errors"].append(f"{symbol_id}: {str(e)}")

        return result
    except Exception as e:
        logger.error(f"Failed to import selected symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.post("/managed-symbols", response_model=ManagedSymbol)
async def create_managed_symbol(request: SymbolCreateRequest):
    """
    Create a new managed symbol.

    The category will be auto-detected based on the symbol name if not specified.
    """
    try:
        symbol = await symbol_service.create_symbol(request)
        return symbol
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.get("/managed-symbols/{symbol_id}", response_model=ManagedSymbol)
async def get_managed_symbol(symbol_id: str):
    """Get a specific managed symbol by ID."""
    try:
        symbol = await symbol_service.get_symbol(symbol_id)
        if not symbol:
            raise HTTPException(status_code=404, detail=f"Symbol '{symbol_id}' not found")
        return symbol
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.put("/managed-symbols/{symbol_id}", response_model=ManagedSymbol)
async def update_managed_symbol(symbol_id: str, request: SymbolUpdateRequest):
    """Update an existing managed symbol."""
    try:
        symbol = await symbol_service.update_symbol(symbol_id, request)
        if not symbol:
            raise HTTPException(status_code=404, detail=f"Symbol '{symbol_id}' not found")
        return symbol
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.delete("/managed-symbols/{symbol_id}")
async def delete_managed_symbol(symbol_id: str):
    """Delete a managed symbol."""
    try:
        deleted = await symbol_service.delete_symbol(symbol_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Symbol '{symbol_id}' not found")
        return {"status": "deleted", "symbol": symbol_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.post("/managed-symbols/{symbol_id}/favorite", response_model=ManagedSymbol)
async def toggle_symbol_favorite(symbol_id: str):
    """Toggle favorite status for a symbol."""
    try:
        symbol = await symbol_service.toggle_favorite(symbol_id)
        if not symbol:
            raise HTTPException(status_code=404, detail=f"Symbol '{symbol_id}' not found")
        return symbol
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle favorite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.post("/managed-symbols/{symbol_id}/refresh", response_model=ManagedSymbol)
async def refresh_symbol_data(symbol_id: str):
    """
    Refresh TimescaleDB data information for a symbol.

    Updates data availability, timestamps, and NHITS model status.
    """
    try:
        symbol = await symbol_service.refresh_symbol_data(symbol_id)
        if not symbol:
            raise HTTPException(status_code=404, detail=f"Symbol '{symbol_id}' not found")
        return symbol
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Config Export/Import ====================

from ..services.config_export_service import (
    config_export_service,
    ConfigExportMetadata,
    ConfigExportResult,
    ConfigImportResult,
)
from fastapi.responses import JSONResponse

config_router = APIRouter()  # Config Export/Import


@config_router.get("/config/exports", response_model=list[ConfigExportMetadata])
async def list_config_exports():
    """List all available configuration exports."""
    try:
        exports = await config_export_service.list_exports()
        return exports
    except Exception as e:
        logger.error(f"Failed to list exports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.post("/config/export", response_model=ConfigExportResult)
async def create_config_export(
    description: Optional[str] = None,
    include_symbols: bool = True,
    include_strategies: bool = True,
    filename: Optional[str] = None,
):
    """
    Create a new configuration export.

    Exports symbols and strategies to a JSON file stored in the data volume.
    """
    try:
        result = await config_export_service.export_config(
            description=description,
            include_symbols=include_symbols,
            include_strategies=include_strategies,
            filename=filename,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to create export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.get("/config/exports/{filename}")
async def get_config_export(filename: str):
    """Get a specific export file for download."""
    try:
        data = await config_export_service.get_export(filename)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Export '{filename}' not found")

        return JSONResponse(
            content=data,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/json",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.delete("/config/exports/{filename}")
async def delete_config_export(filename: str):
    """Delete a saved export file."""
    try:
        success = await config_export_service.delete_export(filename)
        if not success:
            raise HTTPException(status_code=404, detail=f"Export '{filename}' not found")
        return {"success": True, "message": f"Export '{filename}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.post("/config/import", response_model=ConfigImportResult)
async def import_config(
    file: UploadFile = File(...),
    import_symbols: bool = True,
    import_strategies: bool = True,
    overwrite_existing: bool = False,
):
    """
    Import configuration from uploaded JSON file.

    Args:
        file: JSON file containing configuration data
        import_symbols: Import symbols from file
        import_strategies: Import strategies from file
        overwrite_existing: Overwrite existing entries with same ID
    """
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))

        result = await config_export_service.import_config(
            data=data,
            import_symbols=import_symbols,
            import_strategies=import_strategies,
            overwrite_existing=overwrite_existing,
        )
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to import config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.post("/config/import/{filename}", response_model=ConfigImportResult)
async def import_from_saved_export(
    filename: str,
    import_symbols: bool = True,
    import_strategies: bool = True,
    overwrite_existing: bool = False,
):
    """
    Import configuration from a previously saved export file.

    Args:
        filename: Name of the saved export file
        import_symbols: Import symbols from file
        import_strategies: Import strategies from file
        overwrite_existing: Overwrite existing entries with same ID
    """
    try:
        result = await config_export_service.import_from_saved(
            filename=filename,
            import_symbols=import_symbols,
            import_strategies=import_strategies,
            overwrite_existing=overwrite_existing,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to import from saved export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Twelve Data API ====================

@twelvedata_router.get("/twelvedata/status")
async def twelvedata_status():
    """Get Twelve Data service status."""
    return twelvedata_service.get_status()


@twelvedata_router.get("/twelvedata/stocks")
async def get_stocks(
    exchange: Optional[str] = None,
    country: Optional[str] = None,
    symbol_type: str = "Common Stock",
):
    """
    Get list of available stocks from Twelve Data.

    Args:
        exchange: Filter by exchange (e.g., 'NYSE', 'NASDAQ', 'XETRA')
        country: Filter by country (e.g., 'United States', 'Germany')
        symbol_type: Type of symbol (default: 'Common Stock')
    """
    stocks = await twelvedata_service.get_stock_list(
        exchange=exchange,
        country=country,
        symbol_type=symbol_type,
    )
    return {"count": len(stocks), "stocks": stocks}


@twelvedata_router.get("/twelvedata/forex")
async def get_forex_pairs():
    """Get list of available forex pairs from Twelve Data."""
    pairs = await twelvedata_service.get_forex_pairs()
    return {"count": len(pairs), "pairs": pairs}


@twelvedata_router.get("/twelvedata/crypto")
async def get_cryptocurrencies():
    """Get list of available cryptocurrencies from Twelve Data."""
    cryptos = await twelvedata_service.get_cryptocurrencies()
    return {"count": len(cryptos), "cryptocurrencies": cryptos}


@twelvedata_router.get("/twelvedata/etf")
async def get_etfs():
    """Get list of available ETFs from Twelve Data."""
    etfs = await twelvedata_service.get_etf_list()
    return {"count": len(etfs), "etfs": etfs}


@twelvedata_router.get("/twelvedata/indices")
async def get_indices():
    """Get list of available indices from Twelve Data."""
    indices = await twelvedata_service.get_indices()
    return {"count": len(indices), "indices": indices}


@twelvedata_router.get("/twelvedata/exchanges")
async def get_exchanges(asset_type: str = "stock"):
    """
    Get list of available exchanges from Twelve Data.

    Args:
        asset_type: Type of asset ('stock', 'etf', 'index')
    """
    exchanges = await twelvedata_service.get_exchanges(asset_type=asset_type)
    return {"count": len(exchanges), "exchanges": exchanges}


@twelvedata_router.get("/twelvedata/search")
async def search_symbols(query: str, limit: int = 20):
    """
    Search for symbols by name or ticker.

    Args:
        query: Search query (e.g., 'Apple', 'AAPL', 'Tesla')
        limit: Maximum number of results (default: 20)
    """
    results = await twelvedata_service.get_symbol_search(query=query, outputsize=limit)
    return {"count": len(results), "results": results}


@twelvedata_router.get("/twelvedata/quote/{symbol}")
async def get_quote(symbol: str, exchange: Optional[str] = None):
    """
    Get real-time quote for a symbol.

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD', 'BTC/USD')
        exchange: Specific exchange (optional)
    """
    quote = await twelvedata_service.get_quote(symbol=symbol, exchange=exchange)
    return quote


@twelvedata_router.get("/twelvedata/price/{symbol}")
async def get_price(symbol: str, exchange: Optional[str] = None):
    """
    Get current price for a symbol (lightweight endpoint).

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD')
        exchange: Specific exchange (optional)
    """
    price = await twelvedata_service.get_price(symbol=symbol, exchange=exchange)
    return price


@twelvedata_router.get("/twelvedata/time_series/{symbol}")
async def get_time_series(
    symbol: str,
    interval: str = "1day",
    outputsize: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: Optional[str] = None,
):
    """
    Get time series (OHLCV) data for a symbol.

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD', 'BTC/USD')
        interval: Time interval ('1min', '5min', '15min', '30min', '1h', '4h', '1day', '1week', '1month')
        outputsize: Number of data points (max 5000, default: 100)
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD')
        exchange: Specific exchange (optional)
    """
    data = await twelvedata_service.get_time_series(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
    )
    return data


@twelvedata_router.get("/twelvedata/indicator/{symbol}/{indicator}")
async def get_technical_indicator(
    symbol: str,
    indicator: str,
    interval: str = "1day",
    outputsize: int = 100,
    time_period: int = 14,
):
    """
    Get technical indicator data for a symbol.

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD')
        indicator: Indicator name ('sma', 'ema', 'rsi', 'macd', 'bbands', 'stoch', 'adx', 'atr', 'cci', 'obv')
        interval: Time interval (default: '1day')
        outputsize: Number of data points (default: 100)
        time_period: Period for indicator calculation (default: 14)
    """
    data = await twelvedata_service.get_technical_indicators(
        symbol=symbol,
        interval=interval,
        indicator=indicator,
        outputsize=outputsize,
        time_period=time_period,
    )
    return data


@twelvedata_router.get("/twelvedata/earnings")
async def get_earnings_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Get earnings calendar.

    Args:
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD')
    """
    earnings = await twelvedata_service.get_earnings_calendar(
        start_date=start_date,
        end_date=end_date,
    )
    return {"count": len(earnings), "earnings": earnings}


@twelvedata_router.get("/twelvedata/usage")
async def get_api_usage():
    """Get Twelve Data API usage statistics."""
    usage = await twelvedata_service.get_api_usage()
    return usage


# ==================== Live Data ====================

@symbol_router.get("/managed-symbols/{symbol}/live-data")
async def get_symbol_live_data(symbol: str):
    """
    Get live market data for a symbol from EasyInsight and TwelveData APIs.

    Returns the latest values and indicators from both data sources for comparison.

    Args:
        symbol: The symbol to get data for (e.g., 'EURUSD', 'BTCUSD')
    """
    import httpx

    result = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "easyinsight": None,
        "twelvedata": None,
        "errors": []
    }

    # Get symbol info for aliases
    managed_symbol = await symbol_service.get_symbol(symbol)
    aliases = managed_symbol.aliases if managed_symbol else []

    # Fetch EasyInsight data
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.easyinsight_api_url}/symbol-data-full/{symbol}",
                params={"limit": 1}
            )
            if response.status_code == 200:
                response_data = response.json()
                # Handle structured response with columns and data
                data = None
                if isinstance(response_data, dict) and "data" in response_data:
                    data_list = response_data.get("data", [])
                    if data_list:
                        data = data_list[0]
                elif isinstance(response_data, list) and response_data:
                    data = response_data[0]
                elif isinstance(response_data, dict):
                    data = response_data
                if data:
                    result["easyinsight"] = {
                        "source": "EasyInsight API",
                        "snapshot_time": data.get("snapshot_time"),
                        "category": data.get("category"),
                        "price": {
                            "bid": data.get("bid"),
                            "ask": data.get("ask"),
                            "spread": data.get("spread"),
                            "spread_pct": data.get("spread_pct"),
                        },
                        "ohlc": {
                            "m15": {
                                "open": data.get("m15_open"),
                                "high": data.get("m15_high"),
                                "low": data.get("m15_low"),
                                "close": data.get("m15_close"),
                            },
                            "h1": {
                                "open": data.get("h1_open"),
                                "high": data.get("h1_high"),
                                "low": data.get("h1_low"),
                                "close": data.get("h1_close"),
                            },
                            "d1": {
                                "open": data.get("d1_open"),
                                "high": data.get("d1_high"),
                                "low": data.get("d1_low"),
                                "close": data.get("d1_close"),
                            },
                        },
                        "indicators": {
                            "rsi": data.get("rsi"),
                            "macd": {
                                "main": data.get("macd_main"),
                                "signal": data.get("macd_signal"),
                            },
                            "stochastic": {
                                "main": data.get("sto_main"),
                                "signal": data.get("sto_signal"),
                            },
                            "cci": data.get("cci"),
                            "adx": {
                                "main": data.get("adx_main"),
                                "plus_di": data.get("adx_plusdi"),
                                "minus_di": data.get("adx_minusdi"),
                            },
                            "ma_10": data.get("ma_10"),
                            "ichimoku": {
                                "tenkan": data.get("ichimoku_tenkan"),
                                "kijun": data.get("ichimoku_kijun"),
                                "senkou_a": data.get("ichimoku_senkoua"),
                                "senkou_b": data.get("ichimoku_senkoub"),
                                "chikou": data.get("ichimoku_chikou"),
                            },
                            "bollinger": {
                                "upper": data.get("bb_upper"),
                                "base": data.get("bb_base"),
                                "lower": data.get("bb_lower"),
                            },
                            "atr": {
                                "d1": data.get("atr_d1"),
                                "d1_pct": data.get("atr_pct_d1"),
                            },
                            "range_d1": data.get("range_d1"),
                            "pivot_points": {
                                "s1_m5": data.get("s1_level_m5"),
                                "r1_m5": data.get("r1_level_m5"),
                            },
                            "strength": {
                                "h4": data.get("strength_4h"),
                                "d1": data.get("strength_1d"),
                                "w1": data.get("strength_1w"),
                            },
                        },
                    }
            else:
                result["errors"].append(f"EasyInsight: HTTP {response.status_code}")
    except Exception as e:
        result["errors"].append(f"EasyInsight: {str(e)}")

    # Fetch TwelveData quote - try symbol and aliases
    td_symbol = symbol
    # Convert symbol format for TwelveData (e.g., EURUSD -> EUR/USD)
    for alias in aliases:
        if "/" in alias:
            td_symbol = alias
            break

    try:
        quote = await twelvedata_service.get_quote(symbol=td_symbol)
        if quote and "error" not in quote:
            # Calculate spread from bid/ask if available, or estimate from close
            td_bid = quote.get("bid")
            td_ask = quote.get("ask")
            td_spread = None
            td_spread_pct = None

            if td_bid and td_ask:
                try:
                    bid_val = float(td_bid)
                    ask_val = float(td_ask)
                    td_spread = ask_val - bid_val
                    if bid_val > 0:
                        td_spread_pct = (td_spread / bid_val) * 100
                except (ValueError, TypeError):
                    pass

            result["twelvedata"] = {
                "source": "Twelve Data API",
                "symbol_used": td_symbol,
                "name": quote.get("name"),
                "exchange": quote.get("exchange"),
                "currency": quote.get("currency"),
                "datetime": quote.get("datetime"),
                "timestamp": quote.get("timestamp"),
                "bid_ask": {
                    "bid": td_bid,
                    "ask": td_ask,
                    "spread": td_spread,
                    "spread_pct": td_spread_pct,
                },
                "price": {
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "close": quote.get("close"),
                    "previous_close": quote.get("previous_close"),
                },
                "change": quote.get("change"),
                "percent_change": quote.get("percent_change"),
                "volume": quote.get("volume"),
                "average_volume": quote.get("average_volume"),
                "is_market_open": quote.get("is_market_open"),
                "fifty_two_week": {
                    "low": quote.get("fifty_two_week", {}).get("low") if isinstance(quote.get("fifty_two_week"), dict) else None,
                    "high": quote.get("fifty_two_week", {}).get("high") if isinstance(quote.get("fifty_two_week"), dict) else None,
                },
            }
        elif quote and "error" in quote:
            result["errors"].append(f"TwelveData: {quote.get('error')}")
    except Exception as e:
        result["errors"].append(f"TwelveData: {str(e)}")

    return result


# ==================== Router Export ====================

def get_all_routers():
    """
    Return all thematic routers with their tags for inclusion in main app.

    Returns:
        list: List of tuples (router, prefix, tags, description)
    """
    return [
        (system_router, "/api/v1", ["🖥️ System & Monitoring"], {
            "name": "System",
            "description": "Health checks, version info, and system information"
        }),
        (sync_router, "/api/v1", ["🔄 TimescaleDB Sync"], {
            "name": "Sync",
            "description": "TimescaleDB synchronization - start/stop sync service, manual sync, and status monitoring"
        }),
        (trading_router, "/api/v1", ["📊 Trading Analysis"], {
            "name": "Trading",
            "description": "Trading recommendations, symbol analysis, and market insights"
        }),
        (forecast_router, "/api/v1", ["🔮 NHITS Forecast"], {
            "name": "Forecast",
            "description": "Neural price forecasting with NHITS models - generate predictions and view model info"
        }),
        (training_router, "/api/v1", ["🎓 NHITS Training"], {
            "name": "Training",
            "description": "NHITS model training - batch training, progress monitoring, and performance evaluation"
        }),
        (symbol_router, "/api/v1", ["📈 Symbol Management"], {
            "name": "Symbols",
            "description": "Manage trading symbols, categories, and data availability"
        }),
        (strategy_router, "/api/v1", ["🎯 Trading Strategies"], {
            "name": "Strategies",
            "description": "Trading strategy management, import/export, and defaults"
        }),
        (rag_router, "/api/v1", ["🧠 RAG & Knowledge Base"], {
            "name": "RAG",
            "description": "Retrieval-Augmented Generation - document management and semantic search"
        }),
        (llm_router, "/api/v1", ["🤖 LLM Service"], {
            "name": "LLM",
            "description": "Large Language Model management and status"
        }),
        (query_log_router, "/api/v1", ["📝 Query Logs & Analytics"], {
            "name": "Analytics",
            "description": "Query logging, statistics, and audit trails"
        }),
        (twelvedata_router, "/api/v1", ["📈 Twelve Data API"], {
            "name": "Twelve Data",
            "description": "Access to Twelve Data API - stocks, forex, crypto, ETFs, indices, and technical indicators"
        }),
    ]
