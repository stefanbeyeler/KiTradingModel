"""API routes for the KI Trading Model service."""
# Health check now includes NHITS status

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
)
from ..services import AnalysisService, LLMService, StrategyService
from ..services.query_log_service import query_log_service, QueryLogEntry
from ..services.symbol_service import symbol_service


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

@router.get("/forecast/status")
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


@router.get("/forecast/models", response_model=list[ForecastModelInfo])
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


# ==================== NHITS Batch Training Endpoints ====================
# NOTE: These must be defined BEFORE parameterized routes like /forecast/{symbol}

@router.get("/forecast/training/status")
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


@router.get("/forecast/training/symbols")
async def get_trainable_symbols():
    """
    Get list of symbols available for NHITS training.

    Returns all symbols that have sufficient data in TimescaleDB.
    """
    try:
        training_service = get_training_service()

        # Ensure training service has database connection
        if not training_service._db_pool:
            import asyncpg
            pool = await asyncpg.create_pool(
                host=settings.timescaledb_host,
                port=settings.timescaledb_port,
                database=settings.timescaledb_database,
                user=settings.timescaledb_user,
                password=settings.timescaledb_password,
                min_size=1,
                max_size=5
            )
            await training_service.connect(pool)

        symbols = await training_service.get_available_symbols()

        return {
            "count": len(symbols),
            "symbols": symbols
        }

    except Exception as e:
        logger.error(f"Failed to get trainable symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/train-all")
async def train_all_models(
    symbols: list[str] | None = None,
    force: bool = False,
    background: bool = True
):
    """
    Train NHITS models for all (or specified) symbols.

    Parameters:
    - symbols: Optional list of specific symbols to train (default: all available)
    - force: Force retraining even if models are up to date
    - background: Run training in background (default: True)

    Returns training summary or task status if running in background.
    """
    import asyncio
    try:
        if not settings.nhits_enabled:
            raise HTTPException(
                status_code=503,
                detail="NHITS forecasting is disabled. Enable it in settings."
            )

        training_service = get_training_service()

        # Ensure training service has database connection
        if not training_service._db_pool:
            import asyncpg
            pool = await asyncpg.create_pool(
                host=settings.timescaledb_host,
                port=settings.timescaledb_port,
                database=settings.timescaledb_database,
                user=settings.timescaledb_user,
                password=settings.timescaledb_password,
                min_size=1,
                max_size=5
            )
            await training_service.connect(pool)

        if background:
            # Start training in background
            asyncio.create_task(
                training_service.train_all_symbols(
                    symbols=symbols,
                    force=force
                )
            )
            return {
                "status": "started",
                "message": "Training started in background",
                "symbols": symbols or "all available"
            }
        else:
            # Run training synchronously
            result = await training_service.train_all_symbols(
                symbols=symbols,
                force=force
            )
            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/training/cancel")
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


# =============================================================================
# Model Improvement & Performance Endpoints (MUST be before {symbol} routes!)
# =============================================================================

@router.get("/forecast/performance")
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


@router.get("/forecast/evaluated")
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


@router.get("/forecast/retraining-needed")
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


@router.post("/forecast/evaluate")
async def evaluate_predictions():
    """
    Manually trigger evaluation of pending predictions.

    This compares past predictions with actual prices and updates
    model performance metrics.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        from ..main import sync_service

        # Check if sync_service has an active database pool
        if not sync_service._pool:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available. Please ensure RAG sync is enabled."
            )

        evaluated = await model_improvement_service.evaluate_pending_predictions(
            sync_service._pool
        )

        return {
            "success": True,
            "evaluated": evaluated,
            "total_evaluated": sum(evaluated.values()),
            "message": f"Evaluated {sum(evaluated.values())} predictions"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/retrain-poor-performers")
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
            "symbols": symbols
        }
    except Exception as e:
        logger.error(f"Failed to retrain poor performers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NHITS Symbol-specific Endpoints ====================

@router.get("/forecast/{symbol}", response_model=ForecastResult)
async def get_forecast(
    symbol: str,
    horizon: int = 24,
    retrain: bool = False
):
    """
    Generate NHITS price forecast for a symbol.

    Parameters:
    - symbol: Trading symbol (e.g., EURUSD)
    - horizon: Forecast horizon in hours (default: 24, max: 168)
    - retrain: Force model retraining before forecast (default: false)

    Returns predicted prices with confidence intervals for the specified horizon.
    """
    try:
        # Check if NHITS is enabled
        if not settings.nhits_enabled:
            raise HTTPException(
                status_code=503,
                detail="NHITS forecasting is disabled. Enable it in settings."
            )

        # Fetch time series data
        from datetime import timedelta
        time_series = await analysis_service._fetch_time_series(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )

        if not time_series:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")

        # Create config
        config = ForecastConfig(
            symbol=symbol,
            horizon=min(horizon, 168),  # Max 7 days
            retrain=retrain
        )

        # Generate forecast
        forecast_service = get_forecast_service()
        result = await forecast_service.forecast(
            time_series=time_series,
            symbol=symbol,
            config=config
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/{symbol}/train", response_model=ForecastTrainingResult)
async def train_forecast_model(
    symbol: str,
    days: int = 90
):
    """
    Train or retrain the NHITS model for a symbol.

    Parameters:
    - symbol: Trading symbol
    - days: Number of days of historical data to use (default: 90)

    This will train a new model or replace the existing one.
    """
    try:
        if not settings.nhits_enabled:
            raise HTTPException(
                status_code=503,
                detail="NHITS forecasting is disabled. Enable it in settings."
            )

        # Fetch extended historical data
        from datetime import timedelta
        time_series = await analysis_service._fetch_time_series(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now()
        )

        if not time_series:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")

        # Train model
        forecast_service = get_forecast_service()
        result = await forecast_service.train(
            time_series=time_series,
            symbol=symbol
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/{symbol}/model", response_model=ForecastModelInfo)
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

@router.get("/managed-symbols", response_model=list[ManagedSymbol])
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


@router.get("/managed-symbols/stats", response_model=SymbolStats)
async def get_symbol_stats():
    """Get statistics about managed symbols."""
    try:
        stats = await symbol_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get symbol stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/managed-symbols/search")
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


@router.post("/managed-symbols/import", response_model=SymbolImportResult)
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


@router.post("/managed-symbols", response_model=ManagedSymbol)
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


@router.get("/managed-symbols/{symbol_id}", response_model=ManagedSymbol)
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


@router.put("/managed-symbols/{symbol_id}", response_model=ManagedSymbol)
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


@router.delete("/managed-symbols/{symbol_id}")
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


@router.post("/managed-symbols/{symbol_id}/favorite", response_model=ManagedSymbol)
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


@router.post("/managed-symbols/{symbol_id}/refresh", response_model=ManagedSymbol)
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
