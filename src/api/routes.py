"""API routes for the KI Trading Model service."""
# Health check now includes NHITS status

import json
import os
from datetime import datetime, timezone
from typing import Optional, List
import httpx
import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.responses import PlainTextResponse
from loguru import logger

from ..config import settings
from ..config.timeframes import (
    Timeframe,
    normalize_timeframe,
    normalize_timeframe_safe,
    to_twelvedata,
    get_candles_per_day,
    calculate_limit_for_days,
    is_valid_timeframe,
)
from ..config.indicators import (
    IndicatorCategory,
    IndicatorConfig,
    INDICATORS_REGISTRY,
    CATEGORY_DESCRIPTIONS,
    get_indicator,
    get_all_indicators,
    get_enabled_indicators,
    get_indicators_by_category,
    get_twelvedata_indicators,
    get_easyinsight_indicators,
    search_indicators,
    get_indicator_stats,
)
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
# Optional service imports (may not be available in all containers)
try:
    from ..services import AnalysisService
    _analysis_available = True
except ImportError:
    AnalysisService = None
    _analysis_available = False

try:
    from ..services import LLMService
    _llm_available = True
except ImportError:
    LLMService = None
    _llm_available = False

try:
    from ..services import StrategyService
    _strategy_available = True
except ImportError:
    StrategyService = None
    _strategy_available = False

from ..services.query_log_service import query_log_service, QueryLogEntry
from ..services.symbol_service import symbol_service
from ..services.event_based_training_service import event_based_training_service
from ..utils.timezone_utils import to_utc, format_for_display, format_utc_iso, get_timezone_info

# Optional service imports - only available in Data Service container
# These services access external APIs directly (TwelveData, Yahoo Finance)
# Other containers (NHITS, RAG, LLM, etc.) must access data via Data Service HTTP API
try:
    from ..services.twelvedata_service import twelvedata_service
    _twelvedata_available = True
except ImportError:
    twelvedata_service = None  # type: ignore
    _twelvedata_available = False

try:
    from ..services.yfinance_service import yfinance_service
    _yfinance_available = True
except ImportError:
    yfinance_service = None  # type: ignore
    _yfinance_available = False


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
easyinsight_router = APIRouter()  # EasyInsight API (TimescaleDB)
config_router = APIRouter()  # Configuration & Settings
patterns_router = APIRouter()  # Candlestick Pattern Detection
yfinance_router = APIRouter()  # Yahoo Finance API
external_sources_router = APIRouter()  # External Data Sources (Economic, Sentiment, etc.)
backup_router = APIRouter()  # Backup & Restore
indicators_router = APIRouter()  # Technical Indicators Registry

# Service instances (created only if available)
analysis_service = AnalysisService() if _analysis_available else None
llm_service = LLMService() if _llm_available else None
strategy_service = StrategyService() if _strategy_available else None


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


async def get_favorites_from_data_service() -> dict:
    """
    Fetch favorite symbols and their metadata from Data Service.

    Returns:
        Dict mapping symbol names to their metadata (category, is_favorite, etc.)
    """
    import httpx

    data_service_url = getattr(settings, 'data_service_url', 'http://localhost:3001')
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{data_service_url}/api/v1/managed-symbols")
            if response.status_code == 200:
                symbols = response.json()
                return {
                    s["symbol"]: {
                        "is_favorite": s.get("is_favorite", False),
                        "category": s.get("category"),
                        "display_name": s.get("display_name"),
                    }
                    for s in symbols
                }
    except Exception as e:
        logger.warning(f"Failed to fetch favorites from Data Service: {e}")
    return {}


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


@system_router.get("/cache/status", tags=["1. System"])
async def get_cache_status():
    """
    Check Redis cache status and statistics.

    Returns connection status, memory usage, and hit/miss statistics.
    Used by Watchdog for monitoring Redis health.
    """
    try:
        from ..services.cache_service import cache_service

        # Get health check (includes Redis ping)
        health = await cache_service.health_check()

        # Get statistics
        stats = cache_service.get_stats()

        return {
            "status": health.get("status", "unknown"),
            "redis_connected": health.get("redis_connected", False),
            "redis_memory_used": health.get("redis_memory_used", "N/A"),
            "fallback_active": health.get("fallback_active", True),
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache status check failed: {e}")
        return {
            "status": "error",
            "redis_connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
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
    try:
        # Try to extract symbol from query if not provided
        detected_symbol = symbol
        if not detected_symbol:
            # Known trading symbols - more specific patterns to avoid false positives
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

        # Get RAG context if enabled
        rag_context = []
        if use_rag:
            try:
                rag_svc = get_rag_service()
                # Query RAG with the user's question
                rag_results = await rag_svc.query(query, n_results=3, symbol=detected_symbol)
                rag_context = [doc.get("content", "") for doc in rag_results.get("results", [])]
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Build prompt with context
        context_text = ""
        if rag_context:
            context_text = "\n\nRelevanter Kontext:\n" + "\n---\n".join(rag_context)

        symbol_text = f" für {detected_symbol}" if detected_symbol else ""

        system_prompt = f"""Du bist ein erfahrener Trading-Assistent. Beantworte Fragen zu Märkten,
Trading-Strategien und Finanzanalysen auf Deutsch. Sei präzise und hilfreich.
{context_text}"""

        user_prompt = query

        # Call LLM
        response = await llm_service.generate(
            prompt=user_prompt,
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


def parse_tegrastats() -> dict:
    """
    Read tegrastats data from JSON file written by host service.
    Falls back to direct tegrastats call if file not available.
    Returns dict with cpu_temp, gpu_temp, gpu_power_mw, total_power_mw.
    """
    import subprocess
    import re

    result = {
        "cpu_temp": None,
        "gpu_temp": None,
        "gpu_power_mw": None,
        "total_power_mw": None,
        "available": False
    }

    # First try to read from JSON file (written by host service)
    # Check both container-mounted path and local path
    tegrastats_file = "/host_tmp/tegrastats.json"
    if not os.path.exists(tegrastats_file):
        tegrastats_file = "/tmp/tegrastats.json"
    try:
        if os.path.exists(tegrastats_file):
            with open(tegrastats_file, 'r') as f:
                data = json.load(f)
            if data.get("available", False):
                result["cpu_temp"] = data.get("cpu_temp")
                result["gpu_temp"] = data.get("gpu_temp")
                result["gpu_power_mw"] = data.get("gpu_power_mw")
                result["total_power_mw"] = data.get("total_power_mw")
                result["available"] = True
                return result
    except Exception as e:
        logger.debug(f"Could not read tegrastats file: {e}")

    # Fallback: Try running tegrastats directly (works on host, not in container)
    try:
        proc = subprocess.run(
            ['tegrastats', '--interval', '100'],
            capture_output=True,
            text=True,
            timeout=1
        )

        if proc.returncode != 0 or not proc.stdout:
            return result

        line = proc.stdout.strip().split('\n')[0]
        result["available"] = True

        # Parse CPU temperature: cpu@40.218C
        cpu_temp_match = re.search(r'cpu@([\d.]+)C', line)
        if cpu_temp_match:
            result["cpu_temp"] = float(cpu_temp_match.group(1))

        # Parse GPU temperature: gpu@42.156C
        gpu_temp_match = re.search(r'gpu@([\d.]+)C', line)
        if gpu_temp_match:
            result["gpu_temp"] = float(gpu_temp_match.group(1))

        # Parse GPU power: VDD_GPU 9880mW/9880mW (current/average)
        gpu_power_match = re.search(r'VDD_GPU\s+([\d]+)mW', line)
        if gpu_power_match:
            result["gpu_power_mw"] = int(gpu_power_match.group(1))

        # Parse total power: VIN 29942mW/14971mW
        total_power_match = re.search(r'VIN\s+([\d]+)mW', line)
        if total_power_match:
            result["total_power_mw"] = int(total_power_match.group(1))

    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        # tegrastats not available (not on Jetson)
        pass
    except Exception as e:
        logger.warning(f"tegrastats parse error: {e}")

    return result


@system_router.get("/system/metrics")
async def get_system_metrics():
    """
    Get real-time CPU and GPU utilization metrics.

    Returns current usage percentages for monitoring dashboards.
    Updates on each request - designed for polling (e.g., every 2-5 seconds).
    Includes Jetson-specific metrics (temperatures, power) if running on Jetson Orin.
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

        # Try to get Jetson tegrastats data
        tegra = parse_tegrastats()

        metrics = {
            "cpu": {
                "percent": cpu_percent,
                "cores_physical": cpu_count,
                "cores_logical": cpu_count_logical,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "temp_celsius": tegra.get("cpu_temp"),
            },
            "memory": {
                "percent": memory.percent,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
            },
            "gpu": None,
            "power": {
                "gpu_watts": round(tegra["gpu_power_mw"] / 1000, 1) if tegra.get("gpu_power_mw") else None,
                "total_watts": round(tegra["total_power_mw"] / 1000, 1) if tegra.get("total_power_mw") else None,
                "available": tegra.get("available", False)
            },
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
                    "temp_celsius": tegra.get("gpu_temp"),
                    "power_watts": round(tegra["gpu_power_mw"] / 1000, 1) if tegra.get("gpu_power_mw") else None,
                }
            except Exception as e:
                metrics["gpu"] = {"available": True, "error": str(e)}
        else:
            metrics["gpu"] = {"available": False}

        return metrics

    except Exception as e:
        logger.error(f"System metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/system/storage")
async def get_storage_metrics():
    """
    Get storage metrics including disk usage and application data directory sizes.

    Returns disk space and data directory information for the KITradingModel project.
    """
    import psutil
    import os

    def get_directory_size(path: str) -> int:
        """Get total size of a directory in bytes."""
        total_size = 0
        try:
            if os.path.exists(path):
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except (OSError, FileNotFoundError):
                            pass
        except Exception:
            pass
        return total_size

    def format_size(size_bytes: int) -> str:
        """Format bytes to human readable string."""
        if size_bytes >= 1024**3:
            return f"{size_bytes / (1024**3):.2f} GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / (1024**2):.2f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.2f} KB"
        return f"{size_bytes} B"

    try:
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_info = {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": round(disk.percent, 1)
        }

        # Application data directories - these are the mounted volumes
        data_dirs = {
            "models": "/app/data/models",
            "logs": "/app/logs",
            "data": "/app/data"
        }

        volumes = []
        total_app_size = 0

        for name, path in data_dirs.items():
            size = get_directory_size(path)
            total_app_size += size
            if os.path.exists(path):
                volumes.append({
                    "name": name,
                    "path": path,
                    "size": format_size(size),
                    "size_bytes": size
                })

        # Known project containers (static list since we can't query Docker from inside)
        containers = [
            {"name": "trading-nhits", "service": "NHITS Service", "port": 3002},
            {"name": "trading-data", "service": "Data Service", "port": 3001},
            {"name": "trading-rag", "service": "RAG Service", "port": 3003},
            {"name": "trading-llm", "service": "LLM Service", "port": 3004},
            {"name": "trading-frontend", "service": "Frontend", "port": 3000}
        ]

        # Docker usage summary based on data directories
        docker_usage = {
            "app_data_size": format_size(total_app_size),
            "models_size": format_size(get_directory_size("/app/data/models")),
            "logs_size": format_size(get_directory_size("/app/logs")),
            "total_data_size": format_size(get_directory_size("/app/data"))
        }

        return {
            "disk": disk_info,
            "app_storage": {
                "usage": docker_usage,
                "volumes": volumes,
                "containers": containers
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Storage metrics failed: {e}")
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
# Cache Management Endpoints
# ============================================

@system_router.get("/cache/overview", tags=["1. System"])
async def get_cache_overview():
    """
    Get comprehensive cache overview with statistics per category.

    Returns detailed information about:
    - Cache entries per category (OHLCV, INDICATORS, MARKET_DATA, etc.)
    - Symbols and timeframes cached per category
    - TTL configuration per category
    - Redis connection status and memory usage
    """
    from ..services.cache_service import cache_service, CacheCategory, DEFAULT_TTL

    try:
        # Connect if not connected
        if not cache_service._redis_available:
            await cache_service.connect()

        # Get detailed stats
        detailed_stats = await cache_service.get_detailed_stats()
        health = await cache_service.health_check()
        basic_stats = cache_service.get_stats()

        # Add TTL info for all categories
        ttl_config = {cat.value: {"ttl_seconds": ttl, "ttl_display": _format_ttl(ttl)}
                      for cat, ttl in DEFAULT_TTL.items()}

        return {
            "status": "ok",
            "cache_backend": "redis" if health["redis_connected"] else "memory",
            "redis": {
                "connected": health["redis_connected"],
                "memory_used": health["redis_memory_used"],
            },
            "statistics": basic_stats,
            "categories": detailed_stats["categories"],
            "summary": detailed_stats["summary"],
            "ttl_config": ttl_config,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache overview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/cache/category/{category}", tags=["1. System"])
async def get_cache_category_entries(category: str, limit: int = 100):
    """
    Get entries for a specific cache category.

    Args:
        category: Cache category (market, ohlcv, indicators, symbols, metadata, sentiment, economic, onchain, training)
        limit: Maximum number of entries to return (default 100)

    Returns list of cached entries with symbol, timeframe, and TTL information.
    """
    from ..services.cache_service import cache_service, CacheCategory

    try:
        # Validate category
        try:
            cat = CacheCategory(category)
        except ValueError:
            valid_categories = [c.value for c in CacheCategory]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{category}'. Valid: {valid_categories}"
            )

        # Connect if not connected
        if not cache_service._redis_available:
            await cache_service.connect()

        entries = await cache_service.get_category_entries(cat, limit=limit)

        return {
            "category": category,
            "entry_count": len(entries),
            "entries": entries,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache category entries failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.delete("/cache/category/{category}", tags=["1. System"])
async def clear_cache_category(category: str):
    """
    Clear all entries for a specific cache category.

    Args:
        category: Cache category to clear

    Returns number of deleted entries.
    """
    from ..services.cache_service import cache_service, CacheCategory

    try:
        # Validate category
        try:
            cat = CacheCategory(category)
        except ValueError:
            valid_categories = [c.value for c in CacheCategory]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{category}'. Valid: {valid_categories}"
            )

        # Connect if not connected
        if not cache_service._redis_available:
            await cache_service.connect()

        deleted = await cache_service.clear_category(cat)

        return {
            "status": "ok",
            "category": category,
            "deleted_entries": deleted,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache category clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.delete("/cache/all", tags=["1. System"])
async def clear_all_cache():
    """
    Clear the entire cache (all categories).

    WARNING: This will delete all cached data!

    Returns number of deleted entries.
    """
    from ..services.cache_service import cache_service

    try:
        # Connect if not connected
        if not cache_service._redis_available:
            await cache_service.connect()

        deleted = await cache_service.clear_all()

        return {
            "status": "ok",
            "deleted_entries": deleted,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache clear all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _format_ttl(seconds: int) -> str:
    """Format TTL in human-readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    elif seconds < 86400:
        return f"{seconds // 3600}h"
    else:
        return f"{seconds // 86400}d"


# ============================================
# Pre-Fetching Endpoints
# ============================================

@system_router.get("/prefetch/status", tags=["1. System"])
async def get_prefetch_status():
    """
    Get pre-fetching service status and statistics.

    Returns configuration, running state, and fetch statistics.
    """
    from ..services.prefetch_service import prefetch_service

    try:
        stats = prefetch_service.get_stats()
        config = prefetch_service.get_config()

        return {
            "status": "ok",
            "running": stats["running"],
            "config": config,
            "statistics": {
                "last_run": stats["last_run"],
                "total_runs": stats["total_runs"],
                "symbols_fetched": stats["symbols_fetched"],
                "timeframes_fetched": stats["timeframes_fetched"],
                "indicators_fetched": stats.get("indicators_fetched", 0),
                "cache_entries_created": stats["cache_entries_created"],
                "indicator_entries_created": stats.get("indicator_entries_created", 0),
                "errors": stats["errors"],
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prefetch status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.post("/prefetch/start", tags=["1. System"])
async def start_prefetch_service():
    """
    Start the pre-fetching service.

    The service will immediately fetch data for all configured symbols/timeframes
    and then continue with periodic updates.
    """
    from ..services.prefetch_service import prefetch_service

    try:
        await prefetch_service.start()
        stats = prefetch_service.get_stats()

        return {
            "status": "ok",
            "message": "Pre-fetch service started",
            "running": stats["running"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prefetch start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.post("/prefetch/stop", tags=["1. System"])
async def stop_prefetch_service():
    """
    Stop the pre-fetching service.
    """
    from ..services.prefetch_service import prefetch_service

    try:
        await prefetch_service.stop()

        return {
            "status": "ok",
            "message": "Pre-fetch service stopped",
            "running": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prefetch stop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.post("/prefetch/configure", tags=["1. System"])
async def configure_prefetch_service(
    enabled: Optional[bool] = None,
    timeframes: Optional[List[str]] = Query(None),
    max_symbols: Optional[int] = None,
    favorites_only: Optional[bool] = None,
    refresh_interval: Optional[int] = None,
    ohlcv_limit: Optional[int] = None,
    api_delay: Optional[float] = None,
    indicators: Optional[List[str]] = Query(None),
    indicator_limit: Optional[int] = None,
):
    """
    Configure the pre-fetching service.

    Args:
        enabled: Enable/disable pre-fetching
        timeframes: List of timeframes to pre-fetch (e.g. ["1h", "4h", "1day"])
        max_symbols: Maximum number of symbols to pre-fetch
        favorites_only: Only pre-fetch favorite symbols
        refresh_interval: Interval between pre-fetch runs (seconds)
        ohlcv_limit: Number of OHLCV data points to fetch per request
        api_delay: Delay between API calls for rate limiting (seconds)
        indicators: List of indicators to pre-fetch (e.g. ["rsi", "macd", "bbands"])
        indicator_limit: Number of indicator data points to fetch per request
    """
    from ..services.prefetch_service import prefetch_service

    try:
        # Build config dict from non-None parameters
        config_updates = {}
        if enabled is not None:
            config_updates["enabled"] = enabled
        if timeframes is not None:
            config_updates["timeframes"] = timeframes
        if max_symbols is not None:
            config_updates["max_symbols"] = max_symbols
        if favorites_only is not None:
            config_updates["favorites_only"] = favorites_only
        if refresh_interval is not None:
            config_updates["refresh_interval"] = refresh_interval
        if ohlcv_limit is not None:
            config_updates["ohlcv_limit"] = ohlcv_limit
        if api_delay is not None:
            config_updates["api_delay"] = api_delay
        if indicators is not None:
            config_updates["indicators"] = indicators
        if indicator_limit is not None:
            config_updates["indicator_limit"] = indicator_limit

        if config_updates:
            await prefetch_service.configure(**config_updates)

        return {
            "status": "ok",
            "message": "Pre-fetch configuration updated and saved to Redis",
            "updated": config_updates,
            "config": prefetch_service.get_config(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prefetch configure failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.post("/prefetch/run", tags=["1. System"])
async def run_prefetch_now(background_tasks: BackgroundTasks):
    """
    Trigger an immediate pre-fetch run.

    The pre-fetch runs in the background and returns immediately.
    """
    from ..services.prefetch_service import prefetch_service

    try:
        # Run in background
        background_tasks.add_task(prefetch_service._run_prefetch)

        return {
            "status": "ok",
            "message": "Pre-fetch triggered (running in background)",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prefetch run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.post("/prefetch/symbol/{symbol}", tags=["1. System"])
async def prefetch_single_symbol(
    symbol: str,
    timeframes: Optional[List[str]] = Query(None)
):
    """
    Pre-fetch data for a single symbol.

    Args:
        symbol: Symbol to pre-fetch (e.g. "BTCUSD", "EUR/USD")
        timeframes: Optional list of timeframes (default: configured timeframes)

    Returns pre-fetch results for the symbol.
    """
    from ..services.prefetch_service import prefetch_service

    try:
        result = await prefetch_service.prefetch_symbol(symbol, timeframes)

        return {
            "status": "ok",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prefetch symbol failed: {e}")
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

    Returns information about all models that have been trained,
    including favorite status from Data Service.
    """
    try:
        forecast_service = get_forecast_service()
        models = forecast_service.list_models()

        # Fetch favorites from Data Service
        favorites_data = await get_favorites_from_data_service()

        # Enrich models with favorite and category info
        enriched_models = []
        for model in models:
            # Extract base symbol (remove timeframe suffix like _M15, _H1, _D1)
            base_symbol = model.symbol
            for suffix in ["_M15", "_H1", "_D1"]:
                if base_symbol.endswith(suffix):
                    base_symbol = base_symbol[:-len(suffix)]
                    break

            # Get favorite info from Data Service
            symbol_info = favorites_data.get(base_symbol, {})

            # Create enriched model
            enriched = ForecastModelInfo(
                symbol=model.symbol,
                model_exists=model.model_exists,
                model_path=model.model_path,
                last_trained=model.last_trained,
                training_samples=model.training_samples,
                horizon=model.horizon,
                input_size=model.input_size,
                metrics=model.metrics,
                is_favorite=symbol_info.get("is_favorite", False),
                category=symbol_info.get("category"),
            )
            enriched_models.append(enriched)

        return enriched_models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@forecast_router.get("/forecast/models/by-timeframe")
async def list_models_by_timeframe():
    """
    List all trained NHITS models grouped by timeframe.

    Returns models organized by their timeframe suffix (M15, H1, D1).
    """
    try:
        forecast_service = get_forecast_service()
        models = forecast_service.list_models()

        # Group models by timeframe (only M15, H1, D1)
        timeframe_groups = {
            "M15": [],      # 15 minutes
            "H1": [],       # Hourly
            "D1": [],       # Daily
        }

        for model in models:
            if not model.model_exists:
                continue
            symbol = model.symbol

            # Check for timeframe suffix
            if symbol.endswith("_M15"):
                timeframe_groups["M15"].append(symbol.replace("_M15", ""))
            elif symbol.endswith("_H1"):
                timeframe_groups["H1"].append(symbol.replace("_H1", ""))
            elif symbol.endswith("_D1"):
                timeframe_groups["D1"].append(symbol.replace("_D1", ""))
            # Skip models without timeframe suffix (legacy models)

        # Sort symbols in each group
        for key in timeframe_groups:
            timeframe_groups[key].sort()

        total = sum(len(timeframe_groups[tf]) for tf in timeframe_groups)

        return {
            "total_models": total,
            "by_timeframe": {
                "M15": {
                    "count": len(timeframe_groups["M15"]),
                    "label": "15 Minuten (M15)",
                    "symbols": timeframe_groups["M15"]
                },
                "H1": {
                    "count": len(timeframe_groups["H1"]),
                    "label": "Stündlich (H1)",
                    "symbols": timeframe_groups["H1"]
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

    Uses the Data Service API to fetch historical prices from EasyInsight.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service

        # Get pending count before evaluation
        pending_before = sum(len(v) for v in model_improvement_service.pending_feedback.values())

        # Run evaluation using API-based method
        evaluated = await model_improvement_service.evaluate_pending_predictions_via_api()

        # Get counts after evaluation
        pending_after = sum(len(v) for v in model_improvement_service.pending_feedback.values())
        total_evaluated = sum(evaluated.values())

        return {
            "success": True,
            "evaluated_count": total_evaluated,
            "pending_before": pending_before,
            "pending_after": pending_after,
            "by_symbol": evaluated,
            "message": f"Evaluated {total_evaluated} predictions" if total_evaluated > 0 else "No predictions ready for evaluation"
        }
    except Exception as e:
        logger.error(f"Failed to evaluate predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            # Set in_progress flag
            model_improvement_service._retrain_in_progress = True
            model_improvement_service._last_auto_retrain_symbols = symbols
            try:
                logger.info(f"Starting retraining for poor performers: {symbols}")
                for symbol in symbols:
                    try:
                        logger.info(f"Retraining model for {symbol}...")
                        result = await nhits_training_service.train_symbol(symbol, force=True)
                        if result.success:
                            logger.info(f"Successfully retrained {symbol}")
                            model_improvement_service._total_auto_retrains += 1
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
            finally:
                # Always reset in_progress flag
                model_improvement_service._retrain_in_progress = False
                model_improvement_service._last_retrain_time = datetime.utcnow()

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


# ==================== Latest Forecasts ====================

@training_router.get("/forecast/latest-per-model")
async def get_latest_forecasts_per_model():
    """
    Get the latest forecast for each model (symbol/timeframe combination).

    Returns a dictionary with model keys (e.g., "BTCUSD_H1") and their
    latest forecast timestamp and trend direction.
    """
    try:
        import math
        from ..services.model_improvement_service import model_improvement_service

        latest_forecasts = {}  # model_key -> {data + _ts for comparison}

        # Check pending feedback (not yet evaluated)
        for symbol, feedbacks in model_improvement_service.pending_feedback.items():
            for fb in feedbacks:
                model_key = f"{fb.symbol}_{fb.timeframe}" if hasattr(fb, 'timeframe') and fb.timeframe else fb.symbol
                timestamp = fb.timestamp

                # Compare using datetime objects
                existing_ts = latest_forecasts.get(model_key, {}).get("_ts")
                if existing_ts is None or timestamp > existing_ts:
                    # Determine trend direction
                    trend = None
                    if fb.predicted_price and fb.current_price:
                        trend = "up" if fb.predicted_price > fb.current_price else "down"

                    # Sanitize float values (handle inf/nan)
                    current_price = fb.current_price if fb.current_price and math.isfinite(fb.current_price) else None
                    predicted_price = fb.predicted_price if fb.predicted_price and math.isfinite(fb.predicted_price) else None

                    latest_forecasts[model_key] = {
                        "_ts": timestamp,  # Keep datetime for comparison
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        "symbol": fb.symbol,
                        "timeframe": fb.timeframe if hasattr(fb, 'timeframe') else None,
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "trend": trend,
                        "horizon": fb.horizon,
                        "evaluated": False
                    }

        # Also check evaluated feedback
        for symbol, feedbacks in model_improvement_service.evaluated_feedback.items():
            for fb in feedbacks:
                model_key = f"{fb.symbol}_{fb.timeframe}" if hasattr(fb, 'timeframe') and fb.timeframe else fb.symbol
                timestamp = fb.timestamp

                existing_ts = latest_forecasts.get(model_key, {}).get("_ts")
                if existing_ts is None or timestamp > existing_ts:
                    trend = None
                    if fb.predicted_price and fb.current_price:
                        trend = "up" if fb.predicted_price > fb.current_price else "down"

                    # Sanitize float values (handle inf/nan)
                    current_price = fb.current_price if fb.current_price and math.isfinite(fb.current_price) else None
                    predicted_price = fb.predicted_price if fb.predicted_price and math.isfinite(fb.predicted_price) else None
                    error_pct = fb.error_pct if fb.error_pct and math.isfinite(fb.error_pct) else None

                    latest_forecasts[model_key] = {
                        "_ts": timestamp,
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        "symbol": fb.symbol,
                        "timeframe": fb.timeframe if hasattr(fb, 'timeframe') else None,
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "trend": trend,
                        "horizon": fb.horizon,
                        "evaluated": True,
                        "direction_correct": fb.direction_correct,
                        "error_pct": error_pct
                    }

        # Remove internal _ts field before returning
        for key in latest_forecasts:
            if "_ts" in latest_forecasts[key]:
                del latest_forecasts[key]["_ts"]

        return {
            "count": len(latest_forecasts),
            "forecasts": latest_forecasts
        }
    except Exception as e:
        logger.error(f"Failed to get latest forecasts per model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Auto-Evaluation & Auto-Retrain ====================

@training_router.get("/forecast/auto-status")
async def get_auto_evaluation_status():
    """
    Get status of automatic evaluation and retraining system.

    Returns:
    - Auto-evaluation status (enabled, running, last run, total evaluations)
    - Auto-retrain status (enabled, in progress, last run, total retrains)
    - Pending and evaluated prediction counts
    - Symbols needing retraining
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        return model_improvement_service.get_auto_status()
    except Exception as e:
        logger.error(f"Failed to get auto-evaluation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/auto-evaluation/toggle")
async def toggle_auto_evaluation(enabled: bool = True):
    """
    Enable or disable automatic evaluation.

    When enabled, predictions are automatically evaluated against actual prices
    when their horizon expires.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        model_improvement_service.set_auto_evaluation_enabled(enabled)
        return {
            "success": True,
            "auto_evaluation_enabled": enabled,
            "message": f"Auto-evaluation {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        logger.error(f"Failed to toggle auto-evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/auto-retrain/toggle")
async def toggle_auto_retrain(enabled: bool = True):
    """
    Enable or disable automatic retraining.

    When enabled, models are automatically retrained when their performance
    falls below configured thresholds.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        model_improvement_service.set_auto_retrain_enabled(enabled)
        return {
            "success": True,
            "auto_retrain_enabled": enabled,
            "message": f"Auto-retrain {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        logger.error(f"Failed to toggle auto-retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/auto-retrain/cooldown")
async def set_retrain_cooldown(hours: int = 4):
    """
    Set the cooldown period between automatic retrains for the same symbol.

    This prevents excessive retraining of the same model.

    Args:
        hours: Minimum hours between retrains (default: 4, minimum: 1)
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        model_improvement_service.set_retrain_cooldown_hours(hours)
        return {
            "success": True,
            "cooldown_hours": model_improvement_service._retrain_cooldown_hours,
            "message": f"Retrain cooldown set to {hours} hours"
        }
    except Exception as e:
        logger.error(f"Failed to set retrain cooldown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/auto-evaluation/start")
async def start_auto_evaluation(interval_seconds: int = 300):
    """
    Start the automatic evaluation service.

    This starts a background task that periodically evaluates pending predictions
    and triggers auto-retrain when performance thresholds are exceeded.

    Args:
        interval_seconds: Interval between evaluation cycles (default: 300 = 5 minutes)
    """
    try:
        from ..services.model_improvement_service import model_improvement_service

        if model_improvement_service._running:
            return {
                "success": True,
                "message": "Auto-evaluation already running",
                "already_running": True
            }

        await model_improvement_service.start(interval_seconds=interval_seconds)
        return {
            "success": True,
            "message": f"Auto-evaluation started with {interval_seconds}s interval",
            "interval_seconds": interval_seconds
        }
    except Exception as e:
        logger.error(f"Failed to start auto-evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/auto-evaluation/stop")
async def stop_auto_evaluation():
    """
    Stop the automatic evaluation service.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service

        if not model_improvement_service._running:
            return {
                "success": True,
                "message": "Auto-evaluation not running",
                "already_stopped": True
            }

        await model_improvement_service.stop()
        return {
            "success": True,
            "message": "Auto-evaluation stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop auto-evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/manual-retrain")
async def trigger_manual_retrain(symbols: Optional[List[str]] = None):
    """
    Manually trigger retraining for specified symbols.

    If no symbols are specified, retrains all symbols that need it based on
    performance metrics.

    Args:
        symbols: Optional list of symbols to retrain. If None, uses performance-based selection.
    """
    try:
        from ..services.model_improvement_service import model_improvement_service
        result = await model_improvement_service.trigger_manual_retrain(symbols=symbols)
        return result
    except Exception as e:
        logger.error(f"Failed to trigger manual retrain: {e}")
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

        # Fetch time series data using nhits_training_service (works in NHITS container)
        from ..services.nhits_training_service import nhits_training_service
        time_series = await nhits_training_service.get_training_data(
            symbol=symbol,
            days=days_needed,
            timeframe=timeframe,
            use_cache=True
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


# ==================== Training Data Cache Endpoints (Proxy to Data-Service) ====================
# Note: Training data cache is now managed by Data-Service.
# These endpoints proxy to the Data-Service for backwards compatibility.

@forecast_router.get("/forecast/training/cache/stats")
async def get_training_cache_stats():
    """
    Get statistics about the training data cache (from Data-Service).
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://trading-data:3001/api/v1/training-data/cache/stats")
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get cache stats from Data-Service: {e}")
        return {"error": str(e), "total_entries": 0}


@forecast_router.get("/forecast/training/cache/symbols")
async def get_cached_symbols():
    """
    Get list of symbols currently cached (from Data-Service).
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://trading-data:3001/api/v1/training-data/cache/symbols")
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get cached symbols from Data-Service: {e}")
        return {"error": str(e)}


@forecast_router.delete("/forecast/training/cache")
async def clear_training_cache():
    """
    Clear all training data cache (via Data-Service).
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete("http://trading-data:3001/api/v1/training-data/cache")
            return response.json()
    except Exception as e:
        logger.error(f"Failed to clear cache via Data-Service: {e}")
        return {"error": str(e), "removed": 0}


@forecast_router.delete("/forecast/training/cache/expired")
async def cleanup_expired_cache():
    """
    Remove only expired cache entries (via Data-Service).
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete("http://trading-data:3001/api/v1/training-data/cache/expired")
            return response.json()
    except Exception as e:
        logger.error(f"Failed to cleanup expired cache via Data-Service: {e}")
        return {"error": str(e), "removed": 0}


# ==================== Auto-Forecast for Favorites ====================
# IMPORTANT: These routes use training_router because they must be registered
# BEFORE the catch-all /forecast/{symbol} route in forecast_router

@training_router.get("/forecast/favorites")
async def get_favorite_models():
    """
    Get all NHITS models for favorite symbols.

    Returns models filtered to only include those for symbols
    marked as favorites in the Data Service.
    """
    try:
        forecast_service = get_forecast_service()
        models = forecast_service.list_models()

        # Fetch favorites from Data Service
        favorites_data = await get_favorites_from_data_service()
        favorite_symbols = {s for s, info in favorites_data.items() if info.get("is_favorite")}

        # Filter to favorite models only
        favorite_models = []
        for model in models:
            if not model.model_exists:
                continue

            # Extract base symbol
            base_symbol = model.symbol
            for suffix in ["_M15", "_H1", "_D1"]:
                if base_symbol.endswith(suffix):
                    base_symbol = base_symbol[:-len(suffix)]
                    break

            if base_symbol in favorite_symbols:
                symbol_info = favorites_data.get(base_symbol, {})
                favorite_models.append({
                    "symbol": model.symbol,
                    "base_symbol": base_symbol,
                    "timeframe": model.symbol.split("_")[-1] if "_" in model.symbol else "H1",
                    "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                    "category": symbol_info.get("category"),
                    "display_name": symbol_info.get("display_name"),
                })

        return {
            "count": len(favorite_models),
            "models": favorite_models,
        }
    except Exception as e:
        logger.error(f"Failed to get favorite models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@training_router.post("/forecast/favorites/run")
async def run_forecasts_for_favorites(
    timeframe: Optional[str] = None,
    background: bool = False,
    background_tasks: BackgroundTasks = None,
):
    """
    Generate forecasts for all favorite symbols.

    Parameters:
    - timeframe: Specific timeframe (M15, H1, D1) or None for all
    - background: Run in background (returns immediately)

    This generates fresh forecasts for all symbols marked as
    favorites that have trained NHITS models.
    """
    from ..services.auto_forecast_service import auto_forecast_service

    if background and background_tasks:
        background_tasks.add_task(
            auto_forecast_service.run_forecasts_for_favorites,
            timeframe
        )
        return {
            "status": "started",
            "message": f"Forecast generation started in background for timeframe: {timeframe or 'all'}",
        }

    results = await auto_forecast_service.run_forecasts_for_favorites(timeframe)
    return results


@training_router.get("/forecast/favorites/status")
async def get_auto_forecast_status():
    """
    Get status of the auto-forecast service.

    Returns information about the background forecast service
    including last run time and latest forecast results.
    """
    from ..services.auto_forecast_service import auto_forecast_service
    return auto_forecast_service.get_status()


@training_router.get("/forecast/favorites/latest")
async def get_latest_favorite_forecasts():
    """
    Get the latest forecast results for favorite symbols.

    Returns the most recent forecast data for each favorite symbol
    that has been generated by the auto-forecast service.
    """
    from ..services.auto_forecast_service import auto_forecast_service
    forecasts = auto_forecast_service.get_latest_forecasts()
    return {
        "count": len(forecasts),
        "forecasts": forecasts,
    }


@training_router.post("/forecast/favorites/start")
async def start_auto_forecast_service(timeframe: str = "H1"):
    """
    Start the background auto-forecast service.

    Parameters:
    - timeframe: Timeframe for automatic forecasts (M15, H1, D1)

    This starts a background loop that automatically generates
    forecasts for favorite symbols at regular intervals based
    on the timeframe (15min for M15, 1h for H1, 24h for D1).
    """
    from ..services.auto_forecast_service import auto_forecast_service

    if timeframe.upper() not in ["M15", "H1", "D1"]:
        raise HTTPException(status_code=400, detail="Invalid timeframe. Use M15, H1, or D1.")

    await auto_forecast_service.start(timeframe.upper())
    return {
        "status": "started",
        "timeframe": timeframe.upper(),
        "message": f"Auto-forecast service started for {timeframe.upper()}",
    }


@training_router.post("/forecast/favorites/stop")
async def stop_auto_forecast_service():
    """
    Stop the background auto-forecast service.
    """
    from ..services.auto_forecast_service import auto_forecast_service
    await auto_forecast_service.stop()
    return {
        "status": "stopped",
        "message": "Auto-forecast service stopped",
    }


# ==================== Auto-Forecast: Favorites (Timeframe-based) ====================

@training_router.post("/forecast/auto/favorites/start")
async def start_favorites_auto_forecast(
    timeframes: Optional[str] = None
):
    """
    Start automatic forecasting for favorite symbols (timeframe-based).

    Parameters:
    - timeframes: Comma-separated list of timeframes (M15, H1, D1).
                  If not provided, all timeframes are enabled.

    **Intervals:**
    - M15: Every 15 minutes
    - H1: Every hour
    - D1: Once daily

    The service automatically generates forecasts for all favorite symbols
    that have trained models for the specified timeframes.
    """
    from ..services.auto_forecast_service import auto_forecast_service

    tf_list = None
    if timeframes:
        tf_list = [tf.strip().upper() for tf in timeframes.split(",")]
        invalid = [tf for tf in tf_list if tf not in ["M15", "H1", "D1"]]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframes: {invalid}. Use M15, H1, or D1."
            )

    await auto_forecast_service.start_favorites_auto_forecast(tf_list)

    return {
        "status": "started",
        "mode": "favorites",
        "timeframes": tf_list or ["M15", "H1", "D1"],
        "message": f"Favorites auto-forecast started for: {tf_list or ['M15', 'H1', 'D1']}",
    }


@training_router.post("/forecast/auto/favorites/stop")
async def stop_favorites_auto_forecast(
    timeframes: Optional[str] = None
):
    """
    Stop automatic forecasting for favorite symbols.

    Parameters:
    - timeframes: Comma-separated list of timeframes to stop.
                  If not provided, all timeframes are stopped.
    """
    from ..services.auto_forecast_service import auto_forecast_service

    tf_list = None
    if timeframes:
        tf_list = [tf.strip().upper() for tf in timeframes.split(",")]

    await auto_forecast_service.stop_favorites_auto_forecast(tf_list)

    return {
        "status": "stopped",
        "timeframes": tf_list or "all",
        "message": f"Favorites auto-forecast stopped for: {tf_list or 'all timeframes'}",
    }


# ==================== Auto-Forecast: Daily (Non-Favorites) ====================

@training_router.post("/forecast/auto/daily/start")
async def start_daily_auto_forecast(
    scheduled_time: str = "05:00",
    timezone: str = Query(default=None, description="Timezone (default: from settings.display_timezone)")
):
    """
    Start daily automatic forecasting for non-favorite symbols.

    Parameters:
    - scheduled_time: Time in HH:MM format (default: 05:00)
    - timezone: Timezone string (default: from settings.display_timezone)

    The service runs once daily at the specified time and generates
    H1 forecasts for all non-favorite symbols that have trained models.
    """
    from ..services.auto_forecast_service import auto_forecast_service
    import pytz

    # Use default timezone from settings if not provided
    effective_timezone = timezone if timezone else settings.display_timezone

    # Validate time format
    try:
        parts = scheduled_time.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Invalid time range")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid time format. Use HH:MM (e.g., 05:00, 14:30)"
        )

    # Validate timezone
    try:
        pytz.timezone(effective_timezone)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timezone: {effective_timezone}. Use IANA timezone format (e.g., Europe/Zurich)"
        )

    await auto_forecast_service.start_daily_auto_forecast(scheduled_time, effective_timezone)

    return {
        "status": "started",
        "mode": "daily",
        "scheduled_time": scheduled_time,
        "timezone": effective_timezone,
        "message": f"Daily auto-forecast started (scheduled at {scheduled_time} {effective_timezone})",
    }


@training_router.post("/forecast/auto/daily/stop")
async def stop_daily_auto_forecast():
    """
    Stop daily automatic forecasting for non-favorite symbols.
    """
    from ..services.auto_forecast_service import auto_forecast_service
    await auto_forecast_service.stop_daily_auto_forecast()
    return {
        "status": "stopped",
        "mode": "daily",
        "message": "Daily auto-forecast stopped",
    }


@training_router.post("/forecast/auto/daily/schedule")
async def update_daily_schedule(
    scheduled_time: str,
    timezone: Optional[str] = None
):
    """
    Update the daily forecast schedule without restarting.

    Parameters:
    - scheduled_time: Time in HH:MM format
    - timezone: Optional timezone string
    """
    from ..services.auto_forecast_service import auto_forecast_service

    # Validate time format
    try:
        parts = scheduled_time.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Invalid time range")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid time format. Use HH:MM (e.g., 05:00, 14:30)"
        )

    if timezone:
        try:
            import pytz
            pytz.timezone(timezone)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timezone: {timezone}"
            )

    auto_forecast_service.set_daily_schedule(scheduled_time, timezone)

    return {
        "status": "updated",
        "scheduled_time": scheduled_time,
        "timezone": timezone or auto_forecast_service._daily_timezone,
        "message": f"Schedule updated to {scheduled_time}",
    }


@training_router.post("/forecast/auto/daily/run-now")
async def run_daily_forecast_now():
    """
    Manually trigger the daily forecast for non-favorite symbols.

    Runs immediately instead of waiting for the scheduled time.
    """
    from ..services.auto_forecast_service import auto_forecast_service

    result = await auto_forecast_service.run_forecasts_for_non_favorites()

    return {
        "status": "completed",
        "result": result,
    }


# ==================== Auto-Forecast: Combined Status ====================

@training_router.get("/forecast/auto/status")
async def get_auto_forecast_full_status():
    """
    Get comprehensive status of all auto-forecast services.

    Returns status for both:
    - Favorites auto-forecast (timeframe-based)
    - Daily auto-forecast (non-favorites)
    """
    from ..services.auto_forecast_service import auto_forecast_service
    return auto_forecast_service.get_status()


# ==================== Symbol Management Endpoints ====================

@symbol_router.get("/symbols", response_model=list[str])
async def get_symbol_names(
    category: Optional[SymbolCategory] = None,
    status: Optional[SymbolStatus] = None,
    with_data_only: bool = False,
):
    """
    Get list of symbol names.

    Returns a simple list of symbol names for quick lookups.
    Use /managed-symbols for full symbol details.
    """
    try:
        symbols = await symbol_service.get_all_symbols(
            category=category,
            status=status,
            with_data_only=with_data_only,
        )
        return [s.symbol for s in symbols]
    except Exception as e:
        logger.error(f"Failed to get symbol names: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

        # Get already imported symbols (normalized to display format without slashes)
        existing_symbols = {s.symbol.upper().replace("/", "") for s in await symbol_service.get_all_symbols()}

        # Prepare result with import status
        result = []
        for s in symbols_data[:500]:  # Limit to 500 symbols for performance
            symbol_id = s.get("symbol", "")
            # Normalize symbol for comparison (remove slashes)
            symbol_normalized = symbol_id.upper().replace("/", "")
            result.append({
                "symbol": symbol_id,
                "name": s.get("name") or s.get("currency_base", ""),
                "exchange": s.get("exchange", ""),
                "currency": s.get("currency") or s.get("currency_quote", ""),
                "country": s.get("country", ""),
                "type": s.get("type", category),
                "already_imported": symbol_normalized in existing_symbols
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


@symbol_router.post("/managed-symbols/migrate-api-symbols")
async def migrate_api_symbols():
    """
    Migrate all existing symbols to have proper TwelveData and EasyInsight symbol formats.

    This will auto-generate the API-specific symbol formats for all symbols
    that don't already have them set:
    - TwelveData: Uses slash format (e.g., BTC/USD, EUR/USD)
    - EasyInsight: Uses concatenated format (e.g., BTCUSD, EURUSD)

    Returns:
        Migration result with counts of updated, skipped, and errored symbols.
    """
    try:
        result = await symbol_service.migrate_api_symbols()
        logger.info(f"API symbols migration completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to migrate API symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.post("/managed-symbols/cleanup-aliases")
async def cleanup_redundant_aliases():
    """
    Remove aliases that are identical to TwelveData or EasyInsight symbols.

    Returns:
        Cleanup result with counts and list of removed aliases.
    """
    try:
        result = await symbol_service.cleanup_redundant_aliases()
        logger.info(f"Alias cleanup completed: {result['cleaned']} symbols cleaned")
        return result
    except Exception as e:
        logger.error(f"Failed to cleanup aliases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@symbol_router.post("/managed-symbols/migrate-json-to-db")
async def migrate_json_to_db():
    """
    Migrate all symbols from JSON file to TimescaleDB.

    This will read the existing symbols from data/symbols/symbols.json
    and insert/update them in the TimescaleDB symbols table.

    Returns:
        Migration result with counts and any errors.
    """
    try:
        result = await symbol_service.migrate_from_json()
        logger.info(f"JSON to DB migration completed: {result['migrated']} symbols migrated")
        return result
    except Exception as e:
        logger.error(f"Failed to migrate JSON to DB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Data Statistics ====================
# NOTE: This route MUST be defined BEFORE the generic {symbol_id:path} routes

@symbol_router.get("/managed-symbols/data-stats/{symbol:path}")
async def get_symbol_data_stats(symbol: str):
    """
    Get detailed data statistics for a symbol including data range and counts per timeframe.

    Returns:
        - First and last data timestamp
        - Total record count
        - Data points per timeframe (M15, H1, D1)
        - Data gaps analysis
        - Data coverage percentage
    """
    import httpx

    result = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "has_data": False,
        "first_timestamp": None,
        "last_timestamp": None,
        "total_records": 0,
        "timeframes": {
            "M15": {"count": 0, "first": None, "last": None, "coverage_pct": 0},
            "H1": {"count": 0, "first": None, "last": None, "coverage_pct": 0},
            "D1": {"count": 0, "first": None, "last": None, "coverage_pct": 0},
        },
        "data_quality": {
            "gaps_detected": 0,
            "avg_gap_hours": 0,
            "max_gap_hours": 0,
            "completeness_pct": 0,
        },
        "sample_data": [],
        "errors": []
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get symbol info with count
            response = await client.get(f"{settings.easyinsight_api_url}/symbols")
            if response.status_code == 200:
                symbols_data = response.json()
                symbol_info = next((s for s in symbols_data if s.get("symbol") == symbol.upper()), None)
                if symbol_info:
                    result["total_records"] = symbol_info.get("count", 0)
                    result["first_timestamp"] = symbol_info.get("earliest")
                    result["last_timestamp"] = symbol_info.get("latest")
                    result["has_data"] = result["total_records"] > 0

            # Get sample data with timestamps to analyze gaps
            response = await client.get(
                f"{settings.easyinsight_api_url}/symbol-data-full/{symbol}",
                params={"limit": 1000}  # Get last 1000 records for analysis
            )

            if response.status_code == 200:
                response_data = response.json()
                data_list = []
                if isinstance(response_data, dict) and "data" in response_data:
                    data_list = response_data.get("data", [])
                elif isinstance(response_data, list):
                    data_list = response_data

                if data_list:
                    result["has_data"] = True

                    # Analyze timestamps
                    timestamps = []
                    for row in data_list:
                        ts = row.get("snapshot_time")
                        if ts:
                            try:
                                if isinstance(ts, str):
                                    # Parse ISO format
                                    ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+01:00", "+00:00").replace("+00:00", ""))
                                else:
                                    ts_dt = ts
                                timestamps.append(ts_dt)
                            except:
                                pass

                    if timestamps:
                        timestamps.sort()
                        result["first_timestamp"] = timestamps[0].isoformat() if not result["first_timestamp"] else result["first_timestamp"]
                        result["last_timestamp"] = timestamps[-1].isoformat() if not result["last_timestamp"] else result["last_timestamp"]

                        # Analyze gaps (looking for gaps > 2 hours)
                        gaps = []
                        for i in range(1, len(timestamps)):
                            gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                            if gap > 2:  # More than 2 hours gap
                                gaps.append(gap)

                        if gaps:
                            result["data_quality"]["gaps_detected"] = len(gaps)
                            result["data_quality"]["avg_gap_hours"] = round(sum(gaps) / len(gaps), 2)
                            result["data_quality"]["max_gap_hours"] = round(max(gaps), 2)

                        # Calculate completeness (expected vs actual hourly data points)
                        if len(timestamps) >= 2:
                            total_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
                            expected_points = max(1, int(total_hours))
                            result["data_quality"]["completeness_pct"] = round(
                                min(100, (len(timestamps) / expected_points) * 100), 1
                            )

                    # Count data by timeframe (check which OHLC fields are populated)
                    m15_count = sum(1 for row in data_list if row.get("m15_close") is not None)
                    h1_count = sum(1 for row in data_list if row.get("h1_close") is not None)
                    d1_count = sum(1 for row in data_list if row.get("d1_close") is not None)

                    result["timeframes"]["M15"]["count"] = m15_count
                    result["timeframes"]["H1"]["count"] = h1_count
                    result["timeframes"]["D1"]["count"] = d1_count

                    # Calculate coverage percentages based on data range
                    if timestamps and len(timestamps) >= 2:
                        range_hours = max(1, (timestamps[-1] - timestamps[0]).total_seconds() / 3600)
                        result["timeframes"]["M15"]["coverage_pct"] = round(min(100, (m15_count / (range_hours * 4)) * 100), 1)
                        result["timeframes"]["H1"]["coverage_pct"] = round(min(100, (h1_count / range_hours) * 100), 1)
                        result["timeframes"]["D1"]["coverage_pct"] = round(min(100, (d1_count / (range_hours / 24)) * 100), 1)

    except Exception as e:
        logger.error(f"Failed to get data stats for {symbol}: {e}")
        result["errors"].append(str(e))

    # Fetch sample data from TwelveData for chart visualization
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            sample_data = {"source": "TwelveData", "M15": [], "H1": [], "D1": []}

            # Convert symbol format for TwelveData (AUDUSD -> AUD/USD for Forex)
            td_symbol = symbol
            forex_pairs = ["AUD", "EUR", "GBP", "USD", "CHF", "JPY", "CAD", "NZD"]
            if len(symbol) == 6 and symbol[:3] in forex_pairs and symbol[3:] in forex_pairs:
                td_symbol = f"{symbol[:3]}/{symbol[3:]}"

            # Fetch OHLCV for each timeframe from TwelveData
            td_base_url = "https://api.twelvedata.com"
            for tf, td_interval, limit in [("M15", "15min", 50), ("H1", "1h", 50), ("D1", "1day", 30)]:
                try:
                    td_response = await client.get(
                        f"{td_base_url}/time_series",
                        params={
                            "symbol": td_symbol,
                            "interval": td_interval,
                            "outputsize": limit,
                            "apikey": settings.twelvedata_api_key,
                        }
                    )
                    if td_response.status_code == 200:
                        td_data = td_response.json()
                        if "values" in td_data:
                            from zoneinfo import ZoneInfo
                            est_tz = ZoneInfo("America/New_York")
                            utc_tz = ZoneInfo("UTC")

                            candles = []
                            for v in td_data["values"]:
                                if not v.get("close"):
                                    continue
                                # Convert TwelveData EST timestamp to UTC ISO format
                                dt_str = v.get("datetime", "")
                                try:
                                    dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                                    dt_est = dt_naive.replace(tzinfo=est_tz)
                                    dt_utc = dt_est.astimezone(utc_tz)
                                    iso_datetime = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                                except:
                                    iso_datetime = dt_str

                                candles.append({
                                    "datetime": iso_datetime,
                                    "open": float(v.get("open")) if v.get("open") else None,
                                    "high": float(v.get("high")) if v.get("high") else None,
                                    "low": float(v.get("low")) if v.get("low") else None,
                                    "close": float(v.get("close")) if v.get("close") else None,
                                })
                            sample_data[tf] = candles
                except Exception as e:
                    logger.warning(f"Failed to fetch TwelveData {tf} for {symbol}: {e}")

            result["sample_data"] = sample_data
    except Exception as e:
        logger.warning(f"Failed to fetch TwelveData sample data for {symbol}: {e}")
        result["sample_data"] = {"source": "TwelveData", "M15": [], "H1": [], "D1": []}

    return result


# ==================== Live Data ====================
# NOTE: This route MUST be defined BEFORE the generic {symbol_id:path} routes
# to prevent "live-data" being interpreted as a symbol_id

@symbol_router.get("/managed-symbols/live-data/{symbol:path}")
async def get_symbol_live_data(symbol: str):
    """
    Get live market data for a symbol from EasyInsight and TwelveData APIs.

    Returns the latest values and indicators from both data sources for comparison.

    Args:
        symbol: The symbol to get data for (e.g., 'EURUSD', 'BTCUSD')
    """
    import httpx

    now_utc = datetime.now(timezone.utc)
    result = {
        "symbol": symbol,
        "timestamp_utc": format_utc_iso(now_utc),
        "timestamp_display": format_for_display(now_utc),
        "timezone_info": get_timezone_info(),
        "easyinsight": None,
        "twelvedata": None,
        "yfinance": None,
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
                    ei_snapshot_time = data.get("snapshot_time")
                    # Determine EasyInsight symbol used
                    ei_symbol = managed_symbol.easyinsight_symbol if managed_symbol and managed_symbol.easyinsight_symbol else symbol
                    result["easyinsight"] = {
                        "source": "EasyInsight API",
                        "symbol_used": ei_symbol,
                        "snapshot_time_utc": format_utc_iso(ei_snapshot_time),
                        "snapshot_time_display": format_for_display(ei_snapshot_time),
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

    # Fetch TwelveData quote - use twelvedata_symbol from managed symbol
    td_symbol = symbol
    if managed_symbol and managed_symbol.twelvedata_symbol:
        td_symbol = managed_symbol.twelvedata_symbol
    else:
        # Fallback: try aliases or generate from symbol
        for alias in aliases:
            if "/" in alias:
                td_symbol = alias
                break
        else:
            # Generate from symbol pattern (AUDUSD -> AUD/USD)
            td_symbol = symbol_service._generate_twelvedata_symbol(
                symbol, managed_symbol.category if managed_symbol else SymbolCategory.FOREX
            ) or symbol

    try:
        import asyncio

        # Fetch quote and M1 time series in parallel
        quote_task = twelvedata_service.get_quote(symbol=td_symbol)
        m1_task = twelvedata_service.get_time_series(symbol=td_symbol, interval="1min", outputsize=2)

        quote, m1_data = await asyncio.gather(quote_task, m1_task, return_exceptions=True)

        # Handle exceptions
        if isinstance(quote, Exception):
            quote = None
        if isinstance(m1_data, Exception):
            m1_data = None

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

            # Use M1 data for real-time OHLC if available, fallback to quote (daily)
            # IMPORTANT: TwelveData returns datetimes in Exchange timezone (EST for Forex),
            # NOT in UTC. Use Unix timestamps (last_quote_at) when available for accuracy.
            td_datetime = None
            # Prefer Unix timestamp from quote (last_quote_at) - this is unambiguous
            if quote.get("last_quote_at"):
                from datetime import datetime as dt
                td_datetime = dt.fromtimestamp(quote.get("last_quote_at"), tz=timezone.utc)

            m1_ohlc = None
            m1_datetime = None

            if m1_data and "error" not in m1_data and m1_data.get("values"):
                m1_values = m1_data["values"]
                if m1_values and len(m1_values) > 0:
                    latest_m1 = m1_values[0]
                    # M1 datetime is in Exchange timezone - for now, use quote timestamp
                    # as it's more reliable (Unix timestamp)
                    m1_ohlc = {
                        "open": latest_m1.get("open"),
                        "high": latest_m1.get("high"),
                        "low": latest_m1.get("low"),
                        "close": latest_m1.get("close"),
                    }

            # Use quote timestamp (derived from Unix timestamp) as it's timezone-unambiguous
            display_datetime = td_datetime

            result["twelvedata"] = {
                "source": "Twelve Data API",
                "symbol_used": td_symbol,
                "name": quote.get("name"),
                "exchange": quote.get("exchange"),
                "currency": quote.get("currency"),
                "datetime_utc": format_utc_iso(display_datetime),
                "datetime_display": format_for_display(display_datetime),
                "bid_ask": {
                    "bid": td_bid,
                    "ask": td_ask,
                    "spread": td_spread,
                    "spread_pct": td_spread_pct,
                },
                # Use M1 OHLC for real-time price, daily OHLC as secondary
                "price": m1_ohlc or {
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "close": quote.get("close"),
                },
                "price_daily": {
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
                "indicators": {},
            }

            # Fetch TwelveData technical indicators (parallel requests) - use 1min interval for real-time data
            td_interval = "1min"  # Minute data for real-time comparison
            indicator_tasks = {
                "rsi": twelvedata_service.get_rsi(td_symbol, interval=td_interval, outputsize=1),
                "macd": twelvedata_service.get_macd(td_symbol, interval=td_interval, outputsize=1),
                "bbands": twelvedata_service.get_bollinger_bands(td_symbol, interval=td_interval, outputsize=1),
                "stoch": twelvedata_service.get_stochastic(td_symbol, interval=td_interval, outputsize=1),
                "adx": twelvedata_service.get_adx(td_symbol, interval=td_interval, outputsize=1),
                "atr": twelvedata_service.get_atr(td_symbol, interval=td_interval, outputsize=1),
                "ema_10": twelvedata_service.get_ema(td_symbol, interval=td_interval, time_period=10, outputsize=1),
                "ichimoku": twelvedata_service.get_ichimoku(td_symbol, interval=td_interval, outputsize=1),
            }

            # Execute all indicator requests in parallel
            indicator_results = await asyncio.gather(
                *indicator_tasks.values(),
                return_exceptions=True
            )

            # Process results - use the same timestamp as the quote for consistency
            # NOTE: TwelveData indicator datetimes are in Exchange timezone (not UTC),
            # so we use the quote's Unix timestamp which is timezone-unambiguous.
            if display_datetime:
                result["twelvedata"]["indicator_datetime_utc"] = format_utc_iso(display_datetime)
                result["twelvedata"]["indicator_datetime_display"] = format_for_display(display_datetime)

            for name, res in zip(indicator_tasks.keys(), indicator_results):
                if isinstance(res, Exception):
                    continue
                if res and "error" not in res and res.get("values"):
                    values = res["values"]
                    latest = values[0] if isinstance(values, list) and values else values

                    if name == "rsi":
                        result["twelvedata"]["indicators"]["rsi"] = float(latest.get("rsi", 0)) if latest.get("rsi") else None
                    elif name == "macd":
                        result["twelvedata"]["indicators"]["macd"] = {
                            "main": float(latest.get("macd", 0)) if latest.get("macd") else None,
                            "signal": float(latest.get("macd_signal", 0)) if latest.get("macd_signal") else None,
                            "histogram": float(latest.get("macd_hist", 0)) if latest.get("macd_hist") else None,
                        }
                    elif name == "bbands":
                        result["twelvedata"]["indicators"]["bollinger"] = {
                            "upper": float(latest.get("upper_band", 0)) if latest.get("upper_band") else None,
                            "middle": float(latest.get("middle_band", 0)) if latest.get("middle_band") else None,
                            "lower": float(latest.get("lower_band", 0)) if latest.get("lower_band") else None,
                        }
                    elif name == "stoch":
                        result["twelvedata"]["indicators"]["stochastic"] = {
                            "k": float(latest.get("slow_k", 0)) if latest.get("slow_k") else None,
                            "d": float(latest.get("slow_d", 0)) if latest.get("slow_d") else None,
                        }
                    elif name == "adx":
                        result["twelvedata"]["indicators"]["adx"] = float(latest.get("adx", 0)) if latest.get("adx") else None
                    elif name == "atr":
                        result["twelvedata"]["indicators"]["atr"] = float(latest.get("atr", 0)) if latest.get("atr") else None
                    elif name == "ema_10":
                        result["twelvedata"]["indicators"]["ema_10"] = float(latest.get("ema", 0)) if latest.get("ema") else None
                    elif name == "ichimoku":
                        result["twelvedata"]["indicators"]["ichimoku"] = {
                            "tenkan": float(latest.get("tenkan_sen", 0)) if latest.get("tenkan_sen") else None,
                            "kijun": float(latest.get("kijun_sen", 0)) if latest.get("kijun_sen") else None,
                            "senkou_a": float(latest.get("senkou_span_a", 0)) if latest.get("senkou_span_a") else None,
                            "senkou_b": float(latest.get("senkou_span_b", 0)) if latest.get("senkou_span_b") else None,
                            "chikou": float(latest.get("chikou_span", 0)) if latest.get("chikou_span") else None,
                        }

        elif quote and "error" in quote:
            result["errors"].append(f"TwelveData: {quote.get('error')}")
    except Exception as e:
        result["errors"].append(f"TwelveData: {str(e)}")

    # Fetch Yahoo Finance data as additional source
    # Note: yfinance_service is imported conditionally at the top of this file
    try:
        if _yfinance_available and yfinance_service is not None and yfinance_service.is_available():
            # Get daily data for comparison
            yf_data = await yfinance_service.get_time_series(
                symbol=symbol,
                interval="1d",
                outputsize=5  # Last 5 days
            )

            if "error" not in yf_data and yf_data.get("values"):
                values = yf_data["values"]
                latest = values[0] if values else None

                if latest:
                    yf_datetime = latest.get("datetime")
                    result["yfinance"] = {
                        "source": "Yahoo Finance",
                        "symbol_used": yfinance_service._map_symbol(symbol),
                        "datetime_utc": format_utc_iso(yf_datetime),
                        "datetime_display": format_for_display(yf_datetime),
                        "price": {
                            "open": latest.get("open"),
                            "high": latest.get("high"),
                            "low": latest.get("low"),
                            "close": latest.get("close"),
                        },
                        "volume": latest.get("volume"),
                        "data_points": len(values),
                    }

                    # Calculate change from previous day if available
                    if len(values) >= 2:
                        prev = values[1]
                        if prev.get("close") and latest.get("close"):
                            prev_close = float(prev["close"])
                            curr_close = float(latest["close"])
                            change = curr_close - prev_close
                            change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                            result["yfinance"]["change"] = round(change, 5)
                            result["yfinance"]["change_percent"] = round(change_pct, 2)
            elif "error" in yf_data:
                result["errors"].append(f"Yahoo Finance: {yf_data['error']}")
    except Exception as e:
        result["errors"].append(f"Yahoo Finance: {str(e)}")

    return result


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


@symbol_router.get("/managed-symbols/{symbol_id:path}", response_model=ManagedSymbol)
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


@symbol_router.put("/managed-symbols/{symbol_id:path}", response_model=ManagedSymbol)
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


@symbol_router.delete("/managed-symbols/{symbol_id:path}")
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


@symbol_router.post("/managed-symbols/{symbol_id:path}/favorite", response_model=ManagedSymbol)
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


@symbol_router.post("/managed-symbols/{symbol_id:path}/refresh", response_model=ManagedSymbol)
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


@twelvedata_router.get("/twelvedata/quote/{symbol:path}")
async def get_quote(symbol: str, exchange: Optional[str] = None):
    """
    Get real-time quote for a symbol.

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD', 'BTC/USD')
        exchange: Specific exchange (optional)
    """
    quote = await twelvedata_service.get_quote(symbol=symbol, exchange=exchange)
    return quote


@twelvedata_router.get("/twelvedata/price/{symbol:path}")
async def get_price(symbol: str, exchange: Optional[str] = None):
    """
    Get current price for a symbol (lightweight endpoint).

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD')
        exchange: Specific exchange (optional)
    """
    price = await twelvedata_service.get_price(symbol=symbol, exchange=exchange)
    return price


@twelvedata_router.get("/twelvedata/time_series/{symbol:path}")
async def get_time_series(
    symbol: str,
    interval: str = "1day",
    outputsize: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: Optional[str] = None,
    bypass_cache: bool = False,
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
        bypass_cache: Skip cache and fetch fresh data (default: False)
    """
    data = await twelvedata_service.get_time_series(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        bypass_cache=bypass_cache,
    )
    return data


@twelvedata_router.get("/twelvedata/indicator/{symbol:path}/{indicator}")
async def get_technical_indicator(
    symbol: str,
    indicator: str,
    interval: str = "1day",
    outputsize: int = 100,
    time_period: int = 14,
):
    """
    Get technical indicator data for a symbol.

    Supported indicators:
    - Overlap Studies: sma, ema, wma, dema, tema, kama, mama, t3, trima
    - Momentum: rsi, macd, stoch, stochrsi, willr, cci, cmo, roc, mom, ppo, apo, aroon, aroonosc, bop, mfi, dx, adx, adxr, plus_di, minus_di
    - Volatility: bbands, atr, natr, trange
    - Volume: obv, ad, adosc
    - Trend: supertrend, ichimoku, sar
    - Price Transform: avgprice, medprice, typprice, wclprice
    - Pattern: pivot_points_hl

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD', 'BTC/USD')
        indicator: Indicator name (see above for full list)
        interval: Time interval ('1min', '5min', '15min', '30min', '1h', '4h', '1day', '1week')
        outputsize: Number of data points (default: 100, max: 5000)
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


@twelvedata_router.get("/twelvedata/indicators/{symbol:path}")
async def get_multiple_indicators(
    symbol: str,
    indicators: str = "rsi,macd,bbands",
    interval: str = "1day",
    outputsize: int = 100,
):
    """
    Get multiple technical indicators for a symbol in one request.

    Note: Each indicator requires a separate API call (377 credits/min for Grow plan).

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD', 'BTC/USD')
        indicators: Comma-separated list of indicator names (e.g., 'rsi,macd,bbands,stoch')
        interval: Time interval (default: '1day')
        outputsize: Number of data points (default: 100)
    """
    indicator_list = [i.strip().lower() for i in indicators.split(",")]
    data = await twelvedata_service.get_multiple_indicators(
        symbol=symbol,
        indicators=indicator_list,
        interval=interval,
        outputsize=outputsize,
    )
    return data


@twelvedata_router.get("/twelvedata/analysis/{symbol:path}")
async def get_complete_analysis(
    symbol: str,
    interval: str = "1day",
    outputsize: int = 100,
):
    """
    Get complete technical analysis with all major indicators.

    Includes: RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR, CCI, OBV

    Note: This makes 8 API calls with rate limiting, may take time.

    Args:
        symbol: The symbol (e.g., 'AAPL', 'EUR/USD', 'BTC/USD')
        interval: Time interval (default: '1day')
        outputsize: Number of data points (default: 100)
    """
    data = await twelvedata_service.get_complete_analysis(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
    )
    return data


@twelvedata_router.get("/twelvedata/rsi/{symbol:path}")
async def get_rsi(
    symbol: str,
    interval: str = "1day",
    time_period: int = 14,
    outputsize: int = 100,
):
    """Get RSI (Relative Strength Index) indicator."""
    return await twelvedata_service.get_rsi(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/macd/{symbol:path}")
async def get_macd(
    symbol: str,
    interval: str = "1day",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    outputsize: int = 100,
):
    """Get MACD (Moving Average Convergence Divergence) indicator."""
    return await twelvedata_service.get_macd(
        symbol=symbol,
        interval=interval,
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/bbands/{symbol:path}")
async def get_bollinger_bands(
    symbol: str,
    interval: str = "1day",
    time_period: int = 20,
    sd: float = 2.0,
    outputsize: int = 100,
):
    """Get Bollinger Bands indicator."""
    return await twelvedata_service.get_bollinger_bands(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        sd=sd,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/stoch/{symbol:path}")
async def get_stochastic(
    symbol: str,
    interval: str = "1day",
    fast_k_period: int = 14,
    slow_k_period: int = 3,
    slow_d_period: int = 3,
    outputsize: int = 100,
):
    """Get Stochastic Oscillator indicator."""
    return await twelvedata_service.get_stochastic(
        symbol=symbol,
        interval=interval,
        fast_k_period=fast_k_period,
        slow_k_period=slow_k_period,
        slow_d_period=slow_d_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/adx/{symbol:path}")
async def get_adx(
    symbol: str,
    interval: str = "1day",
    time_period: int = 14,
    outputsize: int = 100,
):
    """Get ADX (Average Directional Index) indicator."""
    return await twelvedata_service.get_adx(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/atr/{symbol:path}")
async def get_atr(
    symbol: str,
    interval: str = "1day",
    time_period: int = 14,
    outputsize: int = 100,
):
    """Get ATR (Average True Range) indicator."""
    return await twelvedata_service.get_atr(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/ichimoku/{symbol:path}")
async def get_ichimoku(
    symbol: str,
    interval: str = "1day",
    outputsize: int = 100,
):
    """Get Ichimoku Cloud indicator with all components (Tenkan, Kijun, Senkou A/B, Chikou)."""
    return await twelvedata_service.get_ichimoku(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/supertrend/{symbol:path}")
async def get_supertrend(
    symbol: str,
    interval: str = "1day",
    period: int = 10,
    multiplier: int = 3,
    outputsize: int = 100,
):
    """Get Supertrend indicator."""
    return await twelvedata_service.get_supertrend(
        symbol=symbol,
        interval=interval,
        period=period,
        multiplier=multiplier,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/willr/{symbol:path}")
async def get_williams_r(
    symbol: str,
    interval: str = "1day",
    time_period: int = 14,
    outputsize: int = 100,
):
    """Get Williams %R indicator."""
    return await twelvedata_service.get_williams_r(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/mfi/{symbol:path}")
async def get_mfi(
    symbol: str,
    interval: str = "1day",
    time_period: int = 14,
    outputsize: int = 100,
):
    """Get MFI (Money Flow Index) indicator."""
    return await twelvedata_service.get_mfi(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/obv/{symbol:path}")
async def get_obv(
    symbol: str,
    interval: str = "1day",
    outputsize: int = 100,
):
    """Get OBV (On-Balance Volume) indicator."""
    return await twelvedata_service.get_obv(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/aroon/{symbol:path}")
async def get_aroon(
    symbol: str,
    interval: str = "1day",
    time_period: int = 25,
    outputsize: int = 100,
):
    """Get Aroon indicator (Aroon Up and Aroon Down)."""
    return await twelvedata_service.get_aroon(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/ema/{symbol:path}")
async def get_ema(
    symbol: str,
    interval: str = "1day",
    time_period: int = 20,
    outputsize: int = 100,
):
    """Get EMA (Exponential Moving Average) indicator."""
    return await twelvedata_service.get_ema(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/sma/{symbol:path}")
async def get_sma(
    symbol: str,
    interval: str = "1day",
    time_period: int = 20,
    outputsize: int = 100,
):
    """Get SMA (Simple Moving Average) indicator."""
    return await twelvedata_service.get_sma(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/vwap/{symbol:path}")
async def get_vwap(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 100,
):
    """
    Get VWAP (Volume Weighted Average Price) indicator.

    VWAP is particularly useful for intraday trading as it shows
    the average price weighted by volume. Institutional traders
    often use VWAP as a benchmark for execution quality.

    Note: VWAP resets daily, so it's most useful for intraday intervals.

    Args:
        symbol: Trading symbol (e.g., AAPL, EUR/USD)
        interval: Time interval (default: 1h, recommended: 1min-1h for intraday)
        outputsize: Number of data points to return
    """
    return await twelvedata_service.get_vwap(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/crsi/{symbol:path}")
async def get_connors_rsi(
    symbol: str,
    interval: str = "1day",
    rsi_period: int = 3,
    streak_rsi_period: int = 2,
    pct_rank_period: int = 100,
    outputsize: int = 100,
):
    """
    Get Connors RSI indicator.

    Connors RSI combines three components for better mean-reversion signals:
    1. Short-term RSI (default: 3-period)
    2. Up/Down streak length RSI (default: 2-period)
    3. Percent rank of price change (default: 100-period)

    Better suited for mean-reversion strategies than standard RSI.

    Args:
        symbol: Trading symbol (e.g., AAPL, EUR/USD)
        interval: Time interval
        rsi_period: Period for the RSI calculation (default: 3)
        streak_rsi_period: Period for the streak RSI (default: 2)
        pct_rank_period: Period for percent rank (default: 100)
        outputsize: Number of data points to return
    """
    return await twelvedata_service.get_connors_rsi(
        symbol=symbol,
        interval=interval,
        rsi_period=rsi_period,
        streak_rsi_period=streak_rsi_period,
        pct_rank_period=pct_rank_period,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/linearregslope/{symbol:path}")
async def get_linear_regression_slope(
    symbol: str,
    interval: str = "1day",
    time_period: int = 14,
    series_type: str = "close",
    outputsize: int = 100,
):
    """
    Get Linear Regression Slope indicator.

    Returns the slope of the linear regression line, which quantifies
    trend strength and direction as a numerical value:
    - Positive slope = uptrend
    - Negative slope = downtrend
    - Magnitude indicates trend strength

    Useful as a feature for ML models like NHITS.

    Args:
        symbol: Trading symbol (e.g., AAPL, EUR/USD)
        interval: Time interval
        time_period: Number of periods for regression (default: 14)
        series_type: Price type to use (close, open, high, low)
        outputsize: Number of data points to return
    """
    return await twelvedata_service.get_linear_regression_slope(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        series_type=series_type,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/ht_trendmode/{symbol:path}")
async def get_hilbert_trendmode(
    symbol: str,
    interval: str = "1day",
    series_type: str = "close",
    outputsize: int = 100,
):
    """
    Get Hilbert Transform - Trend vs Cycle Mode indicator.

    Returns a value indicating whether the market is in:
    - Trend mode (value = 1): Trending market, use trend-following strategies
    - Cycle mode (value = 0): Ranging market, use mean-reversion strategies

    Useful for adaptive strategy selection and as a regime filter for ML models.

    Args:
        symbol: Trading symbol (e.g., AAPL, EUR/USD)
        interval: Time interval
        series_type: Price type to use (close, open, high, low)
        outputsize: Number of data points to return
    """
    return await twelvedata_service.get_hilbert_trendmode(
        symbol=symbol,
        interval=interval,
        series_type=series_type,
        outputsize=outputsize,
    )


@twelvedata_router.get("/twelvedata/percent_b/{symbol:path}")
async def get_percent_b(
    symbol: str,
    interval: str = "1day",
    time_period: int = 20,
    sd: float = 2.0,
    ma_type: str = "SMA",
    outputsize: int = 100,
):
    """
    Get Percent B (%B) indicator.

    Shows where price is relative to Bollinger Bands as a normalized value:
    - %B > 1: Price is above upper band (overbought)
    - %B = 1: Price is at upper band
    - %B = 0.5: Price is at middle band (SMA)
    - %B = 0: Price is at lower band
    - %B < 0: Price is below lower band (oversold)

    More suitable for ML models than raw Bollinger Bands as it's normalized (0-1 range).

    Args:
        symbol: Trading symbol (e.g., AAPL, EUR/USD)
        interval: Time interval
        time_period: Period for Bollinger Bands (default: 20)
        sd: Standard deviation multiplier (default: 2.0)
        ma_type: Moving average type (SMA, EMA, etc.)
        outputsize: Number of data points to return
    """
    return await twelvedata_service.get_percent_b(
        symbol=symbol,
        interval=interval,
        time_period=time_period,
        sd=sd,
        ma_type=ma_type,
        outputsize=outputsize,
    )


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


# ==================== Training Data Cache (for NHITS Service) ====================

@twelvedata_router.get("/training-data/{symbol}")
async def get_training_data(
    symbol: str,
    timeframe: str = "H1",
    days: int = 30,
    use_cache: bool = True
):
    """
    Get training data for NHITS model training with caching.

    This endpoint is used by the NHITS service to fetch training data.
    Data sources (in order of priority):
    1. **TwelveData API** (primär) - Konsistente OHLC-Daten für alle Timeframes
    2. **EasyInsight API** (1. Fallback) - Indikatoren und zusätzliche Daten
    3. **Yahoo Finance** (2. Fallback) - Kostenlose historische Daten

    Supported timeframes: M1, M5, M15, M30, H1, H4, D1, W1, MN

    Args:
        symbol: Trading symbol (e.g., BTCUSD, EURUSD)
        timeframe: Timeframe for data (M1, M5, M15, M30, H1, H4, D1, W1, MN)
        days: Number of days of data to fetch
        use_cache: Whether to use cached data if available

    Returns:
        Training data with OHLCV data
    """
    from ..services.training_data_cache_service import training_data_cache
    import httpx
    from datetime import datetime

    # Normalize timeframe using central configuration
    try:
        tf_enum = normalize_timeframe(timeframe)
        tf = tf_enum.value  # Standard format: H1, D1, etc.
    except ValueError:
        # Fallback for invalid timeframes
        tf = timeframe.upper()
        tf_enum = Timeframe.H1

    # Calculate limit using central configuration
    limit = calculate_limit_for_days(tf_enum, days, max_limit=5000)

    # Try to get data from cache first (TwelveData cache)
    rows = None
    from_cache = False
    if use_cache:
        cached_data = training_data_cache.get_cached_data(symbol, tf, "twelvedata")
        if cached_data:
            rows = cached_data
            from_cache = True
            logger.info(f"Using cached TwelveData for {symbol}/{tf}: {len(rows)} rows")

    # PRIMARY: Fetch from TwelveData API
    if rows is None:
        try:
            td_data = await _fetch_twelvedata_training_data(symbol, tf, use_cache)
            if td_data and len(td_data) >= 50:
                return {
                    "symbol": symbol,
                    "timeframe": tf,
                    "source": "twelvedata",
                    "from_cache": False,
                    "count": len(td_data),
                    "data": td_data
                }
            rows = td_data or []
        except Exception as e:
            logger.warning(f"TwelveData API failed for {symbol}: {e}")
            rows = []

    # FALLBACK 1: If TwelveData has insufficient data, try EasyInsight
    if len(rows) < 50:
        logger.info(f"TwelveData insufficient ({len(rows)} rows), trying EasyInsight fallback for {symbol}/{tf}")
        try:
            # Resolve EasyInsight symbol from managed symbols config
            # e.g., N225 -> JP225, BTCUSD -> BTCUSD
            ei_symbol = await symbol_service.get_easyinsight_symbol(symbol)
            if ei_symbol != symbol:
                logger.info(f"Resolved EasyInsight symbol: {symbol} -> {ei_symbol}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{settings.easyinsight_api_url}/symbol-data-full/{ei_symbol}",
                    params={"limit": limit}
                )
                response.raise_for_status()

                data = response.json()
                ei_rows = data.get('data', [])

                if ei_rows and len(ei_rows) >= 50:
                    # Cache EasyInsight data (use original symbol for cache key)
                    if use_cache:
                        training_data_cache.cache_data(symbol, tf, ei_rows, "easyinsight")
                        logger.debug(f"Cached {len(ei_rows)} EasyInsight rows for {symbol}/{tf}")

                    return {
                        "symbol": symbol,  # Return original symbol in response
                        "timeframe": tf,
                        "source": "easyinsight",
                        "from_cache": False,
                        "count": len(ei_rows),
                        "data": ei_rows
                    }

        except Exception as e:
            logger.warning(f"EasyInsight API fallback failed for {symbol}: {e}")

    # Return whatever we have (might be empty or partial)
    source = "twelvedata" if rows else "none"
    return {
        "symbol": symbol,
        "timeframe": tf,
        "source": source,
        "from_cache": from_cache,
        "count": len(rows),
        "data": rows
    }


async def _fetch_twelvedata_training_data(symbol: str, timeframe: str, use_cache: bool) -> list:
    """
    Fetch training data from TwelveData API (primary source for OHLC data).

    Uses central timeframe configuration for consistent mapping.
    Supports all timeframes: M1, M5, M15, M30, H1, H4, D1, W1, MN
    """
    from ..services.training_data_cache_service import training_data_cache
    from ..services.symbol_service import symbol_service

    # Normalize timeframe using central configuration
    tf_enum = normalize_timeframe_safe(timeframe, Timeframe.H1)
    tf = tf_enum.value  # Standard format for cache keys

    # Try to get from cache first
    if use_cache:
        cached_data = training_data_cache.get_cached_data(symbol, tf, "twelvedata")
        if cached_data:
            logger.info(f"Using cached TwelveData for {symbol}/{tf}: {len(cached_data)} rows")
            return cached_data

    # Get TwelveData symbol format
    td_symbol = None
    try:
        sym = await symbol_service.get_symbol(symbol)
        if sym and sym.twelvedata_symbol:
            td_symbol = sym.twelvedata_symbol
    except:
        pass

    # Fallback: convert XXXYYY to XXX/YYY for forex pairs
    if not td_symbol:
        if len(symbol) == 6:
            td_symbol = f"{symbol[:3]}/{symbol[3:]}"
        else:
            td_symbol = symbol

    # Use central timeframe configuration for TwelveData interval
    interval = to_twelvedata(tf_enum)

    # Fetch from TwelveData - max 5000 data points
    try:
        data = await twelvedata_service.get_time_series(
            symbol=td_symbol,
            interval=interval,
            outputsize=5000  # Increased to max for better training data
        )

        values = data.get("values", [])

        # Cache the data with standardized timeframe
        if values and use_cache:
            training_data_cache.cache_data(symbol, tf, values, "twelvedata")
            logger.info(f"Fetched and cached {len(values)} TwelveData rows for {symbol}/{tf}")

        return values

    except Exception as e:
        logger.error(f"Failed to fetch TwelveData for {symbol}/{tf}: {e}")
        return []


@twelvedata_router.get("/training-data/cache/stats")
async def get_training_cache_stats():
    """Get training data cache statistics."""
    from ..services.training_data_cache_service import training_data_cache
    return training_data_cache.get_stats()


@twelvedata_router.get("/training-data/cache/symbols")
async def get_cached_training_symbols():
    """Get list of symbols currently in training data cache."""
    from ..services.training_data_cache_service import training_data_cache
    return training_data_cache.get_cached_symbols()


@twelvedata_router.delete("/training-data/cache")
async def clear_training_cache():
    """Clear all training data cache."""
    from ..services.training_data_cache_service import training_data_cache
    removed = training_data_cache.clear_all()
    return {"removed": removed, "message": f"Cleared {removed} cache files"}


@twelvedata_router.delete("/training-data/cache/expired")
async def cleanup_expired_training_cache():
    """Remove only expired training cache entries."""
    from ..services.training_data_cache_service import training_data_cache
    removed = training_data_cache.cleanup_expired()
    return {"removed": removed, "message": f"Removed {removed} expired cache entries"}


# ==================== Configuration Router ====================

@config_router.get("/config/timezone")
async def get_timezone_config():
    """Get current timezone configuration.

    Returns the configured display timezone and current time in both UTC and local format.
    """
    return get_timezone_info()


@config_router.put("/config/timezone")
async def update_timezone_config(timezone: str):
    """Update the display timezone.

    Note: This updates the runtime setting only. To persist across restarts,
    set the DISPLAY_TIMEZONE environment variable.

    Args:
        timezone: IANA timezone identifier (e.g., 'Europe/Zurich', 'America/New_York', 'UTC')

    Returns:
        Updated timezone configuration
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    # Validate timezone
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timezone: '{timezone}'. Use IANA timezone identifiers like 'Europe/Zurich', 'America/New_York', 'UTC'"
        )

    # Update runtime setting
    settings.display_timezone = timezone

    return {
        "message": f"Timezone updated to {timezone}",
        "note": "This change is not persisted. Set DISPLAY_TIMEZONE environment variable for persistence.",
        **get_timezone_info()
    }


@config_router.get("/config/timezones")
async def list_common_timezones():
    """List common timezones for configuration.

    Returns a list of commonly used IANA timezone identifiers grouped by region.
    """
    return {
        "current": settings.display_timezone,
        "timezones": {
            "Europe": [
                "Europe/Zurich",
                "Europe/Berlin",
                "Europe/London",
                "Europe/Paris",
                "Europe/Amsterdam",
                "Europe/Vienna",
                "Europe/Rome",
                "Europe/Madrid",
                "Europe/Warsaw",
                "Europe/Moscow",
            ],
            "Americas": [
                "America/New_York",
                "America/Chicago",
                "America/Denver",
                "America/Los_Angeles",
                "America/Toronto",
                "America/Sao_Paulo",
                "America/Mexico_City",
            ],
            "Asia": [
                "Asia/Tokyo",
                "Asia/Shanghai",
                "Asia/Hong_Kong",
                "Asia/Singapore",
                "Asia/Dubai",
                "Asia/Kolkata",
                "Asia/Seoul",
            ],
            "Pacific": [
                "Pacific/Auckland",
                "Pacific/Sydney",
                "Australia/Sydney",
                "Australia/Melbourne",
            ],
            "Other": [
                "UTC",
                "GMT",
            ]
        }
    }


# ==================== EasyInsight API (TimescaleDB) ====================


@easyinsight_router.get("/easyinsight/status")
async def get_easyinsight_status():
    """
    Get EasyInsight API connection status and statistics.

    Returns connection health, latency, and available symbol count.
    """
    import time
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.easyinsight_api_url}/symbols")
            latency_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                symbols = response.json()
                return {
                    "status": "connected",
                    "url": settings.easyinsight_api_url,
                    "latency_ms": round(latency_ms, 2),
                    "symbols_available": len(symbols),
                    "api_version": "1.0",
                }
            else:
                return {
                    "status": "error",
                    "url": settings.easyinsight_api_url,
                    "latency_ms": round(latency_ms, 2),
                    "error": f"HTTP {response.status_code}",
                }
    except Exception as e:
        return {
            "status": "disconnected",
            "url": settings.easyinsight_api_url,
            "error": str(e),
        }


@easyinsight_router.get("/easyinsight/symbols")
async def get_easyinsight_symbols():
    """
    Get list of all available symbols from EasyInsight TimescaleDB.

    Returns symbols with their categories, data counts, and metadata.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{settings.easyinsight_api_url}/symbols")
            response.raise_for_status()
            symbols = response.json()

            # Group by category
            categories = {}
            for s in symbols:
                cat = s.get("category", "unknown")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(s.get("symbol"))

            return {
                "total": len(symbols),
                "categories": {k: len(v) for k, v in categories.items()},
                "symbols": symbols,
            }
    except Exception as e:
        logger.error(f"Failed to fetch EasyInsight symbols: {e}")
        raise HTTPException(status_code=502, detail=f"EasyInsight API error: {str(e)}")


@easyinsight_router.get("/easyinsight/ohlcv/{symbol}")
async def get_easyinsight_ohlcv(
    symbol: str,
    limit: int = 500,
):
    """
    Get OHLCV data for a symbol from EasyInsight TimescaleDB.

    Args:
        symbol: Trading symbol (e.g., BTCUSD, EURUSD)
        limit: Number of data points to fetch (max 5000)

    Returns historical OHLCV data with H1 timeframe indicators.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.easyinsight_api_url}/symbol-data-full/{symbol}",
                params={"limit": min(limit, 5000)}
            )
            response.raise_for_status()
            data = response.json()

            rows = data.get("data", [])
            return {
                "symbol": symbol.upper(),
                "source": "easyinsight",
                "count": len(rows),
                "data": rows,
            }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in EasyInsight")
        raise HTTPException(status_code=502, detail=f"EasyInsight API error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to fetch EasyInsight OHLCV for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"EasyInsight API error: {str(e)}")


@easyinsight_router.get("/easyinsight/latest/{symbol}")
async def get_easyinsight_latest(symbol: str):
    """
    Get latest market snapshot for a symbol from EasyInsight.

    Returns the most recent OHLCV data with all available indicators.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.easyinsight_api_url}/symbol-latest-full/{symbol}"
            )
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": symbol.upper(),
                "source": "easyinsight",
                "data": data,
            }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in EasyInsight")
        raise HTTPException(status_code=502, detail=f"EasyInsight API error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to fetch EasyInsight latest for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"EasyInsight API error: {str(e)}")


@easyinsight_router.get("/easyinsight/indicators/{symbol}")
async def get_easyinsight_indicators(
    symbol: str,
    limit: int = 100,
):
    """
    Get technical indicators for a symbol from EasyInsight.

    EasyInsight provides pre-calculated indicators including:
    - RSI, MACD, Bollinger Bands
    - Moving Averages (SMA, EMA)
    - ATR, ADX, Stochastic
    - And more...

    Args:
        symbol: Trading symbol
        limit: Number of data points

    Returns OHLCV data with all available indicators.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.easyinsight_api_url}/symbol-data-full/{symbol}",
                params={"limit": limit}
            )
            response.raise_for_status()
            data = response.json()

            rows = data.get("data", [])

            # Extract indicator columns from first row
            indicators = []
            if rows:
                first_row = rows[0]
                # Find indicator columns (not basic OHLCV)
                basic_cols = {"symbol", "snapshot_time", "h1_open", "h1_high", "h1_low", "h1_close"}
                indicators = [k for k in first_row.keys() if k not in basic_cols and not k.startswith("_")]

            return {
                "symbol": symbol.upper(),
                "source": "easyinsight",
                "count": len(rows),
                "available_indicators": sorted(indicators),
                "data": rows,
            }
    except Exception as e:
        logger.error(f"Failed to fetch EasyInsight indicators for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"EasyInsight API error: {str(e)}")


@easyinsight_router.get("/easyinsight/timeframes")
async def get_easyinsight_timeframes():
    """
    Get list of supported timeframes in EasyInsight.

    Note: EasyInsight primarily stores H1 (1-hour) data with indicators.
    For other timeframes, use TwelveData API.
    """
    return {
        "primary_timeframe": "H1",
        "supported_timeframes": ["H1"],
        "note": "EasyInsight stores H1 data with pre-calculated indicators. For M1, M5, M15, H4, D1, use TwelveData API.",
        "indicator_timeframes": {
            "H1": "Full OHLCV with RSI, MACD, BBands, ATR, ADX, etc.",
        }
    }


# ==================== Yahoo Finance API ====================
# Note: yfinance_service is imported conditionally at the top of this file
# It will be None if yfinance is not installed in the container


@yfinance_router.get("/yfinance/status")
async def yfinance_status():
    """Get Yahoo Finance service status."""
    if not _yfinance_available or yfinance_service is None:
        return {"available": False, "error": "Yahoo Finance service not available in this container"}
    return yfinance_service.get_status()


@yfinance_router.get("/yfinance/symbols")
async def get_yfinance_symbols():
    """Get list of supported Yahoo Finance symbol mappings."""
    if not _yfinance_available:
        raise HTTPException(status_code=503, detail="Yahoo Finance service not available in this container")
    from ..services.yfinance_service import SYMBOL_MAPPING
    return {
        "total_mappings": len(SYMBOL_MAPPING),
        "mappings": SYMBOL_MAPPING,
        "categories": {
            "forex": [s for s in SYMBOL_MAPPING if SYMBOL_MAPPING[s].endswith("=X")],
            "crypto": [s for s in SYMBOL_MAPPING if "-USD" in SYMBOL_MAPPING[s]],
            "indices": [s for s in SYMBOL_MAPPING if SYMBOL_MAPPING[s].startswith("^")],
            "commodities": [s for s in SYMBOL_MAPPING if "=F" in SYMBOL_MAPPING[s]],
        }
    }


@yfinance_router.get("/yfinance/time-series/{symbol}")
async def get_yfinance_time_series(
    symbol: str,
    interval: str = "1d",
    outputsize: int = 100,
):
    """
    Get historical time series data from Yahoo Finance.

    Args:
        symbol: Trading symbol (e.g., BTCUSD, EURUSD, GER40)
        interval: Time interval (M15, H1, H4, D1 or yfinance format: 15m, 1h, 1d)
        outputsize: Number of data points (approximate)

    Yahoo Finance is a free data source with extensive historical data.
    """
    if not _yfinance_available or yfinance_service is None:
        raise HTTPException(status_code=503, detail="Yahoo Finance service not available in this container")
    if not yfinance_service.is_available():
        raise HTTPException(status_code=503, detail="Yahoo Finance service not available (yfinance not installed)")

    result = await yfinance_service.get_time_series(
        symbol=symbol.upper(),
        interval=interval,
        outputsize=outputsize,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@yfinance_router.get("/yfinance/quote/{symbol}")
async def get_yfinance_quote(symbol: str):
    """
    Get current quote/price for a symbol from Yahoo Finance.

    Returns latest price data including open, high, low, close, volume.
    """
    if not _yfinance_available or yfinance_service is None:
        raise HTTPException(status_code=503, detail="Yahoo Finance service not available in this container")
    if not yfinance_service.is_available():
        raise HTTPException(status_code=503, detail="Yahoo Finance service not available")

    result = await yfinance_service.get_time_series(
        symbol=symbol.upper(),
        interval="1d",
        outputsize=1,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    if result.get("values") and len(result["values"]) > 0:
        latest = result["values"][0]
        return {
            "symbol": symbol.upper(),
            "yahoo_symbol": result.get("symbol"),
            "datetime": latest.get("datetime"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "close": latest.get("close"),
            "volume": latest.get("volume"),
        }

    raise HTTPException(status_code=404, detail=f"No data available for {symbol}")


# ==================== Candlestick Pattern Detection ====================

from ..models.candlestick_patterns import (
    PatternScanRequest,
    PatternScanResponse,
    MultiTimeframePatternResult,
    Timeframe,
    PatternCategory,
    PatternDirection,
    PatternType,
)
from ..services.candlestick_pattern_service import candlestick_pattern_service


@patterns_router.get("/patterns/types")
async def get_pattern_types():
    """
    Get all supported candlestick pattern types with descriptions.

    Returns categorized list of all detectable patterns:
    - Reversal patterns (Hammer, Shooting Star, Doji, Engulfing, Morning/Evening Star)
    - Continuation patterns (Three White Soldiers, Three Black Crows)
    - Indecision patterns (Spinning Top, Harami)
    """
    patterns_by_category = {
        "reversal": {
            "bullish": [],
            "bearish": [],
            "neutral": [],
        },
        "continuation": {
            "bullish": [],
            "bearish": [],
        },
        "indecision": {
            "bullish": [],
            "bearish": [],
            "neutral": [],
        },
    }

    for pattern_type in PatternType:
        info = candlestick_pattern_service._pattern_descriptions.get(pattern_type, {})
        category = info.get("category", PatternCategory.INDECISION).value
        direction = info.get("direction", PatternDirection.NEUTRAL).value

        pattern_info = {
            "type": pattern_type.value,
            "description": info.get("description", ""),
            "trading_implication": info.get("implication", ""),
        }

        if direction in patterns_by_category.get(category, {}):
            patterns_by_category[category][direction].append(pattern_info)

    return {
        "total_patterns": len(PatternType),
        "categories": patterns_by_category,
        "supported_timeframes": [tf.value for tf in Timeframe],
    }


@patterns_router.post("/patterns/scan", response_model=PatternScanResponse)
async def scan_patterns(request: PatternScanRequest):
    """
    Scan for candlestick patterns on a symbol across multiple timeframes.

    Performs multi-timeframe analysis (M15, H1, H4, D1) to detect:

    **Reversal Patterns:**
    - Hammer, Inverted Hammer, Shooting Star, Hanging Man
    - Doji (Standard, Dragonfly, Gravestone)
    - Bullish/Bearish Engulfing
    - Morning Star, Evening Star
    - Piercing Line, Dark Cloud Cover

    **Continuation Patterns:**
    - Three White Soldiers, Three Black Crows

    **Indecision Patterns:**
    - Spinning Top
    - Bullish/Bearish Harami
    - Harami Cross

    Returns patterns with confidence scores and trading implications.
    """
    try:
        response = await candlestick_pattern_service.scan_patterns(request)
        return response
    except Exception as e:
        logger.error(f"Pattern scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@patterns_router.get("/patterns/scan/{symbol}", response_model=PatternScanResponse)
async def scan_patterns_simple(
    symbol: str,
    timeframes: str = "M15,H1,H4,D1",
    lookback: int = 100,
    min_confidence: float = 0.5,
):
    """
    Simple GET endpoint for pattern scanning (alternative to POST).

    Args:
        symbol: Trading symbol (e.g., BTCUSD, EURUSD)
        timeframes: Comma-separated timeframes (M15,H1,H4,D1)
        lookback: Number of candles to analyze per timeframe (10-500)
        min_confidence: Minimum pattern confidence threshold (0.0-1.0)

    Example: GET /api/v1/patterns/scan/BTCUSD?timeframes=H1,H4&min_confidence=0.6
    """
    try:
        # Parse timeframes
        tf_list = []
        for tf in timeframes.upper().split(","):
            tf = tf.strip()
            if tf in [t.value for t in Timeframe]:
                tf_list.append(Timeframe(tf))

        if not tf_list:
            tf_list = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]

        request = PatternScanRequest(
            symbol=symbol.upper(),
            timeframes=tf_list,
            lookback_candles=min(max(lookback, 10), 500),
            min_confidence=min(max(min_confidence, 0.0), 1.0),
        )

        response = await candlestick_pattern_service.scan_patterns(request)
        return response
    except Exception as e:
        logger.error(f"Pattern scan failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@patterns_router.get("/patterns/summary/{symbol}")
async def get_pattern_summary(
    symbol: str,
    timeframes: str = "M15,H1,H4,D1",
):
    """
    Get a simplified pattern summary for a symbol.

    Returns a condensed view with:
    - Dominant market direction (bullish/bearish/neutral)
    - Confluence score across timeframes
    - Count of patterns by type
    - Most significant pattern detected
    """
    try:
        tf_list = []
        for tf in timeframes.upper().split(","):
            tf = tf.strip()
            if tf in [t.value for t in Timeframe]:
                tf_list.append(Timeframe(tf))

        if not tf_list:
            tf_list = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]

        request = PatternScanRequest(
            symbol=symbol.upper(),
            timeframes=tf_list,
            lookback_candles=50,
            min_confidence=0.5,
        )

        response = await candlestick_pattern_service.scan_patterns(request)
        result = response.result

        # Build summary
        summary = {
            "symbol": symbol.upper(),
            "scan_timestamp": result.scan_timestamp.isoformat(),
            "dominant_direction": result.dominant_direction.value if result.dominant_direction else "neutral",
            "confluence_score": round(result.confluence_score, 2),
            "total_patterns": result.total_patterns_found,
            "pattern_counts": {
                "bullish": result.bullish_patterns_count,
                "bearish": result.bearish_patterns_count,
                "neutral": result.neutral_patterns_count,
            },
            "timeframe_summary": {},
        }

        # Add per-timeframe summary
        for tf_name, tf_result in [
            ("M15", result.m15),
            ("H1", result.h1),
            ("H4", result.h4),
            ("D1", result.d1),
        ]:
            if tf_result.patterns:
                summary["timeframe_summary"][tf_name] = {
                    "patterns_found": len(tf_result.patterns),
                    "candles_analyzed": tf_result.candles_analyzed,
                    "patterns": [
                        {
                            "type": p.pattern_type.value,
                            "direction": p.direction.value,
                            "confidence": round(p.confidence, 2),
                        }
                        for p in tf_result.patterns
                    ],
                }

        # Add strongest pattern
        if result.strongest_pattern:
            summary["strongest_pattern"] = {
                "type": result.strongest_pattern.pattern_type.value,
                "direction": result.strongest_pattern.direction.value,
                "timeframe": result.strongest_pattern.timeframe.value,
                "confidence": round(result.strongest_pattern.confidence, 2),
                "description": result.strongest_pattern.description,
                "trading_implication": result.strongest_pattern.trading_implication,
            }

        return summary

    except Exception as e:
        logger.error(f"Pattern summary failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@patterns_router.get("/patterns/scan-all")
async def scan_all_symbols(
    min_confidence: float = 0.6,
    timeframes: str = "H1,H4",
):
    """
    Scan all available symbols for candlestick patterns.

    Warning: This can take a while depending on the number of symbols.

    Args:
        min_confidence: Minimum pattern confidence threshold (default: 0.6)
        timeframes: Comma-separated timeframes to scan (default: H1,H4)

    Returns summary of patterns found across all symbols.
    """
    try:
        tf_list = []
        for tf in timeframes.upper().split(","):
            tf = tf.strip()
            if tf in [t.value for t in Timeframe]:
                tf_list.append(Timeframe(tf))

        if not tf_list:
            tf_list = [Timeframe.H1, Timeframe.H4]

        results = await candlestick_pattern_service.scan_all_symbols(
            timeframes=tf_list,
            min_confidence=min_confidence,
        )

        # Build summary
        symbols_with_patterns = []
        for symbol, result in results.items():
            if result.total_patterns_found > 0:
                symbols_with_patterns.append({
                    "symbol": symbol,
                    "total_patterns": result.total_patterns_found,
                    "dominant_direction": result.dominant_direction.value if result.dominant_direction else "neutral",
                    "confluence_score": round(result.confluence_score, 2),
                    "bullish": result.bullish_patterns_count,
                    "bearish": result.bearish_patterns_count,
                })

        # Sort by total patterns (descending)
        symbols_with_patterns.sort(key=lambda x: x["total_patterns"], reverse=True)

        return {
            "total_symbols_scanned": len(results),
            "symbols_with_patterns": len(symbols_with_patterns),
            "timeframes_scanned": [tf.value for tf in tf_list],
            "min_confidence": min_confidence,
            "results": symbols_with_patterns,
        }

    except Exception as e:
        logger.error(f"Scan all symbols failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Pattern History Endpoints ====================

from ..services.pattern_history_service import pattern_history_service


@patterns_router.get("/patterns/history")
async def get_pattern_history(
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    category: Optional[str] = None,
    timeframe: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 100,
):
    """
    Get pattern detection history with optional filters.

    Args:
        symbol: Filter by trading symbol (e.g., BTCUSD)
        direction: Filter by direction (bullish, bearish, neutral)
        category: Filter by category (reversal, continuation, indecision)
        timeframe: Filter by timeframe (M15, H1, H4, D1)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        limit: Maximum number of results (default: 100)

    Returns list of detected patterns sorted by detection time (newest first).
    """
    return pattern_history_service.get_history(
        symbol=symbol,
        direction=direction,
        category=category,
        timeframe=timeframe,
        min_confidence=min_confidence,
        limit=limit,
    )


@patterns_router.get("/patterns/history/by-symbol")
async def get_patterns_by_symbol():
    """
    Get latest patterns grouped by symbol.

    Returns dictionary with symbol as key and list of recent patterns as value.
    Useful for dashboard overview.
    """
    return pattern_history_service.get_latest_by_symbol()


@patterns_router.get("/patterns/history/statistics")
async def get_pattern_statistics():
    """
    Get pattern detection statistics.

    Returns:
        - total_patterns: Total number of patterns in history
        - symbols_with_patterns: Number of symbols with detected patterns
        - by_direction: Count by direction (bullish/bearish/neutral)
        - by_category: Count by category (reversal/continuation/indecision)
        - by_timeframe: Count by timeframe (M15/H1/H4/D1)
        - last_scan: Timestamp of last scan
        - scan_running: Whether automatic scanning is active
    """
    return pattern_history_service.get_statistics()


@patterns_router.post("/patterns/history/scan")
async def trigger_pattern_scan():
    """
    Manually trigger a pattern scan for all symbols.

    Scans all available symbols across configured timeframes
    and stores detected patterns in history.

    Returns number of new patterns found.
    """
    try:
        new_patterns = await pattern_history_service.scan_all_symbols()
        return {
            "status": "completed",
            "new_patterns_found": new_patterns,
            "message": f"Scan complete. Found {new_patterns} new patterns.",
        }
    except Exception as e:
        logger.error(f"Manual pattern scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@patterns_router.post("/patterns/history/start-auto-scan")
async def start_auto_scan():
    """
    Start automatic periodic pattern scanning.

    Scans run every 5 minutes in the background.
    """
    await pattern_history_service.start()
    return {
        "status": "started",
        "message": "Automatic pattern scanning started",
        "interval_seconds": pattern_history_service._scan_interval,
    }


@patterns_router.post("/patterns/history/stop-auto-scan")
async def stop_auto_scan():
    """
    Stop automatic periodic pattern scanning.
    """
    await pattern_history_service.stop()
    return {
        "status": "stopped",
        "message": "Automatic pattern scanning stopped",
    }


@patterns_router.delete("/patterns/history")
async def clear_pattern_history():
    """
    Clear all pattern history.

    Warning: This action cannot be undone.
    """
    pattern_history_service.clear_history()
    return {
        "status": "cleared",
        "message": "Pattern history has been cleared",
    }


@patterns_router.get("/patterns/chart/{symbol}")
async def get_pattern_chart_data(
    symbol: str,
    timeframe: str = "H1",
    timestamp: str = None,
    candles_before: int = 15,
    candles_after: int = 5,
):
    """
    Get OHLCV candle data for visualizing a detected pattern.

    This endpoint returns candle data centered around a pattern detection,
    suitable for rendering a candlestick chart with the pattern highlighted.

    Args:
        symbol: Trading symbol (e.g., BTCUSD, EURUSD)
        timeframe: Pattern timeframe (M5, M15, H1, H4, D1)
        timestamp: ISO 8601 timestamp of the pattern (optional, defaults to latest)
        candles_before: Number of candles to show before pattern (default: 15)
        candles_after: Number of candles to show after pattern (default: 5)

    Returns:
        - symbol: The trading symbol
        - timeframe: The timeframe
        - candles: Array of OHLCV data with is_pattern flag
        - pattern_candle_index: Index of the pattern candle in the array

    Example: GET /api/v1/patterns/chart/BTCUSD?timeframe=H1&candles_before=20
    """
    from datetime import datetime, timezone

    try:
        # Parse timestamp or use current time
        if timestamp:
            try:
                from dateutil import parser as dateutil_parser
                pattern_ts = dateutil_parser.isoparse(timestamp)
                if pattern_ts.tzinfo is None:
                    pattern_ts = pattern_ts.replace(tzinfo=timezone.utc)
            except (ValueError, ImportError):
                pattern_ts = datetime.now(timezone.utc)
        else:
            pattern_ts = datetime.now(timezone.utc)

        # Validate parameters
        candles_before = min(max(candles_before, 5), 50)
        candles_after = min(max(candles_after, 0), 20)

        result = await candlestick_pattern_service.get_pattern_chart_data(
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            pattern_timestamp=pattern_ts,
            candles_before=candles_before,
            candles_after=candles_after,
        )

        return result

    except Exception as e:
        logger.error(f"Pattern chart data failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== External Data Sources ====================

# NOTE: Imports are done lazily inside functions to avoid loading
# aiohttp and other dependencies in services that don't use them (e.g., NHITS)


def get_external_data_fetcher():
    """Get the data fetcher service singleton."""
    from ..services.data_app.external_sources import get_data_fetcher_service
    return get_data_fetcher_service()


def _get_data_source_type():
    """Get DataSourceType enum (lazy import)."""
    from ..services.data_app.external_sources import DataSourceType
    return DataSourceType


def _get_data_priority():
    """Get DataPriority enum (lazy import)."""
    from ..services.data_app.external_sources import DataPriority
    return DataPriority


@external_sources_router.get("/external-sources")
async def get_available_sources():
    """
    Get list of all available external data sources.

    Returns information about each data source including:
    - Source type (economic_calendar, sentiment, etc.)
    - Description
    - Data categories provided
    """
    fetcher = get_external_data_fetcher()
    return {
        "sources": fetcher.get_available_sources(),
        "total_sources": len(fetcher.get_available_sources()),
    }


@external_sources_router.get("/external-sources/economic-calendar")
async def get_economic_calendar(
    symbol: Optional[str] = None,
    days_ahead: int = 7,
    days_back: int = 1
):
    """
    Get economic calendar events.

    Returns upcoming and recent economic events including:
    - Central bank decisions (Fed, ECB, BOJ, etc.)
    - Economic releases (CPI, NFP, GDP, etc.)
    - Market-moving events

    Args:
        symbol: Optional symbol to filter relevant events
        days_ahead: Days to look ahead (default: 7)
        days_back: Days to look back (default: 1)
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_economic_calendar(symbol, days_ahead, days_back)
    return {
        "source": "economic_calendar",
        "symbol": symbol,
        "days_ahead": days_ahead,
        "days_back": days_back,
        "events": [r.to_rag_document() for r in results],
        "total_events": len(results),
    }


@external_sources_router.get("/external-sources/sentiment")
async def get_sentiment_data(
    symbol: Optional[str] = None,
    include_fear_greed: bool = True,
    include_social: bool = True,
    include_options: bool = True,
    include_volatility: bool = True
):
    """
    Get market sentiment data.

    Returns sentiment indicators including:
    - Fear & Greed Index (Crypto and Traditional)
    - Social media sentiment
    - Options market sentiment (Put/Call ratio)
    - Volatility indicators (VIX)
    - Funding rates (Crypto)

    Args:
        symbol: Optional symbol for specific sentiment
        include_fear_greed: Include Fear & Greed Index
        include_social: Include social media sentiment
        include_options: Include options sentiment
        include_volatility: Include VIX/volatility data
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_sentiment(
        symbol, include_fear_greed, include_social, include_options, include_volatility
    )
    return {
        "source": "sentiment",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/onchain/{symbol}")
async def get_onchain_data(
    symbol: str,
    include_whale_alerts: bool = True,
    include_exchange_flows: bool = True,
    include_mining: bool = True,
    include_defi: bool = True
):
    """
    Get on-chain data for a cryptocurrency.

    Returns on-chain metrics including:
    - Whale alerts (large transactions)
    - Exchange inflows/outflows
    - Mining metrics (hashrate, difficulty)
    - DeFi TVL and protocol data

    Args:
        symbol: Crypto symbol (e.g., BTCUSD, ETHUSD)
        include_whale_alerts: Include whale transaction alerts
        include_exchange_flows: Include exchange flow data
        include_mining: Include mining metrics
        include_defi: Include DeFi metrics
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_onchain(
        symbol, include_whale_alerts, include_exchange_flows, include_mining, include_defi
    )
    return {
        "source": "onchain",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/orderbook/{symbol}")
async def get_orderbook_data(
    symbol: str,
    depth: int = 50,
    include_liquidations: bool = True,
    include_cvd: bool = True
):
    """
    Get orderbook and liquidity data.

    Returns orderbook analysis including:
    - Bid/Ask depth analysis
    - Order walls and support/resistance
    - Liquidation levels
    - Cumulative Volume Delta (CVD)

    Args:
        symbol: Trading symbol
        depth: Orderbook depth levels
        include_liquidations: Include liquidation heatmap
        include_cvd: Include CVD analysis
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_orderbook(symbol, depth, include_liquidations, include_cvd)
    return {
        "source": "orderbook",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/macro")
async def get_macro_data(
    symbol: Optional[str] = None,
    include_dxy: bool = True,
    include_bonds: bool = True,
    include_correlations: bool = True,
    include_sectors: bool = True
):
    """
    Get macro economic and correlation data.

    Returns macro data including:
    - DXY (US Dollar Index) analysis
    - Bond yields (2Y, 10Y, 30Y)
    - Cross-asset correlations
    - Sector rotation analysis

    Args:
        symbol: Optional symbol for specific correlations
        include_dxy: Include DXY analysis
        include_bonds: Include bond yield data
        include_correlations: Include correlation matrices
        include_sectors: Include sector data
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_macro(
        symbol, include_dxy, include_bonds, include_correlations, include_sectors
    )
    return {
        "source": "macro_correlation",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/historical-patterns")
async def get_historical_patterns(
    symbol: Optional[str] = None,
    include_seasonality: bool = True,
    include_drawdowns: bool = True,
    include_events: bool = True,
    include_comparable: bool = True
):
    """
    Get historical pattern analysis.

    Returns historical data including:
    - Seasonality patterns (monthly, weekly)
    - Historical drawdowns
    - Event-based analysis (halvings, FOMC)
    - Comparable market phases

    Args:
        symbol: Optional symbol for specific patterns
        include_seasonality: Include seasonality data
        include_drawdowns: Include drawdown history
        include_events: Include event analysis
        include_comparable: Include comparable phases
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_historical_patterns(
        symbol, include_seasonality, include_drawdowns, include_events, include_comparable
    )
    return {
        "source": "historical_patterns",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/technical-levels/{symbol}")
async def get_technical_levels(
    symbol: str,
    include_sr: bool = True,
    include_fib: bool = True,
    include_pivots: bool = True,
    include_vwap: bool = True,
    include_ma: bool = True
):
    """
    Get technical price levels.

    Returns technical levels including:
    - Support/Resistance zones
    - Fibonacci retracements
    - Pivot points (daily, weekly, monthly)
    - VWAP levels
    - Moving average levels

    Args:
        symbol: Trading symbol
        include_sr: Include S/R zones
        include_fib: Include Fibonacci levels
        include_pivots: Include pivot points
        include_vwap: Include VWAP
        include_ma: Include moving averages
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_technical_levels(
        symbol, include_sr, include_fib, include_pivots, include_vwap, include_ma
    )
    return {
        "source": "technical_levels",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/regulatory")
async def get_regulatory_updates(
    symbol: Optional[str] = None,
    include_sec: bool = True,
    include_etf: bool = True,
    include_global: bool = True,
    include_enforcement: bool = True
):
    """
    Get regulatory updates.

    Returns regulatory information including:
    - SEC/CFTC decisions and positions
    - ETF news and approvals
    - Global regulation (EU MiCA, UK FCA, Asia)
    - Enforcement actions

    Args:
        symbol: Optional symbol for specific updates
        include_sec: Include SEC/CFTC updates
        include_etf: Include ETF news
        include_global: Include global regulation
        include_enforcement: Include enforcement actions
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_regulatory(
        symbol, include_sec, include_etf, include_global, include_enforcement
    )
    return {
        "source": "regulatory",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/easyinsight")
async def get_easyinsight_data(
    symbol: Optional[str] = None,
    include_symbols: bool = True,
    include_stats: bool = True,
    include_mt5_logs: bool = True
):
    """
    Get EasyInsight managed data.

    Returns EasyInsight information including:
    - Managed trading symbols
    - Data availability statistics
    - MT5 trading logs

    Args:
        symbol: Optional symbol filter
        include_symbols: Include symbol configurations
        include_stats: Include data statistics
        include_mt5_logs: Include MT5 logs
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_easyinsight(
        symbol, include_symbols, include_stats, include_mt5_logs
    )
    return {
        "source": "easyinsight",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/correlations")
async def get_correlations_data(
    symbol: Optional[str] = None,
    timeframe: str = "30d",
    include_matrix: bool = True,
    include_regime: bool = True
):
    """
    Get asset correlation analysis data.

    Returns correlation data including:
    - Cross-asset correlations (rolling 7d, 30d, 90d windows)
    - Correlation matrix for asset class
    - Correlation regime classification
    - Divergence detection for trading signals
    - Hedge recommendations based on negative correlations

    Args:
        symbol: Optional symbol to analyze correlations for
        timeframe: Correlation window (7d, 30d, 90d)
        include_matrix: Include full correlation matrix
        include_regime: Include correlation regime analysis
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_correlations(
        symbol, timeframe, include_matrix, include_regime
    )
    return {
        "source": "correlations",
        "symbol": symbol,
        "timeframe": timeframe,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/volatility-regime")
async def get_volatility_regime_data(
    symbol: Optional[str] = None,
    include_vix: bool = True,
    include_atr: bool = True,
    include_bollinger: bool = True,
    include_regime: bool = True
):
    """
    Get volatility regime analysis for position sizing and risk management.

    Returns volatility data including:
    - VIX and volatility indices analysis
    - ATR (Average True Range) trends with stop-loss recommendations
    - Bollinger Band width and squeeze detection
    - Volatility regime classification (low, normal, high, extreme)
    - Position sizing recommendations based on volatility

    Args:
        symbol: Optional symbol to analyze
        include_vix: Include VIX analysis
        include_atr: Include ATR trend analysis
        include_bollinger: Include Bollinger width analysis
        include_regime: Include regime classification
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_volatility_regime(
        symbol, include_vix, include_atr, include_bollinger, include_regime
    )
    return {
        "source": "volatility_regime",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.get("/external-sources/institutional-flow")
async def get_institutional_flow_data(
    symbol: Optional[str] = None,
    include_cot: bool = True,
    include_etf: bool = True,
    include_whale: bool = True,
    include_13f: bool = False
):
    """
    Get institutional flow data (Smart Money tracking).

    Returns institutional data including:
    - CFTC Commitment of Traders (COT) Reports
    - ETF Inflows/Outflows (Bitcoin, Gold, major ETFs)
    - Whale wallet activity (crypto)
    - 13F Filings analysis (quarterly, optional)
    - Aggregated Smart Money signal

    Args:
        symbol: Optional symbol to analyze
        include_cot: Include COT report analysis
        include_etf: Include ETF flow data
        include_whale: Include whale tracking (crypto only)
        include_13f: Include 13F filing analysis (quarterly data)
    """
    fetcher = get_external_data_fetcher()
    results = await fetcher.fetch_institutional_flow(
        symbol, include_cot, include_etf, include_whale, include_13f
    )
    return {
        "source": "institutional_flow",
        "symbol": symbol,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.post("/external-sources/fetch-all")
async def fetch_all_sources(
    symbol: Optional[str] = None,
    source_types: Optional[list[str]] = None,
    min_priority: str = "low"
):
    """
    Fetch data from all or selected external sources.

    Args:
        symbol: Optional symbol filter
        source_types: List of source types to fetch (None = all)
            Valid types: economic_calendar, onchain, sentiment, orderbook,
            macro_correlation, historical_pattern, technical_level, regulatory,
            easyinsight, correlations, volatility_regime, institutional_flow
        min_priority: Minimum priority level (critical, high, medium, low)

    Returns:
        Aggregated data from all requested sources
    """
    fetcher = get_external_data_fetcher()
    DataSourceType = _get_data_source_type()
    DataPriority = _get_data_priority()

    # Map string source types to enum
    types_enum = None
    if source_types:
        type_mapping = {
            "economic_calendar": DataSourceType.ECONOMIC_CALENDAR,
            "onchain": DataSourceType.ONCHAIN,
            "sentiment": DataSourceType.SENTIMENT,
            "orderbook": DataSourceType.ORDERBOOK,
            "macro_correlation": DataSourceType.MACRO_CORRELATION,
            "historical_pattern": DataSourceType.HISTORICAL_PATTERN,
            "technical_level": DataSourceType.TECHNICAL_LEVEL,
            "regulatory": DataSourceType.REGULATORY,
            "easyinsight": DataSourceType.EASYINSIGHT,
            "correlations": DataSourceType.CORRELATIONS,
            "volatility_regime": DataSourceType.VOLATILITY_REGIME,
            "institutional_flow": DataSourceType.INSTITUTIONAL_FLOW,
        }
        types_enum = [type_mapping[t] for t in source_types if t in type_mapping]

    # Map priority string to enum
    priority_mapping = {
        "critical": DataPriority.CRITICAL,
        "high": DataPriority.HIGH,
        "medium": DataPriority.MEDIUM,
        "low": DataPriority.LOW,
    }
    priority = priority_mapping.get(min_priority.lower(), DataPriority.LOW)

    results = await fetcher.fetch_all(symbol, types_enum, priority)
    return {
        "symbol": symbol,
        "source_types": source_types or "all",
        "min_priority": min_priority,
        "data": [r.to_rag_document() for r in results],
        "total_items": len(results),
    }


@external_sources_router.post("/external-sources/trading-context/{symbol}")
async def fetch_trading_context(
    symbol: str,
    include_types: Optional[list[str]] = None
):
    """
    Fetch comprehensive trading context for a symbol.

    Provides a complete market view by aggregating data from multiple sources.

    Args:
        symbol: Trading symbol
        include_types: Optional list of data types to include:
            economic, onchain, sentiment, orderbook, macro, patterns, levels,
            regulatory, easyinsight, correlations, volatility, institutional

    Returns:
        Structured trading context with summary of critical events
    """
    fetcher = get_external_data_fetcher()
    context = await fetcher.fetch_trading_context(symbol, include_types)
    return context


# ==================== Backup & Restore ====================

@backup_router.get("/backup/status")
async def get_backup_status():
    """Get current status of backupable data (models and predictions)."""
    from pathlib import Path
    import os

    try:
        # Model info
        from ..services.forecast_service import forecast_service
        model_path = forecast_service.model_path
        models = list(model_path.glob("*_model.pt"))
        metadata_files = list(model_path.glob("*_metadata.json"))

        model_size_bytes = sum(f.stat().st_size for f in models if f.exists())
        metadata_size_bytes = sum(f.stat().st_size for f in metadata_files if f.exists())

        # Prediction info
        from ..services.model_improvement_service import model_improvement_service
        pending_count = sum(len(v) for v in model_improvement_service.pending_feedback.values())
        evaluated_count = sum(len(v) for v in model_improvement_service.evaluated_feedback.values())
        metrics_count = len(model_improvement_service.performance_metrics)

        return {
            "models": {
                "count": len(models),
                "metadata_count": len(metadata_files),
                "size_mb": round((model_size_bytes + metadata_size_bytes) / (1024 * 1024), 2),
                "path": str(model_path),
            },
            "predictions": {
                "pending_count": pending_count,
                "evaluated_count": evaluated_count,
                "symbols_with_metrics": metrics_count,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Backup status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@backup_router.get("/backup/models/download")
async def download_models_backup():
    """Download a backup of all NHITS models and metadata as ZIP file."""
    from pathlib import Path
    import tempfile
    import zipfile
    from fastapi.responses import FileResponse

    try:
        from ..services.forecast_service import forecast_service
        model_path = forecast_service.model_path

        models = list(model_path.glob("*_model.pt"))
        metadata_files = list(model_path.glob("*_metadata.json"))

        if not models:
            raise HTTPException(status_code=404, detail="Keine Modelle zum Backup vorhanden")

        # Create temporary zip file
        filename = f"nhits_models_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
        tmp_path = Path(tempfile.gettempdir()) / filename

        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for model_file in models:
                zf.write(model_file, model_file.name)
            for meta_file in metadata_files:
                zf.write(meta_file, meta_file.name)

            # Add manifest
            manifest = {
                "created_at": datetime.utcnow().isoformat(),
                "models": [m.name for m in models],
                "metadata": [m.name for m in metadata_files],
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        return FileResponse(
            path=str(tmp_path),
            filename=filename,
            media_type="application/zip",
            background=None  # File will be cleaned up after response
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Models backup download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@backup_router.post("/backup/models")
async def create_models_backup_info():
    """Get information about available models for backup (use /backup/models/download for actual download)."""
    from pathlib import Path

    try:
        from ..services.forecast_service import forecast_service
        model_path = forecast_service.model_path

        models = list(model_path.glob("*_model.pt"))
        metadata_files = list(model_path.glob("*_metadata.json"))

        if not models:
            return {
                "success": False,
                "message": "Keine Modelle zum Backup vorhanden",
                "model_count": 0,
            }

        model_size_bytes = sum(f.stat().st_size for f in models if f.exists())
        metadata_size_bytes = sum(f.stat().st_size for f in metadata_files if f.exists())

        return {
            "success": True,
            "message": f"{len(models)} Modelle bereit zum Download",
            "model_count": len(models),
            "metadata_count": len(metadata_files),
            "size_mb": round((model_size_bytes + metadata_size_bytes) / (1024 * 1024), 2),
            "download_url": "/api/v1/backup/models/download",
        }
    except Exception as e:
        logger.error(f"Models backup info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@backup_router.post("/backup/predictions")
async def create_predictions_backup():
    """Create a backup of all prediction data (pending and evaluated)."""
    try:
        from ..services.model_improvement_service import model_improvement_service

        # Convert dataclass objects to dicts
        pending_data = {}
        for symbol, feedbacks in model_improvement_service.pending_feedback.items():
            pending_data[symbol] = [
                {
                    "symbol": fb.symbol,
                    "timestamp": fb.timestamp.isoformat() if fb.timestamp else None,
                    "horizon": fb.horizon,
                    "current_price": fb.current_price,
                    "predicted_price": fb.predicted_price,
                    "actual_price": fb.actual_price,
                    "prediction_error_pct": fb.prediction_error_pct,
                    "direction_correct": fb.direction_correct,
                    "evaluated_at": fb.evaluated_at.isoformat() if fb.evaluated_at else None,
                }
                for fb in feedbacks
            ]

        evaluated_data = {}
        for symbol, feedbacks in model_improvement_service.evaluated_feedback.items():
            evaluated_data[symbol] = [
                {
                    "symbol": fb.symbol,
                    "timestamp": fb.timestamp.isoformat() if fb.timestamp else None,
                    "horizon": fb.horizon,
                    "current_price": fb.current_price,
                    "predicted_price": fb.predicted_price,
                    "actual_price": fb.actual_price,
                    "prediction_error_pct": fb.prediction_error_pct,
                    "direction_correct": fb.direction_correct,
                    "evaluated_at": fb.evaluated_at.isoformat() if fb.evaluated_at else None,
                }
                for fb in feedbacks
            ]

        metrics_data = {}
        for symbol, metrics in model_improvement_service.performance_metrics.items():
            metrics_data[symbol] = {
                "symbol": metrics.symbol,
                "total_predictions": metrics.total_predictions,
                "evaluated_predictions": metrics.evaluated_predictions,
                "avg_error_pct": metrics.avg_error_pct,
                "direction_accuracy": metrics.direction_accuracy,
                "last_updated": metrics.last_updated.isoformat() if metrics.last_updated else None,
                "metrics_1h": metrics.metrics_1h,
                "metrics_4h": metrics.metrics_4h,
                "metrics_24h": metrics.metrics_24h,
                "needs_retraining": metrics.needs_retraining,
                "retraining_reason": metrics.retraining_reason,
            }

        backup_data = {
            "type": "predictions",
            "created_at": datetime.utcnow().isoformat(),
            "pending_feedback": pending_data,
            "evaluated_feedback": evaluated_data,
            "performance_metrics": metrics_data,
        }

        pending_count = sum(len(v) for v in pending_data.values())
        evaluated_count = sum(len(v) for v in evaluated_data.values())

        return {
            "success": True,
            "message": f"Prognose-Daten gesichert: {pending_count} ausstehend, {evaluated_count} evaluiert",
            "pending_count": pending_count,
            "evaluated_count": evaluated_count,
            "metrics_count": len(metrics_data),
            "backup_data": backup_data,
            "filename": f"nhits_predictions_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        }
    except Exception as e:
        logger.error(f"Predictions backup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@backup_router.post("/restore/models")
async def restore_models_backup(file: UploadFile = File(...)):
    """Restore NHITS models from a backup zip file."""
    from pathlib import Path
    import tempfile
    import zipfile

    try:
        from ..services.forecast_service import forecast_service
        model_path = forecast_service.model_path

        # Read uploaded file
        content = await file.read()

        # Save to temp file and extract
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        restored_models = 0
        restored_metadata = 0

        with zipfile.ZipFile(tmp_path, "r") as zf:
            for name in zf.namelist():
                if name == "manifest.json":
                    continue

                # Extract to model path
                target_path = model_path / name
                with zf.open(name) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

                if name.endswith("_model.pt"):
                    restored_models += 1
                elif name.endswith("_metadata.json"):
                    restored_metadata += 1

        Path(tmp_path).unlink()

        return {
            "success": True,
            "message": f"Backup wiederhergestellt: {restored_models} Modelle, {restored_metadata} Metadaten",
            "restored_models": restored_models,
            "restored_metadata": restored_metadata,
        }
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Ungültige ZIP-Datei")
    except Exception as e:
        logger.error(f"Models restore failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@backup_router.post("/restore/predictions")
async def restore_predictions_backup(file: UploadFile = File(...)):
    """Restore prediction data from a backup JSON file."""
    try:
        from ..services.model_improvement_service import (
            model_improvement_service,
            PredictionFeedback,
            ModelPerformanceMetrics,
        )

        # Read and parse JSON
        content = await file.read()
        backup_data = json.loads(content.decode("utf-8"))

        if backup_data.get("type") != "predictions":
            raise HTTPException(status_code=400, detail="Ungültiges Backup-Format (erwartet: predictions)")

        # Restore pending feedback
        restored_pending = 0
        for symbol, feedbacks in backup_data.get("pending_feedback", {}).items():
            if symbol not in model_improvement_service.pending_feedback:
                model_improvement_service.pending_feedback[symbol] = []

            for fb_data in feedbacks:
                fb = PredictionFeedback(
                    symbol=fb_data["symbol"],
                    timestamp=datetime.fromisoformat(fb_data["timestamp"]) if fb_data.get("timestamp") else datetime.utcnow(),
                    horizon=fb_data["horizon"],
                    current_price=fb_data["current_price"],
                    predicted_price=fb_data["predicted_price"],
                    actual_price=fb_data.get("actual_price"),
                    prediction_error_pct=fb_data.get("prediction_error_pct"),
                    direction_correct=fb_data.get("direction_correct"),
                    evaluated_at=datetime.fromisoformat(fb_data["evaluated_at"]) if fb_data.get("evaluated_at") else None,
                )
                model_improvement_service.pending_feedback[symbol].append(fb)
                restored_pending += 1

        # Restore evaluated feedback
        restored_evaluated = 0
        for symbol, feedbacks in backup_data.get("evaluated_feedback", {}).items():
            if symbol not in model_improvement_service.evaluated_feedback:
                model_improvement_service.evaluated_feedback[symbol] = []

            for fb_data in feedbacks:
                fb = PredictionFeedback(
                    symbol=fb_data["symbol"],
                    timestamp=datetime.fromisoformat(fb_data["timestamp"]) if fb_data.get("timestamp") else datetime.utcnow(),
                    horizon=fb_data["horizon"],
                    current_price=fb_data["current_price"],
                    predicted_price=fb_data["predicted_price"],
                    actual_price=fb_data.get("actual_price"),
                    prediction_error_pct=fb_data.get("prediction_error_pct"),
                    direction_correct=fb_data.get("direction_correct"),
                    evaluated_at=datetime.fromisoformat(fb_data["evaluated_at"]) if fb_data.get("evaluated_at") else None,
                )
                model_improvement_service.evaluated_feedback[symbol].append(fb)
                restored_evaluated += 1

        # Restore performance metrics
        restored_metrics = 0
        for symbol, metrics_data in backup_data.get("performance_metrics", {}).items():
            metrics = ModelPerformanceMetrics(
                symbol=metrics_data["symbol"],
                total_predictions=metrics_data.get("total_predictions", 0),
                evaluated_predictions=metrics_data.get("evaluated_predictions", 0),
                avg_error_pct=metrics_data.get("avg_error_pct", 0.0),
                direction_accuracy=metrics_data.get("direction_accuracy", 0.0),
                last_updated=datetime.fromisoformat(metrics_data["last_updated"]) if metrics_data.get("last_updated") else None,
                needs_retraining=metrics_data.get("needs_retraining", False),
                retraining_reason=metrics_data.get("retraining_reason"),
            )
            if metrics_data.get("metrics_1h"):
                metrics.metrics_1h = metrics_data["metrics_1h"]
            if metrics_data.get("metrics_4h"):
                metrics.metrics_4h = metrics_data["metrics_4h"]
            if metrics_data.get("metrics_24h"):
                metrics.metrics_24h = metrics_data["metrics_24h"]

            model_improvement_service.performance_metrics[symbol] = metrics
            restored_metrics += 1

        # Save to persistent storage
        model_improvement_service._save_data()

        return {
            "success": True,
            "message": f"Prognose-Backup wiederhergestellt",
            "restored_pending": restored_pending,
            "restored_evaluated": restored_evaluated,
            "restored_metrics": restored_metrics,
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Ungültige JSON-Datei")
    except Exception as e:
        logger.error(f"Predictions restore failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@backup_router.delete("/backup/predictions/clear")
async def clear_predictions():
    """Clear all prediction data (pending, evaluated, and metrics)."""
    try:
        from ..services.model_improvement_service import model_improvement_service

        pending_count = sum(len(v) for v in model_improvement_service.pending_feedback.values())
        evaluated_count = sum(len(v) for v in model_improvement_service.evaluated_feedback.values())
        metrics_count = len(model_improvement_service.performance_metrics)

        model_improvement_service.pending_feedback.clear()
        model_improvement_service.evaluated_feedback.clear()
        model_improvement_service.performance_metrics.clear()
        model_improvement_service._save_data()

        return {
            "success": True,
            "message": "Alle Prognose-Daten gelöscht",
            "cleared_pending": pending_count,
            "cleared_evaluated": evaluated_count,
            "cleared_metrics": metrics_count,
        }
    except Exception as e:
        logger.error(f"Clear predictions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Indicators Router (Technical Indicators Registry) ====================


@indicators_router.get("/indicators")
async def list_indicators(
    category: Optional[str] = Query(None, description="Filter by category (trend, momentum, volatility, volume, trend_filter, price_transform, ml_features, pattern)"),
    enabled_only: bool = Query(False, description="Only show enabled indicators"),
    search: Optional[str] = Query(None, description="Search by name, id, or description"),
    source: Optional[str] = Query(None, description="Filter by data source (twelvedata, easyinsight)")
):
    """
    List all registered technical indicators.

    Returns comprehensive indicator registry with filtering options.
    Each indicator includes TwelveData/EasyInsight mappings, descriptions,
    strengths, weaknesses, and use cases.
    """
    try:
        # Start with all indicators
        if enabled_only:
            indicators = get_enabled_indicators()
        else:
            indicators = get_all_indicators()

        # Filter by category
        if category:
            try:
                cat_enum = IndicatorCategory(category.lower())
                indicators = [ind for ind in indicators if ind.category == cat_enum]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category: {category}. Valid: {[c.value for c in IndicatorCategory]}"
                )

        # Filter by source
        if source:
            source_lower = source.lower()
            if source_lower == "twelvedata":
                indicators = [ind for ind in indicators if ind.twelvedata_name]
            elif source_lower == "easyinsight":
                indicators = [ind for ind in indicators if ind.easyinsight_name]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid source: {source}. Valid: twelvedata, easyinsight"
                )

        # Search filter
        if search:
            search_results = search_indicators(search)
            search_ids = {ind.id for ind in search_results}
            indicators = [ind for ind in indicators if ind.id in search_ids]

        return {
            "total": len(indicators),
            "indicators": [ind.model_dump() for ind in indicators]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@indicators_router.get("/indicators/stats")
async def get_indicators_statistics():
    """
    Get statistics about registered indicators.

    Returns counts by category, source availability, and enabled/disabled status.
    """
    try:
        stats = get_indicator_stats()
        return {
            "total": stats["total"],
            "enabled": stats["enabled"],
            "disabled": stats["disabled"],
            "twelvedata_supported": stats["twelvedata_supported"],
            "easyinsight_supported": stats["easyinsight_supported"],
            "by_category": stats["by_category"],
            "categories": [
                {
                    "id": cat.value,
                    "name": cat.name,
                    "description": CATEGORY_DESCRIPTIONS.get(cat, ""),
                    "count": stats["by_category"].get(cat.value, 0)
                }
                for cat in IndicatorCategory
            ]
        }
    except Exception as e:
        logger.error(f"Error getting indicator stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@indicators_router.get("/indicators/categories")
async def list_categories():
    """
    List all indicator categories with descriptions.
    """
    return {
        "categories": [
            {
                "id": cat.value,
                "name": cat.name.replace("_", " ").title(),
                "description": CATEGORY_DESCRIPTIONS.get(cat, ""),
                "indicator_count": len(get_indicators_by_category(cat))
            }
            for cat in IndicatorCategory
        ]
    }


@indicators_router.get("/indicators/{indicator_id}")
async def get_indicator_details(indicator_id: str):
    """
    Get detailed information about a specific indicator.

    Returns full documentation including calculation, strengths,
    weaknesses, use cases, and default parameters.
    """
    indicator = get_indicator(indicator_id)
    if not indicator:
        raise HTTPException(
            status_code=404,
            detail=f"Indicator not found: {indicator_id}"
        )

    return {
        "indicator": indicator.model_dump(),
        "category_info": {
            "id": indicator.category.value,
            "name": indicator.category.name.replace("_", " ").title(),
            "description": CATEGORY_DESCRIPTIONS.get(indicator.category, "")
        }
    }


@indicators_router.put("/indicators/{indicator_id}/toggle")
async def toggle_indicator(indicator_id: str, enabled: bool):
    """
    Enable or disable an indicator.

    Disabled indicators won't be used in automatic analysis but
    remain in the registry for reference.
    """
    indicator = get_indicator(indicator_id)
    if not indicator:
        raise HTTPException(
            status_code=404,
            detail=f"Indicator not found: {indicator_id}"
        )

    # Update the indicator in the registry
    INDICATORS_REGISTRY[indicator_id].enabled = enabled

    return {
        "success": True,
        "indicator_id": indicator_id,
        "enabled": enabled,
        "message": f"Indicator {indicator.name} {'enabled' if enabled else 'disabled'}"
    }


@indicators_router.get("/indicators/by-category/{category}")
async def get_indicators_by_category_endpoint(category: str):
    """
    Get all indicators for a specific category.
    """
    try:
        cat_enum = IndicatorCategory(category.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}. Valid: {[c.value for c in IndicatorCategory]}"
        )

    indicators = get_indicators_by_category(cat_enum)
    return {
        "category": {
            "id": cat_enum.value,
            "name": cat_enum.name.replace("_", " ").title(),
            "description": CATEGORY_DESCRIPTIONS.get(cat_enum, "")
        },
        "total": len(indicators),
        "indicators": [ind.model_dump() for ind in indicators]
    }


@indicators_router.get("/indicators/source/twelvedata")
async def get_twelvedata_supported():
    """
    Get all indicators supported by TwelveData API.
    """
    indicators = get_twelvedata_indicators()
    return {
        "source": "twelvedata",
        "total": len(indicators),
        "indicators": [
            {
                "id": ind.id,
                "name": ind.name,
                "twelvedata_name": ind.twelvedata_name,
                "category": ind.category.value,
                "enabled": ind.enabled
            }
            for ind in indicators
        ]
    }


@indicators_router.get("/indicators/source/easyinsight")
async def get_easyinsight_supported():
    """
    Get all indicators supported by EasyInsight API.
    """
    indicators = get_easyinsight_indicators()
    return {
        "source": "easyinsight",
        "total": len(indicators),
        "indicators": [
            {
                "id": ind.id,
                "name": ind.name,
                "easyinsight_name": ind.easyinsight_name,
                "category": ind.category.value,
                "enabled": ind.enabled
            }
            for ind in indicators
        ]
    }


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
        (config_router, "/api/v1", ["⚙️ Configuration"], {
            "name": "Configuration",
            "description": "System configuration and settings management"
        }),
        (patterns_router, "/api/v1", ["🕯️ Candlestick Patterns"], {
            "name": "Patterns",
            "description": "Candlestick pattern detection - reversal, continuation, and indecision patterns with multi-timeframe scanning"
        }),
        (external_sources_router, "/api/v1", ["🌐 External Data Sources"], {
            "name": "External Sources",
            "description": "External data sources - economic calendar, sentiment, on-chain, orderbook, macro, regulatory updates"
        }),
        (backup_router, "/api/v1", ["💾 Backup & Restore"], {
            "name": "Backup",
            "description": "Backup and restore NHITS models and prediction data"
        }),
        (indicators_router, "/api/v1", ["📊 Technical Indicators"], {
            "name": "Indicators",
            "description": "Technical indicators registry with API mappings"
        }),
    ]
