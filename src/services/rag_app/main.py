"""
RAG Service - Retrieval Augmented Generation Microservice

Handles:
- Document Storage & Retrieval (FAISS Vector DB)
- Semantic Search
- Knowledge Base Management
- Document Chunking & Indexing
- Pattern Storage for Trading
- External Data Source Integration (8 sources)
"""

import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import sys
import os
from datetime import datetime, timedelta
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.config import settings
from src.version import VERSION
from src.services.rag_service import RAGService
from src.services.data_fetcher_proxy import DataFetcherProxy, get_data_fetcher_proxy
from src.shared.test_health_router import create_test_health_router
from src.shared.health import get_test_unhealthy_status

# Enums for API compatibility (data sources are now in Data Service)
class DataSourceTypeEnum(str, Enum):
    """Data source type enum for API compatibility."""
    ECONOMIC_CALENDAR = "economic_calendar"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    ORDERBOOK = "orderbook"
    MACRO_CORRELATION = "macro_correlation"
    HISTORICAL_PATTERN = "historical_pattern"
    TECHNICAL_LEVEL = "technical_level"
    REGULATORY = "regulatory"
    EASYINSIGHT = "easyinsight"

class DataPriorityEnum(str, Enum):
    """Priority enum for API compatibility."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Compatibility aliases
DataSourceType = DataSourceTypeEnum
DataPriority = DataPriorityEnum

# Global service instances
rag_service: Optional[RAGService] = None
data_fetcher: Optional[DataFetcherProxy] = None

# Scheduler state
scheduler_task: Optional[asyncio.Task] = None
scheduler_stats = {
    "enabled": False,
    "last_fetch": None,
    "next_fetch": None,
    "total_fetches": 0,
    "total_documents_stored": 0,
    "last_error": None,
    "symbols": [],
    "interval_minutes": 5
}

# Runtime configuration (can be changed via API and persisted)
import json
SCHEDULER_CONFIG_FILE = "/app/data/scheduler_config.json"
scheduler_runtime_config = {
    "interval_minutes": None,  # None = use settings default
    "min_priority": None,      # None = use settings default
}

def load_scheduler_config():
    """Load scheduler config from file."""
    global scheduler_runtime_config
    try:
        with open(SCHEDULER_CONFIG_FILE, 'r') as f:
            saved = json.load(f)
            scheduler_runtime_config.update(saved)
            logger.info(f"Loaded scheduler config: {scheduler_runtime_config}")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not load scheduler config: {e}")

def save_scheduler_config():
    """Save scheduler config to file."""
    try:
        import os
        os.makedirs(os.path.dirname(SCHEDULER_CONFIG_FILE), exist_ok=True)
        with open(SCHEDULER_CONFIG_FILE, 'w') as f:
            json.dump(scheduler_runtime_config, f)
        logger.info(f"Saved scheduler config: {scheduler_runtime_config}")
    except Exception as e:
        logger.error(f"Could not save scheduler config: {e}")

def get_scheduler_interval() -> int:
    """Get effective scheduler interval."""
    if scheduler_runtime_config.get("interval_minutes"):
        return scheduler_runtime_config["interval_minutes"]
    return settings.external_sources_fetch_interval_minutes

def get_scheduler_priority() -> str:
    """Get effective scheduler priority."""
    if scheduler_runtime_config.get("min_priority"):
        return scheduler_runtime_config["min_priority"]
    return settings.external_sources_min_priority

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>RAG</cyan> | <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/rag_service_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level
)

# API Tags for documentation
tags_metadata = [
    {
        "name": "Health & Info",
        "description": "Health checks und Service-Informationen",
    },
    {
        "name": "Documents",
        "description": "Dokumente speichern und verwalten",
    },
    {
        "name": "Query",
        "description": "Semantische Suche in der Wissensbasis",
    },
    {
        "name": "Stats & Management",
        "description": "Datenbank-Statistiken und Index-Verwaltung",
    },
    {
        "name": "Economic Calendar",
        "description": "Wirtschaftskalender (Fed, ECB, NFP, CPI, GDP)",
    },
    {
        "name": "On-Chain Data",
        "description": "On-Chain Metriken (Whale, Exchange Flows, DeFi)",
    },
    {
        "name": "Sentiment",
        "description": "Sentiment-Daten (Fear & Greed, VIX, Funding Rates)",
    },
    {
        "name": "Orderbook & Liquidity",
        "description": "Orderbook-Analyse (Walls, Liquidations, CVD)",
    },
    {
        "name": "Macro & Correlations",
        "description": "Makro-Daten (DXY, Bonds, Korrelationen)",
    },
    {
        "name": "Historical Patterns",
        "description": "Historische Patterns (Saisonalitaet, Drawdowns)",
    },
    {
        "name": "Technical Levels",
        "description": "Technische Levels (S/R, Fibonacci, Pivots)",
    },
    {
        "name": "Regulatory",
        "description": "Regulatorische Updates (SEC, ETFs, Enforcement)",
    },
    {
        "name": "EasyInsight",
        "description": "EasyInsight-Daten (Symbols, MT5 Logs, NHITS)",
    },
    {
        "name": "Candlestick Patterns",
        "description": "Candlestick-Muster (24 Typen, Multi-Timeframe)",
    },
    {
        "name": "Data Ingestion",
        "description": "Daten von externen Quellen abrufen und speichern",
    },
]

# Create FastAPI application
app = FastAPI(
    title="RAG Service - Trading Intelligence",
    description="""
# RAG Service für Trading Intelligence

Retrieval Augmented Generation Service mit **10 integrierten Datenquellen** für umfassende Marktanalyse.

## Features

### 📚 Vector Database (FAISS)
- Semantische Suche über alle Dokumente
- GPU-beschleunigte Embeddings
- Persistente Speicherung

### 📊 10 Externe Datenquellen

| Quelle | Beschreibung |
|--------|--------------|
| **Wirtschaftskalender** | Fed, ECB, NFP, CPI, GDP Events |
| **On-Chain Daten** | Whale Alerts, Exchange Flows, Mining, DeFi |
| **Sentiment** | Fear & Greed, Social Media, Options, VIX |
| **Orderbook** | Walls, Liquidations, CVD, Order Flow |
| **Makro & Korrelationen** | DXY, Bonds, Sektor-Rotation |
| **Historische Patterns** | Saisonalität, Drawdowns, Events |
| **Technische Levels** | S/R, Fibonacci, Pivots, VWAP, MAs |
| **Regulatorische Updates** | SEC, ETFs, Global Regulation |
| **EasyInsight** | Managed Symbols, MT5 Logs, Model Status |
| **Candlestick Patterns** | 24 Pattern-Typen, Multi-Timeframe, Confidence |

## Nutzung

### Trading Context abrufen
```bash
curl -X POST /api/v1/rag/trading-context \\
  -H "Content-Type: application/json" \\
  -d '{"symbol": "BTCUSD"}'
```

### Alle Quellen in RAG laden
```bash
curl -X POST /api/v1/rag/ingest-all-sources?symbol=BTCUSD
```

### Semantische Suche
```bash
curl "/api/v1/rag/query?query=Bitcoin%20support%20levels&symbol=BTCUSD"
```
""",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=os.getenv("ROOT_PATH", ""),
    openapi_tags=tags_metadata,
    contact={
        "name": "KI Trading Model",
        "url": "https://github.com/your-repo/ki-trading-model",
    },
    license_info={
        "name": "MIT",
    },
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test-Health-Router
test_health_router = create_test_health_router("rag")
app.include_router(test_health_router, prefix="/api/v1", tags=["Health & Info"])


# =====================================================
# Request/Response Models
# =====================================================

class DocumentRequest(BaseModel):
    """Request model for adding a document."""
    content: str = Field(..., description="Document content to store")
    document_type: str = Field(..., description="Type of document (analysis, pattern, news, strategy, custom)")
    symbol: Optional[str] = Field(None, description="Trading symbol (e.g., BTCUSD)")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str = Field(..., description="Search query")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    document_types: Optional[list[str]] = Field(None, description="Filter by document types")
    n_results: int = Field(5, ge=1, le=50, description="Number of results to return")


class ChunkRequest(BaseModel):
    """Request model for chunking and adding a large document."""
    content: str = Field(..., description="Large document content to chunk and store")
    document_type: str = Field(..., description="Type of document")
    symbol: Optional[str] = Field(None, description="Trading symbol")
    chunk_size: int = Field(500, ge=100, le=2000, description="Size of each chunk in characters")
    chunk_overlap: int = Field(50, ge=0, le=500, description="Overlap between chunks")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class PatternRequest(BaseModel):
    """Request model for storing a trading pattern."""
    symbol: str = Field(..., description="Trading symbol")
    pattern_name: str = Field(..., description="Name of the pattern")
    description: str = Field(..., description="Pattern description")
    outcome: str = Field(..., description="Pattern outcome/result")
    data_points: Optional[list[dict]] = Field(None, description="Time series data points")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    id: str
    message: str


class QueryResponse(BaseModel):
    """Response model for query operations."""
    query: str
    documents: list[str]
    count: int
    search_time_ms: float


class QueryDetailedResponse(BaseModel):
    """Detailed response model for query operations."""
    query: str
    documents: list[str]
    count: int
    details: dict


# =====================================================
# Lifecycle Events
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_service, data_fetcher

    logger.info("Starting RAG Service...")
    logger.info(f"Version: {VERSION}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"FAISS Directory: {settings.faiss_persist_directory}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Data Service URL: {settings.data_service_url}")

    # Load persistent scheduler configuration
    load_scheduler_config()

    # Initialize RAG Service
    try:
        rag_service = RAGService()
        stats = await rag_service.get_collection_stats()
        logger.info(f"RAG Service initialized - Documents: {stats['document_count']}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        raise

    # Initialize Data Fetcher Proxy (fetches data via Data Service gateway)
    try:
        data_fetcher = get_data_fetcher_proxy()
        logger.info(f"Data Service Client initialized - URL: {settings.data_service_url}")
    except Exception as e:
        logger.error(f"Failed to initialize Data Service Client: {e}")
        # Non-fatal - RAG can work without external data sources

    logger.info("RAG Service started successfully")

    # Start external sources auto-fetch scheduler
    if settings.external_sources_auto_fetch_enabled:
        await start_external_sources_scheduler()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global scheduler_task

    logger.info("Shutting down RAG Service...")

    # Stop scheduler
    if scheduler_task and not scheduler_task.done():
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            logger.info("External sources scheduler stopped")

    if rag_service:
        try:
            await rag_service.persist()
            logger.info("RAG database persisted")
        except Exception as e:
            logger.error(f"Failed to persist RAG database: {e}")

    logger.info("RAG Service stopped")


# =====================================================
# External Sources Auto-Fetch Scheduler
# =====================================================

async def fetch_external_sources_task():
    """Background task that periodically fetches external sources and stores in RAG."""
    global scheduler_stats

    use_managed_symbols = getattr(settings, 'external_sources_use_managed_symbols', True)
    scheduler_stats["enabled"] = True

    logger.info(f"External sources scheduler started - Dynamic symbols: {use_managed_symbols}")

    # Initial delay to let services start up
    await asyncio.sleep(10)

    while True:
        # Get current config values (may have been changed via API)
        interval_minutes = get_scheduler_interval()
        min_priority = get_scheduler_priority()
        scheduler_stats["interval_minutes"] = interval_minutes

        try:
            # Get symbols - either from config or dynamically from Data Service
            if use_managed_symbols and data_fetcher:
                from src.services.data_service_client import get_data_service_client
                client = get_data_service_client()
                symbols = await client.get_managed_symbols(active_only=True)
                if not symbols:
                    # Fallback to configured symbols if Data Service is unavailable
                    symbols = [s.strip() for s in settings.external_sources_symbols.split(",") if s.strip()]
                    logger.warning(f"Could not load symbols from Data Service, using config: {symbols}")
                else:
                    logger.info(f"Loaded {len(symbols)} symbols from Data Service")
            else:
                symbols = [s.strip() for s in settings.external_sources_symbols.split(",") if s.strip()]

            scheduler_stats["symbols"] = symbols

            # Fetch and store for each symbol
            total_stored = 0
            for symbol in symbols:
                try:
                    stored = await fetch_and_store_for_symbol(symbol, min_priority)
                    total_stored += stored
                    if stored > 0:
                        logger.info(f"Fetched and stored {stored} documents for {symbol}")

                    # Also fetch candlestick patterns for the symbol
                    pattern_stored = await fetch_and_store_candlestick_patterns(symbol)
                    total_stored += pattern_stored
                    if pattern_stored > 0:
                        logger.info(f"Fetched and stored {pattern_stored} candlestick patterns for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching external sources for {symbol}: {e}")

            # Also fetch market-wide data (no symbol)
            try:
                stored = await fetch_and_store_for_symbol(None, min_priority)
                total_stored += stored
                logger.info(f"Fetched and stored {stored} market-wide documents")
            except Exception as e:
                logger.error(f"Error fetching market-wide external sources: {e}")

            scheduler_stats["last_fetch"] = datetime.now().isoformat()
            scheduler_stats["total_fetches"] += 1
            scheduler_stats["total_documents_stored"] += total_stored
            scheduler_stats["last_error"] = None

            # Persist RAG database
            if rag_service and total_stored > 0:
                await rag_service.persist()
                logger.info(f"RAG database persisted after storing {total_stored} documents")

            logger.info(f"External sources fetch completed: {total_stored} documents stored in total")

        except Exception as e:
            logger.error(f"Error in external sources scheduler: {e}")
            scheduler_stats["last_error"] = str(e)

        # Schedule next fetch (re-read interval in case it changed)
        interval_minutes = get_scheduler_interval()
        scheduler_stats["next_fetch"] = (datetime.now() + timedelta(minutes=interval_minutes)).isoformat()

        # Wait for next interval
        await asyncio.sleep(interval_minutes * 60)


async def fetch_and_store_for_symbol(symbol: Optional[str], min_priority: str) -> int:
    """Fetch external sources for a symbol and store in RAG."""
    if not data_fetcher or not rag_service:
        return 0

    try:
        # Fetch from Data Service
        results = await data_fetcher.fetch_all(
            symbol=symbol,
            min_priority=min_priority
        )

        if not results:
            return 0

        # Store each result in RAG (duplicates are automatically skipped)
        stored_count = 0
        skipped_count = 0
        for result in results:
            try:
                # Convert to RAG document format
                if hasattr(result, 'to_rag_document'):
                    doc = result.to_rag_document()
                elif isinstance(result, dict):
                    doc = result
                else:
                    continue

                doc_id = await rag_service.add_custom_document(
                    content=doc.get("content", str(result)),
                    document_type=doc.get("document_type", "external_source"),
                    symbol=doc.get("symbol", symbol),
                    metadata=doc.get("metadata", {}),
                    skip_duplicates=True
                )
                if doc_id:
                    stored_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"Failed to store document: {e}")

        if skipped_count > 0:
            logger.debug(f"Skipped {skipped_count} duplicate documents for {symbol or 'market'}")

        return stored_count

    except Exception as e:
        logger.error(f"Error fetching from Data Service: {e}")
        return 0


async def fetch_and_store_candlestick_patterns(symbol: str) -> int:
    """Fetch candlestick patterns for a symbol and store in RAG."""
    if not data_fetcher or not rag_service:
        return 0

    try:
        # Fetch candlestick patterns from Data Service
        # Use H1, H4, D1 timeframes for the scheduler (most relevant for trading)
        results = await data_fetcher.fetch_candlestick_patterns(
            symbol=symbol,
            timeframes=["H1", "H4", "D1"],
            lookback_candles=20,
            min_confidence=0.5
        )

        if not results:
            return 0

        # Store each pattern in RAG (duplicates are automatically skipped)
        stored_count = 0
        skipped_count = 0
        for result in results:
            try:
                if hasattr(result, 'to_rag_document'):
                    doc = result.to_rag_document()
                elif isinstance(result, dict):
                    doc = result
                else:
                    continue

                doc_id = await rag_service.add_custom_document(
                    content=doc.get("content", str(result)),
                    document_type=doc.get("document_type", "candlestick_patterns"),
                    symbol=doc.get("symbol", symbol),
                    metadata=doc.get("metadata", {}),
                    skip_duplicates=True
                )
                if doc_id:
                    stored_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"Failed to store candlestick pattern: {e}")

        if skipped_count > 0:
            logger.debug(f"Skipped {skipped_count} duplicate candlestick patterns for {symbol}")

        return stored_count

    except Exception as e:
        logger.error(f"Error fetching candlestick patterns from Data Service: {e}")
        return 0


async def start_external_sources_scheduler():
    """Start the external sources auto-fetch scheduler."""
    global scheduler_task

    if scheduler_task and not scheduler_task.done():
        logger.warning("External sources scheduler already running")
        return

    scheduler_task = asyncio.create_task(fetch_external_sources_task())
    logger.info("External sources scheduler task created")


async def stop_external_sources_scheduler():
    """Stop the external sources auto-fetch scheduler."""
    global scheduler_task, scheduler_stats

    if scheduler_task and not scheduler_task.done():
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

    scheduler_stats["enabled"] = False
    scheduler_stats["next_fetch"] = None
    logger.info("External sources scheduler stopped")


# =====================================================
# Health & Info Endpoints
# =====================================================

@app.get("/health", tags=["Health & Info"])
async def health_check():
    """
    Health check endpoint.

    Returns service status, version, embedding model info, and document count.
    """
    # Prüfe Test-Unhealthy-Status
    test_status = get_test_unhealthy_status("rag")
    is_unhealthy = test_status.get("test_unhealthy", False)

    stats = None
    if rag_service:
        try:
            stats = await rag_service.get_collection_stats()
        except Exception as e:
            logger.error(f"Health check error: {e}")

    # Bestimme Status
    if is_unhealthy:
        status = "unhealthy"
    elif not rag_service:
        status = "unhealthy"
    else:
        status = "healthy"

    response = {
        "service": "rag",
        "status": status,
        "version": VERSION,
        "embedding_model": settings.embedding_model,
        "device": settings.device,
        "faiss_gpu": stats.get("faiss_gpu", False) if stats else False,
        "document_count": stats.get("document_count", 0) if stats else 0
    }

    # Test-Status hinzufügen wenn aktiv
    if is_unhealthy:
        response["test_unhealthy"] = test_status

    return response


@app.get("/", tags=["Health & Info"])
async def root():
    """
    Root endpoint - Service information.

    Returns basic service info with links to documentation.
    """
    return {
        "service": "RAG Service",
        "version": VERSION,
        "description": "Retrieval Augmented Generation - Vector Search & Knowledge Base",
        "docs": "/docs",
        "health": "/health"
    }


# =====================================================
# External Sources Scheduler Endpoints
# =====================================================

@app.get("/api/v1/rag/scheduler/status", tags=["Data Ingestion"])
async def get_scheduler_status():
    """
    Get the status of the external sources auto-fetch scheduler.

    Returns current scheduler state including:
    - Whether scheduler is enabled/running
    - Last and next fetch times
    - Total fetches and documents stored
    - Configured symbols and interval
    """
    use_managed_symbols = getattr(settings, 'external_sources_use_managed_symbols', True)
    return {
        "scheduler": scheduler_stats,
        "config": {
            "enabled_in_config": settings.external_sources_auto_fetch_enabled,
            "interval_minutes": settings.external_sources_fetch_interval_minutes,
            "symbols": settings.external_sources_symbols or "(dynamic from Data Service)",
            "use_managed_symbols": use_managed_symbols,
            "min_priority": settings.external_sources_min_priority
        }
    }


@app.post("/api/v1/rag/scheduler/start", tags=["Data Ingestion"])
async def start_scheduler():
    """
    Start the external sources auto-fetch scheduler.

    Begins periodic fetching of external data sources and storing in RAG.
    """
    if scheduler_stats["enabled"]:
        return {"status": "already_running", "scheduler": scheduler_stats}

    await start_external_sources_scheduler()
    return {"status": "started", "scheduler": scheduler_stats}


@app.post("/api/v1/rag/scheduler/stop", tags=["Data Ingestion"])
async def stop_scheduler():
    """
    Stop the external sources auto-fetch scheduler.

    Stops periodic fetching. Already stored documents remain in RAG.
    """
    if not scheduler_stats["enabled"]:
        return {"status": "not_running", "scheduler": scheduler_stats}

    await stop_external_sources_scheduler()
    return {"status": "stopped", "scheduler": scheduler_stats}


class SchedulerConfigRequest(BaseModel):
    """Request model for updating scheduler configuration."""
    interval_minutes: Optional[int] = Field(None, ge=1, le=1440, description="Fetch interval in minutes (1-1440)")
    min_priority: Optional[str] = Field(None, description="Minimum priority: critical, high, medium, low")


@app.get("/api/v1/rag/scheduler/config", tags=["Data Ingestion"])
async def get_scheduler_config():
    """
    Get current scheduler runtime configuration.

    Returns the currently active configuration values (runtime or defaults).
    """
    return {
        "interval_minutes": get_scheduler_interval(),
        "min_priority": get_scheduler_priority(),
        "runtime_config": scheduler_runtime_config,
        "defaults": {
            "interval_minutes": settings.external_sources_fetch_interval_minutes,
            "min_priority": settings.external_sources_min_priority
        }
    }


@app.post("/api/v1/rag/scheduler/config", tags=["Data Ingestion"])
async def update_scheduler_config(request: SchedulerConfigRequest):
    """
    Update scheduler runtime configuration.

    Changes take effect immediately and are persisted to disk.
    The scheduler will use the new values on the next fetch cycle.
    """
    global scheduler_runtime_config

    updated = {}

    if request.interval_minutes is not None:
        scheduler_runtime_config["interval_minutes"] = request.interval_minutes
        updated["interval_minutes"] = request.interval_minutes

    if request.min_priority is not None:
        valid_priorities = ["critical", "high", "medium", "low"]
        if request.min_priority.lower() not in valid_priorities:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority. Must be one of: {valid_priorities}"
            )
        scheduler_runtime_config["min_priority"] = request.min_priority.lower()
        updated["min_priority"] = request.min_priority.lower()

    # Persist to file
    save_scheduler_config()

    return {
        "message": "Configuration updated",
        "updated": updated,
        "current_config": {
            "interval_minutes": get_scheduler_interval(),
            "min_priority": get_scheduler_priority()
        }
    }


@app.post("/api/v1/rag/scheduler/fetch-now", tags=["Data Ingestion"])
async def fetch_now(background_tasks: BackgroundTasks):
    """
    Trigger an immediate fetch of all external sources.

    Fetches data from all configured symbols and stores in RAG.
    Runs in background to avoid timeout.
    """
    async def do_fetch():
        global scheduler_stats
        symbols = [s.strip() for s in settings.external_sources_symbols.split(",") if s.strip()]
        min_priority = settings.external_sources_min_priority

        total_stored = 0
        for symbol in symbols:
            try:
                stored = await fetch_and_store_for_symbol(symbol, min_priority)
                total_stored += stored
            except Exception as e:
                logger.error(f"Error fetching for {symbol}: {e}")

        # Market-wide data
        try:
            stored = await fetch_and_store_for_symbol(None, min_priority)
            total_stored += stored
        except Exception as e:
            logger.error(f"Error fetching market-wide data: {e}")

        scheduler_stats["last_fetch"] = datetime.now().isoformat()
        scheduler_stats["total_fetches"] += 1
        scheduler_stats["total_documents_stored"] += total_stored

        if rag_service and total_stored > 0:
            await rag_service.persist()

        logger.info(f"Manual fetch completed: {total_stored} documents stored")

    background_tasks.add_task(do_fetch)
    return {
        "status": "fetch_started",
        "message": "Fetch started in background",
        "symbols": settings.external_sources_symbols.split(",")
    }


# =====================================================
# Document Management Endpoints
# =====================================================

@app.get("/api/v1/rag/documents", tags=["Documents"])
async def list_documents(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sort_order: str = Query("desc", description="Sort order by date: 'asc' or 'desc'")
):
    """
    List documents in the RAG database.

    Returns documents with their metadata for browsing and filtering.
    Sorted by creation date (newest first by default).
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        documents = []
        for i, (doc, meta, doc_id) in enumerate(zip(
            rag_service._documents,
            rag_service._metadatas,
            rag_service._ids
        )):
            # Apply filters
            if symbol and meta.get("symbol") != symbol:
                continue
            if document_type and meta.get("document_type") != document_type:
                continue

            documents.append({
                "id": doc_id,
                "content": doc[:500] + "..." if len(doc) > 500 else doc,
                "document_type": meta.get("document_type", "unknown"),
                "symbol": meta.get("symbol"),
                "created_at": meta.get("timestamp"),
                "metadata": {k: v for k, v in meta.items() if k not in ["document_type", "symbol", "timestamp"]}
            })

        # Sort by created_at (timestamp)
        def get_sort_key(d):
            ts = d.get("created_at")
            if ts is None:
                return ""
            return str(ts)

        documents.sort(key=get_sort_key, reverse=(sort_order.lower() == "desc"))

        # Apply pagination
        total = len(documents)
        documents = documents[offset:offset + limit]

        return {
            "documents": documents,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/documents", response_model=DocumentResponse, tags=["Documents"])
async def add_document(request: DocumentRequest):
    """
    Add a document to the RAG database.

    Stores a document with its embedding for semantic search.
    Supports various document types: analysis, pattern, news, strategy, custom.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        doc_id = await rag_service.add_custom_document(
            content=request.content,
            document_type=request.document_type,
            symbol=request.symbol,
            metadata=request.metadata
        )
        return DocumentResponse(id=doc_id, message="Document added successfully")
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/documents/chunk", response_model=list[DocumentResponse], tags=["Documents"])
async def add_chunked_document(request: ChunkRequest):
    """
    Chunk a large document and add all chunks to the RAG database.

    Automatically splits large documents into overlapping chunks for better retrieval.
    Each chunk is stored separately with metadata linking it to the original document.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        chunks = chunk_text(request.content, request.chunk_size, request.chunk_overlap)
        results = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = request.metadata.copy() if request.metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            doc_id = await rag_service.add_custom_document(
                content=chunk,
                document_type=request.document_type,
                symbol=request.symbol,
                metadata=chunk_metadata
            )
            results.append(DocumentResponse(id=doc_id, message=f"Chunk {i+1}/{len(chunks)} added"))

        logger.info(f"Added {len(chunks)} chunks for document type: {request.document_type}")
        return results
    except Exception as e:
        logger.error(f"Error adding chunked document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/patterns", response_model=DocumentResponse, tags=["Documents"])
async def add_pattern(request: PatternRequest):
    """
    Store a trading pattern in the RAG database.

    Specialized endpoint for storing recognized chart patterns with their outcomes.
    Useful for pattern recognition and historical pattern matching.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        # Format data points if provided
        data_points_str = ""
        if request.data_points:
            data_points_str = "\n".join([
                f"{dp.get('timestamp', '-')}: O={dp.get('open', '-')} H={dp.get('high', '-')} L={dp.get('low', '-')} C={dp.get('close', '-')}"
                for dp in request.data_points
            ])

        content = f"""
Pattern: {request.pattern_name}
Symbol: {request.symbol}
Description: {request.description}

Data Points:
{data_points_str if data_points_str else 'N/A'}

Outcome: {request.outcome}
"""
        doc_id = await rag_service.add_custom_document(
            content=content,
            document_type="pattern",
            symbol=request.symbol,
            metadata={
                "pattern_name": request.pattern_name,
                "outcome": request.outcome
            }
        )
        return DocumentResponse(id=doc_id, message="Pattern stored successfully")
    except Exception as e:
        logger.error(f"Error storing pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/rag/documents", tags=["Documents"])
async def delete_documents(
    symbol: Optional[str] = Query(None, description="Delete documents for this symbol")
):
    """
    Delete documents from the RAG database.

    Delete all documents for a specific symbol. Use with caution.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        deleted_count = await rag_service.delete_documents(symbol=symbol)
        return {
            "message": f"Deleted {deleted_count} documents",
            "symbol": symbol,
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Query Endpoints
# =====================================================

@app.get("/api/v1/rag/query", tags=["Query"])
async def query_documents(
    query: str = Query(..., description="Search query", example="Bitcoin support levels"),
    symbol: Optional[str] = Query(None, description="Filter by symbol", example="BTCUSD"),
    document_types: Optional[str] = Query(None, description="Comma-separated document types", example="analysis,pattern"),
    n_results: int = Query(5, ge=1, le=50, description="Number of results")
):
    """
    Semantic search in the RAG database (GET).

    Finds documents most relevant to the query using vector similarity search.
    Optionally filter by symbol and document types.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        doc_types = document_types.split(",") if document_types else None

        import time
        start = time.time()
        documents = await rag_service.query_relevant_context(
            query=query,
            symbol=symbol,
            n_results=n_results,
            document_types=doc_types
        )
        search_time = (time.time() - start) * 1000

        return QueryResponse(
            query=query,
            documents=documents,
            count=len(documents),
            search_time_ms=round(search_time, 2)
        )
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/query", response_model=QueryResponse, tags=["Query"])
async def query_documents_post(request: QueryRequest):
    """
    Semantic search in the RAG database (POST).

    Same as GET /query but accepts JSON body for complex queries.
    Useful when query text is long or contains special characters.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        import time
        start = time.time()
        documents = await rag_service.query_relevant_context(
            query=request.query,
            symbol=request.symbol,
            n_results=request.n_results,
            document_types=request.document_types
        )
        search_time = (time.time() - start) * 1000

        return QueryResponse(
            query=request.query,
            documents=documents,
            count=len(documents),
            search_time_ms=round(search_time, 2)
        )
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/query/detailed", response_model=QueryDetailedResponse, tags=["Query"])
async def query_documents_detailed(request: QueryRequest):
    """
    Detailed semantic search with metadata.

    Returns query results with additional details:
    - Similarity scores
    - Embedding timing
    - Search timing
    - Document metadata
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        documents, details = await rag_service.query_relevant_context_with_details(
            query=request.query,
            symbol=request.symbol,
            n_results=request.n_results,
            document_types=request.document_types
        )

        return QueryDetailedResponse(
            query=request.query,
            documents=documents,
            count=len(documents),
            details=details
        )
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Stats & Management Endpoints
# =====================================================

@app.get("/api/v1/rag/stats", tags=["Stats & Management"])
async def get_stats():
    """
    Get RAG database statistics.

    Returns:
    - Total document count
    - Documents by type
    - Documents by symbol
    - Embedding model info
    - GPU/CPU status
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        stats = await rag_service.get_collection_stats()

        # Count documents by type
        type_counts = {}
        symbol_counts = {}
        for meta in rag_service._metadatas:
            doc_type = meta.get("document_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            symbol = meta.get("symbol")
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return {
            **stats,
            "documents_by_type": type_counts,
            "documents_by_symbol": symbol_counts,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": rag_service._dimension
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/persist", tags=["Stats & Management"])
async def persist_database():
    """
    Persist the RAG database to disk.

    Saves the FAISS index and metadata to the configured persist directory.
    Called automatically on shutdown, but can be triggered manually.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        await rag_service.persist()
        return {"message": "Database persisted successfully"}
    except Exception as e:
        logger.error(f"Error persisting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/rebuild", tags=["Stats & Management"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Rebuild the FAISS index in background.

    Useful after many deletions to reclaim space and optimize performance.
    Re-generates all embeddings and rebuilds the index from scratch.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    async def do_rebuild():
        try:
            # Force index rebuild by regenerating all embeddings
            logger.info("Starting index rebuild...")
            documents = rag_service._documents.copy()
            metadatas = rag_service._metadatas.copy()
            ids = rag_service._ids.copy()

            # Clear and rebuild
            import faiss
            rag_service._index = faiss.IndexFlatL2(rag_service._dimension)
            rag_service._documents = []
            rag_service._metadatas = []
            rag_service._ids = []

            # Re-add all documents
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                embedding = rag_service._generate_embedding(doc)
                import numpy as np
                rag_service._index.add(np.array([embedding]))
                rag_service._documents.append(doc)
                rag_service._metadatas.append(meta)
                rag_service._ids.append(doc_id)

            await rag_service.persist()
            logger.info(f"Index rebuild complete. {len(documents)} documents reindexed.")
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")

    background_tasks.add_task(do_rebuild)
    return {"message": "Index rebuild started in background"}


# =====================================================
# External Data Sources Endpoints
# =====================================================

class FetchDataRequest(BaseModel):
    """Request model for fetching external data."""
    symbol: Optional[str] = Field(None, description="Trading symbol")
    source_types: Optional[list[DataSourceTypeEnum]] = Field(
        None, description="Specific source types to fetch (None = all)"
    )
    min_priority: str = Field("low", description="Minimum priority: critical, high, medium, low")
    store_in_rag: bool = Field(False, description="Whether to store fetched data in RAG database")


class TradingContextRequest(BaseModel):
    """Request model for fetching comprehensive trading context."""
    symbol: str = Field(..., description="Trading symbol")
    include_types: Optional[list[str]] = Field(
        None,
        description="Data types to include: economic, onchain, sentiment, orderbook, macro, patterns, levels, regulatory, easyinsight"
    )


@app.get("/api/v1/rag/sources", tags=["Data Ingestion"])
async def get_available_sources():
    """
    List available external data sources.

    Returns all 8 data sources with their types and descriptions.
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    return {
        "sources": data_fetcher.get_available_sources(),
        "count": len(data_fetcher.get_available_sources())
    }


@app.post("/api/v1/rag/fetch", tags=["Data Ingestion"])
async def fetch_external_data(request: FetchDataRequest, background_tasks: BackgroundTasks):
    """
    Fetch data from external sources.

    Fetches data from specified sources (or all if not specified).
    Optionally stores results directly in the RAG database.

    **Priority Levels:**
    - `critical`: Immediate market impact (Fed decision, major hack)
    - `high`: Strong influence (CPI release, whale movement)
    - `medium`: Notable impact (earnings, sentiment shift)
    - `low`: Background context (minor news)
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        # Convert enum to DataSourceType
        source_types = None
        if request.source_types:
            source_types = [DataSourceType(st.value) for st in request.source_types]

        # Convert priority string to enum
        priority_map = {
            "critical": DataPriority.CRITICAL,
            "high": DataPriority.HIGH,
            "medium": DataPriority.MEDIUM,
            "low": DataPriority.LOW
        }
        min_priority = priority_map.get(request.min_priority.lower(), DataPriority.LOW)

        # Fetch data
        results = await data_fetcher.fetch_all(
            symbol=request.symbol,
            source_types=source_types,
            min_priority=min_priority
        )

        # Optionally store in RAG
        stored_count = 0
        if request.store_in_rag and rag_service:
            for result in results:
                doc = result.to_rag_document()
                await rag_service.add_custom_document(
                    content=doc["content"],
                    document_type=doc["document_type"],
                    symbol=doc.get("symbol"),
                    metadata=doc.get("metadata")
                )
                stored_count += 1

            # Persist in background
            background_tasks.add_task(rag_service.persist)

        return {
            "success": True,
            "results_count": len(results),
            "stored_in_rag": stored_count,
            "symbol": request.symbol,
            "sources_queried": len(source_types) if source_types else 8,
            "results": [
                {
                    "source": r.source_type.value,
                    "priority": r.priority.value,
                    "content_length": len(r.content),
                    "content_preview": r.content[:300] + "..." if len(r.content) > 300 else r.content
                }
                for r in results[:20]  # Limit preview to 20 results
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching external data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/trading-context", tags=["Data Ingestion"])
async def get_trading_context(request: TradingContextRequest):
    """
    Get comprehensive trading context for a symbol.

    Fetches and organizes data from ALL sources to provide a complete market overview.

    **Returns organized data including:**
    - Critical events and high priority items
    - Summary of all data sources
    - Trading-relevant insights
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        context = await data_fetcher.fetch_trading_context(
            symbol=request.symbol,
            include_types=request.include_types
        )
        return context
    except Exception as e:
        logger.error(f"Error fetching trading context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/economic-calendar", tags=["Economic Calendar"])
async def get_economic_calendar(
    symbol: Optional[str] = Query(None, description="Filter by symbol relevance"),
    days_ahead: int = Query(7, ge=1, le=30, description="Days to look ahead"),
    days_back: int = Query(1, ge=0, le=7, description="Days to look back")
):
    """
    Get upcoming economic calendar events.

    Returns Fed/ECB/BOJ decisions, NFP, CPI, GDP releases, and other market-moving events.

    **Events include:**
    - Central bank decisions (FOMC, ECB, BOJ, BOE)
    - Employment data (NFP, Unemployment)
    - Inflation data (CPI, PPI, PCE)
    - GDP releases
    - PMI data
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_economic_calendar(
            symbol=symbol,
            days_ahead=days_ahead,
            days_back=days_back
        )
        return {
            "events_count": len(results),
            "symbol": symbol,
            "events": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/onchain/{symbol}", tags=["On-Chain Data"])
async def get_onchain_data(
    symbol: str,
    include_whale_alerts: bool = Query(True),
    include_exchange_flows: bool = Query(True),
    include_mining: bool = Query(True),
    include_defi: bool = Query(True)
):
    """
    Get on-chain data for a cryptocurrency.

    **Metrics include:**
    - Whale alerts (large transactions)
    - Exchange inflows/outflows
    - Mining data (hashrate, difficulty)
    - DeFi TVL and protocol metrics
    - MVRV, SOPR, NVT indicators
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_onchain(
            symbol=symbol,
            include_whale_alerts=include_whale_alerts,
            include_exchange_flows=include_exchange_flows,
            include_mining=include_mining,
            include_defi=include_defi
        )
        return {
            "symbol": symbol,
            "metrics_count": len(results),
            "metrics": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching on-chain data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/sentiment", tags=["Sentiment"])
async def get_sentiment_data(
    symbol: Optional[str] = Query(None, description="Trading symbol"),
    include_fear_greed: bool = Query(True),
    include_social: bool = Query(True),
    include_options: bool = Query(True),
    include_volatility: bool = Query(True)
):
    """
    Get market sentiment data.

    **Indicators include:**
    - Fear & Greed Index (crypto)
    - Social media sentiment (Twitter, Reddit)
    - Options Put/Call ratio
    - VIX volatility index
    - Funding rates (perpetuals)
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_sentiment(
            symbol=symbol,
            include_fear_greed=include_fear_greed,
            include_social=include_social,
            include_options=include_options,
            include_volatility=include_volatility
        )
        return {
            "symbol": symbol,
            "indicators_count": len(results),
            "indicators": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching sentiment data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/orderbook/{symbol}", tags=["Orderbook & Liquidity"])
async def get_orderbook_data(
    symbol: str,
    depth: int = Query(50, ge=10, le=200),
    include_liquidations: bool = Query(True),
    include_cvd: bool = Query(True)
):
    """
    Get orderbook and liquidity data.

    **Analysis includes:**
    - Bid/Ask wall detection
    - Liquidation levels (longs/shorts)
    - Open interest analysis
    - Cumulative Volume Delta (CVD)
    - Order flow imbalance
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_orderbook(
            symbol=symbol,
            depth=depth,
            include_liquidations=include_liquidations,
            include_cvd=include_cvd
        )
        return {
            "symbol": symbol,
            "analysis_count": len(results),
            "analysis": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching orderbook data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/macro", tags=["Macro & Correlations"])
async def get_macro_data(
    symbol: Optional[str] = Query(None, description="Trading symbol for context"),
    include_dxy: bool = Query(True),
    include_bonds: bool = Query(True),
    include_correlations: bool = Query(True),
    include_sectors: bool = Query(True)
):
    """
    Get macro and correlation data.

    **Metrics include:**
    - DXY (Dollar Index) analysis
    - Bond yields (2Y, 10Y, 30Y)
    - Cross-asset correlations
    - Sector rotation signals
    - Global liquidity indicators
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_macro(
            symbol=symbol,
            include_dxy=include_dxy,
            include_bonds=include_bonds,
            include_correlations=include_correlations,
            include_sectors=include_sectors
        )
        return {
            "symbol": symbol,
            "metrics_count": len(results),
            "metrics": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching macro data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/historical-patterns", tags=["Historical Patterns"])
async def get_historical_patterns(
    symbol: Optional[str] = Query(None, description="Trading symbol"),
    include_seasonality: bool = Query(True),
    include_drawdowns: bool = Query(True),
    include_events: bool = Query(True),
    include_comparable: bool = Query(True)
):
    """
    Get historical pattern analysis.

    **Patterns include:**
    - Monthly/Weekly seasonality
    - Historical drawdowns analysis
    - Event-based returns (halvings, elections)
    - Comparable market periods
    - Cycle analysis
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_historical_patterns(
            symbol=symbol,
            include_seasonality=include_seasonality,
            include_drawdowns=include_drawdowns,
            include_events=include_events,
            include_comparable=include_comparable
        )
        return {
            "symbol": symbol,
            "patterns_count": len(results),
            "patterns": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching historical patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/technical-levels/{symbol}", tags=["Technical Levels"])
async def get_technical_levels(
    symbol: str,
    include_sr: bool = Query(True, description="Support/Resistance"),
    include_fib: bool = Query(True, description="Fibonacci levels"),
    include_pivots: bool = Query(True, description="Pivot points"),
    include_vwap: bool = Query(True, description="VWAP levels"),
    include_ma: bool = Query(True, description="Moving averages")
):
    """
    Get technical price levels.

    **Levels include:**
    - Support/Resistance zones
    - Fibonacci retracements & extensions
    - Daily/Weekly pivot points
    - VWAP and anchored VWAP
    - Moving averages (SMA, EMA)
    - Volume profile POC/VAH/VAL
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_technical_levels(
            symbol=symbol,
            include_sr=include_sr,
            include_fib=include_fib,
            include_pivots=include_pivots,
            include_vwap=include_vwap,
            include_ma=include_ma
        )
        return {
            "symbol": symbol,
            "levels_count": len(results),
            "levels": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching technical levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/regulatory", tags=["Regulatory"])
async def get_regulatory_updates(
    symbol: Optional[str] = Query(None, description="Trading symbol"),
    include_sec: bool = Query(True),
    include_etf: bool = Query(True),
    include_global: bool = Query(True),
    include_enforcement: bool = Query(True)
):
    """
    Get regulatory updates and news.

    **Updates include:**
    - SEC/CFTC decisions and filings
    - ETF approvals and flow data
    - Global regulation (MiCA, etc.)
    - Enforcement actions
    - Stablecoin regulation
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_regulatory(
            symbol=symbol,
            include_sec=include_sec,
            include_etf=include_etf,
            include_global=include_global,
            include_enforcement=include_enforcement
        )
        return {
            "symbol": symbol,
            "updates_count": len(results),
            "updates": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching regulatory updates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/easyinsight", tags=["EasyInsight"])
async def get_easyinsight_data(
    symbol: Optional[str] = Query(None, description="Filter by specific symbol"),
    include_symbols: bool = Query(True, description="Include managed symbols"),
    include_stats: bool = Query(True, description="Include statistics"),
    include_mt5_logs: bool = Query(True, description="Include MT5 trading logs")
):
    """
    Get EasyInsight managed symbols and MT5 logs.

    **Data includes:**
    - Managed trading symbols and their configurations
    - Symbol statistics (by category, status, model availability)
    - MT5 trading logs and history
    - NHITS model training status
    - TimescaleDB data availability
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_easyinsight(
            symbol=symbol,
            include_symbols=include_symbols,
            include_stats=include_stats,
            include_mt5_logs=include_mt5_logs
        )
        return {
            "symbol": symbol,
            "data_count": len(results),
            "data": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching EasyInsight data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/candlestick-patterns/types", tags=["Candlestick Patterns"])
async def get_candlestick_pattern_types():
    """
    Get all available candlestick pattern types and their descriptions.

    Returns a list of 24 pattern types with:
    - Pattern name and category (reversal, continuation, indecision)
    - Direction (bullish, bearish, neutral)
    - Description and trading implications
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        return await data_fetcher.fetch_candlestick_pattern_types()
    except Exception as e:
        logger.error(f"Error fetching pattern types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/candlestick-patterns/summary/{symbol}", tags=["Candlestick Patterns"])
async def get_candlestick_pattern_summary(symbol: str):
    """
    Get a simplified candlestick pattern summary for a symbol.

    Returns aggregated pattern counts and most significant patterns
    across all timeframes for quick overview.
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        return await data_fetcher.fetch_candlestick_pattern_summary(symbol)
    except Exception as e:
        logger.error(f"Error fetching pattern summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rag/candlestick-patterns/history", tags=["Candlestick Patterns"])
async def get_candlestick_pattern_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    direction: Optional[str] = Query(None, description="Filter by direction: bullish, bearish"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    store_in_rag: bool = Query(False, description="Store patterns in RAG database")
):
    """
    Query historical candlestick pattern detections.

    Returns patterns detected by the Data Service's auto-scan feature.
    Optionally store the results in the RAG database for semantic search.

    **Filters:**
    - Symbol: Specific trading pair
    - Pattern type: e.g., "hammer", "engulfing", "doji"
    - Direction: bullish or bearish
    - Confidence: 0.0 to 1.0
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        results = await data_fetcher.fetch_candlestick_pattern_history(
            symbol=symbol,
            pattern_type=pattern_type,
            direction=direction,
            min_confidence=min_confidence,
            limit=limit
        )

        # Optionally store in RAG
        stored_count = 0
        if store_in_rag and rag_service and results:
            for result in results:
                doc = result.to_rag_document()
                doc_id = await rag_service.add_custom_document(
                    content=doc["content"],
                    document_type=doc["document_type"],
                    symbol=doc.get("symbol"),
                    metadata=doc.get("metadata"),
                    skip_duplicates=True
                )
                if doc_id:
                    stored_count += 1

            if stored_count > 0:
                await rag_service.persist()

        return {
            "filters": {
                "symbol": symbol,
                "pattern_type": pattern_type,
                "direction": direction,
                "min_confidence": min_confidence
            },
            "patterns_count": len(results),
            "stored_in_rag": stored_count,
            "patterns": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching pattern history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/candlestick-patterns/ingest", tags=["Candlestick Patterns"])
async def ingest_candlestick_patterns(
    symbol: Optional[str] = Query(None, description="Symbol to fetch patterns for (None = all managed symbols)"),
    timeframes: Optional[str] = Query("H1,H4,D1", description="Comma-separated timeframes"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence"),
    background_tasks: BackgroundTasks = None
):
    """
    Fetch candlestick patterns for symbol(s) and ingest into RAG database.

    If no symbol is specified, fetches patterns for all managed symbols
    from the Data Service.

    **Process:**
    1. Fetches candlestick patterns from Data Service
    2. Converts to RAG document format
    3. Stores with embeddings (duplicates skipped)
    4. Persists database
    """
    if not data_fetcher or not rag_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    async def do_ingest():
        try:
            tf_list = timeframes.split(",") if timeframes else ["H1", "H4", "D1"]

            # Get symbols to process
            if symbol:
                symbols = [symbol]
            else:
                from src.services.data_service_client import get_data_service_client
                client = get_data_service_client()
                symbols = await client.get_managed_symbols(active_only=True)
                if not symbols:
                    symbols = ["BTCUSD", "ETHUSD"]  # Fallback

            total_stored = 0
            for sym in symbols:
                try:
                    results = await data_fetcher.fetch_candlestick_patterns(
                        symbol=sym,
                        timeframes=tf_list,
                        min_confidence=min_confidence
                    )

                    for result in results:
                        doc = result.to_rag_document()
                        doc_id = await rag_service.add_custom_document(
                            content=doc["content"],
                            document_type=doc["document_type"],
                            symbol=doc.get("symbol"),
                            metadata=doc.get("metadata"),
                            skip_duplicates=True
                        )
                        if doc_id:
                            total_stored += 1

                except Exception as e:
                    logger.error(f"Error ingesting patterns for {sym}: {e}")

            if total_stored > 0:
                await rag_service.persist()

            logger.info(f"Ingested {total_stored} candlestick patterns for {len(symbols)} symbols")
        except Exception as e:
            logger.error(f"Error in candlestick pattern ingestion: {e}")

    if background_tasks:
        background_tasks.add_task(do_ingest)
        return {
            "message": "Candlestick pattern ingestion started in background",
            "symbol": symbol or "all managed symbols",
            "timeframes": timeframes
        }
    else:
        await do_ingest()
        return {
            "message": "Candlestick pattern ingestion completed",
            "symbol": symbol or "all managed symbols",
            "timeframes": timeframes
        }


@app.get("/api/v1/rag/candlestick-patterns/{symbol}", tags=["Candlestick Patterns"])
async def get_candlestick_patterns(
    symbol: str,
    timeframes: Optional[str] = Query(None, description="Comma-separated timeframes: M5,M15,H1,H4,D1"),
    lookback_candles: int = Query(20, ge=5, le=100, description="Number of candles to analyze"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    store_in_rag: bool = Query(False, description="Store detected patterns in RAG database")
):
    """
    Detect candlestick patterns for a symbol across multiple timeframes.

    **Supported Pattern Types (24 total):**

    **Reversal Patterns:**
    - Hammer, Inverted Hammer, Shooting Star, Hanging Man
    - Doji (Standard, Dragonfly, Gravestone)
    - Bullish/Bearish Engulfing
    - Morning Star, Evening Star
    - Piercing Line, Dark Cloud Cover

    **Continuation Patterns:**
    - Three White Soldiers, Three Black Crows
    - Rising/Falling Three Methods

    **Indecision Patterns:**
    - Spinning Top, Bullish/Bearish Harami, Harami Cross

    **Timeframes:** M5, M15, H1, H4, D1
    """
    if not data_fetcher:
        raise HTTPException(status_code=503, detail="Data fetcher not initialized")

    try:
        tf_list = timeframes.split(",") if timeframes else None
        results = await data_fetcher.fetch_candlestick_patterns(
            symbol=symbol,
            timeframes=tf_list,
            lookback_candles=lookback_candles,
            min_confidence=min_confidence
        )

        # Optionally store in RAG
        stored_count = 0
        if store_in_rag and rag_service and results:
            for result in results:
                doc = result.to_rag_document()
                doc_id = await rag_service.add_custom_document(
                    content=doc["content"],
                    document_type=doc["document_type"],
                    symbol=doc.get("symbol"),
                    metadata=doc.get("metadata"),
                    skip_duplicates=True
                )
                if doc_id:
                    stored_count += 1

            if stored_count > 0:
                await rag_service.persist()

        return {
            "symbol": symbol,
            "timeframes": tf_list or ["M5", "M15", "H1", "H4", "D1"],
            "patterns_count": len(results),
            "stored_in_rag": stored_count,
            "patterns": [r.to_rag_document() for r in results]
        }
    except Exception as e:
        logger.error(f"Error fetching candlestick patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/ingest-all-sources", tags=["Data Ingestion"])
async def ingest_all_sources(
    symbol: Optional[str] = Query(None, description="Symbol to fetch for"),
    background_tasks: BackgroundTasks = None
):
    """
    Fetch data from all sources and ingest into RAG database.

    This is useful for populating the RAG database with fresh market data.
    Runs in background if background_tasks is provided.

    **Process:**
    1. Fetches data from all 9 external sources
    2. Converts to RAG document format
    3. Stores with embeddings in FAISS index
    4. Persists database to disk
    """
    if not data_fetcher or not rag_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    async def do_ingest():
        try:
            logger.info(f"Starting full data ingestion for {symbol or 'all symbols'}")

            documents = await data_fetcher.create_rag_documents_batch(symbol=symbol)

            ingested = 0
            for doc in documents:
                await rag_service.add_custom_document(
                    content=doc["content"],
                    document_type=doc["document_type"],
                    symbol=doc.get("symbol"),
                    metadata=doc.get("metadata")
                )
                ingested += 1

            await rag_service.persist()
            logger.info(f"Ingested {ingested} documents from all sources")
        except Exception as e:
            logger.error(f"Error in full ingestion: {e}")

    if background_tasks:
        background_tasks.add_task(do_ingest)
        return {
            "message": "Full ingestion started in background",
            "symbol": symbol
        }
    else:
        await do_ingest()
        return {
            "message": "Full ingestion completed",
            "symbol": symbol
        }


# =====================================================
# Helper Functions
# =====================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars
            search_start = max(end - 100, start)
            for boundary in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                last_boundary = text.rfind(boundary, search_start, end)
                if last_boundary > start:
                    end = last_boundary + len(boundary)
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]  # Filter empty chunks


# =====================================================
# Main Entry Point
# =====================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3003"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=settings.log_level.lower()
    )
