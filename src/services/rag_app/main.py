"""
RAG Service - Retrieval Augmented Generation Microservice

Handles:
- Document Storage & Retrieval (FAISS Vector DB)
- Semantic Search
- Knowledge Base Management
- Document Chunking & Indexing
- Pattern Storage for Trading
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.config import settings
from src.version import VERSION
from src.services.rag_service import RAGService

# Global service instance
rag_service: Optional[RAGService] = None

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

# Create FastAPI application
app = FastAPI(
    title="RAG Service",
    description="Retrieval Augmented Generation Service - Vector Search & Knowledge Base",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=os.getenv("ROOT_PATH", "")
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    global rag_service

    logger.info("Starting RAG Service...")
    logger.info(f"Version: {VERSION}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"FAISS Directory: {settings.faiss_persist_directory}")
    logger.info(f"Device: {settings.device}")

    # Initialize RAG Service
    try:
        rag_service = RAGService()
        stats = await rag_service.get_collection_stats()
        logger.info(f"RAG Service initialized - Documents: {stats['document_count']}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        raise

    logger.info("RAG Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG Service...")

    if rag_service:
        try:
            await rag_service.persist()
            logger.info("RAG database persisted")
        except Exception as e:
            logger.error(f"Failed to persist RAG database: {e}")

    logger.info("RAG Service stopped")


# =====================================================
# Health & Info Endpoints
# =====================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = None
    if rag_service:
        try:
            stats = await rag_service.get_collection_stats()
        except Exception as e:
            logger.error(f"Health check error: {e}")

    return {
        "service": "rag",
        "status": "healthy" if rag_service else "unhealthy",
        "version": VERSION,
        "embedding_model": settings.embedding_model,
        "device": settings.device,
        "faiss_gpu": stats.get("faiss_gpu", False) if stats else False,
        "document_count": stats.get("document_count", 0) if stats else 0
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "RAG Service",
        "version": VERSION,
        "description": "Retrieval Augmented Generation - Vector Search & Knowledge Base",
        "docs": "/docs",
        "health": "/health"
    }


# =====================================================
# Document Management Endpoints
# =====================================================

@app.post("/api/v1/rag/documents", response_model=DocumentResponse)
async def add_document(request: DocumentRequest):
    """Add a document to the RAG database."""
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


@app.post("/api/v1/rag/documents/chunk", response_model=list[DocumentResponse])
async def add_chunked_document(request: ChunkRequest):
    """Chunk a large document and add all chunks to the RAG database."""
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


@app.post("/api/v1/rag/patterns", response_model=DocumentResponse)
async def add_pattern(request: PatternRequest):
    """Store a trading pattern in the RAG database."""
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


@app.delete("/api/v1/rag/documents")
async def delete_documents(
    symbol: Optional[str] = Query(None, description="Delete documents for this symbol")
):
    """Delete documents from the RAG database."""
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

@app.get("/api/v1/rag/query")
async def query_documents(
    query: str = Query(..., description="Search query"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    document_types: Optional[str] = Query(None, description="Comma-separated document types"),
    n_results: int = Query(5, ge=1, le=50, description="Number of results")
):
    """Query relevant documents from the RAG database."""
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


@app.post("/api/v1/rag/query", response_model=QueryResponse)
async def query_documents_post(request: QueryRequest):
    """Query relevant documents from the RAG database (POST version)."""
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


@app.post("/api/v1/rag/query/detailed", response_model=QueryDetailedResponse)
async def query_documents_detailed(request: QueryRequest):
    """Query documents with detailed metadata and timing information."""
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

@app.get("/api/v1/rag/stats")
async def get_stats():
    """Get RAG database statistics."""
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


@app.post("/api/v1/rag/persist")
async def persist_database():
    """Persist the RAG database to disk."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        await rag_service.persist()
        return {"message": "Database persisted successfully"}
    except Exception as e:
        logger.error(f"Error persisting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild the FAISS index (useful after many deletions)."""
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
    port = int(os.getenv("PORT", "3004"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=settings.log_level.lower()
    )
