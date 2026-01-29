"""Time series embedding endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
import numpy as np
import httpx
from loguru import logger

from ..models.schemas import (
    TimeSeriesEmbeddingRequest,
    EmbeddingResponse,
    EmbeddingType,
    SimilarPatternRequest,
    SimilarPatternResponse,
)
from ..services.embedding_service import embedding_service
from ..config import settings
from src.config.microservices import microservices_config

router = APIRouter()

# HTTP Client for Data Service
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create HTTP client for Data Service."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def fetch_ohlcv_from_data_service(symbol: str, timeframe: str, limit: int) -> list:
    """Fetch OHLCV data from Data Service via HTTP."""
    client = await get_http_client()
    data_service_url = getattr(settings, 'data_service_url', microservices_config.data_service_url)

    url = f"{data_service_url}/api/v1/db/ohlcv/{symbol}"
    params = {"timeframe": timeframe, "limit": limit}

    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        # Handle both formats: {"data": [...]} or direct list
        data = result.get("data", result) if isinstance(result, dict) else result
        return data if isinstance(data, list) else []
    except httpx.HTTPStatusError as e:
        logger.error(f"Data Service HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error fetching data from Data Service: {e}")
        raise


@router.post("/embed", response_model=EmbeddingResponse)
async def embed_timeseries(request: TimeSeriesEmbeddingRequest):
    """
    Generate embedding for OHLCV time series.

    Automatically fetches data from Data Service.

    - **symbol**: Trading symbol (e.g., BTCUSD)
    - **timeframe**: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    - **sequence_length**: Number of candles for embedding
    """
    try:
        # Fetch OHLCV data from Data Service via HTTP
        data = await fetch_ohlcv_from_data_service(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.sequence_length
        )

        if not data or len(data) < 10:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient data for {request.symbol}"
            )

        # Convert to numpy array (seq_len, 5) for OHLCV
        # Handle None/null values for volume
        ohlcv = np.array([
            [
                d['open'],
                d['high'],
                d['low'],
                d['close'],
                d.get('volume') if d.get('volume') is not None else 0.0
            ]
            for d in data
        ], dtype=np.float32)

        # Replace any NaN values with 0
        ohlcv = np.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0)

        result = await embedding_service.embed_timeseries(ohlcv)

        # Ensure embeddings are JSON-serializable (no NaN/Inf)
        embeddings = np.nan_to_num(result.embedding, nan=0.0, posinf=0.0, neginf=0.0)

        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimension=result.dimension,
            model=result.model_name,
            embedding_type=result.embedding_type,
            cached=result.cached,
            count=1
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding timeseries for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed-raw", response_model=EmbeddingResponse)
async def embed_raw_timeseries(ohlcv_data: List[List[float]]):
    """
    Generate embedding for raw OHLCV data.

    - **ohlcv_data**: List of [open, high, low, close, volume] arrays
    """
    try:
        ohlcv = np.array(ohlcv_data, dtype=np.float32)

        if ohlcv.ndim == 2 and ohlcv.shape[1] == 5:
            # Single sequence
            pass
        elif ohlcv.ndim == 3 and ohlcv.shape[2] == 5:
            # Batch of sequences
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid OHLCV shape. Expected (seq_len, 5) or (batch, seq_len, 5)"
            )

        result = await embedding_service.embed_timeseries(ohlcv)

        return EmbeddingResponse(
            embeddings=result.embedding.tolist(),
            dimension=result.dimension,
            model=result.model_name,
            embedding_type=result.embedding_type,
            cached=result.cached,
            count=len(result.embedding)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding raw timeseries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed-batch")
async def embed_batch_timeseries(
    requests: List[TimeSeriesEmbeddingRequest]
):
    """
    Batch embedding for multiple symbols.

    Returns embeddings for all requested symbols.
    """
    try:
        results = []
        for req in requests:
            try:
                data = await fetch_ohlcv_from_data_service(
                    symbol=req.symbol,
                    timeframe=req.timeframe,
                    limit=req.sequence_length
                )

                if not data or len(data) < 10:
                    results.append({
                        "symbol": req.symbol,
                        "error": "Insufficient data"
                    })
                    continue

                ohlcv = np.array([
                    [d['open'], d['high'], d['low'], d['close'], d.get('volume', 0)]
                    for d in data
                ], dtype=np.float32)

                result = await embedding_service.embed_timeseries(ohlcv)

                results.append({
                    "symbol": req.symbol,
                    "timeframe": req.timeframe,
                    "embedding": result.embedding[0].tolist(),
                    "dimension": result.dimension,
                    "cached": result.cached
                })
            except Exception as e:
                results.append({
                    "symbol": req.symbol,
                    "error": str(e)
                })

        return {
            "results": results,
            "total": len(requests),
            "successful": len([r for r in results if "embedding" in r])
        }
    except Exception as e:
        logger.error(f"Error in batch embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/find-similar", response_model=SimilarPatternResponse)
async def find_similar_patterns(request: SimilarPatternRequest):
    """
    Find similar historical patterns.

    Uses time series embeddings to search for patterns similar to current price action.

    - **symbol**: Trading symbol
    - **timeframe**: Timeframe
    - **top_k**: Number of similar patterns to return
    - **lookback_days**: How far back to search
    """
    try:
        # Get current pattern (last 50 candles)
        current_data = await fetch_ohlcv_from_data_service(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=50
        )

        if not current_data or len(current_data) < 50:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient current data for {request.symbol}"
            )

        current_ohlcv = np.array([
            [d['open'], d['high'], d['low'], d['close'], d.get('volume', 0)]
            for d in current_data
        ], dtype=np.float32)

        current_result = await embedding_service.embed_timeseries(current_ohlcv)
        current_embedding = current_result.embedding[0]

        # Get historical data
        historical_data = await fetch_ohlcv_from_data_service(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.lookback_days * 24  # Approximation for hourly
        )

        if not historical_data or len(historical_data) < 100:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient historical data for {request.symbol}"
            )

        # Slide window and compute embeddings
        window_size = 50
        step = 10
        similarities = []

        for i in range(0, len(historical_data) - window_size - 50, step):
            window = historical_data[i:i + window_size]
            window_ohlcv = np.array([
                [d['open'], d['high'], d['low'], d['close'], d.get('volume', 0)]
                for d in window
            ], dtype=np.float32)

            result = await embedding_service.embed_timeseries(window_ohlcv)
            similarity = await embedding_service.compute_similarity(
                current_embedding,
                result.embedding[0]
            )

            start_time = window[0].get('timestamp', window[0].get('time', ''))

            similarities.append({
                "start_index": i,
                "start_time": str(start_time),
                "similarity": round(similarity, 4)
            })

        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_patterns = similarities[:request.top_k]

        return SimilarPatternResponse(
            symbol=request.symbol,
            current_pattern_start=str(current_data[0].get('timestamp', '')),
            similar_patterns=top_patterns,
            model=current_result.model_name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
