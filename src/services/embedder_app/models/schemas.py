"""Pydantic schemas for Embedder Service."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingType(str, Enum):
    """Types of embeddings available."""
    TEXT = "text"
    FINANCIAL_TEXT = "financial_text"
    TIMESERIES = "timeseries"
    FEATURES = "features"


class TextEmbeddingRequest(BaseModel):
    """Request for text embedding."""
    texts: List[str] = Field(..., description="List of texts to embed")
    use_finbert: bool = Field(default=False, description="Use FinBERT for financial text")

    model_config = {"json_schema_extra": {"example": {"texts": ["Bitcoin is bullish", "Market crash expected"], "use_finbert": True}}}


class TimeSeriesEmbeddingRequest(BaseModel):
    """Request for time series embedding."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    sequence_length: int = Field(default=200, description="Number of candles for embedding")

    model_config = {"json_schema_extra": {"example": {"symbol": "BTCUSD", "timeframe": "1h", "sequence_length": 200}}}


class FeatureEmbeddingRequest(BaseModel):
    """Request for feature embedding."""
    features: List[List[float]] = Field(..., description="Feature vectors to embed")

    model_config = {"json_schema_extra": {"example": {"features": [[0.5, 0.3, 0.8, 0.2], [0.1, 0.9, 0.4, 0.6]]}}}


class EmbeddingResponse(BaseModel):
    """Response containing embeddings."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    dimension: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Model used for embedding")
    embedding_type: EmbeddingType = Field(..., description="Type of embedding")
    cached: bool = Field(default=False, description="Whether result was from cache")
    count: int = Field(..., description="Number of embeddings generated")


class SimilarityRequest(BaseModel):
    """Request for similarity computation."""
    text1: str = Field(..., description="First text")
    text2: str = Field(..., description="Second text")
    use_finbert: bool = Field(default=False, description="Use FinBERT for financial text")

    model_config = {"json_schema_extra": {"example": {"text1": "Bitcoin price rising", "text2": "BTC bullish momentum", "use_finbert": True}}}


class SimilarityResponse(BaseModel):
    """Response containing similarity score."""
    similarity: float = Field(..., description="Cosine similarity score (0-1)")
    model: str = Field(..., description="Model used")


class SimilarPatternRequest(BaseModel):
    """Request for finding similar patterns."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    top_k: int = Field(default=10, description="Number of similar patterns to return")
    lookback_days: int = Field(default=365, description="Days to search back")


class SimilarPatternResponse(BaseModel):
    """Response containing similar patterns."""
    symbol: str
    current_pattern_start: str
    similar_patterns: List[dict]
    model: str


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    entries: int = Field(..., description="Number of cached entries")
    size_mb: float = Field(..., description="Cache size in MB")
    hit_rate: Optional[float] = Field(default=None, description="Cache hit rate")


class ModelInfoResponse(BaseModel):
    """Information about loaded models."""
    text_embedder: dict
    finbert_embedder: dict
    timeseries_embedder: dict
    feature_embedder: dict
