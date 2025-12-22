"""Main embedding service."""

from typing import Union, List, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger

from ..models.text_embedder import TextEmbedder
from ..models.finbert_embedder import FinBERTEmbedder
from ..models.ts2vec_embedder import TimeSeriesEmbedder
from ..models.feature_autoencoder import FeatureEmbedder
from ..models.schemas import EmbeddingType
from .cache_service import CacheService, cache_service


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    embedding: np.ndarray
    embedding_type: EmbeddingType
    dimension: int
    model_name: str
    cached: bool = False


class EmbeddingService:
    """
    Central service for all embedding operations.

    Features:
    - Multi-modal embeddings (Text, TimeSeries, Features)
    - Automatic caching
    - Batch processing
    - Model hot-swapping
    """

    def __init__(
        self,
        device: str = "cuda",
        cache_enabled: bool = True,
        cache_service_instance: Optional[CacheService] = None
    ):
        """
        Initialize the embedding service.

        Args:
            device: Device to use ('cuda' or 'cpu')
            cache_enabled: Whether to enable caching
            cache_service_instance: Optional custom cache service
        """
        self.device = device
        self.cache_enabled = cache_enabled
        self._cache = cache_service_instance or cache_service

        # Lazy-loaded embedders
        self._text_embedder: Optional[TextEmbedder] = None
        self._finbert_embedder: Optional[FinBERTEmbedder] = None
        self._ts_embedder: Optional[TimeSeriesEmbedder] = None
        self._feature_embedder: Optional[FeatureEmbedder] = None

        logger.info(f"EmbeddingService initialized (device={device}, cache={cache_enabled})")

    @property
    def text_embedder(self) -> TextEmbedder:
        """Get or create text embedder."""
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(device=self.device)
        return self._text_embedder

    @property
    def finbert_embedder(self) -> FinBERTEmbedder:
        """Get or create FinBERT embedder."""
        if self._finbert_embedder is None:
            self._finbert_embedder = FinBERTEmbedder(device=self.device)
        return self._finbert_embedder

    @property
    def ts_embedder(self) -> TimeSeriesEmbedder:
        """Get or create time series embedder."""
        if self._ts_embedder is None:
            self._ts_embedder = TimeSeriesEmbedder(device=self.device)
        return self._ts_embedder

    @property
    def feature_embedder(self) -> FeatureEmbedder:
        """Get or create feature embedder."""
        if self._feature_embedder is None:
            self._feature_embedder = FeatureEmbedder(device=self.device)
        return self._feature_embedder

    async def embed_text(
        self,
        texts: Union[str, List[str]],
        use_finbert: bool = False
    ) -> EmbeddingResult:
        """
        Generate text embeddings.

        Args:
            texts: Text or list of texts
            use_finbert: Use FinBERT for financial text

        Returns:
            EmbeddingResult with embeddings
        """
        embedding_type = EmbeddingType.FINANCIAL_TEXT if use_finbert else EmbeddingType.TEXT
        embedder = self.finbert_embedder if use_finbert else self.text_embedder

        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        if self.cache_enabled:
            cache_key = f"{embedding_type.value}:{hash(tuple(texts))}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return EmbeddingResult(
                    embedding=cached,
                    embedding_type=embedding_type,
                    dimension=cached.shape[-1],
                    model_name=embedder.model_name,
                    cached=True
                )

        # Compute embeddings
        embeddings = embedder.embed(texts)

        # Cache result
        if self.cache_enabled:
            self._cache.set(cache_key, embeddings)

        return EmbeddingResult(
            embedding=embeddings,
            embedding_type=embedding_type,
            dimension=embeddings.shape[-1],
            model_name=embedder.model_name,
            cached=False
        )

    async def embed_timeseries(
        self,
        sequences: np.ndarray
    ) -> EmbeddingResult:
        """
        Generate time series embeddings for OHLCV data.

        Args:
            sequences: numpy array (n_sequences, seq_len, 5) for OHLCV

        Returns:
            EmbeddingResult with embeddings
        """
        # Check cache
        if self.cache_enabled:
            cache_key = f"timeseries:{hash(sequences.tobytes())}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return EmbeddingResult(
                    embedding=cached,
                    embedding_type=EmbeddingType.TIMESERIES,
                    dimension=cached.shape[-1],
                    model_name=self.ts_embedder.model_name,
                    cached=True
                )

        # Compute embeddings
        embeddings = self.ts_embedder.embed(sequences)

        # Cache result
        if self.cache_enabled:
            self._cache.set(cache_key, embeddings)

        return EmbeddingResult(
            embedding=embeddings,
            embedding_type=EmbeddingType.TIMESERIES,
            dimension=embeddings.shape[-1],
            model_name=self.ts_embedder.model_name,
            cached=False
        )

    async def embed_features(
        self,
        features: np.ndarray
    ) -> EmbeddingResult:
        """
        Compress feature vectors.

        Args:
            features: numpy array (n_samples, n_features)

        Returns:
            EmbeddingResult with embeddings
        """
        # Check cache
        if self.cache_enabled:
            cache_key = f"features:{hash(features.tobytes())}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return EmbeddingResult(
                    embedding=cached,
                    embedding_type=EmbeddingType.FEATURES,
                    dimension=cached.shape[-1],
                    model_name=self.feature_embedder.model_name,
                    cached=True
                )

        # Compute embeddings
        embeddings = self.feature_embedder.embed(features)

        # Cache result
        if self.cache_enabled:
            self._cache.set(cache_key, embeddings)

        return EmbeddingResult(
            embedding=embeddings,
            embedding_type=EmbeddingType.FEATURES,
            dimension=embeddings.shape[-1],
            model_name=self.feature_embedder.model_name,
            cached=False
        )

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Flatten if needed
        e1 = embedding1.flatten()
        e2 = embedding2.flatten()

        # Compute cosine similarity
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_model_info(self) -> dict:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        return {
            "text_embedder": {
                "model": self.text_embedder.model_name,
                "dimension": self.text_embedder.embedding_dim,
                "loaded": self.text_embedder.is_loaded()
            },
            "finbert_embedder": {
                "model": self.finbert_embedder.model_name,
                "dimension": self.finbert_embedder.embedding_dim,
                "loaded": self.finbert_embedder.is_loaded()
            },
            "timeseries_embedder": {
                "model": self.ts_embedder.model_name,
                "dimension": self.ts_embedder.embedding_dim,
                "loaded": self.ts_embedder.is_loaded()
            },
            "feature_embedder": {
                "model": self.feature_embedder.model_name,
                "dimension": self.feature_embedder.embedding_dim,
                "loaded": self.feature_embedder.is_loaded()
            }
        }

    def clear_cache(self) -> int:
        """
        Clear the embedding cache.

        Returns:
            Number of entries cleared
        """
        return self._cache.clear()

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        stats = self._cache.get_stats()
        return {
            "entries": stats.entries,
            "size_mb": round(stats.size_mb, 2),
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": round(stats.hit_rate, 4)
        }


# Singleton instance
embedding_service = EmbeddingService()
