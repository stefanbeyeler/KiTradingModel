"""Text embedder using Sentence Transformers."""

import numpy as np
from typing import Union, List, Optional
from loguru import logger

from .base_embedder import BaseEmbedder

# Lazy import to handle missing dependencies gracefully
SentenceTransformer = None


def _load_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
        except ImportError:
            logger.warning("sentence-transformers not installed. TextEmbedder will use fallback.")
            return None
    return SentenceTransformer


class TextEmbedder(BaseEmbedder):
    """
    General-purpose text embedder using Sentence Transformers.
    Optimized for semantic similarity.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda"
    ):
        """
        Initialize the text embedder.

        Args:
            model_name: Sentence Transformer model name
            device: Device to use ('cuda' or 'cpu')
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._embedding_dim: Optional[int] = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        ST = _load_sentence_transformer()
        if ST is None:
            logger.warning("Using fallback random embeddings for text")
            self._embedding_dim = 384
            return

        try:
            logger.info(f"Loading text embedder: {self._model_name}")
            self._model = ST(self._model_name, device=self._device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            self._loaded = True
            logger.info(f"Text embedder loaded. Dimension: {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load text embedder: {e}")
            self._embedding_dim = 384

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim or 384

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        if self._model is None:
            # Fallback: random embeddings (for testing without model)
            logger.warning("Using fallback random embeddings")
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
            show_progress_bar=False
        )

        return embeddings.astype(np.float32)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.embed([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
