"""FinBERT embedder for financial text."""

import numpy as np
from typing import Union, List, Optional
from loguru import logger

from .base_embedder import BaseEmbedder

# Lazy imports
torch = None
AutoTokenizer = None
AutoModel = None


def _load_transformers():
    global torch, AutoTokenizer, AutoModel
    if torch is None:
        try:
            import torch as t
            from transformers import AutoTokenizer as AT, AutoModel as AM
            torch = t
            AutoTokenizer = AT
            AutoModel = AM
        except ImportError:
            logger.warning("transformers/torch not installed. FinBERTEmbedder will use fallback.")
            return False
    return True


class FinBERTEmbedder(BaseEmbedder):
    """
    FinBERT embedder for finance-specific text.
    Optimized for financial news and sentiment.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "cuda"):
        """
        Initialize FinBERT embedder.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self._device_str = device
        self._device = None
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        if not _load_transformers():
            return

        try:
            logger.info(f"Loading FinBERT embedder: {self.MODEL_NAME}")

            self._device = torch.device(
                self._device_str if torch.cuda.is_available() else "cpu"
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME).to(self._device)
            self._model.eval()
            self._loaded = True

            logger.info(f"FinBERT loaded on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (768 for BERT-base)."""
        return 768

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.MODEL_NAME

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for financial text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            numpy array of shape (n_texts, 768)
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        if self._model is None or torch is None:
            # Fallback
            logger.warning("Using fallback random embeddings for FinBERT")
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)

        with torch.no_grad():
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self._device)

            outputs = self._model(**inputs)

            # Mean pooling over token embeddings
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two financial texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.embed([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
