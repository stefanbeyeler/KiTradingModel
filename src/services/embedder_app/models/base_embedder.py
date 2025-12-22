"""Base class for all embedders."""

from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding models."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass

    @abstractmethod
    def embed(self, inputs: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """
        Generate embeddings for the inputs.

        Args:
            inputs: Input data (text, list of texts, or numpy array)

        Returns:
            numpy array of embeddings
        """
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return True
