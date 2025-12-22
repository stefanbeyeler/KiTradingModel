"""Feature Autoencoder for compressing technical indicators."""

import numpy as np
from typing import Optional, Union
from loguru import logger

from .base_embedder import BaseEmbedder

# Lazy imports
torch = None
nn = None


def _load_torch():
    global torch, nn
    if torch is None:
        try:
            import torch as t
            import torch.nn as tnn
            torch = t
            nn = tnn
        except ImportError:
            logger.warning("torch not installed. FeatureEmbedder will use fallback.")
            return False
    return True


class FeatureAutoencoder:
    """
    Autoencoder for feature compression.
    Reduces high-dimensional technical indicators to compact representation.
    """

    def __init__(
        self,
        input_dim: int = 50,      # Number of technical indicators
        latent_dim: int = 128,
        hidden_dims: list = None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self._model = None

    def _build_model(self):
        """Build the PyTorch model."""
        if not _load_torch():
            return None

        class AutoencoderModel(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dims):
                super().__init__()

                # Encoder
                encoder_layers = []
                in_dim = input_dim
                for h_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(in_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    in_dim = h_dim
                encoder_layers.append(nn.Linear(in_dim, latent_dim))
                self.encoder = nn.Sequential(*encoder_layers)

                # Decoder
                decoder_layers = []
                in_dim = latent_dim
                for h_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(in_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    in_dim = h_dim
                decoder_layers.append(nn.Linear(in_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)

            def encode(self, x):
                return self.encoder(x)

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                z = self.encode(x)
                x_recon = self.decode(z)
                return x_recon, z

        return AutoencoderModel(self.input_dim, self.latent_dim, self.hidden_dims)

    def get_model(self):
        """Get or create the model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model


class FeatureEmbedder(BaseEmbedder):
    """
    Feature embedder using autoencoder compression.
    Compresses technical indicators into dense embeddings.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        input_dim: int = 50,
        latent_dim: int = 128
    ):
        """
        Initialize the feature embedder.

        Args:
            model_path: Path to pretrained model weights
            device: Device to use ('cuda' or 'cpu')
            input_dim: Number of input features
            latent_dim: Latent space dimension
        """
        self._model_path = model_path
        self._device_str = device
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._device = None
        self._autoencoder = FeatureAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        if not _load_torch():
            return

        try:
            self._device = torch.device(
                self._device_str if torch.cuda.is_available() else "cpu"
            )

            self._model = self._autoencoder.get_model()
            if self._model is None:
                return

            self._model = self._model.to(self._device)

            if self._model_path:
                try:
                    state_dict = torch.load(self._model_path, map_location=self._device)
                    self._model.load_state_dict(state_dict)
                    logger.info(f"Loaded pretrained FeatureAutoencoder from {self._model_path}")
                except Exception as e:
                    logger.warning(f"Could not load pretrained model: {e}")

            self._model.eval()
            self._loaded = True
            logger.info(f"FeatureEmbedder initialized on {self._device}")
        except Exception as e:
            logger.error(f"Failed to initialize FeatureEmbedder: {e}")

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._latent_dim

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "feature_autoencoder"

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def embed(self, features: Union[np.ndarray, list]) -> np.ndarray:
        """
        Compress feature vectors.

        Args:
            features: numpy array (n_samples, n_features)
                     or (n_features,) for single sample

        Returns:
            numpy array (n_samples, latent_dim)
        """
        self._load_model()

        if isinstance(features, list):
            features = np.array(features)

        if features.ndim == 1:
            features = features[np.newaxis, ...]

        # Handle variable input dimensions
        if features.shape[1] != self._input_dim:
            # Pad or truncate to match expected input
            if features.shape[1] < self._input_dim:
                padding = np.zeros((features.shape[0], self._input_dim - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
            else:
                features = features[:, :self._input_dim]

        if self._model is None or torch is None:
            # Fallback
            logger.warning("Using fallback random embeddings for Features")
            return np.random.randn(len(features), self.embedding_dim).astype(np.float32)

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self._device)
            _, embeddings = self._model(x)

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def reconstruct(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Reconstruct features and compute reconstruction error.

        Args:
            features: numpy array (n_samples, n_features)

        Returns:
            Tuple of (reconstructed features, reconstruction error)
        """
        self._load_model()

        if isinstance(features, list):
            features = np.array(features)

        if features.ndim == 1:
            features = features[np.newaxis, ...]

        if self._model is None or torch is None:
            return features, 0.0

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self._device)
            x_recon, _ = self._model(x)

            # Compute MSE
            mse = torch.mean((x - x_recon) ** 2).item()

        return x_recon.cpu().numpy().astype(np.float32), mse
