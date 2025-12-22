"""Time Series embedder using TS2Vec architecture."""

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
            logger.warning("torch not installed. TimeSeriesEmbedder will use fallback.")
            return False
    return True


class TS2VecEncoder:
    """
    Time Series to Vector Encoder.
    Based on TS2Vec architecture for contrastive learning.

    Note: This is a simplified version. Full implementation would require
    the complete TS2Vec training pipeline.
    """

    def __init__(
        self,
        input_dim: int = 5,       # OHLCV
        hidden_dim: int = 64,
        output_dim: int = 320,
        depth: int = 10,
        kernel_size: int = 3
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self._model = None

    def _build_model(self):
        """Build the PyTorch model."""
        if not _load_torch():
            return None

        class CausalConv1d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
                super().__init__()
                self.padding = (kernel_size - 1) * dilation
                self.conv = nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=self.padding, dilation=dilation
                )

            def forward(self, x):
                out = self.conv(x)
                return out[:, :, :-self.padding] if self.padding > 0 else out

        class TCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
                super().__init__()
                self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
                self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.bn2 = nn.BatchNorm1d(out_channels)
                self.dropout = nn.Dropout(dropout)
                self.relu = nn.ReLU()
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                    if in_channels != out_channels else None

            def forward(self, x):
                residual = self.downsample(x) if self.downsample else x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.dropout(out)
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.dropout(out)
                return self.relu(out + residual)

        class TS2VecModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, depth, kernel_size):
                super().__init__()
                self.input_fc = nn.Linear(input_dim, hidden_dim)
                self.blocks = nn.ModuleList()

                for i in range(depth):
                    dilation = 2 ** i
                    self.blocks.append(TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation))

                self.output_fc = nn.Linear(hidden_dim, output_dim)
                self.layer_norm = nn.LayerNorm(hidden_dim)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                x = self.input_fc(x)  # (batch, seq_len, hidden_dim)
                x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)

                for block in self.blocks:
                    x = block(x)

                # Global average pooling
                x = x.mean(dim=2)  # (batch, hidden_dim)
                x = self.layer_norm(x)
                x = self.output_fc(x)

                # L2 normalize
                x = torch.nn.functional.normalize(x, p=2, dim=1)
                return x

        return TS2VecModel(
            self.input_dim, self.hidden_dim, self.output_dim,
            self.depth, self.kernel_size
        )

    def get_model(self):
        """Get or create the model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model


class TimeSeriesEmbedder(BaseEmbedder):
    """
    Time Series Embedder for OHLCV data.
    Uses dilated causal convolutions for temporal patterns.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        output_dim: int = 320
    ):
        """
        Initialize the time series embedder.

        Args:
            model_path: Path to pretrained model weights
            device: Device to use ('cuda' or 'cpu')
            output_dim: Output embedding dimension
        """
        self._model_path = model_path
        self._device_str = device
        self._output_dim = output_dim
        self._device = None
        self._encoder = TS2VecEncoder(output_dim=output_dim)
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

            self._model = self._encoder.get_model()
            if self._model is None:
                return

            self._model = self._model.to(self._device)

            if self._model_path:
                try:
                    state_dict = torch.load(self._model_path, map_location=self._device)
                    self._model.load_state_dict(state_dict)
                    logger.info(f"Loaded pretrained TS2Vec from {self._model_path}")
                except Exception as e:
                    logger.warning(f"Could not load pretrained model: {e}")

            self._model.eval()
            self._loaded = True
            logger.info(f"TimeSeriesEmbedder initialized on {self._device}")
        except Exception as e:
            logger.error(f"Failed to initialize TimeSeriesEmbedder: {e}")

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._output_dim

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "ts2vec"

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _normalize(self, sequences: np.ndarray) -> np.ndarray:
        """Z-Score normalization per sequence."""
        mean = sequences.mean(axis=(1, 2), keepdims=True)
        std = sequences.std(axis=(1, 2), keepdims=True) + 1e-8
        return (sequences - mean) / std

    def embed(self, sequences: Union[np.ndarray, list]) -> np.ndarray:
        """
        Generate embeddings for OHLCV sequences.

        Args:
            sequences: numpy array (n_sequences, seq_len, 5) for OHLCV
                      or (seq_len, 5) for single sequence

        Returns:
            numpy array (n_sequences, output_dim)
        """
        self._load_model()

        if isinstance(sequences, list):
            sequences = np.array(sequences)

        if sequences.ndim == 2:
            sequences = sequences[np.newaxis, ...]

        # Normalize
        sequences = self._normalize(sequences)

        if self._model is None or torch is None:
            # Fallback
            logger.warning("Using fallback random embeddings for TimeSeries")
            return np.random.randn(len(sequences), self.embedding_dim).astype(np.float32)

        with torch.no_grad():
            x = torch.tensor(sequences, dtype=torch.float32).to(self._device)
            embeddings = self._model(x)

        return embeddings.cpu().numpy().astype(np.float32)

    def embed_from_ohlcv(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """
        Generate embedding from separate OHLCV arrays.

        Args:
            open_prices: Open prices array
            high_prices: High prices array
            low_prices: Low prices array
            close_prices: Close prices array
            volumes: Volume array

        Returns:
            numpy array (1, output_dim)
        """
        # Stack into (seq_len, 5) then add batch dimension
        sequence = np.stack([
            open_prices, high_prices, low_prices, close_prices, volumes
        ], axis=-1)

        return self.embed(sequence)
