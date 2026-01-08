"""Temporal Convolutional Network for Pattern Recognition."""

from typing import List, Optional, Dict
import numpy as np
from loguru import logger

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
            return True
        except ImportError:
            logger.warning("torch not installed. TCN will use fallback.")
            return False
    return True


class CausalConv1d:
    """Causal convolution - only sees past data."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self._module = None

    def get_module(self):
        if self._module is None and _load_torch():
            class CausalConvModule(nn.Module):
                def __init__(self, in_ch, out_ch, k_size, dil, pad):
                    super().__init__()
                    self.padding = pad
                    self.conv = nn.Conv1d(
                        in_ch, out_ch, k_size,
                        padding=pad, dilation=dil
                    )

                def forward(self, x):
                    out = self.conv(x)
                    return out[:, :, :-self.padding] if self.padding > 0 else out

            self._module = CausalConvModule(
                self.in_channels, self.out_channels,
                self.kernel_size, self.dilation, self.padding
            )
        return self._module


class TCNBlock:
    """Residual TCN block with dilated convolutions."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self._module = None

    def get_module(self):
        if self._module is None and _load_torch():
            padding = (self.kernel_size - 1) * self.dilation

            class TCNBlockModule(nn.Module):
                def __init__(self, in_ch, out_ch, k_size, dil, pad, drop):
                    super().__init__()
                    self.padding = pad

                    self.conv1 = nn.Conv1d(in_ch, out_ch, k_size, padding=pad, dilation=dil)
                    self.conv2 = nn.Conv1d(out_ch, out_ch, k_size, padding=pad, dilation=dil)
                    self.bn1 = nn.BatchNorm1d(out_ch)
                    self.bn2 = nn.BatchNorm1d(out_ch)
                    self.dropout = nn.Dropout(drop)
                    self.relu = nn.ReLU()

                    self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

                def forward(self, x):
                    residual = self.downsample(x) if self.downsample else x

                    out = self.conv1(x)
                    out = out[:, :, :-self.padding] if self.padding > 0 else out
                    out = self.relu(self.bn1(out))
                    out = self.dropout(out)

                    out = self.conv2(out)
                    out = out[:, :, :-self.padding] if self.padding > 0 else out
                    out = self.relu(self.bn2(out))
                    out = self.dropout(out)

                    return self.relu(out + residual)

            self._module = TCNBlockModule(
                self.in_channels, self.out_channels,
                self.kernel_size, self.dilation, padding, self.dropout
            )
        return self._module


class TCNPatternClassifier:
    """
    Multi-Label Pattern Classifier with TCN Backbone.

    Detects 16 chart pattern types:
    - Head & Shoulders variants
    - Double/Triple tops and bottoms
    - Triangles (ascending, descending, symmetrical)
    - Flags and wedges
    - Channels
    """

    PATTERN_CLASSES = [
        "head_and_shoulders",
        "inverse_head_and_shoulders",
        "double_top",
        "double_bottom",
        "triple_top",
        "triple_bottom",
        "ascending_triangle",
        "descending_triangle",
        "symmetrical_triangle",
        "bull_flag",
        "bear_flag",
        "cup_and_handle",
        "rising_wedge",
        "falling_wedge",
        "channel_up",
        "channel_down",
    ]

    def __init__(
        self,
        input_channels: int = 5,       # OHLCV
        num_classes: int = 16,
        hidden_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = "cuda"
    ):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels or [64, 128, 256, 512]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device_str = device
        self._model = None
        self._device = None
        self._loaded = False

    def _build_model(self):
        """Build the PyTorch model."""
        if not _load_torch():
            return None

        hidden_channels = self.hidden_channels
        kernel_size = self.kernel_size
        dropout = self.dropout
        input_channels = self.input_channels
        num_classes = self.num_classes

        class TCNClassifierModel(nn.Module):
            def __init__(self):
                super().__init__()

                # Build TCN blocks with proper residual connections
                self.blocks = nn.ModuleList()
                self.residuals = nn.ModuleList()

                in_ch = input_channels
                for i, out_ch in enumerate(hidden_channels):
                    dilation = 2 ** i
                    padding = (kernel_size - 1) * dilation

                    block = nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    self.blocks.append(block)

                    # Residual connection (1x1 conv if channel mismatch)
                    if in_ch != out_ch:
                        self.residuals.append(nn.Conv1d(in_ch, out_ch, 1))
                    else:
                        self.residuals.append(nn.Identity())

                    in_ch = out_ch

                self.global_pool = nn.AdaptiveAvgPool1d(1)

                # Classifier head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_channels[-1], 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes),
                    nn.Sigmoid()  # Multi-label
                )

            def forward(self, x):
                # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
                if x.dim() == 3 and x.size(-1) == input_channels:
                    x = x.transpose(1, 2)

                # Apply TCN blocks with residual connections
                for block, residual in zip(self.blocks, self.residuals):
                    out = block(x)
                    res = residual(x)
                    # Align sizes - trim the longer one to match the shorter
                    min_len = min(out.size(-1), res.size(-1))
                    out = out[:, :, :min_len]
                    res = res[:, :, :min_len]
                    x = nn.functional.relu(out + res)

                # Global pooling
                x = self.global_pool(x).squeeze(-1)

                # Classify
                return self.classifier(x)

        return TCNClassifierModel()

    def load(self, model_path: Optional[str] = None):
        """Load the model."""
        if not _load_torch():
            logger.warning("PyTorch not available, using fallback mode")
            return

        self._device = torch.device(
            self.device_str if torch.cuda.is_available() else "cpu"
        )

        self._model = self._build_model()
        if self._model is None:
            return

        self._model = self._model.to(self._device)

        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                logger.info(f"Loaded TCN model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")

        self._model.eval()
        self._loaded = True

        # Cache parameter count to avoid slow GPU iteration on every info request
        self._cached_num_parameters = sum(p.numel() for p in self._model.parameters())

        # GPU warmup: Run a dummy inference to initialize CUDA context
        # This prevents 30+ second delay on first API request
        self._warmup_gpu()

        logger.info(f"TCN Pattern Classifier initialized on {self._device} ({self._cached_num_parameters:,} parameters)")

    def _warmup_gpu(self):
        """Perform GPU warmup with dummy inference to initialize CUDA context."""
        if self._model is None or self._device is None:
            return

        try:
            import numpy as np
            # Create dummy input matching expected shape (batch, seq_len, channels)
            dummy_input = np.random.randn(1, 200, 5).astype(np.float32)
            dummy_input = self._normalize(dummy_input)

            with torch.no_grad():
                x_tensor = torch.tensor(dummy_input, dtype=torch.float32).to(self._device)
                _ = self._model(x_tensor)
                # Ensure CUDA operations complete
                if self._device.type == 'cuda':
                    torch.cuda.synchronize()

            logger.debug("GPU warmup completed")
        except Exception as e:
            logger.warning(f"GPU warmup failed (non-critical): {e}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_num_parameters(self) -> int:
        """Get cached number of model parameters (fast)."""
        return getattr(self, '_cached_num_parameters', 0)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Predict patterns in the sequence.

        Args:
            x: OHLCV sequence (seq_len, 5) or (batch, seq_len, 5)
            threshold: Confidence threshold

        Returns:
            Dictionary of detected patterns with confidence scores
        """
        if self._model is None:
            # Fallback: random predictions
            return self._fallback_predict(threshold)

        if x.ndim == 2:
            x = x[np.newaxis, ...]

        # Normalize
        x = self._normalize(x)

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self._device)
            probs = self._model(x_tensor)
            probs = probs.cpu().numpy()[0]

        results = {}
        for i, (prob, pattern) in enumerate(zip(probs, self.PATTERN_CLASSES)):
            if prob >= threshold:
                results[pattern] = float(prob)

        return results

    def predict_with_scores(self, x: np.ndarray) -> Dict[str, float]:
        """
        Get all pattern scores (no threshold).

        Args:
            x: OHLCV sequence

        Returns:
            Dictionary of all patterns with scores
        """
        if self._model is None:
            return {p: np.random.random() * 0.3 for p in self.PATTERN_CLASSES}

        if x.ndim == 2:
            x = x[np.newaxis, ...]

        x = self._normalize(x)

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self._device)
            probs = self._model(x_tensor)
            probs = probs.cpu().numpy()[0]

        return {pattern: float(prob) for pattern, prob in zip(self.PATTERN_CLASSES, probs)}

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize sequences."""
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True) + 1e-8
        return (x - mean) / std

    def _fallback_predict(self, threshold: float) -> Dict[str, float]:
        """Fallback predictions for testing without model."""
        results = {}
        for pattern in self.PATTERN_CLASSES:
            prob = np.random.random() * 0.4  # Low random probabilities
            if prob >= threshold:
                results[pattern] = prob
        return results

    def save(self, path: str):
        """Save model weights."""
        if self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info(f"Model saved to {path}")
