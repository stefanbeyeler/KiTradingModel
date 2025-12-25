"""Temporal Convolutional Network model architecture for training."""

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
            logger.warning("torch not installed.")
            return False
    return True


class TCNPatternModel:
    """
    TCN Pattern Classification Model.

    Factory class that creates the actual PyTorch model.
    """

    def __new__(cls, num_inputs: int, num_channels: list, num_classes: int,
                kernel_size: int = 3, dropout: float = 0.2):
        """Create and return the PyTorch model directly."""
        if not _load_torch():
            raise RuntimeError("PyTorch not available")

        class TCNBlock(nn.Module):
            """Residual TCN block with dilated convolutions."""

            def __init__(self, in_ch, out_ch, k_size, dilation, drop):
                super().__init__()
                padding = (k_size - 1) * dilation

                self.conv1 = nn.Conv1d(in_ch, out_ch, k_size, padding=padding, dilation=dilation)
                self.conv2 = nn.Conv1d(out_ch, out_ch, k_size, padding=padding, dilation=dilation)
                self.bn1 = nn.BatchNorm1d(out_ch)
                self.bn2 = nn.BatchNorm1d(out_ch)
                self.dropout = nn.Dropout(drop)
                self.relu = nn.ReLU()
                self.padding = padding

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

                # Align sizes
                min_len = min(out.size(-1), residual.size(-1))
                out = out[:, :, :min_len]
                residual = residual[:, :, :min_len]

                return self.relu(out + residual)

        class TCNClassifier(nn.Module):
            """Full TCN classifier model."""

            def __init__(self, n_inputs, n_channels, n_classes, k_size, drop):
                super().__init__()

                self.blocks = nn.ModuleList()
                in_ch = n_inputs

                for i, out_ch in enumerate(n_channels):
                    dilation = 2 ** i
                    self.blocks.append(TCNBlock(in_ch, out_ch, k_size, dilation, drop))
                    in_ch = out_ch

                self.global_pool = nn.AdaptiveAvgPool1d(1)

                self.classifier = nn.Sequential(
                    nn.Linear(n_channels[-1], 256),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(256, n_classes),
                    nn.Sigmoid()
                )

            def forward(self, x):
                # x shape: (batch, channels, seq_len)
                for block in self.blocks:
                    x = block(x)

                x = self.global_pool(x).squeeze(-1)
                return self.classifier(x)

        return TCNClassifier(num_inputs, num_channels, num_classes, kernel_size, dropout)
