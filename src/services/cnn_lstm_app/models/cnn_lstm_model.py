"""
CNN-LSTM Multi-Task Model

Hybrid CNN-LSTM Architektur mit Multi-Task Learning für:
- Preis-Vorhersage (Regression)
- Pattern-Klassifikation (16 Chart-Patterns)
- Regime-Vorhersage (4 Markt-Regime)

Architektur:
1. CNN-Encoder: Extrahiert lokale Features aus Preis-Windows
2. BiLSTM-Encoder: Verarbeitet sequenzielle Zusammenhaenge
3. Multi-Task Heads: Spezialisierte Ausgabeschichten
"""

from dataclasses import dataclass, field
from typing import Optional

# Lazy import fuer PyTorch (CPU-Fallback wenn nicht verfuegbar)
torch = None
nn = None


def _ensure_torch():
    """Stellt sicher dass PyTorch geladen ist."""
    global torch, nn
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        torch = _torch
        nn = _nn


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CNNLSTMConfig:
    """Konfiguration fuer das CNN-LSTM Modell."""
    # Input
    input_features: int = 25
    sequence_length: int = 168

    # CNN Encoder
    cnn_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.2

    # LSTM Encoder
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    bidirectional: bool = True

    # Output Heads
    num_price_outputs: int = 4      # 1h, 4h, 1d, 1w
    num_pattern_classes: int = 16   # 16 Chart-Patterns
    num_regime_classes: int = 4     # Bull, Bear, Sideways, HighVol

    # Head Hidden Size
    head_hidden_size: int = 128


# Pattern und Regime Klassen
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

REGIME_CLASSES = [
    "bull_trend",
    "bear_trend",
    "sideways",
    "high_volatility",
]


# =============================================================================
# CNN Block
# =============================================================================

class CNNBlock:
    """
    CNN Block mit Conv1d, BatchNorm, ReLU und Dropout.

    Verwendet Residual Connection wenn in_channels == out_channels.
    """

    @staticmethod
    def create(in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.2):
        """Factory-Methode fuer CNN Block."""
        _ensure_torch()

        class _CNNBlock(nn.Module):
            def __init__(self):
                super().__init__()
                padding = kernel_size // 2  # Same padding

                self.conv = nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=padding, bias=False
                )
                self.bn = nn.BatchNorm1d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(dropout)

                # Residual connection wenn Dimensionen passen
                self.residual = None
                if in_channels != out_channels:
                    self.residual = nn.Conv1d(in_channels, out_channels, 1, bias=False)

            def forward(self, x):
                identity = x

                out = self.conv(x)
                out = self.bn(out)
                out = self.relu(out)
                out = self.dropout(out)

                # Residual
                if self.residual is not None:
                    identity = self.residual(identity)

                out = out + identity
                return out

        return _CNNBlock()


# =============================================================================
# Task Heads
# =============================================================================

class PriceHead:
    """Head fuer Preis-Vorhersage (Regression)."""

    @staticmethod
    def create(input_size: int, hidden_size: int, num_outputs: int):
        _ensure_torch()

        class _PriceHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_size // 2, num_outputs)
                )

            def forward(self, x):
                return self.fc(x)

        return _PriceHead()


class PatternHead:
    """Head fuer Pattern-Klassifikation (Multi-Label)."""

    @staticmethod
    def create(input_size: int, hidden_size: int, num_classes: int):
        _ensure_torch()

        class _PatternHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, num_classes)
                )
                # Sigmoid wird im Loss (BCEWithLogitsLoss) angewendet

            def forward(self, x):
                return self.fc(x)

        return _PatternHead()


class RegimeHead:
    """Head fuer Regime-Vorhersage (Multi-Class)."""

    @staticmethod
    def create(input_size: int, hidden_size: int, num_classes: int):
        _ensure_torch()

        class _RegimeHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, num_classes)
                )
                # Softmax wird im Loss (CrossEntropyLoss) angewendet

            def forward(self, x):
                return self.fc(x)

        return _RegimeHead()


# =============================================================================
# Main Model
# =============================================================================

def create_cnn_lstm_model(config: Optional[CNNLSTMConfig] = None):
    """
    Factory-Funktion fuer das CNN-LSTM Multi-Task Modell.

    Args:
        config: Modell-Konfiguration (optional, nutzt Default wenn None)

    Returns:
        PyTorch nn.Module mit forward() Methode
    """
    _ensure_torch()

    if config is None:
        config = CNNLSTMConfig()

    class CNNLSTMMultiTask(nn.Module):
        """
        Hybrid CNN-LSTM Multi-Task Model.

        Input Shape: (batch, sequence_length, input_features)
        Output: Dictionary mit 'price', 'patterns', 'regime' Tensoren
        """

        def __init__(self):
            super().__init__()
            self.config = config

            # =================================================================
            # CNN Encoder
            # =================================================================
            cnn_layers = []
            in_channels = config.input_features

            for out_channels in config.cnn_channels:
                cnn_layers.append(
                    CNNBlock.create(
                        in_channels, out_channels,
                        config.cnn_kernel_size, config.cnn_dropout
                    )
                )
                in_channels = out_channels

            self.cnn_encoder = nn.Sequential(*cnn_layers)

            # =================================================================
            # LSTM Encoder
            # =================================================================
            lstm_input_size = config.cnn_channels[-1]
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                batch_first=True,
                dropout=config.lstm_dropout if config.lstm_layers > 1 else 0,
                bidirectional=config.bidirectional
            )

            # LSTM Output Size (bidirectional verdoppelt)
            lstm_output_size = config.lstm_hidden * (2 if config.bidirectional else 1)

            # =================================================================
            # Attention Layer (optional, fuer bessere Feature-Aggregation)
            # =================================================================
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_size // 2, 1)
            )

            # =================================================================
            # Task Heads
            # =================================================================
            self.price_head = PriceHead.create(
                lstm_output_size, config.head_hidden_size, config.num_price_outputs
            )
            self.pattern_head = PatternHead.create(
                lstm_output_size, config.head_hidden_size, config.num_pattern_classes
            )
            self.regime_head = RegimeHead.create(
                lstm_output_size, config.head_hidden_size, config.num_regime_classes
            )

            # Layer Normalization fuer Stabilisierung
            self.layer_norm = nn.LayerNorm(lstm_output_size)

        def forward(self, x):
            """
            Forward pass durch das Modell.

            Args:
                x: Input Tensor (batch, seq_len, features)

            Returns:
                dict mit 'price', 'patterns', 'regime' Tensoren
            """
            batch_size = x.size(0)

            # CNN erwartet (batch, channels, seq_len)
            x = x.permute(0, 2, 1)

            # CNN Encoder
            cnn_out = self.cnn_encoder(x)

            # Zurueck zu (batch, seq_len, channels) fuer LSTM
            cnn_out = cnn_out.permute(0, 2, 1)

            # LSTM Encoder
            lstm_out, (h_n, c_n) = self.lstm(cnn_out)

            # Attention-basierte Aggregation
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)

            # Layer Normalization
            context = self.layer_norm(context)

            # Task Heads
            price_out = self.price_head(context)
            pattern_out = self.pattern_head(context)
            regime_out = self.regime_head(context)

            return {
                'price': price_out,
                'patterns': pattern_out,
                'regime': regime_out
            }

        def get_num_parameters(self) -> int:
            """Gibt die Gesamtzahl der Parameter zurueck."""
            return sum(p.numel() for p in self.parameters())

        def get_trainable_parameters(self) -> int:
            """Gibt die Anzahl trainierbarer Parameter zurueck."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return CNNLSTMMultiTask()


# =============================================================================
# Model Factory
# =============================================================================

class CNNLSTMModelFactory:
    """Factory fuer CNN-LSTM Modelle mit verschiedenen Konfigurationen."""

    @staticmethod
    def create_default() -> "nn.Module":
        """Erstellt Modell mit Default-Konfiguration."""
        return create_cnn_lstm_model(CNNLSTMConfig())

    @staticmethod
    def create_small() -> "nn.Module":
        """Erstellt kleineres Modell fuer schnelleres Training."""
        config = CNNLSTMConfig(
            cnn_channels=[32, 64, 128],
            lstm_hidden=64,
            lstm_layers=1,
            head_hidden_size=64
        )
        return create_cnn_lstm_model(config)

    @staticmethod
    def create_large() -> "nn.Module":
        """Erstellt groesseres Modell fuer bessere Performance."""
        config = CNNLSTMConfig(
            cnn_channels=[64, 128, 256, 512],
            lstm_hidden=256,
            lstm_layers=3,
            head_hidden_size=256
        )
        return create_cnn_lstm_model(config)

    @staticmethod
    def create_from_config(config: CNNLSTMConfig) -> "nn.Module":
        """Erstellt Modell aus Konfiguration."""
        return create_cnn_lstm_model(config)


# =============================================================================
# Utility Functions
# =============================================================================

def load_model(model_path: str, config: Optional[CNNLSTMConfig] = None, device: str = "cpu"):
    """
    Laedt ein gespeichertes Modell.

    Args:
        model_path: Pfad zur .pt Datei
        config: Modell-Konfiguration (optional)
        device: Zielgeraet (cpu/cuda)

    Returns:
        Geladenes Modell
    """
    _ensure_torch()

    model = create_cnn_lstm_model(config)
    state_dict = torch.load(model_path, map_location=device)

    # Unterstuetze sowohl direkte State Dicts als auch Checkpoints
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def save_model(model, model_path: str, metadata: Optional[dict] = None):
    """
    Speichert ein Modell.

    Args:
        model: PyTorch Modell
        model_path: Zielpfad
        metadata: Optionale Metadaten
    """
    _ensure_torch()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__ if hasattr(model, 'config') else None,
    }

    if metadata:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, model_path)


def get_device() -> str:
    """Gibt das beste verfuegbare Device zurueck."""
    _ensure_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"
