# Umsetzungsvorschlag: Neue Microservices

## Übersicht

Dieses Dokument beschreibt den Umsetzungsvorschlag für drei neue Microservices:

| Service | Port | Zweck |
|---------|------|-------|
| **TCN-Pattern Service** | 3005 | Temporal Convolutional Network für Muster-Erkennung |
| **HMM-Regime Service** | 3006 | Hidden Markov Model + LightGBM für Regime-Erkennung & Scoring |
| **Embedder Service** | 3007 | Zentraler Embedding-Service für alle ML-Modelle |

## Architektur-Integration

```
                    ┌─────────────────────┐
                    │    DATA SERVICE     │  Port 3001
                    │    (Gateway)        │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ TCN-Pattern   │    │  HMM-Regime     │    │    Embedder     │
│ Service       │    │  Service        │    │    Service      │
│ Port 3005     │    │  Port 3006      │    │    Port 3007    │
└───────┬───────┘    └────────┬────────┘    └────────┬────────┘
        │                     │                      │
        │                     ▼                      │
        │            ┌─────────────────┐             │
        │            │  LightGBM       │             │
        │            │  Scorer         │             │
        │            └─────────────────┘             │
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   NHITS Service   │  Port 3002
                    │   RAG Service     │  Port 3003
                    │   LLM Service     │  Port 3004
                    └───────────────────┘
```

---

## 1. TCN-Pattern Service (Port 3005)

### 1.1 Zweck

Temporal Convolutional Networks (TCN) für:
- **Chart-Pattern-Erkennung** (Head & Shoulders, Double Top/Bottom, Triangles, etc.)
- **Sequenz-Klassifikation** von Kursverläufen
- **Multi-Scale Pattern Detection** über verschiedene Zeitfenster

### 1.2 Warum TCN?

| Eigenschaft | Vorteil für Trading |
|-------------|---------------------|
| **Kausale Konvolutionen** | Keine Zukunftsdaten-Leakage |
| **Dilatierte Schichten** | Erfasst lange Abhängigkeiten effizient |
| **Parallelisierbar** | Schneller als RNNs/LSTMs |
| **Residual Connections** | Stabiles Training bei tiefen Netzen |

### 1.3 Verzeichnisstruktur

```
src/services/tcn_app/
├── __init__.py
├── main.py                          # FastAPI Application
├── routers/
│   ├── __init__.py
│   ├── training_router.py           # Training-Endpoints
│   ├── detection_router.py          # Pattern-Detection-Endpoints
│   └── system_router.py             # Health & Monitoring
├── models/
│   ├── __init__.py
│   ├── tcn_model.py                 # TCN-Architektur
│   ├── pattern_classifier.py        # Multi-Label Pattern Classifier
│   └── schemas.py                   # Pydantic-Modelle
├── services/
│   ├── __init__.py
│   ├── tcn_training_service.py      # Training-Logik
│   ├── pattern_detection_service.py # Inference-Logik
│   └── pattern_labeling_service.py  # Automatisches Labeling
└── utils/
    ├── __init__.py
    ├── data_preprocessing.py        # OHLCV → Sequenzen
    └── pattern_definitions.py       # Pattern-Regeln
```

### 1.4 TCN-Architektur

```python
# src/services/tcn_app/models/tcn_model.py

import torch
import torch.nn as nn
from typing import List

class CausalConv1d(nn.Module):
    """Kausale Konvolution - sieht nur vergangene Daten."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Residual TCN Block mit dilatierten Konvolutionen."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x) if self.downsample else x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        return self.relu(out + residual)


class TCNPatternClassifier(nn.Module):
    """Multi-Label Pattern Classifier mit TCN-Backbone."""

    # Unterstützte Pattern-Typen
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
        input_channels: int = 5,      # OHLCV
        num_classes: int = 16,         # Pattern-Typen
        hidden_channels: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # TCN Backbone
        layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(hidden_channels):
            dilation = 2 ** i  # Exponentiell wachsende Dilation
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        # Global Average Pooling + Classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Multi-Label
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, sequence_length) - OHLCV Sequenz
        Returns:
            (batch, num_classes) - Pattern-Wahrscheinlichkeiten
        """
        features = self.tcn(x)
        pooled = self.global_pool(features).squeeze(-1)
        return self.classifier(pooled)

    def detect_patterns(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> dict:
        """Erkennt Pattern mit Konfidenz-Schwelle."""
        with torch.no_grad():
            probs = self.forward(x)

        results = {}
        for i, (prob, pattern) in enumerate(zip(probs[0], self.PATTERN_CLASSES)):
            if prob >= threshold:
                results[pattern] = float(prob)

        return results
```

### 1.5 API-Endpoints

```python
# src/services/tcn_app/routers/detection_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

class PatternDetectionRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    lookback: int = 200
    threshold: float = 0.5
    patterns: Optional[List[str]] = None  # Filter für bestimmte Pattern

class DetectedPattern(BaseModel):
    pattern_type: str
    confidence: float
    start_index: int
    end_index: int
    price_target: Optional[float] = None
    invalidation_level: Optional[float] = None

class PatternDetectionResponse(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    patterns: List[DetectedPattern]
    market_context: dict

@router.post("/detect", response_model=PatternDetectionResponse)
async def detect_patterns(request: PatternDetectionRequest):
    """
    Erkennt Chart-Patterns im aktuellen Kursverlauf.

    - **symbol**: Trading-Symbol (z.B. BTCUSD)
    - **timeframe**: Zeitrahmen (1m, 5m, 15m, 1h, 4h, 1d)
    - **lookback**: Anzahl Kerzen für Analyse
    - **threshold**: Konfidenz-Schwelle (0.0-1.0)
    """
    pass

@router.post("/scan-all")
async def scan_all_symbols(
    timeframe: str = "1h",
    threshold: float = 0.6,
    min_patterns: int = 1
):
    """Scannt alle aktiven Symbole nach Patterns."""
    pass

@router.get("/history/{symbol}")
async def get_pattern_history(
    symbol: str,
    days: int = 30,
    pattern_type: Optional[str] = None
):
    """Historische Pattern-Erkennungen für ein Symbol."""
    pass
```

### 1.6 Training-Service

```python
# src/services/tcn_app/services/tcn_training_service.py

from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import DataLoader
from src.services.data_gateway_service import data_gateway

@dataclass
class TCNTrainingConfig:
    sequence_length: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2

class TCNTrainingService:
    """Service für TCN-Modell-Training."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TCNPatternClassifier] = None
        self.training_in_progress = False

    async def prepare_training_data(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        lookback_days: int = 365
    ) -> DataLoader:
        """
        Bereitet Trainingsdaten vor:
        1. Holt OHLCV-Daten via DataGateway
        2. Generiert Labels via Pattern-Labeling-Service
        3. Erstellt PyTorch DataLoader
        """
        all_sequences = []
        all_labels = []

        for symbol in symbols:
            # Daten via Gateway holen
            data = await data_gateway.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                limit=lookback_days * 24  # Approximation für 1h
            )

            # Pattern-Labels generieren (Semi-supervised)
            sequences, labels = await self._generate_labeled_sequences(data)
            all_sequences.extend(sequences)
            all_labels.extend(labels)

        return self._create_dataloader(all_sequences, all_labels)

    async def train(
        self,
        symbols: list[str],
        config: TCNTrainingConfig
    ) -> dict:
        """Trainiert das TCN-Modell."""
        self.training_in_progress = True

        try:
            # Daten vorbereiten
            train_loader = await self.prepare_training_data(symbols)

            # Modell initialisieren
            self.model = TCNPatternClassifier().to(self.device)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate
            )
            criterion = torch.nn.BCELoss()  # Multi-Label

            # Training Loop
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(config.epochs):
                train_loss = await self._train_epoch(
                    train_loader, optimizer, criterion
                )
                val_loss = await self._validate_epoch(train_loader, criterion)

                # Early Stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    await self._save_checkpoint(epoch, val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        break

            return {
                "status": "completed",
                "final_loss": best_loss,
                "epochs_trained": epoch + 1
            }

        finally:
            self.training_in_progress = False
```

### 1.7 Docker-Konfiguration

```dockerfile
# docker/services/tcn/Dockerfile

FROM python:3.11-slim
ARG SERVICE_PORT=3005

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch mit CUDA
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements-tcn.txt .
RUN pip install --no-cache-dir -r requirements-tcn.txt

COPY src/ ./src/

# Directories
RUN mkdir -p /app/data/models/tcn /app/logs

ENV PYTHONPATH=/app
ENV SERVICE_NAME=tcn
ENV PORT=${SERVICE_PORT}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "src.services.tcn_app.main:app", "--host", "0.0.0.0", "--port", "3005"]
```

---

## 2. HMM-Regime Service (Port 3006)

### 2.1 Zweck

Hidden Markov Model für Regime-Erkennung kombiniert mit LightGBM für Signal-Scoring:

- **Regime-Erkennung**: Bull/Bear/Sideways/High-Volatility Phasen
- **Regime-Wahrscheinlichkeiten**: Probabilistische Zustandsschätzung
- **Signal-Scoring**: LightGBM bewertet Trading-Signale basierend auf aktuellem Regime

### 2.2 Architektur-Konzept

```
OHLCV + Features
       │
       ▼
┌─────────────────┐
│   HMM Module    │
│ (hmmlearn)      │
│                 │
│ States:         │
│ - Bull          │
│ - Bear          │
│ - Sideways      │
│ - High Vol      │
└────────┬────────┘
         │
         │ Regime Probabilities
         ▼
┌─────────────────┐
│ LightGBM Scorer │
│                 │
│ Features:       │
│ - Regime probs  │
│ - Technical     │
│ - Price action  │
│                 │
│ Output:         │
│ - Signal Score  │
│ - Confidence    │
└─────────────────┘
```

### 2.3 Verzeichnisstruktur

```
src/services/hmm_app/
├── __init__.py
├── main.py                          # FastAPI Application
├── routers/
│   ├── __init__.py
│   ├── regime_router.py             # Regime-Detection-Endpoints
│   ├── scoring_router.py            # LightGBM-Scoring-Endpoints
│   ├── training_router.py           # Model-Training
│   └── system_router.py             # Health & Monitoring
├── models/
│   ├── __init__.py
│   ├── hmm_regime_model.py          # HMM-Implementation
│   ├── lightgbm_scorer.py           # LightGBM Signal Scorer
│   └── schemas.py                   # Pydantic-Modelle
├── services/
│   ├── __init__.py
│   ├── regime_detection_service.py  # Regime-Erkennung
│   ├── signal_scoring_service.py    # Signal-Bewertung
│   └── feature_engineering.py       # Feature-Berechnung
└── utils/
    ├── __init__.py
    └── regime_definitions.py        # Regime-Charakteristiken
```

### 2.4 HMM-Regime-Model

```python
# src/services/hmm_app/models/hmm_regime_model.py

import numpy as np
from hmmlearn import hmm
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import joblib

class MarketRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"

@dataclass
class RegimeState:
    regime: MarketRegime
    probability: float
    duration: int  # Kerzen im aktuellen Regime
    transition_probs: dict[MarketRegime, float]

class HMMRegimeModel:
    """
    Hidden Markov Model für Markt-Regime-Erkennung.

    Verwendet Gaussian HMM mit 4 versteckten Zuständen:
    - Bull Trend: Positive Returns, niedrige Volatilität
    - Bear Trend: Negative Returns, erhöhte Volatilität
    - Sideways: Geringe Returns, niedrige Volatilität
    - High Volatility: Hohe Volatilität, ungerichtete Bewegung
    """

    REGIMES = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.SIDEWAYS,
        MarketRegime.HIGH_VOLATILITY,
    ]

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        self.is_fitted = False
        self.regime_mapping: dict[int, MarketRegime] = {}

    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extrahiert Features für HMM:
        - Returns (log)
        - Volatilität (rolling std)
        - Trend-Stärke (SMA deviation)
        """
        returns = np.diff(np.log(prices))

        # Rolling Volatilität (20 Perioden)
        volatility = np.array([
            np.std(returns[max(0, i-20):i+1])
            for i in range(len(returns))
        ])

        # Trend-Stärke (Abweichung von SMA20)
        sma = np.convolve(prices, np.ones(20)/20, mode='valid')
        sma_padded = np.concatenate([np.full(19, sma[0]), sma])
        trend_strength = (prices - sma_padded) / sma_padded

        # Features kombinieren
        features = np.column_stack([
            returns,
            volatility,
            trend_strength[1:]  # Align mit returns
        ])

        return features

    def _map_states_to_regimes(self, features: np.ndarray) -> None:
        """
        Mapped HMM-States zu Markt-Regimes basierend auf State-Charakteristiken.
        """
        means = self.model.means_

        # State-Charakteristiken analysieren
        state_chars = []
        for i in range(self.n_components):
            return_mean = means[i, 0]  # Durchschnittliche Returns
            vol_mean = means[i, 1]     # Durchschnittliche Volatilität

            state_chars.append({
                'state': i,
                'return': return_mean,
                'volatility': vol_mean
            })

        # Sortiere nach Return und Volatilität
        by_return = sorted(state_chars, key=lambda x: x['return'])
        by_vol = sorted(state_chars, key=lambda x: x['volatility'])

        # Mapping (heuristisch)
        self.regime_mapping = {
            by_return[-1]['state']: MarketRegime.BULL_TREND,      # Höchste Returns
            by_return[0]['state']: MarketRegime.BEAR_TREND,       # Niedrigste Returns
            by_vol[-1]['state']: MarketRegime.HIGH_VOLATILITY,    # Höchste Vol
        }

        # Verbleibender State = Sideways
        for i in range(self.n_components):
            if i not in self.regime_mapping:
                self.regime_mapping[i] = MarketRegime.SIDEWAYS

    def fit(self, prices: np.ndarray) -> "HMMRegimeModel":
        """Trainiert das HMM auf historischen Preisdaten."""
        features = self._extract_features(prices)
        self.model.fit(features)
        self._map_states_to_regimes(features)
        self.is_fitted = True
        return self

    def predict_regime(self, prices: np.ndarray) -> RegimeState:
        """
        Erkennt das aktuelle Markt-Regime.

        Returns:
            RegimeState mit aktuellem Regime und Wahrscheinlichkeiten
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        features = self._extract_features(prices)

        # Viterbi für wahrscheinlichste State-Sequenz
        log_prob, states = self.model.decode(features, algorithm="viterbi")
        current_state = states[-1]
        current_regime = self.regime_mapping[current_state]

        # State-Wahrscheinlichkeiten
        posteriors = self.model.predict_proba(features)
        current_probs = posteriors[-1]

        # Regime-Dauer berechnen
        duration = 1
        for i in range(len(states) - 2, -1, -1):
            if states[i] == current_state:
                duration += 1
            else:
                break

        # Übergangswahrscheinlichkeiten
        trans_probs = {}
        for i, regime in enumerate(self.REGIMES):
            for state, mapped_regime in self.regime_mapping.items():
                if mapped_regime == regime:
                    trans_probs[regime] = float(
                        self.model.transmat_[current_state, state]
                    )

        return RegimeState(
            regime=current_regime,
            probability=float(current_probs[current_state]),
            duration=duration,
            transition_probs=trans_probs
        )

    def get_regime_history(
        self,
        prices: np.ndarray
    ) -> list[Tuple[MarketRegime, float]]:
        """Gibt Regime-Historie für alle Zeitpunkte zurück."""
        features = self._extract_features(prices)
        _, states = self.model.decode(features, algorithm="viterbi")
        posteriors = self.model.predict_proba(features)

        history = []
        for i, state in enumerate(states):
            regime = self.regime_mapping[state]
            prob = float(posteriors[i, state])
            history.append((regime, prob))

        return history

    def save(self, path: str) -> None:
        """Speichert Modell und Mapping."""
        joblib.dump({
            'model': self.model,
            'regime_mapping': self.regime_mapping,
            'is_fitted': self.is_fitted
        }, path)

    @classmethod
    def load(cls, path: str) -> "HMMRegimeModel":
        """Lädt gespeichertes Modell."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.regime_mapping = data['regime_mapping']
        instance.is_fitted = data['is_fitted']
        return instance
```

### 2.5 LightGBM Signal Scorer

```python
# src/services/hmm_app/models/lightgbm_scorer.py

import lightgbm as lgb
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class SignalScore:
    signal_type: SignalType
    score: float           # 0-100
    confidence: float      # 0-1
    regime_alignment: str  # "aligned", "neutral", "contrary"
    features_importance: dict[str, float]

class LightGBMSignalScorer:
    """
    LightGBM-basierter Signal-Scorer.

    Bewertet Trading-Signale basierend auf:
    - Aktuellem Markt-Regime (von HMM)
    - Technischen Indikatoren
    - Price Action Features
    - Volumen-Profil
    """

    FEATURE_COLUMNS = [
        # Regime Features (von HMM)
        "regime_bull_prob",
        "regime_bear_prob",
        "regime_sideways_prob",
        "regime_highvol_prob",
        "regime_duration",

        # Technische Indikatoren
        "rsi_14",
        "macd_signal",
        "macd_histogram",
        "bb_position",  # Position in Bollinger Bands (0-1)
        "atr_normalized",

        # Trend Features
        "sma_20_slope",
        "sma_50_slope",
        "ema_cross_signal",  # 1 = bullish cross, -1 = bearish

        # Price Action
        "higher_highs_count",
        "lower_lows_count",
        "last_swing_type",  # 1 = higher high, -1 = lower low

        # Volumen
        "volume_sma_ratio",
        "volume_trend",
    ]

    def __init__(self):
        self.model: Optional[lgb.Booster] = None
        self.is_fitted = False
        self.feature_importance: dict[str, float] = {}

    def _prepare_features(
        self,
        ohlcv: pd.DataFrame,
        regime_probs: dict[str, float],
        regime_duration: int
    ) -> np.ndarray:
        """Berechnet alle Features für Scoring."""
        features = {}

        # Regime Features
        features["regime_bull_prob"] = regime_probs.get("bull_trend", 0)
        features["regime_bear_prob"] = regime_probs.get("bear_trend", 0)
        features["regime_sideways_prob"] = regime_probs.get("sideways", 0)
        features["regime_highvol_prob"] = regime_probs.get("high_volatility", 0)
        features["regime_duration"] = regime_duration

        # Technische Indikatoren (hier vereinfacht)
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        volume = ohlcv['volume'].values

        # RSI
        features["rsi_14"] = self._calc_rsi(close, 14)

        # MACD
        macd, signal, hist = self._calc_macd(close)
        features["macd_signal"] = 1 if macd > signal else -1
        features["macd_histogram"] = hist

        # Bollinger Bands Position
        bb_upper, bb_lower = self._calc_bollinger(close)
        features["bb_position"] = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # ATR
        features["atr_normalized"] = self._calc_atr(high, low, close) / close[-1]

        # SMA Slopes
        sma_20 = np.convolve(close, np.ones(20)/20, mode='valid')
        sma_50 = np.convolve(close, np.ones(50)/50, mode='valid')
        features["sma_20_slope"] = (sma_20[-1] - sma_20[-5]) / sma_20[-5] if len(sma_20) > 5 else 0
        features["sma_50_slope"] = (sma_50[-1] - sma_50[-5]) / sma_50[-5] if len(sma_50) > 5 else 0

        # EMA Cross
        ema_12 = self._calc_ema(close, 12)
        ema_26 = self._calc_ema(close, 26)
        features["ema_cross_signal"] = 1 if ema_12 > ema_26 else -1

        # Price Action
        features["higher_highs_count"] = self._count_higher_highs(high[-20:])
        features["lower_lows_count"] = self._count_lower_lows(low[-20:])
        features["last_swing_type"] = 1 if high[-1] > max(high[-10:-1]) else -1

        # Volume
        vol_sma = np.mean(volume[-20:])
        features["volume_sma_ratio"] = volume[-1] / vol_sma if vol_sma > 0 else 1
        features["volume_trend"] = (np.mean(volume[-5:]) - np.mean(volume[-20:-5])) / vol_sma

        return np.array([features[col] for col in self.FEATURE_COLUMNS])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple] = None
    ) -> "LightGBMSignalScorer":
        """
        Trainiert den LightGBM Scorer.

        Args:
            X: Feature Matrix
            y: Target (Signal-Qualität 0-100)
            eval_set: Validation Set für Early Stopping
        """
        train_data = lgb.Dataset(
            X, label=y,
            feature_name=self.FEATURE_COLUMNS
        )

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        }

        callbacks = [lgb.early_stopping(50)] if eval_set else []
        valid_sets = [lgb.Dataset(eval_set[0], label=eval_set[1])] if eval_set else []

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            callbacks=callbacks
        )

        # Feature Importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.FEATURE_COLUMNS, importance))
        self.is_fitted = True

        return self

    def score_signal(
        self,
        ohlcv: pd.DataFrame,
        regime_probs: dict[str, float],
        regime_duration: int,
        signal_type: SignalType
    ) -> SignalScore:
        """
        Bewertet ein Trading-Signal.

        Returns:
            SignalScore mit Score, Confidence und Regime-Alignment
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        features = self._prepare_features(ohlcv, regime_probs, regime_duration)

        # Prediction
        raw_score = self.model.predict([features])[0]
        score = np.clip(raw_score, 0, 100)

        # Confidence aus Feature-Verteilung
        confidence = self._calculate_confidence(features)

        # Regime Alignment
        alignment = self._check_regime_alignment(
            signal_type, regime_probs
        )

        return SignalScore(
            signal_type=signal_type,
            score=float(score),
            confidence=float(confidence),
            regime_alignment=alignment,
            features_importance=self.feature_importance
        )

    def _check_regime_alignment(
        self,
        signal_type: SignalType,
        regime_probs: dict[str, float]
    ) -> str:
        """Prüft ob Signal zum Regime passt."""
        bull_prob = regime_probs.get("bull_trend", 0)
        bear_prob = regime_probs.get("bear_trend", 0)

        if signal_type == SignalType.LONG:
            if bull_prob > 0.5:
                return "aligned"
            elif bear_prob > 0.5:
                return "contrary"
        elif signal_type == SignalType.SHORT:
            if bear_prob > 0.5:
                return "aligned"
            elif bull_prob > 0.5:
                return "contrary"

        return "neutral"

    # Helper methods für technische Indikatoren
    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_macd(
        self,
        prices: np.ndarray
    ) -> tuple[float, float, float]:
        ema_12 = self._calc_ema(prices, 12)
        ema_26 = self._calc_ema(prices, 26)
        macd = ema_12 - ema_26
        signal = self._calc_ema(np.array([macd]), 9)  # Simplified
        return macd, signal, macd - signal

    def _calc_ema(self, prices: np.ndarray, period: int) -> float:
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(prices, weights, mode='valid')[-1]

    def _calc_bollinger(
        self,
        prices: np.ndarray,
        period: int = 20
    ) -> tuple[float, float]:
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        return sma + 2*std, sma - 2*std

    def _calc_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return np.mean(tr[-period:])

    def _count_higher_highs(self, highs: np.ndarray) -> int:
        count = 0
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                count += 1
        return count

    def _count_lower_lows(self, lows: np.ndarray) -> int:
        count = 0
        for i in range(1, len(lows)):
            if lows[i] < lows[i-1]:
                count += 1
        return count

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Berechnet Konfidenz basierend auf Feature-Qualität."""
        # Vereinfacht: Höhere Konfidenz bei klaren Signalen
        regime_max = max(features[:4])  # Regime-Probs
        rsi = features[5]

        # Klare RSI-Extremwerte = höhere Konfidenz
        rsi_clarity = abs(rsi - 50) / 50

        return (regime_max + rsi_clarity) / 2
```

### 2.6 API-Endpoints

```python
# src/services/hmm_app/routers/regime_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

class RegimeDetectionRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    lookback: int = 500

class RegimeResponse(BaseModel):
    symbol: str
    timestamp: datetime
    current_regime: str
    regime_probability: float
    regime_duration: int
    transition_probabilities: dict[str, float]
    regime_history: Optional[List[dict]] = None

@router.post("/detect", response_model=RegimeResponse)
async def detect_regime(request: RegimeDetectionRequest):
    """
    Erkennt das aktuelle Markt-Regime.

    Regimes:
    - **bull_trend**: Aufwärtstrend mit niedriger Volatilität
    - **bear_trend**: Abwärtstrend mit erhöhter Volatilität
    - **sideways**: Seitwärtsbewegung
    - **high_volatility**: Hohe Volatilität, ungerichtete Bewegung
    """
    pass

@router.get("/history/{symbol}")
async def get_regime_history(
    symbol: str,
    days: int = 30,
    timeframe: str = "1h"
):
    """Historische Regime-Wechsel für ein Symbol."""
    pass


# src/services/hmm_app/routers/scoring_router.py

class SignalScoringRequest(BaseModel):
    symbol: str
    signal_type: str  # "long" oder "short"
    entry_price: Optional[float] = None
    timeframe: str = "1h"

class SignalScoreResponse(BaseModel):
    symbol: str
    signal_type: str
    score: float
    confidence: float
    regime_alignment: str
    current_regime: str
    recommendation: str
    feature_breakdown: dict

@router.post("/score", response_model=SignalScoreResponse)
async def score_signal(request: SignalScoringRequest):
    """
    Bewertet ein Trading-Signal basierend auf aktuellem Regime.

    Score-Interpretation:
    - **80-100**: Starkes Signal, gut zum Regime passend
    - **60-79**: Moderates Signal
    - **40-59**: Schwaches Signal, Vorsicht geboten
    - **0-39**: Schlechtes Signal, möglicherweise gegen Regime
    """
    pass

@router.post("/batch-score")
async def batch_score_signals(signals: List[SignalScoringRequest]):
    """Bewertet mehrere Signale gleichzeitig."""
    pass
```

### 2.7 Docker-Konfiguration

```dockerfile
# docker/services/hmm/Dockerfile

FROM python:3.11-slim
ARG SERVICE_PORT=3006

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \  # Für LightGBM
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hmm.txt .
RUN pip install --no-cache-dir -r requirements-hmm.txt

COPY src/ ./src/

RUN mkdir -p /app/data/models/hmm /app/logs

ENV PYTHONPATH=/app
ENV SERVICE_NAME=hmm
ENV PORT=${SERVICE_PORT}

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "src.services.hmm_app.main:app", "--host", "0.0.0.0", "--port", "3006"]
```

```txt
# requirements-hmm.txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
httpx>=0.25.0
numpy>=1.24.0
pandas>=2.0.0
hmmlearn>=0.3.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
loguru>=0.7.0
```

---

## 3. Embedder Service (Port 3007)

### 3.1 Zweck

Zentraler Embedding-Service für das gesamte System:

- **Einheitliche Embeddings**: Alle Services nutzen dieselben Embedding-Modelle
- **Multi-Modal**: Text, Zeitreihen, und Features
- **Caching**: Effiziente Wiederverwendung berechneter Embeddings
- **Model-Agnostic**: Austauschbare Backend-Modelle

### 3.2 Embedding-Typen

| Typ | Modell | Dimension | Verwendung |
|-----|--------|-----------|------------|
| **Text** | all-MiniLM-L6-v2 | 384 | RAG, Pattern-Beschreibungen |
| **Financial Text** | FinBERT | 768 | News, Sentiment |
| **Time Series** | TS2Vec (selbst trainiert) | 320 | OHLCV-Sequenzen |
| **Market Features** | Autoencoder | 128 | Technische Indikatoren |

### 3.3 Verzeichnisstruktur

```
src/services/embedder_app/
├── __init__.py
├── main.py                          # FastAPI Application
├── routers/
│   ├── __init__.py
│   ├── text_router.py               # Text-Embedding-Endpoints
│   ├── timeseries_router.py         # Zeitreihen-Embeddings
│   ├── feature_router.py            # Feature-Embeddings
│   └── system_router.py             # Health & Monitoring
├── models/
│   ├── __init__.py
│   ├── text_embedder.py             # Sentence Transformers
│   ├── finbert_embedder.py          # FinBERT für Finance
│   ├── ts2vec_embedder.py           # Time Series Embedder
│   ├── feature_autoencoder.py       # Feature Compression
│   └── schemas.py                   # Pydantic-Modelle
├── services/
│   ├── __init__.py
│   ├── embedding_service.py         # Haupt-Service
│   ├── cache_service.py             # Redis/Memory Cache
│   └── batch_processor.py           # Batch-Verarbeitung
└── utils/
    ├── __init__.py
    └── preprocessing.py             # Input-Normalisierung
```

### 3.4 Embedding Models

```python
# src/services/embedder_app/models/text_embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """Basis-Klasse für alle Embedder."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def embed(self, inputs: Union[str, List[str]]) -> np.ndarray:
        pass

class TextEmbedder(BaseEmbedder):
    """
    Allgemeiner Text-Embedder mit Sentence Transformers.
    Optimiert für semantische Ähnlichkeit.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda"
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Erzeugt Embeddings für Text(e).

        Args:
            texts: Einzelner Text oder Liste von Texten

        Returns:
            numpy array der Form (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # Für Cosine Similarity
        )

        return embeddings


# src/services/embedder_app/models/finbert_embedder.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Union

class FinBERTEmbedder(BaseEmbedder):
    """
    FinBERT für Finance-spezifische Texte.
    Optimiert für Finanznachrichten und Sentiment.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return 768

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)

            # Mean Pooling über Token-Embeddings
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Normalisieren
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


# src/services/embedder_app/models/ts2vec_embedder.py

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class TS2VecEncoder(nn.Module):
    """
    Time Series to Vector Encoder.
    Basierend auf TS2Vec Architektur für kontrastives Lernen.
    """

    def __init__(
        self,
        input_dim: int = 5,       # OHLCV
        hidden_dim: int = 64,
        output_dim: int = 320,
        depth: int = 10,
        kernel_size: int = 3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input Projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Dilated Causal Convolutions
        self.convs = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.convs.append(
                nn.Conv1d(
                    hidden_dim, hidden_dim,
                    kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation
                )
            )

        # Output Projection
        self.output_fc = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim)
        """
        # Input projection
        x = self.input_fc(x)  # (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)

        # Dilated convolutions with residual
        for conv in self.convs:
            residual = x
            x = conv(x)
            x = x[:, :, :residual.size(2)]  # Trim padding
            x = self.relu(x) + residual

        # Global average pooling
        x = x.mean(dim=2)  # (batch, hidden_dim)

        # Output projection
        x = self.layer_norm(x)
        x = self.output_fc(x)

        # L2 normalize
        x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x


class TimeSeriesEmbedder(BaseEmbedder):
    """
    Time Series Embedder für OHLCV-Daten.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = TS2VecEncoder().to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return 320

    def embed(self, sequences: np.ndarray) -> np.ndarray:
        """
        Erzeugt Embeddings für OHLCV-Sequenzen.

        Args:
            sequences: numpy array (n_sequences, seq_len, 5) für OHLCV

        Returns:
            numpy array (n_sequences, 320)
        """
        if sequences.ndim == 2:
            sequences = sequences[np.newaxis, ...]

        # Normalisierung pro Sequenz
        sequences = self._normalize(sequences)

        with torch.no_grad():
            x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
            embeddings = self.model(x)

        return embeddings.cpu().numpy()

    def _normalize(self, sequences: np.ndarray) -> np.ndarray:
        """Z-Score Normalisierung pro Sequenz."""
        mean = sequences.mean(axis=(1, 2), keepdims=True)
        std = sequences.std(axis=(1, 2), keepdims=True) + 1e-8
        return (sequences - mean) / std


# src/services/embedder_app/models/feature_autoencoder.py

class FeatureAutoencoder(nn.Module):
    """
    Autoencoder für Feature-Kompression.
    Reduziert hochdimensionale technische Indikatoren.
    """

    def __init__(
        self,
        input_dim: int = 50,      # Anzahl technische Indikatoren
        latent_dim: int = 128,
        hidden_dims: list = [256, 128]
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class FeatureEmbedder(BaseEmbedder):
    """Feature-Embedder mit Autoencoder."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = FeatureAutoencoder().to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return 128

    def embed(self, features: np.ndarray) -> np.ndarray:
        """
        Komprimiert Feature-Vektoren.

        Args:
            features: numpy array (n_samples, n_features)

        Returns:
            numpy array (n_samples, 128)
        """
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            _, embeddings = self.model(x)

        return embeddings.cpu().numpy()
```

### 3.5 Embedding Service

```python
# src/services/embedder_app/services/embedding_service.py

from enum import Enum
from typing import Union, List, Optional
import numpy as np
from dataclasses import dataclass
import hashlib
import json

from ..models.text_embedder import TextEmbedder
from ..models.finbert_embedder import FinBERTEmbedder
from ..models.ts2vec_embedder import TimeSeriesEmbedder
from ..models.feature_autoencoder import FeatureEmbedder

class EmbeddingType(str, Enum):
    TEXT = "text"
    FINANCIAL_TEXT = "financial_text"
    TIMESERIES = "timeseries"
    FEATURES = "features"

@dataclass
class EmbeddingResult:
    embedding: np.ndarray
    embedding_type: EmbeddingType
    dimension: int
    model_name: str
    cached: bool = False

class EmbeddingService:
    """
    Zentraler Service für alle Embedding-Operationen.

    Features:
    - Multi-Modal Embeddings (Text, TimeSeries, Features)
    - Automatisches Caching
    - Batch-Verarbeitung
    - Model Hot-Swapping
    """

    def __init__(self, device: str = "cuda", cache_enabled: bool = True):
        self.device = device
        self.cache_enabled = cache_enabled
        self._cache: dict[str, np.ndarray] = {}

        # Lazy Loading der Modelle
        self._text_embedder: Optional[TextEmbedder] = None
        self._finbert_embedder: Optional[FinBERTEmbedder] = None
        self._ts_embedder: Optional[TimeSeriesEmbedder] = None
        self._feature_embedder: Optional[FeatureEmbedder] = None

    @property
    def text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(device=self.device)
        return self._text_embedder

    @property
    def finbert_embedder(self) -> FinBERTEmbedder:
        if self._finbert_embedder is None:
            self._finbert_embedder = FinBERTEmbedder(device=self.device)
        return self._finbert_embedder

    @property
    def ts_embedder(self) -> TimeSeriesEmbedder:
        if self._ts_embedder is None:
            self._ts_embedder = TimeSeriesEmbedder(device=self.device)
        return self._ts_embedder

    @property
    def feature_embedder(self) -> FeatureEmbedder:
        if self._feature_embedder is None:
            self._feature_embedder = FeatureEmbedder(device=self.device)
        return self._feature_embedder

    def _get_cache_key(self, data: Union[str, list, np.ndarray], embedding_type: EmbeddingType) -> str:
        """Generiert einen Cache-Key für die Eingabe."""
        if isinstance(data, np.ndarray):
            data_str = data.tobytes()
        elif isinstance(data, list):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        hash_input = f"{embedding_type.value}:{data_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    async def embed_text(
        self,
        texts: Union[str, List[str]],
        use_finbert: bool = False
    ) -> EmbeddingResult:
        """
        Erzeugt Text-Embeddings.

        Args:
            texts: Text oder Liste von Texten
            use_finbert: True für Finance-optimierte Embeddings
        """
        embedding_type = EmbeddingType.FINANCIAL_TEXT if use_finbert else EmbeddingType.TEXT

        # Cache Check
        cache_key = self._get_cache_key(texts, embedding_type)
        if self.cache_enabled and cache_key in self._cache:
            return EmbeddingResult(
                embedding=self._cache[cache_key],
                embedding_type=embedding_type,
                dimension=self._cache[cache_key].shape[-1],
                model_name="finbert" if use_finbert else "all-MiniLM-L6-v2",
                cached=True
            )

        # Embedding berechnen
        embedder = self.finbert_embedder if use_finbert else self.text_embedder
        embeddings = embedder.embed(texts)

        # Cachen
        if self.cache_enabled:
            self._cache[cache_key] = embeddings

        return EmbeddingResult(
            embedding=embeddings,
            embedding_type=embedding_type,
            dimension=embeddings.shape[-1],
            model_name="finbert" if use_finbert else "all-MiniLM-L6-v2",
            cached=False
        )

    async def embed_timeseries(
        self,
        sequences: np.ndarray
    ) -> EmbeddingResult:
        """
        Erzeugt Time-Series-Embeddings für OHLCV-Daten.

        Args:
            sequences: numpy array (n_sequences, seq_len, 5)
        """
        cache_key = self._get_cache_key(sequences, EmbeddingType.TIMESERIES)

        if self.cache_enabled and cache_key in self._cache:
            return EmbeddingResult(
                embedding=self._cache[cache_key],
                embedding_type=EmbeddingType.TIMESERIES,
                dimension=320,
                model_name="ts2vec",
                cached=True
            )

        embeddings = self.ts_embedder.embed(sequences)

        if self.cache_enabled:
            self._cache[cache_key] = embeddings

        return EmbeddingResult(
            embedding=embeddings,
            embedding_type=EmbeddingType.TIMESERIES,
            dimension=320,
            model_name="ts2vec",
            cached=False
        )

    async def embed_features(
        self,
        features: np.ndarray
    ) -> EmbeddingResult:
        """
        Komprimiert Feature-Vektoren.

        Args:
            features: numpy array (n_samples, n_features)
        """
        cache_key = self._get_cache_key(features, EmbeddingType.FEATURES)

        if self.cache_enabled and cache_key in self._cache:
            return EmbeddingResult(
                embedding=self._cache[cache_key],
                embedding_type=EmbeddingType.FEATURES,
                dimension=128,
                model_name="feature_autoencoder",
                cached=True
            )

        embeddings = self.feature_embedder.embed(features)

        if self.cache_enabled:
            self._cache[cache_key] = embeddings

        return EmbeddingResult(
            embedding=embeddings,
            embedding_type=EmbeddingType.FEATURES,
            dimension=128,
            model_name="feature_autoencoder",
            cached=False
        )

    async def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Berechnet Cosine Similarity zwischen zwei Embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))

    def clear_cache(self) -> int:
        """Leert den Cache und gibt Anzahl gelöschter Einträge zurück."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_cache_stats(self) -> dict:
        """Gibt Cache-Statistiken zurück."""
        return {
            "entries": len(self._cache),
            "size_mb": sum(
                e.nbytes for e in self._cache.values()
            ) / (1024 * 1024)
        }


# Singleton Instance
embedding_service = EmbeddingService()
```

### 3.6 API-Endpoints

```python
# src/services/embedder_app/routers/text_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class TextEmbeddingRequest(BaseModel):
    texts: List[str]
    use_finbert: bool = False

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    cached: bool

@router.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: TextEmbeddingRequest):
    """
    Erzeugt Text-Embeddings.

    - **texts**: Liste von Texten
    - **use_finbert**: True für Finance-optimierte Embeddings
    """
    pass

@router.post("/similarity")
async def compute_similarity(text1: str, text2: str, use_finbert: bool = False):
    """Berechnet semantische Ähnlichkeit zwischen zwei Texten."""
    pass


# src/services/embedder_app/routers/timeseries_router.py

class TimeSeriesEmbeddingRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    sequence_length: int = 200

@router.post("/embed")
async def embed_timeseries(request: TimeSeriesEmbeddingRequest):
    """
    Erzeugt Embedding für OHLCV-Zeitreihe.

    Holt automatisch Daten via DataGateway.
    """
    pass

@router.post("/embed-batch")
async def embed_timeseries_batch(requests: List[TimeSeriesEmbeddingRequest]):
    """Batch-Verarbeitung für mehrere Symbole."""
    pass

@router.post("/find-similar")
async def find_similar_patterns(
    symbol: str,
    timeframe: str = "1h",
    top_k: int = 10
):
    """
    Findet ähnliche historische Muster.

    Verwendet Time-Series-Embeddings für Similarity Search.
    """
    pass
```

### 3.7 Docker-Konfiguration

```dockerfile
# docker/services/embedder/Dockerfile

FROM nvcr.io/nvidia/pytorch:23.08-py3
ARG SERVICE_PORT=3007

WORKDIR /app

# Modelle vorladen
RUN pip install --no-cache-dir \
    sentence-transformers>=2.2.2 \
    transformers>=4.30.0 \
    faiss-gpu>=1.7.4

# FinBERT vorladen
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('ProsusAI/finbert'); \
    AutoTokenizer.from_pretrained('ProsusAI/finbert')"

# Sentence Transformer vorladen
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY requirements-embedder.txt .
RUN pip install --no-cache-dir -r requirements-embedder.txt

COPY src/ ./src/

RUN mkdir -p /app/data/models/embedder /app/logs /app/cache

ENV PYTHONPATH=/app
ENV SERVICE_NAME=embedder
ENV PORT=${SERVICE_PORT}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "src.services.embedder_app.main:app", "--host", "0.0.0.0", "--port", "3007"]
```

---

## 4. Docker Compose Integration

```yaml
# docker-compose.microservices.yml (Ergänzung)

services:
  # ... bestehende Services ...

  tcn-service:
    image: trading-tcn:latest
    build:
      context: .
      dockerfile: docker/services/tcn/Dockerfile
      args:
        SERVICE_PORT: 3005
    container_name: trading-tcn
    ports:
      - "3005:3005"
    environment:
      - SERVICE_NAME=tcn
      - PORT=3005
      - DATA_SERVICE_URL=http://trading-data:3001
      - EMBEDDER_SERVICE_URL=http://trading-embedder:3007
    volumes:
      - tcn-models:/app/data/models/tcn
      - ./logs:/app/logs
    networks:
      - trading-net
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    depends_on:
      - data-service
      - embedder-service
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3005/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  hmm-service:
    image: trading-hmm:latest
    build:
      context: .
      dockerfile: docker/services/hmm/Dockerfile
      args:
        SERVICE_PORT: 3006
    container_name: trading-hmm
    ports:
      - "3006:3006"
    environment:
      - SERVICE_NAME=hmm
      - PORT=3006
      - DATA_SERVICE_URL=http://trading-data:3001
    volumes:
      - hmm-models:/app/data/models/hmm
      - ./logs:/app/logs
    networks:
      - trading-net
    deploy:
      resources:
        limits:
          memory: 4G
    depends_on:
      - data-service
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3006/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  embedder-service:
    image: trading-embedder:latest
    build:
      context: .
      dockerfile: docker/services/embedder/Dockerfile
      args:
        SERVICE_PORT: 3007
    container_name: trading-embedder
    ports:
      - "3007:3007"
    environment:
      - SERVICE_NAME=embedder
      - PORT=3007
      - EMBEDDING_CACHE_SIZE=10000
    volumes:
      - embedder-models:/app/data/models/embedder
      - embedder-cache:/app/cache
      - ./logs:/app/logs
    networks:
      - trading-net
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 12G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3007/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  tcn-models:
  hmm-models:
  embedder-models:
  embedder-cache:
```

---

## 5. Service-Abhängigkeiten

```
                    ┌─────────────────────┐
                    │    DATA SERVICE     │
                    │     (Port 3001)     │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ TCN-Pattern   │───▶│    Embedder     │◀───│  HMM-Regime     │
│ Service       │    │    Service      │    │  Service        │
│ Port 3005     │    │    Port 3007    │    │  Port 3006      │
└───────┬───────┘    └────────┬────────┘    └────────┬────────┘
        │                     │                      │
        │                     ▼                      │
        │            ┌─────────────────┐             │
        └───────────▶│   RAG Service   │◀────────────┘
                     │   Port 3003     │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │   LLM Service   │
                     │   Port 3004     │
                     └─────────────────┘
```

### Abhängigkeits-Matrix

| Service | Benötigt | Stellt bereit |
|---------|----------|---------------|
| **Embedder** | Data Service | Text/TS/Feature Embeddings |
| **TCN-Pattern** | Data Service, Embedder | Pattern Detection |
| **HMM-Regime** | Data Service | Regime Detection, Signal Scoring |
| **RAG** | Embedder | Vector Search, Knowledge Base |
| **LLM** | RAG, alle ML-Services | Trading Analysis |

---

## 6. API-Übersicht

### Port-Zuweisung

| Service | Port | Swagger UI |
|---------|------|------------|
| Frontend | 3000 | - |
| Data Service | 3001 | /docs |
| NHITS Service | 3002 | /docs |
| RAG Service | 3003 | /docs |
| LLM Service | 3004 | /docs |
| **TCN-Pattern** | **3005** | /docs |
| **HMM-Regime** | **3006** | /docs |
| **Embedder** | **3007** | /docs |

### Endpoint-Übersicht (Neue Services)

#### TCN-Pattern Service (3005)
```
POST /api/v1/detect              - Pattern-Erkennung
POST /api/v1/scan-all            - Alle Symbole scannen
GET  /api/v1/history/{symbol}    - Pattern-Historie
POST /api/v1/train               - Modell trainieren
GET  /api/v1/models              - Verfügbare Modelle
GET  /health                     - Health Check
```

#### HMM-Regime Service (3006)
```
POST /api/v1/regime/detect       - Regime erkennen
GET  /api/v1/regime/history/{symbol} - Regime-Historie
POST /api/v1/scoring/score       - Signal bewerten
POST /api/v1/scoring/batch-score - Batch-Bewertung
POST /api/v1/train/hmm           - HMM trainieren
POST /api/v1/train/scorer        - LightGBM trainieren
GET  /health                     - Health Check
```

#### Embedder Service (3007)
```
POST /api/v1/text/embed          - Text-Embedding
POST /api/v1/text/similarity     - Text-Ähnlichkeit
POST /api/v1/timeseries/embed    - TimeSeries-Embedding
POST /api/v1/timeseries/find-similar - Ähnliche Muster
POST /api/v1/features/embed      - Feature-Embedding
GET  /api/v1/cache/stats         - Cache-Statistiken
DELETE /api/v1/cache             - Cache leeren
GET  /health                     - Health Check
```

---

## 7. Implementierungs-Roadmap

### Phase 1: Embedder Service (Basis)
1. Text-Embedder implementieren
2. API-Endpoints erstellen
3. Caching-Layer aufbauen
4. Docker-Integration
5. Tests schreiben

### Phase 2: HMM-Regime Service
1. HMM-Modell implementieren
2. LightGBM Scorer implementieren
3. Training-Pipeline
4. API-Endpoints
5. Integration mit Data Service

### Phase 3: TCN-Pattern Service
1. TCN-Architektur implementieren
2. Pattern-Labeling-Service
3. Training mit historischen Daten
4. Inference-Pipeline
5. Integration mit Embedder

### Phase 4: Integration
1. RAG-Service Anbindung an Embedder
2. LLM-Service Erweiterung
3. Dashboard-Integration
4. End-to-End Tests

---

## 8. Hardware-Anforderungen

### GPU-Auslastung (Jetson Thor)

| Service | GPU-Bedarf | VRAM |
|---------|------------|------|
| NHITS | Hoch (Training) | 4-8 GB |
| TCN-Pattern | Hoch (Training) | 4-6 GB |
| Embedder | Mittel (Inference) | 4-6 GB |
| HMM-Regime | Niedrig (CPU-lastig) | 0-1 GB |
| RAG (FAISS) | Mittel (Search) | 2-4 GB |
| LLM (Ollama) | Sehr Hoch | 16-32 GB |

**Empfehlung**: Sequentielles Training, paralleler Inference.

---

## 9. Fazit

Die drei neuen Microservices erweitern das KI-Trading-System um:

1. **TCN-Pattern**: Robuste Chart-Pattern-Erkennung mit Deep Learning
2. **HMM-Regime**: Probabilistische Markt-Regime-Erkennung + Signal-Scoring
3. **Embedder**: Zentraler, effizienter Embedding-Service für alle ML-Komponenten

Die Services folgen der bestehenden Architektur:
- Data Gateway Pattern (alle Daten via Data Service)
- Async-First Design
- Docker-native Deployment
- Swagger UI Dokumentation
- Health Checks und Monitoring
