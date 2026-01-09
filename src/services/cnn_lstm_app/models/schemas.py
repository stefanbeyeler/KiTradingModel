"""
Pydantic Schemas für CNN-LSTM Multi-Task Service.

Definiert Request/Response-Modelle für alle API-Endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Markt-Regime Kategorien."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


class PatternType(str, Enum):
    """Chart-Pattern Typen (16 Patterns)."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    CUP_AND_HANDLE = "cup_and_handle"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"


class PriceDirection(str, Enum):
    """Preis-Richtung."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# =============================================================================
# Prediction Schemas
# =============================================================================

class PricePrediction(BaseModel):
    """Preis-Vorhersage Response."""
    current: float = Field(..., description="Aktueller Preis")
    forecast_1h: Optional[float] = Field(None, description="1-Stunden Vorhersage")
    forecast_4h: Optional[float] = Field(None, description="4-Stunden Vorhersage")
    forecast_1d: Optional[float] = Field(None, description="1-Tag Vorhersage")
    forecast_1w: Optional[float] = Field(None, description="1-Wochen Vorhersage")
    direction: PriceDirection = Field(..., description="Prognostizierte Richtung")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Konfidenz (0-1)")
    change_percent_1h: Optional[float] = Field(None, description="Prozentuale Aenderung 1h")
    change_percent_1d: Optional[float] = Field(None, description="Prozentuale Aenderung 1d")


class PatternPrediction(BaseModel):
    """Einzelne Pattern-Vorhersage."""
    type: PatternType = Field(..., description="Pattern-Typ")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Konfidenz (0-1)")
    direction: PriceDirection = Field(..., description="Erwartete Richtung")


class RegimeTransitionProbs(BaseModel):
    """Regime-Uebergangswahrscheinlichkeiten."""
    bull_trend: float = Field(..., ge=0.0, le=1.0)
    bear_trend: float = Field(..., ge=0.0, le=1.0)
    sideways: float = Field(..., ge=0.0, le=1.0)
    high_volatility: float = Field(..., ge=0.0, le=1.0)


class RegimePrediction(BaseModel):
    """Regime-Vorhersage Response."""
    current: MarketRegime = Field(..., description="Aktuelles Regime")
    probability: float = Field(..., ge=0.0, le=1.0, description="Wahrscheinlichkeit")
    transition_probs: RegimeTransitionProbs = Field(
        ..., description="Uebergangswahrscheinlichkeiten"
    )
    regime_duration_bars: Optional[int] = Field(
        None, description="Anzahl Bars im aktuellen Regime"
    )


class MultiTaskPrediction(BaseModel):
    """Kombinierte Multi-Task Vorhersage."""
    price: PricePrediction = Field(..., description="Preis-Vorhersage")
    patterns: list[PatternPrediction] = Field(
        default_factory=list, description="Erkannte Patterns (Konfidenz > 0.5)"
    )
    regime: RegimePrediction = Field(..., description="Regime-Vorhersage")


# =============================================================================
# API Response Schemas
# =============================================================================

class PredictionResponse(BaseModel):
    """Standard API-Response für Vorhersagen."""
    symbol: str = Field(..., description="Trading-Symbol")
    timeframe: str = Field(..., description="Timeframe (normalisiert: H1, D1, etc.)")
    timestamp: datetime = Field(..., description="Vorhersage-Zeitstempel (UTC)")
    predictions: MultiTaskPrediction = Field(..., description="Multi-Task Vorhersagen")
    model_version: str = Field(..., description="Modell-Version")
    inference_time_ms: float = Field(..., description="Inferenz-Zeit in Millisekunden")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSD",
                "timeframe": "H1",
                "timestamp": "2024-01-15T10:30:00Z",
                "predictions": {
                    "price": {
                        "current": 42500.0,
                        "forecast_1h": 42650.0,
                        "forecast_4h": 42800.0,
                        "forecast_1d": 43200.0,
                        "direction": "bullish",
                        "confidence": 0.72,
                        "change_percent_1h": 0.35,
                        "change_percent_1d": 1.65
                    },
                    "patterns": [
                        {
                            "type": "ascending_triangle",
                            "confidence": 0.68,
                            "direction": "bullish"
                        }
                    ],
                    "regime": {
                        "current": "bull_trend",
                        "probability": 0.78,
                        "transition_probs": {
                            "bull_trend": 0.82,
                            "sideways": 0.12,
                            "bear_trend": 0.04,
                            "high_volatility": 0.02
                        }
                    }
                },
                "model_version": "cnn-lstm-v1.0.0-20240115",
                "inference_time_ms": 45.2
            }
        }


class PriceOnlyResponse(BaseModel):
    """Response nur fuer Preis-Vorhersage."""
    symbol: str
    timeframe: str
    timestamp: datetime
    prediction: PricePrediction
    model_version: str
    inference_time_ms: float


class PatternsOnlyResponse(BaseModel):
    """Response nur fuer Pattern-Klassifikation."""
    symbol: str
    timeframe: str
    timestamp: datetime
    patterns: list[PatternPrediction]
    total_patterns_detected: int
    model_version: str
    inference_time_ms: float


class RegimeOnlyResponse(BaseModel):
    """Response nur fuer Regime-Vorhersage."""
    symbol: str
    timeframe: str
    timestamp: datetime
    regime: RegimePrediction
    model_version: str
    inference_time_ms: float


# =============================================================================
# Batch Request/Response
# =============================================================================

class BatchPredictionRequest(BaseModel):
    """Batch-Anfrage fuer mehrere Symbole."""
    symbols: list[str] = Field(..., min_length=1, max_length=50, description="Liste von Symbolen")
    timeframe: str = Field(default="H1", description="Timeframe fuer alle Symbole")
    tasks: list[str] = Field(
        default=["price", "patterns", "regime"],
        description="Zu berechnende Tasks"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["BTCUSD", "EURUSD", "AAPL"],
                "timeframe": "H1",
                "tasks": ["price", "patterns", "regime"]
            }
        }


class BatchPredictionItem(BaseModel):
    """Einzelnes Item in Batch-Response."""
    symbol: str
    success: bool
    prediction: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch-Response fuer mehrere Symbole."""
    timeframe: str
    timestamp: datetime
    total_symbols: int
    successful: int
    failed: int
    results: list[BatchPredictionItem]
    total_inference_time_ms: float


# =============================================================================
# System Schemas
# =============================================================================

class ModelInfo(BaseModel):
    """Informationen ueber ein trainiertes Modell."""
    model_id: str = Field(..., description="Eindeutige Modell-ID")
    version: str = Field(..., description="Modell-Version")
    created_at: datetime = Field(..., description="Erstellungszeitpunkt")
    timeframes: list[str] = Field(..., description="Unterstuetzte Timeframes")
    input_features: int = Field(..., description="Anzahl Input-Features")
    total_parameters: int = Field(..., description="Gesamtzahl Parameter")
    training_samples: Optional[int] = Field(None, description="Anzahl Trainingssamples")
    metrics: Optional[dict] = Field(None, description="Training-Metriken")
    is_active: bool = Field(default=True, description="Aktives Modell fuer Inference")


class ModelsListResponse(BaseModel):
    """Response fuer Modell-Liste."""
    models: list[ModelInfo]
    active_model: Optional[str] = Field(None, description="Aktuell aktives Modell")
    total_models: int


class HealthResponse(BaseModel):
    """Health-Check Response."""
    service: str = Field(default="cnn-lstm", description="Service-Name")
    status: str = Field(..., description="Service-Status")
    version: str = Field(..., description="Service-Version")
    model_loaded: bool = Field(..., description="Modell geladen")
    model_version: Optional[str] = Field(None, description="Geladene Modell-Version")
    gpu_available: bool = Field(default=False, description="GPU verfuegbar")
    uptime_seconds: float = Field(..., description="Uptime in Sekunden")
    timestamp: datetime = Field(..., description="Zeitstempel")

    class Config:
        json_schema_extra = {
            "example": {
                "service": "cnn-lstm",
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "model_version": "cnn-lstm-v1.0.0-20240115",
                "gpu_available": False,
                "uptime_seconds": 3600.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# =============================================================================
# Feature Configuration
# =============================================================================

class FeatureConfig(BaseModel):
    """Konfiguration fuer Feature-Engineering."""
    sequence_length: int = Field(default=168, description="Sequenzlaenge")
    input_features: int = Field(default=25, description="Anzahl Input-Features")
    include_ohlcv: bool = Field(default=True, description="OHLCV inkludieren")
    include_returns: bool = Field(default=True, description="Returns inkludieren")
    include_trend: bool = Field(default=True, description="Trend-Indikatoren inkludieren")
    include_momentum: bool = Field(default=True, description="Momentum-Indikatoren inkludieren")
    include_volatility: bool = Field(default=True, description="Volatilitaets-Indikatoren inkludieren")
    include_volume: bool = Field(default=True, description="Volumen-Indikatoren inkludieren")
    normalize: bool = Field(default=True, description="Features normalisieren")


# =============================================================================
# Timeframe Configuration
# =============================================================================

SEQUENCE_LENGTHS = {
    "M1": 480,   # 8 Stunden
    "M5": 288,   # 24 Stunden
    "M15": 192,  # 48 Stunden
    "M30": 192,  # 4 Tage
    "H1": 168,   # 1 Woche
    "H4": 168,   # 4 Wochen
    "D1": 120,   # ~6 Monate
    "W1": 52,    # 1 Jahr
    "MN": 24,    # 2 Jahre
}


def get_sequence_length(timeframe: str) -> int:
    """Gibt die Sequenzlaenge fuer einen Timeframe zurueck."""
    return SEQUENCE_LENGTHS.get(timeframe.upper(), 168)
