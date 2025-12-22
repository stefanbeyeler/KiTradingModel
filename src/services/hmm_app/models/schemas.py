"""Pydantic schemas for HMM-Regime Service."""

from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


class SignalType(str, Enum):
    """Trading signal types."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TrainingStatus(str, Enum):
    """Training status."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class RegimeDetectionRequest(BaseModel):
    """Request for regime detection."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    lookback: int = Field(default=500, description="Number of candles for analysis")
    include_history: bool = Field(default=False, description="Include regime history")

    model_config = {"json_schema_extra": {"example": {"symbol": "BTCUSD", "timeframe": "1h", "lookback": 500}}}


class RegimeDetectionResponse(BaseModel):
    """Response for regime detection."""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_regime: MarketRegime
    regime_probability: float
    regime_duration: int
    transition_probabilities: Dict[str, float]
    regime_history: Optional[List[Dict]] = None
    market_metrics: Dict


class RegimeHistoryEntry(BaseModel):
    """Single entry in regime history."""
    timestamp: str
    regime: MarketRegime
    probability: float
    duration: Optional[int] = None


class RegimeHistoryResponse(BaseModel):
    """Response for regime history."""
    symbol: str
    timeframe: str
    entries: List[RegimeHistoryEntry]
    total_count: int
    regime_distribution: Dict[str, float]


class SignalScoringRequest(BaseModel):
    """Request for signal scoring."""
    symbol: str = Field(..., description="Trading symbol")
    signal_type: SignalType = Field(..., description="Signal type (long/short)")
    entry_price: Optional[float] = Field(default=None, description="Proposed entry price")
    timeframe: str = Field(default="1h", description="Timeframe")

    model_config = {"json_schema_extra": {"example": {"symbol": "BTCUSD", "signal_type": "long", "timeframe": "1h"}}}


class SignalScoringResponse(BaseModel):
    """Response for signal scoring."""
    symbol: str
    signal_type: SignalType
    score: float = Field(..., description="Signal score (0-100)")
    confidence: float = Field(..., description="Confidence level (0-1)")
    regime_alignment: str = Field(..., description="aligned, neutral, or contrary")
    current_regime: MarketRegime
    recommendation: str
    feature_importance: Dict[str, float]
    risk_assessment: Dict


class BatchSignalRequest(BaseModel):
    """Request for batch signal scoring."""
    signals: List[SignalScoringRequest]


class BatchSignalResponse(BaseModel):
    """Response for batch signal scoring."""
    results: List[SignalScoringResponse]
    total_scored: int
    average_score: float


class TrainingRequest(BaseModel):
    """Request for model training."""
    symbols: List[str] = Field(..., description="Symbols for training")
    timeframe: str = Field(default="1h", description="Timeframe")
    lookback_days: int = Field(default=365, description="Days of historical data")
    model_type: str = Field(default="both", description="hmm, scorer, or both")

    model_config = {"json_schema_extra": {"example": {"symbols": ["BTCUSD", "ETHUSD"], "timeframe": "1h", "lookback_days": 365}}}


class TrainingResponse(BaseModel):
    """Response for training request."""
    status: TrainingStatus
    job_id: Optional[str] = None
    message: str
    hmm_trained: Optional[bool] = None
    scorer_trained: Optional[bool] = None
    metrics: Optional[Dict] = None


class ModelInfoResponse(BaseModel):
    """Information about loaded models."""
    hmm_model: Dict
    scorer_model: Dict
    device: str
