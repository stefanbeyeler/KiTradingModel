"""Pydantic schemas for TCN-Pattern Service."""

from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class PatternType(str, Enum):
    """Supported chart pattern types."""
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


class TrainingStatus(str, Enum):
    """Training status."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class PatternDetectionRequest(BaseModel):
    """Request for pattern detection."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    lookback: int = Field(default=200, description="Number of candles to analyze")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    patterns: Optional[List[str]] = Field(default=None, description="Filter for specific patterns")

    model_config = {"json_schema_extra": {"example": {"symbol": "BTCUSD", "timeframe": "1h", "lookback": 200, "threshold": 0.6}}}


class DetectedPattern(BaseModel):
    """A detected pattern."""
    pattern_type: str = Field(..., description="Type of pattern detected")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    start_index: int = Field(..., description="Start index in the data")
    end_index: int = Field(..., description="End index in the data")
    start_time: Optional[str] = Field(default=None, description="Start timestamp")
    end_time: Optional[str] = Field(default=None, description="End timestamp")
    price_target: Optional[float] = Field(default=None, description="Projected price target")
    invalidation_level: Optional[float] = Field(default=None, description="Pattern invalidation level")
    pattern_height: Optional[float] = Field(default=None, description="Height of the pattern")
    direction: Optional[str] = Field(default=None, description="Bullish or bearish")


class PatternDetectionResponse(BaseModel):
    """Response for pattern detection."""
    symbol: str
    timeframe: str
    timestamp: datetime
    patterns: List[DetectedPattern]
    total_patterns: int
    market_context: dict
    model_version: str


class PatternScanRequest(BaseModel):
    """Request for scanning multiple symbols."""
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to scan (None = all active)")
    timeframe: str = Field(default="1h", description="Timeframe")
    threshold: float = Field(default=0.6, description="Minimum confidence")
    min_patterns: int = Field(default=1, description="Minimum patterns to report")


class SymbolPatternResult(BaseModel):
    """Pattern result for a single symbol."""
    symbol: str
    patterns: List[DetectedPattern]
    scan_time_ms: float


class PatternScanResponse(BaseModel):
    """Response for pattern scan."""
    timestamp: datetime
    timeframe: str
    threshold: float
    results: List[SymbolPatternResult]
    total_symbols: int
    symbols_with_patterns: int
    total_patterns: int
    scan_duration_ms: float


class PatternHistoryEntry(BaseModel):
    """Historical pattern detection."""
    pattern_type: str
    confidence: float
    detected_at: datetime
    start_time: str
    end_time: str
    outcome: Optional[str] = None
    target_reached: Optional[bool] = None
    actual_move: Optional[float] = None


class PatternHistoryResponse(BaseModel):
    """Response for pattern history."""
    symbol: str
    timeframe: str
    patterns: List[PatternHistoryEntry]
    total_count: int
    accuracy_stats: Optional[dict] = None


class TrainingRequest(BaseModel):
    """Request for model training."""
    symbols: List[str] = Field(..., description="Symbols to use for training")
    timeframe: str = Field(default="1h", description="Timeframe")
    lookback_days: int = Field(default=365, description="Days of historical data")
    epochs: int = Field(default=100, description="Training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    validation_split: float = Field(default=0.2, description="Validation split")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")

    model_config = {"json_schema_extra": {"example": {"symbols": ["BTCUSD", "ETHUSD", "EURUSD"], "timeframe": "1h", "lookback_days": 365, "epochs": 100}}}


class TrainingResponse(BaseModel):
    """Response for training request."""
    status: TrainingStatus
    job_id: Optional[str] = None
    message: str = ""
    started_at: Optional[str] = None
    progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    best_loss: Optional[float] = None
    metrics: Optional[dict] = None
    samples_count: Optional[int] = None


class ModelInfoResponse(BaseModel):
    """Information about the loaded model."""
    model_version: str
    trained_on: Optional[datetime] = None
    pattern_classes: List[str]
    num_parameters: int
    input_sequence_length: int
    device: str
    last_training_metrics: Optional[dict] = None
