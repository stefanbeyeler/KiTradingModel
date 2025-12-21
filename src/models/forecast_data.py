"""Pydantic models for NHITS forecasting data."""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class ForecastResult(BaseModel):
    """NHITS forecast result with predictions and confidence intervals."""

    symbol: str
    forecast_timestamp: datetime = Field(default_factory=_utc_now)
    horizon_hours: int = Field(default=24, description="Forecast horizon in hours")
    timeframe: str = Field(
        default="H1",
        description="Timeframe used: M15 (15-min), H1 (hourly), D1 (daily)"
    )

    # Price predictions per hour
    predicted_prices: List[float] = Field(
        default_factory=list,
        description="Predicted prices for each hour in the horizon"
    )
    confidence_low: List[float] = Field(
        default_factory=list,
        description="Lower confidence bound (10th percentile)"
    )
    confidence_high: List[float] = Field(
        default_factory=list,
        description="Upper confidence bound (90th percentile)"
    )

    # Aggregated predictions at key intervals
    predicted_price_1h: Optional[float] = Field(
        default=None,
        description="Predicted price in 1 hour"
    )
    predicted_price_4h: Optional[float] = Field(
        default=None,
        description="Predicted price in 4 hours"
    )
    predicted_price_24h: Optional[float] = Field(
        default=None,
        description="Predicted price in 24 hours"
    )

    # Percentage changes
    current_price: Optional[float] = Field(
        default=None,
        description="Current price at forecast time"
    )
    predicted_change_percent_1h: Optional[float] = Field(
        default=None,
        description="Predicted percentage change in 1 hour"
    )
    predicted_change_percent_4h: Optional[float] = Field(
        default=None,
        description="Predicted percentage change in 4 hours"
    )
    predicted_change_percent_24h: Optional[float] = Field(
        default=None,
        description="Predicted percentage change in 24 hours"
    )

    # Trend probabilities
    trend_up_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability that price will increase"
    )
    trend_down_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability that price will decrease"
    )

    # Volatility estimation
    predicted_volatility: Optional[float] = Field(
        default=None,
        description="Predicted volatility based on confidence interval width"
    )

    # Model confidence and metadata
    model_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall model confidence (inverse of volatility)"
    )
    last_training_date: Optional[datetime] = Field(
        default=None,
        description="When the model was last trained"
    )
    training_samples: Optional[int] = Field(
        default=None,
        description="Number of samples used in training"
    )
    training_mape: Optional[float] = Field(
        default=None,
        description="Mean Absolute Percentage Error from training"
    )

    def to_dict_for_llm(self) -> Dict[str, Any]:
        """Convert to dictionary format suitable for LLM prompt."""
        return {
            "predicted_price_1h": self.predicted_price_1h,
            "predicted_price_4h": self.predicted_price_4h,
            "predicted_price_24h": self.predicted_price_24h,
            "predicted_change_percent_1h": f"{self.predicted_change_percent_1h:+.2f}%" if self.predicted_change_percent_1h else "N/A",
            "predicted_change_percent_4h": f"{self.predicted_change_percent_4h:+.2f}%" if self.predicted_change_percent_4h else "N/A",
            "predicted_change_percent_24h": f"{self.predicted_change_percent_24h:+.2f}%" if self.predicted_change_percent_24h else "N/A",
            "confidence_low_24h": self.confidence_low[-1] if self.confidence_low else None,
            "confidence_high_24h": self.confidence_high[-1] if self.confidence_high else None,
            "trend_up_probability": f"{self.trend_up_probability:.1%}",
            "trend_down_probability": f"{self.trend_down_probability:.1%}",
            "model_confidence": f"{self.model_confidence:.1%}",
            "predicted_volatility": f"{self.predicted_volatility:.2%}" if self.predicted_volatility else "N/A",
        }


class ForecastConfig(BaseModel):
    """Configuration for a forecast request."""

    symbol: str
    horizon: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Forecast horizon in hours (1-168)"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence intervals in forecast"
    )
    retrain: bool = Field(
        default=False,
        description="Force model retraining before forecast"
    )
    timeframe: str = Field(
        default="H1",
        description="Data timeframe: H1 (hourly), M15 (15-min), D1 (daily)"
    )


class TrainingMetrics(BaseModel):
    """Training metrics for NHITS model."""
    final_loss: Optional[float] = None
    n_features: Optional[int] = None
    features_used: Optional[List[str]] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    rmse: Optional[float] = None


class ForecastTrainingResult(BaseModel):
    """Result of model training."""

    symbol: str
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    training_samples: int
    training_duration_seconds: float
    model_path: str
    metrics: TrainingMetrics = Field(
        default_factory=TrainingMetrics,
        description="Training metrics including features used"
    )
    success: bool = True
    error_message: Optional[str] = None


class ForecastModelInfo(BaseModel):
    """Information about a trained forecast model."""

    symbol: str
    model_exists: bool
    model_path: Optional[str] = None
    last_trained: Optional[datetime] = None
    training_samples: Optional[int] = None
    horizon: int = 24
    input_size: int = 168
    metrics: Dict[str, Any] = Field(default_factory=dict)
    is_favorite: bool = Field(default=False, description="Whether this symbol is marked as favorite in Data Service")
    category: Optional[str] = Field(default=None, description="Symbol category (forex, crypto, index, commodity)")


class TrainingProgressTiming(BaseModel):
    """Timing information for training progress."""

    elapsed_seconds: int = Field(description="Time elapsed since training started (seconds)")
    eta_seconds: Optional[int] = Field(
        default=None,
        description="Estimated time remaining (seconds)"
    )
    elapsed_formatted: str = Field(description="Elapsed time in human-readable format (e.g., '4m 30s')")
    eta_formatted: Optional[str] = Field(
        default=None,
        description="ETA in human-readable format (e.g., '6m 15s')"
    )


class TrainingProgressResults(BaseModel):
    """Training results breakdown."""

    successful: int = Field(default=0, description="Number of successfully trained models")
    failed: int = Field(default=0, description="Number of failed training attempts")
    skipped: int = Field(default=0, description="Number of skipped models (already up-to-date)")


class TrainingProgressResponse(BaseModel):
    """Real-time training progress information."""

    training_in_progress: bool = Field(description="Whether training is currently running")
    current_symbol: Optional[str] = Field(
        default=None,
        description="Symbol currently being trained"
    )
    total_symbols: int = Field(default=0, description="Total number of symbols to train")
    completed_symbols: int = Field(default=0, description="Number of symbols completed")
    remaining_symbols: int = Field(default=0, description="Number of symbols remaining")
    progress_percent: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Progress percentage (0-100)"
    )
    results: TrainingProgressResults = Field(
        default_factory=TrainingProgressResults,
        description="Breakdown of training results"
    )
    timing: Optional[TrainingProgressTiming] = Field(
        default=None,
        description="Timing information (only when training is active)"
    )
    cancelling: bool = Field(
        default=False,
        description="Whether training cancellation has been requested"
    )
    started_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when training started"
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message (when training is not active)"
    )
    last_training_run: Optional[str] = Field(
        default=None,
        description="ISO timestamp of last completed training run"
    )
