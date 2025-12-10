"""Pydantic models for trading data and analysis."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class TradeDirection(str, Enum):
    """Trading direction enumeration (new schema)."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalType(str, Enum):
    """Trading signal types (legacy, for backward compatibility)."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class ConfidenceLevel(str, Enum):
    """Confidence levels for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeSeriesData(BaseModel):
    """Time series data point from TimescaleDB."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    additional_data: Optional[dict] = None

    # Technical Indicators from TimescaleDB (optional, for NHITS multi-variate forecasting)
    rsi: Optional[float] = None
    macd_main: Optional[float] = None
    macd_signal: Optional[float] = None
    adx: Optional[float] = None
    adx_plus_di: Optional[float] = None
    adx_minus_di: Optional[float] = None
    atr: Optional[float] = None
    cci: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ma100: Optional[float] = None
    ichimoku_tenkan: Optional[float] = None
    ichimoku_kijun: Optional[float] = None
    strength_4h: Optional[float] = None
    strength_1d: Optional[float] = None
    strength_1w: Optional[float] = None


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None
    obv: Optional[float] = None


class TradingSignal(BaseModel):
    """Individual trading signal."""
    signal_type: SignalType
    indicator: str
    value: float
    description: str


class NHITSForecast(BaseModel):
    """NHITS forecast data embedded in MarketAnalysis."""
    predicted_price_1h: Optional[float] = None
    predicted_price_4h: Optional[float] = None
    predicted_price_24h: Optional[float] = None
    predicted_change_percent_1h: Optional[float] = None
    predicted_change_percent_4h: Optional[float] = None
    predicted_change_percent_24h: Optional[float] = None
    confidence_low_24h: Optional[float] = None
    confidence_high_24h: Optional[float] = None
    trend_up_probability: Optional[float] = None
    trend_down_probability: Optional[float] = None
    model_confidence: Optional[float] = None
    predicted_volatility: Optional[float] = None


class MarketAnalysis(BaseModel):
    """Comprehensive market analysis result."""
    symbol: str
    timestamp: datetime
    current_price: float
    price_change_24h: float
    price_change_7d: Optional[float] = None
    technical_indicators: TechnicalIndicators
    signals: list[TradingSignal]
    trend: str
    volatility: str
    support_levels: list[float] = []
    resistance_levels: list[float] = []

    # NHITS Neural Forecast Data
    nhits_forecast: Optional[NHITSForecast] = Field(
        default=None,
        description="NHITS neural network price forecast"
    )


class OHLCData(BaseModel):
    """OHLC price data for a specific timeframe."""
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None


class PriceData(BaseModel):
    """Current price data including bid/ask spread."""
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


class RSIIndicator(BaseModel):
    """RSI indicator data."""
    value: Optional[float] = None
    period: int = 14
    signal: Optional[str] = None  # overbought, oversold, neutral


class MACDIndicator(BaseModel):
    """MACD indicator data."""
    main_line: Optional[float] = None
    signal_line: Optional[float] = None
    histogram: Optional[float] = None
    parameters: str = "12,26,9"
    trend: Optional[str] = None  # bullish, bearish


class StochasticIndicator(BaseModel):
    """Stochastic oscillator data."""
    k_line: Optional[float] = None
    d_line: Optional[float] = None
    parameters: str = "5,3,3"
    signal: Optional[str] = None  # overbought, oversold, neutral


class CCIIndicator(BaseModel):
    """CCI indicator data."""
    value: Optional[float] = None
    period: int = 14


class ADXIndicator(BaseModel):
    """ADX indicator data."""
    main_line: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    period: int = 14
    trend_strength: Optional[str] = None  # strong, moderate, weak
    trend_direction: Optional[str] = None  # bullish, bearish


class MAIndicator(BaseModel):
    """Moving Average indicator data."""
    value: Optional[float] = None
    type: str = "SMA"
    period: int = 100


class IchimokuIndicator(BaseModel):
    """Ichimoku Cloud indicator data."""
    tenkan_sen: Optional[float] = None
    kijun_sen: Optional[float] = None
    senkou_span_a: Optional[float] = None
    senkou_span_b: Optional[float] = None
    chikou_span: Optional[float] = None
    parameters: str = "9,26,52"
    tk_signal: Optional[str] = None  # bullish, bearish
    cloud_signal: Optional[str] = None  # bullish, bearish


class BollingerBandsIndicator(BaseModel):
    """Bollinger Bands indicator data."""
    upper_band: Optional[float] = None
    middle_band: Optional[float] = None
    lower_band: Optional[float] = None
    period: int = 200
    std_dev: int = 2
    price_position: Optional[str] = None  # at_upper_band, at_lower_band, within_bands


class ATRIndicator(BaseModel):
    """ATR indicator data."""
    value: Optional[float] = None
    timeframe: str = "D1"


class RangeIndicator(BaseModel):
    """Daily range indicator data."""
    value: Optional[float] = None
    timeframe: str = "D1"


class PivotPoints(BaseModel):
    """Pivot points data."""
    r1: Optional[float] = None
    s1: Optional[float] = None
    timeframe: str = "M5"


class StrengthIndicators(BaseModel):
    """Multi-timeframe strength indicators."""
    h4: Optional[float] = None
    d1: Optional[float] = None
    w1: Optional[float] = None


class AllIndicators(BaseModel):
    """All technical indicators used for analysis."""
    rsi: Optional[RSIIndicator] = None
    macd: Optional[MACDIndicator] = None
    stochastic: Optional[StochasticIndicator] = None
    cci: Optional[CCIIndicator] = None
    adx: Optional[ADXIndicator] = None
    ma100: Optional[MAIndicator] = None
    ichimoku: Optional[IchimokuIndicator] = None
    bollinger_bands: Optional[BollingerBandsIndicator] = None
    atr: Optional[ATRIndicator] = None
    range: Optional[RangeIndicator] = None


class MarketDataSnapshot(BaseModel):
    """Complete market data snapshot used for analysis - matches symbol-info endpoint structure."""
    symbol: str
    data_timestamp: Optional[str] = None

    # OHLC Data (Multiple Timeframes)
    ohlc_d1: Optional[OHLCData] = None
    ohlc_h1: Optional[OHLCData] = None
    ohlc_m15: Optional[OHLCData] = None

    # Price Data
    price: Optional[PriceData] = None

    # Technical Indicators
    indicators: Optional[AllIndicators] = None

    # Support/Resistance (Pivot Points)
    pivot_points: Optional[PivotPoints] = None

    # Strength Indicators (Multi-Timeframe)
    strength: Optional[StrengthIndicators] = None


class TradingRecommendation(BaseModel):
    """Trading recommendation generated by the AI with full output schema."""
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # New schema fields (primary)
    direction: TradeDirection = Field(
        default=TradeDirection.NEUTRAL,
        description="Trading direction: LONG, SHORT, or NEUTRAL"
    )
    confidence_score: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Confidence score from 0-100"
    )
    setup_recommendation: str = Field(
        default="",
        description="Detailed description of the trading setup"
    )

    # Price Levels
    entry_price: Optional[float] = Field(
        default=None,
        description="Recommended entry price"
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stop loss price level"
    )
    take_profit_1: Optional[float] = Field(
        default=None,
        description="First take profit target (conservative)"
    )
    take_profit_2: Optional[float] = Field(
        default=None,
        description="Second take profit target (moderate)"
    )
    take_profit_3: Optional[float] = Field(
        default=None,
        description="Third take profit target (aggressive)"
    )

    # Risk Management
    risk_reward_ratio: Optional[float] = Field(
        default=None,
        description="Risk-to-reward ratio"
    )
    recommended_position_size: Optional[float] = Field(
        default=None,
        description="Recommended position size in lots"
    )
    max_risk_percent: Optional[float] = Field(
        default=None,
        description="Maximum risk as percentage of account balance"
    )

    # Analysis Details
    trend_analysis: str = Field(
        default="",
        description="Detailed trend analysis"
    )
    support_resistance: str = Field(
        default="",
        description="Key support and resistance levels"
    )
    key_levels: str = Field(
        default="",
        description="Important price levels to watch"
    )
    risk_factors: str = Field(
        default="",
        description="Potential risks for this trade"
    )
    trade_rationale: str = Field(
        default="",
        description="Why this trade makes sense"
    )

    # Legacy fields (for backward compatibility)
    signal: SignalType = Field(
        default=SignalType.HOLD,
        description="Legacy signal type"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Legacy confidence level"
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Legacy take profit (alias for take_profit_1)"
    )
    reasoning: str = Field(
        default="",
        description="Legacy reasoning field"
    )
    key_factors: list[str] = Field(
        default_factory=list,
        description="Legacy key factors list"
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Legacy risks list"
    )
    timeframe: str = Field(
        default="short_term",
        description="Trading timeframe"
    )

    # Market data used for analysis (new field)
    market_data: Optional[MarketDataSnapshot] = Field(
        default=None,
        description="Complete market data snapshot used for generating this recommendation"
    )


class AnalysisRequest(BaseModel):
    """Request for market analysis."""
    symbol: str
    lookback_days: int = 30
    include_technical: bool = True
    include_sentiment: bool = False
    custom_prompt: Optional[str] = None
    strategy_id: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response containing analysis and recommendation."""
    request_id: str
    symbol: str
    timestamp: datetime
    analysis: MarketAnalysis
    recommendation: TradingRecommendation
    rag_context: list[str] = []
    model_used: str
    processing_time_ms: float


class RAGDocument(BaseModel):
    """Document stored in the RAG system."""
    id: str
    content: str
    metadata: dict
    timestamp: datetime
    symbol: Optional[str] = None
    document_type: str  # e.g., "analysis", "news", "pattern", "indicator"


# ============================================
# Trading Strategy Models
# ============================================

class StrategyType(str, Enum):
    """Types of trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class IndicatorConfig(BaseModel):
    """Configuration for a technical indicator."""
    name: str
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=5.0)
    parameters: dict = Field(default_factory=dict)
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None


class TradingStrategy(BaseModel):
    """Trading strategy configuration."""
    id: str
    name: str
    description: str = ""
    strategy_type: StrategyType = StrategyType.CUSTOM
    risk_level: RiskLevel = RiskLevel.MODERATE

    # Indicator configurations
    indicators: list[IndicatorConfig] = Field(default_factory=list)

    # Signal thresholds
    min_buy_signals: int = Field(default=3, ge=1, le=10)
    min_sell_signals: int = Field(default=3, ge=1, le=10)

    # Risk management
    stop_loss_atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    take_profit_atr_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    max_position_size_percent: float = Field(default=5.0, ge=1.0, le=100.0)

    # Timeframe preferences
    preferred_timeframe: str = "short_term"
    lookback_days: int = Field(default=30, ge=7, le=365)

    # LLM prompt customization
    custom_prompt: Optional[str] = None
    use_rag_context: bool = True
    max_rag_documents: int = Field(default=5, ge=0, le=20)

    # Metadata
    is_active: bool = True
    is_default: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class StrategyCreateRequest(BaseModel):
    """Request to create a new trading strategy."""
    name: str
    description: str = ""
    strategy_type: StrategyType = StrategyType.CUSTOM
    risk_level: RiskLevel = RiskLevel.MODERATE
    indicators: list[IndicatorConfig] = Field(default_factory=list)
    min_buy_signals: int = Field(default=3, ge=1, le=10)
    min_sell_signals: int = Field(default=3, ge=1, le=10)
    stop_loss_atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    take_profit_atr_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    preferred_timeframe: str = "short_term"
    lookback_days: int = Field(default=30, ge=7, le=365)
    custom_prompt: Optional[str] = None
    use_rag_context: bool = True


class StrategyUpdateRequest(BaseModel):
    """Request to update an existing trading strategy."""
    name: Optional[str] = None
    description: Optional[str] = None
    strategy_type: Optional[StrategyType] = None
    risk_level: Optional[RiskLevel] = None
    indicators: Optional[list[IndicatorConfig]] = None
    min_buy_signals: Optional[int] = None
    min_sell_signals: Optional[int] = None
    stop_loss_atr_multiplier: Optional[float] = None
    take_profit_atr_multiplier: Optional[float] = None
    preferred_timeframe: Optional[str] = None
    lookback_days: Optional[int] = None
    custom_prompt: Optional[str] = None
    use_rag_context: Optional[bool] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None
