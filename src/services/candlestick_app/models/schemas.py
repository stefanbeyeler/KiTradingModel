"""Pydantic models for candlestick pattern detection."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, asdict


class PatternCategory(str, Enum):
    """Category of candlestick pattern."""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    INDECISION = "indecision"


class PatternDirection(str, Enum):
    """Direction signal of the pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternStrength(str, Enum):
    """Strength of the pattern signal."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class PatternType(str, Enum):
    """Types of candlestick patterns."""
    # Reversal Patterns - Bullish
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    BULLISH_ENGULFING = "bullish_engulfing"
    MORNING_STAR = "morning_star"
    PIERCING_LINE = "piercing_line"
    DRAGONFLY_DOJI = "dragonfly_doji"
    BULLISH_BELT_HOLD = "bullish_belt_hold"
    BULLISH_COUNTERATTACK = "bullish_counterattack"
    THREE_INSIDE_UP = "three_inside_up"
    BULLISH_ABANDONED_BABY = "bullish_abandoned_baby"
    TOWER_BOTTOM = "tower_bottom"

    # Reversal Patterns - Bearish
    SHOOTING_STAR = "shooting_star"
    HANGING_MAN = "hanging_man"
    BEARISH_ENGULFING = "bearish_engulfing"
    EVENING_STAR = "evening_star"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    GRAVESTONE_DOJI = "gravestone_doji"
    BEARISH_BELT_HOLD = "bearish_belt_hold"
    BEARISH_COUNTERATTACK = "bearish_counterattack"
    THREE_INSIDE_DOWN = "three_inside_down"
    BEARISH_ABANDONED_BABY = "bearish_abandoned_baby"
    TOWER_TOP = "tower_top"
    ADVANCE_BLOCK = "advance_block"
    BEARISH_ISLAND = "bearish_island"
    BULLISH_ISLAND = "bullish_island"

    # Reversal Patterns - Neutral
    DOJI = "doji"

    # Continuation Patterns
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    RISING_THREE_METHODS = "rising_three_methods"
    FALLING_THREE_METHODS = "falling_three_methods"

    # Indecision Patterns
    SPINNING_TOP = "spinning_top"
    BULLISH_HARAMI = "bullish_harami"
    BEARISH_HARAMI = "bearish_harami"
    HARAMI_CROSS = "harami_cross"


class Timeframe(str, Enum):
    """Trading timeframes."""
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class CandleData(BaseModel):
    """Single candle OHLC data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    @property
    def body_size(self) -> float:
        """Absolute size of the candle body."""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> float:
        """Size of upper shadow/wick."""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """Size of lower shadow/wick."""
        return min(self.open, self.close) - self.low

    @property
    def total_range(self) -> float:
        """Total range from high to low."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True if candle closed higher than opened."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True if candle closed lower than opened."""
        return self.close < self.open

    @property
    def is_valid_candle(self) -> bool:
        """
        Check if this candle has valid market data.

        Returns False for:
        - Flat candles where O=H=L=C (missing data, weekend, holiday)
        - Zero or negative range (data error)
        - Candles with no price movement at all

        This helps filter out invalid patterns detected on missing/corrupt data.
        """
        # Minimum range threshold (relative to price level)
        # For most instruments, 0.0001% is essentially zero movement
        min_relative_range = 0.000001

        # Check for zero range (O=H=L=C)
        if self.total_range <= 0:
            return False

        # Check for extremely small range relative to price (essentially flat)
        price_level = (self.high + self.low) / 2
        if price_level > 0:
            relative_range = self.total_range / price_level
            if relative_range < min_relative_range:
                return False

        return True


class DetectedPattern(BaseModel):
    """
    A detected candlestick pattern with trading context.

    Each pattern includes:
    - Type, category, and direction classification
    - Confidence scores (rule-based, AI-based, final combined)
    - Trading implication and description
    - Price and trend context
    """
    pattern_type: PatternType = Field(
        ...,
        description="Type of detected pattern (e.g., hammer, engulfing, doji)"
    )
    category: PatternCategory = Field(
        ...,
        description="Pattern category: reversal, continuation, or indecision"
    )
    direction: PatternDirection = Field(
        ...,
        description="Signal direction: bullish, bearish, or neutral"
    )
    strength: PatternStrength = Field(
        ...,
        description="Signal strength: weak, moderate, or strong"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp of the pattern (last candle)"
    )
    timeframe: Timeframe = Field(
        ...,
        description="Timeframe where pattern was detected"
    )

    # Confidence scores - now with rule/AI breakdown
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final combined confidence score (0.0-1.0)"
    )
    rule_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence from rule-based detection (0.0-1.0)"
    )
    ai_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence from AI model validation (0.0-1.0)"
    )

    # AI validation details
    ai_prediction: Optional[str] = Field(
        default=None,
        description="Pattern type predicted by AI model (may differ from rule-based)"
    )
    ai_agreement: Optional[bool] = Field(
        default=None,
        description="Whether AI agrees with rule-based detection"
    )
    validation_method: str = Field(
        default="rule",
        description="Validation method used: 'rule', 'model', or 'fallback'"
    )

    # Price context
    price_at_detection: float = Field(
        ...,
        description="Price at time of pattern detection"
    )
    candles_involved: int = Field(
        ...,
        ge=1,
        le=5,
        description="Number of candles forming the pattern (1-5)"
    )

    # Optional context
    trend_context: Optional[str] = Field(
        default=None,
        description="Prior trend context: uptrend, downtrend, or sideways"
    )
    support_resistance_nearby: Optional[bool] = Field(
        default=None,
        description="Whether pattern is near a support/resistance level"
    )

    # Description
    description: str = Field(
        default="",
        description="Human-readable description of the pattern"
    )
    trading_implication: str = Field(
        default="",
        description="Trading implication and recommended action"
    )


class TimeframePatterns(BaseModel):
    """Patterns detected in a specific timeframe."""
    timeframe: Timeframe = Field(
        ...,
        description="Timeframe of this analysis"
    )
    patterns: list[DetectedPattern] = Field(
        default_factory=list,
        description="List of detected patterns in this timeframe"
    )
    candles_analyzed: int = Field(
        default=0,
        description="Number of candles analyzed"
    )
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the analysis"
    )


class MultiTimeframePatternResult(BaseModel):
    """
    Result of multi-timeframe pattern scanning.

    Contains pattern analysis across M15, H1, H4, and D1 timeframes with:
    - Per-timeframe pattern lists
    - Aggregated bullish/bearish/neutral counts
    - Confluence score indicating alignment across timeframes
    - Dominant market direction
    """
    symbol: str = Field(
        ...,
        description="Analyzed trading symbol"
    )
    scan_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the scan"
    )

    # Results per timeframe
    m5: TimeframePatterns = Field(
        default_factory=lambda: TimeframePatterns(timeframe=Timeframe.M5),
        description="5-minute timeframe patterns"
    )
    m15: TimeframePatterns = Field(
        default_factory=lambda: TimeframePatterns(timeframe=Timeframe.M15),
        description="15-minute timeframe patterns"
    )
    h1: TimeframePatterns = Field(
        default_factory=lambda: TimeframePatterns(timeframe=Timeframe.H1),
        description="1-hour timeframe patterns"
    )
    h4: TimeframePatterns = Field(
        default_factory=lambda: TimeframePatterns(timeframe=Timeframe.H4),
        description="4-hour timeframe patterns"
    )
    d1: TimeframePatterns = Field(
        default_factory=lambda: TimeframePatterns(timeframe=Timeframe.D1),
        description="Daily timeframe patterns"
    )

    # Summary
    total_patterns_found: int = Field(
        default=0,
        description="Total number of patterns found across all timeframes"
    )
    dominant_direction: Optional[PatternDirection] = Field(
        default=None,
        description="Overall dominant direction (bullish, bearish, or neutral)"
    )
    confluence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confluence score (0.0-1.0). Higher = stronger alignment across timeframes"
    )

    # Aggregated signals
    bullish_patterns_count: int = Field(
        default=0,
        description="Number of bullish patterns across all timeframes"
    )
    bearish_patterns_count: int = Field(
        default=0,
        description="Number of bearish patterns across all timeframes"
    )
    neutral_patterns_count: int = Field(
        default=0,
        description="Number of neutral patterns across all timeframes"
    )

    # Most significant patterns
    strongest_pattern: Optional[DetectedPattern] = Field(
        default=None,
        description="Pattern with highest confidence score"
    )

    def calculate_summary(self):
        """Calculate summary statistics from detected patterns."""
        all_patterns = (
            self.m5.patterns +
            self.m15.patterns +
            self.h1.patterns +
            self.h4.patterns +
            self.d1.patterns
        )

        self.total_patterns_found = len(all_patterns)
        self.bullish_patterns_count = sum(1 for p in all_patterns if p.direction == PatternDirection.BULLISH)
        self.bearish_patterns_count = sum(1 for p in all_patterns if p.direction == PatternDirection.BEARISH)
        self.neutral_patterns_count = sum(1 for p in all_patterns if p.direction == PatternDirection.NEUTRAL)

        # Determine dominant direction
        if self.bullish_patterns_count > self.bearish_patterns_count:
            self.dominant_direction = PatternDirection.BULLISH
        elif self.bearish_patterns_count > self.bullish_patterns_count:
            self.dominant_direction = PatternDirection.BEARISH
        else:
            self.dominant_direction = PatternDirection.NEUTRAL

        # Find strongest pattern
        if all_patterns:
            self.strongest_pattern = max(all_patterns, key=lambda p: p.confidence)

        # Calculate confluence score
        if self.total_patterns_found > 0:
            # Higher score if patterns align across timeframes
            directions = [p.direction for p in all_patterns]
            if self.dominant_direction:
                aligned = sum(1 for d in directions if d == self.dominant_direction)
                self.confluence_score = aligned / len(directions)


class PatternScanRequest(BaseModel):
    """
    Request for candlestick pattern scanning.

    Example:
    ```json
    {
        "symbol": "BTCUSD",
        "timeframes": ["H1", "H4", "D1"],
        "lookback_candles": 100,
        "min_confidence": 0.6
    }
    ```
    """
    symbol: str = Field(
        ...,
        description="Trading symbol (e.g., BTCUSD, EURUSD, XAUUSD)",
        json_schema_extra={"example": "BTCUSD"}
    )
    timeframes: list[Timeframe] = Field(
        default=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        description="Timeframes to scan (M15, H1, H4, D1)"
    )
    lookback_candles: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of candles to analyze per timeframe (10-500)"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0.0-1.0). Higher values = fewer but more reliable patterns"
    )
    include_weak_patterns: bool = Field(
        default=False,
        description="Include weak strength patterns (confidence < 0.5)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "BTCUSD",
                    "timeframes": ["H1", "H4"],
                    "lookback_candles": 100,
                    "min_confidence": 0.6,
                    "include_weak_patterns": False
                }
            ]
        }
    }


class PatternScanResponse(BaseModel):
    """
    Response from candlestick pattern scanning.

    Contains detected patterns across all scanned timeframes with:
    - Per-timeframe pattern lists
    - Summary statistics (bullish/bearish counts)
    - Confluence score (alignment across timeframes)
    - Strongest detected pattern
    """
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracking"
    )
    symbol: str = Field(
        ...,
        description="Scanned trading symbol"
    )
    result: MultiTimeframePatternResult = Field(
        ...,
        description="Multi-timeframe pattern detection results"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    data_source: str = Field(
        default="easyinsight",
        description="Data source used (easyinsight or twelvedata)"
    )


@dataclass
class PatternHistoryEntry:
    """Ein Eintrag in der Pattern-History."""
    id: str
    timestamp: str  # ISO 8601 UTC - Pattern-Zeitpunkt (Kerzen-Close)
    detected_at: str  # ISO 8601 UTC - Scan-Zeitpunkt
    symbol: str
    pattern_type: str
    category: str  # reversal, continuation, indecision
    direction: str  # bullish, bearish, neutral
    strength: str  # weak, moderate, strong
    confidence: float
    timeframe: str
    price_at_detection: float
    description: str
    trading_implication: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
