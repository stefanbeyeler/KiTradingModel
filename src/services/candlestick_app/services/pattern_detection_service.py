"""Candlestick Pattern Detection Service.

Erkennt Candlestick-Muster ueber mehrere Timeframes:
- Reversal: Hammer, Shooting Star, Doji, Engulfing, Morning/Evening Star
- Continuation: Three White Soldiers, Three Black Crows
- Indecision: Spinning Top, Harami

Multi-Timeframe Scanning: M5, M15, H1, H4, D1

WICHTIG: Alle Datenzugriffe erfolgen ueber den DataGatewayService.
"""

import asyncio
import uuid
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

import httpx
from loguru import logger

from ..models.schemas import (
    CandleData,
    DetectedPattern,
    MultiTimeframePatternResult,
    PatternCategory,
    PatternDirection,
    PatternScanRequest,
    PatternScanResponse,
    PatternStrength,
    PatternType,
    Timeframe,
    TimeframePatterns,
)
from .rule_config_service import rule_config_service

# Timeframe-Mapping für TwelveData API
# (Vermeidet Import von src.config.timeframes wegen torch-Abhängigkeit)
TIMEFRAME_TO_TWELVEDATA = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1day",
    "W1": "1week",
    "MN": "1month",
}


def to_twelvedata(timeframe: str) -> str:
    """Konvertiert Standard-Timeframe zu TwelveData-Format."""
    tf_upper = timeframe.upper()
    if tf_upper in TIMEFRAME_TO_TWELVEDATA:
        return TIMEFRAME_TO_TWELVEDATA[tf_upper]
    # Fallback: Wenn bereits TwelveData-Format, direkt zurückgeben
    return timeframe.lower()


# Data Service URL
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")


class CandlestickPatternService:
    """
    Service fuer die Erkennung von Candlestick-Mustern.

    Unterstuetzt:
    - Reversal Patterns (Umkehrmuster)
    - Continuation Patterns (Fortsetzungsmuster)
    - Indecision Patterns (Unentschlossenheitsmuster)
    - Multi-Timeframe Scanning
    """

    def __init__(self):
        self._pattern_descriptions = self._init_pattern_descriptions()
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _init_pattern_descriptions(self) -> dict[PatternType, dict]:
        """Initialize pattern descriptions and trading implications."""
        return {
            # Reversal Patterns - Bullish
            PatternType.HAMMER: {
                "description": "Hammer Pattern - kleine Kerze mit langem unteren Schatten",
                "implication": "Potenzielle bullische Umkehr nach Abwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.INVERTED_HAMMER: {
                "description": "Invertierter Hammer - kleine Kerze mit langem oberen Schatten",
                "implication": "Potenzielle bullische Umkehr, Bestätigung erforderlich",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.BULLISH_ENGULFING: {
                "description": "Bullish Engulfing - große grüne Kerze umschließt vorherige rote Kerze",
                "implication": "Starkes bullisches Umkehrsignal",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.MORNING_STAR: {
                "description": "Morning Star - 3-Kerzen-Muster mit Doji/kleiner Kerze in der Mitte",
                "implication": "Zuverlässiges bullisches Umkehrsignal am Ende eines Abwärtstrends",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.PIERCING_LINE: {
                "description": "Piercing Line - grüne Kerze öffnet unter und schließt über 50% der vorherigen roten Kerze",
                "implication": "Bullisches Umkehrsignal, weniger stark als Engulfing",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.DRAGONFLY_DOJI: {
                "description": "Dragonfly Doji - Doji mit langem unteren Schatten",
                "implication": "Potenzielle bullische Umkehr am Support",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },

            # Reversal Patterns - Bearish
            PatternType.SHOOTING_STAR: {
                "description": "Shooting Star - kleine Kerze mit langem oberen Schatten",
                "implication": "Potenzielle bärische Umkehr nach Aufwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.HANGING_MAN: {
                "description": "Hanging Man - Hammer-Form in einem Aufwärtstrend",
                "implication": "Warnzeichen für potenzielle bärische Umkehr",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.BEARISH_ENGULFING: {
                "description": "Bearish Engulfing - große rote Kerze umschließt vorherige grüne Kerze",
                "implication": "Starkes bärisches Umkehrsignal",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.EVENING_STAR: {
                "description": "Evening Star - 3-Kerzen-Muster mit Doji/kleiner Kerze in der Mitte",
                "implication": "Zuverlässiges bärisches Umkehrsignal am Ende eines Aufwärtstrends",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.DARK_CLOUD_COVER: {
                "description": "Dark Cloud Cover - rote Kerze öffnet über und schließt unter 50% der vorherigen grünen Kerze",
                "implication": "Bärisches Umkehrsignal, weniger stark als Engulfing",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.GRAVESTONE_DOJI: {
                "description": "Gravestone Doji - Doji mit langem oberen Schatten",
                "implication": "Potenzielle bärische Umkehr am Widerstand",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },

            # Reversal - Neutral Doji
            PatternType.DOJI: {
                "description": "Doji - Open und Close fast identisch",
                "implication": "Marktunentschlossenheit, potenzielle Trendwende",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.NEUTRAL,
            },

            # Continuation Patterns
            PatternType.THREE_WHITE_SOLDIERS: {
                "description": "Three White Soldiers - drei aufeinanderfolgende lange grüne Kerzen",
                "implication": "Starkes bullisches Fortsetzungssignal",
                "category": PatternCategory.CONTINUATION,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.THREE_BLACK_CROWS: {
                "description": "Three Black Crows - drei aufeinanderfolgende lange rote Kerzen",
                "implication": "Starkes bärisches Fortsetzungssignal",
                "category": PatternCategory.CONTINUATION,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.RISING_THREE_METHODS: {
                "description": "Rising Three Methods - bullische Fortsetzung mit kleinen Korrekturkerzen",
                "implication": "Bullischer Trend wird wahrscheinlich fortgesetzt",
                "category": PatternCategory.CONTINUATION,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.FALLING_THREE_METHODS: {
                "description": "Falling Three Methods - bärische Fortsetzung mit kleinen Korrekturkerzen",
                "implication": "Bärischer Trend wird wahrscheinlich fortgesetzt",
                "category": PatternCategory.CONTINUATION,
                "direction": PatternDirection.BEARISH,
            },

            # Indecision Patterns
            PatternType.SPINNING_TOP: {
                "description": "Spinning Top - kleine Kerze mit langen Schatten beiderseits",
                "implication": "Marktunentschlossenheit, Konsolidierung möglich",
                "category": PatternCategory.INDECISION,
                "direction": PatternDirection.NEUTRAL,
            },
            PatternType.BULLISH_HARAMI: {
                "description": "Bullish Harami - kleine grüne Kerze innerhalb der vorherigen roten Kerze",
                "implication": "Potenzielle bullische Umkehr, Bestätigung erforderlich",
                "category": PatternCategory.INDECISION,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.BEARISH_HARAMI: {
                "description": "Bearish Harami - kleine rote Kerze innerhalb der vorherigen grünen Kerze",
                "implication": "Potenzielle bärische Umkehr, Bestätigung erforderlich",
                "category": PatternCategory.INDECISION,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.HARAMI_CROSS: {
                "description": "Harami Cross - Doji innerhalb der vorherigen Kerze",
                "implication": "Stärkeres Signal als normales Harami",
                "category": PatternCategory.INDECISION,
                "direction": PatternDirection.NEUTRAL,
            },
            # New Patterns - Belt Hold
            PatternType.BULLISH_BELT_HOLD: {
                "description": "Bullish Belt Hold - lange grüne Kerze ohne unteren Schatten",
                "implication": "Starkes bullisches Signal nach Abwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.BEARISH_BELT_HOLD: {
                "description": "Bearish Belt Hold - lange rote Kerze ohne oberen Schatten",
                "implication": "Starkes bärisches Signal nach Aufwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            # New Patterns - Counterattack
            PatternType.BULLISH_COUNTERATTACK: {
                "description": "Bullish Counterattack - zwei Kerzen schliessen auf gleichem Niveau nach Gap",
                "implication": "Bullisches Umkehrsignal nach Abwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.BEARISH_COUNTERATTACK: {
                "description": "Bearish Counterattack - zwei Kerzen schliessen auf gleichem Niveau nach Gap",
                "implication": "Bärisches Umkehrsignal nach Aufwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            # New Patterns - Three Inside
            PatternType.THREE_INSIDE_UP: {
                "description": "Three Inside Up - Bullish Harami mit Bestätigungskerze",
                "implication": "Bestätigtes bullisches Umkehrsignal",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.THREE_INSIDE_DOWN: {
                "description": "Three Inside Down - Bearish Harami mit Bestätigungskerze",
                "implication": "Bestätigtes bärisches Umkehrsignal",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            # New Patterns - Abandoned Baby
            PatternType.BULLISH_ABANDONED_BABY: {
                "description": "Bullish Abandoned Baby - Doji schwebt unter beiden Nachbarkerzen",
                "implication": "Sehr seltenes, starkes bullisches Umkehrsignal",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.BEARISH_ABANDONED_BABY: {
                "description": "Bearish Abandoned Baby - Doji schwebt über beiden Nachbarkerzen",
                "implication": "Sehr seltenes, starkes bärisches Umkehrsignal",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            # New Patterns - Tower
            PatternType.TOWER_BOTTOM: {
                "description": "Tower Bottom - grosse Abwärtskerze, kleine Kerzen, grosse Aufwärtskerze",
                "implication": "Bullisches Umkehrsignal nach Stauchungsphase",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
            PatternType.TOWER_TOP: {
                "description": "Tower Top - grosse Aufwärtskerze, kleine Kerzen, grosse Abwärtskerze",
                "implication": "Bärisches Umkehrsignal nach Stauchungsphase",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            # New Patterns - Advance Block & Island
            PatternType.ADVANCE_BLOCK: {
                "description": "Advance Block - drei bullische Kerzen mit abnehmender Stärke",
                "implication": "Warnsignal für nachlassende Kaufkraft, bärische Umkehr möglich",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.BEARISH_ISLAND: {
                "description": "Bearish Island - Kerzengruppe isoliert durch Gaps auf beiden Seiten",
                "implication": "Starkes bärisches Umkehrsignal nach Aufwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BEARISH,
            },
            PatternType.BULLISH_ISLAND: {
                "description": "Bullish Island - Kerzengruppe isoliert durch Gaps auf beiden Seiten",
                "implication": "Starkes bullisches Umkehrsignal nach Abwärtstrend",
                "category": PatternCategory.REVERSAL,
                "direction": PatternDirection.BULLISH,
            },
        }

    # ==================== Pattern Detection Methods ====================

    def _is_doji(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check if candle is a Doji (very small body relative to range).

        A Doji is detected when:
        - Candle has valid market data (not a flat/missing data candle)
        - Body is less than configured % of the candle's total range (PRIMARY criterion)

        The body_to_range check is the primary criterion because a true Doji must
        have an extremely small body regardless of market volatility.
        Parameters are configurable via rule_config_service.
        """
        # CRITICAL: Reject flat/invalid candles (missing data, weekend, etc.)
        # These would otherwise be falsely detected as Doji patterns
        if not candle.is_valid_candle:
            return False

        # Get configurable threshold (default to 0.05 = 5%)
        body_to_range_threshold = rule_config_service.get_param("doji", "body_to_range_ratio") or 0.05

        body_to_range_ratio = candle.body_size / candle.total_range if candle.total_range > 0 else 0

        # A Doji MUST have a very small body relative to its range
        # This is the definitive criterion for a Doji
        return body_to_range_ratio < body_to_range_threshold

    def _is_dragonfly_doji(self, candle: CandleData, avg_body: float) -> bool:
        """Check for Dragonfly Doji (long lower shadow, no upper shadow)."""
        if not self._is_doji(candle, avg_body):
            return False
        # Get configurable thresholds
        lower_shadow_min = rule_config_service.get_param("dragonfly_doji", "lower_shadow_min_ratio") or 0.6
        upper_shadow_max = rule_config_service.get_param("dragonfly_doji", "upper_shadow_max_ratio") or 0.1
        return (
            candle.lower_shadow > candle.total_range * lower_shadow_min and
            candle.upper_shadow < candle.total_range * upper_shadow_max
        )

    def _is_gravestone_doji(self, candle: CandleData, avg_body: float) -> bool:
        """Check for Gravestone Doji (long upper shadow, no lower shadow)."""
        if not self._is_doji(candle, avg_body):
            return False
        # Get configurable thresholds
        upper_shadow_min = rule_config_service.get_param("gravestone_doji", "upper_shadow_min_ratio") or 0.6
        lower_shadow_max = rule_config_service.get_param("gravestone_doji", "lower_shadow_max_ratio") or 0.1
        return (
            candle.upper_shadow > candle.total_range * upper_shadow_min and
            candle.lower_shadow < candle.total_range * lower_shadow_max
        )

    def _is_hammer(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Hammer pattern.
        - Small body at the top
        - Long lower shadow (at least 2x body)
        - Little or no upper shadow

        Uses configurable parameters from rule_config_service.
        """
        # Reject invalid/flat candles (missing data)
        if not candle.is_valid_candle:
            return False

        # Get configurable parameters
        params = rule_config_service.get_pattern_params("hammer")
        body_max_ratio = params.get("body_max_ratio", 0.35)
        lower_shadow_min_ratio = params.get("lower_shadow_min_ratio", 0.5)
        upper_shadow_max_ratio = params.get("upper_shadow_max_ratio", 0.15)
        shadow_to_body_min = params.get("shadow_to_body_min", 2.0)

        body_ratio = candle.body_size / candle.total_range
        lower_shadow_ratio = candle.lower_shadow / candle.total_range
        upper_shadow_ratio = candle.upper_shadow / candle.total_range

        return (
            body_ratio < body_max_ratio and
            lower_shadow_ratio > lower_shadow_min_ratio and
            upper_shadow_ratio < upper_shadow_max_ratio and
            candle.lower_shadow >= candle.body_size * shadow_to_body_min
        )

    def _is_inverted_hammer(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Inverted Hammer pattern.
        - Small body at the bottom
        - Long upper shadow (at least 2x body)
        - Little or no lower shadow

        Uses configurable parameters from rule_config_service.
        """
        # Reject invalid/flat candles (missing data)
        if not candle.is_valid_candle:
            return False

        # Get configurable parameters
        params = rule_config_service.get_pattern_params("inverted_hammer")
        body_max_ratio = params.get("body_max_ratio", 0.35)
        upper_shadow_min_ratio = params.get("upper_shadow_min_ratio", 0.5)
        lower_shadow_max_ratio = params.get("lower_shadow_max_ratio", 0.15)
        shadow_to_body_min = params.get("shadow_to_body_min", 2.0)

        body_ratio = candle.body_size / candle.total_range
        upper_shadow_ratio = candle.upper_shadow / candle.total_range
        lower_shadow_ratio = candle.lower_shadow / candle.total_range

        return (
            body_ratio < body_max_ratio and
            upper_shadow_ratio > upper_shadow_min_ratio and
            lower_shadow_ratio < lower_shadow_max_ratio and
            candle.upper_shadow >= candle.body_size * shadow_to_body_min
        )

    def _is_shooting_star(self, candle: CandleData, prev_candle: CandleData, avg_body: float) -> bool:
        """
        Check for Shooting Star pattern (Inverted Hammer after uptrend).

        Uses configurable parameters from rule_config_service.
        """
        # Must be in an uptrend (previous candle bullish)
        if not prev_candle.is_bullish:
            return False

        # Get configurable gap tolerance
        params = rule_config_service.get_pattern_params("shooting_star")
        gap_tolerance = params.get("gap_tolerance", 0.998)

        # Must gap up or open at/near previous high
        if candle.open < prev_candle.close * gap_tolerance:
            return False

        return self._is_inverted_hammer(candle, avg_body)

    def _is_hanging_man(self, candle: CandleData, prev_candle: CandleData, avg_body: float) -> bool:
        """
        Check for Hanging Man pattern (Hammer after uptrend).

        Uses configurable parameters from rule_config_service.
        """
        # Must be in an uptrend (previous candle bullish)
        if not prev_candle.is_bullish:
            return False

        # Get configurable gap tolerance
        params = rule_config_service.get_pattern_params("hanging_man")
        gap_tolerance = params.get("gap_tolerance", 0.998)

        # Must gap up or open at/near previous high
        if candle.open < prev_candle.close * gap_tolerance:
            return False

        return self._is_hammer(candle, avg_body)

    def _is_spinning_top(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Spinning Top pattern.
        - Small body
        - Long shadows on both sides (roughly equal)

        Uses configurable parameters from rule_config_service.
        """
        # Reject invalid/flat candles (missing data)
        if not candle.is_valid_candle:
            return False

        # Get configurable parameters
        params = rule_config_service.get_pattern_params("spinning_top")
        body_max_ratio = params.get("body_max_ratio", 0.3)
        shadow_min_ratio = params.get("shadow_min_ratio", 0.25)
        shadow_balance_max_diff = params.get("shadow_balance_max_diff", 0.2)

        body_ratio = candle.body_size / candle.total_range
        upper_shadow_ratio = candle.upper_shadow / candle.total_range
        lower_shadow_ratio = candle.lower_shadow / candle.total_range

        # Small body, significant shadows on both sides
        return (
            body_ratio < body_max_ratio and
            upper_shadow_ratio > shadow_min_ratio and
            lower_shadow_ratio > shadow_min_ratio and
            abs(upper_shadow_ratio - lower_shadow_ratio) < shadow_balance_max_diff
        )

    def _is_bullish_engulfing(self, current: CandleData, prev: CandleData, avg_body: float = 0) -> bool:
        """
        Check for Bullish Engulfing pattern.
        - Previous candle is bearish (red)
        - Current candle is bullish (green)
        - Current body completely engulfs previous body
        - Both candles must have significant bodies (not dojis)
        - Engulfing candle must be significantly larger than the previous

        A valid Bullish Engulfing requires:
        1. Previous candle: bearish with meaningful body
        2. Current candle: bullish with body SIGNIFICANTLY larger than previous
        3. Current open below previous close (gap down or at same level)
        4. Current close above previous open (completely engulfs)
        5. The engulfing candle must represent a significant price move (>= 0.25%)
        """
        # Basic direction checks - MUST be correct colors
        # prev.is_bearish: close < open (red candle)
        # current.is_bullish: close > open (green candle)
        if not prev.is_bearish or not current.is_bullish:
            return False

        # Double-check: Previous candle must actually have lower close than open
        # This catches edge cases where floating point comparison might fail
        if prev.close >= prev.open:
            return False  # Not a bearish (red) candle

        # Current candle must actually have higher close than open
        if current.close <= current.open:
            return False  # Not a bullish (green) candle

        # Minimum percentage move check - engulfing candle must be significant
        # Body must be at least 0.25% of the price to filter out noise
        price_reference = max(current.open, current.close, prev.open, prev.close)
        if price_reference > 0:
            current_pct_move = (current.body_size / price_reference) * 100
            if current_pct_move < 0.25:
                return False  # Move too small (< 0.25% of price)

        # Minimum body size check - prevent false positives with tiny candles
        # Previous candle must have a meaningful body (at least 25% of its range)
        if prev.total_range > 0:
            prev_body_ratio = prev.body_size / prev.total_range
            if prev_body_ratio < 0.25:
                return False  # Previous candle is too small (almost doji)

        # Current candle must have a meaningful body (at least 40% of its range)
        if current.total_range > 0:
            current_body_ratio = current.body_size / current.total_range
            if current_body_ratio < 0.4:
                return False  # Current candle body too small

        # If avg_body is provided, ensure candles are significant
        if avg_body > 0:
            # Previous candle must be at least 50% of average
            if prev.body_size < avg_body * 0.5:
                return False  # Previous candle too small compared to average
            # Engulfing candle must be at least 100% of average (full size)
            if current.body_size < avg_body:
                return False  # Current candle too small compared to average

        # Engulfing candle must be at least 1.5x the size of the previous candle
        if current.body_size < prev.body_size * 1.5:
            return False  # Not a significant engulfing

        # Engulfing conditions:
        # For bearish prev candle: open is high, close is low
        # Current bullish must: open below prev close, close above prev open
        engulfs_body = (
            current.open <= prev.close and  # Opens at or below prev close
            current.close >= prev.open      # Closes at or above prev open
        )

        return engulfs_body

    def _is_bearish_engulfing(self, current: CandleData, prev: CandleData, avg_body: float = 0) -> bool:
        """
        Check for Bearish Engulfing pattern.
        - Previous candle is bullish (green)
        - Current candle is bearish (red)
        - Current body completely engulfs previous body
        - Both candles must have significant bodies (not dojis)
        - Engulfing candle must be significantly larger than the previous

        A valid Bearish Engulfing requires:
        1. Previous candle: bullish with meaningful body
        2. Current candle: bearish with body SIGNIFICANTLY larger than previous
        3. Current open above previous close (gap up or at same level)
        4. Current close below previous open (completely engulfs)
        5. The engulfing candle must represent a significant price move (>= 0.25%)
        """
        # Basic direction checks - MUST be correct colors
        # prev.is_bullish: close > open (green candle)
        # current.is_bearish: close < open (red candle)
        if not prev.is_bullish or not current.is_bearish:
            return False

        # Double-check: Previous candle must actually have higher close than open
        # This catches edge cases where floating point comparison might fail
        if prev.close <= prev.open:
            return False  # Not a bullish (green) candle

        # Current candle must actually have lower close than open
        if current.close >= current.open:
            return False  # Not a bearish (red) candle

        # Minimum percentage move check - engulfing candle must be significant
        # Body must be at least 0.25% of the price to filter out noise
        price_reference = max(current.open, current.close, prev.open, prev.close)
        if price_reference > 0:
            current_pct_move = (current.body_size / price_reference) * 100
            if current_pct_move < 0.25:
                return False  # Move too small (< 0.25% of price)

        # Minimum body size check - prevent false positives with tiny candles
        # Previous candle must have a meaningful body (at least 25% of its range)
        if prev.total_range > 0:
            prev_body_ratio = prev.body_size / prev.total_range
            if prev_body_ratio < 0.25:
                return False  # Previous candle is too small (almost doji)

        # Current candle must have a meaningful body (at least 40% of its range)
        if current.total_range > 0:
            current_body_ratio = current.body_size / current.total_range
            if current_body_ratio < 0.4:
                return False  # Current candle body too small

        # If avg_body is provided, ensure candles are significant
        if avg_body > 0:
            # Previous candle must be at least 50% of average
            if prev.body_size < avg_body * 0.5:
                return False  # Previous candle too small compared to average
            # Engulfing candle must be at least 100% of average (full size)
            if current.body_size < avg_body:
                return False  # Current candle too small compared to average

        # Engulfing candle must be at least 1.5x the size of the previous candle
        if current.body_size < prev.body_size * 1.5:
            return False  # Not a significant engulfing

        # Engulfing conditions:
        # For bullish prev candle: open is low, close is high
        # Current bearish must: open above prev close, close below prev open
        engulfs_body = (
            current.open >= prev.close and  # Opens at or above prev close
            current.close <= prev.open      # Closes at or below prev open
        )

        return engulfs_body

    def _is_bullish_harami(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Bullish Harami pattern.
        - Previous candle is bearish with large body
        - Current candle is bullish with small body contained within previous
        """
        return (
            prev.is_bearish and
            current.is_bullish and
            current.open > prev.close and
            current.close < prev.open and
            current.body_size < prev.body_size * 0.5
        )

    def _is_bearish_harami(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Bearish Harami pattern.
        - Previous candle is bullish with large body
        - Current candle is bearish with small body contained within previous
        """
        return (
            prev.is_bullish and
            current.is_bearish and
            current.open < prev.close and
            current.close > prev.open and
            current.body_size < prev.body_size * 0.5
        )

    def _is_harami_cross(self, current: CandleData, prev: CandleData, avg_body: float) -> bool:
        """
        Check for Harami Cross pattern.
        - Previous candle has large body
        - Current candle is a Doji contained within previous body
        """
        if not self._is_doji(current, avg_body):
            return False

        return (
            prev.body_size > avg_body * 0.8 and
            max(current.open, current.close) < max(prev.open, prev.close) and
            min(current.open, current.close) > min(prev.open, prev.close)
        )

    def _is_morning_star(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Morning Star pattern (3 candles).

        A valid Morning Star requires:
        1. First candle: LARGE bearish candle (significant selling pressure)
        2. Second candle: SMALL body candle that GAPS DOWN from first close
           - The gap is essential - shows exhaustion of selling
        3. Third candle: LARGE bullish candle that closes above midpoint of first
           - Confirms reversal with strong buying

        Key criteria for a valid Morning Star:
        - First candle must have substantial body (>= 80% of average)
        - Second candle must be very small (< 30% of average) - ideally a Doji
        - Gap down between first close and second high is REQUIRED
        - Third candle must be substantial (>= 80% of average)
        - Third must close above 50% of first candle's body
        """
        # Get configurable thresholds
        first_body_min = rule_config_service.get_param("morning_star", "first_body_min_avg") or 0.8
        second_body_max = rule_config_service.get_param("morning_star", "second_body_max_avg") or 0.3
        third_body_min = rule_config_service.get_param("morning_star", "third_body_min_avg") or 0.8

        # First candle: MUST be bearish with LARGE body
        if not first.is_bearish:
            return False
        if first.body_size < avg_body * first_body_min:
            return False
        # First candle should also have significant range (not just a small red candle)
        if first.total_range < avg_body * 0.5:
            return False

        # Second candle: MUST have SMALL body (ideally Doji-like)
        if second.body_size > avg_body * second_body_max:
            return False

        # CRITICAL: Gap down required - second candle's HIGH must be below first candle's CLOSE
        # This gap shows exhaustion of the downtrend
        if second.high > first.close:
            return False

        # Third candle: MUST be bullish with LARGE body
        if not third.is_bullish:
            return False
        if third.body_size < avg_body * third_body_min:
            return False

        # Third candle MUST close above the midpoint of the first candle's body
        first_midpoint = (first.open + first.close) / 2
        if third.close < first_midpoint:
            return False

        # Additional: Third candle should show strength (significant range)
        if third.total_range < avg_body * 0.5:
            return False

        return True

    def _is_evening_star(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Evening Star pattern (3 candles).

        A valid Evening Star requires:
        1. First candle: LARGE bullish candle (significant buying pressure)
        2. Second candle: SMALL body candle that GAPS UP from first close
           - The gap is essential - shows exhaustion of buying
        3. Third candle: LARGE bearish candle that closes below midpoint of first
           - Confirms reversal with strong selling

        Key criteria for a valid Evening Star:
        - First candle must have substantial body (>= 80% of average)
        - Second candle must be very small (< 30% of average) - ideally a Doji
        - Gap up between first close and second low is REQUIRED
        - Third candle must be substantial (>= 80% of average)
        - Third must close below 50% of first candle's body
        """
        # Get configurable thresholds
        first_body_min = rule_config_service.get_param("evening_star", "first_body_min_avg") or 0.8
        second_body_max = rule_config_service.get_param("evening_star", "second_body_max_avg") or 0.3
        third_body_min = rule_config_service.get_param("evening_star", "third_body_min_avg") or 0.8

        # First candle: MUST be bullish with LARGE body
        if not first.is_bullish:
            return False
        if first.body_size < avg_body * first_body_min:
            return False
        # First candle should also have significant range
        if first.total_range < avg_body * 0.5:
            return False

        # Second candle: MUST have SMALL body (ideally Doji-like)
        if second.body_size > avg_body * second_body_max:
            return False

        # CRITICAL: Gap up required - second candle's LOW must be above first candle's CLOSE
        # This gap shows exhaustion of the uptrend
        if second.low < first.close:
            return False

        # Third candle: MUST be bearish with LARGE body
        if not third.is_bearish:
            return False
        if third.body_size < avg_body * third_body_min:
            return False

        # Third candle MUST close below the midpoint of the first candle's body
        first_midpoint = (first.open + first.close) / 2
        if third.close > first_midpoint:
            return False

        # Additional: Third candle should show strength (significant range)
        if third.total_range < avg_body * 0.5:
            return False

        return True

    def _is_piercing_line(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Piercing Line pattern.
        - Previous candle is bearish
        - Current opens below previous low
        - Current closes above 50% of previous body
        """
        if not prev.is_bearish or not current.is_bullish:
            return False

        prev_midpoint = (prev.open + prev.close) / 2
        return (
            current.open < prev.low and
            current.close > prev_midpoint and
            current.close < prev.open
        )

    def _is_dark_cloud_cover(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Dark Cloud Cover pattern.
        - Previous candle is bullish
        - Current opens above previous high
        - Current closes below 50% of previous body
        """
        if not prev.is_bullish or not current.is_bearish:
            return False

        prev_midpoint = (prev.open + prev.close) / 2
        return (
            current.open > prev.high and
            current.close < prev_midpoint and
            current.close > prev.open
        )

    def _is_three_white_soldiers(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Three White Soldiers pattern.
        - Three consecutive bullish candles
        - Each opens within previous body and closes higher
        - Minimal upper shadows
        """
        candles = [first, second, third]

        # All must be bullish with significant bodies
        for c in candles:
            if not c.is_bullish or c.body_size < avg_body * 0.5:
                return False
            # Upper shadow should be small
            if c.upper_shadow > c.body_size * 0.3:
                return False

        # Each should close higher than previous
        if not (third.close > second.close > first.close):
            return False

        # Each should open within previous body
        if second.open < first.open or second.open > first.close:
            return False
        if third.open < second.open or third.open > second.close:
            return False

        return True

    def _is_three_black_crows(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Three Black Crows pattern.
        - Three consecutive bearish candles
        - Each opens within previous body and closes lower
        - Minimal lower shadows
        """
        candles = [first, second, third]

        # All must be bearish with significant bodies
        for c in candles:
            if not c.is_bearish or c.body_size < avg_body * 0.5:
                return False
            # Lower shadow should be small
            if c.lower_shadow > c.body_size * 0.3:
                return False

        # Each should close lower than previous
        if not (third.close < second.close < first.close):
            return False

        # Each should open within previous body
        if second.open > first.open or second.open < first.close:
            return False
        if third.open > second.open or third.open < second.close:
            return False

        return True

    # ==================== New Pattern Detection Methods ====================

    def _is_bullish_belt_hold(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Bullish Belt Hold pattern.
        - Long bullish body (opens at/near low)
        - Very small or no lower shadow
        - May have small upper shadow

        Uses configurable parameters from rule_config_service.
        """
        if not candle.is_valid_candle or not candle.is_bullish:
            return False

        params = rule_config_service.get_pattern_params("bullish_belt_hold")
        body_min_ratio = params.get("body_min_ratio", 0.75)
        lower_shadow_max_ratio = params.get("lower_shadow_max_ratio", 0.05)
        upper_shadow_max_ratio = params.get("upper_shadow_max_ratio", 0.15)

        body_ratio = candle.body_size / candle.total_range if candle.total_range > 0 else 0
        lower_shadow_ratio = candle.lower_shadow / candle.total_range if candle.total_range > 0 else 0
        upper_shadow_ratio = candle.upper_shadow / candle.total_range if candle.total_range > 0 else 0

        return (
            body_ratio >= body_min_ratio and
            lower_shadow_ratio <= lower_shadow_max_ratio and
            upper_shadow_ratio <= upper_shadow_max_ratio
        )

    def _is_bearish_belt_hold(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Bearish Belt Hold pattern.
        - Long bearish body (opens at/near high)
        - Very small or no upper shadow
        - May have small lower shadow

        Uses configurable parameters from rule_config_service.
        """
        if not candle.is_valid_candle or not candle.is_bearish:
            return False

        params = rule_config_service.get_pattern_params("bearish_belt_hold")
        body_min_ratio = params.get("body_min_ratio", 0.75)
        upper_shadow_max_ratio = params.get("upper_shadow_max_ratio", 0.05)
        lower_shadow_max_ratio = params.get("lower_shadow_max_ratio", 0.15)

        body_ratio = candle.body_size / candle.total_range if candle.total_range > 0 else 0
        upper_shadow_ratio = candle.upper_shadow / candle.total_range if candle.total_range > 0 else 0
        lower_shadow_ratio = candle.lower_shadow / candle.total_range if candle.total_range > 0 else 0

        return (
            body_ratio >= body_min_ratio and
            upper_shadow_ratio <= upper_shadow_max_ratio and
            lower_shadow_ratio <= lower_shadow_max_ratio
        )

    def _is_bullish_counterattack(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Bullish Counterattack pattern.
        - Previous candle is bearish
        - Current candle is bullish with gap down open
        - Both candles close at approximately the same level

        Uses configurable parameters from rule_config_service.
        """
        if not prev.is_bearish or not current.is_bullish:
            return False

        params = rule_config_service.get_pattern_params("bullish_counterattack")
        close_tolerance_pct = params.get("close_tolerance_pct", 0.05)
        body_min_ratio = params.get("body_min_ratio", 0.5)
        gap_min_pct = params.get("gap_min_pct", 0.2)

        # Both candles must have significant bodies
        if prev.total_range > 0:
            prev_body_ratio = prev.body_size / prev.total_range
            if prev_body_ratio < body_min_ratio:
                return False

        if current.total_range > 0:
            curr_body_ratio = current.body_size / current.total_range
            if curr_body_ratio < body_min_ratio:
                return False

        # Gap down required
        price_ref = prev.close
        gap_pct = ((prev.close - current.open) / price_ref) * 100 if price_ref > 0 else 0
        if gap_pct < gap_min_pct:
            return False

        # Closes must be at same level
        close_diff_pct = abs(current.close - prev.close) / prev.close * 100 if prev.close > 0 else float('inf')

        return close_diff_pct <= close_tolerance_pct

    def _is_bearish_counterattack(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Bearish Counterattack pattern.
        - Previous candle is bullish
        - Current candle is bearish with gap up open
        - Both candles close at approximately the same level

        Uses configurable parameters from rule_config_service.
        """
        if not prev.is_bullish or not current.is_bearish:
            return False

        params = rule_config_service.get_pattern_params("bearish_counterattack")
        close_tolerance_pct = params.get("close_tolerance_pct", 0.05)
        body_min_ratio = params.get("body_min_ratio", 0.5)
        gap_min_pct = params.get("gap_min_pct", 0.2)

        # Both candles must have significant bodies
        if prev.total_range > 0:
            prev_body_ratio = prev.body_size / prev.total_range
            if prev_body_ratio < body_min_ratio:
                return False

        if current.total_range > 0:
            curr_body_ratio = current.body_size / current.total_range
            if curr_body_ratio < body_min_ratio:
                return False

        # Gap up required
        price_ref = prev.close
        gap_pct = ((current.open - prev.close) / price_ref) * 100 if price_ref > 0 else 0
        if gap_pct < gap_min_pct:
            return False

        # Closes must be at same level
        close_diff_pct = abs(current.close - prev.close) / prev.close * 100 if prev.close > 0 else float('inf')

        return close_diff_pct <= close_tolerance_pct

    def _is_three_inside_up(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Three Inside Up pattern (bullish reversal).

        STRICT Criteria:
        - First candle: LARGE bearish candle (body >= 60% of range, body > avg_body)
        - Second candle: SMALL bullish candle COMPLETELY within first body (Harami)
        - Third candle: bullish, closes ABOVE first candle's open (confirmation)

        Uses configurable parameters from rule_config_service.
        """
        params = rule_config_service.get_pattern_params("three_inside_up")
        harami_body_max_ratio = params.get("harami_body_max_ratio", 0.5)
        first_body_min_ratio = params.get("first_body_min_ratio", 0.6)

        # First: must be bearish with LARGE body
        if not first.is_bearish:
            return False

        # First candle must have significant body (not a doji or spinning top)
        if first.total_range > 0:
            first_body_ratio = first.body_size / first.total_range
            if first_body_ratio < first_body_min_ratio:
                return False

        # First candle body must be larger than average
        if first.body_size < avg_body * 0.8:
            return False

        # Second: must be bullish
        if not second.is_bullish:
            return False

        # Second body must be COMPLETELY within first body (strict Harami)
        # For bearish first: close is bottom, open is top
        first_body_top = first.open
        first_body_bottom = first.close

        # Second candle's ENTIRE body must be within first body
        second_body_top = max(second.open, second.close)
        second_body_bottom = min(second.open, second.close)

        if second_body_top > first_body_top or second_body_bottom < first_body_bottom:
            return False

        # Second body must be significantly smaller than first
        if second.body_size > first.body_size * harami_body_max_ratio:
            return False

        # Third: must be bullish
        if not third.is_bullish:
            return False

        # Third must close ABOVE first's open (top of first's body)
        # This is the confirmation that breaks the resistance
        if third.close <= first.open:
            return False

        # Third candle should have meaningful body (not a doji)
        if third.total_range > 0 and third.body_size / third.total_range < 0.3:
            return False

        return True

    def _is_three_inside_down(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Three Inside Down pattern (bearish reversal).

        STRICT Criteria:
        - First candle: LARGE bullish candle (body >= 60% of range, body > avg_body)
        - Second candle: SMALL bearish candle COMPLETELY within first body (Harami)
        - Third candle: bearish, closes BELOW first candle's open (confirmation)

        Uses configurable parameters from rule_config_service.
        """
        params = rule_config_service.get_pattern_params("three_inside_down")
        harami_body_max_ratio = params.get("harami_body_max_ratio", 0.5)
        first_body_min_ratio = params.get("first_body_min_ratio", 0.6)

        # First: must be bullish with LARGE body
        if not first.is_bullish:
            return False

        # First candle must have significant body (not a doji or spinning top)
        if first.total_range > 0:
            first_body_ratio = first.body_size / first.total_range
            if first_body_ratio < first_body_min_ratio:
                return False

        # First candle body must be larger than average
        if first.body_size < avg_body * 0.8:
            return False

        # Second: must be bearish
        if not second.is_bearish:
            return False

        # Second body must be COMPLETELY within first body (strict Harami)
        # For bullish first: open is bottom, close is top
        first_body_top = first.close
        first_body_bottom = first.open

        # Second candle's ENTIRE body must be within first body
        second_body_top = max(second.open, second.close)
        second_body_bottom = min(second.open, second.close)

        if second_body_top > first_body_top or second_body_bottom < first_body_bottom:
            return False

        # Second body must be significantly smaller than first
        if second.body_size > first.body_size * harami_body_max_ratio:
            return False

        # Third: must be bearish
        if not third.is_bearish:
            return False

        # Third must close BELOW first's open (bottom of first's body)
        # This is the confirmation that breaks the support
        if third.close >= first.open:
            return False

        # Third candle should have meaningful body (not a doji)
        if third.total_range > 0 and third.body_size / third.total_range < 0.3:
            return False

        return True

    def _is_bullish_abandoned_baby(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Bullish Abandoned Baby pattern.
        - First candle: bearish with large body
        - Second candle: Doji that gaps down (HIGH is below first's LOW)
        - Third candle: bullish with large body, gaps up (LOW is above second's HIGH)

        The Doji must be completely separated from both neighboring candles.
        Uses configurable parameters from rule_config_service.
        """
        params = rule_config_service.get_pattern_params("bullish_abandoned_baby")
        doji_body_max_ratio = params.get("doji_body_max_ratio", 0.1)
        first_body_min_ratio = params.get("first_body_min_ratio", 0.5)
        third_body_min_ratio = params.get("third_body_min_ratio", 0.5)

        # First: bearish with significant body
        if not first.is_bearish:
            return False
        if first.total_range > 0:
            first_body_ratio = first.body_size / first.total_range
            if first_body_ratio < first_body_min_ratio:
                return False

        # Second: Doji (very small body)
        if second.total_range > 0:
            second_body_ratio = second.body_size / second.total_range
            if second_body_ratio > doji_body_max_ratio:
                return False

        # Critical: Gap between first and second (second HIGH must be below first LOW)
        if second.high >= first.low:
            return False

        # Third: bullish with significant body
        if not third.is_bullish:
            return False
        if third.total_range > 0:
            third_body_ratio = third.body_size / third.total_range
            if third_body_ratio < third_body_min_ratio:
                return False

        # Critical: Gap between second and third (third LOW must be above second HIGH)
        if third.low <= second.high:
            return False

        return True

    def _is_bearish_abandoned_baby(
        self,
        first: CandleData,
        second: CandleData,
        third: CandleData,
        avg_body: float
    ) -> bool:
        """
        Check for Bearish Abandoned Baby pattern.
        - First candle: bullish with large body
        - Second candle: Doji that gaps up (LOW is above first's HIGH)
        - Third candle: bearish with large body, gaps down (HIGH is below second's LOW)

        The Doji must be completely separated from both neighboring candles.
        Uses configurable parameters from rule_config_service.
        """
        params = rule_config_service.get_pattern_params("bearish_abandoned_baby")
        doji_body_max_ratio = params.get("doji_body_max_ratio", 0.1)
        first_body_min_ratio = params.get("first_body_min_ratio", 0.5)
        third_body_min_ratio = params.get("third_body_min_ratio", 0.5)

        # First: bullish with significant body
        if not first.is_bullish:
            return False
        if first.total_range > 0:
            first_body_ratio = first.body_size / first.total_range
            if first_body_ratio < first_body_min_ratio:
                return False

        # Second: Doji (very small body)
        if second.total_range > 0:
            second_body_ratio = second.body_size / second.total_range
            if second_body_ratio > doji_body_max_ratio:
                return False

        # Critical: Gap between first and second (second LOW must be above first HIGH)
        if second.low <= first.high:
            return False

        # Third: bearish with significant body
        if not third.is_bearish:
            return False
        if third.total_range > 0:
            third_body_ratio = third.body_size / third.total_range
            if third_body_ratio < third_body_min_ratio:
                return False

        # Critical: Gap between second and third (third HIGH must be below second LOW)
        if third.high >= second.low:
            return False

        return True

    def _is_tower_bottom(self, candles: list[CandleData], avg_body: float) -> bool:
        """
        Check for Tower Bottom pattern.
        - First candle: large bearish body
        - Middle candles (2-10): small bodies in consolidation zone
        - Last candle: large bullish body

        Uses configurable parameters from rule_config_service.
        """
        if len(candles) < 4:  # Min: first + 2 inner + last
            return False

        params = rule_config_service.get_pattern_params("tower_bottom")
        outer_body_min_ratio = params.get("outer_body_min_ratio", 0.6)
        inner_body_max_ratio = params.get("inner_body_max_ratio", 0.4)
        min_inner_candles = int(params.get("min_inner_candles", 2))
        max_inner_candles = int(params.get("max_inner_candles", 10))

        inner_count = len(candles) - 2
        if inner_count < min_inner_candles or inner_count > max_inner_candles:
            return False

        first = candles[0]
        last = candles[-1]
        inner = candles[1:-1]

        # First: large bearish body
        if not first.is_bearish:
            return False
        if first.total_range > 0:
            first_body_ratio = first.body_size / first.total_range
            if first_body_ratio < outer_body_min_ratio:
                return False

        # Last: large bullish body
        if not last.is_bullish:
            return False
        if last.total_range > 0:
            last_body_ratio = last.body_size / last.total_range
            if last_body_ratio < outer_body_min_ratio:
                return False

        # Inner candles: small bodies
        for c in inner:
            if c.total_range > 0:
                body_ratio = c.body_size / c.total_range
                if body_ratio > inner_body_max_ratio:
                    return False

        # Inner candles should be in the upper area of first candle or slightly above
        first_high = max(first.open, first.close)  # Top of first candle's body
        for c in inner:
            if c.low < first.close * 0.99:  # Allow small tolerance
                return False

        return True

    def _is_tower_top(self, candles: list[CandleData], avg_body: float) -> bool:
        """
        Check for Tower Top pattern.
        - First candle: large bullish body
        - Middle candles (2-10): small bodies in consolidation zone
        - Last candle: large bearish body

        Uses configurable parameters from rule_config_service.
        """
        if len(candles) < 4:  # Min: first + 2 inner + last
            return False

        params = rule_config_service.get_pattern_params("tower_top")
        outer_body_min_ratio = params.get("outer_body_min_ratio", 0.6)
        inner_body_max_ratio = params.get("inner_body_max_ratio", 0.4)
        min_inner_candles = int(params.get("min_inner_candles", 2))
        max_inner_candles = int(params.get("max_inner_candles", 10))

        inner_count = len(candles) - 2
        if inner_count < min_inner_candles or inner_count > max_inner_candles:
            return False

        first = candles[0]
        last = candles[-1]
        inner = candles[1:-1]

        # First: large bullish body
        if not first.is_bullish:
            return False
        if first.total_range > 0:
            first_body_ratio = first.body_size / first.total_range
            if first_body_ratio < outer_body_min_ratio:
                return False

        # Last: large bearish body
        if not last.is_bearish:
            return False
        if last.total_range > 0:
            last_body_ratio = last.body_size / last.total_range
            if last_body_ratio < outer_body_min_ratio:
                return False

        # Inner candles: small bodies
        for c in inner:
            if c.total_range > 0:
                body_ratio = c.body_size / c.total_range
                if body_ratio > inner_body_max_ratio:
                    return False

        # Inner candles should be in the upper area of first candle or slightly above
        first_high = max(first.open, first.close)  # Top of first candle's body
        for c in inner:
            if c.high > first.close * 1.01:  # Allow small tolerance above first close
                pass  # Inner candles can be at or above first close
            if c.low < first.open * 0.99:  # Should not go too far below first open
                return False

        return True

    def _is_advance_block(self, c1: CandleData, c2: CandleData, c3: CandleData, avg_body: float) -> bool:
        """
        Check for Advance Block pattern (bearish reversal signal).

        Three bullish candles where:
        - All three candles are bullish
        - Each candle opens within the body of the previous candle
        - Bodies progressively get smaller (decreasing strength)
        - Upper shadows progressively get longer (selling pressure increasing)
        - Appears after an uptrend

        This pattern signals weakening buying pressure and potential reversal.
        """
        # All three must be bullish
        if not (c1.is_bullish and c2.is_bullish and c3.is_bullish):
            return False

        # Each opens within or near the body of the previous
        if c2.open < min(c1.open, c1.close) or c2.open > max(c1.open, c1.close):
            return False
        if c3.open < min(c2.open, c2.close) or c3.open > max(c2.open, c2.close):
            return False

        # Each closes higher than the previous (upward progression)
        if c2.close <= c1.close or c3.close <= c2.close:
            return False

        # Bodies get progressively smaller (key characteristic)
        if not (c1.body_size > c2.body_size > c3.body_size):
            return False

        # Upper shadows get progressively larger (increasing selling pressure)
        if c1.upper_shadow >= c2.upper_shadow or c2.upper_shadow >= c3.upper_shadow:
            return False

        # Third candle should have significant upper shadow
        if c3.total_range > 0:
            upper_shadow_ratio = c3.upper_shadow / c3.total_range
            if upper_shadow_ratio < 0.2:  # At least 20% upper shadow
                return False

        return True

    def _is_bearish_island(self, candles: list[CandleData], avg_body: float) -> bool:
        """
        Check for Bearish Island Reversal pattern.

        Structure:
        - Gap up (first island candle's low > previous candle's high)
        - One or more candles forming the "island"
        - Gap down (last island candle's high < next candle's low becomes current close)

        This is a strong bearish reversal signal when found after an uptrend.
        """
        if len(candles) < 3:
            return False

        # The pattern needs: pre-gap candle, island candle(s), post-gap candle
        # For detection at current position, we look backwards
        pre_gap = candles[0]
        island = candles[1:-1] if len(candles) > 3 else [candles[1]]
        post_gap = candles[-1]

        if not island:
            return False

        first_island = island[0]
        last_island = island[-1]

        # Gap up: first island candle's low > pre-gap candle's high
        gap_up = first_island.low > pre_gap.high

        # Gap down: post-gap candle's high < last island candle's low
        gap_down = post_gap.high < last_island.low

        if not (gap_up and gap_down):
            return False

        # Post-gap candle should be bearish (confirming the reversal)
        if not post_gap.is_bearish:
            return False

        # Pre-gap should be bullish (uptrend context)
        if not pre_gap.is_bullish:
            return False

        return True

    def _is_bullish_island(self, candles: list[CandleData], avg_body: float) -> bool:
        """
        Check for Bullish Island Reversal pattern.

        Structure:
        - Gap down (first island candle's high < previous candle's low)
        - One or more candles forming the "island"
        - Gap up (last island candle's low > next candle's high becomes current close)

        This is a strong bullish reversal signal when found after a downtrend.
        """
        if len(candles) < 3:
            return False

        pre_gap = candles[0]
        island = candles[1:-1] if len(candles) > 3 else [candles[1]]
        post_gap = candles[-1]

        if not island:
            return False

        first_island = island[0]
        last_island = island[-1]

        # Gap down: first island candle's high < pre-gap candle's low
        gap_down = first_island.high < pre_gap.low

        # Gap up: post-gap candle's low > last island candle's high
        gap_up = post_gap.low > last_island.high

        if not (gap_down and gap_up):
            return False

        # Post-gap candle should be bullish (confirming the reversal)
        if not post_gap.is_bullish:
            return False

        # Pre-gap should be bearish (downtrend context)
        if not pre_gap.is_bearish:
            return False

        return True

    # ==================== Main Detection Logic ====================

    def _calculate_avg_body(self, candles: list[CandleData], lookback: int = 20) -> float:
        """Calculate average body size for context."""
        recent = candles[-lookback:] if len(candles) >= lookback else candles
        bodies = [c.body_size for c in recent if c.body_size > 0]
        return sum(bodies) / len(bodies) if bodies else 0

    def _determine_trend(self, candles: list[CandleData], lookback: int = 10) -> str:
        """Determine trend direction from recent candles."""
        if len(candles) < lookback:
            return "sideways"

        recent = candles[-lookback:]
        closes = [c.close for c in recent]

        # Simple linear regression approximation
        first_half_avg = sum(closes[:len(closes)//2]) / (len(closes)//2)
        second_half_avg = sum(closes[len(closes)//2:]) / (len(closes) - len(closes)//2)

        change_pct = (second_half_avg - first_half_avg) / first_half_avg * 100

        if change_pct > 1:
            return "uptrend"
        elif change_pct < -1:
            return "downtrend"
        return "sideways"

    def _create_pattern(
        self,
        pattern_type: PatternType,
        candle: CandleData,
        timeframe: Timeframe,
        confidence: float,
        candles_involved: int = 1,
        trend_context: Optional[str] = None,
        rule_confidence: Optional[float] = None,
        ai_confidence: Optional[float] = None,
        ai_prediction: Optional[str] = None,
        ai_agreement: Optional[bool] = None,
        validation_method: str = "rule",
        pattern_candles: Optional[list[CandleData]] = None
    ) -> DetectedPattern:
        """Create a DetectedPattern object."""
        info = self._pattern_descriptions.get(pattern_type, {})

        # Convert pattern candles to dict format for storage
        pattern_candles_dict = None
        if pattern_candles:
            pattern_candles_dict = [
                {
                    "datetime": c.timestamp.isoformat() if hasattr(c.timestamp, 'isoformat') else str(c.timestamp),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                }
                for c in pattern_candles
            ]

        return DetectedPattern(
            pattern_type=pattern_type,
            category=info.get("category", PatternCategory.INDECISION),
            direction=info.get("direction", PatternDirection.NEUTRAL),
            strength=self._get_strength(confidence),
            timestamp=candle.timestamp,
            timeframe=timeframe,
            confidence=confidence,
            rule_confidence=rule_confidence,
            ai_confidence=ai_confidence,
            ai_prediction=ai_prediction,
            ai_agreement=ai_agreement,
            validation_method=validation_method,
            price_at_detection=candle.close,
            candles_involved=candles_involved,
            trend_context=trend_context,
            description=info.get("description", ""),
            trading_implication=info.get("implication", ""),
            pattern_candles=pattern_candles_dict,
        )

    def _get_strength(self, confidence: float) -> PatternStrength:
        """Convert confidence to pattern strength."""
        if confidence >= 0.75:
            return PatternStrength.STRONG
        elif confidence >= 0.5:
            return PatternStrength.MODERATE
        return PatternStrength.WEAK

    def detect_patterns(
        self,
        candles: list[CandleData],
        timeframe: Timeframe,
        min_confidence: float = 0.5,
        include_weak: bool = False
    ) -> list[DetectedPattern]:
        """
        Detect candlestick patterns in a list of candles.

        Args:
            candles: List of CandleData objects (oldest first)
            timeframe: Timeframe of the data
            min_confidence: Minimum confidence threshold
            include_weak: Include weak strength patterns

        Returns:
            List of detected patterns
        """
        if len(candles) < 3:
            return []

        patterns: list[DetectedPattern] = []
        avg_body = self._calculate_avg_body(candles)
        trend = self._determine_trend(candles)

        # Scan last N candles for patterns (focus on recent)
        scan_depth = min(10, len(candles) - 2)

        for i in range(len(candles) - scan_depth, len(candles)):
            if i < 2:
                continue

            current = candles[i]
            prev = candles[i - 1]
            prev2 = candles[i - 2] if i >= 2 else None

            detected = []

            # === Single Candle Patterns ===

            # Doji variants
            if self._is_dragonfly_doji(current, avg_body):
                confidence = 0.7 if trend == "downtrend" else 0.5
                detected.append((PatternType.DRAGONFLY_DOJI, confidence, 1))
            elif self._is_gravestone_doji(current, avg_body):
                confidence = 0.7 if trend == "uptrend" else 0.5
                detected.append((PatternType.GRAVESTONE_DOJI, confidence, 1))
            elif self._is_doji(current, avg_body):
                confidence = 0.6
                detected.append((PatternType.DOJI, confidence, 1))

            # Hammer / Inverted Hammer
            if self._is_hammer(current, avg_body):
                if trend == "downtrend":
                    confidence = 0.75
                    detected.append((PatternType.HAMMER, confidence, 1))
                elif trend == "uptrend" and self._is_hanging_man(current, prev, avg_body):
                    # Hanging Man requires clear uptrend, not just non-downtrend
                    confidence = 0.65
                    detected.append((PatternType.HANGING_MAN, confidence, 1))

            if self._is_inverted_hammer(current, avg_body):
                if trend == "downtrend":
                    confidence = 0.6
                    detected.append((PatternType.INVERTED_HAMMER, confidence, 1))
                elif trend == "uptrend" and self._is_shooting_star(current, prev, avg_body):
                    # Shooting Star requires clear uptrend, not just non-downtrend
                    confidence = 0.7
                    detected.append((PatternType.SHOOTING_STAR, confidence, 1))

            # Spinning Top
            if self._is_spinning_top(current, avg_body):
                detected.append((PatternType.SPINNING_TOP, 0.55, 1))

            # Belt Hold patterns
            if self._is_bullish_belt_hold(current, avg_body):
                confidence = 0.7 if trend == "downtrend" else 0.5
                detected.append((PatternType.BULLISH_BELT_HOLD, confidence, 1))
            elif self._is_bearish_belt_hold(current, avg_body):
                confidence = 0.7 if trend == "uptrend" else 0.5
                detected.append((PatternType.BEARISH_BELT_HOLD, confidence, 1))

            # === Two Candle Patterns ===

            # Engulfing - pass avg_body for better validation
            if self._is_bullish_engulfing(current, prev, avg_body):
                confidence = 0.8 if trend == "downtrend" else 0.6
                detected.append((PatternType.BULLISH_ENGULFING, confidence, 2))
            elif self._is_bearish_engulfing(current, prev, avg_body):
                confidence = 0.8 if trend == "uptrend" else 0.6
                detected.append((PatternType.BEARISH_ENGULFING, confidence, 2))

            # Harami
            if self._is_harami_cross(current, prev, avg_body):
                confidence = 0.65
                detected.append((PatternType.HARAMI_CROSS, confidence, 2))
            elif self._is_bullish_harami(current, prev):
                confidence = 0.55 if trend == "downtrend" else 0.45
                detected.append((PatternType.BULLISH_HARAMI, confidence, 2))
            elif self._is_bearish_harami(current, prev):
                confidence = 0.55 if trend == "uptrend" else 0.45
                detected.append((PatternType.BEARISH_HARAMI, confidence, 2))

            # Piercing Line / Dark Cloud Cover
            if self._is_piercing_line(current, prev):
                confidence = 0.65 if trend == "downtrend" else 0.5
                detected.append((PatternType.PIERCING_LINE, confidence, 2))
            elif self._is_dark_cloud_cover(current, prev):
                confidence = 0.65 if trend == "uptrend" else 0.5
                detected.append((PatternType.DARK_CLOUD_COVER, confidence, 2))

            # Counterattack patterns
            if self._is_bullish_counterattack(current, prev):
                confidence = 0.65 if trend == "downtrend" else 0.5
                detected.append((PatternType.BULLISH_COUNTERATTACK, confidence, 2))
            elif self._is_bearish_counterattack(current, prev):
                confidence = 0.65 if trend == "uptrend" else 0.5
                detected.append((PatternType.BEARISH_COUNTERATTACK, confidence, 2))

            # === Three Candle Patterns ===
            if prev2 is not None:
                # Morning/Evening Star
                if self._is_morning_star(prev2, prev, current, avg_body):
                    confidence = 0.85 if trend == "downtrend" else 0.65
                    detected.append((PatternType.MORNING_STAR, confidence, 3))
                elif self._is_evening_star(prev2, prev, current, avg_body):
                    confidence = 0.85 if trend == "uptrend" else 0.65
                    detected.append((PatternType.EVENING_STAR, confidence, 3))

                # Three White Soldiers / Three Black Crows
                if self._is_three_white_soldiers(prev2, prev, current, avg_body):
                    confidence = 0.8
                    detected.append((PatternType.THREE_WHITE_SOLDIERS, confidence, 3))
                elif self._is_three_black_crows(prev2, prev, current, avg_body):
                    confidence = 0.8
                    detected.append((PatternType.THREE_BLACK_CROWS, confidence, 3))

                # Three Inside Up/Down
                if self._is_three_inside_up(prev2, prev, current, avg_body):
                    confidence = 0.75 if trend == "downtrend" else 0.6
                    detected.append((PatternType.THREE_INSIDE_UP, confidence, 3))
                elif self._is_three_inside_down(prev2, prev, current, avg_body):
                    confidence = 0.75 if trend == "uptrend" else 0.6
                    detected.append((PatternType.THREE_INSIDE_DOWN, confidence, 3))

                # Abandoned Baby (rare but strong patterns)
                if self._is_bullish_abandoned_baby(prev2, prev, current, avg_body):
                    confidence = 0.9 if trend == "downtrend" else 0.75
                    detected.append((PatternType.BULLISH_ABANDONED_BABY, confidence, 3))
                elif self._is_bearish_abandoned_baby(prev2, prev, current, avg_body):
                    confidence = 0.9 if trend == "uptrend" else 0.75
                    detected.append((PatternType.BEARISH_ABANDONED_BABY, confidence, 3))

                # Advance Block (bearish reversal warning)
                if self._is_advance_block(prev2, prev, current, avg_body):
                    confidence = 0.7 if trend == "uptrend" else 0.55
                    detected.append((PatternType.ADVANCE_BLOCK, confidence, 3))

            # === Island Patterns (multi-candle with gaps) ===
            # Check for Island patterns with varying window sizes (3-6 candles)
            for window_size in range(3, min(7, i + 1)):
                start_idx = i - window_size + 1
                if start_idx >= 0:
                    window_candles = candles[start_idx:i + 1]
                    if self._is_bearish_island(window_candles, avg_body):
                        confidence = 0.85 if trend == "uptrend" else 0.65
                        detected.append((PatternType.BEARISH_ISLAND, confidence, window_size))
                        break  # Found an Island, don't check larger windows
                    elif self._is_bullish_island(window_candles, avg_body):
                        confidence = 0.85 if trend == "downtrend" else 0.65
                        detected.append((PatternType.BULLISH_ISLAND, confidence, window_size))
                        break

            # === Tower Patterns (multi-candle) ===
            # Check for Tower patterns with varying window sizes (4-12 candles)
            for window_size in range(4, min(13, i + 1)):
                start_idx = i - window_size + 1
                if start_idx >= 0:
                    window_candles = candles[start_idx:i + 1]
                    if self._is_tower_bottom(window_candles, avg_body):
                        confidence = 0.7 if trend == "downtrend" else 0.55
                        detected.append((PatternType.TOWER_BOTTOM, confidence, window_size))
                        break  # Found a Tower, don't check larger windows
                    elif self._is_tower_top(window_candles, avg_body):
                        confidence = 0.7 if trend == "uptrend" else 0.55
                        detected.append((PatternType.TOWER_TOP, confidence, window_size))
                        break

            # Create pattern objects for detected patterns
            for pattern_type, confidence, candles_count in detected:
                if confidence >= min_confidence:
                    strength = self._get_strength(confidence)
                    if include_weak or strength != PatternStrength.WEAK:
                        # Extract the pattern candles (oldest to newest)
                        pattern_start_idx = max(0, i - candles_count + 1)
                        pattern_candles_list = candles[pattern_start_idx:i + 1]

                        patterns.append(self._create_pattern(
                            pattern_type=pattern_type,
                            candle=current,
                            timeframe=timeframe,
                            confidence=confidence,
                            candles_involved=candles_count,
                            trend_context=trend,
                            pattern_candles=pattern_candles_list,
                        ))

        return patterns

    def detect_patterns_with_ai(
        self,
        candles: list[CandleData],
        timeframe: Timeframe,
        min_confidence: float = 0.5,
        include_weak: bool = False
    ) -> list[DetectedPattern]:
        """
        Detect candlestick patterns with AI validation.

        Two-stage process:
        1. Rule-based detection - identify pattern candidates
        2. AI validation - validate and adjust confidence scores

        Args:
            candles: List of CandleData objects (oldest first)
            timeframe: Timeframe of the data
            min_confidence: Minimum confidence threshold (applied to final confidence)
            include_weak: Include weak strength patterns

        Returns:
            List of detected patterns with AI-validated confidence scores
        """
        if len(candles) < 3:
            return []

        # Import AI validator here to avoid circular imports
        from .ai_validator_service import ai_validator_service

        patterns: list[DetectedPattern] = []
        avg_body = self._calculate_avg_body(candles)
        trend = self._determine_trend(candles)

        # Prepare OHLCV data for AI validator
        ohlcv_data = [
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume or 0
            }
            for c in candles
        ]

        # Scan last N candles for patterns (focus on recent)
        scan_depth = min(10, len(candles) - 2)

        for i in range(len(candles) - scan_depth, len(candles)):
            if i < 2:
                continue

            current = candles[i]
            prev = candles[i - 1]
            prev2 = candles[i - 2] if i >= 2 else None

            detected = []

            # === Single Candle Patterns ===
            if self._is_dragonfly_doji(current, avg_body):
                confidence = 0.7 if trend == "downtrend" else 0.5
                detected.append((PatternType.DRAGONFLY_DOJI, confidence, 1))
            elif self._is_gravestone_doji(current, avg_body):
                confidence = 0.7 if trend == "uptrend" else 0.5
                detected.append((PatternType.GRAVESTONE_DOJI, confidence, 1))
            elif self._is_doji(current, avg_body):
                detected.append((PatternType.DOJI, 0.6, 1))

            if self._is_hammer(current, avg_body):
                if trend == "downtrend":
                    detected.append((PatternType.HAMMER, 0.75, 1))
                elif trend == "uptrend" and self._is_hanging_man(current, prev, avg_body):
                    # Hanging Man requires clear uptrend, not just non-downtrend
                    detected.append((PatternType.HANGING_MAN, 0.65, 1))

            if self._is_inverted_hammer(current, avg_body):
                if trend == "downtrend":
                    detected.append((PatternType.INVERTED_HAMMER, 0.6, 1))
                elif trend == "uptrend" and self._is_shooting_star(current, prev, avg_body):
                    # Shooting Star requires clear uptrend, not just non-downtrend
                    detected.append((PatternType.SHOOTING_STAR, 0.7, 1))

            if self._is_spinning_top(current, avg_body):
                detected.append((PatternType.SPINNING_TOP, 0.55, 1))

            # Belt Hold patterns
            if self._is_bullish_belt_hold(current, avg_body):
                confidence = 0.7 if trend == "downtrend" else 0.5
                detected.append((PatternType.BULLISH_BELT_HOLD, confidence, 1))
            elif self._is_bearish_belt_hold(current, avg_body):
                confidence = 0.7 if trend == "uptrend" else 0.5
                detected.append((PatternType.BEARISH_BELT_HOLD, confidence, 1))

            # === Two Candle Patterns ===
            if self._is_bullish_engulfing(current, prev, avg_body):
                confidence = 0.8 if trend == "downtrend" else 0.6
                detected.append((PatternType.BULLISH_ENGULFING, confidence, 2))
            elif self._is_bearish_engulfing(current, prev, avg_body):
                confidence = 0.8 if trend == "uptrend" else 0.6
                detected.append((PatternType.BEARISH_ENGULFING, confidence, 2))

            if self._is_harami_cross(current, prev, avg_body):
                detected.append((PatternType.HARAMI_CROSS, 0.65, 2))
            elif self._is_bullish_harami(current, prev):
                confidence = 0.55 if trend == "downtrend" else 0.45
                detected.append((PatternType.BULLISH_HARAMI, confidence, 2))
            elif self._is_bearish_harami(current, prev):
                confidence = 0.55 if trend == "uptrend" else 0.45
                detected.append((PatternType.BEARISH_HARAMI, confidence, 2))

            if self._is_piercing_line(current, prev):
                confidence = 0.65 if trend == "downtrend" else 0.5
                detected.append((PatternType.PIERCING_LINE, confidence, 2))
            elif self._is_dark_cloud_cover(current, prev):
                confidence = 0.65 if trend == "uptrend" else 0.5
                detected.append((PatternType.DARK_CLOUD_COVER, confidence, 2))

            # Counterattack patterns
            if self._is_bullish_counterattack(current, prev):
                confidence = 0.65 if trend == "downtrend" else 0.5
                detected.append((PatternType.BULLISH_COUNTERATTACK, confidence, 2))
            elif self._is_bearish_counterattack(current, prev):
                confidence = 0.65 if trend == "uptrend" else 0.5
                detected.append((PatternType.BEARISH_COUNTERATTACK, confidence, 2))

            # === Three Candle Patterns ===
            if prev2 is not None:
                if self._is_morning_star(prev2, prev, current, avg_body):
                    confidence = 0.85 if trend == "downtrend" else 0.65
                    detected.append((PatternType.MORNING_STAR, confidence, 3))
                elif self._is_evening_star(prev2, prev, current, avg_body):
                    confidence = 0.85 if trend == "uptrend" else 0.65
                    detected.append((PatternType.EVENING_STAR, confidence, 3))

                if self._is_three_white_soldiers(prev2, prev, current, avg_body):
                    detected.append((PatternType.THREE_WHITE_SOLDIERS, 0.8, 3))
                elif self._is_three_black_crows(prev2, prev, current, avg_body):
                    detected.append((PatternType.THREE_BLACK_CROWS, 0.8, 3))

                # Three Inside Up/Down
                if self._is_three_inside_up(prev2, prev, current, avg_body):
                    confidence = 0.75 if trend == "downtrend" else 0.6
                    detected.append((PatternType.THREE_INSIDE_UP, confidence, 3))
                elif self._is_three_inside_down(prev2, prev, current, avg_body):
                    confidence = 0.75 if trend == "uptrend" else 0.6
                    detected.append((PatternType.THREE_INSIDE_DOWN, confidence, 3))

                # Abandoned Baby (rare but strong patterns)
                if self._is_bullish_abandoned_baby(prev2, prev, current, avg_body):
                    confidence = 0.9 if trend == "downtrend" else 0.75
                    detected.append((PatternType.BULLISH_ABANDONED_BABY, confidence, 3))
                elif self._is_bearish_abandoned_baby(prev2, prev, current, avg_body):
                    confidence = 0.9 if trend == "uptrend" else 0.75
                    detected.append((PatternType.BEARISH_ABANDONED_BABY, confidence, 3))

            # === Tower Patterns (multi-candle) ===
            for window_size in range(4, min(13, i + 1)):
                start_idx = i - window_size + 1
                if start_idx >= 0:
                    window_candles = candles[start_idx:i + 1]
                    if self._is_tower_bottom(window_candles, avg_body):
                        confidence = 0.7 if trend == "downtrend" else 0.55
                        detected.append((PatternType.TOWER_BOTTOM, confidence, window_size))
                        break
                    elif self._is_tower_top(window_candles, avg_body):
                        confidence = 0.7 if trend == "uptrend" else 0.55
                        detected.append((PatternType.TOWER_TOP, confidence, window_size))
                        break

            # Validate with AI and create pattern objects
            for pattern_type, rule_confidence, candles_count in detected:
                # Get OHLCV context for this specific pattern
                pattern_ohlcv = ohlcv_data[:i + 1]  # Up to current candle

                # AI validation
                validation = ai_validator_service.validate_pattern(
                    pattern_type=pattern_type.value,
                    rule_confidence=rule_confidence,
                    ohlcv_data=pattern_ohlcv,
                    lookback=20
                )

                final_confidence = validation.final_confidence

                # Apply confidence threshold to final (AI-adjusted) confidence
                if final_confidence >= min_confidence:
                    strength = self._get_strength(final_confidence)
                    if include_weak or strength != PatternStrength.WEAK:
                        # Extract the pattern candles (oldest to newest)
                        # i is the index of the last candle, candles_count is how many candles form the pattern
                        pattern_start_idx = max(0, i - candles_count + 1)
                        pattern_candles_list = candles[pattern_start_idx:i + 1]

                        patterns.append(self._create_pattern(
                            pattern_type=pattern_type,
                            candle=current,
                            timeframe=timeframe,
                            confidence=final_confidence,
                            candles_involved=candles_count,
                            trend_context=trend,
                            rule_confidence=validation.rule_confidence,
                            ai_confidence=validation.ai_confidence,
                            ai_prediction=validation.ai_prediction,
                            ai_agreement=validation.ai_agreement,
                            validation_method=validation.validation_method,
                            pattern_candles=pattern_candles_list,
                        ))

        return patterns

    # ==================== Data Conversion ====================

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Get the number of minutes for a timeframe."""
        tf_map = {
            "M5": 5,
            "M15": 15,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
        }
        return tf_map.get(timeframe.upper(), 60)

    def _get_candle_boundary(self, ts: datetime, timeframe: str) -> datetime:
        """
        Get the candle boundary (start time) for a timestamp.
        For H1: rounds down to the start of the hour.
        For D1: rounds down to the start of the day.
        """
        if timeframe.upper() == "D1":
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe.upper() == "H4":
            hour = (ts.hour // 4) * 4
            return ts.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe.upper() == "H1":
            return ts.replace(minute=0, second=0, microsecond=0)
        elif timeframe.upper() == "M15":
            minute = (ts.minute // 15) * 15
            return ts.replace(minute=minute, second=0, microsecond=0)
        elif timeframe.upper() == "M5":
            minute = (ts.minute // 5) * 5
            return ts.replace(minute=minute, second=0, microsecond=0)
        else:
            return ts.replace(minute=0, second=0, microsecond=0)

    def _aggregate_to_candles(
        self,
        data: list[dict],
        timeframe: str
    ) -> list[dict]:
        """
        Aggregate minute-level data to proper timeframe candles.

        EasyInsight provides minute-by-minute snapshots with rolling OHLC values.
        This function groups by timeframe boundaries and takes the final OHLC
        values for each completed candle.
        """
        if not data:
            return []

        tf_lower = timeframe.lower()
        aggregated: dict[str, dict] = {}

        for row in data:
            try:
                ts_str = row.get("snapshot_time") or row.get("timestamp") or row.get("datetime")
                if not ts_str:
                    continue

                # Parse timestamp
                if isinstance(ts_str, str):
                    try:
                        from dateutil import parser as dateutil_parser
                        ts = dateutil_parser.isoparse(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except (ImportError, ValueError):
                        continue
                else:
                    ts = ts_str

                # Get candle boundary
                boundary = self._get_candle_boundary(ts, timeframe)
                boundary_key = boundary.isoformat()

                # Get OHLC values for the requested timeframe
                open_keys = [f"{tf_lower}_open", "h1_open", "open"]
                high_keys = [f"{tf_lower}_high", "h1_high", "high"]
                low_keys = [f"{tf_lower}_low", "h1_low", "low"]
                close_keys = [f"{tf_lower}_close", "h1_close", "close"]

                open_val = high_val = low_val = close_val = None
                for key in open_keys:
                    if key in row and row[key] is not None:
                        open_val = float(row[key])
                        break
                for key in high_keys:
                    if key in row and row[key] is not None:
                        high_val = float(row[key])
                        break
                for key in low_keys:
                    if key in row and row[key] is not None:
                        low_val = float(row[key])
                        break
                for key in close_keys:
                    if key in row and row[key] is not None:
                        close_val = float(row[key])
                        break

                if not all(v is not None for v in [open_val, high_val, low_val, close_val]):
                    continue

                # Store/update the candle for this boundary
                # We keep the latest snapshot for each candle boundary (most complete OHLC)
                if boundary_key not in aggregated or ts > aggregated[boundary_key]["_last_ts"]:
                    aggregated[boundary_key] = {
                        "snapshot_time": boundary.isoformat(),
                        f"{tf_lower}_open": open_val,
                        f"{tf_lower}_high": high_val,
                        f"{tf_lower}_low": low_val,
                        f"{tf_lower}_close": close_val,
                        "_last_ts": ts,
                    }

            except Exception as e:
                logger.warning(f"Failed to aggregate row: {e}")
                continue

        # Convert to list and sort by time
        result = list(aggregated.values())
        for r in result:
            del r["_last_ts"]  # Remove internal field
        result.sort(key=lambda x: x["snapshot_time"])

        logger.debug(f"Aggregated {len(data)} rows to {len(result)} {timeframe} candles")
        return result

    def _convert_ohlc_to_candles(
        self,
        data: list[dict],
        timeframe: str
    ) -> list[CandleData]:
        """Convert raw OHLC data to CandleData objects."""
        candles = []
        tf_lower = timeframe.lower()

        # Possible field naming patterns
        open_keys = [f"{tf_lower}_open", "h1_open", "open"]
        high_keys = [f"{tf_lower}_high", "h1_high", "high"]
        low_keys = [f"{tf_lower}_low", "h1_low", "low"]
        close_keys = [f"{tf_lower}_close", "h1_close", "close"]

        for row in data:
            try:
                # Get timestamp
                ts = row.get("snapshot_time") or row.get("timestamp") or row.get("datetime")
                if isinstance(ts, str):
                    # Try ISO 8601 format first (with timezone)
                    try:
                        from dateutil import parser as dateutil_parser
                        ts = dateutil_parser.isoparse(ts)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except (ImportError, ValueError):
                        # Fallback: Parse various datetime formats
                        parsed = False
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                            try:
                                ts = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
                                parsed = True
                                break
                            except ValueError:
                                continue
                        if not parsed:
                            continue

                # Get OHLC values
                open_val = None
                for key in open_keys:
                    if key in row and row[key] is not None:
                        open_val = float(row[key])
                        break

                high_val = None
                for key in high_keys:
                    if key in row and row[key] is not None:
                        high_val = float(row[key])
                        break

                low_val = None
                for key in low_keys:
                    if key in row and row[key] is not None:
                        low_val = float(row[key])
                        break

                close_val = None
                for key in close_keys:
                    if key in row and row[key] is not None:
                        close_val = float(row[key])
                        break

                if all(v is not None for v in [open_val, high_val, low_val, close_val]):
                    candles.append(CandleData(
                        timestamp=ts,
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=row.get("volume"),
                    ))

            except Exception as e:
                logger.warning(f"Failed to convert OHLC row: {e}")
                continue

        # Sort by timestamp (oldest first)
        candles.sort(key=lambda c: c.timestamp)
        return candles

    # ==================== Data Gateway Methods ====================

    async def _fetch_historical_data(
        self,
        symbol: str,
        limit: int,
        timeframe: str
    ) -> tuple[list[dict], str]:
        """
        Fetch historical data from Data Service via TwelveData endpoint.

        Returns:
            Tuple of (data, source)
        """
        client = await self._get_client()

        try:
            # Konvertiere Timeframe zu TwelveData-Format (z.B. H1 -> 1h)
            td_interval = to_twelvedata(timeframe)

            url = f"{DATA_SERVICE_URL}/api/v1/twelvedata/time_series/{symbol}"
            params = {
                "interval": td_interval,
                "outputsize": min(limit, 5000),  # TwelveData max limit
            }

            response = await client.get(url, params=params)
            response.raise_for_status()

            result = response.json()

            # TwelveData Response Format:
            # {"meta": {...}, "values": [{"datetime": "...", "open": "...", ...}]}
            values = result.get("values", [])

            # Konvertiere zu Standard-Format für Pattern Detection
            data = []
            for v in values:
                try:
                    data.append({
                        "timestamp": v.get("datetime"),
                        "open": float(v.get("open", 0)),
                        "high": float(v.get("high", 0)),
                        "low": float(v.get("low", 0)),
                        "close": float(v.get("close", 0)),
                        "volume": float(v.get("volume", 0)) if v.get("volume") else 0.0,
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse candle data: {e}")
                    continue

            source = "twelvedata"
            if result.get("meta", {}).get("from_cache"):
                source = "twelvedata_cached"

            return data, source

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return [], "error"
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data: {e}")
            return [], "error"

    async def _get_symbol_names(self) -> list[str]:
        """Get list of available symbols from Data Service."""
        client = await self._get_client()

        try:
            url = f"{DATA_SERVICE_URL}/api/v1/symbols"
            response = await client.get(url)
            response.raise_for_status()

            result = response.json()

            # Handle different response formats
            if isinstance(result, list):
                return [s.get("symbol", s) if isinstance(s, dict) else s for s in result]
            elif isinstance(result, dict) and "symbols" in result:
                symbols = result["symbols"]
                return [s.get("symbol", s) if isinstance(s, dict) else s for s in symbols]

            return []

        except Exception as e:
            logger.error(f"Failed to get symbol names: {e}")
            return []

    # ==================== Public API ====================

    async def scan_patterns(
        self,
        request: PatternScanRequest
    ) -> PatternScanResponse:
        """
        Scan for candlestick patterns across multiple timeframes.

        Args:
            request: PatternScanRequest with symbol and parameters

        Returns:
            PatternScanResponse with detected patterns
        """
        start_time = perf_counter()
        request_id = str(uuid.uuid4())[:8]

        logger.info(f"[{request_id}] Starting pattern scan for {request.symbol}")

        result = MultiTimeframePatternResult(
            symbol=request.symbol,
            scan_timestamp=datetime.now(timezone.utc),
        )

        data_source = "easyinsight"

        # Scan each requested timeframe
        for tf in request.timeframes:
            try:
                # For M5 and H4: TwelveData provides ready candles, no aggregation needed
                # For other timeframes: EasyInsight provides minute snapshots that need aggregation
                use_twelvedata_directly = tf.value.upper() in ("M5", "H4")

                if use_twelvedata_directly:
                    # TwelveData returns actual candles, request exact count needed
                    raw_limit = request.lookback_candles
                else:
                    # EasyInsight provides minute-level snapshots, so for H1 we need ~60x more data
                    tf_minutes = self._get_timeframe_minutes(tf.value)
                    raw_limit = request.lookback_candles * tf_minutes

                # Fetch data via Data Service
                data, source = await self._fetch_historical_data(
                    symbol=request.symbol,
                    limit=raw_limit,
                    timeframe=tf.value,
                )

                if source != "easyinsight":
                    data_source = source

                if not data:
                    logger.warning(f"[{request_id}] No data for {request.symbol} {tf.value}")
                    continue

                # Aggregate minute-level data to proper timeframe candles
                # Skip aggregation for TwelveData (already provides proper candles)
                if source == "twelvedata":
                    aggregated_data = data  # Already proper candles
                else:
                    # EasyInsight provides minute snapshots with rolling OHLC
                    aggregated_data = self._aggregate_to_candles(data, tf.value)

                # Convert to CandleData
                candles = self._convert_ohlc_to_candles(aggregated_data, tf.value)

                # IMPORTANT: Skip the last (most recent) candle as it may still be open/forming
                # TwelveData returns the current candle which changes until it closes
                # Pattern detection on incomplete candles leads to false positives
                if len(candles) > 1:
                    candles = candles[:-1]  # Remove the last (potentially open) candle

                if len(candles) < 3:
                    logger.warning(f"[{request_id}] Insufficient candles for {request.symbol} {tf.value}")
                    continue

                # Detect patterns (with AI validation if model is available)
                patterns = self.detect_patterns_with_ai(
                    candles=candles,
                    timeframe=tf,
                    min_confidence=request.min_confidence,
                    include_weak=request.include_weak_patterns,
                )

                # Store results
                tf_result = TimeframePatterns(
                    timeframe=tf,
                    patterns=patterns,
                    candles_analyzed=len(candles),
                    analysis_timestamp=datetime.now(timezone.utc),
                )

                if tf == Timeframe.M5:
                    result.m5 = tf_result
                elif tf == Timeframe.M15:
                    result.m15 = tf_result
                elif tf == Timeframe.H1:
                    result.h1 = tf_result
                elif tf == Timeframe.H4:
                    result.h4 = tf_result
                elif tf == Timeframe.D1:
                    result.d1 = tf_result

                logger.info(
                    f"[{request_id}] {tf.value}: Found {len(patterns)} patterns in {len(candles)} candles"
                )

            except Exception as e:
                logger.error(f"[{request_id}] Error scanning {tf.value}: {e}")
                continue

        # Calculate summary
        result.calculate_summary()

        processing_time = (perf_counter() - start_time) * 1000

        logger.info(
            f"[{request_id}] Scan complete: {result.total_patterns_found} patterns found "
            f"in {processing_time:.2f}ms"
        )

        return PatternScanResponse(
            request_id=request_id,
            symbol=request.symbol,
            result=result,
            processing_time_ms=processing_time,
            data_source=data_source,
        )

    async def get_pattern_chart_data(
        self,
        symbol: str,
        timeframe: str,
        pattern_timestamp: datetime,
        candles_before: int = 15,
        candles_after: int = 5,
    ) -> dict:
        """
        Get OHLCV data for visualizing a pattern in a chart.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe of the pattern
            pattern_timestamp: Timestamp when the pattern was detected
            candles_before: Number of candles to show before the pattern
            candles_after: Number of candles to show after the pattern

        Returns:
            Dictionary with candle data and pattern highlight info
        """
        # Ensure pattern_timestamp is timezone-aware
        if pattern_timestamp.tzinfo is None:
            pattern_timestamp = pattern_timestamp.replace(tzinfo=timezone.utc)

        # Calculate how many candles we need based on how old the pattern is
        tf_minutes = self._get_timeframe_minutes(timeframe)
        now = datetime.now(timezone.utc)
        time_diff = now - pattern_timestamp
        candles_since_pattern = int(time_diff.total_seconds() / 60 / tf_minutes) + 1

        # Handle case where pattern timestamp is in the future (timezone issues)
        # or pattern is very recent - ensure minimum request
        candles_since_pattern = max(1, candles_since_pattern)

        # We need enough data to cover: pattern + candles_before + some buffer
        total_candles_needed = candles_since_pattern + candles_before + 10  # Extra buffer

        # Limit to reasonable range (minimum 50, maximum 500)
        total_candles_needed = max(50, min(total_candles_needed, 500))

        # Calculate raw data requirement based on timeframe
        use_twelvedata_directly = timeframe.upper() in ("M5", "H4")

        if use_twelvedata_directly:
            raw_limit = total_candles_needed
        else:
            raw_limit = total_candles_needed * tf_minutes

        logger.debug(
            f"Pattern chart: {symbol} {timeframe}, pattern age: {time_diff}, "
            f"candles_since: {candles_since_pattern}, requesting: {total_candles_needed} candles"
        )

        # Fetch data
        data, source = await self._fetch_historical_data(
            symbol=symbol,
            limit=raw_limit,
            timeframe=timeframe,
        )

        if not data:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return {
                "error": f"Keine Daten fuer {symbol} verfuegbar. Moeglicherweise ist das Pattern zu alt.",
                "candles": [],
            }

        # Aggregate if needed
        if source == "twelvedata":
            aggregated_data = data
        else:
            aggregated_data = self._aggregate_to_candles(data, timeframe)

        # Convert to CandleData
        candles = self._convert_ohlc_to_candles(aggregated_data, timeframe)

        if not candles:
            logger.warning(f"No candles after conversion for {symbol} {timeframe}")
            return {
                "error": f"Keine Kerzen-Daten fuer {symbol} {timeframe} verfuegbar.",
                "candles": [],
            }

        logger.debug(f"Converted {len(candles)} candles for {symbol} {timeframe}")

        # Find the pattern candle index by matching timeframe boundary
        pattern_idx = None
        p_boundary = self._get_candle_boundary(pattern_timestamp, timeframe)

        for i, c in enumerate(candles):
            c_boundary = self._get_candle_boundary(c.timestamp, timeframe)
            if c_boundary == p_boundary:
                pattern_idx = i
                break

        # If not found by exact boundary, find closest candle
        if pattern_idx is None:
            min_diff = float('inf')
            for i, c in enumerate(candles):
                diff = abs((c.timestamp - pattern_timestamp).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    pattern_idx = i
            logger.debug(f"Pattern candle not found by boundary, using closest at index {pattern_idx}")

        if pattern_idx is None:
            # Fallback: use the last available candle
            pattern_idx = len(candles) - 1
            logger.warning(f"Could not find pattern candle, using last candle at index {pattern_idx}")

        # Select candles around the pattern
        start_idx = max(0, pattern_idx - candles_before)
        end_idx = min(len(candles), pattern_idx + candles_after + 1)

        selected_candles = candles[start_idx:end_idx]
        highlight_idx = pattern_idx - start_idx

        # Ensure highlight_idx is within bounds
        if highlight_idx < 0 or highlight_idx >= len(selected_candles):
            highlight_idx = len(selected_candles) - 1 if selected_candles else 0

        # Convert to serializable format
        candle_list = []
        for i, c in enumerate(selected_candles):
            candle_list.append({
                "timestamp": c.timestamp.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "is_pattern": i == highlight_idx,
            })

        logger.info(
            f"Pattern chart for {symbol} {timeframe}: "
            f"{len(candle_list)} candles, pattern at index {highlight_idx}"
        )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "pattern_timestamp": pattern_timestamp.isoformat(),
            "data_source": source,
            "candles": candle_list,
            "pattern_candle_index": highlight_idx,
            "total_candles": len(candle_list),
        }

    async def scan_all_symbols(
        self,
        timeframes: Optional[list[Timeframe]] = None,
        min_confidence: float = 0.6,
    ) -> dict[str, MultiTimeframePatternResult]:
        """
        Scan all available symbols for patterns.

        Args:
            timeframes: Timeframes to scan (default: all)
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary of symbol -> pattern results
        """
        if timeframes is None:
            timeframes = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]

        # Get available symbols
        symbols = await self._get_symbol_names()

        logger.info(f"Scanning {len(symbols)} symbols for patterns")

        results = {}

        # Scan symbols concurrently (with limit)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent scans

        async def scan_symbol(symbol: str):
            async with semaphore:
                try:
                    request = PatternScanRequest(
                        symbol=symbol,
                        timeframes=timeframes,
                        min_confidence=min_confidence,
                    )
                    response = await self.scan_patterns(request)
                    return symbol, response.result
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    return symbol, None

        tasks = [scan_symbol(s) for s in symbols]
        scan_results = await asyncio.gather(*tasks)

        for symbol, result in scan_results:
            if result is not None:
                results[symbol] = result

        logger.info(f"Completed scanning {len(results)} symbols")
        return results

    def get_supported_patterns(self) -> list[dict]:
        """Get list of supported patterns with descriptions."""
        patterns = []
        for pattern_type, info in self._pattern_descriptions.items():
            patterns.append({
                "type": pattern_type.value,
                "category": info["category"].value,
                "direction": info["direction"].value,
                "description": info["description"],
                "implication": info["implication"],
            })
        return patterns


# Global singleton instance
candlestick_pattern_service = CandlestickPatternService()
