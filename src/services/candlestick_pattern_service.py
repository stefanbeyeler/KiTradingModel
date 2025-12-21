"""Candlestick Pattern Detection Service.

Erkennt Candlestick-Muster über mehrere Timeframes:
- Reversal: Hammer, Shooting Star, Doji, Engulfing, Morning/Evening Star
- Continuation: Three White Soldiers, Three Black Crows
- Indecision: Spinning Top, Harami

Multi-Timeframe Scanning: M15, H1, H4, D1

WICHTIG: Alle Datenzugriffe erfolgen über den DataGatewayService.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

from loguru import logger

from ..models.candlestick_patterns import (
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
from .data_gateway_service import data_gateway


class CandlestickPatternService:
    """
    Service für die Erkennung von Candlestick-Mustern.

    Unterstützt:
    - Reversal Patterns (Umkehrmuster)
    - Continuation Patterns (Fortsetzungsmuster)
    - Indecision Patterns (Unentschlossenheitsmuster)
    - Multi-Timeframe Scanning
    """

    def __init__(self):
        self._pattern_descriptions = self._init_pattern_descriptions()

    def _init_pattern_descriptions(self) -> dict[PatternType, dict]:
        """Initialize pattern descriptions and trading implications."""
        return {
            # Reversal Patterns - Bullish
            PatternType.HAMMER: {
                "description": "Hammer pattern - kleine Kerze mit langem unteren Schatten",
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
        }

    # ==================== Pattern Detection Methods ====================

    def _is_doji(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check if candle is a Doji (very small body relative to range).

        A Doji is detected when:
        - Body is less than 10% of the candle's total range, OR
        - Body is less than 10% of the average body size

        This dual check ensures Dojis are detected in both volatile and calm markets.
        """
        body_to_range_ratio = candle.body_size / candle.total_range if candle.total_range > 0 else 0
        body_to_avg_ratio = candle.body_size / avg_body if avg_body > 0 else 0

        # A Doji if body is small relative to either the range or the average
        return body_to_range_ratio < 0.1 or (avg_body > 0 and body_to_avg_ratio < 0.1)

    def _is_dragonfly_doji(self, candle: CandleData, avg_body: float) -> bool:
        """Check for Dragonfly Doji (long lower shadow, no upper shadow)."""
        if not self._is_doji(candle, avg_body):
            return False
        return (
            candle.lower_shadow > candle.total_range * 0.6 and
            candle.upper_shadow < candle.total_range * 0.1
        )

    def _is_gravestone_doji(self, candle: CandleData, avg_body: float) -> bool:
        """Check for Gravestone Doji (long upper shadow, no lower shadow)."""
        if not self._is_doji(candle, avg_body):
            return False
        return (
            candle.upper_shadow > candle.total_range * 0.6 and
            candle.lower_shadow < candle.total_range * 0.1
        )

    def _is_hammer(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Hammer pattern.
        - Small body at the top
        - Long lower shadow (at least 2x body)
        - Little or no upper shadow
        """
        if candle.total_range == 0:
            return False

        body_ratio = candle.body_size / candle.total_range
        lower_shadow_ratio = candle.lower_shadow / candle.total_range
        upper_shadow_ratio = candle.upper_shadow / candle.total_range

        return (
            body_ratio < 0.35 and
            lower_shadow_ratio > 0.5 and
            upper_shadow_ratio < 0.15 and
            candle.lower_shadow >= candle.body_size * 2
        )

    def _is_inverted_hammer(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Inverted Hammer pattern.
        - Small body at the bottom
        - Long upper shadow (at least 2x body)
        - Little or no lower shadow
        """
        if candle.total_range == 0:
            return False

        body_ratio = candle.body_size / candle.total_range
        upper_shadow_ratio = candle.upper_shadow / candle.total_range
        lower_shadow_ratio = candle.lower_shadow / candle.total_range

        return (
            body_ratio < 0.35 and
            upper_shadow_ratio > 0.5 and
            lower_shadow_ratio < 0.15 and
            candle.upper_shadow >= candle.body_size * 2
        )

    def _is_shooting_star(self, candle: CandleData, prev_candle: CandleData, avg_body: float) -> bool:
        """
        Check for Shooting Star pattern (Inverted Hammer after uptrend).
        """
        # Must be in an uptrend (previous candle bullish)
        if not prev_candle.is_bullish:
            return False

        # Must gap up or open at/near previous high
        if candle.open < prev_candle.close * 0.998:
            return False

        return self._is_inverted_hammer(candle, avg_body)

    def _is_hanging_man(self, candle: CandleData, prev_candle: CandleData, avg_body: float) -> bool:
        """
        Check for Hanging Man pattern (Hammer after uptrend).
        """
        # Must be in an uptrend (previous candle bullish)
        if not prev_candle.is_bullish:
            return False

        # Must gap up or open at/near previous high
        if candle.open < prev_candle.close * 0.998:
            return False

        return self._is_hammer(candle, avg_body)

    def _is_spinning_top(self, candle: CandleData, avg_body: float) -> bool:
        """
        Check for Spinning Top pattern.
        - Small body
        - Long shadows on both sides (roughly equal)
        """
        if candle.total_range == 0:
            return False

        body_ratio = candle.body_size / candle.total_range
        upper_shadow_ratio = candle.upper_shadow / candle.total_range
        lower_shadow_ratio = candle.lower_shadow / candle.total_range

        # Small body, significant shadows on both sides
        return (
            body_ratio < 0.3 and
            upper_shadow_ratio > 0.25 and
            lower_shadow_ratio > 0.25 and
            abs(upper_shadow_ratio - lower_shadow_ratio) < 0.2  # Roughly equal shadows
        )

    def _is_bullish_engulfing(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Bullish Engulfing pattern.
        - Previous candle is bearish
        - Current candle is bullish
        - Current body completely engulfs previous body
        """
        return (
            prev.is_bearish and
            current.is_bullish and
            current.open < prev.close and
            current.close > prev.open and
            current.body_size > prev.body_size
        )

    def _is_bearish_engulfing(self, current: CandleData, prev: CandleData) -> bool:
        """
        Check for Bearish Engulfing pattern.
        - Previous candle is bullish
        - Current candle is bearish
        - Current body completely engulfs previous body
        """
        return (
            prev.is_bullish and
            current.is_bearish and
            current.open > prev.close and
            current.close < prev.open and
            current.body_size > prev.body_size
        )

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
        1. Large bearish candle
        2. Small body candle (gap down)
        3. Large bullish candle (closes above midpoint of first)
        """
        # First candle: large bearish
        if not first.is_bearish or first.body_size < avg_body * 0.8:
            return False

        # Second candle: small body, gaps down
        if second.body_size > avg_body * 0.5:
            return False
        if max(second.open, second.close) > first.close:
            return False

        # Third candle: large bullish, closes above midpoint of first
        if not third.is_bullish or third.body_size < avg_body * 0.6:
            return False
        first_midpoint = (first.open + first.close) / 2
        if third.close < first_midpoint:
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
        1. Large bullish candle
        2. Small body candle (gap up)
        3. Large bearish candle (closes below midpoint of first)
        """
        # First candle: large bullish
        if not first.is_bullish or first.body_size < avg_body * 0.8:
            return False

        # Second candle: small body, gaps up
        if second.body_size > avg_body * 0.5:
            return False
        if min(second.open, second.close) < first.close:
            return False

        # Third candle: large bearish, closes below midpoint of first
        if not third.is_bearish or third.body_size < avg_body * 0.6:
            return False
        first_midpoint = (first.open + first.close) / 2
        if third.close > first_midpoint:
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
        trend_context: Optional[str] = None
    ) -> DetectedPattern:
        """Create a DetectedPattern object."""
        info = self._pattern_descriptions.get(pattern_type, {})

        return DetectedPattern(
            pattern_type=pattern_type,
            category=info.get("category", PatternCategory.INDECISION),
            direction=info.get("direction", PatternDirection.NEUTRAL),
            strength=self._get_strength(confidence),
            timestamp=candle.timestamp,
            timeframe=timeframe,
            confidence=confidence,
            price_at_detection=candle.close,
            candles_involved=candles_involved,
            trend_context=trend_context,
            description=info.get("description", ""),
            trading_implication=info.get("implication", ""),
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
                elif self._is_hanging_man(current, prev, avg_body):
                    confidence = 0.65
                    detected.append((PatternType.HANGING_MAN, confidence, 1))

            if self._is_inverted_hammer(current, avg_body):
                if trend == "downtrend":
                    confidence = 0.6
                    detected.append((PatternType.INVERTED_HAMMER, confidence, 1))
                elif self._is_shooting_star(current, prev, avg_body):
                    confidence = 0.7
                    detected.append((PatternType.SHOOTING_STAR, confidence, 1))

            # Spinning Top
            if self._is_spinning_top(current, avg_body):
                detected.append((PatternType.SPINNING_TOP, 0.55, 1))

            # === Two Candle Patterns ===

            # Engulfing
            if self._is_bullish_engulfing(current, prev):
                confidence = 0.8 if trend == "downtrend" else 0.6
                detected.append((PatternType.BULLISH_ENGULFING, confidence, 2))
            elif self._is_bearish_engulfing(current, prev):
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

            # Create pattern objects for detected patterns
            for pattern_type, confidence, candles_count in detected:
                if confidence >= min_confidence:
                    strength = self._get_strength(confidence)
                    if include_weak or strength != PatternStrength.WEAK:
                        patterns.append(self._create_pattern(
                            pattern_type=pattern_type,
                            candle=current,
                            timeframe=timeframe,
                            confidence=confidence,
                            candles_involved=candles_count,
                            trend_context=trend,
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

                # Fetch data via DataGatewayService
                data, source = await data_gateway.get_historical_data_with_fallback(
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

                if len(candles) < 3:
                    logger.warning(f"[{request_id}] Insufficient candles for {request.symbol} {tf.value}")
                    continue

                # Detect patterns
                patterns = self.detect_patterns(
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
        data, source = await data_gateway.get_historical_data_with_fallback(
            symbol=symbol,
            limit=raw_limit,
            timeframe=timeframe,
        )

        if not data:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return {
                "error": f"Keine Daten für {symbol} verfügbar. Möglicherweise ist das Pattern zu alt.",
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
                "error": f"Keine Kerzen-Daten für {symbol} {timeframe} verfügbar.",
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
        symbols = await data_gateway.get_symbol_names()

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


# Global singleton instance
candlestick_pattern_service = CandlestickPatternService()
