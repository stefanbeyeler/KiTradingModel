"""Pattern classification utilities for label generation."""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from loguru import logger


class PatternType(str, Enum):
    """Chart pattern types."""
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


@dataclass
class PatternDetection:
    """Detected pattern with metadata."""
    pattern_type: PatternType
    confidence: float
    start_index: int
    end_index: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    price_target: Optional[float] = None
    invalidation_level: Optional[float] = None
    pattern_height: Optional[float] = None
    direction: Optional[str] = None

    def __post_init__(self):
        """Convert numpy types to Python native types."""
        if hasattr(self.confidence, 'item'):
            self.confidence = float(self.confidence)
        if self.price_target is not None and hasattr(self.price_target, 'item'):
            self.price_target = float(self.price_target)
        if self.invalidation_level is not None and hasattr(self.invalidation_level, 'item'):
            self.invalidation_level = float(self.invalidation_level)
        if self.pattern_height is not None and hasattr(self.pattern_height, 'item'):
            self.pattern_height = float(self.pattern_height)
        if hasattr(self.start_index, 'item'):
            self.start_index = int(self.start_index)
        if hasattr(self.end_index, 'item'):
            self.end_index = int(self.end_index)


class PatternClassifier:
    """Rule-based pattern classifier for training label generation."""

    PATTERN_INFO = {
        PatternType.HEAD_AND_SHOULDERS: {"direction": "bearish", "reversal": True},
        PatternType.INVERSE_HEAD_AND_SHOULDERS: {"direction": "bullish", "reversal": True},
        PatternType.DOUBLE_TOP: {"direction": "bearish", "reversal": True},
        PatternType.DOUBLE_BOTTOM: {"direction": "bullish", "reversal": True},
        PatternType.TRIPLE_TOP: {"direction": "bearish", "reversal": True},
        PatternType.TRIPLE_BOTTOM: {"direction": "bullish", "reversal": True},
        PatternType.ASCENDING_TRIANGLE: {"direction": "bullish", "continuation": True},
        PatternType.DESCENDING_TRIANGLE: {"direction": "bearish", "continuation": True},
        PatternType.SYMMETRICAL_TRIANGLE: {"direction": "neutral", "continuation": True},
        PatternType.BULL_FLAG: {"direction": "bullish", "continuation": True},
        PatternType.BEAR_FLAG: {"direction": "bearish", "continuation": True},
        PatternType.CUP_AND_HANDLE: {"direction": "bullish", "reversal": True},
        PatternType.RISING_WEDGE: {"direction": "bearish", "reversal": True},
        PatternType.FALLING_WEDGE: {"direction": "bullish", "reversal": True},
        PatternType.CHANNEL_UP: {"direction": "bullish", "trend": True},
        PatternType.CHANNEL_DOWN: {"direction": "bearish", "trend": True},
    }

    def __init__(self):
        pass

    def find_swing_points(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        window: int = 5
    ) -> tuple[List[int], List[int]]:
        """Find swing highs and lows."""
        swing_highs = []
        swing_lows = []

        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append(i)
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append(i)

        return swing_highs, swing_lows

    def detect_head_and_shoulders(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Optional[PatternDetection]:
        """Detect head and shoulders pattern."""
        swing_highs, swing_lows = self.find_swing_points(highs, lows)

        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None

        for i in range(len(swing_highs) - 2):
            left_shoulder_idx = swing_highs[i]
            head_idx = swing_highs[i + 1]
            right_shoulder_idx = swing_highs[i + 2]

            left_shoulder = highs[left_shoulder_idx]
            head = highs[head_idx]
            right_shoulder = highs[right_shoulder_idx]

            if head > left_shoulder and head > right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff < 0.05:
                    neck_lows = [
                        lows[j] for j in swing_lows
                        if left_shoulder_idx < j < right_shoulder_idx
                    ]

                    if neck_lows:
                        neckline = np.mean(neck_lows)
                        pattern_height = head - neckline
                        target = neckline - pattern_height

                        return PatternDetection(
                            pattern_type=PatternType.HEAD_AND_SHOULDERS,
                            confidence=0.7,
                            start_index=left_shoulder_idx,
                            end_index=right_shoulder_idx,
                            price_target=target,
                            invalidation_level=head,
                            pattern_height=pattern_height,
                            direction="bearish"
                        )

        return None

    def detect_double_top(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Optional[PatternDetection]:
        """Detect double top pattern."""
        swing_highs, swing_lows = self.find_swing_points(highs, lows)

        if len(swing_highs) < 2:
            return None

        for i in range(len(swing_highs) - 1):
            first_top_idx = swing_highs[i]
            second_top_idx = swing_highs[i + 1]

            first_top = highs[first_top_idx]
            second_top = highs[second_top_idx]

            top_diff = abs(first_top - second_top) / first_top
            if top_diff < 0.02:
                valley_lows = [
                    lows[j] for j in swing_lows
                    if first_top_idx < j < second_top_idx
                ]

                if valley_lows:
                    valley = min(valley_lows)
                    pattern_height = max(first_top, second_top) - valley
                    target = valley - pattern_height

                    return PatternDetection(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=0.65,
                        start_index=first_top_idx,
                        end_index=second_top_idx,
                        price_target=target,
                        invalidation_level=max(first_top, second_top),
                        pattern_height=pattern_height,
                        direction="bearish"
                    )

        return None

    def detect_double_bottom(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Optional[PatternDetection]:
        """Detect double bottom pattern."""
        swing_highs, swing_lows = self.find_swing_points(highs, lows)

        if len(swing_lows) < 2:
            return None

        for i in range(len(swing_lows) - 1):
            first_bottom_idx = swing_lows[i]
            second_bottom_idx = swing_lows[i + 1]

            first_bottom = lows[first_bottom_idx]
            second_bottom = lows[second_bottom_idx]

            bottom_diff = abs(first_bottom - second_bottom) / first_bottom
            if bottom_diff < 0.02:
                peak_highs = [
                    highs[j] for j in swing_highs
                    if first_bottom_idx < j < second_bottom_idx
                ]

                if peak_highs:
                    peak = max(peak_highs)
                    pattern_height = peak - min(first_bottom, second_bottom)
                    target = peak + pattern_height

                    return PatternDetection(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=0.65,
                        start_index=first_bottom_idx,
                        end_index=second_bottom_idx,
                        price_target=target,
                        invalidation_level=min(first_bottom, second_bottom),
                        pattern_height=pattern_height,
                        direction="bullish"
                    )

        return None

    def detect_triangle(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Optional[PatternDetection]:
        """Detect triangle patterns."""
        swing_highs, swing_lows = self.find_swing_points(highs, lows)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        recent_highs = swing_highs[-3:]
        recent_lows = swing_lows[-3:]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None

        high_values = [highs[i] for i in recent_highs]
        low_values = [lows[i] for i in recent_lows]

        high_slope = (high_values[-1] - high_values[0]) / len(high_values)
        low_slope = (low_values[-1] - low_values[0]) / len(low_values)

        price_range = max(high_values) - min(low_values)
        high_slope_norm = high_slope / price_range if price_range > 0 else 0
        low_slope_norm = low_slope / price_range if price_range > 0 else 0

        if high_slope_norm < -0.01 and abs(low_slope_norm) < 0.01:
            return PatternDetection(
                pattern_type=PatternType.DESCENDING_TRIANGLE,
                confidence=0.6,
                start_index=min(recent_highs[0], recent_lows[0]),
                end_index=max(recent_highs[-1], recent_lows[-1]),
                direction="bearish"
            )
        elif low_slope_norm > 0.01 and abs(high_slope_norm) < 0.01:
            return PatternDetection(
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                confidence=0.6,
                start_index=min(recent_highs[0], recent_lows[0]),
                end_index=max(recent_highs[-1], recent_lows[-1]),
                direction="bullish"
            )
        elif high_slope_norm < -0.01 and low_slope_norm > 0.01:
            return PatternDetection(
                pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                confidence=0.55,
                start_index=min(recent_highs[0], recent_lows[0]),
                end_index=max(recent_highs[-1], recent_lows[-1]),
                direction="neutral"
            )

        return None

    def detect_all_patterns(self, ohlcv: np.ndarray) -> List[PatternDetection]:
        """Run all pattern detection methods."""
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        closes = ohlcv[:, 3]

        patterns = []

        detectors = [
            self.detect_head_and_shoulders,
            self.detect_double_top,
            self.detect_double_bottom,
            self.detect_triangle,
        ]

        for detector in detectors:
            try:
                result = detector(highs, lows, closes)
                if result:
                    patterns.append(result)
            except Exception as e:
                logger.debug(f"Pattern detection error: {e}")

        return patterns

    def get_pattern_info(self, pattern_type: PatternType) -> Dict:
        """Get information about a pattern type."""
        return self.PATTERN_INFO.get(pattern_type, {})
