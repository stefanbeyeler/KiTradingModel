"""
Pattern Example Generator Service.

Generates ideal example charts for all candlestick pattern types.
These synthetic examples show the "textbook" form of each pattern
for educational and reference purposes.
"""

import io
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.patheffects as path_effects
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - pattern examples disabled")


@dataclass
class SyntheticCandle:
    """A synthetic candle for pattern examples."""
    open: float
    high: float
    low: float
    close: float

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        return self.high - self.low


class PatternExampleService:
    """
    Generates ideal example charts for candlestick patterns.

    Each pattern has a predefined ideal structure that demonstrates
    the textbook form of the pattern for educational purposes.
    """

    def __init__(self):
        self.figure_size = (10, 6)
        self.dpi = 120
        self.colors = {
            "bullish": "#26a69a",            # Teal green
            "bullish_body": "#26a69a",
            "bearish": "#ef5350",            # Red
            "bearish_body": "#ef5350",
            "background": "#131722",         # Dark trading view style
            "grid": "#363c4e",
            "text": "#d1d4dc",
            "highlight_bg": "#1e222d",
            "pattern_label": "#ffffff",
            "annotation": "#9598a1",
        }

        # Pattern definitions with synthetic candle data generators
        self._pattern_generators = {
            # Single candle patterns
            "hammer": self._gen_hammer,
            "inverted_hammer": self._gen_inverted_hammer,
            "shooting_star": self._gen_shooting_star,
            "hanging_man": self._gen_hanging_man,
            "doji": self._gen_doji,
            "dragonfly_doji": self._gen_dragonfly_doji,
            "gravestone_doji": self._gen_gravestone_doji,
            "spinning_top": self._gen_spinning_top,
            "bullish_belt_hold": self._gen_bullish_belt_hold,
            "bearish_belt_hold": self._gen_bearish_belt_hold,

            # Two candle patterns
            "bullish_engulfing": self._gen_bullish_engulfing,
            "bearish_engulfing": self._gen_bearish_engulfing,
            "bullish_harami": self._gen_bullish_harami,
            "bearish_harami": self._gen_bearish_harami,
            "harami_cross": self._gen_harami_cross,
            "piercing_line": self._gen_piercing_line,
            "dark_cloud_cover": self._gen_dark_cloud_cover,
            "bullish_counterattack": self._gen_bullish_counterattack,
            "bearish_counterattack": self._gen_bearish_counterattack,

            # Three candle patterns
            "morning_star": self._gen_morning_star,
            "evening_star": self._gen_evening_star,
            "three_white_soldiers": self._gen_three_white_soldiers,
            "three_black_crows": self._gen_three_black_crows,
            "three_inside_up": self._gen_three_inside_up,
            "three_inside_down": self._gen_three_inside_down,
            "bullish_abandoned_baby": self._gen_bullish_abandoned_baby,
            "bearish_abandoned_baby": self._gen_bearish_abandoned_baby,

            # Multi-candle patterns
            "rising_three_methods": self._gen_rising_three_methods,
            "falling_three_methods": self._gen_falling_three_methods,
            "tower_bottom": self._gen_tower_bottom,
            "tower_top": self._gen_tower_top,
        }

        # Pattern metadata
        self._pattern_info = {
            "hammer": {"direction": "bullish", "candles": 1, "context": "downtrend"},
            "inverted_hammer": {"direction": "bullish", "candles": 1, "context": "downtrend"},
            "shooting_star": {"direction": "bearish", "candles": 1, "context": "uptrend"},
            "hanging_man": {"direction": "bearish", "candles": 1, "context": "uptrend"},
            "doji": {"direction": "neutral", "candles": 1, "context": "any"},
            "dragonfly_doji": {"direction": "bullish", "candles": 1, "context": "downtrend"},
            "gravestone_doji": {"direction": "bearish", "candles": 1, "context": "uptrend"},
            "spinning_top": {"direction": "neutral", "candles": 1, "context": "any"},
            "bullish_belt_hold": {"direction": "bullish", "candles": 1, "context": "downtrend"},
            "bearish_belt_hold": {"direction": "bearish", "candles": 1, "context": "uptrend"},
            "bullish_engulfing": {"direction": "bullish", "candles": 2, "context": "downtrend"},
            "bearish_engulfing": {"direction": "bearish", "candles": 2, "context": "uptrend"},
            "bullish_harami": {"direction": "bullish", "candles": 2, "context": "downtrend"},
            "bearish_harami": {"direction": "bearish", "candles": 2, "context": "uptrend"},
            "harami_cross": {"direction": "neutral", "candles": 2, "context": "any"},
            "piercing_line": {"direction": "bullish", "candles": 2, "context": "downtrend"},
            "dark_cloud_cover": {"direction": "bearish", "candles": 2, "context": "uptrend"},
            "bullish_counterattack": {"direction": "bullish", "candles": 2, "context": "downtrend"},
            "bearish_counterattack": {"direction": "bearish", "candles": 2, "context": "uptrend"},
            "morning_star": {"direction": "bullish", "candles": 3, "context": "downtrend"},
            "evening_star": {"direction": "bearish", "candles": 3, "context": "uptrend"},
            "three_white_soldiers": {"direction": "bullish", "candles": 3, "context": "downtrend"},
            "three_black_crows": {"direction": "bearish", "candles": 3, "context": "uptrend"},
            "three_inside_up": {"direction": "bullish", "candles": 3, "context": "downtrend"},
            "three_inside_down": {"direction": "bearish", "candles": 3, "context": "uptrend"},
            "bullish_abandoned_baby": {"direction": "bullish", "candles": 3, "context": "downtrend"},
            "bearish_abandoned_baby": {"direction": "bearish", "candles": 3, "context": "uptrend"},
            "rising_three_methods": {"direction": "bullish", "candles": 5, "context": "uptrend"},
            "falling_three_methods": {"direction": "bearish", "candles": 5, "context": "downtrend"},
            "tower_bottom": {"direction": "bullish", "candles": 5, "context": "downtrend"},
            "tower_top": {"direction": "bearish", "candles": 5, "context": "uptrend"},
        }

    def get_available_patterns(self) -> List[str]:
        """Get list of all available pattern types."""
        return list(self._pattern_generators.keys())

    def get_pattern_info(self, pattern_type: str) -> Optional[Dict]:
        """Get metadata for a pattern type."""
        return self._pattern_info.get(pattern_type.lower())

    # ==================== Context Generators ====================

    def _gen_downtrend_context(self, start_price: float = 100, candles: int = 4) -> List[SyntheticCandle]:
        """Generate downtrend context candles before pattern."""
        result = []
        price = start_price
        for i in range(candles):
            drop = 2 + (i * 0.5)  # Increasing drops
            o = price
            c = price - drop
            h = o + 0.5
            l = c - 0.3
            result.append(SyntheticCandle(o, h, l, c))
            price = c
        return result

    def _gen_uptrend_context(self, start_price: float = 100, candles: int = 4) -> List[SyntheticCandle]:
        """Generate uptrend context candles before pattern."""
        result = []
        price = start_price
        for i in range(candles):
            rise = 2 + (i * 0.5)
            o = price
            c = price + rise
            h = c + 0.3
            l = o - 0.5
            result.append(SyntheticCandle(o, h, l, c))
            price = c
        return result

    def _gen_sideways_context(self, center_price: float = 100, candles: int = 4) -> List[SyntheticCandle]:
        """Generate sideways context candles."""
        result = []
        for i in range(candles):
            offset = 0.5 if i % 2 == 0 else -0.5
            o = center_price + offset
            c = center_price - offset
            h = max(o, c) + 0.8
            l = min(o, c) - 0.8
            result.append(SyntheticCandle(o, h, l, c))
        return result

    # ==================== Single Candle Patterns ====================

    def _gen_hammer(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Hammer: small body at top, long lower shadow."""
        context = self._gen_downtrend_context(100, 4)
        last_close = context[-1].close

        # Hammer candle: body at top, long lower shadow
        o = last_close - 0.5
        c = last_close  # Bullish close
        body_size = abs(c - o)
        lower_shadow = body_size * 2.5  # Lower shadow 2.5x body
        h = c + 0.2  # Tiny upper shadow
        l = o - lower_shadow

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_inverted_hammer(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Inverted Hammer: small body at bottom, long upper shadow."""
        context = self._gen_downtrend_context(100, 4)
        last_close = context[-1].close

        o = last_close
        c = last_close + 0.5  # Small bullish body
        body_size = abs(c - o)
        upper_shadow = body_size * 2.5
        h = c + upper_shadow
        l = o - 0.2  # Tiny lower shadow

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_shooting_star(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Shooting Star: small body at bottom, long upper shadow (after uptrend)."""
        context = self._gen_uptrend_context(100, 4)
        last_close = context[-1].close

        # Gap up opening
        o = last_close + 1
        c = o - 0.5  # Small bearish body
        body_size = abs(c - o)
        upper_shadow = body_size * 2.5
        h = o + upper_shadow
        l = c - 0.2

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_hanging_man(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Hanging Man: small body at top, long lower shadow (after uptrend)."""
        context = self._gen_uptrend_context(100, 4)
        last_close = context[-1].close

        o = last_close + 0.5
        c = last_close + 1  # Small body at top
        body_size = abs(c - o)
        lower_shadow = body_size * 2.5
        h = c + 0.2
        l = o - lower_shadow

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_doji(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Doji: open equals close, shadows on both sides."""
        context = self._gen_sideways_context(100, 4)

        o = 100
        c = 100.05  # Nearly equal
        h = 101.5
        l = 98.5

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_dragonfly_doji(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Dragonfly Doji: open/close at high, long lower shadow."""
        context = self._gen_downtrend_context(100, 4)
        last_close = context[-1].close

        o = last_close
        c = last_close + 0.05
        h = c + 0.1
        l = o - 4  # Long lower shadow

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_gravestone_doji(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Gravestone Doji: open/close at low, long upper shadow."""
        context = self._gen_uptrend_context(100, 4)
        last_close = context[-1].close

        o = last_close
        c = last_close + 0.05
        h = o + 4  # Long upper shadow
        l = o - 0.1

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_spinning_top(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Spinning Top: small body, long shadows on both sides."""
        context = self._gen_sideways_context(100, 4)

        o = 99.5
        c = 100.5  # Small body
        h = 103  # Long upper shadow
        l = 97   # Long lower shadow

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_bullish_belt_hold(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bullish Belt Hold: long bullish body, opens at low."""
        context = self._gen_downtrend_context(100, 4)
        last_close = context[-1].close

        o = last_close - 1  # Opens with gap down
        l = o  # Opens at low (no lower shadow)
        c = o + 5  # Long bullish body
        h = c + 0.3  # Small upper shadow

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    def _gen_bearish_belt_hold(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bearish Belt Hold: long bearish body, opens at high."""
        context = self._gen_uptrend_context(100, 4)
        last_close = context[-1].close

        o = last_close + 1  # Opens with gap up
        h = o  # Opens at high (no upper shadow)
        c = o - 5  # Long bearish body
        l = c - 0.3  # Small lower shadow

        pattern = [SyntheticCandle(o, h, l, c)]
        return context + pattern, [len(context)]

    # ==================== Two Candle Patterns ====================

    def _gen_bullish_engulfing(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bullish Engulfing: bearish candle followed by larger bullish candle."""
        context = self._gen_downtrend_context(100, 3)
        last_close = context[-1].close

        # First: small bearish
        c1_o = last_close
        c1_c = last_close - 2
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: large bullish that engulfs
        c2_o = c1_c - 0.5  # Opens below first close
        c2_c = c1_o + 1    # Closes above first open
        c2_h = c2_c + 0.3
        c2_l = c2_o - 0.3
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_bearish_engulfing(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bearish Engulfing: bullish candle followed by larger bearish candle."""
        context = self._gen_uptrend_context(100, 3)
        last_close = context[-1].close

        # First: small bullish
        c1_o = last_close
        c1_c = last_close + 2
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: large bearish that engulfs
        c2_o = c1_c + 0.5  # Opens above first close
        c2_c = c1_o - 1    # Closes below first open
        c2_h = c2_o + 0.3
        c2_l = c2_c - 0.3
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_bullish_harami(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bullish Harami: large bearish followed by small bullish inside."""
        context = self._gen_downtrend_context(100, 3)
        last_close = context[-1].close

        # First: large bearish
        c1_o = last_close
        c1_c = last_close - 4
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: small bullish inside first body
        c2_o = c1_c + 1
        c2_c = c1_o - 1.5
        c2_h = c2_c + 0.2
        c2_l = c2_o - 0.2
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_bearish_harami(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bearish Harami: large bullish followed by small bearish inside."""
        context = self._gen_uptrend_context(100, 3)
        last_close = context[-1].close

        # First: large bullish
        c1_o = last_close
        c1_c = last_close + 4
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: small bearish inside first body
        c2_o = c1_c - 1
        c2_c = c1_o + 1.5
        c2_h = c2_o + 0.2
        c2_l = c2_c - 0.2
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_harami_cross(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Harami Cross: large candle followed by doji inside."""
        context = self._gen_downtrend_context(100, 3)
        last_close = context[-1].close

        # First: large bearish
        c1_o = last_close
        c1_c = last_close - 4
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: doji inside first body
        mid = (c1_o + c1_c) / 2
        c2_o = mid
        c2_c = mid + 0.05
        c2_h = mid + 0.8
        c2_l = mid - 0.8
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_piercing_line(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Piercing Line: bearish, then bullish opening below but closing above midpoint."""
        context = self._gen_downtrend_context(100, 3)
        last_close = context[-1].close

        # First: bearish
        c1_o = last_close
        c1_c = last_close - 4
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.5
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: opens below first low, closes above midpoint
        c2_o = c1_l - 0.5
        midpoint = (c1_o + c1_c) / 2
        c2_c = midpoint + 1  # Above midpoint
        c2_h = c2_c + 0.3
        c2_l = c2_o - 0.2
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_dark_cloud_cover(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Dark Cloud Cover: bullish, then bearish opening above but closing below midpoint."""
        context = self._gen_uptrend_context(100, 3)
        last_close = context[-1].close

        # First: bullish
        c1_o = last_close
        c1_c = last_close + 4
        c1_h = c1_c + 0.5
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: opens above first high, closes below midpoint
        c2_o = c1_h + 0.5
        midpoint = (c1_o + c1_c) / 2
        c2_c = midpoint - 1  # Below midpoint
        c2_h = c2_o + 0.2
        c2_l = c2_c - 0.3
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_bullish_counterattack(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bullish Counterattack: bearish, gap down, bullish closing at same level."""
        context = self._gen_downtrend_context(100, 3)
        last_close = context[-1].close

        # First: bearish
        c1_o = last_close
        c1_c = last_close - 3
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: gap down open, closes at same level as first close
        c2_o = c1_c - 2  # Gap down
        c2_c = c1_c  # Same close level
        c2_h = c2_c + 0.3
        c2_l = c2_o - 0.3
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    def _gen_bearish_counterattack(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bearish Counterattack: bullish, gap up, bearish closing at same level."""
        context = self._gen_uptrend_context(100, 3)
        last_close = context[-1].close

        # First: bullish
        c1_o = last_close
        c1_c = last_close + 3
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: gap up open, closes at same level as first close
        c2_o = c1_c + 2  # Gap up
        c2_c = c1_c  # Same close level
        c2_h = c2_o + 0.3
        c2_l = c2_c - 0.3
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        pattern = [candle1, candle2]
        return context + pattern, [len(context), len(context) + 1]

    # ==================== Three Candle Patterns ====================

    def _gen_morning_star(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Morning Star: bearish, small/doji, bullish."""
        context = self._gen_downtrend_context(100, 3)
        last_close = context[-1].close

        # First: large bearish
        c1_o = last_close
        c1_c = last_close - 4
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: small body with gap down (star)
        c2_o = c1_c - 1
        c2_c = c2_o + 0.3
        c2_h = c2_c + 0.5
        c2_l = c2_o - 0.5
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        # Third: large bullish closing into first body
        c3_o = c2_c + 0.5
        c3_c = c1_o - 1  # Closes into first body
        c3_h = c3_c + 0.3
        c3_l = c3_o - 0.3
        candle3 = SyntheticCandle(c3_o, c3_h, c3_l, c3_c)

        pattern = [candle1, candle2, candle3]
        return context + pattern, [len(context), len(context) + 1, len(context) + 2]

    def _gen_evening_star(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Evening Star: bullish, small/doji, bearish."""
        context = self._gen_uptrend_context(100, 3)
        last_close = context[-1].close

        # First: large bullish
        c1_o = last_close
        c1_c = last_close + 4
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: small body with gap up (star)
        c2_o = c1_c + 1
        c2_c = c2_o - 0.3
        c2_h = c2_o + 0.5
        c2_l = c2_c - 0.5
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        # Third: large bearish closing into first body
        c3_o = c2_c - 0.5
        c3_c = c1_o + 1  # Closes into first body
        c3_h = c3_o + 0.3
        c3_l = c3_c - 0.3
        candle3 = SyntheticCandle(c3_o, c3_h, c3_l, c3_c)

        pattern = [candle1, candle2, candle3]
        return context + pattern, [len(context), len(context) + 1, len(context) + 2]

    def _gen_three_white_soldiers(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Three White Soldiers: three consecutive bullish candles."""
        context = self._gen_downtrend_context(100, 2)
        last_close = context[-1].close

        candles = []
        price = last_close
        for i in range(3):
            o = price + (0.2 if i > 0 else 0)  # Open within previous body
            c = o + 3  # Large bullish body
            h = c + 0.3
            l = o - 0.2
            candles.append(SyntheticCandle(o, h, l, c))
            price = c

        return context + candles, [len(context), len(context) + 1, len(context) + 2]

    def _gen_three_black_crows(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Three Black Crows: three consecutive bearish candles."""
        context = self._gen_uptrend_context(100, 2)
        last_close = context[-1].close

        candles = []
        price = last_close
        for i in range(3):
            o = price - (0.2 if i > 0 else 0)  # Open within previous body
            c = o - 3  # Large bearish body
            h = o + 0.2
            l = c - 0.3
            candles.append(SyntheticCandle(o, h, l, c))
            price = c

        return context + candles, [len(context), len(context) + 1, len(context) + 2]

    def _gen_three_inside_up(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Three Inside Up: bearish, small bullish harami, bullish confirmation."""
        context = self._gen_downtrend_context(100, 2)
        last_close = context[-1].close

        # First: large bearish
        c1_o = last_close
        c1_c = last_close - 4
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: small bullish inside first body
        c2_o = c1_c + 0.8
        c2_c = c1_o - 1.2
        c2_h = c2_c + 0.2
        c2_l = c2_o - 0.2
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        # Third: bullish closing above first open
        c3_o = c2_c
        c3_c = c1_o + 1  # Closes above first open
        c3_h = c3_c + 0.3
        c3_l = c3_o - 0.2
        candle3 = SyntheticCandle(c3_o, c3_h, c3_l, c3_c)

        pattern = [candle1, candle2, candle3]
        return context + pattern, [len(context), len(context) + 1, len(context) + 2]

    def _gen_three_inside_down(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Three Inside Down: bullish, small bearish harami, bearish confirmation."""
        context = self._gen_uptrend_context(100, 2)
        last_close = context[-1].close

        # First: large bullish
        c1_o = last_close
        c1_c = last_close + 4
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: small bearish inside first body
        c2_o = c1_c - 0.8
        c2_c = c1_o + 1.2
        c2_h = c2_o + 0.2
        c2_l = c2_c - 0.2
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        # Third: bearish closing below first open
        c3_o = c2_c
        c3_c = c1_o - 1  # Closes below first open
        c3_h = c3_o + 0.2
        c3_l = c3_c - 0.3
        candle3 = SyntheticCandle(c3_o, c3_h, c3_l, c3_c)

        pattern = [candle1, candle2, candle3]
        return context + pattern, [len(context), len(context) + 1, len(context) + 2]

    def _gen_bullish_abandoned_baby(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bullish Abandoned Baby: bearish, doji with gaps, bullish."""
        context = self._gen_downtrend_context(100, 2)
        last_close = context[-1].close

        # First: large bearish
        c1_o = last_close
        c1_c = last_close - 4
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: doji with gap down (high below first low)
        c2_o = c1_l - 1.5
        c2_c = c2_o + 0.05
        c2_h = c2_o + 0.4
        c2_l = c2_o - 0.4
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        # Third: large bullish with gap up (low above doji high)
        c3_o = c2_h + 1
        c3_c = c1_o - 0.5
        c3_h = c3_c + 0.3
        c3_l = c3_o - 0.2
        candle3 = SyntheticCandle(c3_o, c3_h, c3_l, c3_c)

        pattern = [candle1, candle2, candle3]
        return context + pattern, [len(context), len(context) + 1, len(context) + 2]

    def _gen_bearish_abandoned_baby(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Bearish Abandoned Baby: bullish, doji with gaps, bearish."""
        context = self._gen_uptrend_context(100, 2)
        last_close = context[-1].close

        # First: large bullish
        c1_o = last_close
        c1_c = last_close + 4
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Second: doji with gap up (low above first high)
        c2_o = c1_h + 1.5
        c2_c = c2_o - 0.05
        c2_h = c2_o + 0.4
        c2_l = c2_o - 0.4
        candle2 = SyntheticCandle(c2_o, c2_h, c2_l, c2_c)

        # Third: large bearish with gap down (high below doji low)
        c3_o = c2_l - 1
        c3_c = c1_o + 0.5
        c3_h = c3_o + 0.2
        c3_l = c3_c - 0.3
        candle3 = SyntheticCandle(c3_o, c3_h, c3_l, c3_c)

        pattern = [candle1, candle2, candle3]
        return context + pattern, [len(context), len(context) + 1, len(context) + 2]

    # ==================== Multi-Candle Patterns ====================

    def _gen_rising_three_methods(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Rising Three Methods: bullish, 3 small bearish, bullish continuation."""
        context = self._gen_uptrend_context(100, 2)
        last_close = context[-1].close

        # First: large bullish
        c1_o = last_close
        c1_c = last_close + 5
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Three small bearish candles within first body
        inner_candles = []
        price = c1_c - 0.5
        for i in range(3):
            o = price
            c = o - 1
            h = o + 0.2
            l = c - 0.2
            inner_candles.append(SyntheticCandle(o, h, l, c))
            price = c

        # Last: large bullish continuing trend
        c5_o = inner_candles[-1].close + 0.3
        c5_c = c1_c + 2
        c5_h = c5_c + 0.3
        c5_l = c5_o - 0.2
        candle5 = SyntheticCandle(c5_o, c5_h, c5_l, c5_c)

        pattern = [candle1] + inner_candles + [candle5]
        return context + pattern, list(range(len(context), len(context) + 5))

    def _gen_falling_three_methods(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Falling Three Methods: bearish, 3 small bullish, bearish continuation."""
        context = self._gen_downtrend_context(100, 2)
        last_close = context[-1].close

        # First: large bearish
        c1_o = last_close
        c1_c = last_close - 5
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Three small bullish candles within first body
        inner_candles = []
        price = c1_c + 0.5
        for i in range(3):
            o = price
            c = o + 1
            h = c + 0.2
            l = o - 0.2
            inner_candles.append(SyntheticCandle(o, h, l, c))
            price = c

        # Last: large bearish continuing trend
        c5_o = inner_candles[-1].close - 0.3
        c5_c = c1_c - 2
        c5_h = c5_o + 0.2
        c5_l = c5_c - 0.3
        candle5 = SyntheticCandle(c5_o, c5_h, c5_l, c5_c)

        pattern = [candle1] + inner_candles + [candle5]
        return context + pattern, list(range(len(context), len(context) + 5))

    def _gen_tower_bottom(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Tower Bottom: large bearish, small consolidation candles, large bullish."""
        context = self._gen_downtrend_context(100, 2)
        last_close = context[-1].close

        # First: large bearish (tower wall)
        c1_o = last_close
        c1_c = last_close - 5
        c1_h = c1_o + 0.3
        c1_l = c1_c - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Three small consolidation candles
        inner_candles = []
        center = c1_c + 0.5
        for i in range(3):
            offset = 0.3 if i % 2 == 0 else -0.3
            o = center + offset
            c = center - offset + 0.2
            h = max(o, c) + 0.3
            l = min(o, c) - 0.3
            inner_candles.append(SyntheticCandle(o, h, l, c))

        # Last: large bullish (tower wall)
        c5_o = inner_candles[-1].close
        c5_c = c1_o  # Closes at first candle's open level
        c5_h = c5_c + 0.3
        c5_l = c5_o - 0.3
        candle5 = SyntheticCandle(c5_o, c5_h, c5_l, c5_c)

        pattern = [candle1] + inner_candles + [candle5]
        return context + pattern, list(range(len(context), len(context) + 5))

    def _gen_tower_top(self) -> Tuple[List[SyntheticCandle], List[int]]:
        """Tower Top: large bullish, small consolidation candles, large bearish."""
        context = self._gen_uptrend_context(100, 2)
        last_close = context[-1].close

        # First: large bullish (tower wall)
        c1_o = last_close
        c1_c = last_close + 5
        c1_h = c1_c + 0.3
        c1_l = c1_o - 0.3
        candle1 = SyntheticCandle(c1_o, c1_h, c1_l, c1_c)

        # Three small consolidation candles
        inner_candles = []
        center = c1_c - 0.5
        for i in range(3):
            offset = 0.3 if i % 2 == 0 else -0.3
            o = center + offset
            c = center - offset - 0.2
            h = max(o, c) + 0.3
            l = min(o, c) - 0.3
            inner_candles.append(SyntheticCandle(o, h, l, c))

        # Last: large bearish (tower wall)
        c5_o = inner_candles[-1].close
        c5_c = c1_o  # Closes at first candle's open level
        c5_h = c5_o + 0.3
        c5_l = c5_c - 0.3
        candle5 = SyntheticCandle(c5_o, c5_h, c5_l, c5_c)

        pattern = [candle1] + inner_candles + [candle5]
        return context + pattern, list(range(len(context), len(context) + 5))

    # ==================== Chart Rendering ====================

    def render_example_chart(
        self,
        pattern_type: str,
        show_labels: bool = True,
        compact: bool = False
    ) -> Optional[str]:
        """
        Render an example chart for a pattern type.

        Args:
            pattern_type: Name of the pattern
            show_labels: Whether to show labels and annotations
            compact: Smaller size for thumbnails

        Returns:
            Base64 encoded PNG image, or None if pattern not found
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot render chart - matplotlib not available")
            return None

        pattern_lower = pattern_type.lower()
        if pattern_lower not in self._pattern_generators:
            logger.warning(f"Unknown pattern type: {pattern_type}")
            return None

        try:
            # Generate synthetic data
            candles, pattern_indices = self._pattern_generators[pattern_lower]()
            info = self._pattern_info.get(pattern_lower, {})
            direction = info.get("direction", "neutral")

            # Figure size
            fig_size = (8, 4) if compact else self.figure_size

            # Create figure
            fig, ax = plt.subplots(figsize=fig_size, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["background"])

            # Plot candlesticks
            for i, candle in enumerate(candles):
                is_pattern = i in pattern_indices
                is_bullish = candle.is_bullish

                # Choose colors
                if is_bullish:
                    color = self.colors["bullish"]
                    body_color = self.colors["background"]  # Hollow
                else:
                    color = self.colors["bearish"]
                    body_color = self.colors["bearish"]  # Filled

                # Highlight pattern candles
                if is_pattern:
                    # Background highlight
                    highlight_color = self.colors["bullish"] if direction == "bullish" else \
                                     self.colors["bearish"] if direction == "bearish" else \
                                     "#ff9800"
                    ax.axvspan(i - 0.45, i + 0.45, color=highlight_color, alpha=0.15)

                # Draw wick
                line_width = 2 if is_pattern else 1
                ax.plot([i, i], [candle.low, candle.high], color=color, linewidth=line_width)

                # Draw body
                body_height = max(candle.body_size, (candle.total_range) * 0.01)
                rect = Rectangle(
                    (i - 0.35, candle.body_bottom),
                    0.7,
                    body_height,
                    facecolor=body_color,
                    edgecolor=color,
                    linewidth=2 if is_pattern else 1
                )
                ax.add_patch(rect)

            # Style
            ax.set_xlim(-0.5, len(candles) - 0.5)
            all_lows = [c.low for c in candles]
            all_highs = [c.high for c in candles]
            y_padding = (max(all_highs) - min(all_lows)) * 0.15
            ax.set_ylim(min(all_lows) - y_padding, max(all_highs) + y_padding * (1.5 if show_labels else 1))

            ax.grid(True, alpha=0.2, color=self.colors["grid"], linestyle='--')
            ax.tick_params(colors=self.colors["text"], labelsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if show_labels:
                # Title
                title_color = self.colors["bullish"] if direction == "bullish" else \
                             self.colors["bearish"] if direction == "bearish" else \
                             "#ff9800"
                title = pattern_type.replace("_", " ").title()
                ax.set_title(title, color=title_color, fontsize=14, fontweight="bold", pad=10)

                # Direction badge
                if direction != "neutral":
                    badge_text = "↑ BULLISH" if direction == "bullish" else "↓ BEARISH"
                    ax.text(
                        0.98, 0.98, badge_text,
                        transform=ax.transAxes,
                        fontsize=9,
                        color=title_color,
                        ha='right', va='top',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors["highlight_bg"],
                                  edgecolor=title_color, alpha=0.8)
                    )

                # Pattern bracket
                if pattern_indices:
                    pattern_start = min(pattern_indices)
                    pattern_end = max(pattern_indices)
                    pattern_center = (pattern_start + pattern_end) / 2
                    y_top = max(all_highs) + y_padding * 0.5

                    # Bracket line
                    ax.plot([pattern_start - 0.3, pattern_end + 0.3], [y_top, y_top],
                           color=self.colors["annotation"], linewidth=1.5)
                    ax.plot([pattern_start - 0.3, pattern_start - 0.3], [y_top, y_top - y_padding * 0.2],
                           color=self.colors["annotation"], linewidth=1.5)
                    ax.plot([pattern_end + 0.3, pattern_end + 0.3], [y_top, y_top - y_padding * 0.2],
                           color=self.colors["annotation"], linewidth=1.5)

            plt.tight_layout()

            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, facecolor=self.colors["background"],
                       bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)

            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return image_base64

        except Exception as e:
            logger.error(f"Failed to render example chart for {pattern_type}: {e}")
            return None

    def get_all_examples(self, compact: bool = False) -> Dict[str, str]:
        """
        Generate example charts for all available patterns.

        Returns:
            Dictionary mapping pattern type to base64 image
        """
        examples = {}
        for pattern_type in self._pattern_generators.keys():
            image = self.render_example_chart(pattern_type, show_labels=True, compact=compact)
            if image:
                examples[pattern_type] = image
        return examples


# Global singleton
pattern_example_service = PatternExampleService()
