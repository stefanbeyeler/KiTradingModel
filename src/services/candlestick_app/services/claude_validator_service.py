"""
Claude Vision Validator Service for Candlestick Pattern QA.

This service generates chart images of detected patterns and sends them
to Claude AI for visual validation. This provides an external quality
assurance mechanism for pattern detection.

Architecture:
    1. Rule-based detection -> Pattern detected
    2. Chart rendering -> Visual representation of pattern
    3. Claude Vision API -> External validation
    4. Result -> Agreement/disagreement with confidence
"""

import os
import io
import base64
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from loguru import logger

# Try to import matplotlib for chart rendering
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - chart rendering disabled")


class ValidationStatus(str, Enum):
    """Status of Claude validation."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ClaudeValidationResult:
    """Result of Claude AI validation for a pattern."""
    pattern_id: str
    pattern_type: str
    symbol: str
    timeframe: str

    # Claude's assessment
    claude_agrees: bool
    claude_confidence: float  # 0.0 - 1.0
    claude_pattern_type: Optional[str]  # What Claude thinks it is
    claude_reasoning: str

    # Quality metrics
    visual_quality_score: float  # How well-formed the pattern looks
    market_context_score: float  # How appropriate the context is

    # Metadata
    validation_timestamp: str
    model_used: str
    status: ValidationStatus
    error_message: Optional[str] = None

    # Chart image (base64) for reference
    chart_image_base64: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class ChartRenderer:
    """
    Renders candlestick charts for pattern visualization.

    Uses unified color scheme from src/config/chart_colors.py to ensure
    consistency between manual validation (Frontend) and Claude AI validation.
    """

    def __init__(self):
        self.figure_size = (12, 6)
        self.dpi = 100
        # Unified colors - matching Frontend JavaScript implementation
        # See: src/config/chart_colors.py for documentation
        self.colors = {
            "bullish": "#4caf50",            # Green - same as Frontend
            "bullish_highlight": "#66bb6a",  # Lighter green for highlighted
            "bearish": "#f44336",            # Red - same as Frontend
            "bearish_highlight": "#ef5350",  # Lighter red for highlighted
            "neutral": "#ff9800",            # Orange for neutral patterns
            "background": "#1a1a2e",         # Dark background - same as Frontend
            "grid": "#333333",               # Grid lines
            "text": "#ffffff",               # White text
            # MUCH stronger highlight for Claude to identify - 50% alpha instead of 20%
            "highlight_bullish": (76/255, 175/255, 80/255, 0.5),   # RGBA tuple - STRONG
            "highlight_bearish": (244/255, 67/255, 54/255, 0.5),   # RGBA tuple - STRONG
            "highlight_neutral": (255/255, 152/255, 0/255, 0.5),   # RGBA tuple - STRONG
            # Border color for pattern candles
            "pattern_border": "#ffffff",     # White border around pattern candles
        }

    def render_pattern_chart(
        self,
        ohlcv_data: List[Dict],
        pattern_type: str,
        pattern_candles: int = 3,
        context_before: int = 5,
        context_after: int = 5,
        direction: str = "neutral"
    ) -> Optional[str]:
        """
        Render a candlestick chart with the pattern centered.

        Args:
            ohlcv_data: List of OHLCV dictionaries (pattern at end of data)
            pattern_type: Name of the pattern
            pattern_candles: Number of candles in the pattern
            context_before: Number of context candles before pattern
            context_after: Number of context candles after pattern (empty space)
            direction: Pattern direction ("bullish", "bearish", or "neutral")

        Returns:
            Base64 encoded PNG image, or None if rendering fails
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot render chart - matplotlib not available")
            return None

        try:
            # Pattern is at the end of ohlcv_data
            # We take: context_before + pattern_candles from the end
            candles_needed = context_before + pattern_candles

            if len(ohlcv_data) >= candles_needed:
                data = ohlcv_data[-candles_needed:]
            else:
                data = ohlcv_data

            # Pattern starts at index: context_before (0-indexed)
            # Pattern ends at: len(data) - 1
            pattern_start_idx = len(data) - pattern_candles

            # Total display width: data + empty space after
            total_display = len(data) + context_after

            # Determine highlight color based on direction
            highlight_color_key = f"highlight_{direction}" if direction in ["bullish", "bearish"] else "highlight_neutral"
            highlight_color = self.colors.get(highlight_color_key, self.colors["highlight_neutral"])

            # Determine label color based on direction
            label_color = self.colors.get(direction, self.colors["neutral"])

            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["background"])

            # Plot candlesticks
            for i, candle in enumerate(data):
                o = float(candle.get("open", candle.get("o", 0)))
                h = float(candle.get("high", candle.get("h", 0)))
                l = float(candle.get("low", candle.get("l", 0)))
                c = float(candle.get("close", candle.get("c", 0)))

                is_bullish = c >= o
                is_pattern_candle = i >= pattern_start_idx

                # Select color based on candle direction and highlight state
                if is_bullish:
                    color = self.colors["bullish_highlight"] if is_pattern_candle else self.colors["bullish"]
                else:
                    color = self.colors["bearish_highlight"] if is_pattern_candle else self.colors["bearish"]

                # Highlight pattern candles with STRONG background + white border
                if is_pattern_candle:
                    # Strong colored background
                    ax.axvspan(
                        i - 0.45, i + 0.45,
                        color=highlight_color[:3] if isinstance(highlight_color, tuple) else highlight_color,
                        alpha=highlight_color[3] if isinstance(highlight_color, tuple) else 0.5
                    )
                    # Add white dashed border around the pattern area for clarity
                    ax.axvline(x=i - 0.45, color=self.colors["pattern_border"], linestyle='--', linewidth=1.5, alpha=0.8)
                    ax.axvline(x=i + 0.45, color=self.colors["pattern_border"], linestyle='--', linewidth=1.5, alpha=0.8)

                # Draw wick (high-low line) - MUCH thicker for highlighted candles
                line_width = 3 if is_pattern_candle else 1
                ax.plot([i, i], [l, h], color=color, linewidth=line_width)

                # Draw body
                body_bottom = min(o, c)
                body_height = abs(c - o)
                if body_height < 0.0001:  # Doji
                    body_height = (h - l) * 0.01

                # Bullish candles: hollow (dark fill with colored border)
                # Bearish candles: filled
                # Pattern candles get WHITE border for maximum visibility
                if is_bullish:
                    rect = Rectangle(
                        (i - 0.3, body_bottom),
                        0.6,
                        body_height,
                        facecolor='#1a1a2e',  # Dark fill for hollow effect
                        edgecolor=self.colors["pattern_border"] if is_pattern_candle else color,
                        linewidth=3 if is_pattern_candle else 1
                    )
                else:
                    rect = Rectangle(
                        (i - 0.3, body_bottom),
                        0.6,
                        body_height,
                        facecolor=color,
                        edgecolor=self.colors["pattern_border"] if is_pattern_candle else color,
                        linewidth=3 if is_pattern_candle else 1
                    )
                ax.add_patch(rect)

            # Draw empty placeholder candles after pattern (gray outlines)
            if context_after > 0:
                # Get average candle size for placeholder
                all_ranges = [float(c.get("high", c.get("h", 0))) - float(c.get("low", c.get("l", 0))) for c in data]
                avg_range = sum(all_ranges) / len(all_ranges) if all_ranges else 0
                last_close = float(data[-1].get("close", data[-1].get("c", 0)))

                for j in range(context_after):
                    x_pos = len(data) + j
                    # Draw a subtle placeholder line
                    ax.plot(
                        [x_pos, x_pos],
                        [last_close - avg_range * 0.3, last_close + avg_range * 0.3],
                        color=self.colors["grid"],
                        linewidth=1,
                        linestyle=':'
                    )

            # Style the chart
            ax.set_xlim(-0.5, total_display - 0.5)

            # Calculate y-axis limits with padding
            all_highs = [float(c.get("high", c.get("h", 0))) for c in data]
            all_lows = [float(c.get("low", c.get("l", 0))) for c in data]
            y_min = min(all_lows)
            y_max = max(all_highs)
            y_padding = (y_max - y_min) * 0.15
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Grid
            ax.grid(True, alpha=0.3, color=self.colors["grid"])
            ax.tick_params(colors=self.colors["text"])

            # Title with direction-based color
            ax.set_title(
                f"Pattern: {pattern_type.replace('_', ' ').title()}",
                color=label_color,
                fontsize=14,
                fontweight="bold"
            )

            # Add annotation for pattern area with direction-based color
            pattern_center = pattern_start_idx + pattern_candles / 2 - 0.5
            ax.annotate(
                "Pattern",
                xy=(pattern_center, y_max),
                xytext=(pattern_center, y_max + y_padding * 0.4),
                color=label_color,
                fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=label_color)
            )

            # Remove spines
            for spine in ax.spines.values():
                spine.set_color(self.colors["grid"])

            plt.tight_layout()

            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, facecolor=self.colors["background"])
            buf.seek(0)

            # Encode to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            plt.close(fig)

            return image_base64

        except Exception as e:
            logger.error(f"Failed to render chart: {e}")
            return None


class ClaudeValidatorService:
    """
    Service for validating candlestick patterns using Claude Vision API.

    This provides external quality assurance by having Claude analyze
    chart images and provide feedback on pattern validity.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = os.getenv("CLAUDE_VALIDATOR_MODEL", "claude-sonnet-4-20250514")
        self.chart_renderer = ChartRenderer()

        # Validation history
        self._validation_history: List[ClaudeValidationResult] = []
        self._history_file = Path(os.getenv("DATA_DIR", "/app/data")) / "claude_validations.json"
        self._load_history()

        # Rate limiting
        self._last_request_time = None
        self._min_request_interval = 1.0  # Minimum seconds between requests

        # Cache for validated patterns
        self._validation_cache: Dict[str, ClaudeValidationResult] = {}
        self._cache_max_size = 100

        logger.info(f"Claude Validator initialized with model: {self.model}")

    def _load_history(self):
        """Load validation history from file."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Keep only last 1000 entries
                    self._validation_history = data[-1000:]
                logger.info(f"Loaded {len(self._validation_history)} validation history entries")
        except Exception as e:
            logger.warning(f"Failed to load validation history: {e}")
            self._validation_history = []

    def _save_history(self):
        """Save validation history to file."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump(self._validation_history[-1000:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save validation history: {e}")

    def _get_pattern_candle_count(self, pattern_type: str) -> int:
        """Get the number of candles involved in a pattern."""
        single_candle = ["doji", "dragonfly_doji", "gravestone_doji", "hammer",
                        "inverted_hammer", "shooting_star", "hanging_man", "spinning_top"]
        two_candle = ["bullish_engulfing", "bearish_engulfing", "bullish_harami",
                     "bearish_harami", "harami_cross", "piercing_line", "dark_cloud_cover"]
        three_candle = ["morning_star", "evening_star", "three_white_soldiers",
                       "three_black_crows", "rising_three_methods", "falling_three_methods"]

        pattern_lower = pattern_type.lower()
        if pattern_lower in single_candle:
            return 1
        elif pattern_lower in two_candle:
            return 2
        elif pattern_lower in three_candle:
            return 3
        else:
            return 2  # Default

    def _get_pattern_direction(self, pattern_type: str) -> str:
        """
        Get the direction (bullish/bearish/neutral) of a pattern type.

        This ensures consistent color-coding between Frontend and Backend rendering.
        """
        bullish_patterns = [
            "hammer", "inverted_hammer", "bullish_engulfing", "bullish_harami",
            "morning_star", "three_white_soldiers", "piercing_line",
            "dragonfly_doji", "rising_three_methods", "tweezer_bottom"
        ]
        bearish_patterns = [
            "hanging_man", "shooting_star", "bearish_engulfing", "bearish_harami",
            "evening_star", "three_black_crows", "dark_cloud_cover",
            "gravestone_doji", "falling_three_methods", "tweezer_top"
        ]
        neutral_patterns = [
            "doji", "spinning_top", "harami_cross", "inside_bar"
        ]

        pattern_lower = pattern_type.lower()
        if pattern_lower in bullish_patterns:
            return "bullish"
        elif pattern_lower in bearish_patterns:
            return "bearish"
        else:
            return "neutral"

    def _calculate_candle_metrics(self, candle: Dict) -> Dict[str, float]:
        """
        Calculate precise metrics for a candle to provide to Claude.

        Returns:
            Dict with body_ratio, upper_shadow_ratio, lower_shadow_ratio, body_position
        """
        o = float(candle.get("open", candle.get("o", 0)))
        h = float(candle.get("high", candle.get("h", 0)))
        l = float(candle.get("low", candle.get("l", 0)))
        c = float(candle.get("close", candle.get("c", 0)))

        total_range = h - l
        if total_range <= 0:
            return {
                "open": o, "high": h, "low": l, "close": c,
                "total_range": 0, "body_size": 0,
                "body_top": o, "body_bottom": o,
                "body_ratio": 0, "upper_shadow_ratio": 0, "lower_shadow_ratio": 0,
                "body_position": "middle", "is_bullish": c >= o
            }

        body_size = abs(c - o)
        body_ratio = body_size / total_range

        body_top = max(o, c)
        body_bottom = min(o, c)

        upper_shadow = h - body_top
        lower_shadow = body_bottom - l

        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range

        # Body position: where is the body within the candle?
        body_center = (body_top + body_bottom) / 2
        candle_center = (h + l) / 2

        if body_center > candle_center + total_range * 0.15:
            body_position = "top"
        elif body_center < candle_center - total_range * 0.15:
            body_position = "bottom"
        else:
            body_position = "middle"

        return {
            "open": round(o, 6),
            "high": round(h, 6),
            "low": round(l, 6),
            "close": round(c, 6),
            "total_range": round(total_range, 6),
            "body_size": round(body_size, 6),
            "body_top": round(body_top, 6),
            "body_bottom": round(body_bottom, 6),
            "body_ratio": round(body_ratio * 100, 1),  # As percentage
            "upper_shadow_ratio": round(upper_shadow_ratio * 100, 1),
            "lower_shadow_ratio": round(lower_shadow_ratio * 100, 1),
            "body_position": body_position,
            "is_bullish": c >= o
        }

    def _get_pattern_criteria(self, pattern_type: str) -> str:
        """Get specific criteria for each pattern type."""
        criteria = {
            "shooting_star": """
**STRICT Shooting Star Criteria (ALL must be met):**
- Small real body near the LOW of the candle (body in bottom 1/3)
- Long UPPER shadow at least 2x the body length
- Little to NO lower shadow (less than 10% of total range)
- Appears after an uptrend (prior candles should be bullish/rising)
- Color of body (red or green) is less important than position""",

            "hanging_man": """
**STRICT Hanging Man Criteria (ALL must be met):**
- Small real body near the HIGH of the candle (body in top 1/3)
- Long LOWER shadow at least 2x the body length
- Little to NO upper shadow (less than 10% of total range)
- Appears after an uptrend (prior candles should be bullish/rising)
- Color of body is less important than position""",

            "hammer": """
**STRICT Hammer Criteria (ALL must be met):**
- Small real body near the HIGH of the candle (body in top 1/3)
- Long LOWER shadow at least 2x the body length
- Little to NO upper shadow (less than 10% of total range)
- Appears after a downtrend (prior candles should be bearish/falling)
- Color of body is less important than position""",

            "inverted_hammer": """
**STRICT Inverted Hammer Criteria (ALL must be met):**
- Small real body near the LOW of the candle (body in bottom 1/3)
- Long UPPER shadow at least 2x the body length
- Little to NO lower shadow (less than 10% of total range)
- Appears after a downtrend (prior candles should be bearish/falling)
- Color of body is less important than position""",

            "doji": """
**STRICT Doji Criteria (ALL must be met):**
- Extremely small or no body (open ≈ close, less than 5% of total range)
- Upper and lower shadows can vary but body must be tiny
- The cross/plus shape is the defining characteristic""",

            "bullish_engulfing": """
**STRICT Bullish Engulfing Criteria (ALL must be met):**
- Previous candle must be bearish (red/filled)
- Current candle must be bullish (green/hollow)
- Current body COMPLETELY engulfs previous body
- Shadows don't need to engulf, only the BODY""",

            "bearish_engulfing": """
**STRICT Bearish Engulfing Criteria (ALL must be met):**
- Previous candle must be bullish (green/hollow)
- Current candle must be bearish (red/filled)
- Current body COMPLETELY engulfs previous body
- Shadows don't need to engulf, only the BODY""",
        }
        return criteria.get(pattern_type.lower(), "")

    def _build_validation_prompt(
        self,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        candle_metrics: Optional[Dict] = None,
        prev_candle_metrics: Optional[Dict] = None
    ) -> str:
        """Build the prompt for Claude to validate a pattern."""
        specific_criteria = self._get_pattern_criteria(pattern_type)

        # Build metrics section if available
        metrics_section = ""
        validation_instruction = ""

        # Check if this is a multi-candle pattern (engulfing, harami, etc.)
        is_engulfing = pattern_type.lower() in ["bullish_engulfing", "bearish_engulfing"]
        is_harami = pattern_type.lower() in ["bullish_harami", "bearish_harami", "harami_cross"]
        is_two_candle = is_engulfing or is_harami

        if candle_metrics and is_two_candle and prev_candle_metrics:
            # Multi-candle pattern: show BOTH candles
            metrics_section = f"""
**EXAKTE OHLC-DATEN FÜR BEIDE KERZEN (verwende diese Zahlen, keine visuelle Schätzung!):**

**VORHERIGE KERZE (Kerze 1):**
- Open: {prev_candle_metrics['open']}
- High: {prev_candle_metrics['high']}
- Low: {prev_candle_metrics['low']}
- Close: {prev_candle_metrics['close']}
- Body Top: {prev_candle_metrics['body_top']}
- Body Bottom: {prev_candle_metrics['body_bottom']}
- Kerzenrichtung: {"Bullish (grün)" if prev_candle_metrics['is_bullish'] else "Bearish (rot)"}

**AKTUELLE KERZE (Kerze 2 - Pattern-Kerze):**
- Open: {candle_metrics['open']}
- High: {candle_metrics['high']}
- Low: {candle_metrics['low']}
- Close: {candle_metrics['close']}
- Body Top: {candle_metrics['body_top']}
- Body Bottom: {candle_metrics['body_bottom']}
- Kerzenrichtung: {"Bullish (grün)" if candle_metrics['is_bullish'] else "Bearish (rot)"}
"""
            # Automatic validation for Engulfing patterns
            if is_engulfing:
                curr_body_top = candle_metrics['body_top']
                curr_body_bottom = candle_metrics['body_bottom']
                prev_body_top = prev_candle_metrics['body_top']
                prev_body_bottom = prev_candle_metrics['body_bottom']

                # Check direction requirements
                if pattern_type.lower() == "bearish_engulfing":
                    direction_ok = prev_candle_metrics['is_bullish'] and not candle_metrics['is_bullish']
                    direction_msg = "Vorherige Kerze bullish (grün) UND aktuelle Kerze bearish (rot)"
                else:  # bullish_engulfing
                    direction_ok = not prev_candle_metrics['is_bullish'] and candle_metrics['is_bullish']
                    direction_msg = "Vorherige Kerze bearish (rot) UND aktuelle Kerze bullish (grün)"

                # Check engulfing: current body must completely contain previous body
                engulfs = curr_body_top >= prev_body_top and curr_body_bottom <= prev_body_bottom

                metrics_section += f"""
**ENGULFING-PRÜFUNG (automatisch berechnet):**
- Richtungsbedingung ({direction_msg}): {"✅ ERFÜLLT" if direction_ok else "❌ NICHT ERFÜLLT"}
- Vorherige Body-Grenzen: Top={prev_body_top}, Bottom={prev_body_bottom}
- Aktuelle Body-Grenzen: Top={curr_body_top}, Bottom={curr_body_bottom}
- Umhüllt vollständig (curr_top >= prev_top UND curr_bottom <= prev_bottom): {"✅ JA" if engulfs else "❌ NEIN"}
"""
                if direction_ok and engulfs:
                    validation_instruction = f"""
**AUTOMATISCHE BESTÄTIGUNG ERFORDERLICH:**
Die mathematische Prüfung zeigt:
- Richtung korrekt: Vorherige Kerze {"bullish" if prev_candle_metrics['is_bullish'] else "bearish"}, aktuelle Kerze {"bullish" if candle_metrics['is_bullish'] else "bearish"}
- Body der aktuellen Kerze ({curr_body_bottom} bis {curr_body_top}) umhüllt Body der vorherigen Kerze ({prev_body_bottom} bis {prev_body_top}) vollständig
Du MUSST agrees=true setzen. Dies ist ein GÜLTIGES {pattern_type.replace('_', ' ').title()} Pattern."""
                else:
                    reasons = []
                    if not direction_ok:
                        reasons.append(f"Richtung falsch - erwartet: {direction_msg}")
                    if not engulfs:
                        reasons.append(f"Body umhüllt nicht vollständig (aktuell: {curr_body_bottom}-{curr_body_top}, vorher: {prev_body_bottom}-{prev_body_top})")
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Die mathematische Prüfung ist fehlgeschlagen:
{chr(10).join('- ' + r for r in reasons)}
Du MUSST agrees=false setzen."""

            elif is_harami:
                curr_body_top = candle_metrics['body_top']
                curr_body_bottom = candle_metrics['body_bottom']
                prev_body_top = prev_candle_metrics['body_top']
                prev_body_bottom = prev_candle_metrics['body_bottom']

                # Harami: previous body contains current body
                contained = prev_body_top >= curr_body_top and prev_body_bottom <= curr_body_bottom

                metrics_section += f"""
**HARAMI-PRÜFUNG (automatisch berechnet):**
- Vorherige Body-Grenzen: Top={prev_body_top}, Bottom={prev_body_bottom}
- Aktuelle Body-Grenzen: Top={curr_body_top}, Bottom={curr_body_bottom}
- Enthalten in vorheriger Kerze (prev_top >= curr_top UND prev_bottom <= curr_bottom): {"✅ JA" if contained else "❌ NEIN"}
"""
                if contained:
                    validation_instruction = """
**Metrik-Prüfung BESTANDEN:** Aktuelle Kerze ist innerhalb der vorherigen Kerze enthalten."""
                else:
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Die aktuelle Kerze ({curr_body_bottom} bis {curr_body_top}) ist NICHT vollständig in der vorherigen Kerze ({prev_body_bottom} bis {prev_body_top}) enthalten.
Du MUSST agrees=false setzen."""

        elif candle_metrics:
            # Single candle pattern
            metrics_section = f"""
**EXAKTE OHLC-DATEN DER PATTERN-KERZE (verwende diese Zahlen, keine visuelle Schätzung!):**
- Open: {candle_metrics['open']}
- High: {candle_metrics['high']}
- Low: {candle_metrics['low']}
- Close: {candle_metrics['close']}
- Total Range (High-Low): {candle_metrics['total_range']}
- Body Size: {candle_metrics['body_size']}
- **Body Ratio: {candle_metrics['body_ratio']}%** (body / total range)
- Upper Shadow Ratio: {candle_metrics['upper_shadow_ratio']}%
- Lower Shadow Ratio: {candle_metrics['lower_shadow_ratio']}%
- Body Position: {candle_metrics['body_position']}
- Kerzenrichtung: {"Bullish (grün)" if candle_metrics['is_bullish'] else "Bearish (rot)"}
"""
            # Add specific validation based on pattern type and actual metrics
            body_ratio = candle_metrics['body_ratio']
            upper_ratio = candle_metrics['upper_shadow_ratio']
            lower_ratio = candle_metrics['lower_shadow_ratio']

            if pattern_type.lower() == "doji":
                if body_ratio > 5:
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Die Body-Ratio beträgt {body_ratio}%, was > 5% ist. Dies KANN KEIN Doji sein.
Du MUSST agrees=false setzen."""
                else:
                    validation_instruction = f"""
**Metrik-Prüfung BESTANDEN:** Body-Ratio {body_ratio}% ist < 5%, konsistent mit Doji."""

            elif pattern_type.lower() == "shooting_star":
                if body_ratio > 35:
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Die Body-Ratio beträgt {body_ratio}%, was > 35% ist. Dies KANN KEIN Shooting Star sein.
Du MUSST agrees=false setzen."""
                elif upper_ratio < 2 * body_ratio:
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Oberer Schatten ({upper_ratio}%) ist nicht mindestens 2x der Körper ({body_ratio}%).
Du MUSST agrees=false setzen."""
                else:
                    validation_instruction = f"""
**Metrik-Prüfung BESTANDEN:** Body-Ratio {body_ratio}% < 35%, oberer Schatten {upper_ratio}%."""

            elif pattern_type.lower() == "hammer":
                if body_ratio > 35:
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Die Body-Ratio beträgt {body_ratio}%, was > 35% ist. Dies KANN KEIN Hammer sein.
Du MUSST agrees=false setzen."""
                elif lower_ratio < 2 * body_ratio:
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Unterer Schatten ({lower_ratio}%) ist nicht mindestens 2x der Körper ({body_ratio}%).
Du MUSST agrees=false setzen."""
                else:
                    validation_instruction = f"""
**Metrik-Prüfung BESTANDEN:** Body-Ratio {body_ratio}% < 35%, unterer Schatten {lower_ratio}%."""

            elif pattern_type.lower() in ["hanging_man", "inverted_hammer"]:
                if body_ratio > 35:
                    validation_instruction = f"""
**AUTOMATISCHE ABLEHNUNG ERFORDERLICH:**
Die Body-Ratio beträgt {body_ratio}%, was > 35% ist. Dies KANN KEIN {pattern_type.replace('_', ' ').title()} sein.
Du MUSST agrees=false setzen."""

        return f"""Du bist ein Experte für technische Analyse, spezialisiert auf Candlestick-Muster-Erkennung.
Du musst die bereitgestellten Metriken verwenden und die automatischen Prüfungen respektieren.

**Symbol:** {symbol}
**Timeframe:** {timeframe}
**Behauptetes Muster:** {pattern_type.replace('_', ' ').title()}
{metrics_section}
{validation_instruction}
{specific_criteria}

**WIE DU DIE PATTERN-KERZEN IM CHART IDENTIFIZIERST:**
Die Pattern-Kerzen sind markiert mit:
- Einem STARKEN farbigen Hintergrund (orange/grün/rot)
- WEISSEN GESTRICHELTEN vertikalen Linien auf beiden Seiten
- Einem WEISSEN RAND um den Kerzenkörper

**KRITISCH: VERWENDE DIE EXAKTEN METRIKEN OBEN, KEINE VISUELLE SCHÄTZUNG!**
Die bereitgestellten Metriken sind aus den tatsächlichen OHLC-Daten berechnet und 100% genau.
Die automatischen Prüfungen (AUTOMATISCHE BESTÄTIGUNG/ABLEHNUNG) MÜSSEN respektiert werden.

**FÜR ENGULFING-MUSTER:**
- Nur der KÖRPER muss umhüllt werden, NICHT die Schatten
- Die mathematische Prüfung oben ist verbindlich

Antworte in diesem exakten JSON-Format:
```json
{{
    "agrees": true/false,
    "confidence": 0.0-1.0,
    "detected_pattern": "pattern_name_oder_null",
    "visual_quality": 0.0-1.0,
    "market_context": 0.0-1.0,
    "reasoning": "Kurze Begründung auf DEUTSCH mit Bezug auf die tatsächlichen Metriken"
}}
```

Wenn oben AUTOMATISCHE BESTÄTIGUNG angegeben ist, MUSST du agrees=true setzen.
Wenn oben AUTOMATISCHE ABLEHNUNG angegeben ist, MUSST du agrees=false setzen.
WICHTIG: Die "reasoning" Begründung MUSS auf Deutsch sein!"""

    async def _rate_limit(self):
        """Apply rate limiting between API requests."""
        if self._last_request_time is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = datetime.now(timezone.utc)

    async def validate_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        ohlcv_data: List[Dict],
        force: bool = False
    ) -> ClaudeValidationResult:
        """
        Validate a detected pattern using Claude Vision API.

        Args:
            pattern_id: Unique identifier for the pattern
            pattern_type: Type of pattern detected
            symbol: Trading symbol
            timeframe: Chart timeframe
            ohlcv_data: OHLCV data including pattern and context
            force: Force validation even if cached

        Returns:
            ClaudeValidationResult with Claude's assessment
        """
        # Check cache
        if not force and pattern_id in self._validation_cache:
            logger.debug(f"Returning cached validation for {pattern_id}")
            return self._validation_cache[pattern_id]

        # Check if API key is configured
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not configured - skipping Claude validation")
            return ClaudeValidationResult(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                claude_agrees=True,  # Assume correct if can't validate
                claude_confidence=0.0,
                claude_pattern_type=None,
                claude_reasoning="Claude validation not configured (no API key)",
                visual_quality_score=0.0,
                market_context_score=0.0,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                model_used="none",
                status=ValidationStatus.SKIPPED
            )

        try:
            # Render chart with direction-based colors for consistency with Frontend
            pattern_candles = self._get_pattern_candle_count(pattern_type)
            direction = self._get_pattern_direction(pattern_type)
            chart_base64 = self.chart_renderer.render_pattern_chart(
                ohlcv_data=ohlcv_data,
                pattern_type=pattern_type,
                pattern_candles=pattern_candles,
                context_before=5,
                context_after=5,
                direction=direction
            )

            if not chart_base64:
                raise ValueError("Failed to render chart image")

            # Calculate metrics for the pattern candle (last candle in data)
            candle_metrics = None
            prev_candle_metrics = None

            # Check if this is a multi-candle pattern
            is_two_candle = pattern_type.lower() in [
                "bullish_engulfing", "bearish_engulfing",
                "bullish_harami", "bearish_harami", "harami_cross",
                "piercing_line", "dark_cloud_cover"
            ]

            if ohlcv_data:
                # Get the pattern candle (last one)
                pattern_candle = ohlcv_data[-1]
                candle_metrics = self._calculate_candle_metrics(pattern_candle)

                # For 2-candle patterns, also get the previous candle
                if is_two_candle and len(ohlcv_data) >= 2:
                    prev_candle = ohlcv_data[-2]
                    prev_candle_metrics = self._calculate_candle_metrics(prev_candle)
                    logger.info(
                        f"Two-candle pattern {pattern_type}: "
                        f"prev_body={prev_candle_metrics['body_bottom']}-{prev_candle_metrics['body_top']} "
                        f"({'bullish' if prev_candle_metrics['is_bullish'] else 'bearish'}), "
                        f"curr_body={candle_metrics['body_bottom']}-{candle_metrics['body_top']} "
                        f"({'bullish' if candle_metrics['is_bullish'] else 'bearish'})"
                    )
                else:
                    logger.info(
                        f"Pattern candle metrics for {pattern_type}: "
                        f"body_ratio={candle_metrics['body_ratio']}%, "
                        f"upper_shadow={candle_metrics['upper_shadow_ratio']}%, "
                        f"lower_shadow={candle_metrics['lower_shadow_ratio']}%"
                    )

            # Build prompt with metrics (including prev_candle for 2-candle patterns)
            prompt = self._build_validation_prompt(
                pattern_type, symbol, timeframe, candle_metrics, prev_candle_metrics
            )

            # Apply rate limiting
            await self._rate_limit()

            # Call Claude API
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 1024,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": chart_base64
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ]
                    }
                )

                if response.status_code != 200:
                    raise ValueError(f"API error: {response.status_code} - {response.text}")

                result_data = response.json()
                content = result_data.get("content", [{}])[0].get("text", "")

            # Parse Claude's response
            validation_result = self._parse_claude_response(
                content=content,
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                chart_base64=chart_base64
            )

            # Cache result
            self._cache_result(validation_result)

            # Add to history
            self._validation_history.append(validation_result.to_dict())
            self._save_history()

            logger.info(
                f"Claude validation for {pattern_type} ({symbol}/{timeframe}): "
                f"agrees={validation_result.claude_agrees}, "
                f"confidence={validation_result.claude_confidence:.2f}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Claude validation failed: {e}")
            return ClaudeValidationResult(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                claude_agrees=True,  # Fail open
                claude_confidence=0.0,
                claude_pattern_type=None,
                claude_reasoning=f"Validation error: {str(e)}",
                visual_quality_score=0.0,
                market_context_score=0.0,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                model_used=self.model,
                status=ValidationStatus.ERROR,
                error_message=str(e)
            )

    def _parse_claude_response(
        self,
        content: str,
        pattern_id: str,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        chart_base64: Optional[str] = None
    ) -> ClaudeValidationResult:
        """Parse Claude's JSON response into a validation result."""
        try:
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            return ClaudeValidationResult(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                claude_agrees=data.get("agrees", True),
                claude_confidence=float(data.get("confidence", 0.5)),
                claude_pattern_type=data.get("detected_pattern"),
                claude_reasoning=data.get("reasoning", "No reasoning provided"),
                visual_quality_score=float(data.get("visual_quality", 0.5)),
                market_context_score=float(data.get("market_context", 0.5)),
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                model_used=self.model,
                status=ValidationStatus.VALIDATED if data.get("agrees") else ValidationStatus.REJECTED,
                chart_image_base64=chart_base64
            )

        except Exception as e:
            logger.warning(f"Failed to parse Claude response: {e}")
            return ClaudeValidationResult(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                claude_agrees=True,
                claude_confidence=0.5,
                claude_pattern_type=None,
                claude_reasoning=f"Response parsing failed: {content[:200]}...",
                visual_quality_score=0.5,
                market_context_score=0.5,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                model_used=self.model,
                status=ValidationStatus.ERROR,
                error_message=str(e),
                chart_image_base64=chart_base64
            )

    def _cache_result(self, result: ClaudeValidationResult):
        """Cache a validation result."""
        self._validation_cache[result.pattern_id] = result

        # Enforce cache size limit
        if len(self._validation_cache) > self._cache_max_size:
            # Remove oldest entries
            oldest_keys = list(self._validation_cache.keys())[:-self._cache_max_size]
            for key in oldest_keys:
                del self._validation_cache[key]

    async def validate_patterns_batch(
        self,
        patterns: List[Dict],
        ohlcv_data_map: Dict[str, List[Dict]],
        max_validations: int = 10
    ) -> List[ClaudeValidationResult]:
        """
        Validate multiple patterns in batch.

        Args:
            patterns: List of pattern dictionaries
            ohlcv_data_map: Map of pattern_id -> OHLCV data
            max_validations: Maximum number of validations to perform

        Returns:
            List of validation results
        """
        results = []

        for pattern in patterns[:max_validations]:
            pattern_id = pattern.get("id", "")
            ohlcv_data = ohlcv_data_map.get(pattern_id, [])

            if not ohlcv_data:
                logger.warning(f"No OHLCV data for pattern {pattern_id}")
                continue

            result = await self.validate_pattern(
                pattern_id=pattern_id,
                pattern_type=pattern.get("pattern_type", "unknown"),
                symbol=pattern.get("symbol", ""),
                timeframe=pattern.get("timeframe", ""),
                ohlcv_data=ohlcv_data
            )
            results.append(result)

        return results

    def get_validation_history(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
        pattern_type: Optional[str] = None,
        status: Optional[ValidationStatus] = None
    ) -> List[Dict]:
        """Get validation history with optional filters."""
        history = self._validation_history

        if symbol:
            history = [h for h in history if h.get("symbol") == symbol.upper()]

        if pattern_type:
            history = [h for h in history if h.get("pattern_type") == pattern_type.lower()]

        if status:
            history = [h for h in history if h.get("status") == status.value]

        return list(reversed(history[-limit:]))

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about Claude validations."""
        if not self._validation_history:
            return {
                "total_validations": 0,
                "agreements": 0,
                "rejections": 0,
                "errors": 0,
                "agreement_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_visual_quality": 0.0,
                "avg_market_context": 0.0,
                "by_pattern": {},
                "by_symbol": {}
            }

        total = len(self._validation_history)
        agreements = sum(1 for h in self._validation_history if h.get("claude_agrees"))
        rejections = sum(1 for h in self._validation_history
                        if h.get("status") == ValidationStatus.REJECTED.value)
        errors = sum(1 for h in self._validation_history
                    if h.get("status") == ValidationStatus.ERROR.value)

        confidences = [h.get("claude_confidence", 0) for h in self._validation_history
                      if h.get("claude_confidence")]
        visual_scores = [h.get("visual_quality_score", 0) for h in self._validation_history
                        if h.get("visual_quality_score")]
        context_scores = [h.get("market_context_score", 0) for h in self._validation_history
                         if h.get("market_context_score")]

        # By pattern type
        by_pattern = {}
        for h in self._validation_history:
            pt = h.get("pattern_type", "unknown")
            if pt not in by_pattern:
                by_pattern[pt] = {"total": 0, "agreed": 0}
            by_pattern[pt]["total"] += 1
            if h.get("claude_agrees"):
                by_pattern[pt]["agreed"] += 1

        # By symbol
        by_symbol = {}
        for h in self._validation_history:
            sym = h.get("symbol", "unknown")
            if sym not in by_symbol:
                by_symbol[sym] = {"total": 0, "agreed": 0}
            by_symbol[sym]["total"] += 1
            if h.get("claude_agrees"):
                by_symbol[sym]["agreed"] += 1

        return {
            "total_validations": total,
            "agreements": agreements,
            "rejections": rejections,
            "errors": errors,
            "agreement_rate": round(agreements / total * 100, 1) if total > 0 else 0.0,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "avg_visual_quality": round(sum(visual_scores) / len(visual_scores), 3) if visual_scores else 0.0,
            "avg_market_context": round(sum(context_scores) / len(context_scores), 3) if context_scores else 0.0,
            "by_pattern": by_pattern,
            "by_symbol": by_symbol
        }

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "enabled": bool(self.api_key),
            "model": self.model,
            "matplotlib_available": MATPLOTLIB_AVAILABLE,
            "cache_size": len(self._validation_cache),
            "history_size": len(self._validation_history),
            "api_configured": bool(self.api_key)
        }


# Global singleton
claude_validator_service = ClaudeValidatorService()
