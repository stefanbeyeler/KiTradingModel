"""
Unified Chart Color Configuration for Candlestick Pattern Visualization.

This configuration ensures consistent visual representation across:
- Frontend (JavaScript Canvas rendering)
- Backend (Matplotlib rendering for Claude Vision Validator)

Both manual validation and AI-based validation use the same color scheme
to ensure fair and consistent pattern assessment.
"""

from typing import Dict, Any


# === UNIFIED COLOR PALETTE ===
# These colors are used by both Frontend and Backend chart renderers

CHART_COLORS: Dict[str, str] = {
    # Candle Colors
    "bullish": "#4caf50",           # Green - bullish candles
    "bullish_highlight": "#66bb6a", # Lighter green for highlighted bullish
    "bearish": "#f44336",           # Red - bearish candles
    "bearish_highlight": "#ef5350", # Lighter red for highlighted bearish
    "neutral": "#ff9800",           # Orange - neutral/indecision patterns

    # Background & Grid
    "background": "#1a1a2e",        # Dark background
    "grid": "rgba(255, 255, 255, 0.05)",  # Subtle grid lines

    # Highlight Colors (pattern area background)
    "highlight_bullish": "rgba(76, 175, 80, 0.2)",   # Green tint
    "highlight_bearish": "rgba(244, 67, 54, 0.2)",   # Red tint
    "highlight_neutral": "rgba(255, 152, 0, 0.2)",   # Orange tint

    # Text & Labels
    "text_primary": "#ffffff",      # Primary text (white)
    "text_secondary": "#888888",    # Secondary text (gray)
    "text_highlight": "#ffffff",    # Highlighted text

    # Pattern Label Colors (same as direction colors)
    "label_bullish": "#4caf50",
    "label_bearish": "#f44336",
    "label_neutral": "#ff9800",
}


# === MATPLOTLIB-SPECIFIC COLORS ===
# Converted for Matplotlib (no rgba support in some contexts)

MATPLOTLIB_COLORS: Dict[str, str] = {
    "bullish": "#4caf50",
    "bullish_highlight": "#66bb6a",
    "bearish": "#f44336",
    "bearish_highlight": "#ef5350",
    "neutral": "#ff9800",
    "background": "#1a1a2e",
    "grid": "#333333",              # Solid color for matplotlib
    "highlight": "#FFD700",         # Gold for pattern marker arrow
    "text": "#ffffff",
}

# Highlight alpha for matplotlib (separate value)
MATPLOTLIB_HIGHLIGHT_ALPHA = 0.2


def get_direction_color(direction: str, highlighted: bool = False) -> str:
    """Get the appropriate color for a candle direction."""
    if direction == "bullish":
        return CHART_COLORS["bullish_highlight"] if highlighted else CHART_COLORS["bullish"]
    elif direction == "bearish":
        return CHART_COLORS["bearish_highlight"] if highlighted else CHART_COLORS["bearish"]
    else:
        return CHART_COLORS["neutral"]


def get_highlight_background(direction: str) -> str:
    """Get the highlight background color for a pattern direction."""
    if direction == "bullish":
        return CHART_COLORS["highlight_bullish"]
    elif direction == "bearish":
        return CHART_COLORS["highlight_bearish"]
    else:
        return CHART_COLORS["highlight_neutral"]


def get_label_color(direction: str) -> str:
    """Get the label color for a pattern direction."""
    if direction == "bullish":
        return CHART_COLORS["label_bullish"]
    elif direction == "bearish":
        return CHART_COLORS["label_bearish"]
    else:
        return CHART_COLORS["label_neutral"]


def to_javascript_config() -> Dict[str, Any]:
    """
    Export color configuration for JavaScript frontend.

    This can be used to generate a JavaScript config file or
    be served via an API endpoint.
    """
    return {
        "colors": CHART_COLORS,
        "version": "1.0.0",
        "description": "Unified chart colors for candlestick pattern visualization"
    }


# === COLOR DOCUMENTATION ===
"""
Color Usage Guide:

1. CANDLE BODY & WICK:
   - Bullish (close >= open): Use "bullish" color
   - Bearish (close < open): Use "bearish" color
   - When candle is part of highlighted pattern: Use "_highlight" variant

2. PATTERN HIGHLIGHT BACKGROUND:
   - Use "highlight_bullish/bearish/neutral" based on pattern direction
   - This creates a semi-transparent overlay behind pattern candles

3. PATTERN LABELS:
   - Use "label_bullish/bearish/neutral" for the pattern name badge
   - Text inside badge should be "text_primary" (white)

4. CHART BACKGROUND:
   - Use "background" for the main chart area
   - Use "grid" for subtle horizontal grid lines

5. TIMESTAMPS & SECONDARY INFO:
   - Normal: "text_secondary"
   - Highlighted candle: "text_highlight"
"""
