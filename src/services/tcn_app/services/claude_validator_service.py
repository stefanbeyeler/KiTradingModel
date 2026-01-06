"""
Claude Vision Validator Service for TCN Chart Pattern QA.

This service generates chart images of detected TCN patterns and sends them
to Claude AI for visual validation. This provides an external quality
assurance mechanism for chart pattern detection (Head & Shoulders, Triangles, etc.).

Architecture:
    1. TCN detection -> Pattern detected
    2. Chart rendering -> Visual representation with pattern overlay
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
from typing import Optional, Dict, List, Any
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
    from matplotlib.patches import Rectangle, Polygon
    import numpy as np
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
class TCNClaudeValidationResult:
    """Result of Claude AI validation for a TCN pattern."""
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


class TCNChartRenderer:
    """
    Renders candlestick charts for TCN pattern visualization.
    Includes pattern overlay lines for Head & Shoulders, Triangles, etc.
    """

    def __init__(self):
        self.figure_size = (14, 7)
        self.dpi = 100
        self.colors = {
            "bullish": "#4caf50",
            "bearish": "#f44336",
            "neutral": "#ff9800",
            "background": "#1a1a2e",
            "grid": "#333333",
            "text": "#ffffff",
            "pattern_line": "#00bcd4",  # Cyan for pattern lines
            "neckline": "#ffeb3b",  # Yellow for necklines
            "support": "#4caf50",  # Green for support
            "resistance": "#f44336",  # Red for resistance
            "highlight_zone": (0, 188, 212, 0.2),  # Cyan with alpha
        }

    def render_tcn_pattern_chart(
        self,
        ohlcv_data: List[Dict],
        pattern_type: str,
        pattern_points: Optional[List[Dict]] = None,
        direction: str = "neutral",
        pattern_start_time: Optional[str] = None,
        pattern_end_time: Optional[str] = None
    ) -> Optional[str]:
        """
        Render a candlestick chart with TCN pattern overlay.

        Args:
            ohlcv_data: List of OHLCV dictionaries
            pattern_type: Name of the pattern (e.g., head_and_shoulders)
            pattern_points: Key points for pattern visualization
            direction: Pattern direction (bullish, bearish, neutral)
            pattern_start_time: When pattern started
            pattern_end_time: When pattern completed

        Returns:
            Base64 encoded PNG image, or None if rendering fails
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot render chart - matplotlib not available")
            return None

        try:
            if not ohlcv_data or len(ohlcv_data) < 10:
                logger.warning("Not enough OHLCV data for chart rendering")
                return None

            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["background"])

            # Find pattern zone indices
            pattern_start_idx = 0
            pattern_end_idx = len(ohlcv_data) - 1

            if pattern_start_time and pattern_end_time:
                start_clean = pattern_start_time.replace("T", " ").split("+")[0].split(".")[0]
                end_clean = pattern_end_time.replace("T", " ").split("+")[0].split(".")[0]

                for i, candle in enumerate(ohlcv_data):
                    ts = candle.get("timestamp", candle.get("datetime", ""))
                    ts_clean = ts.replace("T", " ").split("+")[0].split(".")[0]
                    if ts_clean == start_clean:
                        pattern_start_idx = i
                    if ts_clean == end_clean:
                        pattern_end_idx = i

            # Plot candlesticks
            all_highs = []
            all_lows = []

            for i, candle in enumerate(ohlcv_data):
                o = float(candle.get("open", 0))
                h = float(candle.get("high", 0))
                l = float(candle.get("low", 0))
                c = float(candle.get("close", 0))

                if h == 0 or l == 0:
                    continue

                all_highs.append(h)
                all_lows.append(l)

                is_bullish = c >= o
                is_pattern_candle = pattern_start_idx <= i <= pattern_end_idx

                # Select color
                if is_bullish:
                    color = self.colors["bullish"]
                else:
                    color = self.colors["bearish"]

                # Highlight pattern zone
                if is_pattern_candle:
                    ax.axvspan(
                        i - 0.4, i + 0.4,
                        color=self.colors["highlight_zone"][:3],
                        alpha=self.colors["highlight_zone"][3]
                    )

                # Draw wick
                line_width = 2 if is_pattern_candle else 1
                ax.plot([i, i], [l, h], color=color, linewidth=line_width)

                # Draw body
                body_bottom = min(o, c)
                body_height = abs(c - o)
                if body_height < 0.0001:
                    body_height = (h - l) * 0.01

                if is_bullish:
                    rect = Rectangle(
                        (i - 0.3, body_bottom), 0.6, body_height,
                        facecolor=self.colors["background"],
                        edgecolor=color,
                        linewidth=2 if is_pattern_candle else 1
                    )
                else:
                    rect = Rectangle(
                        (i - 0.3, body_bottom), 0.6, body_height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=2 if is_pattern_candle else 1
                    )
                ax.add_patch(rect)

            # Draw pattern lines if pattern_points provided
            if pattern_points and len(pattern_points) >= 2:
                self._draw_pattern_overlay(ax, ohlcv_data, pattern_type, pattern_points)

            # Style the chart
            if all_highs and all_lows:
                y_min = min(all_lows)
                y_max = max(all_highs)
                y_padding = (y_max - y_min) * 0.15
                ax.set_ylim(y_min - y_padding, y_max + y_padding)

            ax.set_xlim(-1, len(ohlcv_data))
            ax.grid(True, alpha=0.3, color=self.colors["grid"])
            ax.tick_params(colors=self.colors["text"])

            # Title with direction color
            title_color = self.colors.get(direction, self.colors["neutral"])
            pattern_name = pattern_type.replace("_", " ").title()
            ax.set_title(
                f"TCN Pattern: {pattern_name}",
                color=title_color,
                fontsize=14,
                fontweight="bold"
            )

            # Add legend for pattern elements
            ax.text(
                0.02, 0.98,
                f"Direction: {direction.upper()}",
                transform=ax.transAxes,
                color=title_color,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=self.colors["background"], alpha=0.8)
            )

            for spine in ax.spines.values():
                spine.set_color(self.colors["grid"])

            plt.tight_layout()

            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, facecolor=self.colors["background"])
            buf.seek(0)

            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return image_base64

        except Exception as e:
            logger.error(f"Failed to render TCN chart: {e}")
            return None

    def _draw_pattern_overlay(
        self,
        ax,
        ohlcv_data: List[Dict],
        pattern_type: str,
        pattern_points: List[Dict]
    ):
        """Draw pattern-specific overlay lines."""
        try:
            # Map pattern point indices to chart x-coordinates
            # Pattern points have 'index' relative to original data
            point_coords = []

            for point in pattern_points:
                idx = point.get("index", 0)
                price = point.get("price", 0)
                point_type = point.get("point_type", "")

                # Clamp index to valid range
                if idx < 0:
                    idx = 0
                if idx >= len(ohlcv_data):
                    idx = len(ohlcv_data) - 1

                point_coords.append({
                    "x": idx,
                    "y": price,
                    "type": point_type
                })

            if not point_coords:
                return

            # Draw based on pattern type
            pattern_lower = pattern_type.lower()

            if "head_and_shoulders" in pattern_lower or "inverse_head_and_shoulders" in pattern_lower:
                self._draw_head_and_shoulders(ax, point_coords)
            elif "double_top" in pattern_lower or "double_bottom" in pattern_lower:
                self._draw_double_pattern(ax, point_coords)
            elif "triple_top" in pattern_lower or "triple_bottom" in pattern_lower:
                self._draw_triple_pattern(ax, point_coords)
            elif "triangle" in pattern_lower:
                self._draw_triangle(ax, point_coords)
            elif "wedge" in pattern_lower:
                self._draw_wedge(ax, point_coords)
            elif "channel" in pattern_lower:
                self._draw_channel(ax, point_coords)
            elif "flag" in pattern_lower:
                self._draw_flag(ax, point_coords)
            else:
                # Generic: connect all points
                self._draw_generic_pattern(ax, point_coords)

        except Exception as e:
            logger.warning(f"Failed to draw pattern overlay: {e}")

    def _draw_head_and_shoulders(self, ax, points: List[Dict]):
        """Draw Head and Shoulders pattern lines."""
        # Find key points
        left_shoulder = next((p for p in points if "left_shoulder" in p["type"]), None)
        head = next((p for p in points if p["type"] == "head"), None)
        right_shoulder = next((p for p in points if "right_shoulder" in p["type"]), None)
        neckline_left = next((p for p in points if "neckline_left" in p["type"]), None)
        neckline_right = next((p for p in points if "neckline_right" in p["type"]), None)

        # Draw shoulder-head-shoulder line
        shoulder_points = [p for p in [left_shoulder, head, right_shoulder] if p]
        if len(shoulder_points) >= 2:
            xs = [p["x"] for p in shoulder_points]
            ys = [p["y"] for p in shoulder_points]
            ax.plot(xs, ys, color=self.colors["pattern_line"], linewidth=2, linestyle='-', marker='o', markersize=8)

        # Draw neckline
        neckline_points = [p for p in [neckline_left, neckline_right] if p]
        if len(neckline_points) >= 2:
            xs = [p["x"] for p in neckline_points]
            ys = [p["y"] for p in neckline_points]
            # Extend neckline
            ax.plot(xs, ys, color=self.colors["neckline"], linewidth=2, linestyle='--')

        # Add labels
        if head:
            ax.annotate("Head", (head["x"], head["y"]), textcoords="offset points",
                       xytext=(0, 10), ha='center', color=self.colors["text"], fontsize=9)
        if left_shoulder:
            ax.annotate("L.S.", (left_shoulder["x"], left_shoulder["y"]), textcoords="offset points",
                       xytext=(0, 10), ha='center', color=self.colors["text"], fontsize=9)
        if right_shoulder:
            ax.annotate("R.S.", (right_shoulder["x"], right_shoulder["y"]), textcoords="offset points",
                       xytext=(0, 10), ha='center', color=self.colors["text"], fontsize=9)

    def _draw_double_pattern(self, ax, points: List[Dict]):
        """Draw Double Top/Bottom pattern."""
        tops = [p for p in points if "top" in p["type"] or "peak" in p["type"]]
        bottoms = [p for p in points if "bottom" in p["type"] or "trough" in p["type"]]

        key_points = tops if len(tops) >= 2 else bottoms

        if len(key_points) >= 2:
            xs = [p["x"] for p in key_points[:2]]
            ys = [p["y"] for p in key_points[:2]]
            # Draw horizontal resistance/support line
            avg_y = sum(ys) / len(ys)
            ax.axhline(y=avg_y, color=self.colors["pattern_line"], linewidth=2, linestyle='--', alpha=0.8)
            ax.plot(xs, ys, color=self.colors["pattern_line"], linewidth=2, marker='o', markersize=8, linestyle='')

    def _draw_triple_pattern(self, ax, points: List[Dict]):
        """Draw Triple Top/Bottom pattern."""
        tops = [p for p in points if "top" in p["type"] or "peak" in p["type"]]
        bottoms = [p for p in points if "bottom" in p["type"] or "trough" in p["type"]]

        key_points = tops if len(tops) >= 3 else bottoms

        if len(key_points) >= 3:
            xs = [p["x"] for p in key_points[:3]]
            ys = [p["y"] for p in key_points[:3]]
            avg_y = sum(ys) / len(ys)
            ax.axhline(y=avg_y, color=self.colors["pattern_line"], linewidth=2, linestyle='--', alpha=0.8)
            ax.plot(xs, ys, color=self.colors["pattern_line"], linewidth=2, marker='o', markersize=8, linestyle='')

    def _draw_triangle(self, ax, points: List[Dict]):
        """Draw Triangle pattern (ascending, descending, symmetrical)."""
        upper_points = [p for p in points if "upper" in p["type"] or "resistance" in p["type"] or "high" in p["type"]]
        lower_points = [p for p in points if "lower" in p["type"] or "support" in p["type"] or "low" in p["type"]]

        # Draw upper trendline
        if len(upper_points) >= 2:
            xs = [p["x"] for p in upper_points]
            ys = [p["y"] for p in upper_points]
            ax.plot(xs, ys, color=self.colors["resistance"], linewidth=2, linestyle='-')

        # Draw lower trendline
        if len(lower_points) >= 2:
            xs = [p["x"] for p in lower_points]
            ys = [p["y"] for p in lower_points]
            ax.plot(xs, ys, color=self.colors["support"], linewidth=2, linestyle='-')

    def _draw_wedge(self, ax, points: List[Dict]):
        """Draw Wedge pattern."""
        self._draw_triangle(ax, points)  # Same visualization as triangle

    def _draw_channel(self, ax, points: List[Dict]):
        """Draw Channel pattern."""
        self._draw_triangle(ax, points)  # Same visualization

    def _draw_flag(self, ax, points: List[Dict]):
        """Draw Flag pattern."""
        self._draw_triangle(ax, points)

    def _draw_generic_pattern(self, ax, points: List[Dict]):
        """Draw generic pattern by connecting all points."""
        if len(points) >= 2:
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            ax.plot(xs, ys, color=self.colors["pattern_line"], linewidth=2, marker='o', markersize=6)


class TCNClaudeValidatorService:
    """
    Service for validating TCN chart patterns using Claude Vision API.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = os.getenv("CLAUDE_VALIDATOR_MODEL", "claude-sonnet-4-20250514")
        self.chart_renderer = TCNChartRenderer()

        # Validation history
        self._validation_history: List[Dict] = []
        self._history_file = Path(os.getenv("DATA_DIR", "/app/data")) / "tcn_claude_validations.json"
        self._load_history()

        # Rate limiting
        self._last_request_time = None
        self._min_request_interval = 1.0

        # Cache
        self._validation_cache: Dict[str, TCNClaudeValidationResult] = {}
        self._cache_max_size = 100

        logger.info(f"TCN Claude Validator initialized with model: {self.model}")

    def _load_history(self):
        """Load validation history from file."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._validation_history = data[-1000:]
                logger.info(f"Loaded {len(self._validation_history)} TCN validation history entries")
        except Exception as e:
            logger.warning(f"Failed to load TCN validation history: {e}")
            self._validation_history = []

    def _save_history(self):
        """Save validation history to file."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump(self._validation_history[-1000:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save TCN validation history: {e}")

    def _get_pattern_criteria(self, pattern_type: str) -> str:
        """Get specific criteria for each TCN pattern type."""
        criteria = {
            "head_and_shoulders": """
**Head and Shoulders Kriterien:**
- Drei deutliche Hochpunkte: linke Schulter, Kopf (höchster), rechte Schulter
- Kopf muss deutlich höher sein als beide Schultern
- Schultern sollten ungefähr auf gleicher Höhe sein (±5%)
- Nackenlinie verbindet die Tiefs zwischen den Hochpunkten
- Volumen typischerweise höher am Kopf als an den Schultern
- Bearish reversal pattern nach Aufwärtstrend""",

            "inverse_head_and_shoulders": """
**Inverse Head and Shoulders Kriterien:**
- Drei deutliche Tiefpunkte: linke Schulter, Kopf (tiefster), rechte Schulter
- Kopf muss deutlich tiefer sein als beide Schultern
- Schultern sollten ungefähr auf gleicher Höhe sein (±5%)
- Nackenlinie verbindet die Hochs zwischen den Tiefpunkten
- Bullish reversal pattern nach Abwärtstrend""",

            "double_top": """
**Double Top Kriterien:**
- Zwei deutliche Hochpunkte auf ungefähr gleichem Niveau (±2%)
- Dazwischen ein deutliches Tal (Pullback von mindestens 10%)
- Bearish reversal pattern nach Aufwärtstrend
- Bestätigung durch Bruch der Unterstützung (Tal-Niveau)""",

            "double_bottom": """
**Double Bottom Kriterien:**
- Zwei deutliche Tiefpunkte auf ungefähr gleichem Niveau (±2%)
- Dazwischen ein deutlicher Peak (Rally von mindestens 10%)
- Bullish reversal pattern nach Abwärtstrend
- Bestätigung durch Bruch des Widerstands (Peak-Niveau)""",

            "triple_top": """
**Triple Top Kriterien:**
- Drei deutliche Hochpunkte auf ungefähr gleichem Niveau
- Dazwischen zwei Täler
- Bearish reversal pattern
- Stärker als Double Top""",

            "triple_bottom": """
**Triple Bottom Kriterien:**
- Drei deutliche Tiefpunkte auf ungefähr gleichem Niveau
- Dazwischen zwei Peaks
- Bullish reversal pattern
- Stärker als Double Bottom""",

            "ascending_triangle": """
**Ascending Triangle Kriterien:**
- Horizontale Widerstandslinie (mindestens 2 Berührungen)
- Steigende Unterstützungslinie (höhere Tiefs)
- Typischerweise bullish continuation pattern
- Volumen nimmt meist ab während Formation""",

            "descending_triangle": """
**Descending Triangle Kriterien:**
- Horizontale Unterstützungslinie (mindestens 2 Berührungen)
- Fallende Widerstandslinie (tiefere Hochs)
- Typischerweise bearish continuation pattern""",

            "symmetrical_triangle": """
**Symmetrical Triangle Kriterien:**
- Konvergierende Trendlinien (tiefere Hochs UND höhere Tiefs)
- Mindestens 4 Berührungspunkte (2 pro Linie)
- Neutral - Ausbruchsrichtung bestimmt Trend
- Volumen nimmt während Formation ab""",

            "rising_wedge": """
**Rising Wedge Kriterien:**
- Beide Trendlinien steigen, aber konvergieren
- Obere Linie steigt langsamer als untere
- Bearish pattern (trotz steigender Preise)
- Oft Reversal nach Aufwärtstrend""",

            "falling_wedge": """
**Falling Wedge Kriterien:**
- Beide Trendlinien fallen, aber konvergieren
- Untere Linie fällt langsamer als obere
- Bullish pattern (trotz fallender Preise)
- Oft Reversal nach Abwärtstrend""",

            "bull_flag": """
**Bull Flag Kriterien:**
- Starker vorheriger Aufwärtstrend (Flaggenstange)
- Konsolidierung in einem abwärts geneigten Kanal
- Kurze Dauer (typisch 1-3 Wochen)
- Bullish continuation pattern""",

            "bear_flag": """
**Bear Flag Kriterien:**
- Starker vorheriger Abwärtstrend (Flaggenstange)
- Konsolidierung in einem aufwärts geneigten Kanal
- Kurze Dauer (typisch 1-3 Wochen)
- Bearish continuation pattern""",

            "channel_up": """
**Channel Up (Aufwärtskanal) Kriterien:**
- Parallele Trendlinien mit Aufwärtsneigung
- Mindestens 2 Berührungen pro Linie
- Preise bewegen sich zwischen den Linien""",

            "channel_down": """
**Channel Down (Abwärtskanal) Kriterien:**
- Parallele Trendlinien mit Abwärtsneigung
- Mindestens 2 Berührungen pro Linie
- Preise bewegen sich zwischen den Linien""",
        }
        return criteria.get(pattern_type.lower(), "")

    def _build_validation_prompt(
        self,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        direction: str,
        confidence: float
    ) -> str:
        """Build the prompt for Claude to validate a TCN pattern."""
        specific_criteria = self._get_pattern_criteria(pattern_type)
        pattern_name = pattern_type.replace("_", " ").title()

        return f"""Du bist ein Experte für technische Analyse, spezialisiert auf Chart-Pattern-Erkennung.

**Symbol:** {symbol}
**Timeframe:** {timeframe}
**Erkanntes Pattern:** {pattern_name}
**Erkannte Richtung:** {direction}
**Erkennungs-Konfidenz:** {confidence:.1%}

{specific_criteria}

**DEINE AUFGABE:**
Analysiere den Chart und bewerte, ob das markierte Pattern korrekt erkannt wurde.

Das Pattern-Bereich ist mit einem **hellblauen/türkisen Hintergrund** hervorgehoben.
Die Pattern-Linien (falls vorhanden) zeigen die wichtigen Punkte:
- **Cyan-Linien**: Verbinden die Hauptpunkte des Patterns
- **Gelbe gestrichelte Linie**: Nackenlinie (bei H&S)
- **Grüne Linie**: Unterstützung
- **Rote Linie**: Widerstand

**PRÜFE:**
1. Ist das Pattern visuell erkennbar?
2. Entspricht es den klassischen Kriterien?
3. Ist die Richtung (bullish/bearish) korrekt?
4. Ist der Marktkontext passend (vorheriger Trend)?

Antworte in diesem exakten JSON-Format:
```json
{{
    "agrees": true/false,
    "confidence": 0.0-1.0,
    "detected_pattern": "pattern_name_oder_null_wenn_anderes",
    "visual_quality": 0.0-1.0,
    "market_context": 0.0-1.0,
    "reasoning": "Kurze Begründung auf DEUTSCH"
}}
```

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
        direction: str,
        confidence: float,
        ohlcv_data: List[Dict],
        pattern_points: Optional[List[Dict]] = None,
        pattern_start_time: Optional[str] = None,
        pattern_end_time: Optional[str] = None,
        force: bool = False
    ) -> TCNClaudeValidationResult:
        """
        Validate a detected TCN pattern using Claude Vision API.

        Args:
            pattern_id: Unique identifier for the pattern
            pattern_type: Type of pattern detected
            symbol: Trading symbol
            timeframe: Chart timeframe
            direction: Pattern direction (bullish/bearish)
            confidence: Detection confidence
            ohlcv_data: OHLCV data for chart rendering
            pattern_points: Key points for pattern visualization
            pattern_start_time: When pattern started
            pattern_end_time: When pattern completed
            force: Force validation even if cached

        Returns:
            TCNClaudeValidationResult with Claude's assessment
        """
        # Check cache
        if not force and pattern_id in self._validation_cache:
            logger.debug(f"Returning cached TCN validation for {pattern_id}")
            return self._validation_cache[pattern_id]

        # Check if API key is configured
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not configured - skipping Claude validation")
            return TCNClaudeValidationResult(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                claude_agrees=True,
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
            # Render chart
            chart_base64 = self.chart_renderer.render_tcn_pattern_chart(
                ohlcv_data=ohlcv_data,
                pattern_type=pattern_type,
                pattern_points=pattern_points,
                direction=direction,
                pattern_start_time=pattern_start_time,
                pattern_end_time=pattern_end_time
            )

            if not chart_base64:
                raise ValueError("Failed to render chart image")

            # Build prompt
            prompt = self._build_validation_prompt(
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                confidence=confidence
            )

            # Rate limit
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

            # Parse response
            validation_result = self._parse_claude_response(
                content=content,
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                chart_base64=chart_base64
            )

            # Cache and save
            self._validation_cache[pattern_id] = validation_result
            if len(self._validation_cache) > self._cache_max_size:
                oldest_keys = list(self._validation_cache.keys())[:-self._cache_max_size]
                for key in oldest_keys:
                    del self._validation_cache[key]

            self._validation_history.append(validation_result.to_dict())
            self._save_history()

            logger.info(
                f"TCN Claude validation for {pattern_type} ({symbol}/{timeframe}): "
                f"agrees={validation_result.claude_agrees}, "
                f"confidence={validation_result.claude_confidence:.2f}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"TCN Claude validation failed: {e}")
            return TCNClaudeValidationResult(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                claude_agrees=True,
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
    ) -> TCNClaudeValidationResult:
        """Parse Claude's JSON response."""
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            return TCNClaudeValidationResult(
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
            return TCNClaudeValidationResult(
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
                "by_pattern": {},
                "by_symbol": {}
            }

        total = len(self._validation_history)
        agreements = sum(1 for h in self._validation_history if h.get("claude_agrees"))
        rejections = sum(1 for h in self._validation_history if h.get("status") == "rejected")
        errors = sum(1 for h in self._validation_history if h.get("status") == "error")

        by_pattern = {}
        for h in self._validation_history:
            pt = h.get("pattern_type", "unknown")
            if pt not in by_pattern:
                by_pattern[pt] = {"total": 0, "agreed": 0}
            by_pattern[pt]["total"] += 1
            if h.get("claude_agrees"):
                by_pattern[pt]["agreed"] += 1

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

    def clear_history(self):
        """Clear validation history."""
        count = len(self._validation_history)
        self._validation_history = []
        self._validation_cache = {}
        self._save_history()
        logger.info(f"Cleared {count} TCN validation history entries")
        return {"cleared": count}


# Global singleton
tcn_claude_validator_service = TCNClaudeValidatorService()
