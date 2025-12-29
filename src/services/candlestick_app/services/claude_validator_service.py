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
    """Renders candlestick charts for pattern visualization."""

    def __init__(self):
        self.figure_size = (12, 6)
        self.dpi = 100
        self.colors = {
            "bullish": "#26A69A",  # Green
            "bearish": "#EF5350",  # Red
            "highlight": "#FFD700",  # Gold for pattern highlight
            "background": "#1E1E1E",  # Dark background
            "grid": "#333333",
            "text": "#FFFFFF"
        }

    def render_pattern_chart(
        self,
        ohlcv_data: List[Dict],
        pattern_type: str,
        pattern_candles: int = 3,
        context_candles: int = 10
    ) -> Optional[str]:
        """
        Render a candlestick chart highlighting the pattern.

        Args:
            ohlcv_data: List of OHLCV dictionaries
            pattern_type: Name of the pattern
            pattern_candles: Number of candles in the pattern
            context_candles: Number of context candles before pattern

        Returns:
            Base64 encoded PNG image, or None if rendering fails
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot render chart - matplotlib not available")
            return None

        try:
            # Calculate how many candles to show
            total_candles = pattern_candles + context_candles
            data = ohlcv_data[-total_candles:] if len(ohlcv_data) >= total_candles else ohlcv_data

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
                color = self.colors["bullish"] if is_bullish else self.colors["bearish"]

                # Highlight pattern candles
                is_pattern_candle = i >= len(data) - pattern_candles
                if is_pattern_candle:
                    # Add highlight background
                    ax.axvspan(
                        i - 0.4, i + 0.4,
                        alpha=0.2,
                        color=self.colors["highlight"]
                    )

                # Draw wick (high-low line)
                ax.plot([i, i], [l, h], color=color, linewidth=1)

                # Draw body
                body_bottom = min(o, c)
                body_height = abs(c - o)
                if body_height < 0.0001:  # Doji
                    body_height = (h - l) * 0.01

                rect = Rectangle(
                    (i - 0.3, body_bottom),
                    0.6,
                    body_height,
                    facecolor=color if is_bullish else color,
                    edgecolor=color,
                    linewidth=1
                )
                ax.add_patch(rect)

            # Style the chart
            ax.set_xlim(-0.5, len(data) - 0.5)

            # Calculate y-axis limits with padding
            all_highs = [float(c.get("high", c.get("h", 0))) for c in data]
            all_lows = [float(c.get("low", c.get("l", 0))) for c in data]
            y_min = min(all_lows)
            y_max = max(all_highs)
            y_padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Grid
            ax.grid(True, alpha=0.3, color=self.colors["grid"])
            ax.tick_params(colors=self.colors["text"])

            # Title
            ax.set_title(
                f"Pattern: {pattern_type.replace('_', ' ').title()}",
                color=self.colors["text"],
                fontsize=14,
                fontweight="bold"
            )

            # Add annotation for pattern area
            ax.annotate(
                "Pattern",
                xy=(len(data) - pattern_candles/2 - 0.5, y_max),
                xytext=(len(data) - pattern_candles/2 - 0.5, y_max + y_padding * 0.5),
                color=self.colors["highlight"],
                fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=self.colors["highlight"])
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

    def _build_validation_prompt(self, pattern_type: str, symbol: str, timeframe: str) -> str:
        """Build the prompt for Claude to validate a pattern."""
        return f"""You are an expert technical analyst specializing in candlestick pattern recognition.

Analyze this candlestick chart image and evaluate whether the highlighted pattern is a valid **{pattern_type.replace('_', ' ').title()}** pattern.

**Symbol:** {symbol}
**Timeframe:** {timeframe}
**Claimed Pattern:** {pattern_type.replace('_', ' ').title()}

Please evaluate:

1. **Pattern Validity**: Does this truly match the characteristics of a {pattern_type.replace('_', ' ').title()}?
   - Check candle body sizes, shadow lengths, and relative positions
   - Consider the classic definition of this pattern

2. **Visual Quality Score (0.0-1.0)**: How well-formed is this pattern?
   - 1.0 = Textbook perfect example
   - 0.5 = Acceptable but not ideal
   - 0.0 = Does not meet basic criteria

3. **Market Context Score (0.0-1.0)**: Is the pattern appearing in appropriate context?
   - Consider if reversal patterns appear after trends
   - Consider if continuation patterns appear within trends

4. **Your Assessment**: What pattern do YOU see? (if different from claimed)

Respond in this exact JSON format:
```json
{{
    "agrees": true/false,
    "confidence": 0.0-1.0,
    "detected_pattern": "pattern_name_or_null",
    "visual_quality": 0.0-1.0,
    "market_context": 0.0-1.0,
    "reasoning": "Brief explanation of your assessment"
}}
```

Be strict but fair in your assessment. Only confirm patterns that clearly meet the definition."""

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
            # Render chart
            pattern_candles = self._get_pattern_candle_count(pattern_type)
            chart_base64 = self.chart_renderer.render_pattern_chart(
                ohlcv_data=ohlcv_data,
                pattern_type=pattern_type,
                pattern_candles=pattern_candles,
                context_candles=15
            )

            if not chart_base64:
                raise ValueError("Failed to render chart image")

            # Build prompt
            prompt = self._build_validation_prompt(pattern_type, symbol, timeframe)

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
