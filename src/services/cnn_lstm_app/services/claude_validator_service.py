"""
Claude Vision Validator Service fuer CNN-LSTM Predictions.

Validiert Predictions visuell mittels Claude Vision API:
- Preis-Trend-Analyse (Chart-basiert)
- Pattern-Erkennung (visuell)
- Regime-Klassifikation (visuell)

Analog zum claude_validator_service.py des Candlestick Service,
aber angepasst fuer Multi-Task Predictions.
"""

import asyncio
import base64
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Status der Claude-Validierung."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ClaudePredictionValidation:
    """Ergebnis der Claude-Validierung fuer eine CNN-LSTM Prediction."""
    prediction_id: str
    symbol: str
    timeframe: str
    validation_timestamp: str

    # Price Validation
    price_direction_agrees: bool
    price_confidence: float
    price_reasoning: str

    # Pattern Validation
    patterns_confirmed: List[str]
    patterns_rejected: List[str]
    patterns_suggested: List[str]
    pattern_confidence: float
    pattern_reasoning: str

    # Regime Validation
    regime_agrees: bool
    regime_suggested: Optional[str]
    regime_confidence: float
    regime_reasoning: str

    # Overall
    overall_agrees: bool
    overall_confidence: float
    overall_reasoning: str

    # Metadata
    model_used: str
    status: ValidationStatus
    error_message: Optional[str] = None
    chart_image_base64: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class PredictionChartRenderer:
    """
    Rendert Charts fuer CNN-LSTM Prediction Validierung.

    Erstellt Candlestick-Charts mit Annotations fuer:
    - Vorhergesagte Richtung
    - Erkannte Patterns
    - Aktuelles Regime
    """

    def __init__(self):
        self.figure_size = (14, 8)
        self.dpi = 100
        self.colors = {
            "bullish": "#4caf50",
            "bearish": "#f44336",
            "neutral": "#ff9800",
            "background": "#1a1a2e",
            "grid": "#333333",
            "text": "#ffffff",
            "forecast": "#00bcd4",
            "regime_bull": "#4caf50",
            "regime_bear": "#f44336",
            "regime_sideways": "#ff9800",
            "regime_highvol": "#9c27b0",
        }

    def render_prediction_chart(
        self,
        ohlcv_data: List[Dict],
        prediction: Dict,
        show_forecast: bool = True,
        context_candles: int = 30
    ) -> Optional[str]:
        """
        Render a candlestick chart with prediction annotations.

        Args:
            ohlcv_data: List of OHLCV dictionaries
            prediction: Prediction data from CNN-LSTM
            show_forecast: Whether to show forecast lines
            context_candles: Number of candles to show

        Returns:
            Base64 encoded PNG image, or None if rendering fails
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot render chart - matplotlib not available")
            return None

        try:
            # Use last N candles
            data = ohlcv_data[-context_candles:] if len(ohlcv_data) > context_candles else ohlcv_data

            if not data:
                return None

            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=self.figure_size,
                facecolor=self.colors["background"],
                gridspec_kw={'height_ratios': [3, 1]}
            )

            ax1.set_facecolor(self.colors["background"])
            ax2.set_facecolor(self.colors["background"])

            # Plot candlesticks
            for i, candle in enumerate(data):
                o = float(candle.get("open", candle.get("o", 0)))
                h = float(candle.get("high", candle.get("h", 0)))
                l = float(candle.get("low", candle.get("l", 0)))
                c = float(candle.get("close", candle.get("c", 0)))

                is_bullish = c >= o
                color = self.colors["bullish"] if is_bullish else self.colors["bearish"]

                # Draw wick
                ax1.plot([i, i], [l, h], color=color, linewidth=1)

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
                        linewidth=1
                    )
                else:
                    rect = Rectangle(
                        (i - 0.3, body_bottom), 0.6, body_height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=1
                    )
                ax1.add_patch(rect)

            # Get price data
            price_pred = prediction.get("predictions", {}).get("price", {})
            current_price = price_pred.get("current", float(data[-1].get("close", 0)))
            direction = price_pred.get("direction", "neutral")
            confidence = price_pred.get("confidence", 0.0)

            # Draw forecast if available
            if show_forecast:
                forecast_1h = price_pred.get("forecast_1h")
                forecast_1d = price_pred.get("forecast_1d")

                last_idx = len(data) - 1

                if forecast_1h:
                    ax1.plot(
                        [last_idx, last_idx + 2],
                        [current_price, forecast_1h],
                        color=self.colors["forecast"],
                        linewidth=2,
                        linestyle='--',
                        label=f"1h: {forecast_1h:.2f}"
                    )

                if forecast_1d:
                    ax1.plot(
                        [last_idx, last_idx + 5],
                        [current_price, forecast_1d],
                        color=self.colors["forecast"],
                        linewidth=2,
                        linestyle=':',
                        label=f"1d: {forecast_1d:.2f}"
                    )

            # Direction annotation
            direction_color = self.colors.get(direction, self.colors["neutral"])
            ax1.annotate(
                f"Direction: {direction.upper()}\nConfidence: {confidence:.1%}",
                xy=(len(data) - 1, current_price),
                xytext=(len(data) + 1, current_price),
                fontsize=10,
                color=direction_color,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=self.colors["background"], edgecolor=direction_color),
                arrowprops=dict(arrowstyle='->', color=direction_color)
            )

            # Pattern annotations
            patterns = prediction.get("predictions", {}).get("patterns", [])
            if patterns:
                pattern_text = "Patterns:\n" + "\n".join([
                    f"• {p.get('type', 'unknown')} ({p.get('confidence', 0):.0%})"
                    for p in patterns[:3]
                ])
                ax1.text(
                    0.02, 0.98, pattern_text,
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    color=self.colors["text"],
                    bbox=dict(boxstyle='round', facecolor=self.colors["background"], alpha=0.8)
                )

            # Regime annotation
            regime = prediction.get("predictions", {}).get("regime", {})
            regime_current = regime.get("current", "sideways")
            regime_prob = regime.get("probability", 0.0)
            regime_color = self.colors.get(f"regime_{regime_current.replace('_', '')}", self.colors["neutral"])

            ax1.text(
                0.98, 0.98, f"Regime: {regime_current.upper()}\n({regime_prob:.0%})",
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                color=regime_color,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=self.colors["background"], edgecolor=regime_color)
            )

            # Style main chart
            all_highs = [float(c.get("high", c.get("h", 0))) for c in data]
            all_lows = [float(c.get("low", c.get("l", 0))) for c in data]
            y_min = min(all_lows)
            y_max = max(all_highs)
            y_padding = (y_max - y_min) * 0.15

            ax1.set_xlim(-0.5, len(data) + 6)
            ax1.set_ylim(y_min - y_padding, y_max + y_padding)
            ax1.grid(True, alpha=0.3, color=self.colors["grid"])
            ax1.tick_params(colors=self.colors["text"])
            ax1.set_title(
                f"{prediction.get('symbol', 'Unknown')} - {prediction.get('timeframe', '')}",
                color=self.colors["text"],
                fontsize=14,
                fontweight='bold'
            )

            # Volume subplot
            volumes = [float(c.get("volume", c.get("v", 0))) for c in data]
            colors = [
                self.colors["bullish"] if float(c.get("close", 0)) >= float(c.get("open", 0))
                else self.colors["bearish"]
                for c in data
            ]
            ax2.bar(range(len(data)), volumes, color=colors, alpha=0.7)
            ax2.set_xlim(-0.5, len(data) + 6)
            ax2.grid(True, alpha=0.3, color=self.colors["grid"])
            ax2.tick_params(colors=self.colors["text"])
            ax2.set_ylabel("Volume", color=self.colors["text"])

            # Remove spines
            for ax in [ax1, ax2]:
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
            logger.error(f"Failed to render prediction chart: {e}")
            return None


class ClaudePredictionValidatorService:
    """
    Service fuer die Validierung von CNN-LSTM Predictions mittels Claude Vision API.

    Validiert alle drei Tasks:
    - Price Direction (visuell aus Chart)
    - Patterns (visuell aus Chart)
    - Regime (visuell aus Chart)
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = os.getenv("CLAUDE_VALIDATOR_MODEL", "claude-sonnet-4-20250514")
        self.chart_renderer = PredictionChartRenderer()

        # Validation history
        self._validation_history: List[Dict] = []
        self._history_file = Path(os.getenv("DATA_DIR", "/app/data")) / "cnn_lstm_claude_validations.json"
        self._load_history()

        # Rate limiting
        self._last_request_time = None
        self._min_request_interval = 1.0

        # Cache
        self._validation_cache: Dict[str, ClaudePredictionValidation] = {}
        self._cache_max_size = 100

        logger.info(f"ClaudePredictionValidatorService initialized with model: {self.model}")

    def _load_history(self):
        """Load validation history from file."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    self._validation_history = json.load(f)[-1000:]
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

    async def _rate_limit(self):
        """Apply rate limiting between API requests."""
        if self._last_request_time is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = datetime.now(timezone.utc)

    def _build_validation_prompt(
        self,
        prediction: Dict,
        symbol: str,
        timeframe: str
    ) -> str:
        """Build the prompt for Claude to validate a prediction."""
        price_pred = prediction.get("predictions", {}).get("price", {})
        patterns = prediction.get("predictions", {}).get("patterns", [])
        regime = prediction.get("predictions", {}).get("regime", {})

        patterns_str = ", ".join([
            f"{p.get('type', 'unknown')} ({p.get('confidence', 0):.0%})"
            for p in patterns
        ]) if patterns else "keine"

        return f"""Du bist ein Experte fuer technische Analyse. Analysiere diesen Chart und validiere die CNN-LSTM Prediction.

**Symbol:** {symbol}
**Timeframe:** {timeframe}

**PREDICTION ZU VALIDIEREN:**

1. **Preis-Richtung:** {price_pred.get('direction', 'unknown')}
   - Konfidenz: {price_pred.get('confidence', 0):.0%}
   - 1h Forecast: {price_pred.get('forecast_1h', 'N/A')}
   - 1d Forecast: {price_pred.get('forecast_1d', 'N/A')}

2. **Erkannte Patterns:** {patterns_str}

3. **Markt-Regime:** {regime.get('current', 'unknown')}
   - Konfidenz: {regime.get('probability', 0):.0%}

**DEINE AUFGABE:**

Analysiere den Chart visuell und bewerte:
1. Ist die vorhergesagte Richtung (bullish/bearish/neutral) basierend auf dem Chart plausibel?
2. Sind die erkannten Patterns im Chart sichtbar?
3. Entspricht das erkannte Regime (bull_trend/bear_trend/sideways/high_volatility) dem Chart?

Antworte in diesem exakten JSON-Format:
```json
{{
    "price_direction_agrees": true/false,
    "price_confidence": 0.0-1.0,
    "price_reasoning": "Kurze Begruendung auf Deutsch",

    "patterns_confirmed": ["pattern1", "pattern2"],
    "patterns_rejected": ["pattern3"],
    "patterns_suggested": ["nicht erkannte patterns"],
    "pattern_confidence": 0.0-1.0,
    "pattern_reasoning": "Kurze Begruendung auf Deutsch",

    "regime_agrees": true/false,
    "regime_suggested": "vorgeschlagenes regime wenn anders",
    "regime_confidence": 0.0-1.0,
    "regime_reasoning": "Kurze Begruendung auf Deutsch",

    "overall_agrees": true/false,
    "overall_confidence": 0.0-1.0,
    "overall_reasoning": "Zusammenfassung auf Deutsch"
}}
```

WICHTIG: Alle Begruendungen muessen auf Deutsch sein!"""

    def _parse_claude_response(
        self,
        content: str,
        prediction_id: str,
        symbol: str,
        timeframe: str,
        chart_base64: Optional[str] = None
    ) -> ClaudePredictionValidation:
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

            return ClaudePredictionValidation(
                prediction_id=prediction_id,
                symbol=symbol,
                timeframe=timeframe,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),

                price_direction_agrees=data.get("price_direction_agrees", True),
                price_confidence=float(data.get("price_confidence", 0.5)),
                price_reasoning=data.get("price_reasoning", "Keine Begruendung"),

                patterns_confirmed=data.get("patterns_confirmed", []),
                patterns_rejected=data.get("patterns_rejected", []),
                patterns_suggested=data.get("patterns_suggested", []),
                pattern_confidence=float(data.get("pattern_confidence", 0.5)),
                pattern_reasoning=data.get("pattern_reasoning", "Keine Begruendung"),

                regime_agrees=data.get("regime_agrees", True),
                regime_suggested=data.get("regime_suggested"),
                regime_confidence=float(data.get("regime_confidence", 0.5)),
                regime_reasoning=data.get("regime_reasoning", "Keine Begruendung"),

                overall_agrees=data.get("overall_agrees", True),
                overall_confidence=float(data.get("overall_confidence", 0.5)),
                overall_reasoning=data.get("overall_reasoning", "Keine Begruendung"),

                model_used=self.model,
                status=ValidationStatus.VALIDATED if data.get("overall_agrees") else ValidationStatus.REJECTED,
                chart_image_base64=chart_base64
            )

        except Exception as e:
            logger.warning(f"Failed to parse Claude response: {e}")
            return ClaudePredictionValidation(
                prediction_id=prediction_id,
                symbol=symbol,
                timeframe=timeframe,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                price_direction_agrees=True,
                price_confidence=0.5,
                price_reasoning=f"Parse error: {str(e)[:100]}",
                patterns_confirmed=[],
                patterns_rejected=[],
                patterns_suggested=[],
                pattern_confidence=0.5,
                pattern_reasoning="Parse error",
                regime_agrees=True,
                regime_suggested=None,
                regime_confidence=0.5,
                regime_reasoning="Parse error",
                overall_agrees=True,
                overall_confidence=0.5,
                overall_reasoning=f"Response parse failed: {content[:200]}",
                model_used=self.model,
                status=ValidationStatus.ERROR,
                error_message=str(e),
                chart_image_base64=chart_base64
            )

    async def validate_prediction(
        self,
        prediction_id: str,
        prediction: Dict,
        ohlcv_data: List[Dict],
        force: bool = False
    ) -> ClaudePredictionValidation:
        """
        Validate a CNN-LSTM prediction using Claude Vision API.

        Args:
            prediction_id: Unique identifier for the prediction
            prediction: Full prediction response from CNN-LSTM
            ohlcv_data: OHLCV data for chart rendering
            force: Force validation even if cached

        Returns:
            ClaudePredictionValidation with Claude's assessment
        """
        symbol = prediction.get("symbol", "UNKNOWN")
        timeframe = prediction.get("timeframe", "H1")

        # Check cache
        if not force and prediction_id in self._validation_cache:
            logger.debug(f"Returning cached validation for {prediction_id}")
            return self._validation_cache[prediction_id]

        # Check if API key is configured
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not configured - skipping Claude validation")
            return ClaudePredictionValidation(
                prediction_id=prediction_id,
                symbol=symbol,
                timeframe=timeframe,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                price_direction_agrees=True,
                price_confidence=0.0,
                price_reasoning="Claude validation not configured",
                patterns_confirmed=[],
                patterns_rejected=[],
                patterns_suggested=[],
                pattern_confidence=0.0,
                pattern_reasoning="Claude validation not configured",
                regime_agrees=True,
                regime_suggested=None,
                regime_confidence=0.0,
                regime_reasoning="Claude validation not configured",
                overall_agrees=True,
                overall_confidence=0.0,
                overall_reasoning="Claude validation not configured (no API key)",
                model_used="none",
                status=ValidationStatus.SKIPPED
            )

        try:
            # Render chart
            chart_base64 = self.chart_renderer.render_prediction_chart(
                ohlcv_data=ohlcv_data,
                prediction=prediction,
                show_forecast=True,
                context_candles=30
            )

            if not chart_base64:
                raise ValueError("Failed to render chart image")

            # Build prompt
            prompt = self._build_validation_prompt(prediction, symbol, timeframe)

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
                        "max_tokens": 2048,
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
                prediction_id=prediction_id,
                symbol=symbol,
                timeframe=timeframe,
                chart_base64=chart_base64
            )

            # Cache result
            self._validation_cache[prediction_id] = validation_result
            if len(self._validation_cache) > self._cache_max_size:
                oldest_keys = list(self._validation_cache.keys())[:-self._cache_max_size]
                for key in oldest_keys:
                    del self._validation_cache[key]

            # Add to history
            self._validation_history.append(validation_result.to_dict())
            self._save_history()

            logger.info(
                f"Claude validation for {symbol}/{timeframe}: "
                f"agrees={validation_result.overall_agrees}, "
                f"confidence={validation_result.overall_confidence:.2f}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Claude validation failed: {e}")
            return ClaudePredictionValidation(
                prediction_id=prediction_id,
                symbol=symbol,
                timeframe=timeframe,
                validation_timestamp=datetime.now(timezone.utc).isoformat(),
                price_direction_agrees=True,
                price_confidence=0.0,
                price_reasoning=f"Validation error: {str(e)}",
                patterns_confirmed=[],
                patterns_rejected=[],
                patterns_suggested=[],
                pattern_confidence=0.0,
                pattern_reasoning="Validation error",
                regime_agrees=True,
                regime_suggested=None,
                regime_confidence=0.0,
                regime_reasoning="Validation error",
                overall_agrees=True,
                overall_confidence=0.0,
                overall_reasoning=f"Validation error: {str(e)}",
                model_used=self.model,
                status=ValidationStatus.ERROR,
                error_message=str(e)
            )

    def get_validation_history(
        self,
        limit: int = 50,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """Get validation history with optional filters."""
        history = self._validation_history

        if symbol:
            history = [h for h in history if h.get("symbol") == symbol.upper()]

        return list(reversed(history[-limit:]))

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about Claude validations."""
        if not self._validation_history:
            return {
                "total_validations": 0,
                "agreements": 0,
                "rejections": 0,
                "agreement_rate": 0.0,
                "avg_confidence": 0.0,
                "by_task": {
                    "price": {"agreed": 0, "total": 0},
                    "patterns": {"agreed": 0, "total": 0},
                    "regime": {"agreed": 0, "total": 0}
                }
            }

        total = len(self._validation_history)
        agreements = sum(1 for h in self._validation_history if h.get("overall_agrees"))
        price_agreed = sum(1 for h in self._validation_history if h.get("price_direction_agrees"))
        regime_agreed = sum(1 for h in self._validation_history if h.get("regime_agrees"))

        confidences = [h.get("overall_confidence", 0) for h in self._validation_history]

        return {
            "total_validations": total,
            "agreements": agreements,
            "rejections": total - agreements,
            "agreement_rate": round(agreements / total * 100, 1) if total > 0 else 0.0,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "by_task": {
                "price": {
                    "agreed": price_agreed,
                    "total": total,
                    "rate": round(price_agreed / total * 100, 1) if total > 0 else 0.0
                },
                "regime": {
                    "agreed": regime_agreed,
                    "total": total,
                    "rate": round(regime_agreed / total * 100, 1) if total > 0 else 0.0
                }
            }
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

    def clear_memory(self):
        """Clear all in-memory data."""
        history_count = len(self._validation_history)
        cache_count = len(self._validation_cache)

        self._validation_history = []
        self._validation_cache = {}

        logger.info(f"Claude validator memory cleared: history={history_count}, cache={cache_count}")

        return {
            "cleared": True,
            "validation_history_cleared": history_count,
            "validation_cache_cleared": cache_count
        }


# Global singleton
claude_prediction_validator_service = ClaudePredictionValidatorService()
