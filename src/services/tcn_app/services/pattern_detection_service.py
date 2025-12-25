"""Pattern detection service."""

import os
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
from loguru import logger

from ..models.tcn_model import TCNPatternClassifier
from ..models.pattern_classifier import PatternClassifier, PatternDetection, PatternType, PatternPoint
from ..models.schemas import DetectedPattern, PatternDetectionResponse


class PatternDetectionService:
    """
    Service for detecting chart patterns.

    Combines TCN deep learning with rule-based detection
    for robust pattern recognition.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the pattern detection service.

        Args:
            device: Device for TCN model ('cuda' or 'cpu')
        """
        self.device = device
        self.tcn_model = TCNPatternClassifier(device=device)
        self.rule_classifier = PatternClassifier()
        self._model_loaded = False
        self._model_version = "1.0.0"
        self._model_path: Optional[str] = None

    def load_model(self, model_path: Optional[str] = None):
        """Load the TCN model."""
        try:
            self.tcn_model.load(model_path)
            self._model_loaded = True
            self._model_path = model_path
            logger.info(f"TCN Pattern model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load TCN model: {e}")
            self._model_loaded = False

    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """
        Hot-reload the model from disk.

        Called by training service when a new model is available.

        Args:
            model_path: Path to new model. If None, uses latest.pt symlink.

        Returns:
            True if reload successful
        """
        # Default to latest.pt if no path provided
        if model_path is None:
            model_path = os.getenv("TCN_MODEL_PATH", "data/models/tcn/latest.pt")

        # Check if model exists
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return False

        try:
            # Reinitialize the TCN model
            self.tcn_model = TCNPatternClassifier(device=self.device)
            self.tcn_model.load(model_path)
            self._model_loaded = True
            self._model_path = model_path
            logger.info(f"TCN model hot-reloaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "loaded": self._model_loaded,
            "path": self._model_path,
            "version": self._model_version,
            "device": self.device,
            "parameters": self.tcn_model.get_num_parameters() if self._model_loaded else 0
        }

    def _generate_pattern_points(
        self,
        pattern_type: str,
        ohlcv: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> Optional[List[PatternPoint]]:
        """
        Generate approximate pattern points for TCN-detected patterns.

        Uses swing point detection to find key levels within the pattern window.
        """
        try:
            # Extract data window
            window = ohlcv[start_idx:end_idx + 1]
            if len(window) < 5:
                return None

            highs = window[:, 1]  # High prices
            lows = window[:, 2]   # Low prices

            # Find swing points in window
            swing_highs, swing_lows = self.rule_classifier.find_swing_points(highs, lows, window=3)

            points = []

            if pattern_type in ["head_and_shoulders", "triple_top"]:
                # Find highest points for head/shoulders
                if len(swing_highs) >= 3:
                    sorted_highs = sorted(swing_highs, key=lambda i: highs[i], reverse=True)
                    head_i = sorted_highs[0]
                    shoulders = sorted([h for h in sorted_highs[1:3]])
                    points = [
                        PatternPoint(start_idx + shoulders[0], float(highs[shoulders[0]]), "left_shoulder"),
                        PatternPoint(start_idx + head_i, float(highs[head_i]), "head"),
                        PatternPoint(start_idx + shoulders[1], float(highs[shoulders[1]]), "right_shoulder"),
                    ]
                    # Neckline from lows
                    if len(swing_lows) >= 2:
                        nl_left = min(swing_lows)
                        nl_right = max(swing_lows)
                        points.append(PatternPoint(start_idx + nl_left, float(lows[nl_left]), "neckline_left"))
                        points.append(PatternPoint(start_idx + nl_right, float(lows[nl_right]), "neckline_right"))

            elif pattern_type in ["inverse_head_and_shoulders", "triple_bottom"]:
                # Find lowest points
                if len(swing_lows) >= 3:
                    sorted_lows = sorted(swing_lows, key=lambda i: lows[i])
                    head_i = sorted_lows[0]
                    shoulders = sorted([l for l in sorted_lows[1:3]])
                    points = [
                        PatternPoint(start_idx + shoulders[0], float(lows[shoulders[0]]), "left_shoulder"),
                        PatternPoint(start_idx + head_i, float(lows[head_i]), "head"),
                        PatternPoint(start_idx + shoulders[1], float(lows[shoulders[1]]), "right_shoulder"),
                    ]
                    # Neckline from highs
                    if len(swing_highs) >= 2:
                        nl_left = min(swing_highs)
                        nl_right = max(swing_highs)
                        points.append(PatternPoint(start_idx + nl_left, float(highs[nl_left]), "neckline_left"))
                        points.append(PatternPoint(start_idx + nl_right, float(highs[nl_right]), "neckline_right"))

            elif pattern_type == "double_top":
                if len(swing_highs) >= 2:
                    top1_i, top2_i = swing_highs[0], swing_highs[-1]
                    valley_i = swing_lows[len(swing_lows)//2] if swing_lows else len(window)//2
                    points = [
                        PatternPoint(start_idx + top1_i, float(highs[top1_i]), "top_1"),
                        PatternPoint(start_idx + valley_i, float(lows[valley_i]) if valley_i < len(lows) else float(np.min(lows)), "valley"),
                        PatternPoint(start_idx + top2_i, float(highs[top2_i]), "top_2"),
                    ]

            elif pattern_type == "double_bottom":
                if len(swing_lows) >= 2:
                    bot1_i, bot2_i = swing_lows[0], swing_lows[-1]
                    peak_i = swing_highs[len(swing_highs)//2] if swing_highs else len(window)//2
                    points = [
                        PatternPoint(start_idx + bot1_i, float(lows[bot1_i]), "bottom_1"),
                        PatternPoint(start_idx + peak_i, float(highs[peak_i]) if peak_i < len(highs) else float(np.max(highs)), "peak"),
                        PatternPoint(start_idx + bot2_i, float(lows[bot2_i]), "bottom_2"),
                    ]

            elif pattern_type in ["ascending_triangle", "descending_triangle", "symmetrical_triangle"]:
                # Triangle uses trendlines
                if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                    points = [
                        PatternPoint(start_idx + swing_highs[0], float(highs[swing_highs[0]]), "resistance_start"),
                        PatternPoint(start_idx + swing_highs[-1], float(highs[swing_highs[-1]]), "resistance_end"),
                        PatternPoint(start_idx + swing_lows[0], float(lows[swing_lows[0]]), "support_start"),
                        PatternPoint(start_idx + swing_lows[-1], float(lows[swing_lows[-1]]), "support_end"),
                    ]

            elif pattern_type in ["channel_up", "channel_down"]:
                # Channel uses parallel lines
                if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                    points = [
                        PatternPoint(start_idx, float(highs[0]), "upper_start"),
                        PatternPoint(end_idx, float(highs[-1]), "upper_end"),
                        PatternPoint(start_idx, float(lows[0]), "lower_start"),
                        PatternPoint(end_idx, float(lows[-1]), "lower_end"),
                    ]

            elif pattern_type in ["bull_flag", "bear_flag"]:
                # Flag has pole and flag portion
                mid = len(window) // 3
                points = [
                    PatternPoint(start_idx, float(lows[0]) if pattern_type == "bull_flag" else float(highs[0]), "pole_start"),
                    PatternPoint(start_idx + mid, float(highs[mid]) if pattern_type == "bull_flag" else float(lows[mid]), "pole_end"),
                    PatternPoint(end_idx, float(window[-1, 3]), "flag_end"),  # Close price
                ]

            elif pattern_type in ["rising_wedge", "falling_wedge"]:
                if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                    points = [
                        PatternPoint(start_idx + swing_highs[0], float(highs[swing_highs[0]]), "upper_start"),
                        PatternPoint(start_idx + swing_highs[-1], float(highs[swing_highs[-1]]), "upper_end"),
                        PatternPoint(start_idx + swing_lows[0], float(lows[swing_lows[0]]), "lower_start"),
                        PatternPoint(start_idx + swing_lows[-1], float(lows[swing_lows[-1]]), "lower_end"),
                    ]

            return points if points else None

        except Exception as e:
            logger.debug(f"Error generating pattern points for {pattern_type}: {e}")
            return None

    async def detect_patterns(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 200,
        threshold: float = 0.5,
        pattern_filter: Optional[List[str]] = None
    ) -> PatternDetectionResponse:
        """
        Detect patterns for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            lookback: Number of candles to analyze
            threshold: Confidence threshold
            pattern_filter: Optional list of patterns to detect

        Returns:
            PatternDetectionResponse with detected patterns
        """
        try:
            # Fetch OHLCV data
            from src.services.data_gateway_service import data_gateway

            data = await data_gateway.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback
            )

            if not data or len(data) < 50:
                return PatternDetectionResponse(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    patterns=[],
                    total_patterns=0,
                    market_context={},
                    model_version=self._model_version
                )

            # Map timeframe to field prefix
            tf_map = {"1h": "h1", "4h": "h1", "1d": "d1", "15m": "m15", "m15": "m15", "h1": "h1", "d1": "d1"}
            prefix = tf_map.get(timeframe.lower(), "h1")

            # Convert to numpy array - handle both direct OHLC and prefixed fields
            ohlcv = np.array([
                [
                    d.get('open') or d.get(f'{prefix}_open', 0),
                    d.get('high') or d.get(f'{prefix}_high', 0),
                    d.get('low') or d.get(f'{prefix}_low', 0),
                    d.get('close') or d.get(f'{prefix}_close', 0),
                    d.get('volume', 0)
                ]
                for d in data
            ], dtype=np.float32)

            timestamps = [d.get('timestamp', d.get('time', '')) for d in data]

            # Detect patterns
            detected = await self._detect_in_sequence(
                ohlcv, timestamps, threshold, pattern_filter
            )

            # Get market context
            context = self._get_market_context(ohlcv)

            return PatternDetectionResponse(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                patterns=detected,
                total_patterns=len(detected),
                market_context=context,
                model_version=self._model_version
            )

        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
            return PatternDetectionResponse(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                patterns=[],
                total_patterns=0,
                market_context={"error": str(e)},
                model_version=self._model_version
            )

    async def _detect_in_sequence(
        self,
        ohlcv: np.ndarray,
        timestamps: List[str],
        threshold: float,
        pattern_filter: Optional[List[str]]
    ) -> List[DetectedPattern]:
        """
        Detect patterns in OHLCV sequence.

        Combines TCN predictions with rule-based detection.
        """
        detected = []

        # TCN predictions
        if self._model_loaded:
            tcn_predictions = self.tcn_model.predict(ohlcv, threshold)

            for pattern_type, confidence in tcn_predictions.items():
                if pattern_filter and pattern_type not in pattern_filter:
                    continue

                # Estimate pattern location (simplified)
                pattern_info = self.rule_classifier.get_pattern_info(
                    PatternType(pattern_type)
                )

                start_idx = max(0, len(ohlcv) - 50)
                end_idx = len(ohlcv) - 1

                # Generate pattern points for visualization
                pattern_points = self._generate_pattern_points(
                    pattern_type, ohlcv, start_idx, end_idx
                )
                pattern_points_dicts = None
                if pattern_points:
                    pattern_points_dicts = [p.to_dict() for p in pattern_points]

                detected.append(DetectedPattern(
                    pattern_type=pattern_type,
                    confidence=round(confidence, 4),
                    start_index=start_idx,
                    end_index=end_idx,
                    start_time=str(timestamps[start_idx]) if timestamps and start_idx < len(timestamps) else None,
                    end_time=str(timestamps[end_idx]) if timestamps and end_idx < len(timestamps) else None,
                    direction=pattern_info.get("direction"),
                    pattern_points=pattern_points_dicts
                ))

        # Rule-based detection (supplement)
        rule_patterns = self.rule_classifier.detect_all_patterns(ohlcv)

        for pattern in rule_patterns:
            if pattern_filter and pattern.pattern_type.value not in pattern_filter:
                continue

            # Check if already detected by TCN
            tcn_detected = any(
                d.pattern_type == pattern.pattern_type.value
                for d in detected
            )

            if not tcn_detected:
                # Get pattern points from rule-based detection
                pattern_points_dicts = pattern.get_pattern_points_as_dicts()

                detected.append(DetectedPattern(
                    pattern_type=pattern.pattern_type.value,
                    confidence=round(pattern.confidence, 4),
                    start_index=pattern.start_index,
                    end_index=pattern.end_index,
                    start_time=str(timestamps[pattern.start_index]) if timestamps and pattern.start_index < len(timestamps) else None,
                    end_time=str(timestamps[pattern.end_index]) if timestamps and pattern.end_index < len(timestamps) else None,
                    price_target=pattern.price_target,
                    invalidation_level=pattern.invalidation_level,
                    pattern_height=pattern.pattern_height,
                    direction=pattern.direction,
                    pattern_points=pattern_points_dicts
                ))

        # Sort by confidence
        detected.sort(key=lambda x: x.confidence, reverse=True)

        return detected

    async def scan_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        threshold: float = 0.6,
        min_patterns: int = 1
    ) -> List[Dict]:
        """
        Scan multiple symbols for patterns.

        Args:
            symbols: List of symbols to scan
            timeframe: Timeframe
            threshold: Minimum confidence
            min_patterns: Minimum patterns to report

        Returns:
            List of results per symbol
        """
        import time
        results = []

        for symbol in symbols:
            start_time = time.time()

            try:
                response = await self.detect_patterns(
                    symbol=symbol,
                    timeframe=timeframe,
                    threshold=threshold
                )

                if len(response.patterns) >= min_patterns:
                    results.append({
                        "symbol": symbol,
                        "patterns": [p.model_dump() for p in response.patterns],
                        "scan_time_ms": round((time.time() - start_time) * 1000, 2)
                    })

            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")

        return results

    def _get_market_context(self, ohlcv: np.ndarray) -> Dict:
        """
        Get market context for pattern interpretation.
        """
        closes = ohlcv[:, 3]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]

        # Calculate basic metrics
        current_price = closes[-1]
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price

        # Trend direction
        if current_price > sma_20 > sma_50:
            trend = "bullish"
        elif current_price < sma_20 < sma_50:
            trend = "bearish"
        else:
            trend = "neutral"

        # Volatility
        returns = np.diff(np.log(closes))
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # Recent high/low
        recent_high = np.max(highs[-20:]) if len(highs) >= 20 else highs[-1]
        recent_low = np.min(lows[-20:]) if len(lows) >= 20 else lows[-1]

        price_range_pct = float((recent_high - recent_low) / recent_low * 100) if recent_low > 0 else 0.0

        return {
            "trend": trend,
            "price": round(float(current_price), 4),
            "sma_20": round(float(sma_20), 4),
            "sma_50": round(float(sma_50), 4),
            "volatility": round(float(volatility), 4),
            "recent_high": round(float(recent_high), 4),
            "recent_low": round(float(recent_low), 4),
            "price_range_pct": round(price_range_pct, 2)
        }

    def get_supported_patterns(self) -> List[str]:
        """Get list of supported pattern types."""
        return TCNPatternClassifier.PATTERN_CLASSES


# Singleton instance
pattern_detection_service = PatternDetectionService()
