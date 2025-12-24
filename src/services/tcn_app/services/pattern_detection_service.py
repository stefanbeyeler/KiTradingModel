"""Pattern detection service."""

from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
from loguru import logger

from ..models.tcn_model import TCNPatternClassifier
from ..models.pattern_classifier import PatternClassifier, PatternDetection, PatternType
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

    def load_model(self, model_path: Optional[str] = None):
        """Load the TCN model."""
        try:
            self.tcn_model.load(model_path)
            self._model_loaded = True
            logger.info("TCN Pattern model loaded")
        except Exception as e:
            logger.warning(f"Could not load TCN model: {e}")
            self._model_loaded = False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

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

                detected.append(DetectedPattern(
                    pattern_type=pattern_type,
                    confidence=round(confidence, 4),
                    start_index=max(0, len(ohlcv) - 50),
                    end_index=len(ohlcv) - 1,
                    start_time=str(timestamps[max(0, len(ohlcv) - 50)]) if timestamps else None,
                    end_time=str(timestamps[-1]) if timestamps else None,
                    direction=pattern_info.get("direction")
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
                detected.append(DetectedPattern(
                    pattern_type=pattern.pattern_type.value,
                    confidence=round(pattern.confidence, 4),
                    start_index=pattern.start_index,
                    end_index=pattern.end_index,
                    start_time=str(timestamps[pattern.start_index]) if timestamps else None,
                    end_time=str(timestamps[pattern.end_index]) if timestamps else None,
                    price_target=pattern.price_target,
                    invalidation_level=pattern.invalidation_level,
                    pattern_height=pattern.pattern_height,
                    direction=pattern.direction
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
