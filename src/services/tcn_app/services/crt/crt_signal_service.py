"""CRT Signal Service - Generates trading signals with multi-service integration.

Combines CRT detection with:
- HMM Regime Filter (distribution/accumulation phases)
- NHITS Forecast Confirmation (directional bias)
- TCN Pattern Confluence (chart patterns)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from loguru import logger
import httpx

from src.utils.session_utils import SessionType, get_session_for_h4_candle

from .session_manager import SessionManager, session_manager
from .range_tracker import RangeTracker, CRTRange, CRTState, range_tracker
from .purge_detector import PurgeDetector, PurgeEvent, ReEntryEvent, purge_detector
from src.config.microservices import microservices_config

UTC_TZ = timezone.utc


@dataclass
class CRTSignal:
    """Complete CRT Trading Signal with multi-service confirmation."""
    symbol: str
    direction: str                  # "long" or "short"
    signal_time: datetime

    # CRT Core
    crt_range: CRTRange
    purge_event: PurgeEvent
    reentry_event: ReEntryEvent

    # Trade Parameters
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    position_size_percent: float    # Recommended position (based on risk)

    # Confidence Factors
    crt_confidence: float           # CRT pattern quality (0-1)
    regime_alignment: float         # HMM regime alignment (0-1)
    forecast_alignment: float       # NHITS forecast alignment (0-1)
    pattern_alignment: float        # TCN pattern alignment (0-1)
    total_confidence: float         # Weighted average

    # Risk Metrics
    risk_reward_ratio: float
    max_risk_percent: float

    # Metadata
    session_type: str
    invalidation_price: float       # Price where signal becomes invalid

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "signal_time": self.signal_time.isoformat(),
            "trade": {
                "entry_price": self.entry_price,
                "stop_loss": self.stop_loss,
                "take_profit_1": self.take_profit_1,
                "take_profit_2": self.take_profit_2,
                "invalidation_price": self.invalidation_price,
                "risk_reward_ratio": round(self.risk_reward_ratio, 2),
                "position_size_percent": round(self.position_size_percent, 2),
            },
            "confidence": {
                "total": round(self.total_confidence, 3),
                "crt": round(self.crt_confidence, 3),
                "regime": round(self.regime_alignment, 3),
                "forecast": round(self.forecast_alignment, 3),
                "pattern": round(self.pattern_alignment, 3),
            },
            "crt_details": {
                "session_type": self.session_type,
                "range": self.crt_range.to_dict() if self.crt_range else None,
                "purge": self.purge_event.to_dict() if self.purge_event else None,
                "reentry": self.reentry_event.to_dict() if self.reentry_event else None,
            },
            "max_risk_percent": self.max_risk_percent,
        }


class CRTSignalService:
    """
    Generates CRT signals with multi-service integration.

    Integration:
    - HMM Service (3004): Regime filter
    - NHITS Service (3002): Directional bias
    - TCN Service (3003): Pattern confluence
    """

    # Service URLs (from central config)
    HMM_SERVICE_URL = microservices_config.hmm_service_url
    NHITS_SERVICE_URL = microservices_config.nhits_service_url
    TCN_SERVICE_URL = microservices_config.tcn_service_url

    # Confidence weights
    WEIGHTS = {
        "crt": 0.40,       # CRT pattern quality
        "regime": 0.25,    # HMM regime
        "forecast": 0.20,  # NHITS forecast
        "pattern": 0.15,   # TCN pattern
    }

    # Risk settings
    DEFAULT_MAX_RISK_PERCENT = 2.0
    MIN_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        session_mgr: Optional[SessionManager] = None,
        range_trk: Optional[RangeTracker] = None,
        purge_det: Optional[PurgeDetector] = None,
    ):
        """Initialize CRT Signal Service."""
        self.session_manager = session_mgr or session_manager
        self.range_tracker = range_trk or range_tracker
        self.purge_detector = purge_det or purge_detector
        self._http_client: Optional[httpx.AsyncClient] = None
        logger.info("CRT SignalService initialized")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    async def analyze_symbol(
        self,
        symbol: str,
        include_service_integration: bool = True,
    ) -> dict:
        """
        Perform full CRT analysis for a symbol.

        Args:
            symbol: Trading symbol
            include_service_integration: Include HMM/NHITS/TCN checks

        Returns:
            Complete analysis dict
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.now(UTC_TZ).isoformat(),
            "session_info": self.session_manager.get_session_info(),
            "crt_status": None,
            "signal": None,
            "service_alignment": None,
        }

        # Get current range status
        crt_range = self.range_tracker.get_active_range(symbol)

        if crt_range:
            result["crt_status"] = {
                "has_range": True,
                "range": crt_range.to_dict(),
                "state": crt_range.state.value,
            }

            # Check for signal
            if crt_range.has_signal:
                signal = await self._create_signal_from_range(
                    symbol, crt_range, include_service_integration
                )
                if signal:
                    result["signal"] = signal.to_dict()
        else:
            result["crt_status"] = {
                "has_range": False,
                "reason": "No active CRT range",
                "next_session": self.session_manager.get_next_key_session(),
            }

        # Get service alignments if requested
        if include_service_integration:
            result["service_alignment"] = await self._get_service_alignments(symbol, "neutral")

        return result

    async def scan_symbols(
        self,
        symbols: List[str],
        min_confidence: float = 0.5,
    ) -> List[CRTSignal]:
        """
        Scan multiple symbols for CRT signals.

        Args:
            symbols: List of symbols to scan
            min_confidence: Minimum confidence threshold

        Returns:
            List of CRT signals above threshold
        """
        signals = []

        for symbol in symbols:
            try:
                crt_range = self.range_tracker.get_active_range(symbol)
                if crt_range and crt_range.has_signal:
                    signal = await self._create_signal_from_range(
                        symbol, crt_range, include_service_integration=True
                    )
                    if signal and signal.total_confidence >= min_confidence:
                        signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        return signals

    async def process_candle(
        self,
        symbol: str,
        candle: dict,
        is_close: bool = True,
    ) -> Optional[CRTSignal]:
        """
        Process a new candle for CRT detection.

        Args:
            symbol: Trading symbol
            candle: Candle data {datetime, open, high, low, close, volume}
            is_close: True if this is a closed candle

        Returns:
            CRTSignal if generated, None otherwise
        """
        close_price = candle.get("close", 0)
        high_price = candle.get("high", 0)
        low_price = candle.get("low", 0)

        # Update range tracker with price
        crt_range = self.range_tracker.update_price(
            symbol=symbol,
            current_price=close_price,
            is_candle_close=is_close,
        )

        if not crt_range:
            return None

        # Check if we have a new signal
        if crt_range.has_signal and crt_range.signal_time:
            # Only return signal if it's fresh (within last minute)
            age_seconds = (datetime.now(UTC_TZ) - crt_range.signal_time).total_seconds()
            if age_seconds < 60:
                return await self._create_signal_from_range(
                    symbol, crt_range, include_service_integration=True
                )

        return None

    async def create_range_from_h4(
        self,
        symbol: str,
        h4_candle: dict,
    ) -> Optional[CRTRange]:
        """
        Create a CRT range from an H4 candle.

        Args:
            symbol: Trading symbol
            h4_candle: H4 candle data

        Returns:
            CRTRange if created, None otherwise
        """
        candle_start = self._parse_candle_time(h4_candle)
        session_type = get_session_for_h4_candle(candle_start)

        if not session_type:
            logger.debug(f"CRT {symbol}: H4 candle is not a key session candle")
            return None

        return self.range_tracker.create_range(
            symbol=symbol,
            session_type=session_type,
            candle_start=candle_start,
            candle_end=candle_start.replace(hour=candle_start.hour + 4),
            open_price=h4_candle.get("open", 0),
            high_price=h4_candle.get("high", 0),
            low_price=h4_candle.get("low", 0),
            close_price=h4_candle.get("close", 0),
            volume=h4_candle.get("volume", 0),
        )

    async def _create_signal_from_range(
        self,
        symbol: str,
        crt_range: CRTRange,
        include_service_integration: bool = True,
    ) -> Optional[CRTSignal]:
        """Create a CRTSignal from an active range with signal."""
        if not crt_range.has_signal:
            return None

        direction = crt_range.signal_direction

        # Reconstruct purge event
        purge = PurgeEvent(
            direction=crt_range.purge_direction or "unknown",
            purge_price=crt_range.purge_price or 0,
            purge_wick=abs((crt_range.purge_price or 0) - (
                crt_range.crt_high if crt_range.purge_direction == "above" else crt_range.crt_low
            )),
            purge_wick_percent=0,  # Calculated below
            purge_time=crt_range.purge_time or datetime.now(UTC_TZ),
            candles_since_range=0,
        )

        # Calculate purge wick percent
        if crt_range.purge_direction == "above":
            purge.purge_wick_percent = (purge.purge_wick / crt_range.crt_high) * 100
        else:
            purge.purge_wick_percent = (purge.purge_wick / crt_range.crt_low) * 100

        # Create re-entry event
        entry_price = crt_range.signal_price or 0
        if direction == "short":
            stop_loss = purge.purge_price * 1.001  # 0.1% buffer
            take_profit_1 = crt_range.crt_low
            risk = stop_loss - entry_price
            reward = entry_price - take_profit_1
        else:
            stop_loss = purge.purge_price * 0.999  # 0.1% buffer
            take_profit_1 = crt_range.crt_high
            risk = entry_price - stop_loss
            reward = take_profit_1 - entry_price

        rr_ratio = reward / risk if risk > 0 else 0

        reentry = ReEntryEvent(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=None,
            risk_reward_1=rr_ratio,
            risk_reward_2=None,
            risk_amount=risk,
            confidence=0.5,  # Will be recalculated
            reentry_time=crt_range.signal_time or datetime.now(UTC_TZ),
            reentry_candle_close=entry_price,
        )

        # Get service alignments
        if include_service_integration:
            alignments = await self._get_service_alignments(symbol, direction)
        else:
            alignments = {
                "regime": 0.5,
                "forecast": 0.5,
                "pattern": 0.5,
            }

        # Calculate CRT confidence
        crt_confidence = self._calculate_crt_confidence(crt_range, purge, reentry)

        # Calculate total confidence
        total_confidence = (
            self.WEIGHTS["crt"] * crt_confidence +
            self.WEIGHTS["regime"] * alignments["regime"] +
            self.WEIGHTS["forecast"] * alignments["forecast"] +
            self.WEIGHTS["pattern"] * alignments["pattern"]
        )

        # Calculate position size
        position_size = self._calculate_position_size(
            risk, entry_price, total_confidence
        )

        # Invalidation price
        invalidation = purge.purge_price

        return CRTSignal(
            symbol=symbol,
            direction=direction,
            signal_time=crt_range.signal_time or datetime.now(UTC_TZ),
            crt_range=crt_range,
            purge_event=purge,
            reentry_event=reentry,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=None,
            position_size_percent=position_size,
            crt_confidence=crt_confidence,
            regime_alignment=alignments["regime"],
            forecast_alignment=alignments["forecast"],
            pattern_alignment=alignments["pattern"],
            total_confidence=total_confidence,
            risk_reward_ratio=rr_ratio,
            max_risk_percent=self.DEFAULT_MAX_RISK_PERCENT,
            session_type=crt_range.session_type.value,
            invalidation_price=invalidation,
        )

    def _calculate_crt_confidence(
        self,
        crt_range: CRTRange,
        purge: PurgeEvent,
        reentry: ReEntryEvent,
    ) -> float:
        """Calculate CRT-specific confidence."""
        confidence = 0.5

        # R:R bonus
        if reentry.risk_reward_1 >= 2.0:
            confidence += 0.2
        elif reentry.risk_reward_1 >= 1.5:
            confidence += 0.15
        elif reentry.risk_reward_1 >= 1.0:
            confidence += 0.1

        # Purge wick bonus
        if purge.purge_wick_percent >= 0.5:
            confidence += 0.15
        elif purge.purge_wick_percent >= 0.2:
            confidence += 0.1

        # Range freshness
        if crt_range.age_hours <= 4:
            confidence += 0.05

        return min(confidence, 1.0)

    async def _get_service_alignments(
        self,
        symbol: str,
        direction: str,
    ) -> dict:
        """Get alignment scores from other services."""
        alignments = {
            "regime": 0.5,    # Default neutral
            "forecast": 0.5,
            "pattern": 0.5,
        }

        try:
            # Run all service calls in parallel
            results = await asyncio.gather(
                self._get_regime_alignment(symbol, direction),
                self._get_forecast_alignment(symbol, direction),
                self._get_pattern_alignment(symbol, direction),
                return_exceptions=True,
            )

            if not isinstance(results[0], Exception):
                alignments["regime"] = results[0]
            if not isinstance(results[1], Exception):
                alignments["forecast"] = results[1]
            if not isinstance(results[2], Exception):
                alignments["pattern"] = results[2]

        except Exception as e:
            logger.warning(f"Error getting service alignments: {e}")

        return alignments

    async def _get_regime_alignment(self, symbol: str, direction: str) -> float:
        """Get HMM regime alignment score."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.HMM_SERVICE_URL}/api/v1/regime/{symbol}",
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                regime = data.get("regime", "").lower()

                # Alignment logic
                if direction == "short":
                    if regime in ["distribution", "trending_down"]:
                        return 1.0
                    elif regime == "ranging":
                        return 0.6
                    elif regime in ["accumulation", "trending_up"]:
                        return 0.2
                else:  # long
                    if regime in ["accumulation", "trending_up"]:
                        return 1.0
                    elif regime == "ranging":
                        return 0.6
                    elif regime in ["distribution", "trending_down"]:
                        return 0.2

        except Exception as e:
            logger.debug(f"HMM service unavailable: {e}")

        return 0.5

    async def _get_forecast_alignment(self, symbol: str, direction: str) -> float:
        """Get NHITS forecast alignment score."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.NHITS_SERVICE_URL}/api/v1/forecast/{symbol}",
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                forecast_change = data.get("forecast_change_percent", 0)

                # Alignment logic
                if direction == "short":
                    if forecast_change < -1.0:
                        return 1.0
                    elif forecast_change < -0.5:
                        return 0.8
                    elif forecast_change < 0:
                        return 0.6
                    elif forecast_change > 1.0:
                        return 0.2
                    else:
                        return 0.4
                else:  # long
                    if forecast_change > 1.0:
                        return 1.0
                    elif forecast_change > 0.5:
                        return 0.8
                    elif forecast_change > 0:
                        return 0.6
                    elif forecast_change < -1.0:
                        return 0.2
                    else:
                        return 0.4

        except Exception as e:
            logger.debug(f"NHITS service unavailable: {e}")

        return 0.5

    async def _get_pattern_alignment(self, symbol: str, direction: str) -> float:
        """Get TCN pattern alignment score."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.TCN_SERVICE_URL}/api/v1/detect/{symbol}",
                params={"timeframe": "H4", "threshold": 0.5},
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                patterns = data.get("patterns", [])

                bearish_patterns = [
                    "head_and_shoulders", "double_top", "triple_top",
                    "rising_wedge", "bear_flag", "descending_triangle"
                ]
                bullish_patterns = [
                    "inverse_head_and_shoulders", "double_bottom", "triple_bottom",
                    "falling_wedge", "bull_flag", "ascending_triangle", "cup_and_handle"
                ]

                bearish_count = sum(
                    1 for p in patterns if p.get("pattern_type") in bearish_patterns
                )
                bullish_count = sum(
                    1 for p in patterns if p.get("pattern_type") in bullish_patterns
                )

                if direction == "short":
                    if bearish_count > bullish_count:
                        return 0.8 + min(bearish_count * 0.05, 0.2)
                    elif bearish_count > 0:
                        return 0.6
                    elif bullish_count > 0:
                        return 0.3
                else:  # long
                    if bullish_count > bearish_count:
                        return 0.8 + min(bullish_count * 0.05, 0.2)
                    elif bullish_count > 0:
                        return 0.6
                    elif bearish_count > 0:
                        return 0.3

        except Exception as e:
            logger.debug(f"TCN pattern check failed: {e}")

        return 0.5

    def _calculate_position_size(
        self,
        risk_amount: float,
        entry_price: float,
        confidence: float,
    ) -> float:
        """Calculate recommended position size as % of portfolio."""
        # Base position based on risk
        if entry_price <= 0 or risk_amount <= 0:
            return 0

        # Adjust by confidence
        if confidence >= 0.8:
            risk_percent = self.DEFAULT_MAX_RISK_PERCENT
        elif confidence >= 0.7:
            risk_percent = self.DEFAULT_MAX_RISK_PERCENT * 0.75
        elif confidence >= 0.6:
            risk_percent = self.DEFAULT_MAX_RISK_PERCENT * 0.5
        else:
            risk_percent = self.DEFAULT_MAX_RISK_PERCENT * 0.25

        return risk_percent

    def _parse_candle_time(self, candle: dict) -> datetime:
        """Parse candle timestamp."""
        ts = candle.get("datetime") or candle.get("timestamp") or candle.get("time")

        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=UTC_TZ)
            return ts

        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=UTC_TZ)
                return dt
            except ValueError:
                pass

        return datetime.now(UTC_TZ)

    def get_status(self) -> dict:
        """Get service status."""
        return {
            "service": "CRT Signal Service",
            "range_tracker": self.range_tracker.get_status(),
            "session_info": self.session_manager.get_session_info(),
            "config": {
                "weights": self.WEIGHTS,
                "min_confidence": self.MIN_CONFIDENCE_THRESHOLD,
                "max_risk_percent": self.DEFAULT_MAX_RISK_PERCENT,
            },
            "service_urls": {
                "hmm": self.HMM_SERVICE_URL,
                "nhits": self.NHITS_SERVICE_URL,
                "tcn": self.TCN_SERVICE_URL,
            }
        }

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()


# Singleton instance
crt_signal_service = CRTSignalService()
