"""Purge Detector - Detects liquidity sweeps and re-entries for CRT.

A "purge" is when price sweeps above/below the CRT range to take out
liquidity (stop losses) before reversing back into the range.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List
from loguru import logger

from .range_tracker import CRTRange, CRTState

UTC_TZ = timezone.utc


@dataclass
class PurgeEvent:
    """Details of a liquidity purge event."""
    direction: str                  # "above" or "below"
    purge_price: float              # Highest/lowest price during purge
    purge_wick: float               # How far beyond range (in price units)
    purge_wick_percent: float       # How far beyond range (in %)
    purge_time: datetime
    candles_since_range: int        # Number of candles since range was defined
    volume_at_purge: Optional[float] = None
    volume_ratio: Optional[float] = None  # Volume vs average

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "direction": self.direction,
            "purge_price": self.purge_price,
            "purge_wick": round(self.purge_wick, 5),
            "purge_wick_percent": round(self.purge_wick_percent, 3),
            "purge_time": self.purge_time.isoformat(),
            "candles_since_range": self.candles_since_range,
            "volume_at_purge": self.volume_at_purge,
            "volume_ratio": round(self.volume_ratio, 2) if self.volume_ratio else None,
        }


@dataclass
class ReEntryEvent:
    """Details of a re-entry into the CRT range."""
    direction: str                  # "long" or "short"
    entry_price: float              # Recommended entry price
    stop_loss: float                # Stop loss (beyond purge)
    take_profit_1: float            # TP1: Opposite side of range
    take_profit_2: Optional[float]  # TP2: Extended target
    risk_reward_1: float            # R:R to TP1
    risk_reward_2: Optional[float]  # R:R to TP2
    risk_amount: float              # Risk in price units
    confidence: float               # Signal confidence (0-1)
    reentry_time: datetime
    reentry_candle_close: float     # Close price of re-entry candle

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "risk_reward_1": round(self.risk_reward_1, 2),
            "risk_reward_2": round(self.risk_reward_2, 2) if self.risk_reward_2 else None,
            "risk_amount": round(self.risk_amount, 5),
            "confidence": round(self.confidence, 3),
            "reentry_time": self.reentry_time.isoformat(),
            "reentry_candle_close": self.reentry_candle_close,
        }


class PurgeDetector:
    """
    Detects purge (liquidity sweep) and re-entry events.

    Purge Criteria:
    - Price must exceed CRT High/Low by at least PURGE_MIN_PERCENT
    - Must occur within PURGE_MAX_HOURS after range definition
    - Volume spike increases confidence

    Re-Entry Criteria:
    - Candle must CLOSE back inside the range (not just wick)
    - Must happen after a valid purge
    """

    # Configuration
    PURGE_MIN_PERCENT = 0.05        # 0.05% minimum purge beyond range
    PURGE_MAX_HOURS = 12            # Max hours to wait for purge
    STOP_BUFFER_PERCENT = 0.1       # Buffer beyond purge for stop loss
    MIN_RISK_REWARD = 1.0           # Minimum R:R for valid signal
    VOLUME_SPIKE_THRESHOLD = 1.5    # Volume ratio for spike detection

    def __init__(self):
        """Initialize PurgeDetector."""
        logger.info("CRT PurgeDetector initialized")

    def detect_purge(
        self,
        crt_range: CRTRange,
        ltf_candles: List[dict],
        avg_volume: Optional[float] = None,
    ) -> Optional[PurgeEvent]:
        """
        Detect a purge event in the LTF candles.

        Args:
            crt_range: The CRT range to check against
            ltf_candles: List of lower timeframe candles (M5/M15)
                         Each candle: {datetime, open, high, low, close, volume}
            avg_volume: Average volume for comparison

        Returns:
            PurgeEvent if purge detected, None otherwise
        """
        if not ltf_candles:
            return None

        # Already has a purge
        if crt_range.state in [CRTState.PURGE_ABOVE, CRTState.PURGE_BELOW]:
            return self._get_existing_purge(crt_range, ltf_candles, avg_volume)

        # Check for new purge
        highest_above = None
        lowest_below = None
        purge_candle_above = None
        purge_candle_below = None

        for i, candle in enumerate(ltf_candles):
            high = candle.get("high", 0)
            low = candle.get("low", 0)

            # Check purge above
            if high > crt_range.crt_high:
                if highest_above is None or high > highest_above:
                    highest_above = high
                    purge_candle_above = (i, candle)

            # Check purge below
            if low < crt_range.crt_low:
                if lowest_below is None or low < lowest_below:
                    lowest_below = low
                    purge_candle_below = (i, candle)

        # Determine which purge to return (if any)
        purge_above_percent = 0
        purge_below_percent = 0

        if highest_above:
            purge_above_percent = ((highest_above - crt_range.crt_high) / crt_range.crt_high) * 100

        if lowest_below:
            purge_below_percent = ((crt_range.crt_low - lowest_below) / crt_range.crt_low) * 100

        # Return the larger purge if it meets threshold
        if purge_above_percent >= self.PURGE_MIN_PERCENT and purge_above_percent >= purge_below_percent:
            idx, candle = purge_candle_above
            volume = candle.get("volume", 0)
            volume_ratio = (volume / avg_volume) if avg_volume and avg_volume > 0 else None

            return PurgeEvent(
                direction="above",
                purge_price=highest_above,
                purge_wick=highest_above - crt_range.crt_high,
                purge_wick_percent=purge_above_percent,
                purge_time=self._parse_candle_time(candle),
                candles_since_range=idx,
                volume_at_purge=volume,
                volume_ratio=volume_ratio,
            )

        if purge_below_percent >= self.PURGE_MIN_PERCENT:
            idx, candle = purge_candle_below
            volume = candle.get("volume", 0)
            volume_ratio = (volume / avg_volume) if avg_volume and avg_volume > 0 else None

            return PurgeEvent(
                direction="below",
                purge_price=lowest_below,
                purge_wick=crt_range.crt_low - lowest_below,
                purge_wick_percent=purge_below_percent,
                purge_time=self._parse_candle_time(candle),
                candles_since_range=idx,
                volume_at_purge=volume,
                volume_ratio=volume_ratio,
            )

        return None

    def _get_existing_purge(
        self,
        crt_range: CRTRange,
        ltf_candles: List[dict],
        avg_volume: Optional[float],
    ) -> Optional[PurgeEvent]:
        """Reconstruct purge event from existing range state."""
        if not crt_range.purge_price or not crt_range.purge_direction:
            return None

        if crt_range.purge_direction == "above":
            wick = crt_range.purge_price - crt_range.crt_high
            wick_percent = (wick / crt_range.crt_high) * 100
        else:
            wick = crt_range.crt_low - crt_range.purge_price
            wick_percent = (wick / crt_range.crt_low) * 100

        return PurgeEvent(
            direction=crt_range.purge_direction,
            purge_price=crt_range.purge_price,
            purge_wick=wick,
            purge_wick_percent=wick_percent,
            purge_time=crt_range.purge_time or datetime.now(UTC_TZ),
            candles_since_range=0,
            volume_at_purge=None,
            volume_ratio=None,
        )

    def detect_reentry(
        self,
        crt_range: CRTRange,
        purge: PurgeEvent,
        ltf_candles: List[dict],
        tp2_level: Optional[float] = None,
    ) -> Optional[ReEntryEvent]:
        """
        Detect a re-entry into the CRT range after purge.

        Args:
            crt_range: The CRT range
            purge: The purge event that occurred
            ltf_candles: List of candles to check for re-entry
            tp2_level: Optional extended take profit level

        Returns:
            ReEntryEvent if re-entry detected, None otherwise
        """
        if not ltf_candles:
            return None

        # Find candle that closes back inside range
        for candle in ltf_candles:
            close = candle.get("close", 0)

            # Check if close is inside range
            if crt_range.crt_low <= close <= crt_range.crt_high:
                # Valid re-entry!
                return self._create_reentry_event(
                    crt_range=crt_range,
                    purge=purge,
                    reentry_candle=candle,
                    tp2_level=tp2_level,
                )

        return None

    def _create_reentry_event(
        self,
        crt_range: CRTRange,
        purge: PurgeEvent,
        reentry_candle: dict,
        tp2_level: Optional[float],
    ) -> ReEntryEvent:
        """Create a ReEntryEvent with calculated trade parameters."""
        close = reentry_candle.get("close", 0)
        reentry_time = self._parse_candle_time(reentry_candle)

        if purge.direction == "above":
            # Short signal after purge above
            direction = "short"
            entry_price = close
            stop_loss = purge.purge_price * (1 + self.STOP_BUFFER_PERCENT / 100)
            take_profit_1 = crt_range.crt_low
            take_profit_2 = tp2_level if tp2_level and tp2_level < take_profit_1 else None

            risk = stop_loss - entry_price
            reward_1 = entry_price - take_profit_1
            reward_2 = (entry_price - take_profit_2) if take_profit_2 else None

        else:
            # Long signal after purge below
            direction = "long"
            entry_price = close
            stop_loss = purge.purge_price * (1 - self.STOP_BUFFER_PERCENT / 100)
            take_profit_1 = crt_range.crt_high
            take_profit_2 = tp2_level if tp2_level and tp2_level > take_profit_1 else None

            risk = entry_price - stop_loss
            reward_1 = take_profit_1 - entry_price
            reward_2 = (take_profit_2 - entry_price) if take_profit_2 else None

        # Calculate R:R
        rr_1 = reward_1 / risk if risk > 0 else 0
        rr_2 = (reward_2 / risk) if reward_2 and risk > 0 else None

        # Calculate confidence
        confidence = self._calculate_confidence(crt_range, purge, rr_1)

        return ReEntryEvent(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_reward_1=rr_1,
            risk_reward_2=rr_2,
            risk_amount=risk,
            confidence=confidence,
            reentry_time=reentry_time,
            reentry_candle_close=close,
        )

    def _calculate_confidence(
        self,
        crt_range: CRTRange,
        purge: PurgeEvent,
        risk_reward: float,
    ) -> float:
        """
        Calculate signal confidence based on multiple factors.

        Factors:
        - Risk/Reward ratio
        - Purge wick size (deeper = better)
        - Volume spike
        - Range age (fresher = better)
        """
        confidence = 0.5  # Base confidence

        # R:R bonus (up to +0.2)
        if risk_reward >= 2.0:
            confidence += 0.2
        elif risk_reward >= 1.5:
            confidence += 0.15
        elif risk_reward >= 1.0:
            confidence += 0.1

        # Purge wick bonus (up to +0.15)
        if purge.purge_wick_percent >= 0.5:
            confidence += 0.15
        elif purge.purge_wick_percent >= 0.2:
            confidence += 0.1
        elif purge.purge_wick_percent >= 0.1:
            confidence += 0.05

        # Volume spike bonus (up to +0.1)
        if purge.volume_ratio and purge.volume_ratio >= self.VOLUME_SPIKE_THRESHOLD:
            confidence += 0.1

        # Range freshness bonus (up to +0.05)
        if crt_range.age_hours <= 4:
            confidence += 0.05
        elif crt_range.age_hours <= 8:
            confidence += 0.025

        return min(confidence, 1.0)  # Cap at 1.0

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

    def is_valid_signal(self, reentry: ReEntryEvent) -> bool:
        """
        Check if re-entry signal meets minimum criteria.

        Args:
            reentry: The re-entry event

        Returns:
            True if signal is valid
        """
        # Minimum R:R check
        if reentry.risk_reward_1 < self.MIN_RISK_REWARD:
            return False

        # Risk must be positive
        if reentry.risk_amount <= 0:
            return False

        return True


# Singleton instance
purge_detector = PurgeDetector()
