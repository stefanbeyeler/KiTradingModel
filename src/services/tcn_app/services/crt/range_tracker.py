"""CRT Range Tracker - State Machine for Candle Range Theory.

Manages the lifecycle of CRT ranges:
WAITING -> RANGE_DEFINED -> PURGE_ABOVE/BELOW -> SIGNAL_LONG/SHORT -> INVALIDATED
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Dict
from loguru import logger

from src.utils.session_utils import SessionType, UTC_TZ


class CRTState(str, Enum):
    """CRT State Machine States."""
    WAITING = "waiting"               # Waiting for key session candle
    RANGE_DEFINED = "range_defined"   # Range defined, waiting for purge
    PURGE_ABOVE = "purge_above"       # Purge above CRT High detected
    PURGE_BELOW = "purge_below"       # Purge below CRT Low detected
    SIGNAL_LONG = "signal_long"       # Long signal (re-entry after purge below)
    SIGNAL_SHORT = "signal_short"     # Short signal (re-entry after purge above)
    INVALIDATED = "invalidated"       # Range invalidated (timeout or displacement)


@dataclass
class CRTRange:
    """CRT Range Definition and State."""
    symbol: str
    session_type: SessionType
    candle_start: datetime          # H4 Candle start time (UTC)
    candle_end: datetime            # H4 Candle end time (UTC)
    crt_high: float                 # Range High
    crt_low: float                  # Range Low
    crt_open: float                 # Candle Open
    crt_close: float                # Candle Close
    volume: float                   # Candle Volume
    state: CRTState = CRTState.RANGE_DEFINED
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC_TZ))

    # Purge tracking
    purge_price: Optional[float] = None
    purge_time: Optional[datetime] = None
    purge_direction: Optional[str] = None  # "above" or "below"

    # Signal tracking
    signal_price: Optional[float] = None
    signal_time: Optional[datetime] = None
    signal_direction: Optional[str] = None  # "long" or "short"

    # Invalidation
    invalidation_reason: Optional[str] = None
    invalidated_at: Optional[datetime] = None

    @property
    def range_size(self) -> float:
        """Calculate range size in price units."""
        return self.crt_high - self.crt_low

    @property
    def range_percent(self) -> float:
        """Calculate range size as percentage of price."""
        mid = (self.crt_high + self.crt_low) / 2
        return (self.range_size / mid) * 100 if mid > 0 else 0

    @property
    def is_bullish_candle(self) -> bool:
        """Check if the defining candle was bullish."""
        return self.crt_close > self.crt_open

    @property
    def age_hours(self) -> float:
        """Get age of range in hours."""
        now = datetime.now(UTC_TZ)
        return (now - self.created_at).total_seconds() / 3600

    @property
    def is_active(self) -> bool:
        """Check if range is still active (not invalidated or signaled)."""
        return self.state in [CRTState.RANGE_DEFINED, CRTState.PURGE_ABOVE, CRTState.PURGE_BELOW]

    @property
    def has_signal(self) -> bool:
        """Check if range has generated a signal."""
        return self.state in [CRTState.SIGNAL_LONG, CRTState.SIGNAL_SHORT]

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "symbol": self.symbol,
            "session_type": self.session_type.value,
            "candle_start": self.candle_start.isoformat(),
            "candle_end": self.candle_end.isoformat(),
            "crt_high": self.crt_high,
            "crt_low": self.crt_low,
            "crt_open": self.crt_open,
            "crt_close": self.crt_close,
            "volume": self.volume,
            "range_size": self.range_size,
            "range_percent": round(self.range_percent, 3),
            "is_bullish_candle": self.is_bullish_candle,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "age_hours": round(self.age_hours, 2),
            "purge": {
                "price": self.purge_price,
                "time": self.purge_time.isoformat() if self.purge_time else None,
                "direction": self.purge_direction,
            } if self.purge_price else None,
            "signal": {
                "price": self.signal_price,
                "time": self.signal_time.isoformat() if self.signal_time else None,
                "direction": self.signal_direction,
            } if self.signal_price else None,
            "invalidation": {
                "reason": self.invalidation_reason,
                "time": self.invalidated_at.isoformat() if self.invalidated_at else None,
            } if self.invalidation_reason else None,
        }


class RangeTracker:
    """
    Manages active CRT ranges for all symbols.

    Rules:
    - Only ONE active range per symbol at a time
    - Ranges are invalidated after MAX_RANGE_AGE_HOURS
    - Ranges are invalidated on full displacement without re-entry
    """

    # Configuration
    MAX_RANGE_AGE_HOURS = 24.0      # Max age before auto-invalidation
    MIN_RANGE_PERCENT = 0.1        # Minimum range size (0.1%)
    MAX_RANGE_PERCENT = 5.0        # Maximum range size (5%)
    PURGE_MIN_PERCENT = 0.05       # Minimum purge beyond range (0.05%)

    def __init__(self):
        """Initialize RangeTracker."""
        self._active_ranges: Dict[str, CRTRange] = {}
        self._history: Dict[str, list[CRTRange]] = {}
        logger.info("CRT RangeTracker initialized")

    def create_range(
        self,
        symbol: str,
        session_type: SessionType,
        candle_start: datetime,
        candle_end: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0.0,
    ) -> Optional[CRTRange]:
        """
        Create a new CRT range from an H4 candle.

        Args:
            symbol: Trading symbol
            session_type: Type of session (london_pre, london, ny)
            candle_start: H4 candle start time
            candle_end: H4 candle end time
            open_price: Candle open
            high_price: Candle high (CRT High)
            low_price: Candle low (CRT Low)
            close_price: Candle close
            volume: Candle volume

        Returns:
            CRTRange if valid, None if range is invalid
        """
        # Calculate range percentage
        mid_price = (high_price + low_price) / 2
        range_percent = ((high_price - low_price) / mid_price) * 100 if mid_price > 0 else 0

        # Validate range size
        if range_percent < self.MIN_RANGE_PERCENT:
            logger.debug(f"CRT {symbol}: Range too small ({range_percent:.3f}%)")
            return None

        if range_percent > self.MAX_RANGE_PERCENT:
            logger.debug(f"CRT {symbol}: Range too large ({range_percent:.3f}%)")
            return None

        # Invalidate existing range if present
        if symbol in self._active_ranges:
            old_range = self._active_ranges[symbol]
            self._invalidate_range(symbol, "New range created")

        # Create new range
        new_range = CRTRange(
            symbol=symbol,
            session_type=session_type,
            candle_start=candle_start,
            candle_end=candle_end,
            crt_high=high_price,
            crt_low=low_price,
            crt_open=open_price,
            crt_close=close_price,
            volume=volume,
            state=CRTState.RANGE_DEFINED,
        )

        self._active_ranges[symbol] = new_range

        logger.info(
            f"CRT {symbol}: New range created - "
            f"High: {high_price:.2f}, Low: {low_price:.2f}, "
            f"Size: {range_percent:.2f}%, Session: {session_type.value}"
        )

        return new_range

    def get_active_range(self, symbol: str) -> Optional[CRTRange]:
        """
        Get the active range for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            CRTRange if active, None otherwise
        """
        crt_range = self._active_ranges.get(symbol)

        if crt_range:
            # Check for timeout
            if crt_range.age_hours > self.MAX_RANGE_AGE_HOURS:
                self._invalidate_range(symbol, "Timeout - range expired")
                return None

        return crt_range

    def update_price(
        self,
        symbol: str,
        current_price: float,
        current_time: Optional[datetime] = None,
        is_candle_close: bool = False,
    ) -> Optional[CRTRange]:
        """
        Update range state based on current price.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_time: Current time (default: now)
            is_candle_close: True if this is a candle close price

        Returns:
            Updated CRTRange or None
        """
        crt_range = self.get_active_range(symbol)
        if not crt_range:
            return None

        if current_time is None:
            current_time = datetime.now(UTC_TZ)

        # State machine transitions
        if crt_range.state == CRTState.RANGE_DEFINED:
            # Check for purge above
            if current_price > crt_range.crt_high:
                purge_percent = ((current_price - crt_range.crt_high) / crt_range.crt_high) * 100
                if purge_percent >= self.PURGE_MIN_PERCENT:
                    crt_range.state = CRTState.PURGE_ABOVE
                    crt_range.purge_price = current_price
                    crt_range.purge_time = current_time
                    crt_range.purge_direction = "above"
                    logger.info(f"CRT {symbol}: Purge ABOVE detected at {current_price:.2f}")

            # Check for purge below
            elif current_price < crt_range.crt_low:
                purge_percent = ((crt_range.crt_low - current_price) / crt_range.crt_low) * 100
                if purge_percent >= self.PURGE_MIN_PERCENT:
                    crt_range.state = CRTState.PURGE_BELOW
                    crt_range.purge_price = current_price
                    crt_range.purge_time = current_time
                    crt_range.purge_direction = "below"
                    logger.info(f"CRT {symbol}: Purge BELOW detected at {current_price:.2f}")

        elif crt_range.state == CRTState.PURGE_ABOVE:
            # Update purge high if price goes higher
            if current_price > crt_range.purge_price:
                crt_range.purge_price = current_price
                crt_range.purge_time = current_time

            # Check for re-entry (only on candle close)
            if is_candle_close and crt_range.crt_low <= current_price <= crt_range.crt_high:
                crt_range.state = CRTState.SIGNAL_SHORT
                crt_range.signal_price = current_price
                crt_range.signal_time = current_time
                crt_range.signal_direction = "short"
                logger.info(f"CRT {symbol}: SHORT SIGNAL at {current_price:.2f} (re-entry after purge above)")

        elif crt_range.state == CRTState.PURGE_BELOW:
            # Update purge low if price goes lower
            if current_price < crt_range.purge_price:
                crt_range.purge_price = current_price
                crt_range.purge_time = current_time

            # Check for re-entry (only on candle close)
            if is_candle_close and crt_range.crt_low <= current_price <= crt_range.crt_high:
                crt_range.state = CRTState.SIGNAL_LONG
                crt_range.signal_price = current_price
                crt_range.signal_time = current_time
                crt_range.signal_direction = "long"
                logger.info(f"CRT {symbol}: LONG SIGNAL at {current_price:.2f} (re-entry after purge below)")

        return crt_range

    def check_purge(self, symbol: str, price: float) -> Optional[str]:
        """
        Quick check if price has purged the range.

        Args:
            symbol: Trading symbol
            price: Price to check

        Returns:
            "above", "below", or None
        """
        crt_range = self.get_active_range(symbol)
        if not crt_range or crt_range.state != CRTState.RANGE_DEFINED:
            return None

        if price > crt_range.crt_high:
            return "above"
        elif price < crt_range.crt_low:
            return "below"

        return None

    def check_reentry(self, symbol: str, close_price: float) -> bool:
        """
        Check if price has re-entered the range.

        Args:
            symbol: Trading symbol
            close_price: Candle close price

        Returns:
            True if re-entry detected
        """
        crt_range = self.get_active_range(symbol)
        if not crt_range:
            return False

        if crt_range.state not in [CRTState.PURGE_ABOVE, CRTState.PURGE_BELOW]:
            return False

        return crt_range.crt_low <= close_price <= crt_range.crt_high

    def invalidate_range(self, symbol: str, reason: str) -> bool:
        """
        Manually invalidate a range.

        Args:
            symbol: Trading symbol
            reason: Reason for invalidation

        Returns:
            True if range was invalidated
        """
        return self._invalidate_range(symbol, reason)

    def _invalidate_range(self, symbol: str, reason: str) -> bool:
        """Internal method to invalidate a range."""
        crt_range = self._active_ranges.get(symbol)
        if not crt_range:
            return False

        crt_range.state = CRTState.INVALIDATED
        crt_range.invalidation_reason = reason
        crt_range.invalidated_at = datetime.now(UTC_TZ)

        # Move to history
        if symbol not in self._history:
            self._history[symbol] = []
        self._history[symbol].append(crt_range)

        # Keep only last 100 ranges per symbol
        if len(self._history[symbol]) > 100:
            self._history[symbol] = self._history[symbol][-100:]

        # Remove from active
        del self._active_ranges[symbol]

        logger.info(f"CRT {symbol}: Range invalidated - {reason}")
        return True

    def get_all_active_ranges(self) -> Dict[str, CRTRange]:
        """Get all active ranges."""
        # Clean up expired ranges first
        expired = []
        for symbol, crt_range in self._active_ranges.items():
            if crt_range.age_hours > self.MAX_RANGE_AGE_HOURS:
                expired.append(symbol)

        for symbol in expired:
            self._invalidate_range(symbol, "Timeout - range expired")

        return self._active_ranges.copy()

    def get_ranges_with_signals(self) -> list[CRTRange]:
        """Get all ranges that have generated signals."""
        return [r for r in self._active_ranges.values() if r.has_signal]

    def get_history(self, symbol: str, limit: int = 20) -> list[dict]:
        """
        Get range history for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum entries to return

        Returns:
            List of historical ranges as dicts
        """
        history = self._history.get(symbol, [])
        return [r.to_dict() for r in history[-limit:]]

    def get_status(self) -> dict:
        """Get overall tracker status."""
        active_ranges = self.get_all_active_ranges()

        return {
            "active_ranges_count": len(active_ranges),
            "ranges_with_purge": sum(
                1 for r in active_ranges.values()
                if r.state in [CRTState.PURGE_ABOVE, CRTState.PURGE_BELOW]
            ),
            "ranges_with_signal": sum(
                1 for r in active_ranges.values()
                if r.has_signal
            ),
            "symbols_tracked": list(active_ranges.keys()),
            "config": {
                "max_range_age_hours": self.MAX_RANGE_AGE_HOURS,
                "min_range_percent": self.MIN_RANGE_PERCENT,
                "max_range_percent": self.MAX_RANGE_PERCENT,
                "purge_min_percent": self.PURGE_MIN_PERCENT,
            }
        }


# Singleton instance
range_tracker = RangeTracker()
