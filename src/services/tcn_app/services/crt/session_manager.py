"""Session Manager for CRT Detection.

Manages CRT-relevant session times and H4 candle tracking.
Key sessions: 1 AM, 5 AM, 9 AM EST (London Pre, London, NY Open).
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from loguru import logger

from .session_utils import (
    SessionType,
    get_est_time,
    get_h4_candle_start,
    get_h4_candle_end,
    get_previous_h4_candle,
    is_key_session_hour,
    get_current_session,
    get_session_for_h4_candle,
    get_next_key_session,
    get_key_session_h4_candles,
    is_h4_candle_complete,
    get_session_info,
    UTC_TZ,
)


class SessionManager:
    """
    Manages CRT session tracking and H4 candle monitoring.

    Responsibilities:
    - Track current session type
    - Identify key session H4 candles
    - Provide session timing information
    """

    def __init__(self):
        """Initialize SessionManager."""
        self._last_h4_start: Optional[datetime] = None
        self._current_session: Optional[SessionType] = None
        logger.info("CRT SessionManager initialized")

    def get_current_time_utc(self) -> datetime:
        """Get current UTC time."""
        return datetime.now(UTC_TZ)

    def get_session_info(self, timestamp: Optional[datetime] = None) -> dict:
        """
        Get comprehensive session information.

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            Dict with session details
        """
        return get_session_info(timestamp)

    def get_current_session(self, timestamp: Optional[datetime] = None) -> SessionType:
        """
        Get the current session type.

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            SessionType enum
        """
        return get_current_session(timestamp)

    def is_key_session_active(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if we're currently in a key session hour.

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            True if at 1 AM, 5 AM, or 9 AM EST
        """
        return is_key_session_hour(timestamp)

    def get_current_h4_candle(self, timestamp: Optional[datetime] = None) -> dict:
        """
        Get current H4 candle boundaries.

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            Dict with start, end, is_complete, session_type
        """
        if timestamp is None:
            timestamp = self.get_current_time_utc()

        h4_start = get_h4_candle_start(timestamp)
        h4_end = get_h4_candle_end(timestamp)
        session_type = get_session_for_h4_candle(h4_start)

        return {
            "start_utc": h4_start,
            "end_utc": h4_end,
            "is_complete": is_h4_candle_complete(h4_start, timestamp),
            "session_type": session_type,
            "is_key_session_candle": session_type is not None,
        }

    def get_last_completed_h4_candle(self, timestamp: Optional[datetime] = None) -> dict:
        """
        Get the last completed H4 candle boundaries.

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            Dict with start, end, session_type
        """
        if timestamp is None:
            timestamp = self.get_current_time_utc()

        prev_start, prev_end = get_previous_h4_candle(timestamp)
        session_type = get_session_for_h4_candle(prev_start)

        return {
            "start_utc": prev_start,
            "end_utc": prev_end,
            "session_type": session_type,
            "is_key_session_candle": session_type is not None,
        }

    def get_next_key_session(self, timestamp: Optional[datetime] = None) -> dict:
        """
        Get information about the next key session.

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            Dict with session_type, times, hours_until
        """
        return get_next_key_session(timestamp)

    def get_todays_key_sessions(self, date: Optional[datetime] = None) -> list[dict]:
        """
        Get all key session H4 candles for today.

        Args:
            date: Date to check (default: today)

        Returns:
            List of key session candle info
        """
        return get_key_session_h4_candles(date)

    def should_create_range(self, timestamp: Optional[datetime] = None) -> dict:
        """
        Determine if a new CRT range should be created now.

        A range should be created when:
        1. A key session H4 candle has just completed
        2. We don't already have an active range for this session

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            Dict with should_create, reason, session_type, h4_candle
        """
        if timestamp is None:
            timestamp = self.get_current_time_utc()

        # Get current H4 candle
        current_h4 = self.get_current_h4_candle(timestamp)

        # Check if we just completed an H4 candle
        if current_h4["is_complete"]:
            # Actually check the current (just completed) candle
            h4_start = current_h4["start_utc"]
        else:
            # Check the previous completed candle
            prev_h4 = self.get_last_completed_h4_candle(timestamp)
            h4_start = prev_h4["start_utc"]

        session_type = get_session_for_h4_candle(h4_start)

        if session_type is not None:
            return {
                "should_create": True,
                "reason": f"Key session candle completed: {session_type.value}",
                "session_type": session_type,
                "h4_candle_start": h4_start,
                "h4_candle_end": h4_start + timedelta(hours=4),
            }

        return {
            "should_create": False,
            "reason": "Not a key session H4 candle",
            "session_type": None,
            "h4_candle_start": h4_start,
            "h4_candle_end": h4_start + timedelta(hours=4),
        }

    def is_within_trading_hours(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if within active trading hours (not weekend).

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            True if market is likely open
        """
        if timestamp is None:
            timestamp = self.get_current_time_utc()

        est_time = get_est_time(timestamp)

        # Check weekend
        if est_time.weekday() >= 5:
            # Sunday after 5 PM EST is OK (futures open)
            if est_time.weekday() == 6 and est_time.hour >= 17:
                return True
            # Saturday before 5 PM EST might still have some markets
            if est_time.weekday() == 5 and est_time.hour < 17:
                return True
            return False

        return True

    def get_est_hour(self, timestamp: Optional[datetime] = None) -> int:
        """
        Get current EST hour.

        Args:
            timestamp: UTC datetime (default: current time)

        Returns:
            Hour in EST (0-23)
        """
        if timestamp is None:
            timestamp = self.get_current_time_utc()

        return get_est_time(timestamp).hour


# Singleton instance
session_manager = SessionManager()
