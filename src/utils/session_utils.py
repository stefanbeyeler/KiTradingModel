"""Session time utilities for CRT (Candle Range Theory) detection.

Handles EST/EDT timezone conversions and H4 candle boundary calculations
for the key CRT session times: 1 AM, 5 AM, 9 AM EST.

Usage:
    from src.utils.session_utils import (
        get_h4_candle_start,
        get_current_session,
        is_key_session_hour,
        get_next_key_session,
    )
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo
from enum import Enum


class SessionType(str, Enum):
    """CRT Session Types based on market opens."""
    LONDON_PRE = "london_pre"    # 1 AM EST - London Pre-Market
    LONDON = "london"            # 5 AM EST - London Open
    NEW_YORK = "new_york"        # 9 AM EST - New York Open
    ASIAN = "asian"              # Outside key sessions
    OFF_HOURS = "off_hours"      # Weekend/holidays


# EST/EDT Timezone
EST_TZ = ZoneInfo("America/New_York")
UTC_TZ = timezone.utc

# CRT Key Session Hours (in EST/EDT local time)
CRT_KEY_HOURS_EST = {
    1: SessionType.LONDON_PRE,   # 1 AM EST
    5: SessionType.LONDON,       # 5 AM EST
    9: SessionType.NEW_YORK,     # 9 AM EST
}

# H4 Candle boundaries (UTC hours)
H4_BOUNDARIES_UTC = [0, 4, 8, 12, 16, 20]


def get_est_time(utc_time: Optional[datetime] = None) -> datetime:
    """
    Convert UTC time to EST/EDT (handles DST automatically).

    Args:
        utc_time: UTC datetime (default: current time)

    Returns:
        datetime in EST/EDT timezone
    """
    if utc_time is None:
        utc_time = datetime.now(UTC_TZ)
    elif utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=UTC_TZ)

    return utc_time.astimezone(EST_TZ)


def get_utc_time(est_time: datetime) -> datetime:
    """
    Convert EST/EDT time to UTC.

    Args:
        est_time: datetime in EST/EDT timezone

    Returns:
        datetime in UTC
    """
    if est_time.tzinfo is None:
        est_time = est_time.replace(tzinfo=EST_TZ)

    return est_time.astimezone(UTC_TZ)


def get_h4_candle_start(timestamp: Optional[datetime] = None) -> datetime:
    """
    Get the start time of the current H4 candle.

    H4 candles start at: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC

    Args:
        timestamp: UTC datetime (default: current time)

    Returns:
        Start of current H4 candle in UTC
    """
    if timestamp is None:
        timestamp = datetime.now(UTC_TZ)
    elif timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC_TZ)

    # Convert to UTC for calculation
    utc_time = timestamp.astimezone(UTC_TZ)

    # Find the most recent H4 boundary
    current_hour = utc_time.hour
    h4_start_hour = (current_hour // 4) * 4

    return utc_time.replace(
        hour=h4_start_hour,
        minute=0,
        second=0,
        microsecond=0
    )


def get_h4_candle_end(timestamp: Optional[datetime] = None) -> datetime:
    """
    Get the end time of the current H4 candle.

    Args:
        timestamp: UTC datetime (default: current time)

    Returns:
        End of current H4 candle in UTC
    """
    start = get_h4_candle_start(timestamp)
    return start + timedelta(hours=4)


def get_previous_h4_candle(timestamp: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Get the start and end times of the previous (completed) H4 candle.

    Args:
        timestamp: UTC datetime (default: current time)

    Returns:
        Tuple of (start, end) for previous H4 candle in UTC
    """
    current_start = get_h4_candle_start(timestamp)
    prev_start = current_start - timedelta(hours=4)
    prev_end = current_start

    return prev_start, prev_end


def is_key_session_hour(timestamp: Optional[datetime] = None) -> bool:
    """
    Check if the current time is at a CRT key session hour.

    Key sessions: 1 AM, 5 AM, 9 AM EST

    Args:
        timestamp: UTC datetime (default: current time)

    Returns:
        True if current hour is a key session hour
    """
    if timestamp is None:
        timestamp = datetime.now(UTC_TZ)

    est_time = get_est_time(timestamp)
    return est_time.hour in CRT_KEY_HOURS_EST


def get_current_session(timestamp: Optional[datetime] = None) -> SessionType:
    """
    Get the current session type based on EST time.

    Args:
        timestamp: UTC datetime (default: current time)

    Returns:
        SessionType enum value
    """
    if timestamp is None:
        timestamp = datetime.now(UTC_TZ)

    est_time = get_est_time(timestamp)
    hour = est_time.hour

    # Check if weekend
    if est_time.weekday() >= 5:  # Saturday or Sunday
        return SessionType.OFF_HOURS

    # Determine session based on EST hour
    if hour in CRT_KEY_HOURS_EST:
        return CRT_KEY_HOURS_EST[hour]
    elif 0 <= hour < 1:
        return SessionType.ASIAN
    elif 1 <= hour < 5:
        return SessionType.LONDON_PRE
    elif 5 <= hour < 9:
        return SessionType.LONDON
    elif 9 <= hour < 17:
        return SessionType.NEW_YORK
    else:
        return SessionType.ASIAN


def get_session_for_h4_candle(h4_start: datetime) -> Optional[SessionType]:
    """
    Determine which key session (if any) an H4 candle belongs to.

    An H4 candle is considered a "key session candle" if it contains
    one of the key session hours (1 AM, 5 AM, 9 AM EST).

    Args:
        h4_start: Start time of H4 candle (UTC)

    Returns:
        SessionType if this is a key session candle, None otherwise
    """
    h4_end = h4_start + timedelta(hours=4)

    # Check each hour within the H4 candle
    for hour_offset in range(4):
        check_time = h4_start + timedelta(hours=hour_offset)
        est_time = get_est_time(check_time)

        if est_time.hour in CRT_KEY_HOURS_EST:
            return CRT_KEY_HOURS_EST[est_time.hour]

    return None


def get_next_key_session(timestamp: Optional[datetime] = None) -> dict:
    """
    Get information about the next key CRT session.

    Args:
        timestamp: UTC datetime (default: current time)

    Returns:
        Dict with session_type, start_time_utc, start_time_est, hours_until
    """
    if timestamp is None:
        timestamp = datetime.now(UTC_TZ)

    est_time = get_est_time(timestamp)
    current_hour = est_time.hour

    # Find next key hour
    key_hours = sorted(CRT_KEY_HOURS_EST.keys())

    next_hour = None
    days_ahead = 0

    # Check if any key hour is still coming today
    for h in key_hours:
        if h > current_hour:
            next_hour = h
            break

    # If no key hour today, get first one tomorrow
    if next_hour is None:
        next_hour = key_hours[0]
        days_ahead = 1

    # Skip weekends
    next_date = est_time.date() + timedelta(days=days_ahead)
    while next_date.weekday() >= 5:  # Saturday or Sunday
        next_date += timedelta(days=1)
        days_ahead += 1

    # Construct next session time
    next_session_est = datetime(
        next_date.year,
        next_date.month,
        next_date.day,
        next_hour,
        0,
        0,
        tzinfo=EST_TZ
    )

    next_session_utc = get_utc_time(next_session_est)

    # Calculate hours until
    hours_until = (next_session_utc - timestamp).total_seconds() / 3600

    return {
        "session_type": CRT_KEY_HOURS_EST[next_hour].value,
        "start_time_utc": next_session_utc.isoformat(),
        "start_time_est": next_session_est.strftime("%Y-%m-%d %H:%M EST"),
        "hours_until": round(hours_until, 2),
        "h4_candle_start": get_h4_candle_start(next_session_utc).isoformat(),
    }


def get_key_session_h4_candles(date: Optional[datetime] = None) -> list[dict]:
    """
    Get all H4 candles that contain key sessions for a given date.

    Args:
        date: Date to check (default: today)

    Returns:
        List of dicts with h4_start, h4_end, session_type
    """
    if date is None:
        date = datetime.now(UTC_TZ)

    # Get the date in EST
    est_date = get_est_time(date).date()

    # Skip weekends
    if est_date.weekday() >= 5:
        return []

    results = []

    for hour, session_type in CRT_KEY_HOURS_EST.items():
        # Create EST datetime for this key hour
        key_time_est = datetime(
            est_date.year,
            est_date.month,
            est_date.day,
            hour,
            0,
            0,
            tzinfo=EST_TZ
        )

        # Convert to UTC
        key_time_utc = get_utc_time(key_time_est)

        # Get H4 candle containing this time
        h4_start = get_h4_candle_start(key_time_utc)
        h4_end = h4_start + timedelta(hours=4)

        results.append({
            "session_type": session_type.value,
            "key_hour_est": hour,
            "key_time_utc": key_time_utc.isoformat(),
            "h4_start_utc": h4_start.isoformat(),
            "h4_end_utc": h4_end.isoformat(),
        })

    return results


def is_h4_candle_complete(h4_start: datetime, current_time: Optional[datetime] = None) -> bool:
    """
    Check if an H4 candle is complete (closed).

    Args:
        h4_start: Start time of H4 candle
        current_time: Current time (default: now)

    Returns:
        True if the candle has closed
    """
    if current_time is None:
        current_time = datetime.now(UTC_TZ)

    h4_end = h4_start + timedelta(hours=4)
    return current_time >= h4_end


def get_session_info(timestamp: Optional[datetime] = None) -> dict:
    """
    Get comprehensive session information for the given time.

    Args:
        timestamp: UTC datetime (default: current time)

    Returns:
        Dict with all session-related information
    """
    if timestamp is None:
        timestamp = datetime.now(UTC_TZ)

    est_time = get_est_time(timestamp)
    h4_start = get_h4_candle_start(timestamp)
    h4_end = get_h4_candle_end(timestamp)
    current_session = get_current_session(timestamp)
    h4_session = get_session_for_h4_candle(h4_start)
    next_session = get_next_key_session(timestamp)

    return {
        "current_time_utc": timestamp.isoformat(),
        "current_time_est": est_time.strftime("%Y-%m-%d %H:%M:%S EST"),
        "current_session": current_session.value,
        "is_key_session_hour": is_key_session_hour(timestamp),
        "h4_candle": {
            "start_utc": h4_start.isoformat(),
            "end_utc": h4_end.isoformat(),
            "is_complete": is_h4_candle_complete(h4_start, timestamp),
            "session_type": h4_session.value if h4_session else None,
            "is_key_session_candle": h4_session is not None,
        },
        "next_key_session": next_session,
        "is_weekend": est_time.weekday() >= 5,
    }
