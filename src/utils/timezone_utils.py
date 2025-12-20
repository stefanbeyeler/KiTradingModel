"""Timezone utilities for consistent timestamp handling.

All timestamps are stored internally in UTC and converted to the configured
display timezone for presentation.

Usage:
    from src.utils.timezone_utils import to_utc, format_for_display

    # Parse and normalize to UTC
    utc_time = to_utc("2024-01-15T10:30:45Z")
    utc_time = to_utc("2024-01-15T11:30:45+01:00")

    # Format for display in configured timezone
    display_str = format_for_display(utc_time)  # "15.01.2024, 11:30:45 CET"
"""

from datetime import datetime, timezone
from typing import Optional, Union
from zoneinfo import ZoneInfo

from src.config.settings import settings


def get_display_timezone() -> ZoneInfo:
    """Get the configured display timezone."""
    return ZoneInfo(settings.display_timezone)


def get_current_utc() -> datetime:
    """Get the current time in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)


def parse_timestamp(timestamp: Union[str, datetime, None]) -> Optional[datetime]:
    """Parse a timestamp string into a timezone-aware datetime.

    Handles various formats:
    - ISO 8601 with Z suffix: "2024-01-15T10:30:45Z"
    - ISO 8601 with offset: "2024-01-15T10:30:45+01:00"
    - ISO 8601 without timezone: "2024-01-15T10:30:45" (assumed UTC)
    - TwelveData format: "2024-01-15 10:30:45" (assumed UTC)

    Args:
        timestamp: String or datetime to parse

    Returns:
        Timezone-aware datetime in its original timezone, or None if parsing fails
    """
    if timestamp is None:
        return None

    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp

    if not isinstance(timestamp, str):
        return None

    ts = timestamp.strip()

    try:
        # Handle Z suffix (UTC)
        if ts.endswith('Z'):
            ts = ts[:-1] + '+00:00'

        # Handle space separator (TwelveData format)
        if ' ' in ts and 'T' not in ts:
            ts = ts.replace(' ', 'T')

        # Parse ISO format
        parsed = datetime.fromisoformat(ts)

        # If no timezone info, assume UTC
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    except (ValueError, TypeError):
        return None


def to_utc(timestamp: Union[str, datetime, None]) -> Optional[datetime]:
    """Convert a timestamp to UTC.

    Args:
        timestamp: String or datetime to convert

    Returns:
        Timezone-aware datetime in UTC, or None if parsing fails
    """
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return None

    return parsed.astimezone(timezone.utc)


def to_display_timezone(timestamp: Union[str, datetime, None]) -> Optional[datetime]:
    """Convert a timestamp to the configured display timezone.

    Args:
        timestamp: String or datetime to convert

    Returns:
        Timezone-aware datetime in display timezone, or None if parsing fails
    """
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return None

    return parsed.astimezone(get_display_timezone())


def format_for_display(
    timestamp: Union[str, datetime, None],
    include_timezone: bool = True,
    format_override: Optional[str] = None
) -> Optional[str]:
    """Format a timestamp for display in the configured timezone.

    Args:
        timestamp: String or datetime to format
        include_timezone: Whether to append timezone abbreviation (e.g., "CET")
        format_override: Override the default datetime format

    Returns:
        Formatted string in display timezone, or None if parsing fails
    """
    local_time = to_display_timezone(timestamp)
    if local_time is None:
        return None

    fmt = format_override or settings.datetime_format
    formatted = local_time.strftime(fmt)

    if include_timezone:
        tz_abbr = local_time.strftime('%Z')
        formatted = f"{formatted} {tz_abbr}"

    return formatted


def format_utc_iso(timestamp: Union[str, datetime, None]) -> Optional[str]:
    """Format a timestamp as ISO 8601 UTC string.

    Args:
        timestamp: String or datetime to format

    Returns:
        ISO 8601 formatted string with Z suffix, or None if parsing fails
    """
    utc_time = to_utc(timestamp)
    if utc_time is None:
        return None

    return utc_time.strftime('%Y-%m-%dT%H:%M:%SZ')


def get_timezone_info() -> dict:
    """Get information about the current timezone configuration.

    Returns:
        Dictionary with timezone configuration details
    """
    tz = get_display_timezone()
    now_utc = get_current_utc()
    now_local = now_utc.astimezone(tz)

    return {
        "timezone": settings.display_timezone,
        "abbreviation": now_local.strftime('%Z'),
        "utc_offset": now_local.strftime('%z'),
        "utc_offset_hours": now_local.utcoffset().total_seconds() / 3600 if now_local.utcoffset() else 0,
        "current_utc": format_utc_iso(now_utc),
        "current_local": format_for_display(now_utc),
        "date_format": settings.date_format,
        "time_format": settings.time_format,
        "datetime_format": settings.datetime_format,
    }
