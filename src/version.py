"""Version configuration for KI Trading Model."""

from datetime import datetime

# Release version - update this for each release
VERSION = "1.0.0"

# Release date/time - update this for each release
RELEASE_DATE = "2025-11-29T12:00:00"


def get_version_info() -> dict:
    """Get version information as dictionary."""
    return {
        "version": f"v{VERSION}",
        "release_date": RELEASE_DATE,
        "release_date_formatted": datetime.fromisoformat(RELEASE_DATE).strftime("%d.%m.%Y %H:%M")
    }
