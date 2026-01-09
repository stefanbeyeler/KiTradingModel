"""Database configuration for TimescaleDB.

This module provides database-specific configuration including:
- Chunk intervals for hypertables
- Retention policies
- Compression settings
- Table configurations
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TableType(str, Enum):
    """Types of tables in the database."""
    OHLCV = "ohlcv"
    INDICATORS = "indicators"
    INDICATORS_MA = "indicators_ma"
    INDICATORS_MOMENTUM = "indicators_momentum"
    INDICATORS_VOLATILITY = "indicators_volatility"
    INDICATORS_TREND = "indicators_trend"
    INDICATORS_VOLUME = "indicators_volume"
    INDICATORS_LEVELS = "indicators_levels"
    MARKET_SNAPSHOTS = "market_snapshots"
    SYMBOLS = "symbols"
    DATA_FRESHNESS = "data_freshness"
    ECONOMIC_EVENTS = "economic_events"
    SENTIMENT_DATA = "sentiment_data"
    ONCHAIN_DATA = "onchain_data"


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe table."""
    table_suffix: str        # e.g., "m1", "h1", "d1"
    chunk_interval: str      # e.g., "1 day", "7 days"
    retention_days: Optional[int]  # None = no retention policy
    compression_after_days: int


# Timeframe-specific OHLCV table configurations
OHLCV_CONFIGS = {
    "M1": TimeframeConfig("m1", "1 day", 30, 7),
    "M5": TimeframeConfig("m5", "1 day", 90, 7),
    "M15": TimeframeConfig("m15", "7 days", 180, 14),
    "M30": TimeframeConfig("m30", "7 days", 180, 14),
    "H1": TimeframeConfig("h1", "7 days", None, 30),
    "H4": TimeframeConfig("h4", "30 days", None, 60),
    "D1": TimeframeConfig("d1", "365 days", None, 90),
    "W1": TimeframeConfig("w1", "365 days", None, 90),
    "MN": TimeframeConfig("mn", "365 days", None, 90),
}


# Indicator table chunk intervals
INDICATOR_CHUNK_INTERVALS = {
    TableType.INDICATORS: "7 days",
    TableType.INDICATORS_MA: "7 days",
    TableType.INDICATORS_MOMENTUM: "7 days",
    TableType.INDICATORS_VOLATILITY: "7 days",
    TableType.INDICATORS_TREND: "7 days",
    TableType.INDICATORS_VOLUME: "7 days",
    TableType.INDICATORS_LEVELS: "30 days",
}


# Retention policies (in days) for indicator tables
INDICATOR_RETENTION = {
    TableType.INDICATORS: 30,           # JSONB table - shorter retention
    TableType.INDICATORS_MA: 180,
    TableType.INDICATORS_MOMENTUM: 180,
    TableType.INDICATORS_VOLATILITY: 180,
    TableType.INDICATORS_TREND: 180,
    TableType.INDICATORS_VOLUME: 180,
    TableType.INDICATORS_LEVELS: 365,   # Pivot points - longer for backtesting
}


# Compression policies (after X days)
INDICATOR_COMPRESSION = {
    TableType.INDICATORS: 7,
    TableType.INDICATORS_MA: 7,
    TableType.INDICATORS_MOMENTUM: 7,
    TableType.INDICATORS_VOLATILITY: 7,
    TableType.INDICATORS_TREND: 7,
    TableType.INDICATORS_VOLUME: 7,
    TableType.INDICATORS_LEVELS: 30,
}


def get_ohlcv_table_name(timeframe: str) -> str:
    """Get the OHLCV table name for a timeframe.

    Args:
        timeframe: Standard timeframe (M1, H1, D1, etc.)

    Returns:
        Table name (e.g., "ohlcv_h1")
    """
    config = OHLCV_CONFIGS.get(timeframe.upper())
    if config:
        return f"ohlcv_{config.table_suffix}"
    raise ValueError(f"Unknown timeframe: {timeframe}")


def get_chunk_interval(timeframe: str) -> str:
    """Get the chunk interval for a timeframe.

    Args:
        timeframe: Standard timeframe (M1, H1, D1, etc.)

    Returns:
        Chunk interval string (e.g., "7 days")
    """
    config = OHLCV_CONFIGS.get(timeframe.upper())
    if config:
        return config.chunk_interval
    raise ValueError(f"Unknown timeframe: {timeframe}")


# All supported timeframes
SUPPORTED_TIMEFRAMES = list(OHLCV_CONFIGS.keys())


# Freshness thresholds (how old data can be before refresh is needed)
FRESHNESS_THRESHOLDS = {
    "M1": 120,      # 2 minutes
    "M5": 600,      # 10 minutes
    "M15": 1800,    # 30 minutes
    "M30": 3600,    # 1 hour
    "H1": 7200,     # 2 hours
    "H4": 28800,    # 8 hours
    "D1": 86400,    # 1 day
    "W1": 604800,   # 1 week
    "MN": 2592000,  # 30 days
}
