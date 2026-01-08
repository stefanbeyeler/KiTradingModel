"""Data Repository - 3-Layer Caching Strategy.

This module implements the Repository Pattern for data access:
1. Redis Cache (Hot Data) - Fast access
2. TimescaleDB (Persistence) - Long-term storage
3. External APIs (Fallback) - TwelveData, EasyInsight, YFinance

The repository orchestrates these layers to provide seamless data access
with automatic caching and persistence.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import logging

from loguru import logger

from .timescaledb_service import timescaledb_service
from .cache_service import cache_service, CacheCategory
from ..config.timeframes import Timeframe, normalize_timeframe
from ..config.database import FRESHNESS_THRESHOLDS


class DataRepository:
    """
    Repository Pattern for data access.

    Implements the 3-Layer-Caching-Strategy:
    1. Redis Cache (Hot Data)
    2. TimescaleDB (Persistent Storage)
    3. External APIs (Fallback)
    """

    def __init__(self):
        self._db = timescaledb_service
        self._cache = cache_service
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize repository connections."""
        if self._initialized:
            return

        # Connect to Redis cache
        await self._cache.connect()

        # Initialize TimescaleDB if available
        if self._db.is_available:
            await self._db.initialize()

        self._initialized = True
        logger.info("DataRepository initialized")

    async def close(self) -> None:
        """Close repository connections."""
        await self._cache.disconnect()
        await self._db.close()
        self._initialized = False

    # ==================== OHLCV Operations ====================

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        force_refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Get OHLCV data with 3-layer caching.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, H1, D1, etc.)
            limit: Maximum data points
            start_time: Start time filter
            end_time: End time filter
            force_refresh: Skip cache and fetch fresh data

        Returns:
            Tuple of (data, source) where source is 'cache', 'db', or 'api_required'
        """
        await self._ensure_initialized()

        tf = normalize_timeframe(timeframe)
        cache_key_params = {
            "limit": limit,
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None,
        }

        # 1. Redis Cache Check (if not force_refresh)
        if not force_refresh:
            cached = await self._cache.get(
                CacheCategory.OHLCV,
                symbol,
                tf.value,
                params=cache_key_params,
            )
            if cached:
                logger.debug(f"Cache HIT: {symbol}/{tf.value}")
                return cached, "cache"

        # 2. TimescaleDB Check
        if self._db.is_available:
            db_data = await self._db.get_ohlcv(
                symbol=symbol,
                timeframe=tf.value,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            )

            if db_data and self._is_data_fresh(db_data, tf):
                # Cache in Redis
                await self._cache.set(
                    CacheCategory.OHLCV,
                    db_data,
                    symbol,
                    tf.value,
                    params=cache_key_params,
                )
                logger.debug(f"DB HIT: {symbol}/{tf.value} ({len(db_data)} rows)")
                return db_data, "db"

        # 3. Data needs to be fetched from API
        logger.debug(f"DB MISS: {symbol}/{tf.value} - API fetch required")
        return [], "api_required"

    async def save_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Save OHLCV data (DB + Cache invalidation).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data list
            source: Data source identifier

        Returns:
            Number of rows saved
        """
        if not data:
            return 0

        await self._ensure_initialized()
        tf = normalize_timeframe(timeframe)
        count = 0

        # Save to TimescaleDB if available
        if self._db.is_available:
            count = await self._db.upsert_ohlcv(
                symbol=symbol,
                timeframe=tf.value,
                data=data,
                source=source,
            )

        # Invalidate cache (will be refreshed on next request)
        await self._cache.delete_pattern(
            f"*:{CacheCategory.OHLCV.value}:{symbol}:{tf.value}:*"
        )

        logger.info(f"Saved {count} OHLCV rows: {symbol}/{tf.value} from {source}")
        return count

    # ==================== Indicator Operations ====================

    async def get_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        limit: int = 100,
        force_refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """Get indicators with caching."""
        await self._ensure_initialized()
        tf = normalize_timeframe(timeframe)
        cache_key_params = {"indicator": indicator_name, "limit": limit}

        # 1. Redis Cache
        if not force_refresh:
            cached = await self._cache.get(
                CacheCategory.INDICATORS,
                symbol,
                tf.value,
                params=cache_key_params,
            )
            if cached:
                return cached, "cache"

        # 2. TimescaleDB
        if self._db.is_available:
            db_data = await self._db.get_indicators(
                symbol=symbol,
                timeframe=tf.value,
                indicator_name=indicator_name,
                limit=limit,
            )

            if db_data:
                await self._cache.set(
                    CacheCategory.INDICATORS,
                    db_data,
                    symbol,
                    tf.value,
                    params=cache_key_params,
                )
                return db_data, "db"

        return [], "api_required"

    async def save_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        data: list[dict[str, Any]],
        parameters: dict[str, Any],
        source: str,
    ) -> int:
        """Save indicators to DB + invalidate cache."""
        if not data:
            return 0

        await self._ensure_initialized()
        tf = normalize_timeframe(timeframe)
        count = 0

        if self._db.is_available:
            count = await self._db.upsert_indicators(
                symbol=symbol,
                timeframe=tf.value,
                indicator_name=indicator_name,
                data=data,
                parameters=parameters,
                source=source,
            )

        # Invalidate cache
        await self._cache.delete_pattern(
            f"*:{CacheCategory.INDICATORS.value}:{symbol}:{tf.value}:*"
        )

        return count

    # ==================== Optimized Indicator Access ====================

    async def get_all_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        force_refresh: bool = False,
    ) -> tuple[dict[str, list[dict[str, Any]]], str]:
        """Get all indicators from optimized tables."""
        await self._ensure_initialized()
        tf = normalize_timeframe(timeframe)
        cache_key_params = {"type": "all", "limit": limit}

        # 1. Redis Cache
        if not force_refresh:
            cached = await self._cache.get(
                CacheCategory.INDICATORS,
                symbol,
                tf.value,
                params=cache_key_params,
            )
            if cached:
                return cached, "cache"

        # 2. TimescaleDB
        if self._db.is_available:
            db_data = await self._db.get_all_indicators(
                symbol=symbol,
                timeframe=tf.value,
                limit=limit,
            )

            if db_data and any(db_data.values()):
                await self._cache.set(
                    CacheCategory.INDICATORS,
                    db_data,
                    symbol,
                    tf.value,
                    params=cache_key_params,
                )
                return db_data, "db"

        return {}, "api_required"

    # ==================== Freshness Tracking ====================

    async def get_freshness(
        self,
        symbol: str,
        timeframe: str,
        data_type: str = "ohlcv",
    ) -> Optional[dict[str, Any]]:
        """Get data freshness status."""
        await self._ensure_initialized()

        if self._db.is_available:
            return await self._db.get_freshness(symbol, timeframe, data_type)

        return None

    async def is_data_stale(
        self,
        symbol: str,
        timeframe: str,
        data_type: str = "ohlcv",
    ) -> bool:
        """Check if data needs refreshing."""
        freshness = await self.get_freshness(symbol, timeframe, data_type)

        if not freshness:
            return True  # No data = stale

        tf = normalize_timeframe(timeframe)
        threshold = FRESHNESS_THRESHOLDS.get(tf.value, 3600)

        last_updated = datetime.fromisoformat(freshness["last_updated"])
        now = datetime.now(timezone.utc)
        age_seconds = (now - last_updated).total_seconds()

        return age_seconds > threshold

    # ==================== Helper Methods ====================

    def _is_data_fresh(
        self,
        data: list[dict[str, Any]],
        timeframe: Timeframe,
    ) -> bool:
        """
        Check if data is fresh enough.

        Based on timeframe-specific freshness rules.
        """
        if not data:
            return False

        # Find latest timestamp
        try:
            latest = max(
                datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                if isinstance(row.get("timestamp"), str)
                else row.get("timestamp", datetime.min.replace(tzinfo=timezone.utc))
                for row in data
            )
        except (ValueError, TypeError):
            return False

        # Freshness intervals per timeframe
        freshness_intervals = {
            Timeframe.M1: timedelta(minutes=2),
            Timeframe.M5: timedelta(minutes=10),
            Timeframe.M15: timedelta(minutes=30),
            Timeframe.M30: timedelta(hours=1),
            Timeframe.M45: timedelta(hours=2),
            Timeframe.H1: timedelta(hours=2),
            Timeframe.H2: timedelta(hours=4),
            Timeframe.H4: timedelta(hours=8),
            Timeframe.D1: timedelta(days=1),
            Timeframe.W1: timedelta(days=7),
            Timeframe.MN: timedelta(days=30),
        }

        max_age = freshness_intervals.get(timeframe, timedelta(hours=1))

        # Handle timezone-aware comparison
        now = datetime.now(timezone.utc)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)

        return now - latest < max_age

    async def _ensure_initialized(self) -> None:
        """Ensure repository is initialized."""
        if not self._initialized:
            await self.initialize()

    # ==================== Statistics ====================

    async def get_statistics(self) -> dict[str, Any]:
        """Get repository statistics."""
        stats = {
            "cache": self._cache.get_stats(),
            "db_available": self._db.is_available,
        }

        if self._db.is_available:
            stats["db"] = await self._db.get_statistics()

        return stats


# Singleton instance
data_repository = DataRepository()
