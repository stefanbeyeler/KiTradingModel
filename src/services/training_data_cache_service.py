"""
Training Data Cache Service.

Provides persistent caching of training data from EasyInsight and TwelveData APIs
to reduce API calls during batch NHITS model training.

Data is cached on disk with TTL per timeframe. Cache entries expire automatically
and are cleaned up periodically. This is especially important for Twelve Data API
which has daily request limits.
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata for cached data."""
    symbol: str
    timeframe: str
    source: str  # 'easyinsight' or 'twelvedata'
    created_at: str
    expires_at: str
    row_count: int
    data_hash: str


class TrainingDataCacheService:
    """
    Service for caching training data to reduce API calls.

    Caches raw API responses on disk with configurable TTL per timeframe.
    Automatically cleans up expired cache entries and can clear all cache
    after training completes.
    """

    # TTL in hours per timeframe - optimized for reducing API calls
    # Especially important for Twelve Data which has daily request limits
    DEFAULT_TTL = {
        "M15": 6,    # 6 hours - reasonable for intraday retraining
        "H1": 12,    # 12 hours - covers multiple training runs per day
        "D1": 48,    # 48 hours - D1 data changes slowly (1 new candle/day)
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache service.

        Args:
            cache_dir: Directory for cache files. Defaults to data/training_cache
        """
        if cache_dir is None:
            # Use the same data directory as other services
            base_path = Path(settings.nhits_model_path).parent
            cache_dir = base_path / "training_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._bytes_saved = 0

        logger.info(f"Training data cache initialized at {self.cache_dir}")

    def _get_cache_key(self, symbol: str, timeframe: str, source: str = "easyinsight") -> str:
        """Generate unique cache key for symbol/timeframe/source combination."""
        return f"{source}_{symbol}_{timeframe}".replace("/", "_").replace("\\", "_")

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"{cache_key}.json"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get file path for cache metadata."""
        return self.cache_dir / f"{cache_key}.meta.json"

    def _compute_hash(self, data: List[Dict]) -> str:
        """Compute hash of data for integrity checking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_ttl_hours(self, timeframe: str) -> int:
        """Get TTL in hours for a timeframe."""
        return self.DEFAULT_TTL.get(timeframe.upper(), 4)

    def is_cached(self, symbol: str, timeframe: str, source: str = "easyinsight") -> bool:
        """Check if valid (non-expired) cache exists for symbol/timeframe."""
        cache_key = self._get_cache_key(symbol, timeframe, source)
        metadata_path = self._get_metadata_path(cache_key)
        cache_path = self._get_cache_path(cache_key)

        if not metadata_path.exists() or not cache_path.exists():
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            expires_at = datetime.fromisoformat(metadata['expires_at'])
            return datetime.utcnow() < expires_at
        except Exception as e:
            logger.warning(f"Error checking cache for {cache_key}: {e}")
            return False

    def get_cached_data(
        self,
        symbol: str,
        timeframe: str,
        source: str = "easyinsight"
    ) -> Optional[List[Dict]]:
        """
        Get cached data if available and not expired.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M15, H1, D1)
            source: Data source (easyinsight or twelvedata)

        Returns:
            Cached data rows or None if cache miss/expired
        """
        cache_key = self._get_cache_key(symbol, timeframe, source)

        if not self.is_cached(symbol, timeframe, source):
            self._cache_misses += 1
            return None

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            self._cache_hits += 1
            self._bytes_saved += cache_path.stat().st_size
            logger.debug(f"Cache hit for {cache_key}: {len(data)} rows")
            return data

        except Exception as e:
            logger.warning(f"Error reading cache for {cache_key}: {e}")
            self._cache_misses += 1
            return None

    def cache_data(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict],
        source: str = "easyinsight",
        ttl_hours: Optional[int] = None
    ) -> bool:
        """
        Cache data for a symbol/timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M15, H1, D1)
            data: Raw API response data to cache
            source: Data source (easyinsight or twelvedata)
            ttl_hours: Override default TTL

        Returns:
            True if successfully cached
        """
        if not data:
            return False

        cache_key = self._get_cache_key(symbol, timeframe, source)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        if ttl_hours is None:
            ttl_hours = self._get_ttl_hours(timeframe)

        try:
            # Create metadata
            now = datetime.utcnow()
            metadata = CacheMetadata(
                symbol=symbol,
                timeframe=timeframe,
                source=source,
                created_at=now.isoformat(),
                expires_at=(now + timedelta(hours=ttl_hours)).isoformat(),
                row_count=len(data),
                data_hash=self._compute_hash(data)
            )

            # Write cache file
            with open(cache_path, 'w') as f:
                json.dump(data, f)

            # Write metadata
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)

            logger.debug(
                f"Cached {len(data)} rows for {cache_key} "
                f"(expires in {ttl_hours}h)"
            )
            return True

        except Exception as e:
            logger.error(f"Error caching data for {cache_key}: {e}")
            return False

    def invalidate(self, symbol: str, timeframe: str, source: str = "easyinsight") -> bool:
        """Invalidate (delete) cache for a specific symbol/timeframe."""
        cache_key = self._get_cache_key(symbol, timeframe, source)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        deleted = False
        try:
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True

            if deleted:
                logger.debug(f"Invalidated cache for {cache_key}")
            return deleted
        except Exception as e:
            logger.error(f"Error invalidating cache for {cache_key}: {e}")
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries.

        Returns:
            Number of entries removed
        """
        removed = 0
        now = datetime.utcnow()

        try:
            for meta_file in self.cache_dir.glob("*.meta.json"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                    expires_at = datetime.fromisoformat(metadata['expires_at'])
                    if now >= expires_at:
                        # Remove expired entry
                        cache_key = meta_file.stem.replace('.meta', '')
                        cache_file = self.cache_dir / f"{cache_key}.json"

                        if cache_file.exists():
                            cache_file.unlink()
                        meta_file.unlink()
                        removed += 1

                except Exception as e:
                    logger.warning(f"Error processing {meta_file}: {e}")

            if removed > 0:
                logger.info(f"Cleaned up {removed} expired cache entries")
            return removed

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return removed

    def clear_all(self) -> int:
        """Clear all cached data.

        Returns:
            Number of files removed
        """
        removed = 0
        try:
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
                removed += 1

            logger.info(f"Cleared all cache: {removed} files removed")
            return removed

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return removed

    def clear_for_symbols(self, symbols: List[str]) -> int:
        """Clear cache for specific symbols (all timeframes).

        Args:
            symbols: List of symbols to clear

        Returns:
            Number of entries removed
        """
        removed = 0
        for symbol in symbols:
            for timeframe in ["M15", "H1", "D1"]:
                for source in ["easyinsight", "twelvedata"]:
                    if self.invalidate(symbol, timeframe, source):
                        removed += 1

        if removed > 0:
            logger.info(f"Cleared cache for {len(symbols)} symbols: {removed} entries removed")
        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = 0
        total_size = 0
        by_timeframe = {"M15": 0, "H1": 0, "D1": 0}
        by_source = {"easyinsight": 0, "twelvedata": 0}
        expired_count = 0
        now = datetime.utcnow()

        try:
            for meta_file in self.cache_dir.glob("*.meta.json"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                    total_entries += 1

                    # Count by timeframe
                    tf = metadata.get('timeframe', 'H1')
                    if tf in by_timeframe:
                        by_timeframe[tf] += 1

                    # Count by source
                    source = metadata.get('source', 'easyinsight')
                    if source in by_source:
                        by_source[source] += 1

                    # Check if expired
                    expires_at = datetime.fromisoformat(metadata['expires_at'])
                    if now >= expires_at:
                        expired_count += 1

                    # Calculate size
                    cache_key = meta_file.stem.replace('.meta', '')
                    cache_file = self.cache_dir / f"{cache_key}.json"
                    if cache_file.exists():
                        total_size += cache_file.stat().st_size

                except Exception:
                    pass

            return {
                "cache_dir": str(self.cache_dir),
                "total_entries": total_entries,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "by_timeframe": by_timeframe,
                "by_source": by_source,
                "expired_count": expired_count,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": round(
                    self._cache_hits / max(1, self._cache_hits + self._cache_misses) * 100, 1
                ),
                "bytes_saved": self._bytes_saved,
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    def get_cached_symbols(self) -> Dict[str, List[str]]:
        """Get list of cached symbols grouped by timeframe.

        Returns:
            Dict mapping timeframe to list of cached symbols
        """
        result = {"M15": [], "H1": [], "D1": []}
        now = datetime.utcnow()

        try:
            for meta_file in self.cache_dir.glob("*.meta.json"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                    # Only include non-expired entries
                    expires_at = datetime.fromisoformat(metadata['expires_at'])
                    if now < expires_at:
                        tf = metadata.get('timeframe', 'H1')
                        symbol = metadata.get('symbol', '')
                        if tf in result and symbol:
                            result[tf].append(symbol)

                except Exception:
                    pass

            # Sort lists
            for tf in result:
                result[tf].sort()

            return result

        except Exception as e:
            logger.error(f"Error getting cached symbols: {e}")
            return result


# Global instance
training_data_cache = TrainingDataCacheService()
