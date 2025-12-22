"""Cache service for embeddings."""

import hashlib
import json
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry."""
    data: np.ndarray
    created_at: datetime
    hits: int = 0
    size_bytes: int = 0


@dataclass
class CacheStats:
    """Cache statistics."""
    entries: int = 0
    size_mb: float = 0.0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0


class CacheService:
    """
    In-memory cache for embeddings.

    Features:
    - LRU eviction
    - TTL support
    - Size limits
    - Statistics tracking
    """

    def __init__(
        self,
        max_entries: int = 10000,
        max_size_mb: float = 500.0,
        ttl_minutes: int = 60
    ):
        """
        Initialize the cache service.

        Args:
            max_entries: Maximum number of cache entries
            max_size_mb: Maximum cache size in MB
            ttl_minutes: Time to live for entries in minutes
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._ttl = timedelta(minutes=ttl_minutes)
        self._hits = 0
        self._misses = 0
        self._current_size = 0

    def _generate_key(self, data: Any, prefix: str = "") -> str:
        """Generate a cache key for the data."""
        if isinstance(data, np.ndarray):
            data_str = data.tobytes()
        elif isinstance(data, (list, dict)):
            data_str = json.dumps(data, sort_keys=True).encode()
        else:
            data_str = str(data).encode()

        hash_val = hashlib.sha256(data_str).hexdigest()[:32]
        return f"{prefix}:{hash_val}" if prefix else hash_val

    def _evict_if_needed(self):
        """Evict entries if cache is full."""
        # Evict expired entries first
        now = datetime.now()
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v.created_at > self._ttl
        ]
        for key in expired_keys:
            self._remove(key)

        # If still over limit, evict least recently used
        while len(self._cache) >= self._max_entries or self._current_size >= self._max_size_bytes:
            if not self._cache:
                break

            # Find entry with fewest hits (simple LRU proxy)
            min_key = min(self._cache.keys(), key=lambda k: self._cache[k].hits)
            self._remove(min_key)

    def _remove(self, key: str):
        """Remove an entry from cache."""
        if key in self._cache:
            self._current_size -= self._cache[key].size_bytes
            del self._cache[key]

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get an entry from cache.

        Args:
            key: Cache key

        Returns:
            Cached embedding or None if not found
        """
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check TTL
        if datetime.now() - entry.created_at > self._ttl:
            self._remove(key)
            self._misses += 1
            return None

        entry.hits += 1
        self._hits += 1
        return entry.data

    def set(self, key: str, data: np.ndarray):
        """
        Store an entry in cache.

        Args:
            key: Cache key
            data: Embedding data to cache
        """
        self._evict_if_needed()

        size_bytes = data.nbytes
        entry = CacheEntry(
            data=data,
            created_at=datetime.now(),
            size_bytes=size_bytes
        )

        self._cache[key] = entry
        self._current_size += size_bytes

    def get_or_compute(
        self,
        data: Any,
        compute_fn: callable,
        prefix: str = ""
    ) -> tuple[np.ndarray, bool]:
        """
        Get from cache or compute and store.

        Args:
            data: Input data for key generation
            compute_fn: Function to compute embedding if not cached
            prefix: Key prefix for namespacing

        Returns:
            Tuple of (embedding, was_cached)
        """
        key = self._generate_key(data, prefix)

        cached = self.get(key)
        if cached is not None:
            return cached, True

        result = compute_fn()
        self.set(key, result)
        return result, False

    def clear(self) -> int:
        """
        Clear the cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        self._current_size = 0
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats object
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return CacheStats(
            entries=len(self._cache),
            size_mb=self._current_size / (1024 * 1024),
            hits=self._hits,
            misses=self._misses,
            hit_rate=hit_rate
        )


# Singleton instance
cache_service = CacheService()
