"""
Unified Cache Service für das KI Trading Model.

Der Data Service fungiert als zentrales Gateway für alle externen Daten.
Dieser Cache Service stellt eine einheitliche Caching-Schicht bereit.

Caching-Strategie:
- Redis als primärer Cache (verteilt, persistent optional)
- In-Memory Fallback wenn Redis nicht verfügbar
- Automatische TTL-basierte Invalidierung
- JSON-Serialisierung für komplexe Datenstrukturen
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum
import os

from loguru import logger

# Redis-Import mit Fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis nicht verfügbar - verwende In-Memory Cache")


class CacheCategory(str, Enum):
    """Cache-Kategorien mit vordefinierten TTL-Werten."""

    # Marktdaten - kurze TTL da sich häufig ändern
    MARKET_DATA = "market"          # 60s - Echtzeit-Kurse
    OHLCV = "ohlcv"                  # 300s - Kerzendaten

    # Indikatoren - mittlere TTL
    INDICATORS = "indicators"        # 300s - Technische Indikatoren

    # Referenzdaten - längere TTL
    SYMBOLS = "symbols"              # 3600s - Symbol-Listen
    METADATA = "metadata"            # 3600s - Metadaten

    # Externe Quellen - variable TTL
    SENTIMENT = "sentiment"          # 900s - Sentiment-Daten
    ECONOMIC = "economic"            # 1800s - Wirtschaftskalender
    ONCHAIN = "onchain"              # 600s - On-Chain Daten

    # Training-Daten - lange TTL
    TRAINING = "training"            # 21600s (6h) - Training-Daten


# Standard-TTL pro Kategorie (in Sekunden)
DEFAULT_TTL = {
    CacheCategory.MARKET_DATA: 60,
    CacheCategory.OHLCV: 300,
    CacheCategory.INDICATORS: 300,
    CacheCategory.SYMBOLS: 3600,
    CacheCategory.METADATA: 3600,
    CacheCategory.SENTIMENT: 900,
    CacheCategory.ECONOMIC: 1800,
    CacheCategory.ONCHAIN: 600,
    CacheCategory.TRAINING: 21600,
}


class CacheService:
    """
    Einheitlicher Cache Service mit Redis-Backend.

    Features:
    - Redis als primärer Cache
    - In-Memory Fallback bei Verbindungsproblemen
    - Automatische Serialisierung/Deserialisierung
    - Statistik-Tracking (Hits, Misses, Bytes)
    - Kategorisierte TTL-Verwaltung
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        prefix: str = "trading",
        default_ttl: int = 300,
    ):
        """
        Initialisiert den Cache Service.

        Args:
            redis_url: Redis-Verbindungs-URL (z.B. redis://localhost:6379)
            prefix: Prefix für alle Cache-Keys
            default_ttl: Standard-TTL in Sekunden
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://trading-redis:6379")
        self.prefix = prefix
        self.default_ttl = default_ttl

        # Redis-Client (lazy initialization)
        self._redis: Optional[redis.Redis] = None
        self._redis_available = False

        # In-Memory Fallback Cache
        self._memory_cache: dict[str, tuple[datetime, int, Any]] = {}

        # Statistiken
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "bytes_saved": 0,
            "redis_errors": 0,
        }

        logger.info(f"CacheService initialisiert - Redis URL: {self.redis_url}")

    async def connect(self) -> bool:
        """
        Stellt Verbindung zu Redis her.

        Returns:
            True wenn Verbindung erfolgreich, False sonst
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis-Bibliothek nicht installiert - verwende In-Memory Cache")
            return False

        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Teste Verbindung
            await self._redis.ping()
            self._redis_available = True
            logger.info("Redis-Verbindung erfolgreich hergestellt")
            return True
        except Exception as e:
            logger.warning(f"Redis-Verbindung fehlgeschlagen: {e} - verwende In-Memory Cache")
            self._redis_available = False
            return False

    async def disconnect(self) -> None:
        """Trennt die Redis-Verbindung."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._redis_available = False
            logger.info("Redis-Verbindung getrennt")

    def _build_key(self, category: CacheCategory, *args) -> str:
        """
        Erstellt einen eindeutigen Cache-Key.

        Args:
            category: Cache-Kategorie
            *args: Key-Komponenten

        Returns:
            Formatierter Cache-Key
        """
        components = [self.prefix, category.value] + [str(a) for a in args]
        return ":".join(components)

    def _hash_params(self, params: dict) -> str:
        """
        Erstellt einen Hash aus Parametern für den Cache-Key.

        Args:
            params: Dictionary mit Parametern

        Returns:
            MD5-Hash der Parameter
        """
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()[:12]

    def _serialize(self, data: Any) -> str:
        """Serialisiert Daten für den Cache."""
        return json.dumps(data, default=str)

    def _deserialize(self, data: str) -> Any:
        """Deserialisiert Daten aus dem Cache."""
        return json.loads(data)

    async def get(
        self,
        category: CacheCategory,
        *key_parts,
        params: Optional[dict] = None,
    ) -> Optional[Any]:
        """
        Holt Daten aus dem Cache.

        Args:
            category: Cache-Kategorie
            *key_parts: Key-Komponenten
            params: Optionale Parameter für Key-Hash

        Returns:
            Gecachte Daten oder None
        """
        key_components = list(key_parts)
        if params:
            key_components.append(self._hash_params(params))

        cache_key = self._build_key(category, *key_components)

        # Versuche Redis
        if self._redis_available and self._redis:
            try:
                data = await self._redis.get(cache_key)
                if data:
                    self._stats["hits"] += 1
                    self._stats["bytes_saved"] += len(data)
                    return self._deserialize(data)
                self._stats["misses"] += 1
                return None
            except Exception as e:
                self._stats["redis_errors"] += 1
                logger.warning(f"Redis GET Fehler: {e}")

        # Fallback: In-Memory Cache
        if cache_key in self._memory_cache:
            expires, _, data = self._memory_cache[cache_key]
            if expires > datetime.now(timezone.utc):
                self._stats["hits"] += 1
                return data
            # Abgelaufen - entfernen
            del self._memory_cache[cache_key]

        self._stats["misses"] += 1
        return None

    async def set(
        self,
        category: CacheCategory,
        data: Any,
        *key_parts,
        params: Optional[dict] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Speichert Daten im Cache.

        Args:
            category: Cache-Kategorie
            data: Zu cachende Daten
            *key_parts: Key-Komponenten
            params: Optionale Parameter für Key-Hash
            ttl: Time-to-Live in Sekunden (überschreibt Kategorie-Standard)

        Returns:
            True bei Erfolg
        """
        key_components = list(key_parts)
        if params:
            key_components.append(self._hash_params(params))

        cache_key = self._build_key(category, *key_components)
        effective_ttl = ttl or DEFAULT_TTL.get(category, self.default_ttl)

        serialized = self._serialize(data)
        self._stats["sets"] += 1

        # Versuche Redis
        if self._redis_available and self._redis:
            try:
                await self._redis.setex(cache_key, effective_ttl, serialized)
                return True
            except Exception as e:
                self._stats["redis_errors"] += 1
                logger.warning(f"Redis SET Fehler: {e}")

        # Fallback: In-Memory Cache
        from datetime import timedelta
        expires = datetime.now(timezone.utc) + timedelta(seconds=effective_ttl)
        self._memory_cache[cache_key] = (expires, len(serialized), data)
        return True

    async def delete(
        self,
        category: CacheCategory,
        *key_parts,
        params: Optional[dict] = None,
    ) -> bool:
        """
        Löscht einen Eintrag aus dem Cache.

        Args:
            category: Cache-Kategorie
            *key_parts: Key-Komponenten
            params: Optionale Parameter für Key-Hash

        Returns:
            True bei Erfolg
        """
        key_components = list(key_parts)
        if params:
            key_components.append(self._hash_params(params))

        cache_key = self._build_key(category, *key_components)
        self._stats["deletes"] += 1

        # Versuche Redis
        if self._redis_available and self._redis:
            try:
                await self._redis.delete(cache_key)
                return True
            except Exception as e:
                self._stats["redis_errors"] += 1
                logger.warning(f"Redis DELETE Fehler: {e}")

        # Fallback: In-Memory Cache
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        return True

    async def delete_pattern(self, pattern: str) -> int:
        """
        Löscht alle Keys die einem Pattern entsprechen.

        Args:
            pattern: Redis-Pattern (z.B. "trading:market:*")

        Returns:
            Anzahl gelöschter Keys
        """
        deleted = 0

        if self._redis_available and self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=pattern)
                    if keys:
                        await self._redis.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
                return deleted
            except Exception as e:
                self._stats["redis_errors"] += 1
                logger.warning(f"Redis SCAN/DELETE Fehler: {e}")

        # Fallback: In-Memory Cache
        import fnmatch
        keys_to_delete = [k for k in self._memory_cache.keys() if fnmatch.fnmatch(k, pattern)]
        for key in keys_to_delete:
            del self._memory_cache[key]
            deleted += 1

        return deleted

    async def clear_category(self, category: CacheCategory) -> int:
        """
        Löscht alle Einträge einer Kategorie.

        Args:
            category: Cache-Kategorie

        Returns:
            Anzahl gelöschter Einträge
        """
        pattern = f"{self.prefix}:{category.value}:*"
        return await self.delete_pattern(pattern)

    async def clear_all(self) -> int:
        """
        Löscht den gesamten Cache.

        Returns:
            Anzahl gelöschter Einträge
        """
        pattern = f"{self.prefix}:*"
        return await self.delete_pattern(pattern)

    async def cleanup_expired(self) -> int:
        """
        Entfernt abgelaufene Einträge aus dem In-Memory Cache.

        Returns:
            Anzahl entfernter Einträge
        """
        now = datetime.now(timezone.utc)
        expired_keys = [
            key for key, (expires, _, _) in self._memory_cache.items()
            if expires <= now
        ]

        for key in expired_keys:
            del self._memory_cache[key]

        if expired_keys:
            logger.debug(f"Cache Cleanup: {len(expired_keys)} abgelaufene Einträge entfernt")

        return len(expired_keys)

    def get_stats(self) -> dict:
        """
        Gibt Cache-Statistiken zurück.

        Returns:
            Dictionary mit Statistiken
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0.0

        memory_size = sum(size for _, size, _ in self._memory_cache.values())

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "sets": self._stats["sets"],
            "deletes": self._stats["deletes"],
            "bytes_saved": self._stats["bytes_saved"],
            "redis_errors": self._stats["redis_errors"],
            "redis_available": self._redis_available,
            "memory_cache_entries": len(self._memory_cache),
            "memory_cache_bytes": memory_size,
        }

    async def health_check(self) -> dict:
        """
        Prüft den Gesundheitszustand des Cache.

        Returns:
            Health-Status Dictionary
        """
        redis_healthy = False
        redis_info = {}

        if self._redis_available and self._redis:
            try:
                await self._redis.ping()
                redis_info = await self._redis.info("memory")
                redis_healthy = True
            except Exception as e:
                logger.warning(f"Redis Health Check fehlgeschlagen: {e}")

        return {
            "status": "healthy" if redis_healthy else "degraded",
            "redis_connected": redis_healthy,
            "redis_memory_used": redis_info.get("used_memory_human", "N/A"),
            "fallback_active": not redis_healthy,
            "memory_cache_entries": len(self._memory_cache),
            "stats": self.get_stats(),
        }


# Singleton-Instanz
cache_service = CacheService()


async def get_cache() -> CacheService:
    """
    Factory-Funktion für Dependency Injection.

    Returns:
        CacheService Singleton
    """
    if not cache_service._redis_available:
        await cache_service.connect()
    return cache_service
