"""Base classes for RAG data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from enum import Enum


class DataSourceType(str, Enum):
    """Types of data sources for RAG."""
    ECONOMIC_CALENDAR = "economic_calendar"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    ORDERBOOK = "orderbook"
    MACRO_CORRELATION = "macro_correlation"
    HISTORICAL_PATTERN = "historical_pattern"
    TECHNICAL_LEVEL = "technical_level"
    REGULATORY = "regulatory"


class DataPriority(str, Enum):
    """Priority levels for data relevance."""
    CRITICAL = "critical"  # Immediate market impact (Fed decision, major hack)
    HIGH = "high"          # Strong influence (CPI release, whale movement)
    MEDIUM = "medium"      # Notable impact (earnings, sentiment shift)
    LOW = "low"            # Background context (minor news, general trends)


@dataclass
class DataSourceResult:
    """Result from a data source fetch operation."""
    source_type: DataSourceType
    content: str
    symbol: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: DataPriority = DataPriority.MEDIUM
    metadata: dict = field(default_factory=dict)
    raw_data: Optional[Any] = None

    def to_rag_document(self) -> dict:
        """Convert to RAG document format."""
        return {
            "content": self.content,
            "document_type": self.source_type.value,
            "symbol": self.symbol,
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "priority": self.priority.value,
                **self.metadata
            }
        }


class DataSourceBase(ABC):
    """Abstract base class for all data sources."""

    source_type: DataSourceType

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes default

    @abstractmethod
    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """Fetch data from the source.

        Args:
            symbol: Optional trading symbol to filter by
            **kwargs: Source-specific parameters

        Returns:
            List of DataSourceResult objects
        """
        pass

    @abstractmethod
    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch data formatted for RAG ingestion.

        Args:
            symbol: Optional trading symbol
            **kwargs: Source-specific parameters

        Returns:
            List of dicts ready for RAG document storage
        """
        pass

    def _get_cache_key(self, symbol: Optional[str], **kwargs) -> str:
        """Generate cache key for requests."""
        parts = [self.source_type.value]
        if symbol:
            parts.append(symbol)
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}={v}")
        return ":".join(parts)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        cached_time, _ = self._cache[cache_key]
        return (datetime.utcnow() - cached_time).total_seconds() < self._cache_ttl

    def _get_cached(self, cache_key: str) -> Optional[list[DataSourceResult]]:
        """Get cached results if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        return None

    def _set_cache(self, cache_key: str, results: list[DataSourceResult]):
        """Cache results."""
        self._cache[cache_key] = (datetime.utcnow(), results)
