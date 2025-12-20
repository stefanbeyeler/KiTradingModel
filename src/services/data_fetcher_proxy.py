"""Data Fetcher Proxy - Proxies data requests through Data Service.

This proxy forwards all external data requests to the Data Service (gateway architecture).
If the Data Service is not available, it logs an error (no fallback - architecture rule).
"""

from typing import Optional
from loguru import logger

from .data_service_client import DataServiceClient, get_data_service_client


class DataFetcherProxy:
    """Proxy for accessing external data sources via Data Service."""

    def __init__(self):
        """Initialize the proxy with Data Service client."""
        self._client = get_data_service_client()
        logger.info("DataFetcherProxy initialized")

    def get_available_sources(self) -> list[dict]:
        """Get available sources (synchronous wrapper)."""
        # This needs to be called asynchronously
        return [
            {"type": "economic_calendar", "name": "ECONOMIC_CALENDAR", "description": "Economic calendar events (Fed, CPI, NFP, etc.)"},
            {"type": "onchain", "name": "ONCHAIN", "description": "On-chain crypto metrics (whale alerts, exchange flows, mining)"},
            {"type": "sentiment", "name": "SENTIMENT", "description": "Market sentiment (Fear & Greed, social, options, VIX)"},
            {"type": "orderbook", "name": "ORDERBOOK", "description": "Orderbook data (walls, liquidations, CVD, order flow)"},
            {"type": "macro_correlation", "name": "MACRO_CORRELATION", "description": "Macro data (DXY, bonds, correlations, sectors)"},
            {"type": "historical_pattern", "name": "HISTORICAL_PATTERN", "description": "Historical patterns (seasonality, drawdowns, events)"},
            {"type": "technical_level", "name": "TECHNICAL_LEVEL", "description": "Technical levels (S/R, Fibonacci, pivots, VWAP, MAs)"},
            {"type": "regulatory", "name": "REGULATORY", "description": "Regulatory updates (SEC, ETFs, global regulation)"},
            {"type": "easyinsight", "name": "EASYINSIGHT", "description": "EasyInsight data (managed symbols, MT5 logs, model status)"},
        ]

    async def fetch_all(
        self,
        symbol: Optional[str] = None,
        source_types=None,  # Accept enum or list
        min_priority=None
    ) -> list:
        """Fetch data from all or selected sources via Data Service."""
        # Convert enums to strings if needed
        source_type_strs = None
        if source_types:
            source_type_strs = [st.value if hasattr(st, 'value') else str(st) for st in source_types]

        priority_str = "low"
        if min_priority:
            priority_str = min_priority.value if hasattr(min_priority, 'value') else str(min_priority)

        result = await self._client.fetch_all(symbol, source_type_strs, priority_str)

        if "error" in result:
            logger.error(f"Error fetching from Data Service: {result['error']}")
            return []

        # Convert to DataSourceResult-like objects
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_trading_context(self, symbol: str, include_types: Optional[list[str]] = None) -> dict:
        """Fetch comprehensive trading context via Data Service."""
        return await self._client.fetch_trading_context(symbol, include_types)

    async def fetch_economic_calendar(self, symbol: Optional[str] = None, days_ahead: int = 7, days_back: int = 1) -> list:
        """Fetch economic calendar events via Data Service."""
        result = await self._client.get_economic_calendar(symbol, days_ahead, days_back)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("events", [])]

    async def fetch_sentiment(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch sentiment data via Data Service."""
        result = await self._client.get_sentiment(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_onchain(self, symbol: str, **kwargs) -> list:
        """Fetch on-chain data via Data Service."""
        result = await self._client.get_onchain(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_orderbook(self, symbol: str, **kwargs) -> list:
        """Fetch orderbook data via Data Service."""
        result = await self._client.get_orderbook(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_macro(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch macro data via Data Service."""
        result = await self._client.get_macro(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_historical_patterns(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch historical patterns via Data Service."""
        result = await self._client.get_historical_patterns(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_technical_levels(self, symbol: str, **kwargs) -> list:
        """Fetch technical levels via Data Service."""
        result = await self._client.get_technical_levels(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_regulatory(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch regulatory updates via Data Service."""
        result = await self._client.get_regulatory(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_easyinsight(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch EasyInsight data via Data Service."""
        result = await self._client.get_easyinsight(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def create_rag_documents_batch(self, symbol: Optional[str] = None, source_types=None) -> list[dict]:
        """Create RAG documents from all sources via Data Service."""
        results = await self.fetch_all(symbol, source_types)
        return [r.to_rag_document() for r in results]


class ProxyResult:
    """Wrapper class to make Data Service responses compatible with DataSourceResult interface."""

    def __init__(self, data: dict):
        self._data = data
        self.content = data.get("content", "")
        self.source_type = type('obj', (object,), {'value': data.get("document_type", "unknown")})()
        self.priority = type('obj', (object,), {'value': data.get("metadata", {}).get("priority", "medium")})()
        self.symbol = data.get("symbol")
        self.metadata = data.get("metadata", {})

    def to_rag_document(self) -> dict:
        """Return the original document format."""
        return self._data


# Singleton instance
_proxy: Optional[DataFetcherProxy] = None


def get_data_fetcher_proxy() -> DataFetcherProxy:
    """Get or create the singleton DataFetcherProxy instance."""
    global _proxy
    if _proxy is None:
        _proxy = DataFetcherProxy()
    return _proxy
