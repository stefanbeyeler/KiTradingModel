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
            {"type": "candlestick_patterns", "name": "CANDLESTICK_PATTERNS", "description": "Candlestick pattern analysis (24 patterns, multi-timeframe)"},
            {"type": "correlations", "name": "CORRELATIONS", "description": "Asset correlations (cross-asset, divergences, hedge recommendations)"},
            {"type": "volatility_regime", "name": "VOLATILITY_REGIME", "description": "Volatility regime (VIX, ATR, Bollinger, position sizing)"},
            {"type": "institutional_flow", "name": "INSTITUTIONAL_FLOW", "description": "Institutional flows (COT reports, ETF flows, whale tracking)"},
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

    async def fetch_correlations(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch asset correlation data via Data Service."""
        result = await self._client.get_correlations(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_volatility_regime(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch volatility regime data via Data Service."""
        result = await self._client.get_volatility_regime(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_institutional_flow(self, symbol: Optional[str] = None, **kwargs) -> list:
        """Fetch institutional flow data (COT, ETF, whale) via Data Service."""
        result = await self._client.get_institutional_flow(symbol, **kwargs)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            return []
        return [ProxyResult(item) for item in result.get("data", [])]

    async def fetch_candlestick_patterns(
        self,
        symbol: str,
        timeframes: Optional[list[str]] = None,
        lookback_candles: int = 20,
        min_confidence: float = 0.5
    ) -> list:
        """Fetch candlestick patterns via Data Service.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            timeframes: List of timeframes (e.g., ["M5", "H1", "D1"])
            lookback_candles: Number of candles to analyze
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            List of ProxyResult objects containing detected patterns
        """
        result = await self._client.get_candlestick_patterns(
            symbol=symbol,
            timeframes=timeframes,
            lookback_candles=lookback_candles,
            min_confidence=min_confidence
        )
        if "error" in result:
            logger.error(f"Error fetching candlestick patterns: {result['error']}")
            return []

        # Get the result object (Data Service returns {request_id, symbol, result: {...}})
        pattern_result = result.get("result", result)

        # Convert pattern results to RAG-compatible format
        # Data Service returns patterns under keys like "m5", "m15", "h1", "h4", "d1"
        documents = []
        timeframe_keys = ["m5", "m15", "h1", "h4", "d1"]
        for tf_key in timeframe_keys:
            tf_data = pattern_result.get(tf_key, {})
            if not tf_data:
                continue
            timeframe = tf_data.get("timeframe", tf_key.upper())
            patterns = tf_data.get("patterns", [])

            for pattern in patterns:
                # Create a document for each detected pattern
                content = self._format_pattern_content(symbol, timeframe, pattern)
                doc = {
                    "content": content,
                    "document_type": "candlestick_patterns",
                    "symbol": symbol,
                    "metadata": {
                        "timeframe": timeframe,
                        "pattern_type": pattern.get("pattern_type"),
                        "direction": pattern.get("direction"),
                        "confidence": pattern.get("confidence"),
                        "strength": pattern.get("strength"),
                        "category": pattern.get("category"),
                        "priority": "high" if pattern.get("confidence", 0) >= 0.7 else "medium",
                    }
                }
                documents.append(ProxyResult(doc))

        return documents

    def _format_pattern_content(self, symbol: str, timeframe: str, pattern: dict) -> str:
        """Format a candlestick pattern into readable content for RAG."""
        pattern_type = pattern.get("pattern_type", "Unknown")
        direction = pattern.get("direction", "neutral")
        confidence = pattern.get("confidence", 0)
        strength = pattern.get("strength", "moderate")
        description = pattern.get("description", "")
        trading_implication = pattern.get("trading_implication", "")

        return f"""Candlestick Pattern: {pattern_type}
Symbol: {symbol}
Timeframe: {timeframe}
Direction: {direction}
Confidence: {confidence:.1%}
Strength: {strength}

Description: {description}

Trading Implication: {trading_implication}"""

    async def fetch_candlestick_pattern_types(self) -> dict:
        """Fetch all available candlestick pattern types."""
        return await self._client.get_candlestick_pattern_types()

    async def fetch_candlestick_pattern_summary(self, symbol: str) -> dict:
        """Fetch simplified candlestick pattern summary for a symbol."""
        return await self._client.get_candlestick_pattern_summary(symbol)

    async def fetch_candlestick_pattern_history(
        self,
        symbol: Optional[str] = None,
        pattern_type: Optional[str] = None,
        direction: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> list:
        """Fetch historical candlestick pattern detections."""
        result = await self._client.get_candlestick_pattern_history(
            symbol=symbol,
            pattern_type=pattern_type,
            direction=direction,
            min_confidence=min_confidence,
            limit=limit
        )
        if "error" in result:
            logger.error(f"Error fetching pattern history: {result['error']}")
            return []

        # Convert to RAG-compatible format
        documents = []
        for entry in result.get("patterns", []):
            content = self._format_pattern_content(
                entry.get("symbol", ""),
                entry.get("timeframe", ""),
                entry
            )
            doc = {
                "content": content,
                "document_type": "candlestick_patterns",
                "symbol": entry.get("symbol"),
                "metadata": {
                    "timeframe": entry.get("timeframe"),
                    "pattern_type": entry.get("pattern_type"),
                    "direction": entry.get("direction"),
                    "confidence": entry.get("confidence"),
                    "detected_at": entry.get("detected_at"),
                    "priority": "high" if entry.get("confidence", 0) >= 0.7 else "medium",
                }
            }
            documents.append(ProxyResult(doc))

        return documents

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
