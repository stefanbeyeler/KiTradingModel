"""Data Fetcher Service - Central service for fetching and coordinating all RAG data sources."""

import asyncio
from datetime import datetime
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority
from .economic_calendar import EconomicCalendarSource
from .onchain_data import OnChainDataSource
from .sentiment_data import SentimentDataSource
from .orderbook_data import OrderbookDataSource
from .macro_correlation import MacroCorrelationSource
from .historical_patterns import HistoricalPatternsSource
from .technical_levels import TechnicalLevelsSource
from .regulatory_updates import RegulatoryUpdatesSource


class DataFetcherService:
    """
    Central service that coordinates all RAG data sources.

    This service provides:
    - Unified interface for fetching from all data sources
    - Parallel fetching for performance
    - Priority-based filtering
    - Caching coordination
    - RAG document formatting
    """

    def __init__(self):
        """Initialize all data sources."""
        self._sources: dict[DataSourceType, DataSourceBase] = {
            DataSourceType.ECONOMIC_CALENDAR: EconomicCalendarSource(),
            DataSourceType.ONCHAIN: OnChainDataSource(),
            DataSourceType.SENTIMENT: SentimentDataSource(),
            DataSourceType.ORDERBOOK: OrderbookDataSource(),
            DataSourceType.MACRO_CORRELATION: MacroCorrelationSource(),
            DataSourceType.HISTORICAL_PATTERN: HistoricalPatternsSource(),
            DataSourceType.TECHNICAL_LEVEL: TechnicalLevelsSource(),
            DataSourceType.REGULATORY: RegulatoryUpdatesSource(),
        }
        logger.info(f"DataFetcherService initialized with {len(self._sources)} data sources")

    async def fetch_all(
        self,
        symbol: Optional[str] = None,
        source_types: Optional[list[DataSourceType]] = None,
        min_priority: DataPriority = DataPriority.LOW,
        **kwargs
    ) -> list[DataSourceResult]:
        """
        Fetch data from all or selected sources in parallel.

        Args:
            symbol: Trading symbol for context
            source_types: List of specific source types to fetch (None = all)
            min_priority: Minimum priority level to include
            **kwargs: Additional parameters passed to sources

        Returns:
            List of all results from all sources
        """
        sources_to_fetch = (
            {st: self._sources[st] for st in source_types if st in self._sources}
            if source_types
            else self._sources
        )

        if not sources_to_fetch:
            logger.warning("No sources to fetch")
            return []

        logger.info(f"Fetching from {len(sources_to_fetch)} data sources for {symbol or 'market'}")
        start_time = datetime.now()

        # Fetch all sources in parallel
        tasks = [
            source.fetch(symbol, **kwargs)
            for source in sources_to_fetch.values()
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        results = []
        for i, result in enumerate(all_results):
            source_type = list(sources_to_fetch.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching from {source_type}: {result}")
                continue
            results.extend(result)

        # Filter by priority
        priority_order = [DataPriority.CRITICAL, DataPriority.HIGH, DataPriority.MEDIUM, DataPriority.LOW]
        min_priority_idx = priority_order.index(min_priority)
        filtered_results = [
            r for r in results
            if priority_order.index(r.priority) <= min_priority_idx
        ]

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Fetched {len(results)} results ({len(filtered_results)} after priority filter) "
            f"in {elapsed:.2f}s"
        )

        return filtered_results

    async def fetch_for_rag(
        self,
        symbol: Optional[str] = None,
        source_types: Optional[list[DataSourceType]] = None,
        min_priority: DataPriority = DataPriority.LOW,
        **kwargs
    ) -> list[dict]:
        """
        Fetch data formatted for RAG document storage.

        Args:
            symbol: Trading symbol
            source_types: Specific sources to fetch
            min_priority: Minimum priority
            **kwargs: Additional parameters

        Returns:
            List of dicts ready for RAG ingestion
        """
        results = await self.fetch_all(symbol, source_types, min_priority, **kwargs)
        return [r.to_rag_document() for r in results]

    async def fetch_economic_calendar(
        self,
        symbol: Optional[str] = None,
        days_ahead: int = 7,
        days_back: int = 1
    ) -> list[DataSourceResult]:
        """Fetch economic calendar events."""
        return await self._sources[DataSourceType.ECONOMIC_CALENDAR].fetch(
            symbol, days_ahead=days_ahead, days_back=days_back
        )

    async def fetch_onchain(
        self,
        symbol: str,
        include_whale_alerts: bool = True,
        include_exchange_flows: bool = True,
        include_mining: bool = True,
        include_defi: bool = True
    ) -> list[DataSourceResult]:
        """Fetch on-chain data for a cryptocurrency."""
        return await self._sources[DataSourceType.ONCHAIN].fetch(
            symbol,
            include_whale_alerts=include_whale_alerts,
            include_exchange_flows=include_exchange_flows,
            include_mining=include_mining,
            include_defi=include_defi
        )

    async def fetch_sentiment(
        self,
        symbol: Optional[str] = None,
        include_fear_greed: bool = True,
        include_social: bool = True,
        include_options: bool = True,
        include_volatility: bool = True
    ) -> list[DataSourceResult]:
        """Fetch sentiment data."""
        return await self._sources[DataSourceType.SENTIMENT].fetch(
            symbol,
            include_fear_greed=include_fear_greed,
            include_social=include_social,
            include_options=include_options,
            include_volatility=include_volatility
        )

    async def fetch_orderbook(
        self,
        symbol: str,
        depth: int = 50,
        include_liquidations: bool = True,
        include_cvd: bool = True
    ) -> list[DataSourceResult]:
        """Fetch orderbook and liquidity data."""
        return await self._sources[DataSourceType.ORDERBOOK].fetch(
            symbol,
            depth=depth,
            include_liquidations=include_liquidations,
            include_cvd=include_cvd
        )

    async def fetch_macro(
        self,
        symbol: Optional[str] = None,
        include_dxy: bool = True,
        include_bonds: bool = True,
        include_correlations: bool = True,
        include_sectors: bool = True
    ) -> list[DataSourceResult]:
        """Fetch macro and correlation data."""
        return await self._sources[DataSourceType.MACRO_CORRELATION].fetch(
            symbol,
            include_dxy=include_dxy,
            include_bonds=include_bonds,
            include_correlations=include_correlations,
            include_sectors=include_sectors
        )

    async def fetch_historical_patterns(
        self,
        symbol: Optional[str] = None,
        include_seasonality: bool = True,
        include_drawdowns: bool = True,
        include_events: bool = True,
        include_comparable: bool = True
    ) -> list[DataSourceResult]:
        """Fetch historical pattern data."""
        return await self._sources[DataSourceType.HISTORICAL_PATTERN].fetch(
            symbol,
            include_seasonality=include_seasonality,
            include_drawdowns=include_drawdowns,
            include_events=include_events,
            include_comparable=include_comparable
        )

    async def fetch_technical_levels(
        self,
        symbol: str,
        include_sr: bool = True,
        include_fib: bool = True,
        include_pivots: bool = True,
        include_vwap: bool = True,
        include_ma: bool = True
    ) -> list[DataSourceResult]:
        """Fetch technical price levels."""
        return await self._sources[DataSourceType.TECHNICAL_LEVEL].fetch(
            symbol,
            include_sr=include_sr,
            include_fib=include_fib,
            include_pivots=include_pivots,
            include_vwap=include_vwap,
            include_ma=include_ma
        )

    async def fetch_regulatory(
        self,
        symbol: Optional[str] = None,
        include_sec: bool = True,
        include_etf: bool = True,
        include_global: bool = True,
        include_enforcement: bool = True
    ) -> list[DataSourceResult]:
        """Fetch regulatory updates."""
        return await self._sources[DataSourceType.REGULATORY].fetch(
            symbol,
            include_sec=include_sec,
            include_etf=include_etf,
            include_global=include_global,
            include_enforcement=include_enforcement
        )

    async def fetch_trading_context(
        self,
        symbol: str,
        include_types: Optional[list[str]] = None
    ) -> dict:
        """
        Fetch comprehensive trading context for a symbol.

        This method provides a complete market view by fetching
        relevant data from multiple sources and organizing it
        into a structured format.

        Args:
            symbol: Trading symbol
            include_types: Optional list of data types to include:
                - 'economic': Economic calendar events
                - 'onchain': On-chain metrics (crypto only)
                - 'sentiment': Market sentiment
                - 'orderbook': Orderbook/liquidity data
                - 'macro': Macro correlations
                - 'patterns': Historical patterns
                - 'levels': Technical levels
                - 'regulatory': Regulatory updates
                Default: All types

        Returns:
            Dict with organized trading context
        """
        all_types = [
            'economic', 'onchain', 'sentiment', 'orderbook',
            'macro', 'patterns', 'levels', 'regulatory'
        ]
        types_to_fetch = include_types or all_types

        context = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'data': {},
            'summary': {
                'critical_events': [],
                'high_priority_items': [],
                'key_levels': [],
                'sentiment_overview': None
            }
        }

        # Map types to source types and fetch methods
        type_mapping = {
            'economic': (DataSourceType.ECONOMIC_CALENDAR, self.fetch_economic_calendar),
            'onchain': (DataSourceType.ONCHAIN, self.fetch_onchain),
            'sentiment': (DataSourceType.SENTIMENT, self.fetch_sentiment),
            'orderbook': (DataSourceType.ORDERBOOK, self.fetch_orderbook),
            'macro': (DataSourceType.MACRO_CORRELATION, self.fetch_macro),
            'patterns': (DataSourceType.HISTORICAL_PATTERN, self.fetch_historical_patterns),
            'levels': (DataSourceType.TECHNICAL_LEVEL, self.fetch_technical_levels),
            'regulatory': (DataSourceType.REGULATORY, self.fetch_regulatory),
        }

        # Fetch in parallel
        tasks = {}
        for type_name in types_to_fetch:
            if type_name in type_mapping:
                source_type, fetch_method = type_mapping[type_name]
                if type_name in ['orderbook', 'levels', 'onchain']:
                    # These require symbol
                    tasks[type_name] = fetch_method(symbol)
                else:
                    tasks[type_name] = fetch_method(symbol)

        results = await asyncio.gather(
            *[tasks[k] for k in tasks.keys()],
            return_exceptions=True
        )

        # Process results
        for type_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {type_name}: {result}")
                context['data'][type_name] = {'error': str(result)}
            else:
                context['data'][type_name] = [r.to_rag_document() for r in result]

                # Extract critical and high priority items
                for item in result:
                    if item.priority == DataPriority.CRITICAL:
                        context['summary']['critical_events'].append({
                            'source': type_name,
                            'content': item.content[:500] + '...' if len(item.content) > 500 else item.content
                        })
                    elif item.priority == DataPriority.HIGH:
                        context['summary']['high_priority_items'].append({
                            'source': type_name,
                            'type': item.metadata.get('metric_type', 'unknown')
                        })

        return context

    async def create_rag_documents_batch(
        self,
        symbol: Optional[str] = None,
        source_types: Optional[list[DataSourceType]] = None
    ) -> list[dict]:
        """
        Create a batch of RAG documents for storage.

        This is the main method for populating the RAG database
        with fresh data from all sources.

        Args:
            symbol: Optional symbol filter
            source_types: Optional list of source types

        Returns:
            List of RAG documents ready for storage
        """
        results = await self.fetch_all(symbol, source_types)

        documents = []
        for result in results:
            doc = result.to_rag_document()
            documents.append(doc)

        logger.info(f"Created {len(documents)} RAG documents")
        return documents

    def get_available_sources(self) -> list[dict]:
        """Get information about available data sources."""
        return [
            {
                'type': source_type.value,
                'name': source_type.name,
                'description': self._get_source_description(source_type)
            }
            for source_type in self._sources.keys()
        ]

    def _get_source_description(self, source_type: DataSourceType) -> str:
        """Get description for a data source type."""
        descriptions = {
            DataSourceType.ECONOMIC_CALENDAR: "Economic calendar events (Fed, CPI, NFP, etc.)",
            DataSourceType.ONCHAIN: "On-chain crypto metrics (whale alerts, exchange flows, mining)",
            DataSourceType.SENTIMENT: "Market sentiment (Fear & Greed, social, options, VIX)",
            DataSourceType.ORDERBOOK: "Orderbook data (walls, liquidations, CVD, order flow)",
            DataSourceType.MACRO_CORRELATION: "Macro data (DXY, bonds, correlations, sectors)",
            DataSourceType.HISTORICAL_PATTERN: "Historical patterns (seasonality, drawdowns, events)",
            DataSourceType.TECHNICAL_LEVEL: "Technical levels (S/R, Fibonacci, pivots, VWAP, MAs)",
            DataSourceType.REGULATORY: "Regulatory updates (SEC, ETFs, global regulation)",
        }
        return descriptions.get(source_type, "Unknown data source")


# Singleton instance
_fetcher_service: Optional[DataFetcherService] = None


def get_data_fetcher_service() -> DataFetcherService:
    """Get or create the singleton DataFetcherService instance."""
    global _fetcher_service
    if _fetcher_service is None:
        _fetcher_service = DataFetcherService()
    return _fetcher_service
