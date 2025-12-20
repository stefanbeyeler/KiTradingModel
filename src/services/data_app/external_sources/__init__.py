"""External Data Sources - Central gateway for all external data integrations.

This module provides access to various external data sources for trading intelligence.
All external data fetching is centralized here in the Data Service as per architecture rules.
"""

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority
from .economic_calendar import EconomicCalendarSource
from .onchain_data import OnChainDataSource
from .sentiment_data import SentimentDataSource
from .orderbook_data import OrderbookDataSource
from .macro_correlation import MacroCorrelationSource
from .historical_patterns import HistoricalPatternsSource
from .technical_levels import TechnicalLevelsSource
from .regulatory_updates import RegulatoryUpdatesSource
from .easyinsight_data import EasyInsightDataSource
from .data_fetcher_service import DataFetcherService, get_data_fetcher_service

__all__ = [
    "DataSourceBase",
    "DataSourceResult",
    "DataSourceType",
    "DataPriority",
    "EconomicCalendarSource",
    "OnChainDataSource",
    "SentimentDataSource",
    "OrderbookDataSource",
    "MacroCorrelationSource",
    "HistoricalPatternsSource",
    "TechnicalLevelsSource",
    "RegulatoryUpdatesSource",
    "EasyInsightDataSource",
    "DataFetcherService",
    "get_data_fetcher_service",
]
