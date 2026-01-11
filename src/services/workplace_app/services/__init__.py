"""Services für Trading Workplace."""

from .signal_aggregator import SignalAggregatorService, signal_aggregator
from .scoring_service import ScoringService, scoring_service
from .watchlist_service import WatchlistService, watchlist_service
from .scanner_service import ScannerService, scanner_service
from .deep_analysis_service import DeepAnalysisService, deep_analysis_service

__all__ = [
    "SignalAggregatorService",
    "signal_aggregator",
    "ScoringService",
    "scoring_service",
    "WatchlistService",
    "watchlist_service",
    "ScannerService",
    "scanner_service",
    "DeepAnalysisService",
    "deep_analysis_service",
]
