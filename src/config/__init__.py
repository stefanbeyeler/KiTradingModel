"""Configuration module for KI Trading Model."""

from .settings import Settings, settings
from .output_schema import (
    TradeDirection,
    TradingAnalysisOutput,
    QuickRecommendationOutput,
    MultiSymbolOutput,
    OUTPUT_SCHEMA_JSON,
    get_output_schema_prompt,
)

__all__ = [
    "Settings",
    "settings",
    "TradeDirection",
    "TradingAnalysisOutput",
    "QuickRecommendationOutput",
    "MultiSymbolOutput",
    "OUTPUT_SCHEMA_JSON",
    "get_output_schema_prompt",
]
