"""Output schema configuration for trading analysis responses."""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TradeDirection(str, Enum):
    """Trading direction enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TradingAnalysisOutput(BaseModel):
    """
    Schema for trading analysis output.

    This defines the structure of the response returned by the KI trading model
    when analyzing a symbol for trading opportunities.
    """

    # Trade Direction and Confidence
    direction: TradeDirection = Field(
        description="Trading direction recommendation: LONG, SHORT, or NEUTRAL"
    )
    confidence_score: int = Field(
        ge=0,
        le=100,
        description="Confidence score from 0-100 indicating strength of the recommendation"
    )

    # Setup Description
    setup_recommendation: str = Field(
        description="Detailed description of the trading setup and why it's valid or not"
    )

    # Price Levels
    entry_price: float = Field(
        ge=0,
        description="Recommended entry price for the trade"
    )
    stop_loss: float = Field(
        ge=0,
        description="Stop loss price level to limit potential losses"
    )
    take_profit_1: float = Field(
        ge=0,
        description="First take profit target (conservative)"
    )
    take_profit_2: float = Field(
        ge=0,
        description="Second take profit target (moderate)"
    )
    take_profit_3: float = Field(
        ge=0,
        description="Third take profit target (aggressive)"
    )

    # Risk Management
    risk_reward_ratio: float = Field(
        ge=0,
        description="Risk-to-reward ratio (e.g., 2.5 means 2.5:1 reward to risk)"
    )
    recommended_position_size: float = Field(
        ge=0,
        description="Recommended position size in lots"
    )
    max_risk_percent: float = Field(
        ge=0,
        le=100,
        description="Maximum risk as percentage of account balance"
    )

    # Analysis Details
    trend_analysis: str = Field(
        description="Detailed analysis of current trend direction, strength, and multi-timeframe alignment"
    )
    support_resistance: str = Field(
        description="Key support and resistance levels with distances from current price"
    )
    key_levels: str = Field(
        description="Important price levels to watch (psychological levels, Fibonacci, etc.)"
    )
    risk_factors: str = Field(
        description="Potential risks and what could invalidate this trade setup"
    )
    trade_rationale: str = Field(
        description="Why this trade makes sense according to the trading strategy"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "direction": "LONG",
                "confidence_score": 75,
                "setup_recommendation": "Strong bullish setup with RSI oversold bounce and MACD crossover",
                "entry_price": 1.08500,
                "stop_loss": 1.08200,
                "take_profit_1": 1.08800,
                "take_profit_2": 1.09100,
                "take_profit_3": 1.09500,
                "risk_reward_ratio": 2.5,
                "recommended_position_size": 0.10,
                "max_risk_percent": 1.0,
                "trend_analysis": "D1 bullish trend, H1 pullback completed, M15 showing reversal signals",
                "support_resistance": "Support at 1.0820, Resistance at 1.0900 and 1.0950",
                "key_levels": "Psychological level 1.0900, Fibonacci 61.8% at 1.0875",
                "risk_factors": "ECB speech tomorrow, potential volatility spike",
                "trade_rationale": "Trend continuation trade with favorable risk-reward"
            }
        }


class QuickRecommendationOutput(BaseModel):
    """
    Simplified output schema for quick recommendations.

    Used for fast analysis endpoints that return basic trading signals.
    """

    direction: TradeDirection = Field(
        description="Trading direction: LONG, SHORT, or NEUTRAL"
    )
    confidence_score: int = Field(
        ge=0,
        le=100,
        description="Confidence score from 0-100"
    )
    entry_price: float = Field(
        ge=0,
        description="Recommended entry price"
    )
    stop_loss: float = Field(
        ge=0,
        description="Stop loss price level"
    )
    take_profit_1: float = Field(
        ge=0,
        description="Primary take profit target"
    )
    risk_reward_ratio: float = Field(
        ge=0,
        description="Risk-to-reward ratio"
    )
    summary: str = Field(
        description="Brief summary of the trading setup"
    )


class MultiSymbolOutput(BaseModel):
    """
    Output schema for multi-symbol analysis.

    Contains analysis results for multiple trading symbols.
    """

    timestamp: str = Field(
        description="ISO timestamp when analysis was performed"
    )
    analyses: List[TradingAnalysisOutput] = Field(
        description="List of trading analyses for each symbol"
    )
    best_opportunity: Optional[str] = Field(
        default=None,
        description="Symbol with the highest confidence score"
    )
    market_overview: str = Field(
        description="General market conditions overview"
    )


# JSON Schema for external use (e.g., LLM prompts)
OUTPUT_SCHEMA_JSON = {
    "direction": "LONG | SHORT | NEUTRAL",
    "confidence_score": "0-100",
    "setup_recommendation": "Detailed description of the trading setup and why it's valid or not",
    "entry_price": 0.00000,
    "stop_loss": 0.00000,
    "take_profit_1": 0.00000,
    "take_profit_2": 0.00000,
    "take_profit_3": 0.00000,
    "risk_reward_ratio": 0.0,
    "recommended_position_size": 0.00,
    "max_risk_percent": 0.0,
    "trend_analysis": "Detailed analysis of current trend direction, strength, and multi-timeframe alignment",
    "support_resistance": "Key support and resistance levels with distances from current price",
    "key_levels": "Important price levels to watch (psychological levels, Fibonacci, etc.)",
    "risk_factors": "Potential risks and what could invalidate this trade setup",
    "trade_rationale": "Why this trade makes sense according to the trading strategy"
}


def get_output_schema_prompt() -> str:
    """
    Returns the output schema as a formatted string for LLM prompts.

    This can be included in the system prompt to ensure the LLM
    returns responses in the correct format.
    """
    import json
    return f"""
Du musst deine Analyse im folgenden JSON-Format zurückgeben:

```json
{json.dumps(OUTPUT_SCHEMA_JSON, indent=2, ensure_ascii=False)}
```

Wichtige Hinweise:
- direction: Muss exakt "LONG", "SHORT" oder "NEUTRAL" sein
- confidence_score: Ganzzahl zwischen 0 und 100
- Alle Preise mit 5 Dezimalstellen für Forex, 2 für Indizes
- risk_reward_ratio: Mindestens 1.5 für gültige Setups
- max_risk_percent: Maximal 2% pro Trade empfohlen
"""
