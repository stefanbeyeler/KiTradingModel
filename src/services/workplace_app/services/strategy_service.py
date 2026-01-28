"""
Strategy Service - Management of trading strategies.

Verwaltet Trading-Strategien mit JSON-Persistenz.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from ..config import settings


# ============================================
# Enums und Models (lokal definiert)
# ============================================

from enum import Enum
from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    """Types of trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class IndicatorConfig(BaseModel):
    """Configuration for a technical indicator."""
    name: str
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=5.0)
    parameters: dict = Field(default_factory=dict)
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None


class TradingStrategy(BaseModel):
    """Trading strategy configuration."""
    id: str
    name: str
    description: str = ""
    strategy_type: StrategyType = StrategyType.CUSTOM
    risk_level: RiskLevel = RiskLevel.MODERATE

    # Indicator configurations
    indicators: list[IndicatorConfig] = Field(default_factory=list)

    # Signal thresholds
    min_buy_signals: int = Field(default=3, ge=1, le=10)
    min_sell_signals: int = Field(default=3, ge=1, le=10)

    # Risk management
    stop_loss_atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    take_profit_atr_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    max_position_size_percent: float = Field(default=5.0, ge=1.0, le=100.0)

    # Timeframe preferences
    preferred_timeframe: str = "short_term"
    lookback_days: int = Field(default=30, ge=7, le=365)

    # LLM prompt customization
    custom_prompt: Optional[str] = None
    use_rag_context: bool = True
    max_rag_documents: int = Field(default=5, ge=0, le=20)

    # Metadata
    is_active: bool = True
    is_default: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class StrategyCreateRequest(BaseModel):
    """Request to create a new trading strategy."""
    name: str
    description: str = ""
    strategy_type: StrategyType = StrategyType.CUSTOM
    risk_level: RiskLevel = RiskLevel.MODERATE
    indicators: list[IndicatorConfig] = Field(default_factory=list)
    min_buy_signals: int = Field(default=3, ge=1, le=10)
    min_sell_signals: int = Field(default=3, ge=1, le=10)
    stop_loss_atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    take_profit_atr_multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    preferred_timeframe: str = "short_term"
    lookback_days: int = Field(default=30, ge=7, le=365)
    custom_prompt: Optional[str] = None
    use_rag_context: bool = True


class StrategyUpdateRequest(BaseModel):
    """Request to update an existing trading strategy."""
    name: Optional[str] = None
    description: Optional[str] = None
    strategy_type: Optional[StrategyType] = None
    risk_level: Optional[RiskLevel] = None
    indicators: Optional[list[IndicatorConfig]] = None
    min_buy_signals: Optional[int] = None
    min_sell_signals: Optional[int] = None
    stop_loss_atr_multiplier: Optional[float] = None
    take_profit_atr_multiplier: Optional[float] = None
    preferred_timeframe: Optional[str] = None
    lookback_days: Optional[int] = None
    custom_prompt: Optional[str] = None
    use_rag_context: Optional[bool] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None


# ============================================
# Strategy Service
# ============================================

class StrategyService:
    """Service for managing trading strategies."""

    def __init__(self):
        self._strategies: dict[str, TradingStrategy] = {}
        self._storage_path = os.path.join(settings.strategies_file)
        self._load_strategies()
        self._ensure_default_strategies()

    def _load_strategies(self):
        """Load strategies from disk."""
        if os.path.exists(self._storage_path):
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for strategy_data in data:
                        strategy = TradingStrategy(**strategy_data)
                        self._strategies[strategy.id] = strategy
                logger.info(f"Loaded {len(self._strategies)} strategies from disk")
            except Exception as e:
                logger.error(f"Failed to load strategies: {e}")
                self._strategies = {}

    def _save_strategies(self):
        """Save strategies to disk."""
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        try:
            data = [s.model_dump(mode="json") for s in self._strategies.values()]
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved {len(self._strategies)} strategies to disk")
        except Exception as e:
            logger.error(f"Failed to save strategies: {e}")

    def _ensure_default_strategies(self):
        """Ensure default strategies exist."""
        if not self._strategies:
            self._create_default_strategies()
            self._save_strategies()

    def _create_default_strategies(self):
        """Create built-in default strategies."""

        # 1. Trend Following Strategy
        trend_following = TradingStrategy(
            id="default_trend_following",
            name="Trend Following",
            description="Folgt etablierten Trends mit Moving Averages und MACD. Geeignet für längere Haltedauern.",
            strategy_type=StrategyType.TREND_FOLLOWING,
            risk_level=RiskLevel.MODERATE,
            indicators=[
                IndicatorConfig(name="SMA_20", enabled=True, weight=1.0),
                IndicatorConfig(name="SMA_50", enabled=True, weight=1.5),
                IndicatorConfig(name="SMA_200", enabled=True, weight=2.0),
                IndicatorConfig(name="MACD", enabled=True, weight=1.5),
                IndicatorConfig(name="RSI", enabled=True, weight=0.5, buy_threshold=40, sell_threshold=60),
            ],
            min_buy_signals=3,
            min_sell_signals=3,
            stop_loss_atr_multiplier=2.5,
            take_profit_atr_multiplier=4.0,
            preferred_timeframe="medium_term",
            lookback_days=60,
            custom_prompt="Fokussiere auf langfristige Trends und Moving Average Crossovers.",
            is_default=True,
        )
        self._strategies[trend_following.id] = trend_following

        # 2. Mean Reversion Strategy
        mean_reversion = TradingStrategy(
            id="default_mean_reversion",
            name="Mean Reversion",
            description="Kauft bei Überverkauft, verkauft bei Überkauft. Nutzt RSI und Bollinger Bands.",
            strategy_type=StrategyType.MEAN_REVERSION,
            risk_level=RiskLevel.MODERATE,
            indicators=[
                IndicatorConfig(name="RSI", enabled=True, weight=2.0, buy_threshold=30, sell_threshold=70),
                IndicatorConfig(name="Bollinger_Bands", enabled=True, weight=2.0),
                IndicatorConfig(name="SMA_20", enabled=True, weight=1.0),
            ],
            min_buy_signals=2,
            min_sell_signals=2,
            stop_loss_atr_multiplier=1.5,
            take_profit_atr_multiplier=2.0,
            preferred_timeframe="short_term",
            lookback_days=20,
            custom_prompt="Suche nach extremen RSI-Werten und Preisen nahe den Bollinger Bändern.",
        )
        self._strategies[mean_reversion.id] = mean_reversion

        # 3. Momentum Strategy
        momentum = TradingStrategy(
            id="default_momentum",
            name="Momentum",
            description="Nutzt starke Preisbewegungen. Aggressiver Ansatz mit höherem Risiko.",
            strategy_type=StrategyType.MOMENTUM,
            risk_level=RiskLevel.AGGRESSIVE,
            indicators=[
                IndicatorConfig(name="RSI", enabled=True, weight=1.5, buy_threshold=50, sell_threshold=50),
                IndicatorConfig(name="MACD", enabled=True, weight=2.0),
                IndicatorConfig(name="EMA_12", enabled=True, weight=1.5),
                IndicatorConfig(name="EMA_26", enabled=True, weight=1.0),
            ],
            min_buy_signals=2,
            min_sell_signals=2,
            stop_loss_atr_multiplier=1.5,
            take_profit_atr_multiplier=3.0,
            preferred_timeframe="short_term",
            lookback_days=14,
            custom_prompt="Fokussiere auf starke Momentum-Signale und MACD-Divergenzen.",
        )
        self._strategies[momentum.id] = momentum

        # 4. Conservative Strategy
        conservative = TradingStrategy(
            id="default_conservative",
            name="Konservativ",
            description="Vorsichtiger Ansatz mit strengen Signalkriterien. Minimiert Risiko.",
            strategy_type=StrategyType.SWING,
            risk_level=RiskLevel.CONSERVATIVE,
            indicators=[
                IndicatorConfig(name="SMA_50", enabled=True, weight=1.5),
                IndicatorConfig(name="SMA_200", enabled=True, weight=2.0),
                IndicatorConfig(name="RSI", enabled=True, weight=1.0, buy_threshold=25, sell_threshold=75),
                IndicatorConfig(name="MACD", enabled=True, weight=1.0),
                IndicatorConfig(name="Bollinger_Bands", enabled=True, weight=1.0),
            ],
            min_buy_signals=4,
            min_sell_signals=4,
            stop_loss_atr_multiplier=3.0,
            take_profit_atr_multiplier=2.0,
            preferred_timeframe="long_term",
            lookback_days=90,
            custom_prompt="Nur bei starker Konfluenz mehrerer Indikatoren handeln.",
        )
        self._strategies[conservative.id] = conservative

        # 5. Scalping Strategy
        scalping = TradingStrategy(
            id="default_scalping",
            name="Scalping",
            description="Kurzfristige Trades mit schnellen Ein- und Ausstiegen. Für aktive Trader.",
            strategy_type=StrategyType.SCALPING,
            risk_level=RiskLevel.AGGRESSIVE,
            indicators=[
                IndicatorConfig(name="RSI", enabled=True, weight=2.0, buy_threshold=35, sell_threshold=65),
                IndicatorConfig(name="EMA_12", enabled=True, weight=1.5),
                IndicatorConfig(name="Bollinger_Bands", enabled=True, weight=1.5),
            ],
            min_buy_signals=2,
            min_sell_signals=2,
            stop_loss_atr_multiplier=1.0,
            take_profit_atr_multiplier=1.5,
            preferred_timeframe="short_term",
            lookback_days=7,
            custom_prompt="Kurze Haltedauer mit engen Stop-Loss und Take-Profit Levels.",
        )
        self._strategies[scalping.id] = scalping

        logger.info(f"Created {len(self._strategies)} default strategies")

    async def get_all_strategies(self, include_inactive: bool = False) -> list[TradingStrategy]:
        """Get all trading strategies."""
        strategies = list(self._strategies.values())
        if not include_inactive:
            strategies = [s for s in strategies if s.is_active]
        return sorted(strategies, key=lambda s: (not s.is_default, s.name))

    async def get_strategy(self, strategy_id: str) -> Optional[TradingStrategy]:
        """Get a specific strategy by ID."""
        return self._strategies.get(strategy_id)

    async def get_default_strategy(self) -> Optional[TradingStrategy]:
        """Get the default strategy."""
        for strategy in self._strategies.values():
            if strategy.is_default and strategy.is_active:
                return strategy
        # Fallback to first active strategy
        active = [s for s in self._strategies.values() if s.is_active]
        return active[0] if active else None

    async def create_strategy(self, request: StrategyCreateRequest) -> TradingStrategy:
        """Create a new trading strategy."""
        strategy_id = f"custom_{uuid.uuid4().hex[:8]}"

        strategy = TradingStrategy(
            id=strategy_id,
            name=request.name,
            description=request.description,
            strategy_type=request.strategy_type,
            risk_level=request.risk_level,
            indicators=request.indicators if request.indicators else self._get_default_indicators(request.strategy_type),
            min_buy_signals=request.min_buy_signals,
            min_sell_signals=request.min_sell_signals,
            stop_loss_atr_multiplier=request.stop_loss_atr_multiplier,
            take_profit_atr_multiplier=request.take_profit_atr_multiplier,
            preferred_timeframe=request.preferred_timeframe,
            lookback_days=request.lookback_days,
            custom_prompt=request.custom_prompt,
            use_rag_context=request.use_rag_context,
        )

        self._strategies[strategy.id] = strategy
        self._save_strategies()
        logger.info(f"Created strategy: {strategy.name} ({strategy.id})")
        return strategy

    async def update_strategy(self, strategy_id: str, request: StrategyUpdateRequest) -> Optional[TradingStrategy]:
        """Update an existing strategy."""
        if strategy_id not in self._strategies:
            return None

        strategy = self._strategies[strategy_id]
        update_data = request.model_dump(exclude_unset=True)

        # Handle setting new default
        if update_data.get("is_default"):
            for s in self._strategies.values():
                s.is_default = False

        # Update fields
        for field, value in update_data.items():
            if hasattr(strategy, field):
                setattr(strategy, field, value)

        strategy.updated_at = datetime.utcnow()
        self._save_strategies()
        logger.info(f"Updated strategy: {strategy.name} ({strategy.id})")
        return strategy

    async def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy (only custom strategies can be deleted)."""
        if strategy_id not in self._strategies:
            return False

        strategy = self._strategies[strategy_id]
        if strategy_id.startswith("default_"):
            logger.warning(f"Cannot delete default strategy: {strategy_id}")
            return False

        del self._strategies[strategy_id]
        self._save_strategies()
        logger.info(f"Deleted strategy: {strategy.name} ({strategy_id})")
        return True

    async def set_default_strategy(self, strategy_id: str) -> Optional[TradingStrategy]:
        """Set a strategy as the default."""
        if strategy_id not in self._strategies:
            return None

        for s in self._strategies.values():
            s.is_default = False

        strategy = self._strategies[strategy_id]
        strategy.is_default = True
        strategy.updated_at = datetime.utcnow()
        self._save_strategies()
        logger.info(f"Set default strategy: {strategy.name}")
        return strategy

    def _get_default_indicators(self, strategy_type: StrategyType) -> list[IndicatorConfig]:
        """Get default indicators for a strategy type."""
        if strategy_type == StrategyType.TREND_FOLLOWING:
            return [
                IndicatorConfig(name="SMA_20", enabled=True, weight=1.0),
                IndicatorConfig(name="SMA_50", enabled=True, weight=1.5),
                IndicatorConfig(name="MACD", enabled=True, weight=1.5),
            ]
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return [
                IndicatorConfig(name="RSI", enabled=True, weight=2.0, buy_threshold=30, sell_threshold=70),
                IndicatorConfig(name="Bollinger_Bands", enabled=True, weight=2.0),
            ]
        elif strategy_type == StrategyType.MOMENTUM:
            return [
                IndicatorConfig(name="RSI", enabled=True, weight=1.5),
                IndicatorConfig(name="MACD", enabled=True, weight=2.0),
            ]
        else:
            return [
                IndicatorConfig(name="RSI", enabled=True, weight=1.0),
                IndicatorConfig(name="MACD", enabled=True, weight=1.0),
                IndicatorConfig(name="SMA_20", enabled=True, weight=1.0),
            ]

    def export_strategy_to_markdown(self, strategy: TradingStrategy) -> str:
        """Export a strategy to Markdown format."""
        lines = []

        # Header
        lines.append(f"# Trading Strategie: {strategy.name}")
        lines.append("")
        lines.append(f"**ID:** `{strategy.id}`  ")
        lines.append(f"**Typ:** {strategy.strategy_type.value}  ")
        lines.append(f"**Risikolevel:** {strategy.risk_level.value}  ")
        lines.append(f"**Standard:** {'Ja' if strategy.is_default else 'Nein'}  ")
        lines.append(f"**Aktiv:** {'Ja' if strategy.is_active else 'Nein'}  ")
        lines.append("")

        # Description
        lines.append("## Beschreibung")
        lines.append("")
        lines.append(strategy.description if strategy.description else "*Keine Beschreibung*")
        lines.append("")

        # Indicators
        lines.append("## Indikatoren")
        lines.append("")
        if strategy.indicators:
            lines.append("| Indikator | Aktiv | Gewichtung | Kauf-Schwelle | Verkauf-Schwelle |")
            lines.append("|-----------|-------|------------|---------------|------------------|")
            for ind in strategy.indicators:
                buy_th = str(ind.buy_threshold) if ind.buy_threshold is not None else "-"
                sell_th = str(ind.sell_threshold) if ind.sell_threshold is not None else "-"
                lines.append(f"| {ind.name} | {'ja' if ind.enabled else 'nein'} | {ind.weight} | {buy_th} | {sell_th} |")
        else:
            lines.append("*Keine Indikatoren konfiguriert*")
        lines.append("")

        # Signal Settings
        lines.append("## Signal-Einstellungen")
        lines.append("")
        lines.append(f"- **Min. Kaufsignale:** {strategy.min_buy_signals}")
        lines.append(f"- **Min. Verkaufssignale:** {strategy.min_sell_signals}")
        lines.append("")

        # Risk Management
        lines.append("## Risikomanagement")
        lines.append("")
        lines.append(f"- **Stop-Loss ATR-Multiplikator:** {strategy.stop_loss_atr_multiplier}")
        lines.append(f"- **Take-Profit ATR-Multiplikator:** {strategy.take_profit_atr_multiplier}")
        lines.append(f"- **Max. Positionsgroesse:** {strategy.max_position_size_percent}%")
        lines.append("")

        # Timeframe
        lines.append("## Zeitrahmen")
        lines.append("")
        lines.append(f"- **Bevorzugter Zeitrahmen:** {strategy.preferred_timeframe}")
        lines.append(f"- **Lookback-Tage:** {strategy.lookback_days}")
        lines.append("")

        # Custom Prompt
        lines.append("## Benutzerdefinierter Prompt")
        lines.append("")
        if strategy.custom_prompt:
            lines.append("```")
            lines.append(strategy.custom_prompt)
            lines.append("```")
        else:
            lines.append("*Kein benutzerdefinierter Prompt*")
        lines.append("")

        # Metadata
        lines.append("---")
        lines.append("")
        lines.append(f"*Erstellt:* {strategy.created_at.isoformat()}  ")
        lines.append(f"*Aktualisiert:* {strategy.updated_at.isoformat()}")

        return "\n".join(lines)

    async def import_strategy(self, data: dict) -> TradingStrategy:
        """
        Import a complete strategy from dict (for config import).

        If strategy with same ID exists, it will be updated.
        Otherwise, a new strategy will be created.
        """
        strategy_id = data.get("id")

        if not strategy_id:
            # Generate new ID if not provided
            strategy_id = f"imported_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            data["id"] = strategy_id

        # Parse indicators if they exist
        indicators = []
        for ind_data in data.get("indicators", []):
            if isinstance(ind_data, dict):
                indicators.append(IndicatorConfig(**ind_data))
            elif isinstance(ind_data, IndicatorConfig):
                indicators.append(ind_data)

        # Parse enums
        strategy_type = data.get("strategy_type", "custom")
        if isinstance(strategy_type, str):
            try:
                strategy_type = StrategyType(strategy_type.lower())
            except ValueError:
                strategy_type = StrategyType.CUSTOM

        risk_level = data.get("risk_level", "moderate")
        if isinstance(risk_level, str):
            try:
                risk_level = RiskLevel(risk_level.lower())
            except ValueError:
                risk_level = RiskLevel.MODERATE

        # Create strategy object
        strategy = TradingStrategy(
            id=strategy_id,
            name=data.get("name", "Imported Strategy"),
            description=data.get("description", ""),
            strategy_type=strategy_type,
            risk_level=risk_level,
            indicators=indicators,
            min_buy_signals=data.get("min_buy_signals", 3),
            min_sell_signals=data.get("min_sell_signals", 3),
            stop_loss_atr_multiplier=data.get("stop_loss_atr_multiplier", 2.0),
            take_profit_atr_multiplier=data.get("take_profit_atr_multiplier", 3.0),
            max_position_size_percent=data.get("max_position_size_percent", 5.0),
            preferred_timeframe=data.get("preferred_timeframe", "short_term"),
            lookback_days=data.get("lookback_days", 30),
            custom_prompt=data.get("custom_prompt"),
            use_rag_context=data.get("use_rag_context", True),
            is_active=data.get("is_active", True),
            is_default=data.get("is_default", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Store strategy
        self._strategies[strategy_id] = strategy
        self._save_strategies()
        logger.info(f"Imported strategy: {strategy.name} (ID: {strategy_id})")

        return strategy


# Singleton-Instanz
strategy_service = StrategyService()
