"""Strategy Service - Management of trading strategies."""

import json
import os
import uuid
from datetime import datetime
from typing import Optional
from loguru import logger

from ..config import settings
from ..models.trading_data import (
    TradingStrategy,
    StrategyType,
    RiskLevel,
    IndicatorConfig,
    StrategyCreateRequest,
    StrategyUpdateRequest,
)


class StrategyService:
    """Service for managing trading strategies."""

    def __init__(self):
        self._strategies: dict[str, TradingStrategy] = {}
        self._storage_path = os.path.join(settings.faiss_persist_directory, "strategies.json")
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
            logger.info(f"Saved {len(self._strategies)} strategies to disk")
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
            custom_prompt="Fokussiere auf langfristige Trends und Moving Average Crossovers. Ignoriere kurzfristige Schwankungen.",
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
            custom_prompt="Suche nach extremen RSI-Werten und Preisen nahe den Bollinger Bändern für Umkehrsignale.",
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
            custom_prompt="Fokussiere auf starke Momentum-Signale. MACD-Histogramm und RSI-Divergenzen sind besonders wichtig.",
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
            custom_prompt="Sei sehr vorsichtig mit Empfehlungen. Nur bei starker Konfluenz mehrerer Indikatoren handeln.",
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
            custom_prompt="Kurze Haltedauer. Schnelle Reaktion auf Signale. Enge Stop-Loss und Take-Profit Levels.",
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

    def get_strategy_prompt(self, strategy: TradingStrategy) -> str:
        """Generate a custom prompt based on the strategy configuration."""
        prompt_parts = []

        # Strategy type description
        type_descriptions = {
            StrategyType.TREND_FOLLOWING: "Fokussiere auf Trendfolge-Signale und Moving Average Crossovers.",
            StrategyType.MEAN_REVERSION: "Suche nach Umkehrsignalen bei überkauften/überverkauften Bedingungen.",
            StrategyType.MOMENTUM: "Nutze starke Momentum-Signale für schnelle Trades.",
            StrategyType.BREAKOUT: "Identifiziere Ausbrüche aus Konsolidierungszonen.",
            StrategyType.SCALPING: "Kurze Haltedauer mit engen Stop-Loss und Take-Profit Levels.",
            StrategyType.SWING: "Mittelfristige Trades mit ausgewogenem Risiko-Rendite-Verhältnis.",
            StrategyType.CUSTOM: "",
        }

        if strategy.strategy_type in type_descriptions:
            prompt_parts.append(type_descriptions[strategy.strategy_type])

        # Risk level
        risk_descriptions = {
            RiskLevel.CONSERVATIVE: "Sei sehr vorsichtig und empfehle nur bei starker Signal-Konfluenz.",
            RiskLevel.MODERATE: "Balance zwischen Chancen und Risiken.",
            RiskLevel.AGGRESSIVE: "Akzeptiere höheres Risiko für potentiell höhere Renditen.",
        }
        prompt_parts.append(risk_descriptions.get(strategy.risk_level, ""))

        # Custom prompt
        if strategy.custom_prompt:
            prompt_parts.append(strategy.custom_prompt)

        # Indicator weights
        if strategy.indicators:
            important_indicators = [i.name for i in strategy.indicators if i.weight >= 1.5 and i.enabled]
            if important_indicators:
                prompt_parts.append(f"Besonders wichtige Indikatoren: {', '.join(important_indicators)}")

        return " ".join(filter(None, prompt_parts))
