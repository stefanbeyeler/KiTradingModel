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
                lines.append(f"| {ind.name} | {'✓' if ind.enabled else '✗'} | {ind.weight} | {buy_th} | {sell_th} |")
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
        lines.append(f"- **Max. Positionsgröße:** {strategy.max_position_size_percent}%")
        lines.append("")

        # Timeframe
        lines.append("## Zeitrahmen")
        lines.append("")
        lines.append(f"- **Bevorzugter Zeitrahmen:** {strategy.preferred_timeframe}")
        lines.append(f"- **Lookback-Tage:** {strategy.lookback_days}")
        lines.append("")

        # RAG Settings
        lines.append("## RAG-Einstellungen")
        lines.append("")
        lines.append(f"- **RAG-Kontext verwenden:** {'Ja' if strategy.use_rag_context else 'Nein'}")
        lines.append(f"- **Max. RAG-Dokumente:** {strategy.max_rag_documents}")
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

    def import_strategy_from_markdown(self, markdown_content: str) -> Optional[TradingStrategy]:
        """Import a strategy from Markdown format."""
        import re

        try:
            # Extract name from header
            name_match = re.search(r'^# Trading Strategie: (.+)$', markdown_content, re.MULTILINE)
            if not name_match:
                logger.error("Could not find strategy name in markdown")
                return None
            name = name_match.group(1).strip()

            # Extract ID (optional for import)
            id_match = re.search(r'\*\*ID:\*\* `([^`]+)`', markdown_content)
            original_id = id_match.group(1) if id_match else None

            # Extract strategy type
            type_match = re.search(r'\*\*Typ:\*\* (\w+)', markdown_content)
            strategy_type = StrategyType.CUSTOM
            if type_match:
                try:
                    strategy_type = StrategyType(type_match.group(1))
                except ValueError:
                    pass

            # Extract risk level
            risk_match = re.search(r'\*\*Risikolevel:\*\* (\w+)', markdown_content)
            risk_level = RiskLevel.MODERATE
            if risk_match:
                try:
                    risk_level = RiskLevel(risk_match.group(1))
                except ValueError:
                    pass

            # Extract is_default
            default_match = re.search(r'\*\*Standard:\*\* (Ja|Nein)', markdown_content)
            is_default = default_match and default_match.group(1) == "Ja"

            # Extract is_active
            active_match = re.search(r'\*\*Aktiv:\*\* (Ja|Nein)', markdown_content)
            is_active = not active_match or active_match.group(1) == "Ja"

            # Extract description
            desc_match = re.search(r'## Beschreibung\s*\n\s*\n(.+?)(?=\n\n## |\Z)', markdown_content, re.DOTALL)
            description = ""
            if desc_match:
                desc_text = desc_match.group(1).strip()
                if desc_text != "*Keine Beschreibung*":
                    description = desc_text

            # Extract indicators from table
            indicators = []
            indicator_table = re.search(r'\| Indikator \|.*?\n\|[-|]+\n(.*?)(?=\n\n|\n\*|\Z)', markdown_content, re.DOTALL)
            if indicator_table:
                rows = indicator_table.group(1).strip().split('\n')
                for row in rows:
                    cols = [c.strip() for c in row.split('|')[1:-1]]
                    if len(cols) >= 5:
                        ind_name = cols[0]
                        enabled = '✓' in cols[1]
                        weight = float(cols[2]) if cols[2] else 1.0
                        buy_th = float(cols[3]) if cols[3] != '-' else None
                        sell_th = float(cols[4]) if cols[4] != '-' else None
                        indicators.append(IndicatorConfig(
                            name=ind_name,
                            enabled=enabled,
                            weight=weight,
                            buy_threshold=buy_th,
                            sell_threshold=sell_th
                        ))

            # Extract signal settings
            min_buy = 3
            min_sell = 3
            buy_match = re.search(r'\*\*Min\. Kaufsignale:\*\* (\d+)', markdown_content)
            if buy_match:
                min_buy = int(buy_match.group(1))
            sell_match = re.search(r'\*\*Min\. Verkaufssignale:\*\* (\d+)', markdown_content)
            if sell_match:
                min_sell = int(sell_match.group(1))

            # Extract risk management
            sl_match = re.search(r'\*\*Stop-Loss ATR-Multiplikator:\*\* ([\d.]+)', markdown_content)
            stop_loss = float(sl_match.group(1)) if sl_match else 2.0

            tp_match = re.search(r'\*\*Take-Profit ATR-Multiplikator:\*\* ([\d.]+)', markdown_content)
            take_profit = float(tp_match.group(1)) if tp_match else 3.0

            pos_match = re.search(r'\*\*Max\. Positionsgröße:\*\* ([\d.]+)%', markdown_content)
            max_position = float(pos_match.group(1)) if pos_match else 5.0

            # Extract timeframe settings
            tf_match = re.search(r'\*\*Bevorzugter Zeitrahmen:\*\* (\w+)', markdown_content)
            timeframe = tf_match.group(1) if tf_match else "short_term"

            lb_match = re.search(r'\*\*Lookback-Tage:\*\* (\d+)', markdown_content)
            lookback = int(lb_match.group(1)) if lb_match else 30

            # Extract RAG settings
            rag_match = re.search(r'\*\*RAG-Kontext verwenden:\*\* (Ja|Nein)', markdown_content)
            use_rag = not rag_match or rag_match.group(1) == "Ja"

            rag_docs_match = re.search(r'\*\*Max\. RAG-Dokumente:\*\* (\d+)', markdown_content)
            max_rag_docs = int(rag_docs_match.group(1)) if rag_docs_match else 5

            # Extract custom prompt
            custom_prompt = None
            prompt_match = re.search(r'## Benutzerdefinierter Prompt\s*\n\s*\n```\n(.+?)\n```', markdown_content, re.DOTALL)
            if prompt_match:
                custom_prompt = prompt_match.group(1).strip()

            # Generate new ID for imported strategy
            new_id = f"imported_{uuid.uuid4().hex[:8]}"

            strategy = TradingStrategy(
                id=new_id,
                name=name,
                description=description,
                strategy_type=strategy_type,
                risk_level=risk_level,
                indicators=indicators,
                min_buy_signals=min_buy,
                min_sell_signals=min_sell,
                stop_loss_atr_multiplier=stop_loss,
                take_profit_atr_multiplier=take_profit,
                max_position_size_percent=max_position,
                preferred_timeframe=timeframe,
                lookback_days=lookback,
                custom_prompt=custom_prompt,
                use_rag_context=use_rag,
                max_rag_documents=max_rag_docs,
                is_active=is_active,
                is_default=False,  # Never import as default
            )

            logger.info(f"Imported strategy from markdown: {strategy.name}")
            return strategy

        except Exception as e:
            logger.error(f"Failed to import strategy from markdown: {e}")
            return None

    async def import_and_save_strategy(self, markdown_content: str) -> Optional[TradingStrategy]:
        """Import a strategy from Markdown and save it."""
        strategy = self.import_strategy_from_markdown(markdown_content)
        if strategy:
            self._strategies[strategy.id] = strategy
            self._save_strategies()
            return strategy
        return None
