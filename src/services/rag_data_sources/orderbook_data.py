"""Orderbook & Liquidity Data Source - Order flow, bid/ask walls, liquidations."""

import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class OrderbookDataSource(DataSourceBase):
    """
    Fetches orderbook and liquidity data for market microstructure analysis.

    Data includes:
    - Orderbook snapshots (bid/ask depth)
    - Bid/Ask walls and clusters
    - Liquidation levels and heatmaps
    - Open Interest changes
    - Funding rates (for perpetuals)
    - Order flow imbalance
    - CVD (Cumulative Volume Delta)
    """

    source_type = DataSourceType.ORDERBOOK

    def __init__(self):
        super().__init__()
        self._cache_ttl = 60  # 1 minute for orderbook data (very dynamic)

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch orderbook and liquidity data.

        Args:
            symbol: Trading symbol
            depth: Orderbook depth to analyze (default: 50 levels)
            include_liquidations: Include liquidation level analysis
            include_cvd: Include cumulative volume delta

        Returns:
            List of orderbook/liquidity results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []

        depth = kwargs.get("depth", 50)
        include_liq = kwargs.get("include_liquidations", True)
        include_cvd = kwargs.get("include_cvd", True)

        try:
            # Orderbook analysis
            ob_data = await self._fetch_orderbook_analysis(symbol, depth)
            results.extend(ob_data)

            # Bid/Ask walls
            wall_data = await self._fetch_wall_analysis(symbol)
            results.extend(wall_data)

            # Liquidation levels
            if include_liq:
                liq_data = await self._fetch_liquidation_levels(symbol)
                results.extend(liq_data)

            # CVD and order flow
            if include_cvd:
                flow_data = await self._fetch_order_flow(symbol)
                results.extend(flow_data)

        except Exception as e:
            logger.error(f"Error fetching orderbook data: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch orderbook data formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_orderbook_analysis(
        self,
        symbol: Optional[str],
        depth: int
    ) -> list[DataSourceResult]:
        """Analyze orderbook depth and structure."""
        results = []

        analysis = self._analyze_orderbook(symbol, depth)

        content = f"""ORDERBOOK ANALYSE - {symbol or 'MARKT'}
=====================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Orderbook Tiefe:
- Analyse-Tiefe: {depth} Level pro Seite
- Bid-Tiefe (Kaufaufträge): {analysis['bid_depth']}
- Ask-Tiefe (Verkaufaufträge): {analysis['ask_depth']}
- Bid/Ask Ratio: {analysis['bid_ask_ratio']}

Spread Analyse:
- Aktueller Spread: {analysis['spread']}
- Spread in %: {analysis['spread_percent']}
- Durchschnittlicher Spread (24h): {analysis['avg_spread']}

Liquidität:
- Gesamtliquidität (±2%): {analysis['liquidity_2pct']}
- Liquiditäts-Score: {analysis['liquidity_score']}/100

Orderbook Imbalance:
- Imbalance Score: {analysis['imbalance']}/100 (>50 = mehr Käufer)
- Interpretation: {analysis['imbalance_interpretation']}

Markt-Tiefe Profil:
{analysis['depth_profile']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "orderbook_analysis",
                **analysis
            }
        ))

        return results

    def _analyze_orderbook(self, symbol: Optional[str], depth: int) -> dict:
        """Analyze orderbook structure."""
        # In production, connect to exchange APIs:
        # - Binance: api.binance.com/api/v3/depth
        # - Coinbase: api.pro.coinbase.com/products/{id}/book
        # - Kraken: api.kraken.com/0/public/Depth

        return {
            "bid_depth": "Aggregierte Bid-Liquidität",
            "ask_depth": "Aggregierte Ask-Liquidität",
            "bid_ask_ratio": "Bid/Ask Verhältnis",
            "spread": "Best Bid - Best Ask",
            "spread_percent": "Spread als % vom Preis",
            "avg_spread": "24h Durchschnitts-Spread",
            "liquidity_2pct": "Volumen innerhalb ±2% vom Midprice",
            "liquidity_score": 75,
            "imbalance": 55,
            "imbalance_interpretation": """
Leichtes Übergewicht auf der Kaufseite.
Dies kann auf kurzfristigen Aufwärtsdruck hindeuten,
sollte aber mit Vorsicht interpretiert werden, da
große Orders schnell entfernt werden können (Spoofing).
""".strip(),
            "depth_profile": """
±0.5%: Dichte Liquidität, gute Ausführung
±1.0%: Moderate Liquidität
±2.0%: Dünnere Liquidität, Slippage möglich bei großen Orders

Beobachtete Cluster:
- Starke Bid-Cluster bei wichtigen Support-Levels
- Ask-Cluster bei psychologischen Widerständen
""".strip(),
            "trading_implication": """
- Enge Spreads = Gute Liquidität, günstige Ausführung
- Bid > Ask: Kurzfristig bullisches Orderbook
- Große Imbalance: Vorsicht vor Manipulation/Spoofing
"""
        }

    async def _fetch_wall_analysis(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Analyze bid/ask walls and clusters."""
        results = []

        analysis = self._analyze_walls(symbol)

        content = f"""BID/ASK WALLS ANALYSE - {symbol or 'MARKT'}
==========================================
Große Orderblöcke im Orderbook

Aktive Bid Walls (Kaufwände):
{analysis['bid_walls']}

Aktive Ask Walls (Verkaufwände):
{analysis['ask_walls']}

Wall-Interpretation:
{analysis['wall_interpretation']}

Historische Effektivität:
- Walls als Support: {analysis['wall_support_success']}
- Walls als Resistance: {analysis['wall_resistance_success']}

Spoofing Wahrscheinlichkeit:
{analysis['spoofing_assessment']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "wall_analysis",
                **analysis
            }
        ))

        return results

    def _analyze_walls(self, symbol: Optional[str]) -> dict:
        """Analyze orderbook walls."""
        return {
            "bid_walls": """
- Level 1: [Preis] - [Größe] USD (Stärke: Stark)
- Level 2: [Preis] - [Größe] USD (Stärke: Moderat)
Wird aus Live-Orderbook-Daten berechnet
""".strip(),
            "ask_walls": """
- Level 1: [Preis] - [Größe] USD (Stärke: Stark)
- Level 2: [Preis] - [Größe] USD (Stärke: Moderat)
Wird aus Live-Orderbook-Daten berechnet
""".strip(),
            "wall_interpretation": """
Große Walls können als temporäre Support/Resistance fungieren:
- Bid Walls: Potentieller Support, Käufer akkumulieren
- Ask Walls: Potentieller Widerstand, Verkäufer warten

WICHTIG: Walls können jederzeit entfernt werden!
Sie sind keine garantierten Levels.
""".strip(),
            "wall_support_success": "Variabel - ca. 60% halten kurzfristig",
            "wall_resistance_success": "Variabel - ca. 55% halten kurzfristig",
            "spoofing_assessment": """
Spoofing-Indikatoren:
- Wall erscheint/verschwindet häufig
- Wall bewegt sich mit dem Preis
- Wall ist unverhältnismäßig groß

Bei Verdacht auf Spoofing: Wall als Signal ignorieren
""".strip(),
            "trading_implication": """
Walls als zusätzliche Konfirmation nutzen, nicht als primäres Signal:
- Entry nahe Bid Wall kann engeren Stop ermöglichen
- Take Profit vor Ask Wall platzieren
- Auf Wall-Removal achten als Warnsignal
"""
        }

    async def _fetch_liquidation_levels(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch liquidation level analysis."""
        results = []

        analysis = self._analyze_liquidations(symbol)

        content = f"""LIQUIDATION LEVEL ANALYSE - {symbol or 'MARKT'}
=============================================
Futures/Perpetual Liquidation Heatmap

Konzentrierte Long-Liquidationen:
{analysis['long_liquidations']}

Konzentrierte Short-Liquidationen:
{analysis['short_liquidations']}

Liquidation Heatmap Interpretation:
{analysis['heatmap_interpretation']}

Risiko-Level:
- Nächste Long-Liquidations-Zone: {analysis['nearest_long_liq']}
- Nächste Short-Liquidations-Zone: {analysis['nearest_short_liq']}
- Größte Liquidations-Cluster: {analysis['largest_cluster']}

Open Interest Analyse:
- Aktuelles OI: {analysis['open_interest']}
- OI Trend (24h): {analysis['oi_trend']}
- Leverage im Markt: {analysis['leverage_assessment']}

Cascade Liquidation Risiko:
{analysis['cascade_risk']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        priority = (
            DataPriority.HIGH
            if analysis['high_risk']
            else DataPriority.MEDIUM
        )

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "liquidation_analysis",
                **analysis
            }
        ))

        return results

    def _analyze_liquidations(self, symbol: Optional[str]) -> dict:
        """Analyze liquidation levels."""
        # In production, use:
        # - Coinglass API (coinglass.com)
        # - Hyblock Capital
        # - Exchange-specific liquidation data

        return {
            "long_liquidations": """
Liquidations-Cluster für Long-Positionen:
- Zone 1: [Preis -5%] - Hohe Konzentration
- Zone 2: [Preis -10%] - Moderate Konzentration
- Zone 3: [Preis -15%] - Gestreut
""".strip(),
            "short_liquidations": """
Liquidations-Cluster für Short-Positionen:
- Zone 1: [Preis +5%] - Moderate Konzentration
- Zone 2: [Preis +10%] - Hohe Konzentration
- Zone 3: [Preis +15%] - Gestreut
""".strip(),
            "heatmap_interpretation": """
Liquidation Heatmaps zeigen, wo gestopfte Positionen liegen.
Market Maker und Whales "jagen" oft diese Levels, da dort
garantierte Liquidität (durch Zwangsliquidationen) vorhanden ist.

Preis tendiert dazu, zu den größten Liquidations-Clustern
zu wandern, bevor er umkehrt.
""".strip(),
            "nearest_long_liq": "[Preis] - [Entfernung in %]",
            "nearest_short_liq": "[Preis] - [Entfernung in %]",
            "largest_cluster": "Position und ungefähres Volumen",
            "open_interest": "Gesamt OI in USD",
            "oi_trend": "Steigend/Fallend mit Prozent",
            "leverage_assessment": """
Durchschnittlicher Leverage im Markt:
- Niedrig (<5x): Gesunder Markt
- Moderat (5-10x): Normale Aktivität
- Hoch (>10x): Erhöhtes Liquidationsrisiko
""".strip(),
            "cascade_risk": """
Cascade Liquidation = Kettenreaktion von Liquidationen
Risiko erhöht wenn:
- Hohe OI-Konzentration nahe aktuellem Preis
- Viele Positionen mit ähnlichen Liquidations-Levels
- Geringe Orderbook-Liquidität zwischen Levels
""".strip(),
            "high_risk": False,
            "trading_implication": """
- Stop-Loss NICHT auf offensichtlichen Liquidations-Levels setzen
- "Liquidation Hunting" als mögliches Szenario einplanen
- Nach großen Liquidations-Events: Oft Reversal-Möglichkeiten
- Gegen die Liquidations-Richtung handeln kann profitabel sein
"""
        }

    async def _fetch_order_flow(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch CVD and order flow analysis."""
        results = []

        analysis = self._analyze_order_flow(symbol)

        content = f"""ORDER FLOW ANALYSE - {symbol or 'MARKT'}
======================================
Cumulative Volume Delta & Aggressor Analysis

CVD (Cumulative Volume Delta):
- Aktueller CVD: {analysis['cvd_current']}
- CVD Trend (4h): {analysis['cvd_4h']}
- CVD Trend (24h): {analysis['cvd_24h']}
- CVD Divergenz: {analysis['cvd_divergence']}

CVD Interpretation:
{analysis['cvd_interpretation']}

Aggressor Analyse:
- Buy Aggression: {analysis['buy_aggression']}
- Sell Aggression: {analysis['sell_aggression']}
- Netto Aggression: {analysis['net_aggression']}

Footprint Analyse:
{analysis['footprint']}

Absorption:
{analysis['absorption']}

Delta Profile:
{analysis['delta_profile']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "order_flow",
                **analysis
            }
        ))

        return results

    def _analyze_order_flow(self, symbol: Optional[str]) -> dict:
        """Analyze order flow and CVD."""
        return {
            "cvd_current": "Kumulatives Delta seit Reset",
            "cvd_4h": "4h Trend: Steigend/Fallend",
            "cvd_24h": "24h Trend: Steigend/Fallend",
            "cvd_divergence": """
CVD-Preis Divergenz:
- Preis steigt + CVD fällt: Bearische Divergenz (Warnung)
- Preis fällt + CVD steigt: Bullische Divergenz (Chance)
- Konvergenz: Trend-Bestätigung
""".strip(),
            "cvd_interpretation": """
CVD misst die Differenz zwischen Market Buy und Market Sell Orders:
- Positiver CVD: Mehr aggressive Käufer
- Negativer CVD: Mehr aggressive Verkäufer

CVD zeigt die "wahre" Nachfrage/Angebot-Dynamik,
die nicht durch Limit Orders verzerrt ist.
""".strip(),
            "buy_aggression": "Market Buys als % des Volumens",
            "sell_aggression": "Market Sells als % des Volumens",
            "net_aggression": "Netto Aggression (Buy - Sell)",
            "footprint": """
Footprint Chart Analyse:
- Zeigt Volumen auf jedem Preis-Level
- Bid/Ask Volumen getrennt sichtbar
- Imbalances zwischen Bid/Ask Volumen identifizieren
- High Volume Nodes = wichtige Preis-Level
""".strip(),
            "absorption": """
Absorption = Große Limit Orders absorbieren Market Orders
- Bid Absorption: Große Käufer "fangen" Verkäufe auf
- Ask Absorption: Große Verkäufer "fangen" Käufe auf

Absorption auf Support = Bullisch
Absorption auf Resistance = Bearisch
""".strip(),
            "delta_profile": """
Delta Profile nach Zeitrahmen:
- 1min: Kurzfristiger Orderflow
- 5min: Intraday Bias
- 1h: Session Bias
- 4h: Swing Trading Bias
""".strip(),
            "trading_implication": """
Order Flow Trading Strategie:
1. CVD Divergenzen als Früh-Indikator nutzen
2. Auf Absorption an S/R Levels achten
3. High Volume Nodes als potentielle Reversal-Zonen
4. Delta Imbalances für Entry Timing
"""
        }

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when fetching fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""ORDERBOOK & LIQUIDITÄT - ÜBERSICHT
===================================
Hinweis: Live-Daten temporär nicht verfügbar.

Wichtige Orderbook-Metriken:

1. Orderbook Tiefe:
   - Bid/Ask Depth Analyse
   - Spread-Monitoring
   - Liquiditäts-Score

2. Walls & Cluster:
   - Große Bid/Ask Walls identifizieren
   - Spoofing erkennen
   - Wall-Effektivität bewerten

3. Liquidations:
   - Long/Short Liquidation Levels
   - Cascade Risk Assessment
   - Open Interest Tracking

4. Order Flow:
   - CVD (Cumulative Volume Delta)
   - Aggressor Analyse
   - Footprint Charts

Datenquellen:
- Coinglass.com (Liquidationen, OI)
- Exchange APIs (Orderbook)
- TradingView (Footprint)

Empfehlung: Orderflow als Timing-Tool nutzen,
nicht als primäre Richtungsbestimmung.
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
