"""Macro Correlation Data Source - DXY, bonds, correlations, sector rotation."""

import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class MacroCorrelationSource(DataSourceBase):
    """
    Fetches macro market data and cross-asset correlations.

    Data includes:
    - DXY (US Dollar Index)
    - Bond Yields (10Y Treasury, etc.)
    - Gold/Silver prices and correlations
    - Cross-asset correlation matrices
    - Sector rotation analysis
    - Risk-on/Risk-off indicators
    - Global liquidity metrics
    """

    source_type = DataSourceType.MACRO_CORRELATION

    # Asset correlations reference
    TYPICAL_CORRELATIONS = {
        ("BTCUSD", "DXY"): -0.6,
        ("BTCUSD", "SPX"): 0.5,
        ("BTCUSD", "GOLD"): 0.3,
        ("XAUUSD", "DXY"): -0.8,
        ("XAUUSD", "REAL_YIELDS"): -0.7,
        ("SPX", "VIX"): -0.8,
        ("EURUSD", "DXY"): -0.95,
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 1800  # 30 minutes for macro data

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch macro and correlation data.

        Args:
            symbol: Trading symbol for context
            include_dxy: Include DXY analysis
            include_bonds: Include bond yield analysis
            include_correlations: Include correlation matrix
            include_sectors: Include sector rotation

        Returns:
            List of macro/correlation results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []

        include_dxy = kwargs.get("include_dxy", True)
        include_bonds = kwargs.get("include_bonds", True)
        include_corr = kwargs.get("include_correlations", True)
        include_sectors = kwargs.get("include_sectors", True)

        try:
            if include_dxy:
                dxy_data = await self._fetch_dxy_analysis(symbol)
                results.extend(dxy_data)

            if include_bonds:
                bond_data = await self._fetch_bond_analysis(symbol)
                results.extend(bond_data)

            if include_corr:
                corr_data = await self._fetch_correlation_analysis(symbol)
                results.extend(corr_data)

            if include_sectors:
                sector_data = await self._fetch_sector_rotation(symbol)
                results.extend(sector_data)

            # Global liquidity and risk metrics
            liquidity_data = await self._fetch_liquidity_metrics()
            results.extend(liquidity_data)

        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch macro data formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_dxy_analysis(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch and analyze DXY (US Dollar Index)."""
        results = []

        analysis = self._analyze_dxy(symbol)

        content = f"""DXY (US DOLLAR INDEX) ANALYSE
==============================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Aktueller Stand:
- DXY Preis: {analysis['dxy_price']}
- 24h Änderung: {analysis['change_24h']}
- 7d Änderung: {analysis['change_7d']}
- 30d Änderung: {analysis['change_30d']}

Technische Levels:
- Support: {analysis['support']}
- Resistance: {analysis['resistance']}
- 200-Tage MA: {analysis['ma200']}
- Trend: {analysis['trend']}

DXY Komponenten:
- EUR (57.6%): {analysis['eur_component']}
- JPY (13.6%): {analysis['jpy_component']}
- GBP (11.9%): {analysis['gbp_component']}
- CAD (9.1%): {analysis['cad_component']}
- SEK (4.2%): {analysis['sek_component']}
- CHF (3.6%): {analysis['chf_component']}

DXY Auswirkungen auf andere Assets:
{analysis['impact_analysis']}

Implikation für {symbol or 'Trading'}:
{analysis['symbol_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "dxy_analysis",
                **analysis
            }
        ))

        return results

    def _analyze_dxy(self, symbol: Optional[str]) -> dict:
        """Analyze DXY data."""
        # Symbol-specific implications
        symbol_implications = {
            "BTCUSD": """
Bitcoin und DXY sind typischerweise negativ korreliert (-0.5 bis -0.7).
- Starker DXY: Negativer Druck auf BTC (Risk-off, USD-Stärke)
- Schwacher DXY: Positiver Druck auf BTC (Risk-on, Suche nach Alternativen)

Ausnahmen: In extremen Risk-off Situationen kann beides fallen (Liquiditätskrise).
""",
            "XAUUSD": """
Gold und DXY sind stark negativ korreliert (-0.7 bis -0.9).
- Starker DXY: Negativer Druck auf Gold (Gold in USD teurer)
- Schwacher DXY: Positiver Druck auf Gold (Inflation-Hedge)

Gold-DXY Korrelation ist eine der stärksten im Markt.
""",
            "EURUSD": """
EUR/USD und DXY sind nahezu perfekt negativ korreliert (-0.95+).
EUR macht 57.6% des DXY aus.
- Starker DXY = Schwacher EUR/USD
- Schwacher DXY = Starker EUR/USD
""",
        }

        return {
            "dxy_price": "Wird aus Marktdaten geladen",
            "change_24h": "24h Änderung in %",
            "change_7d": "7d Änderung in %",
            "change_30d": "30d Änderung in %",
            "support": "Nächste Support-Levels",
            "resistance": "Nächste Resistance-Levels",
            "ma200": "200-Tage Moving Average",
            "trend": "Aktueller Trend (Aufwärts/Abwärts/Seitwärts)",
            "eur_component": "EUR Stärke/Schwäche",
            "jpy_component": "JPY Stärke/Schwäche",
            "gbp_component": "GBP Stärke/Schwäche",
            "cad_component": "CAD Stärke/Schwäche",
            "sek_component": "SEK Stärke/Schwäche",
            "chf_component": "CHF Stärke/Schwäche",
            "impact_analysis": """
DXY Stärke = Druck auf:
- Rohstoffe (Gold, Silber, Öl)
- Emerging Markets
- Kryptowährungen
- Unternehmensgewinne mit USD-Exposure

DXY Schwäche = Positiv für:
- Rohstoffe
- Risk Assets
- Exporteure in andere Währungen
""".strip(),
            "symbol_implication": symbol_implications.get(symbol, """
Generell gilt:
- DXY Stärke = Risk-off Umfeld
- DXY Schwäche = Risk-on Umfeld

Immer die spezifische Korrelation des Assets beachten.
""").strip()
        }

    async def _fetch_bond_analysis(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch and analyze bond yields."""
        results = []

        analysis = self._analyze_bonds(symbol)

        content = f"""ANLEIHEN & ZINSEN ANALYSE
==========================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

US Treasury Yields:
- 2-Jahr: {analysis['us_2y']}
- 10-Jahr: {analysis['us_10y']}
- 30-Jahr: {analysis['us_30y']}

Yield Curve:
- 2s10s Spread: {analysis['spread_2s10s']}
- 3m10y Spread: {analysis['spread_3m10y']}
- Yield Curve Status: {analysis['curve_status']}

Yield Curve Interpretation:
{analysis['curve_interpretation']}

Real Yields (10Y TIPS):
- Aktuell: {analysis['real_yield']}
- Trend: {analysis['real_yield_trend']}

Real Yield Implikation:
{analysis['real_yield_implication']}

Fed Funds Rate:
- Aktuell: {analysis['fed_funds']}
- Markterwartung (nächste Meeting): {analysis['fed_expectation']}
- Terminal Rate Prognose: {analysis['terminal_rate']}

Globale Anleihen:
- Deutschland 10Y: {analysis['germany_10y']}
- Japan 10Y: {analysis['japan_10y']}
- UK 10Y: {analysis['uk_10y']}

Implikation für {symbol or 'Trading'}:
{analysis['symbol_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "bond_analysis",
                **analysis
            }
        ))

        return results

    def _analyze_bonds(self, symbol: Optional[str]) -> dict:
        """Analyze bond market data."""
        symbol_implications = {
            "BTCUSD": """
Bitcoin reagiert auf Realzinsen:
- Steigende Realzinsen: Negativ für BTC (Opportunitätskosten)
- Fallende Realzinsen: Positiv für BTC (Suche nach Rendite)

Yield Curve Inversion kann Rezessions-Hedging in BTC treiben.
""",
            "XAUUSD": """
Gold ist stark abhängig von Realzinsen:
- Steigende Realzinsen: Sehr negativ für Gold
- Negative Realzinsen: Sehr positiv für Gold

Gold = Zero-Yield Asset, daher direkte Konkurrenz zu Bonds.
""",
        }

        return {
            "us_2y": "2-Jahr Treasury Yield",
            "us_10y": "10-Jahr Treasury Yield",
            "us_30y": "30-Jahr Treasury Yield",
            "spread_2s10s": "2s10s Spread (normal: positiv)",
            "spread_3m10y": "3m10y Spread (normal: positiv)",
            "curve_status": "Normal/Flach/Invertiert",
            "curve_interpretation": """
Yield Curve Signale:
- Normale Kurve (steigend): Gesunde Wirtschaftserwartung
- Flache Kurve: Unsicherheit, Übergang
- Invertierte Kurve: Historischer Rezessions-Indikator
  (Durchschnittlich 12-18 Monate vor Rezession)

WICHTIG: Inversion allein ist nicht timing-genau.
Oft folgt erst Rally, dann Crash.
""".strip(),
            "real_yield": "10Y TIPS (inflationsbereinigt)",
            "real_yield_trend": "Trend der letzten Wochen",
            "real_yield_implication": """
Real Yields = Nominalzins - Inflation
- Positive Real Yields: Anleihen attraktiv vs. Risk Assets
- Negative Real Yields: Anleger suchen Alternativen

Real Yields sind einer der wichtigsten Makro-Faktoren für:
- Gold
- Growth Stocks
- Kryptowährungen
""".strip(),
            "fed_funds": "Aktueller Fed Funds Rate",
            "fed_expectation": "CME FedWatch Wahrscheinlichkeiten",
            "terminal_rate": "Markterwartung für Höchststand",
            "germany_10y": "Bund Yield",
            "japan_10y": "JGB Yield",
            "uk_10y": "Gilt Yield",
            "symbol_implication": symbol_implications.get(symbol, """
Anleihen beeinflussen alle Asset-Klassen:
- Steigende Yields: Druck auf Bewertungen
- Fallende Yields: Support für Risk Assets

Die Geschwindigkeit der Änderung ist oft wichtiger als das Level.
""").strip()
        }

    async def _fetch_correlation_analysis(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch cross-asset correlation analysis."""
        results = []

        analysis = self._analyze_correlations(symbol)

        content = f"""KORRELATIONS-ANALYSE
=====================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
Zeitraum: 30-Tage Rolling Correlation

{f"Korrelationen für {symbol}:" if symbol else "Wichtige Markt-Korrelationen:"}

{analysis['correlation_matrix']}

Aktuelle vs. Historische Korrelationen:
{analysis['correlation_changes']}

Korrelations-Regime:
- Aktuelles Regime: {analysis['current_regime']}
- Regime-Stabilität: {analysis['regime_stability']}

Interpretation:
{analysis['interpretation']}

Divergenzen (Trading-Opportunities):
{analysis['divergences']}

Risk-On / Risk-Off Indikator:
{analysis['risk_indicator']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "correlation_analysis",
                **analysis
            }
        ))

        return results

    def _analyze_correlations(self, symbol: Optional[str]) -> dict:
        """Analyze cross-asset correlations."""
        return {
            "correlation_matrix": """
                BTC     ETH     SPX     DXY     GOLD
    BTC         1.00    0.85    0.45   -0.55    0.30
    ETH         0.85    1.00    0.50   -0.50    0.25
    SPX         0.45    0.50    1.00   -0.20   -0.10
    DXY        -0.55   -0.50   -0.20    1.00   -0.75
    GOLD        0.30    0.25   -0.10   -0.75    1.00

(Werte sind illustrativ - Live-Daten würden berechnet)
""".strip(),
            "correlation_changes": """
Korrelations-Verschiebungen (30d vs. 90d):
- BTC-SPX: Korrelation [steigend/fallend]
- BTC-DXY: Korrelation [steigend/fallend]
- Gold-Realzinsen: [stabil/verändert]

Signifikante Verschiebungen können Regime-Wechsel anzeigen.
""".strip(),
            "current_regime": "Risk-On / Risk-Off / Übergang",
            "regime_stability": "Stabil / Instabil / Wechselnd",
            "interpretation": """
Korrelations-Interpretation:
1. Hohe Korrelationen (>0.7): Assets bewegen sich zusammen
2. Negative Korrelationen (<-0.5): Hedging-Möglichkeiten
3. Niedrige Korrelationen (±0.3): Diversifikations-Potential

In Krisen tendieren Korrelationen zu +1 (alles fällt zusammen).
""".strip(),
            "divergences": """
Aktuelle Divergenzen zwischen korrelierten Assets:
- [Asset A] vs [Asset B]: Divergenz seit [Zeitraum]
- Historische Convergenz-Wahrscheinlichkeit: [%]
- Trading-Opportunity: [Long A / Short B] oder umgekehrt

Divergenzen sind Mean-Reversion Opportunities.
""".strip(),
            "risk_indicator": """
Risk-On/Risk-Off Composite:
- VIX Level: [Niedrig/Normal/Erhöht/Hoch]
- Credit Spreads: [Eng/Normal/Weit]
- High-Yield Demand: [Stark/Normal/Schwach]
- Safe-Haven Flows: [Niedrig/Normal/Hoch]

Gesamtindikator: [Risk-On / Neutral / Risk-Off]
""".strip(),
            "trading_implication": """
Korrelations-basierte Strategien:
1. Pairs Trading bei Divergenzen
2. Hedging mit negativ korrelierten Assets
3. Portfolio-Diversifikation optimieren
4. Regime-Wechsel als Warnsignal nutzen
"""
        }

    async def _fetch_sector_rotation(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch sector rotation analysis."""
        results = []

        analysis = self._analyze_sector_rotation()

        content = f"""SEKTOR-ROTATION ANALYSE
========================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Aktuelle Sektor-Performance (vs. SPX):

Führende Sektoren:
{analysis['leading_sectors']}

Lagging Sektoren:
{analysis['lagging_sectors']}

Rotations-Phase:
- Aktuelle Phase: {analysis['current_phase']}
- Phase-Charakteristik: {analysis['phase_characteristic']}

Wirtschaftszyklus-Position:
{analysis['cycle_position']}

Sektor-Momentum:
{analysis['sector_momentum']}

Smart Money Flow:
{analysis['smart_money']}

Implikation für Crypto:
{analysis['crypto_implication']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "sector_rotation",
                **analysis
            }
        ))

        return results

    def _analyze_sector_rotation(self) -> dict:
        """Analyze sector rotation patterns."""
        return {
            "leading_sectors": """
1. [Sektor] - +X% vs SPX (30d)
2. [Sektor] - +X% vs SPX (30d)
3. [Sektor] - +X% vs SPX (30d)
""".strip(),
            "lagging_sectors": """
1. [Sektor] - -X% vs SPX (30d)
2. [Sektor] - -X% vs SPX (30d)
3. [Sektor] - -X% vs SPX (30d)
""".strip(),
            "current_phase": "Early Cycle / Mid Cycle / Late Cycle / Recession",
            "phase_characteristic": """
Typische Sektor-Performance nach Phase:
- Early Cycle: Consumer Discretionary, Financials, Real Estate
- Mid Cycle: Technology, Industrials, Materials
- Late Cycle: Energy, Healthcare, Consumer Staples
- Recession: Utilities, Consumer Staples, Healthcare
""".strip(),
            "cycle_position": """
Indikatoren für Zyklusphase:
- PMI Trend: [Steigend/Fallend]
- Yield Curve: [Normal/Flach/Invertiert]
- Credit Conditions: [Locker/Eng]
- Labor Market: [Stark/Schwächend]

Geschätzte Position: [Phase] mit [Konfidenz]
""".strip(),
            "sector_momentum": """
Momentum-Ranking (1M/3M/6M):
- Tech: [+/-] / [+/-] / [+/-]
- Financials: [+/-] / [+/-] / [+/-]
- Energy: [+/-] / [+/-] / [+/-]
- Healthcare: [+/-] / [+/-] / [+/-]
""".strip(),
            "smart_money": """
Institutionelle Flows:
- Größte Inflows: [Sektoren]
- Größte Outflows: [Sektoren]
- 13F Filing Trends: [Beobachtungen]
""".strip(),
            "crypto_implication": """
Crypto-Sektor Korrelation:
- Tech-Führerschaft oft positiv für Crypto (Risk-On)
- Defensive Sektor-Führerschaft negativ (Risk-Off)
- Crypto verhält sich wie "leveraged Tech"

Altcoins korrelieren stärker mit Tech/Growth als BTC.
""".strip(),
            "trading_implication": """
Sektor-Rotation nutzen:
1. Führende Sektoren für Long-Bias
2. Lagging Sektoren für Shorts/Hedges
3. Rotation als Frühindikator für Zykluswende
4. Crypto-Exposure an Sektor-Regime anpassen
"""
        }

    async def _fetch_liquidity_metrics(self) -> list[DataSourceResult]:
        """Fetch global liquidity metrics."""
        results = []

        analysis = self._analyze_liquidity()

        content = f"""GLOBALE LIQUIDITÄTS-ANALYSE
============================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Zentralbank-Bilanzen:
- Fed Balance Sheet: {analysis['fed_balance']}
- ECB Balance Sheet: {analysis['ecb_balance']}
- BOJ Balance Sheet: {analysis['boj_balance']}
- Global M2: {analysis['global_m2']}

Liquiditäts-Trend:
- 30d Änderung: {analysis['liquidity_30d']}
- 90d Änderung: {analysis['liquidity_90d']}
- YTD Änderung: {analysis['liquidity_ytd']}

Liquiditäts-Interpretation:
{analysis['interpretation']}

Reverse Repo & TGA:
- Fed RRP: {analysis['fed_rrp']}
- Treasury General Account: {analysis['tga']}
- Netto-Liquiditätseffekt: {analysis['net_liquidity']}

Credit Conditions:
- Investment Grade Spreads: {analysis['ig_spreads']}
- High Yield Spreads: {analysis['hy_spreads']}
- Financial Conditions Index: {analysis['fci']}

Historischer Kontext:
{analysis['historical_context']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "liquidity_analysis",
                **analysis
            }
        ))

        return results

    def _analyze_liquidity(self) -> dict:
        """Analyze global liquidity conditions."""
        return {
            "fed_balance": "Fed Bilanz in Billionen USD",
            "ecb_balance": "EZB Bilanz in Billionen EUR",
            "boj_balance": "BOJ Bilanz in Billionen JPY",
            "global_m2": "Globale M2 Geldmenge",
            "liquidity_30d": "30-Tage Liquiditätsänderung",
            "liquidity_90d": "90-Tage Liquiditätsänderung",
            "liquidity_ytd": "Year-to-Date Liquiditätsänderung",
            "interpretation": """
Globale Liquidität und Asset-Preise:
- Steigende Liquidität: Positiv für Risk Assets (Aktien, Crypto)
- Fallende Liquidität (QT): Druck auf Bewertungen
- Liquidität ist der "Treibstoff" für Asset-Inflation

Bitcoin korreliert stark mit der globalen Liquidität.
2020-2021: Expansion → Bull Market
2022: Kontraktion → Bear Market
""".strip(),
            "fed_rrp": "Fed Reverse Repo Facility",
            "tga": "Treasury General Account Balance",
            "net_liquidity": """
Netto-Liquidität = Fed Balance - RRP - TGA
Dies ist die "echte" Liquidität im System.
Änderungen in RRP/TGA können Fed-Policy überlagern.
""".strip(),
            "ig_spreads": "Investment Grade Credit Spreads",
            "hy_spreads": "High Yield Credit Spreads",
            "fci": "Financial Conditions Index (eng/locker)",
            "historical_context": """
Liquiditäts-Regime:
- 2008-2020: QE Era, massive Expansion
- 2022-2023: QT Era, Kontraktion
- 2024+: Abhängig von Wirtschaftsdaten

Liquidität folgt typischerweise einem 18-24 Monat Zyklus.
""".strip(),
            "trading_implication": """
Liquiditäts-basiertes Trading:
1. Expanding Liquidity: Long Risk Assets
2. Contracting Liquidity: Defensiv positionieren
3. Liquiditäts-Wendepunkte sind oft Preis-Wendepunkte
4. Fed Policy + RRP/TGA zusammen betrachten
"""
        }

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when fetching fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""MAKRO & KORRELATIONEN - ÜBERSICHT
==================================
Hinweis: Live-Daten temporär nicht verfügbar.

Wichtige Makro-Metriken:

1. DXY (US Dollar Index):
   - Beeinflusst alle USD-denominierten Assets
   - Negativ korreliert mit Gold, Crypto, Rohstoffen

2. Bond Yields:
   - Realzinsen = Nominalzins - Inflation
   - Wichtig für Bewertungen aller Assets
   - Yield Curve als Rezessions-Indikator

3. Korrelationen:
   - Ändern sich über Zeit (Regime-abhängig)
   - In Krisen oft Korrelation → 1
   - Divergenzen = Trading Opportunities

4. Sektor-Rotation:
   - Zeigt Position im Wirtschaftszyklus
   - Führende Sektoren als Frühindikator

5. Globale Liquidität:
   - Zentralbank-Bilanzen beobachten
   - Liquidität = Treibstoff für Asset-Preise

Datenquellen:
- FRED (Federal Reserve)
- TradingView
- Investing.com
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
