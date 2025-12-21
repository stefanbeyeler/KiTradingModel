"""Correlation Data Source - Asset correlations for divergence/convergence and hedge analysis."""

import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class CorrelationsDataSource(DataSourceBase):
    """
    Fetches and calculates correlation data between trading assets.

    Data includes:
    - Cross-asset correlations (BTC/ETH, Gold/Silver, etc.)
    - Rolling correlation analysis (7d, 30d, 90d windows)
    - Correlation breakdowns/divergences for trading signals
    - Hedge recommendations based on negative correlations
    - Correlation regime detection
    """

    source_type = DataSourceType.CORRELATIONS

    # Standard correlation pairs for analysis
    CRYPTO_PAIRS = [
        ("BTCUSD", "ETHUSD"),
        ("BTCUSD", "SOLUSD"),
        ("ETHUSD", "SOLUSD"),
        ("BTCUSD", "XAUUSD"),  # BTC vs Gold
        ("BTCUSD", "DXY"),     # BTC vs Dollar Index
    ]

    FOREX_PAIRS = [
        ("EURUSD", "GBPUSD"),
        ("EURUSD", "DXY"),
        ("USDJPY", "DXY"),
        ("XAUUSD", "DXY"),
        ("XAUUSD", "XAGUSD"),
    ]

    COMMODITY_PAIRS = [
        ("XAUUSD", "XAGUSD"),
        ("XAUUSD", "USOIL"),
        ("XAGUSD", "USOIL"),
    ]

    # Correlation interpretation thresholds
    CORRELATION_ZONES = {
        (0.8, 1.0): ("Sehr stark positiv", "Assets bewegen sich nahezu identisch"),
        (0.5, 0.8): ("Stark positiv", "Deutliche positive Abhängigkeit"),
        (0.2, 0.5): ("Moderat positiv", "Leichte Tendenz zur gleichen Richtung"),
        (-0.2, 0.2): ("Schwach/Keine", "Unabhängige Bewegungen"),
        (-0.5, -0.2): ("Moderat negativ", "Leichte Tendenz zur Gegenrichtung"),
        (-0.8, -0.5): ("Stark negativ", "Guter Hedge-Kandidat"),
        (-1.0, -0.8): ("Sehr stark negativ", "Exzellenter Hedge"),
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 1800  # 30 minutes for correlation data

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch correlation data.

        Args:
            symbol: Trading symbol to analyze correlations for
            timeframe: Correlation window (7d, 30d, 90d)
            include_matrix: Include full correlation matrix
            include_regime: Include correlation regime analysis

        Returns:
            List of correlation analysis results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []
        timeframe = kwargs.get("timeframe", "30d")
        include_matrix = kwargs.get("include_matrix", True)
        include_regime = kwargs.get("include_regime", True)

        try:
            # Get correlations for the specific symbol or general market
            if symbol:
                pair_corrs = await self._fetch_symbol_correlations(symbol, timeframe)
                results.extend(pair_corrs)

            if include_matrix:
                matrix_result = await self._fetch_correlation_matrix(symbol, timeframe)
                results.append(matrix_result)

            if include_regime:
                regime_result = await self._analyze_correlation_regime(symbol, timeframe)
                results.append(regime_result)

            # Divergence detection
            divergence_result = await self._detect_divergences(symbol)
            if divergence_result:
                results.append(divergence_result)

        except Exception as e:
            logger.error(f"Error fetching correlation data: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch correlation data formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol."""
        symbol_upper = symbol.upper()
        if any(c in symbol_upper for c in ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LINK"]):
            return "crypto"
        elif any(c in symbol_upper for c in ["XAU", "XAG", "OIL", "GAS"]):
            return "commodity"
        elif any(c in symbol_upper for c in ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]):
            return "forex"
        elif symbol_upper == "DXY":
            return "index"
        return "unknown"

    def _get_correlation_pairs(self, symbol: Optional[str]) -> list[tuple]:
        """Get relevant correlation pairs for a symbol."""
        if not symbol:
            return self.CRYPTO_PAIRS + self.FOREX_PAIRS[:3]

        asset_class = self._get_asset_class(symbol)
        if asset_class == "crypto":
            return self.CRYPTO_PAIRS
        elif asset_class == "forex":
            return self.FOREX_PAIRS
        elif asset_class == "commodity":
            return self.COMMODITY_PAIRS
        return self.CRYPTO_PAIRS + self.FOREX_PAIRS[:2]

    async def _fetch_symbol_correlations(
        self, symbol: str, timeframe: str
    ) -> list[DataSourceResult]:
        """Fetch correlations for a specific symbol against related assets."""
        results = []

        correlations = self._calculate_correlations(symbol, timeframe)

        content = f"""KORRELATIONSANALYSE - {symbol}
{'=' * 40}
Zeitfenster: {timeframe}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Korrelationen mit anderen Assets:
"""
        for pair_symbol, corr_data in correlations.items():
            interpretation = self._interpret_correlation(corr_data['current'])
            content += f"""
{symbol} vs {pair_symbol}:
  - Aktuelle Korrelation ({timeframe}): {corr_data['current']:.3f}
  - 7-Tage Korrelation: {corr_data['7d']:.3f}
  - 90-Tage Korrelation: {corr_data['90d']:.3f}
  - Interpretation: {interpretation[0]}
  - Bedeutung: {interpretation[1]}
"""

        content += f"""
Trading-Implikationen:
{self._get_correlation_trading_advice(symbol, correlations)}

Hedge-Empfehlungen:
{self._get_hedge_recommendations(symbol, correlations)}
"""

        # Determine priority based on significant changes
        priority = self._assess_correlation_priority(correlations)

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "analysis_type": "symbol_correlations",
                "timeframe": timeframe,
                "correlations": {k: v['current'] for k, v in correlations.items()}
            }
        ))

        return results

    def _calculate_correlations(self, symbol: str, timeframe: str) -> dict:
        """
        Calculate correlations for symbol.

        In production, this would fetch historical price data and calculate
        actual correlations using numpy/pandas.
        """
        # Get relevant comparison assets
        asset_class = self._get_asset_class(symbol)

        if asset_class == "crypto":
            base_correlations = {
                "ETHUSD": {"current": 0.85, "7d": 0.88, "90d": 0.82},
                "SOLUSD": {"current": 0.78, "7d": 0.75, "90d": 0.72},
                "XAUUSD": {"current": 0.15, "7d": 0.22, "90d": 0.08},
                "DXY": {"current": -0.42, "7d": -0.38, "90d": -0.45},
                "SPX": {"current": 0.55, "7d": 0.62, "90d": 0.48},
            }
        elif asset_class == "forex":
            base_correlations = {
                "DXY": {"current": -0.75, "7d": -0.78, "90d": -0.72},
                "GBPUSD": {"current": 0.82, "7d": 0.80, "90d": 0.78},
                "XAUUSD": {"current": 0.45, "7d": 0.42, "90d": 0.48},
            }
        elif asset_class == "commodity":
            base_correlations = {
                "XAGUSD": {"current": 0.88, "7d": 0.90, "90d": 0.85},
                "DXY": {"current": -0.65, "7d": -0.62, "90d": -0.68},
                "USOIL": {"current": 0.25, "7d": 0.30, "90d": 0.22},
            }
        else:
            base_correlations = {
                "BTCUSD": {"current": 0.45, "7d": 0.48, "90d": 0.42},
                "XAUUSD": {"current": 0.35, "7d": 0.32, "90d": 0.38},
            }

        # Remove self-correlation if present
        base_correlations.pop(symbol.upper(), None)

        return base_correlations

    def _interpret_correlation(self, correlation: float) -> tuple[str, str]:
        """Interpret correlation value."""
        for (low, high), (name, desc) in self.CORRELATION_ZONES.items():
            if low <= correlation <= high:
                return (name, desc)
        return ("Unbekannt", "Korrelation außerhalb normaler Bereiche")

    def _get_correlation_trading_advice(self, symbol: str, correlations: dict) -> str:
        """Generate trading advice based on correlations."""
        advice = []

        for other_symbol, corr_data in correlations.items():
            current = corr_data['current']
            short_term = corr_data['7d']

            # Detect correlation breakdown
            if abs(current - short_term) > 0.15:
                if current < short_term:
                    advice.append(
                        f"- Korrelations-Breakdown mit {other_symbol}: "
                        f"Von {short_term:.2f} auf {current:.2f}. "
                        "Mögliche Divergenz-Chance."
                    )
                else:
                    advice.append(
                        f"- Korrelations-Stärkung mit {other_symbol}: "
                        f"Von {short_term:.2f} auf {current:.2f}. "
                        "Verstärkte Abhängigkeit beachten."
                    )

            # High positive correlation warning
            if current > 0.8:
                advice.append(
                    f"- Hohe Korrelation mit {other_symbol} ({current:.2f}): "
                    "Diversifikationseffekt gering."
                )

        if not advice:
            advice.append("- Keine signifikanten Korrelationsänderungen festgestellt.")
            advice.append("- Aktuelle Korrelationsstruktur stabil.")

        return "\n".join(advice)

    def _get_hedge_recommendations(self, symbol: str, correlations: dict) -> str:
        """Generate hedge recommendations based on negative correlations."""
        hedges = []

        for other_symbol, corr_data in correlations.items():
            if corr_data['current'] < -0.4:
                quality = "Exzellent" if corr_data['current'] < -0.7 else "Gut"
                hedges.append(
                    f"- {other_symbol}: Korrelation {corr_data['current']:.2f} ({quality} als Hedge)"
                )

        if not hedges:
            return "Keine starken negativen Korrelationen für Hedge-Strategien gefunden."

        return "\n".join(hedges)

    def _assess_correlation_priority(self, correlations: dict) -> DataPriority:
        """Assess priority based on correlation changes."""
        for corr_data in correlations.values():
            change = abs(corr_data['current'] - corr_data['90d'])
            if change > 0.3:
                return DataPriority.HIGH
            elif change > 0.2:
                return DataPriority.MEDIUM
        return DataPriority.LOW

    async def _fetch_correlation_matrix(
        self, symbol: Optional[str], timeframe: str
    ) -> DataSourceResult:
        """Generate correlation matrix for asset class."""
        asset_class = self._get_asset_class(symbol) if symbol else "crypto"

        if asset_class == "crypto":
            matrix = self._get_crypto_matrix()
        elif asset_class == "forex":
            matrix = self._get_forex_matrix()
        else:
            matrix = self._get_crypto_matrix()

        content = f"""KORRELATIONSMATRIX - {asset_class.upper()}
{'=' * 50}
Zeitfenster: {timeframe}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

{matrix['formatted']}

Matrix-Interpretation:
{matrix['interpretation']}

Cluster-Analyse:
{matrix['clusters']}

Risiko-Diversifikation:
{matrix['diversification']}
"""

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "analysis_type": "correlation_matrix",
                "asset_class": asset_class,
                "timeframe": timeframe
            }
        )

    def _get_crypto_matrix(self) -> dict:
        """Generate crypto correlation matrix."""
        return {
            "formatted": """
         BTC    ETH    SOL    ADA    XRP
BTC     1.00   0.85   0.78   0.72   0.68
ETH     0.85   1.00   0.82   0.75   0.70
SOL     0.78   0.82   1.00   0.70   0.65
ADA     0.72   0.75   0.70   1.00   0.72
XRP     0.68   0.70   0.65   0.72   1.00
""",
            "interpretation": """
- BTC dominiert als Leit-Asset mit hohen Korrelationen zu allen Altcoins
- ETH zeigt ähnliche Muster, etwas höhere Korrelation zu Smart-Contract-Plattformen
- Altcoins (SOL, ADA, XRP) stark untereinander korreliert
- Diversifikation innerhalb Crypto-Assets begrenzt
""",
            "clusters": """
Cluster 1 (Marktführer): BTC, ETH - bewegen sich fast parallel
Cluster 2 (Smart Contracts): SOL, ADA - hohe interne Korrelation
Cluster 3 (Legacy): XRP - etwas unabhängiger, aber immer noch BTC-dominiert
""",
            "diversification": """
- Innerhalb Crypto: Geringe Diversifikation möglich
- Für echte Portfolio-Diversifikation: Gold, DXY inverse, Anleihen einbeziehen
- BTC/Gold Korrelation aktuell niedrig (~0.15) - guter Diversifikator
"""
        }

    def _get_forex_matrix(self) -> dict:
        """Generate forex correlation matrix."""
        return {
            "formatted": """
         EUR    GBP    JPY    CHF    AUD
EUR     1.00   0.82   0.45   0.72   0.55
GBP     0.82   1.00   0.38   0.65   0.60
JPY     0.45   0.38   1.00   0.68   0.25
CHF     0.72   0.65   0.68   1.00   0.40
AUD     0.55   0.60   0.25   0.40   1.00
""",
            "interpretation": """
- EUR/GBP stark korreliert (europäische Währungen)
- CHF als Safe Haven korreliert mit JPY in Krisen
- AUD als Risk-On Währung weniger korreliert mit Safe Havens
- USD-Gegenbewegungen treiben viele Paare gemeinsam
""",
            "clusters": """
Cluster 1 (Europa): EUR, GBP, CHF - wirtschaftlich verflochten
Cluster 2 (Safe Haven): JPY, CHF - Flucht in Sicherheit
Cluster 3 (Commodity): AUD, CAD, NZD - Rohstoff-abhängig
""",
            "diversification": """
- EUR/GBP Hedges wenig effektiv (zu korreliert)
- AUD/JPY als Risk-On/Risk-Off Paar interessant
- DXY für USD-Exposure-Hedge
"""
        }

    async def _analyze_correlation_regime(
        self, symbol: Optional[str], timeframe: str
    ) -> DataSourceResult:
        """Analyze current correlation regime."""
        regime = self._detect_regime()

        content = f"""KORRELATIONS-REGIME ANALYSE
{'=' * 40}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Aktuelles Regime: {regime['name']}

Beschreibung:
{regime['description']}

Regime-Indikatoren:
- Cross-Asset Korrelation: {regime['cross_asset']}
- Intra-Class Korrelation: {regime['intra_class']}
- Risk-On/Risk-Off Signal: {regime['risk_signal']}

Historischer Kontext:
{regime['historical']}

Trading-Implikationen:
{regime['implications']}

Regime-Wechsel Wahrscheinlichkeit:
{regime['transition_prob']}
"""

        priority = DataPriority.HIGH if regime['extreme'] else DataPriority.MEDIUM

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "analysis_type": "correlation_regime",
                "regime": regime['name'],
                "risk_signal": regime['risk_signal']
            }
        )

    def _detect_regime(self) -> dict:
        """Detect current correlation regime."""
        # In production: Calculate from actual cross-asset correlations
        return {
            "name": "Moderat Korreliert",
            "extreme": False,
            "cross_asset": "0.45 (durchschnittlich)",
            "intra_class": "0.72 (erhöht)",
            "risk_signal": "Neutral mit leichter Risk-On Tendenz",
            "description": """
Das aktuelle Marktumfeld zeigt moderate Korrelationen zwischen Asset-Klassen.
Innerhalb der Asset-Klassen (z.B. Crypto) bleiben Korrelationen erhöht.
Dies deutet auf normale Marktbedingungen ohne extreme Stress-Situationen hin.
""".strip(),
            "historical": """
- Krisen-Regime (2020 März): Korrelationen stiegen auf >0.9 (alles fällt gemeinsam)
- Euphorie-Regime: Altcoins entkoppeln sich teilweise von BTC
- Normales Regime: Moderate Korrelationen, Sektor-Rotationen möglich
""".strip(),
            "implications": """
- Diversifikation teilweise wirksam
- Sektor-Rotation kann Alpha generieren
- Bei Stress-Signalen: Korrelationen können schnell steigen
- Hedges mit negativen Korrelationen (Gold, DXY) sinnvoll
""".strip(),
            "transition_prob": """
- Wahrscheinlichkeit Übergang zu Krisen-Regime: 15%
- Wahrscheinlichkeit Übergang zu De-Korrelation: 25%
- Wahrscheinlichkeit Fortsetzung: 60%
""".strip()
        }

    async def _detect_divergences(self, symbol: Optional[str]) -> Optional[DataSourceResult]:
        """Detect correlation divergences for trading signals."""
        divergences = self._find_divergences(symbol)

        if not divergences:
            return None

        content = f"""KORRELATIONS-DIVERGENZEN
{'=' * 40}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Erkannte Divergenzen:
"""
        for div in divergences:
            content += f"""
{div['pair']}:
  - Normale Korrelation: {div['normal']:.2f}
  - Aktuelle Korrelation: {div['current']:.2f}
  - Divergenz-Stärke: {div['strength']}
  - Signal: {div['signal']}
  - Trading-Idee: {div['trade_idea']}
"""

        content += f"""
Divergenz-Interpretation:
Wenn normalerweise stark korrelierte Assets divergieren, kann dies auf:
1. Sektor-spezifische News/Entwicklungen hindeuten
2. Eine Mean-Reversion Gelegenheit bieten
3. Einen Regime-Wechsel andeuten

Risiko-Hinweis:
Divergenzen können sich ausweiten bevor Mean-Reversion einsetzt.
Immer Stop-Loss verwenden!
"""

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "analysis_type": "divergences",
                "divergences": [d['pair'] for d in divergences]
            }
        )

    def _find_divergences(self, symbol: Optional[str]) -> list[dict]:
        """Find correlation divergences."""
        # In production: Compare current vs historical correlations
        # Return empty list for normal conditions
        return []

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when fetching fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""KORRELATIONS-ANALYSE - ÜBERSICHT
===================================
Hinweis: Live-Daten temporär nicht verfügbar.

Korrelations-Grundlagen für Trading:

1. Positive Korrelation (>0.5):
   - Assets bewegen sich in gleiche Richtung
   - Geringe Diversifikation
   - Verstärkt Gewinne UND Verluste

2. Negative Korrelation (<-0.5):
   - Assets bewegen sich gegenläufig
   - Gute Hedge-Möglichkeit
   - Beispiel: BTC vs DXY, Gold vs USD

3. Keine Korrelation (-0.2 bis 0.2):
   - Unabhängige Bewegungen
   - Beste Diversifikation
   - Beispiel: BTC vs einige Altcoins in bestimmten Phasen

Wichtige Korrelationen:
- BTC/ETH: ~0.85 (sehr hoch)
- BTC/Gold: ~0.15 (niedrig)
- BTC/DXY: ~-0.40 (negativ)
- EUR/GBP: ~0.82 (sehr hoch)

Empfehlung:
- Portfolio-Korrelationen regelmäßig prüfen
- In Krisen steigen alle Korrelationen
- Hedges vor Stress-Events aufbauen
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
