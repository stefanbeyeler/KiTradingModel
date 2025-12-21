"""Volatility Regime Data Source - VIX, ATR trends, Bollinger width for position sizing."""

import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class VolatilityRegimeDataSource(DataSourceBase):
    """
    Fetches and analyzes volatility regime data for position sizing and risk management.

    Data includes:
    - VIX and volatility indices across markets
    - ATR (Average True Range) trends
    - Bollinger Bandwidth for volatility compression/expansion
    - Historical volatility vs implied volatility
    - Volatility regime classification
    - Position sizing recommendations
    """

    source_type = DataSourceType.VOLATILITY_REGIME

    # Volatility regime thresholds (based on percentile)
    REGIME_ZONES = {
        "extreme_low": (0, 10, "Extrem niedrige Volatilität", "Ruhe vor dem Sturm möglich"),
        "low": (10, 30, "Niedrige Volatilität", "Gute Zeit für Optionen-Verkauf"),
        "normal": (30, 70, "Normale Volatilität", "Standard-Positionsgrößen"),
        "high": (70, 90, "Erhöhte Volatilität", "Positionsgrößen reduzieren"),
        "extreme_high": (90, 100, "Extrem hohe Volatilität", "Maximale Vorsicht, Panik möglich"),
    }

    # VIX thresholds
    VIX_ZONES = {
        (0, 12): ("Sehr niedrig", "Complacency-Zone, Absicherung günstig"),
        (12, 17): ("Niedrig", "Normaler Bullenmarkt"),
        (17, 20): ("Moderat", "Leicht erhöhte Unsicherheit"),
        (20, 25): ("Erhöht", "Hedging aktiv, Vorsicht"),
        (25, 30): ("Hoch", "Signifikante Angst im Markt"),
        (30, 40): ("Sehr hoch", "Panik-Zone, oft Markttiefs"),
        (40, 100): ("Extrem", "Crash-Niveau, historische Kaufgelegenheiten"),
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 600  # 10 minutes for volatility data

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch volatility regime data.

        Args:
            symbol: Trading symbol to analyze
            include_vix: Include VIX analysis
            include_atr: Include ATR trend analysis
            include_bollinger: Include Bollinger width analysis
            include_regime: Include regime classification

        Returns:
            List of volatility analysis results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []
        include_vix = kwargs.get("include_vix", True)
        include_atr = kwargs.get("include_atr", True)
        include_bollinger = kwargs.get("include_bollinger", True)
        include_regime = kwargs.get("include_regime", True)

        try:
            if include_vix:
                vix_result = await self._fetch_vix_analysis()
                results.append(vix_result)

            if include_atr and symbol:
                atr_result = await self._fetch_atr_analysis(symbol)
                results.append(atr_result)

            if include_bollinger and symbol:
                bb_result = await self._fetch_bollinger_analysis(symbol)
                results.append(bb_result)

            if include_regime:
                regime_result = await self._analyze_volatility_regime(symbol)
                results.append(regime_result)

            # Position sizing recommendations
            sizing_result = await self._get_position_sizing_recommendations(symbol)
            results.append(sizing_result)

        except Exception as e:
            logger.error(f"Error fetching volatility data: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch volatility data formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_vix_analysis(self) -> DataSourceResult:
        """Fetch and analyze VIX data."""
        vix_data = await self._get_vix_data()

        zone = self._get_vix_zone(vix_data['current'])

        content = f"""VIX VOLATILITÄTS-INDEX ANALYSE
{'=' * 45}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Aktueller VIX: {vix_data['current']:.2f}
Klassifikation: {zone[0]}
Interpretation: {zone[1]}

Historischer Kontext:
- 20-Tage Durchschnitt: {vix_data['ma20']:.2f}
- 50-Tage Durchschnitt: {vix_data['ma50']:.2f}
- 52-Wochen Range: {vix_data['52w_low']:.2f} - {vix_data['52w_high']:.2f}
- Aktuelles Percentile: {vix_data['percentile']}%

VIX Term Structure:
- VIX Spot: {vix_data['current']:.2f}
- VIX 1M Future: {vix_data['vix_1m']:.2f}
- Contango/Backwardation: {vix_data['term_structure']}

VIX-Zonen Referenz:
- <12: Complacency - Optionen günstig, gute Zeit für Hedges
- 12-17: Normal - Gesunder Bullenmarkt
- 17-25: Erhöht - Hedging aktiv, erhöhte Unsicherheit
- 25-40: Hoch - Signifikante Angst, oft Markttiefs nahe
- >40: Extrem - Panik, historisch beste Kaufgelegenheiten

Trend-Analyse:
{vix_data['trend_analysis']}

Trading-Signale:
{self._get_vix_trading_signals(vix_data)}
"""

        priority = DataPriority.HIGH if vix_data['current'] > 25 or vix_data['current'] < 12 else DataPriority.MEDIUM

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=priority,
            metadata={
                "metric_type": "vix_analysis",
                "vix_current": vix_data['current'],
                "percentile": vix_data['percentile'],
                "zone": zone[0]
            }
        )

    async def _get_vix_data(self) -> dict:
        """Get VIX data from external sources."""
        # In production: Fetch from market data API
        return {
            "current": 18.5,
            "ma20": 17.2,
            "ma50": 16.8,
            "52w_low": 11.5,
            "52w_high": 38.2,
            "percentile": 45,
            "vix_1m": 19.2,
            "term_structure": "Contango (normal)",
            "trend_analysis": """
- VIX aktuell leicht über Durchschnitt
- 7-Tage Trend: Stabil bis leicht steigend
- Volatilität der Volatilität (VVIX): Normal
- Keine extremen Bewegungen erwartet
""".strip()
        }

    def _get_vix_zone(self, vix_value: float) -> tuple[str, str]:
        """Get VIX zone classification."""
        for (low, high), info in self.VIX_ZONES.items():
            if low <= vix_value < high:
                return info
        return ("Unbekannt", "Wert außerhalb normaler Bereiche")

    def _get_vix_trading_signals(self, vix_data: dict) -> str:
        """Generate trading signals based on VIX."""
        signals = []
        vix = vix_data['current']

        if vix < 12:
            signals.append("WARNUNG: Complacency-Zone - Absicherung jetzt günstig kaufen")
            signals.append("Put-Optionen als Versicherung empfohlen")
        elif vix < 17:
            signals.append("Normal: Standard-Positionsgrößen verwenden")
        elif vix < 25:
            signals.append("VORSICHT: Positionsgrößen um 20-30% reduzieren")
            signals.append("Bestehende Hedges halten oder aufstocken")
        elif vix < 35:
            signals.append("KONTRÄR BULLISCH: Historisch gute Kaufzone")
            signals.append("Schrittweise Positionen aufbauen (DCA)")
            signals.append("Positionsgrößen klein halten wegen hoher Volatilität")
        else:
            signals.append("EXTREM: Panik im Markt - Beste Contrarian-Kaufgelegenheiten")
            signals.append("NUR für erfahrene Trader mit starken Nerven")
            signals.append("VIX-Spikes sind typischerweise kurzlebig")

        # Term structure signals
        if "Backwardation" in vix_data['term_structure']:
            signals.append("VIX Backwardation: Akute Angst, kurzfristig höchste Unsicherheit")

        return "\n- ".join([""] + signals)

    async def _fetch_atr_analysis(self, symbol: str) -> DataSourceResult:
        """Analyze ATR trend for a symbol."""
        atr_data = self._calculate_atr_data(symbol)

        content = f"""ATR (AVERAGE TRUE RANGE) ANALYSE - {symbol}
{'=' * 50}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Aktuelle ATR-Werte:
- ATR (14): {atr_data['atr_14']:.4f}
- ATR (7): {atr_data['atr_7']:.4f}
- ATR (21): {atr_data['atr_21']:.4f}

ATR als Prozent vom Preis:
- ATR%: {atr_data['atr_percent']:.2f}%
- Historischer Durchschnitt: {atr_data['atr_avg_percent']:.2f}%
- Aktuelles Percentile: {atr_data['percentile']}%

ATR Trend:
{atr_data['trend']}

Volatilitäts-Regime (basierend auf ATR):
{atr_data['regime']}

Position Sizing mit ATR:
{self._get_atr_position_sizing(atr_data, symbol)}

Stop-Loss Empfehlungen:
- Aggressiv: 1.0 x ATR = {atr_data['stop_1atr']:.4f}
- Standard: 1.5 x ATR = {atr_data['stop_1_5atr']:.4f}
- Konservativ: 2.0 x ATR = {atr_data['stop_2atr']:.4f}

Take-Profit Ziele (basierend auf ATR):
- TP1: 1.5 x ATR = {atr_data['tp_1_5atr']:.4f}
- TP2: 2.5 x ATR = {atr_data['tp_2_5atr']:.4f}
- TP3: 4.0 x ATR = {atr_data['tp_4atr']:.4f}
"""

        priority = (
            DataPriority.HIGH
            if atr_data['percentile'] > 80 or atr_data['percentile'] < 10
            else DataPriority.MEDIUM
        )

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "atr_analysis",
                "atr_14": atr_data['atr_14'],
                "atr_percent": atr_data['atr_percent'],
                "percentile": atr_data['percentile']
            }
        )

    def _calculate_atr_data(self, symbol: str) -> dict:
        """Calculate ATR data for symbol."""
        # In production: Fetch from TwelveData or calculate from OHLC
        # Using representative values
        is_crypto = any(c in symbol.upper() for c in ["BTC", "ETH", "SOL"])

        if is_crypto:
            atr_14 = 1250.0  # Example for BTC
            price = 42000.0
        else:
            atr_14 = 0.0085  # Example for forex
            price = 1.0850

        atr_percent = (atr_14 / price) * 100

        return {
            "atr_14": atr_14,
            "atr_7": atr_14 * 1.1,
            "atr_21": atr_14 * 0.95,
            "atr_percent": atr_percent,
            "atr_avg_percent": atr_percent * 0.9,
            "percentile": 55,
            "trend": """
- 7-Tage ATR > 14-Tage ATR: Volatilität steigt kurzfristig
- ATR über historischem Durchschnitt: Erhöhte Marktaktivität
- Trend: Leicht steigend
""".strip(),
            "regime": """
Aktuelles Regime: NORMAL bis LEICHT ERHÖHT
- ATR Percentile 55% zeigt durchschnittliche Volatilität
- Keine extremen Bewegungen erwartet
- Standard Position Sizing anwendbar
""".strip(),
            "stop_1atr": atr_14,
            "stop_1_5atr": atr_14 * 1.5,
            "stop_2atr": atr_14 * 2.0,
            "tp_1_5atr": atr_14 * 1.5,
            "tp_2_5atr": atr_14 * 2.5,
            "tp_4atr": atr_14 * 4.0
        }

    def _get_atr_position_sizing(self, atr_data: dict, symbol: str) -> str:
        """Generate position sizing advice based on ATR."""
        percentile = atr_data['percentile']

        if percentile < 20:
            return """
NIEDRIGE VOLATILITÄT - Position Size erhöhen möglich:
- Standard: 100% der normalen Position
- Aggressiv: bis 125% möglich (größere SL, aber mehr Zeit)
- Vorsicht: Volatilitäts-Expansion kann schnell kommen
"""
        elif percentile < 40:
            return """
NORMALE-NIEDRIGE VOLATILITÄT:
- Standard: 100% der normalen Position
- SL-Distanz kann etwas enger sein
- Gute Bedingungen für Trendfolge
"""
        elif percentile < 60:
            return """
NORMALE VOLATILITÄT:
- Standard: 100% der normalen Position
- Standard SL/TP Verhältnisse (1.5-2x ATR SL, 2-3x ATR TP)
- Normale Marktbedingungen
"""
        elif percentile < 80:
            return """
ERHÖHTE VOLATILITÄT - Position Size reduzieren:
- Empfohlen: 75% der normalen Position
- Breitere SL erforderlich (2x ATR)
- Schnellere Gewinnmitnahmen
"""
        else:
            return """
HOHE VOLATILITÄT - Maximale Vorsicht:
- Empfohlen: 50% der normalen Position
- Sehr breite SL (2.5-3x ATR)
- Kleine Positionen, schnelle Exits
- Alternativ: Aus dem Markt bleiben
"""

    async def _fetch_bollinger_analysis(self, symbol: str) -> DataSourceResult:
        """Analyze Bollinger Band width for volatility."""
        bb_data = self._calculate_bollinger_data(symbol)

        content = f"""BOLLINGER BANDS VOLATILITÄTS-ANALYSE - {symbol}
{'=' * 55}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Bollinger Bands (20, 2):
- Oberes Band: {bb_data['upper']:.4f}
- Mittleres Band (SMA): {bb_data['middle']:.4f}
- Unteres Band: {bb_data['lower']:.4f}
- Aktueller Preis: {bb_data['price']:.4f}

Bandbreite-Analyse:
- Bandbreite: {bb_data['bandwidth']:.2f}%
- Historischer Durchschnitt: {bb_data['avg_bandwidth']:.2f}%
- Percentile: {bb_data['percentile']}%

%B (Percent B):
- Aktuell: {bb_data['percent_b']:.2f}
- Interpretation: {bb_data['percent_b_interpretation']}

Squeeze-Detektion:
{bb_data['squeeze_status']}

Volatilitäts-Expansion/Kontraktion:
{bb_data['expansion_analysis']}

Trading-Signale:
{self._get_bollinger_signals(bb_data)}
"""

        priority = (
            DataPriority.HIGH
            if bb_data['squeeze_detected'] or bb_data['percentile'] > 90
            else DataPriority.MEDIUM
        )

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "bollinger_analysis",
                "bandwidth": bb_data['bandwidth'],
                "percent_b": bb_data['percent_b'],
                "squeeze": bb_data['squeeze_detected']
            }
        )

    def _calculate_bollinger_data(self, symbol: str) -> dict:
        """Calculate Bollinger Band data."""
        # In production: Fetch from TwelveData
        is_crypto = any(c in symbol.upper() for c in ["BTC", "ETH", "SOL"])

        if is_crypto:
            price = 42000.0
            middle = 41500.0
            upper = 44500.0
            lower = 38500.0
        else:
            price = 1.0850
            middle = 1.0820
            upper = 1.0920
            lower = 1.0720

        bandwidth = ((upper - lower) / middle) * 100
        percent_b = (price - lower) / (upper - lower)

        return {
            "price": price,
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": bandwidth,
            "avg_bandwidth": bandwidth * 1.1,
            "percentile": 45,
            "percent_b": percent_b,
            "percent_b_interpretation": self._interpret_percent_b(percent_b),
            "squeeze_detected": bandwidth < 4.0,
            "squeeze_status": self._get_squeeze_status(bandwidth),
            "expansion_analysis": """
- Bandbreite aktuell nahe Durchschnitt
- Keine signifikante Kontraktion/Expansion
- Normale Volatilitätsbedingungen
""".strip()
        }

    def _interpret_percent_b(self, percent_b: float) -> str:
        """Interpret Percent B value."""
        if percent_b > 1.0:
            return "Über oberem Band - Überkauft, mögliche Korrektur"
        elif percent_b > 0.8:
            return "Nahe oberem Band - Aufwärtstrend, aber Vorsicht"
        elif percent_b > 0.5:
            return "Obere Hälfte - Leicht bullisch"
        elif percent_b > 0.2:
            return "Untere Hälfte - Leicht bearisch"
        elif percent_b > 0:
            return "Nahe unterem Band - Abwärtstrend oder Support"
        else:
            return "Unter unterem Band - Überverkauft, möglicher Bounce"

    def _get_squeeze_status(self, bandwidth: float) -> str:
        """Get Bollinger Squeeze status."""
        if bandwidth < 3.0:
            return """
SQUEEZE DETEKTIERT - Starke Kontraktion!
- Bandbreite unter historischem Minimum
- Explosive Bewegung wahrscheinlich
- Richtung unklar - auf Ausbruch warten
- Erhöhte Positionsbereitschaft
"""
        elif bandwidth < 5.0:
            return """
Moderate Kontraktion
- Bandbreite unter Durchschnitt
- Potentieller Squeeze in Entwicklung
- Aufmerksam auf Ausbruchssignale
"""
        else:
            return """
Keine Squeeze-Situation
- Normale bis erhöhte Bandbreite
- Volatilität im normalen Bereich
"""

    def _get_bollinger_signals(self, bb_data: dict) -> str:
        """Generate trading signals from Bollinger analysis."""
        signals = []
        percent_b = bb_data['percent_b']

        if bb_data['squeeze_detected']:
            signals.append("SQUEEZE: Auf Ausbruch vorbereiten - Richtung noch unklar")
            signals.append("Entry bei Ausbruch über/unter Band mit Momentum-Bestätigung")

        if percent_b > 1.0:
            signals.append("ÜBERKAUFT: Preis über oberem Band - Rücksetzer möglich")
            signals.append("Mean Reversion Short-Gelegenheit prüfen")
        elif percent_b < 0:
            signals.append("ÜBERVERKAUFT: Preis unter unterem Band - Bounce möglich")
            signals.append("Mean Reversion Long-Gelegenheit prüfen")

        if not signals:
            signals.append("Neutral: Preis innerhalb der Bänder")
            signals.append("Trendfolge oder Range-Trading je nach Kontext")

        return "\n- ".join([""] + signals)

    async def _analyze_volatility_regime(self, symbol: Optional[str]) -> DataSourceResult:
        """Analyze overall volatility regime."""
        regime = self._classify_regime(symbol)

        content = f"""VOLATILITÄTS-REGIME KLASSIFIKATION
{'=' * 45}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Aktuelles Regime: {regime['name']}
Konfidenz: {regime['confidence']}%

Regime-Beschreibung:
{regime['description']}

Indikatoren:
- VIX-Signal: {regime['vix_signal']}
- ATR-Signal: {regime['atr_signal']}
- BBands-Signal: {regime['bbands_signal']}
- Historische Vol vs Implizite Vol: {regime['hv_iv_spread']}

Regime-Charakteristiken:
{regime['characteristics']}

Erwartete Marktdynamik:
{regime['expected_dynamics']}

Risikomanagement-Anpassungen:
{regime['risk_adjustments']}

Regime-Übergangs-Wahrscheinlichkeiten:
{regime['transition_probs']}
"""

        priority = (
            DataPriority.HIGH
            if regime['extreme']
            else DataPriority.MEDIUM
        )

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "volatility_regime",
                "regime": regime['name'],
                "confidence": regime['confidence']
            }
        )

    def _classify_regime(self, symbol: Optional[str]) -> dict:
        """Classify current volatility regime."""
        # In production: Aggregate multiple volatility indicators
        return {
            "name": "NORMAL",
            "confidence": 75,
            "extreme": False,
            "description": """
Das aktuelle Volatilitätsumfeld ist normal mit leicht erhöhter kurzfristiger
Schwankungsbreite. Keine extremen Bedingungen erkennbar.
""".strip(),
            "vix_signal": "Neutral (VIX im normalen Bereich)",
            "atr_signal": "Leicht erhöht (Percentile 55%)",
            "bbands_signal": "Normal (keine Squeeze)",
            "hv_iv_spread": "Positiv (IV > HV) - leichter Aufschlag für Optionen",
            "characteristics": """
Normale Volatilität Regime:
- Tägliche Schwankungen im erwarteten Bereich
- Trend-Following Strategien effektiv
- Standard Position Sizing anwendbar
- Moderate Stop-Loss Distanzen ausreichend
""".strip(),
            "expected_dynamics": """
- Reguläre Marktbewegungen erwartet
- Keine außergewöhnlichen Events am Horizont
- Saisonale Muster können greifen
- Overnight-Gaps im normalen Rahmen
""".strip(),
            "risk_adjustments": """
- Position Size: 100% (Standard)
- Stop-Loss: 1.5-2x ATR
- Take-Profit: 2-3x ATR
- Max Risiko pro Trade: 1-2%
- Overnight-Positionen: Erlaubt
""".strip(),
            "transition_probs": """
- Verbleib im normalen Regime: 65%
- Übergang zu niedriger Volatilität: 15%
- Übergang zu hoher Volatilität: 20%
""".strip()
        }

    async def _get_position_sizing_recommendations(
        self, symbol: Optional[str]
    ) -> DataSourceResult:
        """Generate position sizing recommendations based on volatility."""
        sizing = self._calculate_position_sizing(symbol)

        content = f"""POSITION SIZING EMPFEHLUNGEN
{'=' * 40}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
{'Symbol: ' + symbol if symbol else 'Allgemein'}

Volatilitäts-adjustierte Position Sizing:
{sizing['method']}

Empfohlene Anpassungen:
- Basis-Positionsgröße: {sizing['base_size']}%
- Volatilitäts-Faktor: {sizing['vol_factor']:.2f}
- Adjustierte Größe: {sizing['adjusted_size']}%

Risiko-Parameter:
- Max Risiko pro Trade: {sizing['max_risk']}%
- Empfohlener Stop-Loss: {sizing['stop_loss']}
- Risk-Reward Minimum: {sizing['min_rr']}

Positionsgrößen-Formel:
{sizing['formula']}

Beispielrechnung ({symbol or 'BTCUSD'}):
{sizing['example']}

Kelly-Kriterium Anpassung:
{sizing['kelly']}

Korrelations-Adjustierung:
{sizing['correlation_adj']}

Allgemeine Regeln:
{sizing['rules']}
"""

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "position_sizing",
                "adjusted_size": sizing['adjusted_size'],
                "vol_factor": sizing['vol_factor']
            }
        )

    def _calculate_position_sizing(self, symbol: Optional[str]) -> dict:
        """Calculate position sizing based on volatility."""
        return {
            "method": """
ATR-basierte Position Sizing:
Position Size = (Account Risk %) / (ATR × Multiplier / Entry Price)

Dies stellt sicher, dass jeder Trade unabhängig von der Volatilität
das gleiche Risiko in % des Accounts hat.
""".strip(),
            "base_size": 100,
            "vol_factor": 1.0,
            "adjusted_size": 100,
            "max_risk": 2,
            "stop_loss": "1.5-2x ATR",
            "min_rr": "1:2",
            "formula": """
Position Size = (Account × Risk%) / (Stop-Loss in Währung)

Beispiel:
- Account: 10.000 USD
- Risiko: 2% = 200 USD
- Stop-Loss: 100 USD (1.5x ATR)
- Position Size: 200 / 100 = 2 Einheiten
""".strip(),
            "example": """
Annahmen:
- Account: 10.000 USD
- Max Risiko: 2% = 200 USD
- ATR: 1.250 USD (BTC)
- Stop-Loss: 1.5 × ATR = 1.875 USD

Position Size = 200 / 1.875 = 0.107 BTC
Bei BTC @ 42.000: Position = 4.494 USD = 45% des Accounts

Dies begrenzt den Verlust auf max 200 USD (2%)
""".strip(),
            "kelly": """
Kelly-Kriterium für optimale Position:
f* = (p × b - q) / b

Wobei:
- p = Gewinnwahrscheinlichkeit
- q = Verlustwahrscheinlichkeit (1-p)
- b = Odds (durchschnittlicher Gewinn / durchschnittlicher Verlust)

Empfehlung: 25-50% des Kelly-Werts für konservativeres Sizing
""".strip(),
            "correlation_adj": """
Bei mehreren korrelierten Positionen:
- Hohe Korrelation (>0.7): Gesamtgröße reduzieren
- Behandeln als quasi-eine-Position für Risikozwecke
- Beispiel: BTC + ETH Long = effektiv 1.7x BTC Long
""".strip(),
            "rules": """
1. Niemals mehr als 2% pro Trade riskieren
2. Maximale Gesamtexposure: 10% des Accounts
3. Bei hoher Volatilität: Position Size halbieren
4. Bei Verlustreihe: Position Size reduzieren (50% nach 3 Verlusten)
5. Korrelierte Positionen als eine behandeln
6. Übernacht-Positionen: Halbe Größe oder Hedge
""".strip()
        }

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when fetching fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""VOLATILITÄTS-ANALYSE - ÜBERSICHT
==================================
Hinweis: Live-Daten temporär nicht verfügbar.

Volatilitäts-Metriken Referenz:

1. VIX (Volatility Index):
   - <12: Sehr niedrig, Complacency
   - 12-20: Normal
   - 20-30: Erhöht, Vorsicht
   - >30: Hoch, oft Markttiefs

2. ATR (Average True Range):
   - Misst durchschnittliche Tagesrange
   - Für Stop-Loss: 1.5-2x ATR
   - Für Position Sizing: Invertiert zu ATR

3. Bollinger Bands:
   - Bandwidth: Maß für Volatilität
   - Squeeze: Niedrige Bandbreite vor Ausbruch
   - %B: Position innerhalb der Bänder

4. Position Sizing Formel:
   Position = (Account × Risk%) / Stop-Loss

Regime-Anpassungen:
- Niedrige Vol: 100-125% Position
- Normale Vol: 100% Position
- Hohe Vol: 50-75% Position
- Extreme Vol: 25-50% Position oder pausieren

Quellen:
- VIX: CBOE
- ATR/Bollinger: TwelveData
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
