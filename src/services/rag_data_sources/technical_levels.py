"""Technical Levels Data Source - S/R zones, Fibonacci, pivots, VWAP."""

from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class TechnicalLevelsSource(DataSourceBase):
    """
    Provides key technical price levels for trading.

    Data includes:
    - Support/Resistance zones (historical importance)
    - Fibonacci retracements for different timeframes
    - Pivot Points (Daily, Weekly, Monthly)
    - VWAP levels
    - Moving Average levels
    - Volume Profile key levels
    """

    source_type = DataSourceType.TECHNICAL_LEVEL

    # Fibonacci ratios
    FIB_RATIOS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]

    def __init__(self):
        super().__init__()
        self._cache_ttl = 3600  # 1 hour for technical levels

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch technical price levels.

        Args:
            symbol: Trading symbol
            include_sr: Include Support/Resistance analysis
            include_fib: Include Fibonacci levels
            include_pivots: Include pivot points
            include_vwap: Include VWAP analysis
            include_ma: Include Moving Average levels

        Returns:
            List of technical level results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []

        include_sr = kwargs.get("include_sr", True)
        include_fib = kwargs.get("include_fib", True)
        include_pivots = kwargs.get("include_pivots", True)
        include_vwap = kwargs.get("include_vwap", True)
        include_ma = kwargs.get("include_ma", True)

        try:
            if include_sr:
                sr_data = await self._fetch_support_resistance(symbol)
                results.extend(sr_data)

            if include_fib:
                fib_data = await self._fetch_fibonacci_levels(symbol)
                results.extend(fib_data)

            if include_pivots:
                pivot_data = await self._fetch_pivot_points(symbol)
                results.extend(pivot_data)

            if include_vwap:
                vwap_data = await self._fetch_vwap_levels(symbol)
                results.extend(vwap_data)

            if include_ma:
                ma_data = await self._fetch_ma_levels(symbol)
                results.extend(ma_data)

            # Volume Profile levels
            vp_data = await self._fetch_volume_profile(symbol)
            results.extend(vp_data)

        except Exception as e:
            logger.error(f"Error fetching technical levels: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch technical levels formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_support_resistance(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch Support/Resistance zone analysis."""
        results = []

        analysis = self._analyze_sr_levels(symbol)

        content = f"""SUPPORT/RESISTANCE ANALYSE - {symbol or 'MARKT'}
===============================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

MAJOR SUPPORT ZONES (stark):
{analysis['major_support']}

MAJOR RESISTANCE ZONES (stark):
{analysis['major_resistance']}

MINOR SUPPORT (schwächer):
{analysis['minor_support']}

MINOR RESISTANCE (schwächer):
{analysis['minor_resistance']}

Psychologische Level:
{analysis['psychological']}

Historische Pivot-Punkte:
{analysis['historical_pivots']}

S/R Stärke-Bewertung:
{analysis['strength_criteria']}

Aktuelle Position:
{analysis['current_position']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "support_resistance",
                **analysis
            }
        ))

        return results

    def _analyze_sr_levels(self, symbol: Optional[str]) -> dict:
        """Analyze Support/Resistance levels."""
        # In production, this would analyze actual price data
        return {
            "major_support": """
1. [Preis] - Stärke: ★★★★★ (5/5)
   - Historische Touches: X mal
   - Zeitrahmen: Mehrere Weekly Bounces
   - Confluence: 200 MA, Fib 0.618

2. [Preis] - Stärke: ★★★★☆ (4/5)
   - Historische Touches: X mal
   - Zeitrahmen: Daily Support
   - Confluence: Vorheriges ATH

3. [Preis] - Stärke: ★★★☆☆ (3/5)
   - Historische Touches: X mal
   - Zeitrahmen: Recent Support
   - Confluence: VWAP, Pivot
""".strip(),
            "major_resistance": """
1. [Preis] - Stärke: ★★★★★ (5/5)
   - Historische Touches: X mal
   - Zeitrahmen: ATH Level
   - Confluence: Psychologisches Level

2. [Preis] - Stärke: ★★★★☆ (4/5)
   - Historische Touches: X mal
   - Zeitrahmen: Weekly Resistance
   - Confluence: Fib Extension

3. [Preis] - Stärke: ★★★☆☆ (3/5)
   - Historische Touches: X mal
   - Zeitrahmen: Daily Resistance
   - Confluence: 50 MA
""".strip(),
            "minor_support": """
- [Preis] - Kurzfristig, wenige Touches
- [Preis] - Intraday Level
- [Preis] - Recent Low
""".strip(),
            "minor_resistance": """
- [Preis] - Kurzfristig, wenige Touches
- [Preis] - Intraday Level
- [Preis] - Recent High
""".strip(),
            "psychological": """
Psychologische Level (runde Zahlen):
- [z.B. 100,000 für BTC]
- [z.B. 50,000]
- [z.B. 75,000]

Diese Level wirken als magische Anziehungspunkte
und oft auch als kurzfristige S/R.
""".strip(),
            "historical_pivots": """
Wichtige historische Wendepunkte:
- ATH: [Preis] am [Datum]
- ATL (relevant): [Preis] am [Datum]
- Cycle High: [Preis]
- Cycle Low: [Preis]
- Last Major Swing High: [Preis]
- Last Major Swing Low: [Preis]
""".strip(),
            "strength_criteria": """
S/R Stärke-Kriterien:
★ = Einmal getestet
★★ = Mehrfach getestet (2-3x)
★★★ = Oft getestet (4-5x)
★★★★ = Sehr oft getestet + Confluence
★★★★★ = Extremes Level + Multi-TF + High Volume

Confluence erhöht Stärke:
- MA Zusammentreffen
- Fibonacci Level
- Volume Profile POC
- Trend Lines
- Round Numbers
""".strip(),
            "current_position": """
Aktuelle Preis-Position:
- Preis: [aktuell]
- Nächster Support: [Preis] (Abstand: X%)
- Nächste Resistance: [Preis] (Abstand: X%)
- Risk/Reward zu nächsten Levels: X:Y
""".strip(),
            "trading_implication": """
S/R Trading Strategien:
1. Breakout: Entry über Resistance, SL unter Breakout-Level
2. Bounce: Entry am Support mit engem SL darunter
3. Retest: Warten auf Retest nach Breakout
4. Failed Breakout: Reversal bei Fake-Out

Risk Management:
- SL immer hinter nächstem Level
- Position Size basierend auf S/R Abstand
"""
        }

    async def _fetch_fibonacci_levels(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch Fibonacci retracement and extension levels."""
        results = []

        analysis = self._analyze_fibonacci(symbol)

        content = f"""FIBONACCI ANALYSE - {symbol or 'MARKT'}
======================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

FIBONACCI RETRACEMENTS (vom letzten Swing):
{analysis['retracements']}

FIBONACCI EXTENSIONS:
{analysis['extensions']}

Multi-Timeframe Fibonacci:
{analysis['mtf_fib']}

Golden Pocket Zone:
{analysis['golden_pocket']}

Fibonacci Confluence Zones:
{analysis['confluence_zones']}

Fibonacci Interpretation:
{analysis['interpretation']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "fibonacci",
                **analysis
            }
        ))

        return results

    def _analyze_fibonacci(self, symbol: Optional[str]) -> dict:
        """Analyze Fibonacci levels."""
        return {
            "retracements": """
Swing High: [Preis] | Swing Low: [Preis]

Level    | Ratio  | Preis      | Status
---------|--------|------------|--------
Fib 0.0  | 0%     | [Preis]    | Low
Fib 0.236| 23.6%  | [Preis]    | Shallow
Fib 0.382| 38.2%  | [Preis]    | Moderate
Fib 0.5  | 50%    | [Preis]    | Mid-Point
Fib 0.618| 61.8%  | [Preis]    | Golden Ratio
Fib 0.786| 78.6%  | [Preis]    | Deep
Fib 1.0  | 100%   | [Preis]    | High

Aktueller Preis bei: ~Fib [X.XXX]
""".strip(),
            "extensions": """
Über dem Swing High:

Level    | Ratio   | Preis      | Typ
---------|---------|------------|--------
Fib 1.0  | 100%    | [Preis]    | Swing High
Fib 1.272| 127.2%  | [Preis]    | Ext 1
Fib 1.618| 161.8%  | [Preis]    | Golden Ext
Fib 2.0  | 200%    | [Preis]    | Measured Move
Fib 2.618| 261.8%  | [Preis]    | Extended
Fib 3.618| 361.8%  | [Preis]    | Extreme

Extensions als Take-Profit Ziele nutzen.
""".strip(),
            "mtf_fib": """
Multi-Timeframe Fibonacci Confluence:

Weekly Swing (langfristig):
- 0.382: [Preis]
- 0.618: [Preis]

Daily Swing (mittelfristig):
- 0.382: [Preis]
- 0.618: [Preis]

4H Swing (kurzfristig):
- 0.382: [Preis]
- 0.618: [Preis]

MTF Confluence bei: [Preis-Zonen]
""".strip(),
            "golden_pocket": """
Golden Pocket Zone (0.618 - 0.65):
- Untere Grenze: [Preis] (0.618)
- Obere Grenze: [Preis] (0.65)

Der Golden Pocket ist statistisch die stärkste
Retracement-Zone für Bounces in Trends.

Statistik:
- Bounce-Rate aus Golden Pocket: ~70%
- Durchschnittlicher Bounce: +15-25%
- Failure Rate: ~30% (führt zu tieferem Retracement)
""".strip(),
            "confluence_zones": """
Fibonacci Confluence mit anderen Levels:

Zone 1: [Preis-Range]
- Fib 0.618 (Daily)
- 200 MA
- Vorheriges Swing High
→ Starke Confluence Zone

Zone 2: [Preis-Range]
- Fib 0.5 (Weekly)
- Psychologisches Level
→ Moderate Confluence Zone

Zone 3: [Preis-Range]
- Fib 0.382 (4H)
- VWAP
→ Kurzfristige Confluence
""".strip(),
            "interpretation": """
Fibonacci Interpretation:
- 0.236-0.382: Schwaches Retracement (starker Trend)
- 0.382-0.618: Normales Retracement (gesunder Trend)
- 0.618-0.786: Tiefes Retracement (Trend gefährdet)
- >0.786: Sehr tiefes Retracement (Trendwende wahrscheinlich)

Fibonacci funktioniert am besten:
- In klaren Trends
- Mit Confluence anderer Level
- Auf höheren Timeframes
""".strip(),
            "trading_implication": """
Fibonacci Trading Strategien:
1. Retracement Entry: Im Golden Pocket (0.618-0.65) kaufen
2. Extension Target: 1.618 als primäres Ziel
3. Confluence Trading: Entry nur bei Multi-Level Confluence
4. Invalidation: Unter 0.786 ist der Trend gefährdet

Risk Management:
- Stop-Loss unter nächstem Fib-Level
- Take-Profit gestaffelt an Extensions
"""
        }

    async def _fetch_pivot_points(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch Pivot Point calculations."""
        results = []

        analysis = self._analyze_pivots(symbol)

        content = f"""PIVOT POINTS ANALYSE - {symbol or 'MARKT'}
=========================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

DAILY PIVOTS (Standard):
{analysis['daily_pivots']}

WEEKLY PIVOTS:
{analysis['weekly_pivots']}

MONTHLY PIVOTS:
{analysis['monthly_pivots']}

Camarilla Pivots:
{analysis['camarilla']}

Pivot Interpretation:
{analysis['interpretation']}

Aktuelle Position relativ zu Pivots:
{analysis['current_position']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "pivot_points",
                **analysis
            }
        ))

        return results

    def _analyze_pivots(self, symbol: Optional[str]) -> dict:
        """Analyze Pivot Points."""
        return {
            "daily_pivots": """
Standard Pivot Berechnung:
PP = (High + Low + Close) / 3

Level   | Preis      | Beschreibung
--------|------------|------------------
R3      | [Preis]    | Extreme Resistance
R2      | [Preis]    | Strong Resistance
R1      | [Preis]    | First Resistance
PP      | [Preis]    | Pivot Point (Neutral)
S1      | [Preis]    | First Support
S2      | [Preis]    | Strong Support
S3      | [Preis]    | Extreme Support

Berechnung basiert auf: Vorherige Tageskerze
""".strip(),
            "weekly_pivots": """
Weekly Pivot Levels:

Level   | Preis      | Wichtigkeit
--------|------------|------------------
R3      | [Preis]    | ★★★
R2      | [Preis]    | ★★★★
R1      | [Preis]    | ★★★★★
WPP     | [Preis]    | ★★★★★ (Hauptpivot)
S1      | [Preis]    | ★★★★★
S2      | [Preis]    | ★★★★
S3      | [Preis]    | ★★★

Weekly Pivots wichtiger als Daily für Swing Trading.
""".strip(),
            "monthly_pivots": """
Monthly Pivot Levels:

Level   | Preis      | Wichtigkeit
--------|------------|------------------
R2      | [Preis]    | ★★★★
R1      | [Preis]    | ★★★★★
MPP     | [Preis]    | ★★★★★ (Hauptpivot)
S1      | [Preis]    | ★★★★★
S2      | [Preis]    | ★★★★

Monthly Pivots = Major S/R für Position Trading.
""".strip(),
            "camarilla": """
Camarilla Pivot Levels (für Intraday):

Level   | Preis      | Trading-Regel
--------|------------|------------------
H4      | [Preis]    | Breakout Long Target
H3      | [Preis]    | Short Entry (Fade)
H2      | [Preis]    | Minor Resistance
H1      | [Preis]    | Minor Resistance
L1      | [Preis]    | Minor Support
L2      | [Preis]    | Minor Support
L3      | [Preis]    | Long Entry (Fade)
L4      | [Preis]    | Breakout Short Target

Camarilla-Strategie:
- Zwischen H3 und L3: Range Trading (Fade Extremes)
- Außerhalb H4/L4: Breakout Trading
""".strip(),
            "interpretation": """
Pivot Point Interpretation:
- Preis über PP: Bullish Bias
- Preis unter PP: Bearish Bias
- PP = Equilibrium (Entscheidungszone)

Typisches Verhalten:
- 70% der Zeit bewegt sich Preis zwischen S1 und R1
- Breakout über R1/unter S1: Trend-Tage
- R2/S2: Oft Reversal-Zonen
- R3/S3: Extreme, selten erreicht
""".strip(),
            "current_position": """
Aktuelle Position:
- Preis: [aktuell]
- Über/Unter Daily PP: [Position]
- Über/Unter Weekly PP: [Position]
- Über/Unter Monthly PP: [Position]

Nächste Levels:
- Resistance: [Level] ([Abstand])
- Support: [Level] ([Abstand])
""".strip(),
            "trading_implication": """
Pivot Trading Strategien:

1. PP Bounce: Entry am PP mit Ziel R1/S1
2. S1/R1 Fade: Counter-Trend Trade an S1/R1
3. Breakout: Entry über R1/unter S1, Ziel R2/S2
4. Multi-TF Confluence: Daily Pivot + Weekly Pivot = Starke Zone

Risk Management:
- Stop hinter nächstem Pivot Level
- Partial TP an jedem Pivot Level
"""
        }

    async def _fetch_vwap_levels(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch VWAP analysis."""
        results = []

        analysis = self._analyze_vwap(symbol)

        content = f"""VWAP ANALYSE - {symbol or 'MARKT'}
==================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

VWAP LEVELS:
{analysis['vwap_levels']}

Anchored VWAPs:
{analysis['anchored_vwaps']}

VWAP Standard-Abweichungen:
{analysis['vwap_bands']}

VWAP Interpretation:
{analysis['interpretation']}

Aktuelle Position:
{analysis['current_position']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "vwap",
                **analysis
            }
        ))

        return results

    def _analyze_vwap(self, symbol: Optional[str]) -> dict:
        """Analyze VWAP levels."""
        return {
            "vwap_levels": """
Volume Weighted Average Price:

Zeitraum     | VWAP Preis | Trend
-------------|------------|--------
Session      | [Preis]    | [Über/Unter]
Daily        | [Preis]    | [Über/Unter]
Weekly       | [Preis]    | [Über/Unter]
Monthly      | [Preis]    | [Über/Unter]

VWAP = Durchschnittspreis gewichtet nach Volumen
       (fairer Preis für den Zeitraum)
""".strip(),
            "anchored_vwaps": """
Anchored VWAPs (von wichtigen Punkten):

Anchor Point          | VWAP Preis | Beschreibung
----------------------|------------|------------------
ATH                   | [Preis]    | Von Allzeithoch
Cycle Low             | [Preis]    | Vom Zyklustief
Halving (BTC)         | [Preis]    | Vom Halving Event
Year Start            | [Preis]    | Vom Jahresanfang
Quarter Start         | [Preis]    | Vom Quartalsbeginn
Month Start           | [Preis]    | Vom Monatsanfang
Last Swing Low        | [Preis]    | Vom letzten Tief
Last Swing High       | [Preis]    | Vom letzten Hoch

Anchored VWAPs zeigen fairen Wert seit Event.
""".strip(),
            "vwap_bands": """
VWAP Standard-Abweichungen:

Band    | Obere      | Untere     | Statistik
--------|------------|------------|----------
VWAP    | [Preis]    | -          | Mean
+1 SD   | [Preis]    | [Preis]    | 68% der Zeit
+2 SD   | [Preis]    | [Preis]    | 95% der Zeit
+3 SD   | [Preis]    | [Preis]    | 99% der Zeit

Extreme Abweichungen (>2 SD) sind Mean-Reversion Kandidaten.
""".strip(),
            "interpretation": """
VWAP Interpretation:

1. Preis über VWAP: Käufer dominieren, bullish
2. Preis unter VWAP: Verkäufer dominieren, bearish
3. Preis am VWAP: Equilibrium, Entscheidungszone

VWAP als Support/Resistance:
- In Aufwärtstrends: VWAP = Support
- In Abwärtstrends: VWAP = Resistance
- In Ranges: VWAP = Pivot/Midpoint

Institutionelle Nutzung:
- Große Orders werden relativ zu VWAP ausgeführt
- Ziel: Besser als VWAP kaufen/verkaufen
""".strip(),
            "current_position": """
Aktuelle Position relativ zu VWAPs:
- Session VWAP: [Über/Unter] um [X%]
- Daily VWAP: [Über/Unter] um [X%]
- Weekly VWAP: [Über/Unter] um [X%]

SD-Position: [X] Standard-Abweichungen [über/unter]
→ [Interpretation: Normal/Überkauft/Überverkauft]
""".strip(),
            "trading_implication": """
VWAP Trading Strategien:

1. VWAP Bounce: Long am VWAP im Aufwärtstrend
2. VWAP Fade: Short bei +2 SD, Long bei -2 SD
3. VWAP Breakout: Preis schließt dauerhaft über/unter
4. Anchored VWAP: S/R an wichtigen Anchor-VWAPs

Risk Management:
- Stop unter VWAP (Long) / über VWAP (Short)
- Bei >2 SD: Reversal wahrscheinlich
"""
        }

    async def _fetch_ma_levels(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch Moving Average levels."""
        results = []

        analysis = self._analyze_ma_levels(symbol)

        content = f"""MOVING AVERAGE ANALYSE - {symbol or 'MARKT'}
==========================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

SIMPLE MOVING AVERAGES (SMA):
{analysis['sma_levels']}

EXPONENTIAL MOVING AVERAGES (EMA):
{analysis['ema_levels']}

MA Crossover Status:
{analysis['crossovers']}

MA Cloud/Ribbon:
{analysis['ma_ribbon']}

Death Cross / Golden Cross:
{analysis['crosses']}

Aktuelle Position:
{analysis['current_position']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "moving_averages",
                **analysis
            }
        ))

        return results

    def _analyze_ma_levels(self, symbol: Optional[str]) -> dict:
        """Analyze Moving Average levels."""
        return {
            "sma_levels": """
SMA Levels (Daily Close):

MA       | Preis      | Abstand  | Status
---------|------------|----------|--------
SMA 20   | [Preis]    | [X%]     | [Über/Unter]
SMA 50   | [Preis]    | [X%]     | [Über/Unter]
SMA 100  | [Preis]    | [X%]     | [Über/Unter]
SMA 200  | [Preis]    | [X%]     | [Über/Unter]

SMA 200 = Wichtigster langfristiger MA
          (Bull/Bear Market Grenze)
""".strip(),
            "ema_levels": """
EMA Levels (Daily Close):

MA       | Preis      | Abstand  | Status
---------|------------|----------|--------
EMA 9    | [Preis]    | [X%]     | [Über/Unter]
EMA 21   | [Preis]    | [X%]     | [Über/Unter]
EMA 50   | [Preis]    | [X%]     | [Über/Unter]
EMA 200  | [Preis]    | [X%]     | [Über/Unter]

EMAs reagieren schneller auf Preisänderungen.
""".strip(),
            "crossovers": """
MA Crossover Status:

Crossover        | Status     | Datum      | Signal
-----------------|------------|------------|--------
9 EMA / 21 EMA   | [Status]   | [Datum]    | Kurzfristig
20 SMA / 50 SMA  | [Status]   | [Datum]    | Mittelfristig
50 SMA / 200 SMA | [Status]   | [Datum]    | Langfristig

Bullish: Schneller MA über langsamem MA
Bearish: Schneller MA unter langsamem MA
""".strip(),
            "ma_ribbon": """
MA Ribbon Analyse:

Ribbon Status: [Expanding/Contracting/Twisted]

Expanding: Trend wird stärker
Contracting: Trend wird schwächer
Twisted: Trendwende möglich

MA Reihenfolge (bullisch):
Preis > EMA9 > EMA21 > SMA50 > SMA100 > SMA200

MA Reihenfolge (bearisch):
Preis < EMA9 < EMA21 < SMA50 < SMA100 < SMA200
""".strip(),
            "crosses": """
Golden Cross / Death Cross Status:

Golden Cross (50 kreuzt über 200):
- Letztes Datum: [Datum]
- Performance danach: [+X%]
- Historische Erfolgsrate: ~70%

Death Cross (50 kreuzt unter 200):
- Letztes Datum: [Datum]
- Performance danach: [-X%]
- Historische Erfolgsrate: ~65%

WICHTIG: Lagging Indikator - bestätigt Trends,
         sagt sie nicht voraus.
""".strip(),
            "current_position": """
Aktuelle Position:
- Über/Unter SMA 200: [Status] (Bull/Bear Market)
- Über/Unter SMA 50: [Status] (Medium Term)
- Über/Unter EMA 21: [Status] (Short Term)

MA Cluster Zone: [Preis-Range]
(Bereich mit mehreren MAs = Starke S/R Zone)
""".strip(),
            "trading_implication": """
MA Trading Strategien:

1. Trend Following: Long über MA200, Short unter MA200
2. MA Bounce: Entry am MA mit Stop darunter
3. MA Crossover: Entry bei Cross, Exit bei Re-Cross
4. MA Cluster: Trade S/R an MA-Konzentrations-Zonen

Risk Management:
- Nie gegen alle MAs handeln
- MA200 ist die wichtigste Linie im Sand
"""
        }

    async def _fetch_volume_profile(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch Volume Profile analysis."""
        results = []

        analysis = self._analyze_volume_profile(symbol)

        content = f"""VOLUME PROFILE ANALYSE - {symbol or 'MARKT'}
==========================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

VOLUME PROFILE KEY LEVELS:
{analysis['key_levels']}

POC (Point of Control):
{analysis['poc']}

Value Area:
{analysis['value_area']}

High/Low Volume Nodes:
{analysis['hvn_lvn']}

Volume Profile Interpretation:
{analysis['interpretation']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "volume_profile",
                **analysis
            }
        ))

        return results

    def _analyze_volume_profile(self, symbol: Optional[str]) -> dict:
        """Analyze Volume Profile levels."""
        return {
            "key_levels": """
Volume Profile Schlüssel-Level:

Level Type | Preis      | Beschreibung
-----------|------------|------------------
POC        | [Preis]    | Höchstes Volumen
VAH        | [Preis]    | Value Area High
VAL        | [Preis]    | Value Area Low
HVN 1      | [Preis]    | High Volume Node
HVN 2      | [Preis]    | High Volume Node
LVN 1      | [Preis]    | Low Volume Node
LVN 2      | [Preis]    | Low Volume Node
""".strip(),
            "poc": """
Point of Control (POC):
- Preis: [aktueller POC]
- Bedeutung: Fairster Preis für den Zeitraum
             (meistes Volumen gehandelt)

POC als S/R:
- Starker Support/Resistance
- Preis tendiert zum POC zurückzukehren
- Institutional Reference Point

Developing POC: [Preis] (sich entwickelnder POC)
""".strip(),
            "value_area": """
Value Area (70% des Volumens):
- Value Area High (VAH): [Preis]
- Value Area Low (VAL): [Preis]
- Value Area Range: [Breite in %]

Interpretation:
- Preis in Value Area: Akzeptierter Wert
- Preis über VAH: Überkauft relativ zu Volumen
- Preis unter VAL: Überverkauft relativ zu Volumen

Value Area = 70% des gesamten Volumens
(statistisch signifikanter Preisbereich)
""".strip(),
            "hvn_lvn": """
High Volume Nodes (HVN):
- HVN = Preislevel mit hohem Handelsvolumen
- Fungieren als Support/Resistance
- Preis verweilt oft an HVNs

HVN Levels:
1. [Preis] - [Volumen%]
2. [Preis] - [Volumen%]
3. [Preis] - [Volumen%]

Low Volume Nodes (LVN):
- LVN = Preislevel mit geringem Volumen
- Preis bewegt sich schnell durch LVNs
- Potentielle Breakout/Breakdown Zonen

LVN Levels:
1. [Preis-Range]
2. [Preis-Range]
""".strip(),
            "interpretation": """
Volume Profile Interpretation:

1. Balance vs. Imbalance:
   - Enge VA = Balance/Consolidation
   - Weite VA = Range Trading
   - Bewegung zu LVN = Potentieller Trend-Ausbruch

2. Profile Shapes:
   - P-Shape: Selling exhaustion (bullish)
   - b-Shape: Buying exhaustion (bearish)
   - D-Shape: Balanced market
   - B-Shape: Double distribution (two fair values)

3. Naked POCs:
   - POCs die nie re-visited wurden
   - Starke Magnet-Wirkung für Preis
""".strip(),
            "trading_implication": """
Volume Profile Trading:

1. POC Reversion: Trade zum POC wenn Preis weit entfernt
2. VA Edge Trade: Long an VAL, Short an VAH
3. LVN Breakout: Schnelle Moves durch LVNs
4. HVN Support: Erwarte Support/Resistance an HVNs

Risk Management:
- Stop hinter nächstem HVN
- Targets an POC oder gegenüberliegendem VA Edge
"""
        }

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when analysis fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""TECHNISCHE LEVELS - ÜBERSICHT
==============================

Wichtige technische Level-Typen:

1. Support/Resistance:
   - Historische Pivot-Punkte
   - Psychologische Level (runde Zahlen)
   - Vorherige Highs/Lows

2. Fibonacci:
   - Retracements: 0.382, 0.5, 0.618
   - Golden Pocket: 0.618-0.65
   - Extensions: 1.272, 1.618

3. Pivot Points:
   - Daily/Weekly/Monthly
   - Standard und Camarilla

4. VWAP:
   - Session/Daily/Weekly
   - Anchored VWAPs

5. Moving Averages:
   - SMA 50, 100, 200
   - EMA 9, 21, 50

6. Volume Profile:
   - POC, VAH, VAL
   - High/Low Volume Nodes

Empfehlung:
Confluence ist der Schlüssel - Level wo mehrere
Methoden übereinstimmen sind am stärksten.
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
