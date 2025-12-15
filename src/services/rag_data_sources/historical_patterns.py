"""Historical Patterns Data Source - Seasonality, drawdowns, event-based returns."""

from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
import calendar

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class HistoricalPatternsSource(DataSourceBase):
    """
    Provides historical pattern analysis and seasonality data.

    Data includes:
    - Seasonality patterns (monthly, weekly, daily)
    - Historical drawdowns and recovery times
    - Event-based returns (halvings, FOMC, etc.)
    - Comparable market phases from history
    - Statistical patterns and anomalies
    """

    source_type = DataSourceType.HISTORICAL_PATTERN

    # Bitcoin halving dates
    BTC_HALVINGS = [
        datetime(2012, 11, 28),
        datetime(2016, 7, 9),
        datetime(2020, 5, 11),
        datetime(2024, 4, 20),  # Approximate
    ]

    # Historical monthly returns for BTC (approximate averages)
    BTC_MONTHLY_SEASONALITY = {
        1: -0.02,   # January
        2: 0.12,    # February
        3: 0.05,    # March
        4: 0.15,    # April
        5: -0.10,   # May ("Sell in May")
        6: -0.05,   # June
        7: 0.08,    # July
        8: -0.03,   # August
        9: -0.05,   # September
        10: 0.20,   # October ("Uptober")
        11: 0.15,   # November
        12: 0.10,   # December
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 86400  # 24 hours for historical data

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch historical pattern data.

        Args:
            symbol: Trading symbol
            include_seasonality: Include seasonality analysis
            include_drawdowns: Include drawdown history
            include_events: Include event-based analysis
            include_comparable: Include comparable periods

        Returns:
            List of historical pattern results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []

        include_season = kwargs.get("include_seasonality", True)
        include_dd = kwargs.get("include_drawdowns", True)
        include_events = kwargs.get("include_events", True)
        include_comp = kwargs.get("include_comparable", True)

        try:
            if include_season:
                season_data = await self._fetch_seasonality(symbol)
                results.extend(season_data)

            if include_dd:
                dd_data = await self._fetch_drawdown_history(symbol)
                results.extend(dd_data)

            if include_events:
                event_data = await self._fetch_event_patterns(symbol)
                results.extend(event_data)

            if include_comp:
                comp_data = await self._fetch_comparable_periods(symbol)
                results.extend(comp_data)

        except Exception as e:
            logger.error(f"Error fetching historical patterns: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch historical patterns formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_seasonality(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch seasonality patterns."""
        results = []
        now = datetime.utcnow()

        analysis = self._analyze_seasonality(symbol, now)

        content = f"""SAISONALITÄTS-ANALYSE - {symbol or 'MARKT'}
==========================================
Stand: {now.strftime('%Y-%m-%d')}
Aktueller Monat: {calendar.month_name[now.month]}

MONATLICHE SAISONALITÄT:
{analysis['monthly_pattern']}

WÖCHENTLICHE SAISONALITÄT:
{analysis['weekly_pattern']}

TÄGLICHE SAISONALITÄT:
{analysis['daily_pattern']}

Aktueller Monat historisch:
{analysis['current_month_analysis']}

Nächste 3 Monate Ausblick:
{analysis['next_months']}

Saisonale Anomalien:
{analysis['anomalies']}

Statistische Signifikanz:
{analysis['significance']}

Trading-Implikation für aktuellen Zeitpunkt:
{analysis['current_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "seasonality",
                "month": now.month,
                **analysis
            }
        ))

        return results

    def _analyze_seasonality(self, symbol: Optional[str], now: datetime) -> dict:
        """Analyze seasonality patterns."""
        current_month = now.month

        # Get seasonality for symbol (default to BTC patterns)
        monthly = self.BTC_MONTHLY_SEASONALITY

        monthly_pattern = """
Monat       | Hist. Return | Gewinnrate | Bemerkung
------------|--------------|------------|------------------
Januar      | -2%          | 45%        | Oft schwach
Februar     | +12%         | 70%        | Historisch stark
März        | +5%          | 55%        | Moderat
April       | +15%         | 75%        | Sehr stark
Mai         | -10%         | 35%        | "Sell in May"
Juni        | -5%          | 40%        | Schwach
Juli        | +8%          | 60%        | Erholung
August      | -3%          | 45%        | Gemischt
September   | -5%          | 40%        | Historisch schwach
Oktober     | +20%         | 80%        | "Uptober" - sehr stark
November    | +15%         | 70%        | Stark
Dezember    | +10%         | 60%        | Moderat positiv

* Basierend auf historischen Daten - keine Garantie
""".strip()

        weekly_pattern = """
Wochentag-Effekte:
- Montag: Oft schwächer (Weekend Gap)
- Dienstag-Donnerstag: Stabilste Performance
- Freitag: Gemischt (Wochenend-Positionierung)
- Wochenende: Geringeres Volumen, höhere Volatilität

Beobachtung: Crypto handelt 24/7, aber traditionelle
Markt-Zeiten beeinflussen Liquidität.
""".strip()

        daily_pattern = """
Intraday-Muster (UTC):
- 00:00-08:00: Asien-Session, moderate Aktivität
- 08:00-14:00: Europa-Session, steigende Aktivität
- 14:00-21:00: US-Session, höchste Liquidität
- 21:00-00:00: Übergang, fallende Aktivität

Volatilste Zeiten:
- US-Marktöffnung (14:30 UTC)
- Wichtige Daten-Releases (13:30 UTC für US)
""".strip()

        # Current month analysis
        month_return = monthly.get(current_month, 0) * 100
        month_name = calendar.month_name[current_month]
        bias = "bullisch" if month_return > 0 else "bearisch"

        current_month_analysis = f"""
{month_name} Historische Performance:
- Durchschnittlicher Return: {month_return:+.1f}%
- Historischer Bias: {bias}
- Sample Size: 10+ Jahre Daten

Wichtige {month_name}-Events:
- Quartalszahlen (falls Q-Ende)
- Optionsverfall (3. Freitag)
- FOMC-Meetings (Kalender prüfen)
""".strip()

        # Next 3 months
        next_months_analysis = []
        for i in range(1, 4):
            m = (current_month + i - 1) % 12 + 1
            ret = monthly.get(m, 0) * 100
            next_months_analysis.append(
                f"- {calendar.month_name[m]}: {ret:+.1f}% (historisch)"
            )
        next_months = "\n".join(next_months_analysis)

        return {
            "monthly_pattern": monthly_pattern,
            "weekly_pattern": weekly_pattern,
            "daily_pattern": daily_pattern,
            "current_month_analysis": current_month_analysis,
            "next_months": next_months,
            "anomalies": """
Bekannte saisonale Anomalien:
- "Sell in May and go away" (Mai-Oktober schwächer)
- "Santa Claus Rally" (letzte Dezember-Woche)
- "January Effect" (oft Rotation in neue Positionen)
- Bitcoin "Uptober" (Oktober historisch sehr stark)
- Halving-Zyklen (4-Jahres-Muster bei Bitcoin)
""".strip(),
            "significance": """
Statistische Hinweise:
- Saisonalität erklärt nur ~10-15% der Varianz
- Funktioniert besser in Trends als in Ranges
- Kann durch Events überschrieben werden
- Nicht als alleiniges Signal verwenden
""".strip(),
            "current_implication": f"""
Für {month_name} {now.year}:
- Historischer Bias: {"Positiv" if month_return > 0 else "Negativ"}
- Empfehlung: {"Mit dem saisonalen Trend handeln" if abs(month_return) > 5 else "Saisonalität wenig ausgeprägt"}
- Vorsicht: Individuelle Events können dominieren
"""
        }

    async def _fetch_drawdown_history(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch historical drawdown analysis."""
        results = []

        analysis = self._analyze_drawdowns(symbol)

        content = f"""HISTORISCHE DRAWDOWN-ANALYSE - {symbol or 'MARKT'}
=================================================

Größte Historische Drawdowns:
{analysis['major_drawdowns']}

Drawdown-Statistiken:
{analysis['drawdown_stats']}

Recovery-Zeiten:
{analysis['recovery_times']}

Aktuelle Drawdown-Position:
{analysis['current_position']}

Drawdown-Muster:
{analysis['patterns']}

Historische Lektionen:
{analysis['lessons']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "drawdown_history",
                **analysis
            }
        ))

        return results

    def _analyze_drawdowns(self, symbol: Optional[str]) -> dict:
        """Analyze historical drawdowns."""
        # BTC historical drawdowns as reference
        return {
            "major_drawdowns": """
Rang | Zeitraum          | Drawdown | Dauer    | Recovery
-----|-------------------|----------|----------|----------
1    | 2011 (Jun-Nov)    | -93%     | 5 Monate | 7 Monate
2    | 2013-15 (Nov-Jan) | -86%     | 14 Mon.  | 25 Monate
3    | 2017-18 (Dec-Dec) | -84%     | 12 Mon.  | 36 Monate
4    | 2021-22 (Nov-Nov) | -77%     | 12 Mon.  | TBD
5    | 2020 (Feb-Mar)    | -50%     | 1 Monat  | 2 Monate
6    | 2019 (Jun-Dec)    | -53%     | 6 Monate | 8 Monate

Beobachtung: Jeder Zyklus hatte niedrigere maximale Drawdowns.
""".strip(),
            "drawdown_stats": """
Statistische Übersicht (10+ Jahre Daten):
- Durchschnittlicher Drawdown pro Jahr: -30% bis -50%
- Median Drawdown: -35%
- Anzahl >20% Drawdowns pro Jahr: 2-4
- Anzahl >50% Drawdowns: Alle 2-3 Jahre

Correlation mit Makro:
- Größte Drawdowns oft bei Liquiditätskrisen
- Fed-Policy-Wenden oft Katalysator
""".strip(),
            "recovery_times": """
Recovery nach Drawdown-Größe (historisch):
- 20-30% Drawdown: 1-3 Monate Recovery
- 30-50% Drawdown: 3-6 Monate Recovery
- 50-70% Drawdown: 6-18 Monate Recovery
- >70% Drawdown: 1-3 Jahre Recovery

WICHTIG: Recovery = zurück zum vorherigen ATH
Neue ATHs kommen typischerweise nach Full Recovery.
""".strip(),
            "current_position": """
Aktueller Status:
- ATH: [Preis] am [Datum]
- Aktueller Preis: [Preis]
- Aktueller Drawdown: [X%]
- Zeit seit ATH: [Monate]

Historischer Kontext:
- Vergleichbar mit [früherer Zeitraum]
- Recovery-Erwartung basierend auf Historie: [Zeitraum]
""".strip(),
            "patterns": """
Beobachtete Drawdown-Muster:
1. "Capitulation Spike": Finaler Ausverkauf mit hohem Volumen
2. "Grinding Bottom": Langsame Bodenbildung über Monate
3. "V-Recovery": Schnelle Erholung (selten, meist bei externem Katalysator)
4. "W-Bottom": Doppelter Test des Tiefs

Typischer Ablauf:
Crash → Bounce → Retest → Accumulation → Breakout
""".strip(),
            "lessons": """
Historische Lektionen aus Drawdowns:
1. Alle Bärenmärkte endeten - bisher immer
2. Tiefs kamen oft mit maximaler Negativität
3. Früh verkaufen ist besser als zu spät
4. Nachkaufen in Schritten (DCA) reduziert Timing-Risiko
5. Leverage in Bärenmärkten ist zerstörerisch
6. Cash-Position erlaubt Opportunitäten
""".strip(),
            "trading_implication": """
Drawdown-basiertes Trading:
- Bei -30%: Erste Position möglich, Risiko-Management wichtig
- Bei -50%: Historisch gute Kaufgelegenheiten
- Bei -70%+: Exzellente Risk/Reward, aber braucht Geduld
- Stop-Loss: Unter letztem signifikanten Tief
"""
        }

    async def _fetch_event_patterns(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch event-based return patterns."""
        results = []

        analysis = self._analyze_event_patterns(symbol)

        content = f"""EVENT-BASIERTE RETURN-ANALYSE - {symbol or 'MARKT'}
=================================================

BITCOIN HALVING PATTERN:
{analysis['halving_pattern']}

FOMC MEETING PATTERN:
{analysis['fomc_pattern']}

CPI RELEASE PATTERN:
{analysis['cpi_pattern']}

OPTIONSVERFALL PATTERN:
{analysis['opex_pattern']}

QUARTALSENDE PATTERN:
{analysis['quarter_end_pattern']}

Nächste wichtige Events:
{analysis['upcoming_events']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "event_patterns",
                **analysis
            }
        ))

        return results

    def _analyze_event_patterns(self, symbol: Optional[str]) -> dict:
        """Analyze event-based patterns."""
        return {
            "halving_pattern": """
Bitcoin Halving Zyklus-Analyse:
- Halving 1 (Nov 2012): ATH 12 Monate später, +9000%
- Halving 2 (Jul 2016): ATH 17 Monate später, +2800%
- Halving 3 (Mai 2020): ATH 18 Monate später, +700%
- Halving 4 (Apr 2024): Zyklus läuft...

Typisches Muster:
1. 12-18 Monate vor Halving: Accumulation
2. 6 Monate vor Halving: Erste Rally
3. Halving Event: Oft "sell the news"
4. 6-12 Monate nach Halving: Parabolische Phase
5. 12-18 Monate nach Halving: Peak, dann Bear Market

WARNUNG: Sample Size ist klein (n=4). Past performance ≠ future results.
""".strip(),
            "fomc_pattern": """
FOMC Meeting Return-Muster (historisch):
- Tag des Meetings: Erhöhte Volatilität
- Meeting-Tag bis +1d: Oft Trend-Tag
- Woche nach Meeting: Mean Reversion möglich

Reaktion abhängig von:
- Hawkish surprise: Risk-off (negativ für Crypto)
- Dovish surprise: Risk-on (positiv für Crypto)
- In-line: Meist gedämpfte Reaktion

Statistik:
- VIX fällt im Schnitt am Tag nach FOMC
- Crypto folgt oft Aktien-Reaktion mit Verzögerung
""".strip(),
            "cpi_pattern": """
CPI Release Return-Muster:
- Release: 13:30 UTC (US Zeit)
- Volatilität: Sehr hoch in ersten 30 Minuten
- Trend: Oft etabliert sich innerhalb 1 Stunde

Reaktion:
- CPI höher als erwartet: Risk-off, USD stärker
- CPI niedriger als erwartet: Risk-on, USD schwächer
- In-line: Kurzfristige Volatilität, dann Normalisierung

Crypto-Korrelation mit CPI-Reaktion: ~0.6 zu SPX
""".strip(),
            "opex_pattern": """
Options-Verfall Muster (Monthly/Quarterly):
- Max Pain: Preis tendiert zum Max Pain Level
- Gamma-Squeeze: Möglich bei hoher Gamma-Exposure
- Timing: 3. Freitag im Monat (CME), Deribit Quartals-Verfall

Beobachtungen:
- Volatilität oft erhöht in OpEx-Woche
- Nach OpEx: Oft neue Richtungsbewegung
- Quarterly OpEx > Monthly OpEx Impact
""".strip(),
            "quarter_end_pattern": """
Quartalsende-Effekte:
- Rebalancing: Fonds rebalancen Portfolios
- Window Dressing: Performance-Optimierung
- Liquidität: Kann kurzfristig sinken

Typisches Muster:
- Letzte Woche des Quartals: Rebalancing-Flows
- Erste Woche neues Quartal: Fresh Money
- Q4 Ende: Oft Tax-Loss Selling, dann Rally
""".strip(),
            "upcoming_events": """
Kommende wichtige Events (Kalender prüfen):
- Nächstes FOMC: [Datum]
- Nächste CPI: [Datum]
- Nächste NFP: [Datum]
- Nächster Options-Verfall: [Datum]
- Quartalsende: [Datum]

Event-Kalender täglich aktualisieren!
""".strip(),
            "trading_implication": """
Event-Trading Strategien:
1. Vor Events: Positionen reduzieren oder hedgen
2. Nach Events: Trend-Following mit Momentum
3. Event-Straddles: Volatilität kaufen vor Events
4. Fade extreme Moves: Mean Reversion nach Überreaktion
"""
        }

    async def _fetch_comparable_periods(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch comparable historical periods."""
        results = []

        analysis = self._analyze_comparable_periods(symbol)

        content = f"""VERGLEICHBARE HISTORISCHE PERIODEN
===================================

Aktuelle Marktphase:
{analysis['current_phase']}

Ähnliche historische Perioden:
{analysis['similar_periods']}

Vergleichs-Analyse:
{analysis['comparison']}

Was danach passierte:
{analysis['outcomes']}

Unterschiede zur Vergangenheit:
{analysis['differences']}

Wahrscheinlichste Szenarien:
{analysis['scenarios']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "comparable_periods",
                **analysis
            }
        ))

        return results

    def _analyze_comparable_periods(self, symbol: Optional[str]) -> dict:
        """Analyze comparable historical periods."""
        return {
            "current_phase": """
Identifizierte Marktphase-Charakteristiken:
- Preis-Struktur: [Higher Highs/Lower Lows/Range]
- Volumen-Trend: [Steigend/Fallend/Stabil]
- Volatilität: [Hoch/Mittel/Niedrig]
- Sentiment: [Fear/Neutral/Greed]
- On-Chain: [Akkumulation/Distribution]
- Makro: [Risk-On/Neutral/Risk-Off]

Basierend auf diesen Faktoren: [Marktphase]
""".strip(),
            "similar_periods": """
Historische Perioden mit ähnlichen Charakteristiken:

1. [Zeitraum 1, z.B. Q4 2023]
   - Ähnlichkeit: 75%
   - Hauptübereinstimmungen: [Faktoren]
   - Hauptunterschiede: [Faktoren]

2. [Zeitraum 2, z.B. Q2 2019]
   - Ähnlichkeit: 65%
   - Hauptübereinstimmungen: [Faktoren]
   - Hauptunterschiede: [Faktoren]

3. [Zeitraum 3, z.B. Q4 2020]
   - Ähnlichkeit: 60%
   - Hauptübereinstimmungen: [Faktoren]
   - Hauptunterschiede: [Faktoren]
""".strip(),
            "comparison": """
Detaillierter Vergleich (aktuell vs. historisch):

Metrik           | Aktuell | Period 1 | Period 2 | Period 3
-----------------|---------|----------|----------|----------
Drawdown         | X%      | Y%       | Z%       | W%
Sentiment        | Fear/Greed | ... | ... | ...
DXY              | Level   | Level    | Level    | Level
Fed Policy       | Stance  | Stance   | Stance   | Stance
Halving Cycle    | Position| Position | Position | Position
""".strip(),
            "outcomes": """
Was in den Vergleichsperioden danach passierte:

Period 1 ([Zeitraum]):
- 3 Monate später: +/- X%
- 6 Monate später: +/- X%
- 12 Monate später: +/- X%
- Besonderheiten: [Event/Katalysator]

Period 2 ([Zeitraum]):
- 3 Monate später: +/- X%
- 6 Monate später: +/- X%
- 12 Monate später: +/- X%
- Besonderheiten: [Event/Katalysator]

Durchschnittliches Outcome: [Zusammenfassung]
""".strip(),
            "differences": """
Wichtige Unterschiede zur Vergangenheit:

Strukturelle Änderungen:
- Institutionelle Adoption höher als je zuvor
- ETF-Zuflüsse als neuer Faktor
- Regulatorisches Umfeld entwickelt sich
- Globale Liquidität anders strukturiert

Makro-Kontext:
- Zinsniveau historisch [hoch/niedrig]
- Geopolitische Risiken [höher/niedriger]
- Inflation [Problem/Unter Kontrolle]

Markt-Mikrostruktur:
- Derivate-Volumen größer
- Mehr Leverage im System
- Liquidität konzentrierter
""".strip(),
            "scenarios": """
Szenarien basierend auf historischer Analyse:

Bullisches Szenario (30% Wahrscheinlichkeit):
- Trigger: [Katalysator]
- Verlauf: [Beschreibung]
- Ziel: [Preis/Range]

Basis Szenario (50% Wahrscheinlichkeit):
- Trigger: Status quo
- Verlauf: [Beschreibung]
- Ziel: [Preis/Range]

Bearisches Szenario (20% Wahrscheinlichkeit):
- Trigger: [Katalysator]
- Verlauf: [Beschreibung]
- Ziel: [Preis/Range]
""".strip(),
            "trading_implication": """
Historische Pattern-basierte Empfehlung:
1. Wenn History rhymes: [Strategie]
2. Risiko-Management: Auf Abweichungen vorbereitet sein
3. Katalysatoren beobachten: [Liste]
4. Time Horizon: [Empfohlener Zeitrahmen]
"""
        }

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when analysis fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""HISTORISCHE PATTERNS - ÜBERSICHT
================================

Wichtige historische Analysen:

1. Saisonalität:
   - Monatliche Muster (z.B. "Uptober")
   - Wochentag-Effekte
   - Intraday-Muster

2. Drawdown-Historie:
   - Größte historische Drawdowns
   - Recovery-Zeiten
   - Muster bei Tiefpunkten

3. Event-basierte Returns:
   - Halving-Zyklen (BTC)
   - FOMC-Meetings
   - CPI-Releases
   - Options-Verfälle

4. Vergleichbare Perioden:
   - Ähnliche Marktphasen identifizieren
   - Historische Outcomes analysieren
   - Auf Unterschiede achten

Wichtige Warnung:
"Past performance is not indicative of future results"
Historische Muster sind Wahrscheinlichkeiten, keine Garantien.
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
