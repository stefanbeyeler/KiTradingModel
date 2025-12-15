"""Regulatory Updates Data Source - SEC, CFTC, global regulations, ETF updates."""

from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class RegulatoryUpdatesSource(DataSourceBase):
    """
    Provides regulatory news and updates relevant to trading.

    Data includes:
    - SEC/CFTC decisions and rulings
    - ETF approvals and flow data
    - Global regulatory changes (EU, Asia, etc.)
    - Enforcement actions
    - Policy proposals and comment periods
    - Stablecoin regulations
    - Exchange/Broker regulations
    """

    source_type = DataSourceType.REGULATORY

    # Regulatory bodies
    REGULATORS = {
        "US": ["SEC", "CFTC", "FinCEN", "OCC", "Federal Reserve"],
        "EU": ["ESMA", "EBA", "MiCA Framework"],
        "UK": ["FCA"],
        "Asia": ["MAS (Singapore)", "SFC (Hong Kong)", "FSA (Japan)", "VARA (Dubai)"],
        "Global": ["FATF", "BIS", "IOSCO"]
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 3600  # 1 hour for regulatory news

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch regulatory updates.

        Args:
            symbol: Trading symbol for context
            include_sec: Include SEC updates
            include_etf: Include ETF flow/approval data
            include_global: Include global regulatory news
            include_enforcement: Include enforcement actions

        Returns:
            List of regulatory update results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []

        include_sec = kwargs.get("include_sec", True)
        include_etf = kwargs.get("include_etf", True)
        include_global = kwargs.get("include_global", True)
        include_enforce = kwargs.get("include_enforcement", True)

        try:
            if include_sec:
                sec_data = await self._fetch_sec_updates(symbol)
                results.extend(sec_data)

            if include_etf:
                etf_data = await self._fetch_etf_updates(symbol)
                results.extend(etf_data)

            if include_global:
                global_data = await self._fetch_global_regulatory(symbol)
                results.extend(global_data)

            if include_enforce:
                enforce_data = await self._fetch_enforcement_actions(symbol)
                results.extend(enforce_data)

            # Stablecoin specific regulations
            stable_data = await self._fetch_stablecoin_regulation()
            results.extend(stable_data)

        except Exception as e:
            logger.error(f"Error fetching regulatory updates: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch regulatory updates formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_sec_updates(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch SEC related updates."""
        results = []

        analysis = self._analyze_sec_updates(symbol)

        content = f"""SEC REGULATORISCHE UPDATES
===========================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

AKTUELLE SEC POSITIONEN:
{analysis['current_positions']}

ANHÄNGIGE ENTSCHEIDUNGEN:
{analysis['pending_decisions']}

LAUFENDE FÄLLE:
{analysis['ongoing_cases']}

SEC KRYPTO-KLASSIFIZIERUNG:
{analysis['crypto_classification']}

SEC KOMMENTARFRISTEN:
{analysis['comment_periods']}

HISTORISCHE ENTSCHEIDUNGEN (Präzedenzfälle):
{analysis['historical_decisions']}

Markt-Implikationen:
{analysis['market_implications']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "sec_updates",
                **analysis
            }
        ))

        return results

    def _analyze_sec_updates(self, symbol: Optional[str]) -> dict:
        """Analyze SEC regulatory landscape."""
        return {
            "current_positions": """
SEC Aktuelle Haltung zu Krypto-Assets:
- Bitcoin: Warenähnlich (Commodity) - NICHT Security
- Ethereum: Position nach Merge unklar
- Andere Altcoins: Oft als Securities eingestuft
- Stablecoins: Potenziell Securities je nach Struktur

Howey Test Kriterien (Security Definition):
1. Investment of money
2. In a common enterprise
3. With expectation of profits
4. Derived from efforts of others

SEC prüft jeden Token individuell nach Howey.
""".strip(),
            "pending_decisions": """
Ausstehende regulatorische Entscheidungen:

1. Spot Ethereum ETF
   - Status: [Anhängig/Genehmigt/Abgelehnt]
   - Deadline: [Datum]
   - Antragsteller: [Liste]

2. Weitere Krypto-Produkte
   - Solana ETF: [Status]
   - Index-Produkte: [Status]

3. Exchange-Registrierungen
   - [Exchange] vs SEC: [Status]

WICHTIG: Deadlines können verschoben werden.
""".strip(),
            "ongoing_cases": """
Wichtige laufende SEC-Fälle:

1. SEC vs. Ripple (XRP)
   - Status: Teilentscheidung - XRP nicht per se Security
   - Implikation: Präzedenzfall für andere Token
   - Nächste Schritte: [Update]

2. SEC vs. Coinbase
   - Status: Laufend
   - Vorwürfe: Unregistrierte Securities
   - Implikation: Exchange-Regulierung

3. SEC vs. Binance
   - Status: Laufend
   - Vorwürfe: Multiple Verstöße
   - Implikation: Globale Börsenregulierung

Fälle können Monate bis Jahre dauern.
""".strip(),
            "crypto_classification": """
SEC Krypto-Klassifizierung (aktueller Stand):

Als Securities eingestuft:
- Die meisten ICO Tokens
- Tokens mit Profit-Versprechen
- Tokens aus Pre-Sales an Investoren

NICHT als Securities:
- Bitcoin (klar)
- Ethereum (nach aktueller Position)
- Sufficiently decentralized Tokens

Grauzone:
- Viele Top-50 Altcoins
- DeFi Tokens
- NFTs (fallabhängig)
""".strip(),
            "comment_periods": """
Offene Kommentarfristen:

1. [Regelvorschlag]: Frist bis [Datum]
2. [Regelvorschlag]: Frist bis [Datum]

Kommentarperioden = Öffentlichkeit kann Feedback geben
vor finaler Regelverabschiedung.
""".strip(),
            "historical_decisions": """
Wichtige Präzedenzfälle:

- 2017: DAO Report - Token können Securities sein
- 2019: Framework für Investment Contract Analysis
- 2023: Ripple Urteil - programmatic sales != Security
- 2024: Spot BTC ETFs genehmigt

Diese Entscheidungen beeinflussen zukünftige Fälle.
""".strip(),
            "market_implications": """
Markt reagiert auf SEC News:

Positive Ereignisse:
- ETF-Genehmigungen: +5-20% typisch
- Favorable Urteile: +10-30% für betroffenen Token
- Klare Regulatory Clarity: Bullisch für Sektor

Negative Ereignisse:
- Klagen/Wells Notices: -10-30%
- Ablehnungen: -5-15%
- Enforcement Actions: Stark negativ für betroffene Token
""".strip(),
            "trading_implication": """
SEC News Trading:

1. Pre-Event: Positionen vor Deadlines reduzieren
2. Post-Event: Schnelle Reaktion auf News
3. Langfristig: Regulatory Clarity = Bullisch

Risk Management:
- Bei SEC-Uncertainty: Exposure reduzieren
- Nach negativer News: Nicht in fallende Messer greifen
- Bei positiver News: Nicht FOMO kaufen
"""
        }

    async def _fetch_etf_updates(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch ETF approval and flow data."""
        results = []

        analysis = self._analyze_etf_data(symbol)

        content = f"""ETF UPDATES & FLOWS
====================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

GENEHMIGTE KRYPTO-ETFs:
{analysis['approved_etfs']}

ETF FLOW DATEN:
{analysis['flow_data']}

AUSSTEHENDE ETF-ANTRÄGE:
{analysis['pending_applications']}

ETF IMPACT ANALYSE:
{analysis['impact_analysis']}

HISTORISCHE ETF LAUNCHES:
{analysis['historical_launches']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "etf_updates",
                **analysis
            }
        ))

        return results

    def _analyze_etf_data(self, symbol: Optional[str]) -> dict:
        """Analyze ETF landscape."""
        return {
            "approved_etfs": """
Genehmigte US Spot Bitcoin ETFs (seit Jan 2024):

Ticker | Anbieter      | AUM          | Gebühren
-------|---------------|--------------|----------
IBIT   | BlackRock     | $XX Mrd      | 0.25%
FBTC   | Fidelity      | $XX Mrd      | 0.25%
ARKB   | ARK/21Shares  | $XX Mrd      | 0.21%
BITB   | Bitwise       | $XX Mrd      | 0.20%
HODL   | VanEck        | $XX Mrd      | 0.25%
GBTC   | Grayscale     | $XX Mrd      | 1.50%

Gesamt AUM: $XX Milliarden
(Daten werden täglich aktualisiert)

Spot Ethereum ETFs:
- Status: [Genehmigt/Anhängig]
- Produkte: [Liste]
""".strip(),
            "flow_data": """
ETF Flow Daten (Aggregiert):

Zeitraum | Netto-Flow | Kumulativ
---------|------------|----------
Heute    | $XXX Mio   | $XX Mrd
7 Tage   | $XXX Mio   | -
30 Tage  | $XXX Mio   | -
YTD      | $XXX Mrd   | -

Flow-Trend: [Positiv/Negativ/Neutral]

Top Zuflüsse: [ETFs]
Top Abflüsse: [ETFs, oft GBTC]

Flow-Daten als Leading Indicator:
- Starke Zuflüsse: Bullisch
- Anhaltende Abflüsse: Bearisch
""".strip(),
            "pending_applications": """
Ausstehende ETF-Anträge:

Spot Ethereum:
1. [Anbieter] - Deadline: [Datum]
2. [Anbieter] - Deadline: [Datum]

Andere Krypto-ETFs:
1. Solana ETF - [Anbieter] - Status: [Early Stage]
2. XRP ETF - [Anbieter] - Status: [Spekulativ]
3. Multi-Crypto Index - [Anbieter] - Status: [Review]

SEC kann Deadlines um bis zu 240 Tage verschieben.
""".strip(),
            "impact_analysis": """
ETF Impact auf Preise:

Historischer Bitcoin ETF Impact:
- Vor Genehmigung (Q4 2023): +50-100% Rally
- Bei Genehmigung: "Sell the news" -15%
- 3 Monate danach: +40%
- Langfristig: Strukturelle Nachfrage

Erwarteter ETH ETF Impact:
- Ähnliches Muster wahrscheinlich
- Aber: Markt bereits erwartet es
- Möglicher Impact: +20-50% bei Genehmigung

Daily Flow Impact:
- $100M+ Zuflüsse: Tendenziell bullisch
- $100M+ Abflüsse: Tendenziell bearisch
- Flows vs. Preis: ~0.5 Korrelation
""".strip(),
            "historical_launches": """
Historische Krypto-ETF Launches:

2021: Bitcoin Futures ETFs (BITO)
- Launch: Oktober 2021
- Preis bei Launch: ~$63,000 (ATH)
- 3 Monate später: -50%

2024: Spot Bitcoin ETFs
- Launch: Januar 2024
- Preis bei Launch: ~$46,000
- Entwicklung: [Update]

Lektion: ETF Launch ≠ garantiert bullisch
Oft bereits "eingepreist" vor Launch.
""".strip(),
            "trading_implication": """
ETF Flow-basiertes Trading:

1. Flow Momentum: Starke Flows = Trend-Bestätigung
2. Flow Divergenz: Abflüsse bei Rallye = Warnsignal
3. Pre-Deadline: Volatilität vor ETF Entscheidungen
4. Post-Launch: Oft "sell the news"

Datenquellen für Flows:
- Farside Investors
- SoSoValue
- Bloomberg Terminal
"""
        }

    async def _fetch_global_regulatory(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch global regulatory updates."""
        results = []

        analysis = self._analyze_global_regulation()

        content = f"""GLOBALE REGULIERUNG UPDATES
=============================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

EUROPÄISCHE UNION (MiCA):
{analysis['eu_mica']}

GROSSBRITANNIEN:
{analysis['uk_regulation']}

ASIEN-PAZIFIK:
{analysis['asia_regulation']}

ANDERE JURISDIKTIONEN:
{analysis['other_jurisdictions']}

REGULATORISCHE ARBITRAGE:
{analysis['regulatory_arbitrage']}

GLOBAL COORDINATION:
{analysis['global_coordination']}

Implikationen für Markt:
{analysis['market_implications']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "global_regulatory",
                **analysis
            }
        ))

        return results

    def _analyze_global_regulation(self) -> dict:
        """Analyze global regulatory landscape."""
        return {
            "eu_mica": """
Markets in Crypto-Assets (MiCA) Framework:
- Status: In Kraft seit 2024
- Scope: Alle EU-Mitgliedsstaaten

Kernpunkte:
1. Lizenzierung für Crypto Asset Service Provider (CASPs)
2. Stablecoin-Regulierung (strenge Reserve-Anforderungen)
3. Whitepaper-Anforderungen für Token
4. Verbraucherschutz-Regeln

Impact:
- Rechtssicherheit für EU-Markt
- Höhere Compliance-Kosten
- Mögliche Konsolidierung kleiner Player
- Stablecoins müssen EU-Reserven halten
""".strip(),
            "uk_regulation": """
UK Crypto Regulation (FCA):
- Status: Entwickelnd

Aktuelle Regeln:
- Crypto-Marketing Regeln (stark)
- Registrierungspflicht für Anbieter
- AML/KYC Anforderungen

Geplant:
- Umfassendes Crypto-Regime
- Staking-Regulierung
- DeFi-Ansätze

UK Position: Pro-Innovation aber mit Guardrails
""".strip(),
            "asia_regulation": """
Asien-Pazifik Regulierung:

Singapur (MAS):
- Lizenziertes Regime
- Stablecoin Framework
- Pro-Innovation Ansatz

Hong Kong (SFC):
- Retail Crypto Handel erlaubt (2024)
- Lizenzierung für Börsen
- Spot ETF Anträge

Japan (FSA):
- Streng reguliert seit 2017
- Stablecoin-Gesetze
- Separierung Kundengelder

Dubai (VARA):
- Neue Crypto-freundliche Zone
- Weltweite Unternehmen siedeln sich an
""".strip(),
            "other_jurisdictions": """
Weitere wichtige Jurisdiktionen:

Schweiz:
- Sehr krypto-freundlich
- Crypto Valley Zug
- DLT Gesetz

Bermuda/Bahamas:
- Offshore-freundlich
- FTX war dort ansässig (Warnung)

Australien:
- Entwickelt Rahmenwerk
- Moderate Position

Indien:
- 30% Steuer auf Crypto
- TDS auf Transaktionen
- Keine klare Regulierung
""".strip(),
            "regulatory_arbitrage": """
Regulatorische Arbitrage:

Unternehmen wählen Jurisdiktionen basierend auf:
1. Regulatorische Klarheit
2. Steuerliche Behandlung
3. Banking-Zugang
4. Talent-Pool

Trends:
- Weg von USA (Unsicherheit)
- Hin zu Dubai, Singapur, EU
- Dezentrale Protokolle: Keine Jurisdiktion

Risiko: Regulatorisches Risiko bleibt für User
die über diese Anbieter handeln.
""".strip(),
            "global_coordination": """
Globale Regulierungs-Koordination:

FATF (Financial Action Task Force):
- Travel Rule: Crypto-Transfers melden
- AML Standards für alle Jurisdiktionen

BIS (Bank for International Settlements):
- CBDC Forschung
- Krypto-Asset Behandlung für Banken

IOSCO:
- Globale Standards für Crypto-Märkte
- Cross-Border Koordination

Trend: Zunehmende globale Harmonisierung
""".strip(),
            "market_implications": """
Regulatorische Klarheit = Langfristig bullisch
- Institutionelle Adoption benötigt Regeln
- Retail-Schutz erhöht Vertrauen
- Aber: Kurzfristig kann Regulierung belasten

Jurisdiktions-Arbitrage:
- Projekte verschieben sich zu freundlichen Zonen
- User in restriktiven Zonen limitiert
- Globale Fragmentierung möglich
"""
        }

    async def _fetch_enforcement_actions(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch regulatory enforcement actions."""
        results = []

        analysis = self._analyze_enforcement()

        content = f"""ENFORCEMENT ACTIONS & WARNUNGEN
================================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

AKTUELLE ENFORCEMENT ACTIONS:
{analysis['current_actions']}

WELLS NOTICES:
{analysis['wells_notices']}

SETTLEMENTS:
{analysis['settlements']}

WARNUNGEN & RED FLAGS:
{analysis['warnings']}

PRÄVENTIVE MASSNAHMEN:
{analysis['preventive_measures']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "enforcement",
                **analysis
            }
        ))

        return results

    def _analyze_enforcement(self) -> dict:
        """Analyze enforcement landscape."""
        return {
            "current_actions": """
Laufende Enforcement-Verfahren:

SEC:
1. [Unternehmen/Projekt] - Vorwurf: [Beschreibung]
2. [Unternehmen/Projekt] - Vorwurf: [Beschreibung]

CFTC:
1. [Unternehmen/Projekt] - Vorwurf: [Beschreibung]

DOJ (Criminal):
1. [Fall] - Status: [Laufend/Abgeschlossen]

WICHTIG: Laufende Verfahren = erhöhtes Risiko
für betroffene Token und assoziierte Assets.
""".strip(),
            "wells_notices": """
Wells Notices (SEC):
= Formelle Warnung vor möglicher Klage

Aktuelle Wells Notices:
1. [Unternehmen] - Datum: [X], Response: [Y]
2. [Unternehmen] - Datum: [X], Response: [Y]

Nach Wells Notice:
- Unternehmen kann antworten
- SEC entscheidet ob Klage
- Typisch 60-90 Tage bis Entscheidung

Markt-Reaktion auf Wells Notice: Oft -20-40%
""".strip(),
            "settlements": """
Wichtige Settlements:

1. [Unternehmen] - $XXX Mio
   - Vorwurf: [Beschreibung]
   - Auflage: [Bedingungen]

2. [Unternehmen] - $XXX Mio
   - Vorwurf: [Beschreibung]
   - Auflage: [Bedingungen]

Settlements = Nicht Schuldeingeständnis
aber: Zahlung + Compliance-Auflagen
""".strip(),
            "warnings": """
Aktive Warnungen & Red Flags:

Ponzi/Betrug Warnungen:
- [Projekt] - Warnung von [Behörde]
- [Projekt] - Investigation aktiv

Unregistrierte Securities:
- Tokens die als Securities eingestuft wurden
- Projekte mit SEC Vorladungen

Operational Risks:
- Börsen mit Banking-Problemen
- Projekte mit Team-Exits
- Smart Contract Vulnerabilities

IMMER DYOR (Do Your Own Research)
""".strip(),
            "preventive_measures": """
Selbstschutz vor Regulatorischem Risiko:

1. Diversifikation:
   - Nicht alles in einem Token
   - Nicht alles auf einer Börse

2. Research:
   - Team und Backing prüfen
   - Rechtliche Struktur verstehen
   - Community und Transparenz

3. Custody:
   - Self-Custody für langfristige Holdings
   - Regulierte Custodians für Institutionen

4. Steuer-Compliance:
   - Transaktionen dokumentieren
   - Lokale Steuerregeln befolgen
""".strip(),
            "trading_implication": """
Enforcement News Trading:

Auf Enforcement News reagieren:
- Klage/Investigation: Sofort Exit erwägen
- Wells Notice: Risiko-Reduktion
- Settlement: Oft End-of-Uncertainty Bounce
- Klage gewonnen: Starke Rally möglich

Risk Management:
- Exposure zu "unter Investigation" Tokens minimieren
- Regulatory Risk Premium einkalkulieren
"""
        }

    async def _fetch_stablecoin_regulation(self) -> list[DataSourceResult]:
        """Fetch stablecoin-specific regulatory updates."""
        results = []

        analysis = self._analyze_stablecoin_regulation()

        content = f"""STABLECOIN REGULIERUNG
=======================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

US STABLECOIN POLITIK:
{analysis['us_policy']}

EU STABLECOIN (MiCA):
{analysis['eu_policy']}

GLOBALE STABLECOIN REGULIERUNG:
{analysis['global_policy']}

RISIKEN & COMPLIANCE:
{analysis['risks']}

AUSWIRKUNGEN AUF MARKT:
{analysis['market_impact']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "stablecoin_regulation",
                **analysis
            }
        ))

        return results

    def _analyze_stablecoin_regulation(self) -> dict:
        """Analyze stablecoin regulatory landscape."""
        return {
            "us_policy": """
US Stablecoin Regulierung:

Aktuelle Position:
- Keine spezifische Bundesgesetzgebung
- SEC: Einige Stablecoins könnten Securities sein
- Federal Reserve: Beobachtet Risiken

Geplante Gesetzgebung:
- Stablecoin-Gesetzesentwürfe im Kongress
- Reserve-Anforderungen diskutiert
- Bank-Lizenzierung für Emittenten

Tether (USDT):
- Größter Stablecoin
- Keine US-Lizenz
- Reserve-Zusammensetzung teils unklar

Circle (USDC):
- Transparentere Reserven
- Besser reguliert
- IPO geplant
""".strip(),
            "eu_policy": """
EU MiCA Stablecoin-Regeln:

Anforderungen ab 2024:
1. Vollständig durch liquide Assets gedeckt
2. Reserven bei EU-Banken halten
3. E-Money-Lizenz erforderlich
4. Tägliche Reserve-Berichte
5. Rücknahmerechte für Holder

Impact:
- Tether (USDT) nicht MiCA-konform
- USDC arbeitet an EU-Compliance
- EU-native Stablecoins entstehen

Volume Cap: €200 Mio/Tag für Nicht-EUR Stablecoins
(könnte USDT/USDC Nutzung in EU limitieren)
""".strip(),
            "global_policy": """
Globale Stablecoin-Ansätze:

Singapur:
- Stablecoin Framework veröffentlicht
- SGD-gedeckte Stablecoins gefördert

UK:
- Stablecoin-Regulierung geplant
- Fokus auf systemische Stablecoins

Japan:
- Nur Yen-gedeckte Stablecoins erlaubt
- Strenge Emittenten-Regeln

Hong Kong:
- Stablecoin-Lizenzierung entwickelt

Trend: Globale Verschärfung der Regeln
""".strip(),
            "risks": """
Stablecoin Risiken:

1. Regulatorisches Risiko:
   - De-Listing von Börsen möglich
   - Banking-Partner können abspringen
   - Jurisdiktions-Beschränkungen

2. Reserve-Risiko:
   - Nicht alle Stablecoins gleich gedeckt
   - Commercial Paper vs. Cash/T-Bills
   - Transparenz variiert stark

3. Smart Contract Risiko:
   - Blacklisting durch Emittenten möglich
   - Upgrade-Risiken

4. Systemisches Risiko:
   - USDT Failure wäre katastrophal
   - Markt sehr konzentriert

Empfehlung: Stablecoin-Exposure diversifizieren
""".strip(),
            "market_impact": """
Stablecoin Regulatory Impact:

Bullisch:
- Klare Regulierung = Institutionelle Adoption
- Transparente Reserven = mehr Vertrauen
- CBDCs könnten Crypto-Brücken sein

Bearisch:
- Zu strenge Regeln = Liquiditätsverlust
- USDT-Probleme = Markt-Crash-Risiko
- Regulatorische Fragmentierung

Stablecoins = Lebenssaft der Crypto-Märkte
Regulierung dieser Assets ist kritisch für
die gesamte Marktstruktur.
"""
        }

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when fetch fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""REGULATORISCHE UPDATES - ÜBERSICHT
===================================

Wichtige Regulierungsbereiche:

1. SEC (US Securities):
   - Krypto als Securities
   - ETF Genehmigungen
   - Enforcement Actions

2. CFTC (US Commodities):
   - Bitcoin als Commodity
   - Derivate-Regulierung

3. EU MiCA:
   - Umfassendes Krypto-Framework
   - Stablecoin-Regeln
   - Lizenzierungen

4. Globale Regulierung:
   - FATF Travel Rule
   - Nationale Ansätze variieren
   - Regulatorische Arbitrage

5. ETF Flows:
   - Tägliche Zu-/Abflüsse
   - Neuer Demand Driver

6. Stablecoins:
   - Reserve-Anforderungen
   - Emittenten-Regulierung

Datenquellen:
- SEC.gov
- CFTC.gov
- Farside Investors (ETF Flows)
- Crypto-News Outlets
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
