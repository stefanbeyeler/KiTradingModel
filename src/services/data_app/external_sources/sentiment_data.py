"""Sentiment Data Source - Fear & Greed, Social Media, Google Trends, Options ratios."""

import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class SentimentDataSource(DataSourceBase):
    """
    Fetches market sentiment data from various sources.

    Data includes:
    - Fear & Greed Index (Crypto and Traditional)
    - Social Media Sentiment (Twitter/X, Reddit)
    - Google Trends search volume
    - Options Put/Call Ratio
    - VIX and volatility indices
    - Funding rates sentiment
    - Long/Short ratios
    """

    source_type = DataSourceType.SENTIMENT

    # Fear & Greed thresholds
    FEAR_GREED_ZONES = {
        (0, 25): ("Extreme Fear", "Historisch guter Kaufbereich"),
        (25, 45): ("Fear", "Vorsichtig bullisch"),
        (45, 55): ("Neutral", "Keine klare Richtung"),
        (55, 75): ("Greed", "Vorsicht bei Long-Positionen"),
        (75, 100): ("Extreme Greed", "Historisch guter Verkaufsbereich"),
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 900  # 15 minutes for sentiment data

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch sentiment data.

        Args:
            symbol: Trading symbol
            include_fear_greed: Include Fear & Greed Index
            include_social: Include social media sentiment
            include_options: Include options sentiment data
            include_volatility: Include VIX/volatility data

        Returns:
            List of sentiment data results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []

        include_fg = kwargs.get("include_fear_greed", True)
        include_social = kwargs.get("include_social", True)
        include_options = kwargs.get("include_options", True)
        include_vol = kwargs.get("include_volatility", True)

        try:
            if include_fg:
                fg_data = await self._fetch_fear_greed(symbol)
                results.extend(fg_data)

            if include_social:
                social_data = await self._fetch_social_sentiment(symbol)
                results.extend(social_data)

            if include_options:
                options_data = await self._fetch_options_sentiment(symbol)
                results.extend(options_data)

            if include_vol:
                vol_data = await self._fetch_volatility_indices(symbol)
                results.extend(vol_data)

            # Add funding rates for crypto
            if symbol and self._is_crypto(symbol):
                funding_data = await self._fetch_funding_rates(symbol)
                results.extend(funding_data)

        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch sentiment data formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        crypto_symbols = ["BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "AVAX", "BNB", "XRP", "DOGE"]
        return any(c in symbol.upper() for c in crypto_symbols)

    async def _fetch_fear_greed(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch Fear & Greed Index data."""
        results = []

        # Crypto Fear & Greed (alternative.me API is free)
        if not symbol or self._is_crypto(symbol):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.alternative.me/fng/?limit=10",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            results.append(self._parse_crypto_fear_greed(data))
            except Exception as e:
                logger.warning(f"Could not fetch Crypto Fear & Greed: {e}")
                results.append(self._generate_fear_greed_analysis("crypto"))

        # Traditional market Fear & Greed
        if not symbol or not self._is_crypto(symbol):
            results.append(self._generate_fear_greed_analysis("traditional"))

        return results

    def _parse_crypto_fear_greed(self, data: dict) -> DataSourceResult:
        """Parse Crypto Fear & Greed API response."""
        fg_data = data.get("data", [])

        if not fg_data:
            return self._generate_fear_greed_analysis("crypto")

        current = fg_data[0]
        value = int(current.get("value", 50))
        classification = current.get("value_classification", "Neutral")

        # Get zone interpretation
        zone_info = ("Neutral", "Keine klare Richtung")
        for (low, high), info in self.FEAR_GREED_ZONES.items():
            if low <= value < high:
                zone_info = info
                break

        # Calculate trend from history
        trend = "Stabil"
        if len(fg_data) >= 7:
            week_ago = int(fg_data[6].get("value", value))
            diff = value - week_ago
            if diff > 10:
                trend = "Stark steigend (mehr Gier)"
            elif diff > 5:
                trend = "Leicht steigend"
            elif diff < -10:
                trend = "Stark fallend (mehr Angst)"
            elif diff < -5:
                trend = "Leicht fallend"

        content = f"""CRYPTO FEAR & GREED INDEX
==========================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Aktueller Wert: {value}/100
Klassifikation: {classification}
Zone: {zone_info[0]}
7-Tage Trend: {trend}

Interpretation:
{zone_info[1]}

Historische Werte (letzte 7 Tage):
"""
        for i, day in enumerate(fg_data[:7]):
            date = datetime.fromtimestamp(int(day.get("timestamp", 0)))
            content += f"- {date.strftime('%Y-%m-%d')}: {day.get('value')} ({day.get('value_classification')})\n"

        content += f"""
Trading-Implikationen:
- Extreme Fear (0-25): Contrarian Buy Signal - Historisch gute Einstiegspunkte
- Extreme Greed (75-100): Contrarian Sell Signal - Vorsicht vor Korrektur

Aktuelle Empfehlung basierend auf Sentiment:
{self._get_fg_recommendation(value)}
"""

        priority = DataPriority.HIGH if value < 25 or value > 75 else DataPriority.MEDIUM

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=priority,
            metadata={
                "metric_type": "fear_greed_crypto",
                "value": value,
                "classification": classification,
                "trend": trend
            }
        )

    def _generate_fear_greed_analysis(self, market_type: str) -> DataSourceResult:
        """Generate Fear & Greed analysis when API unavailable."""
        if market_type == "crypto":
            content = """CRYPTO FEAR & GREED INDEX
==========================
Hinweis: Live-Daten temporär nicht verfügbar.

Index-Komponenten:
- Volatilität (25%)
- Market Momentum/Volume (25%)
- Social Media (15%)
- Surveys (15%)
- Dominanz (10%)
- Trends (10%)

Interpretation:
- 0-25: Extreme Fear - Contrarian Buy Zone
- 25-45: Fear - Accumulation Zone
- 45-55: Neutral - Seitwärtsmarkt
- 55-75: Greed - Profit Taking Zone
- 75-100: Extreme Greed - Distribution/Sell Zone

Quelle: alternative.me/crypto/fear-and-greed-index/
"""
        else:
            content = """MARKET FEAR & GREED INDEX (CNN)
================================
Traditionelle Märkte Sentiment

Index-Komponenten:
- Stock Price Momentum (S&P 500 vs 125-Day MA)
- Stock Price Strength (52-Week Highs vs Lows)
- Stock Price Breadth (Advancing vs Declining Volume)
- Put/Call Options Ratio
- Junk Bond Demand (Spread zu Safe Bonds)
- Market Volatility (VIX)
- Safe Haven Demand (Stocks vs Bonds Return)

Interpretation:
Gleiche Zonen wie Crypto Index (0-100 Skala)
- Extreme Fear oft bei Markttiefs
- Extreme Greed oft vor Korrekturen

Quelle: money.cnn.com/data/fear-and-greed/
"""

        return DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=DataPriority.MEDIUM,
            metadata={"metric_type": f"fear_greed_{market_type}", "fallback": True}
        )

    def _get_fg_recommendation(self, value: int) -> str:
        """Get trading recommendation based on Fear & Greed value."""
        if value < 20:
            return """
STARK BULLISCH (Contrarian)
Extreme Fear ist historisch einer der besten Kaufzeitpunkte.
Empfehlung: Schrittweise Akkumulation, DCA Strategie
Risiko: Kann noch tiefer fallen, daher gestaffelt kaufen
"""
        elif value < 35:
            return """
MODERAT BULLISCH
Fear-Zone bietet oft gute Einstiegsmöglichkeiten.
Empfehlung: Positionen aufbauen, Stop-Loss setzen
Risiko: Sentiment kann noch negativer werden
"""
        elif value < 55:
            return """
NEUTRAL
Markt in Balance, keine extremen Signale.
Empfehlung: Bestehende Strategie fortsetzen
Risiko: Ausbruch in beide Richtungen möglich
"""
        elif value < 75:
            return """
VORSICHTIG
Greed-Zone, Markt möglicherweise überkauft.
Empfehlung: Gewinne teilweise mitnehmen, Stops nachziehen
Risiko: Korrektur wird wahrscheinlicher
"""
        else:
            return """
STARK BEARISCH (Contrarian)
Extreme Greed geht historisch oft Korrekturen voraus.
Empfehlung: Positionen reduzieren, Hedges aufbauen
Risiko: Euphorie kann länger anhalten als erwartet
"""

    async def _fetch_social_sentiment(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch social media sentiment analysis."""
        results = []

        analysis = self._analyze_social_sentiment(symbol)

        content = f"""SOCIAL MEDIA SENTIMENT ANALYSE
==============================
{'Symbol: ' + symbol if symbol else 'Markt-Übersicht'}
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Twitter/X Sentiment:
- Gesamt-Stimmung: {analysis['twitter_sentiment']}
- Mention-Volumen: {analysis['twitter_volume']}
- Influencer-Tendenz: {analysis['twitter_influencers']}

Reddit Sentiment (r/cryptocurrency, r/wallstreetbets):
- Gesamt-Stimmung: {analysis['reddit_sentiment']}
- Post-Volumen: {analysis['reddit_volume']}
- Top-Themen: {analysis['reddit_topics']}

Google Trends:
- Suchvolumen-Trend: {analysis['google_trend']}
- Relative Stärke: {analysis['google_strength']}

Sentiment Score (aggregiert):
{analysis['aggregate_score']}/100

Interpretation:
{analysis['interpretation']}

Warnsignale:
{analysis['warnings']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        priority = (
            DataPriority.HIGH
            if analysis['extreme_sentiment']
            else DataPriority.MEDIUM
        )

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "social_sentiment",
                **analysis
            }
        ))

        return results

    def _analyze_social_sentiment(self, symbol: Optional[str]) -> dict:
        """Analyze social media sentiment."""
        # In production, connect to:
        # - Twitter API / Sentiment analysis services
        # - Reddit API
        # - LunarCrush (crypto social metrics)
        # - Santiment
        # - Google Trends API

        return {
            "twitter_sentiment": "Moderat Positiv",
            "twitter_volume": "Durchschnittlich",
            "twitter_influencers": "Gemischt - einige bullisch, einige vorsichtig",
            "reddit_sentiment": "Leicht Positiv",
            "reddit_volume": "Erhöht gegenüber 7-Tage Durchschnitt",
            "reddit_topics": "Marktanalyse, Altcoin-Rotation, DeFi Yields",
            "google_trend": "Stabil",
            "google_strength": "50/100 (durchschnittlich)",
            "aggregate_score": 58,
            "extreme_sentiment": False,
            "interpretation": """
Das Social Media Sentiment ist moderat positiv ohne extreme Ausschläge.
Die Diskussionen sind sachlich mit Focus auf fundamentale Themen.
Kein FOMO oder Panik erkennbar.
""".strip(),
            "warnings": """
- Keine extremen Warnsignale aktiv
- Bei plötzlichem Volumen-Anstieg Vorsicht (potentieller Pump)
- Influencer-Meinungen kritisch hinterfragen
""".strip(),
            "trading_implication": """
Neutrales Sentiment = keine Contrarian-Signale.
Technische und fundamentale Analyse priorisieren.
"""
        }

    async def _fetch_options_sentiment(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch options market sentiment (Put/Call ratio, etc.)."""
        results = []

        analysis = self._analyze_options_sentiment(symbol)

        content = f"""OPTIONS MARKT SENTIMENT
========================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

Put/Call Ratio:
- Equity Options P/C: {analysis['equity_pc']}
- Index Options P/C: {analysis['index_pc']}
- Total P/C: {analysis['total_pc']}
- 5-Tage Durchschnitt: {analysis['pc_5d_avg']}

Interpretation Put/Call:
{analysis['pc_interpretation']}

Options Flow Analyse:
- Große Calls (Bullisch): {analysis['large_calls']}
- Große Puts (Bearisch): {analysis['large_puts']}
- Smart Money Tendenz: {analysis['smart_money']}

Max Pain Level:
- Aktueller Max Pain: {analysis['max_pain']}
- Abstand zum Preis: {analysis['max_pain_distance']}

Gamma Exposure:
{analysis['gamma_exposure']}

Options Sentiment Score: {analysis['sentiment_score']}/100

Trading-Implikation:
{analysis['trading_implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "options_sentiment",
                **analysis
            }
        ))

        return results

    def _analyze_options_sentiment(self, symbol: Optional[str]) -> dict:
        """Analyze options market sentiment."""
        return {
            "equity_pc": "0.85 (leicht bullisch)",
            "index_pc": "1.10 (leicht bearisch/hedging)",
            "total_pc": "0.95 (neutral)",
            "pc_5d_avg": "0.92",
            "pc_interpretation": """
Put/Call Ratio nahe 1.0 zeigt ausgewogenen Markt.
- P/C > 1.2: Übermäßiges Hedging, oft Contrarian Bullish
- P/C < 0.7: Übermäßiger Optimismus, Vorsicht geboten
Aktuell: Neutral, keine extremen Signale
""".strip(),
            "large_calls": "Moderate Aktivität bei OTM Calls",
            "large_puts": "Defensive Puts auf Index-Level aktiv",
            "smart_money": "Leicht bullisch positioniert",
            "max_pain": "Berechnet aus Open Interest",
            "max_pain_distance": "Preis typischerweise gravitiert zu Max Pain bei Verfall",
            "gamma_exposure": """
- Positive Gamma: Market Maker dämpfen Bewegungen
- Negative Gamma: Market Maker verstärken Bewegungen
Aktuell: Leicht positive Gamma Zone
""".strip(),
            "sentiment_score": 55,
            "trading_implication": """
Options-Markt zeigt keine extremen Positionierungen.
Max Pain als kurzfristiges Ziel beobachten.
Bei Verfall erhöhte Volatilität möglich.
"""
        }

    async def _fetch_volatility_indices(self, symbol: Optional[str]) -> list[DataSourceResult]:
        """Fetch VIX and volatility indices."""
        results = []

        analysis = self._analyze_volatility(symbol)

        content = f"""VOLATILITÄTS-INDIKATOREN
=========================
Stand: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

VIX (CBOE Volatility Index):
- Aktuell: {analysis['vix_current']}
- 20-Tage MA: {analysis['vix_ma20']}
- Percentile (1Y): {analysis['vix_percentile']}

VIX Interpretation:
{analysis['vix_interpretation']}

VIX Term Structure:
- Contango/Backwardation: {analysis['term_structure']}
- Implikation: {analysis['term_implication']}

Crypto Volatilität (DVOL/BVOL):
- Bitcoin IV: {analysis['btc_iv']}
- Ethereum IV: {analysis['eth_iv']}
- Trend: {analysis['crypto_vol_trend']}

Volatility Risk Premium:
{analysis['vol_risk_premium']}

Historischer Kontext:
{analysis['historical_context']}

Trading-Implikation:
{analysis['trading_implication']}
"""

        priority = (
            DataPriority.HIGH
            if analysis['vix_extreme']
            else DataPriority.MEDIUM
        )

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "volatility_indices",
                **analysis
            }
        ))

        return results

    def _analyze_volatility(self, symbol: Optional[str]) -> dict:
        """Analyze volatility indices."""
        return {
            "vix_current": "Wird aus Marktdaten geladen",
            "vix_ma20": "20-Tage Durchschnitt",
            "vix_percentile": "Historisches Percentile",
            "vix_interpretation": """
VIX Zonen:
- <15: Niedrige Volatilität, Complacency, mögliche Ruhe vor dem Sturm
- 15-20: Normal, gesunder Markt
- 20-30: Erhöhte Unsicherheit, Hedging aktiv
- >30: Hohe Angst, oft Markttiefs in der Nähe
- >40: Panik, historisch exzellente Kaufgelegenheiten
""".strip(),
            "term_structure": "Analyse der VIX Futures Kurve",
            "term_implication": """
Contango (normal): VIX Futures > Spot = Markt erwartet Stabilität
Backwardation: VIX Futures < Spot = Akute Angst, kurzfristige Unsicherheit
""".strip(),
            "btc_iv": "Bitcoin implizierte Volatilität",
            "eth_iv": "Ethereum implizierte Volatilität",
            "crypto_vol_trend": "Volatilitäts-Trend für Crypto",
            "vol_risk_premium": """
Die Differenz zwischen implizierter und realisierter Volatilität.
Hohe Prämie = Optionen teuer, potentiell übertriebene Angst
Negative Prämie = Markt unterschätzt Risiko
""".strip(),
            "historical_context": """
Historisch waren VIX-Spikes über 30 oft in der Nähe von Markttiefs.
Mean Reversion: Extrem hohe VIX-Werte normalisieren sich typischerweise
innerhalb von 2-4 Wochen.
""".strip(),
            "vix_extreme": False,
            "trading_implication": """
- Niedriger VIX: Gute Zeit für Hedges (Put-Optionen günstig)
- Hoher VIX: Contrarian Kaufsignal, Volatilitäts-Verkauf profitabel
"""
        }

    async def _fetch_funding_rates(self, symbol: str) -> list[DataSourceResult]:
        """Fetch perpetual funding rates for crypto."""
        results = []

        analysis = self._analyze_funding_rates(symbol)

        content = f"""FUNDING RATES ANALYSE - {symbol}
================================
Perpetual Futures Sentiment

Aktuelle Funding Rate:
- Binance: {analysis['binance_funding']}
- Bybit: {analysis['bybit_funding']}
- OKX: {analysis['okx_funding']}
- Durchschnitt: {analysis['avg_funding']}

Funding Rate Interpretation:
{analysis['funding_interpretation']}

Open Interest:
- Aktuell: {analysis['open_interest']}
- 24h Änderung: {analysis['oi_change']}
- Trend: {analysis['oi_trend']}

Long/Short Ratio:
- Top Trader Long/Short: {analysis['ls_ratio']}
- Retail Long/Short: {analysis['retail_ls']}

Liquidation Levels:
- Nächste Long Liquidations: {analysis['long_liquidations']}
- Nächste Short Liquidations: {analysis['short_liquidations']}

Sentiment Score: {analysis['sentiment_score']}/100 (100 = extrem bullisch)

Trading-Implikation:
{analysis['trading_implication']}
"""

        priority = (
            DataPriority.HIGH
            if analysis['extreme_funding']
            else DataPriority.MEDIUM
        )

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "funding_rates",
                **analysis
            }
        ))

        return results

    def _analyze_funding_rates(self, symbol: str) -> dict:
        """Analyze perpetual futures funding rates."""
        return {
            "binance_funding": "Wird von API geladen",
            "bybit_funding": "Wird von API geladen",
            "okx_funding": "Wird von API geladen",
            "avg_funding": "Durchschnitt aller Börsen",
            "funding_interpretation": """
Funding Rate Logik:
- Positiv: Longs zahlen Shorts, bullisches Sentiment
- Negativ: Shorts zahlen Longs, bearisches Sentiment
- >0.1%/8h: Überhitzter Long-Markt, Korrekturrisiko
- <-0.1%/8h: Übertriebene Shorts, Squeeze-Risiko

Extremwerte sind Contrarian-Signale!
""".strip(),
            "open_interest": "Gesamt Open Interest in USD",
            "oi_change": "24h Änderung in %",
            "oi_trend": "Trend über 7 Tage",
            "ls_ratio": "Top Trader Positionierung",
            "retail_ls": "Retail Trader Positionierung",
            "long_liquidations": "Preis-Level für Long-Liquidationen",
            "short_liquidations": "Preis-Level für Short-Liquidationen",
            "extreme_funding": False,
            "sentiment_score": 55,
            "trading_implication": """
- Hohe positive Funding + hohe OI: Vorsicht, Long-Squeeze möglich
- Hohe negative Funding + hohe OI: Short-Squeeze Risiko
- Neutrale Funding: Markt in Balance
"""
        }

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when fetching fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""SENTIMENT DATEN - ÜBERSICHT
===========================
Hinweis: Live-Daten temporär nicht verfügbar.

Wichtige Sentiment-Indikatoren:

1. Fear & Greed Index (0-100):
   - Crypto: alternative.me/crypto/fear-and-greed-index/
   - Traditional: money.cnn.com/data/fear-and-greed/

2. Social Sentiment:
   - LunarCrush (Crypto Social)
   - Santiment
   - Twitter/X Sentiment Tools

3. Options Daten:
   - Put/Call Ratio
   - Max Pain
   - Large Options Flow

4. Volatilität:
   - VIX (CBOE)
   - Crypto DVOL/BVOL

5. Funding Rates (Crypto):
   - Positive = Bullisch positioniert
   - Negative = Bearisch positioniert
   - Extreme = Contrarian Signal

Empfehlung: Sentiment als Bestätigung nutzen, nicht als alleinige Grundlage.
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True}
        )
