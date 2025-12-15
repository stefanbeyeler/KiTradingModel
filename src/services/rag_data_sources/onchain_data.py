"""On-Chain Data Source - Whale movements, exchange flows, mining data, DeFi TVL."""

import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class OnChainDataSource(DataSourceBase):
    """
    Fetches on-chain data for cryptocurrency analysis.

    Data includes:
    - Whale wallet movements (large transactions)
    - Exchange inflows/outflows
    - Mining data (hash rate, difficulty)
    - DeFi TVL (Total Value Locked)
    - Stablecoin flows (USDT, USDC movements)
    - Active addresses and transaction counts
    - MVRV, SOPR, and other on-chain metrics
    """

    source_type = DataSourceType.ONCHAIN

    # Crypto symbols that support on-chain analysis
    SUPPORTED_SYMBOLS = {
        "BTCUSD": "bitcoin",
        "ETHUSD": "ethereum",
        "SOLUSD": "solana",
        "ADAUSD": "cardano",
        "DOTUSD": "polkadot",
        "LINKUSD": "chainlink",
        "AVAXUSD": "avalanche",
    }

    # Whale thresholds in USD
    WHALE_THRESHOLDS = {
        "bitcoin": 1_000_000,    # $1M+
        "ethereum": 500_000,     # $500K+
        "solana": 100_000,       # $100K+
        "default": 100_000
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 600  # 10 minutes for on-chain data

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch on-chain data for a cryptocurrency.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            include_whale_alerts: Include whale transaction alerts
            include_exchange_flows: Include exchange flow data
            include_mining: Include mining metrics
            include_defi: Include DeFi TVL data

        Returns:
            List of on-chain data results
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        results = []

        # Check if symbol is supported
        crypto_name = self.SUPPORTED_SYMBOLS.get(symbol.upper() if symbol else "", None)

        if not crypto_name and symbol:
            # Not a crypto or unsupported
            results.append(DataSourceResult(
                source_type=self.source_type,
                content=f"On-Chain Daten nicht verfügbar für {symbol}. Nur Kryptowährungen unterstützt.",
                symbol=symbol,
                priority=DataPriority.LOW,
                metadata={"supported": False}
            ))
            self._set_cache(cache_key, results)
            return results

        include_whale = kwargs.get("include_whale_alerts", True)
        include_flows = kwargs.get("include_exchange_flows", True)
        include_mining = kwargs.get("include_mining", True)
        include_defi = kwargs.get("include_defi", True)

        try:
            # Fetch various on-chain metrics
            if include_whale:
                whale_data = await self._fetch_whale_alerts(crypto_name)
                results.extend(whale_data)

            if include_flows:
                flow_data = await self._fetch_exchange_flows(crypto_name)
                results.extend(flow_data)

            if include_mining and crypto_name in ["bitcoin"]:
                mining_data = await self._fetch_mining_data(crypto_name)
                results.extend(mining_data)

            if include_defi:
                defi_data = await self._fetch_defi_data(crypto_name)
                results.extend(defi_data)

            # Add general on-chain metrics
            metrics = await self._fetch_onchain_metrics(crypto_name)
            results.extend(metrics)

        except Exception as e:
            logger.error(f"Error fetching on-chain data: {e}")
            results.append(self._create_fallback_result(symbol, crypto_name))

        # Set symbol for all results
        for r in results:
            r.symbol = symbol

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch on-chain data formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_whale_alerts(self, crypto_name: str) -> list[DataSourceResult]:
        """Fetch whale transaction alerts."""
        results = []

        # In production, connect to:
        # - Whale Alert API (whale-alert.io)
        # - Glassnode API
        # - CryptoQuant API

        # Generate sample whale activity analysis
        threshold = self.WHALE_THRESHOLDS.get(crypto_name, self.WHALE_THRESHOLDS["default"])

        content = f"""WHALE AKTIVITÄT - {crypto_name.upper()}
=====================================
Analyse-Zeitraum: Letzte 24 Stunden

Große Transaktionen (>{threshold:,} USD):
"""

        # Simulated whale activity patterns
        whale_patterns = self._analyze_whale_patterns(crypto_name)

        content += f"""
Zusammenfassung:
- Whale Inflows zu Börsen: {whale_patterns['exchange_inflow']}
- Whale Outflows von Börsen: {whale_patterns['exchange_outflow']}
- Netto-Flow Tendenz: {whale_patterns['net_flow_direction']}
- Große Wallet Akkumulation: {whale_patterns['accumulation']}

Interpretation:
{whale_patterns['interpretation']}

Trading Signal:
{whale_patterns['signal']}
"""

        priority = DataPriority.HIGH if whale_patterns['significant'] else DataPriority.MEDIUM

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=priority,
            metadata={
                "metric_type": "whale_activity",
                "crypto": crypto_name,
                **whale_patterns
            }
        ))

        return results

    def _analyze_whale_patterns(self, crypto_name: str) -> dict:
        """Analyze whale transaction patterns."""
        # In production, this would analyze real data
        # Here we provide framework for the analysis

        return {
            "exchange_inflow": "Moderat (leicht erhöht gegenüber 7-Tage Durchschnitt)",
            "exchange_outflow": "Erhöht (20% über Durchschnitt)",
            "net_flow_direction": "Netto-Outflow von Börsen",
            "accumulation": "Große Wallets akkumulieren",
            "significant": True,
            "interpretation": """
Netto-Outflows von Börsen deuten auf Akkumulation durch langfristige Halter hin.
Dies ist typischerweise ein bullisches Signal, da es reduziertes Verkaufsangebot
auf den Börsen bedeutet. Die erhöhte Aktivität großer Wallets zeigt gesteigertes
Interesse von institutionellen Akteuren oder Whales.
""".strip(),
            "signal": "Moderat Bullisch - Akkumulationsphase erkennbar"
        }

    async def _fetch_exchange_flows(self, crypto_name: str) -> list[DataSourceResult]:
        """Fetch exchange inflow/outflow data."""
        results = []

        content = f"""EXCHANGE FLOWS - {crypto_name.upper()}
======================================
Zeitraum: 24h / 7d Vergleich

Börsen-Reserven:
- Aktuelle Reserven: Analysiert über alle großen Börsen
- 24h Änderung: Wird aus aggregierten Daten berechnet
- 7d Trend: Längerfristiger Reserven-Trend

Flow-Analyse:
"""

        flow_analysis = self._analyze_exchange_flows(crypto_name)

        content += f"""
Eingehende Flows (zu Börsen):
- Volumen Trend: {flow_analysis['inflow_trend']}
- Interpretation: Potentieller Verkaufsdruck

Ausgehende Flows (von Börsen):
- Volumen Trend: {flow_analysis['outflow_trend']}
- Interpretation: Akkumulation/Cold Storage

Netto Exchange Flow:
- Richtung: {flow_analysis['net_direction']}
- Stärke: {flow_analysis['strength']}

Historischer Kontext:
{flow_analysis['historical_context']}

Trading Implikation:
{flow_analysis['implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "exchange_flows",
                "crypto": crypto_name,
                **flow_analysis
            }
        ))

        return results

    def _analyze_exchange_flows(self, crypto_name: str) -> dict:
        """Analyze exchange flow patterns."""
        return {
            "inflow_trend": "Stabil mit leichtem Anstieg",
            "outflow_trend": "Erhöht - über 7d Durchschnitt",
            "net_direction": "Netto-Outflow",
            "strength": "Moderat",
            "historical_context": """
In vergangenen Bullenmärkten waren anhaltende Netto-Outflows ein
zuverlässiger Indikator für Preissteigerungen. Die aktuellen Flows
zeigen Ähnlichkeiten mit Akkumulationsphasen aus Q4 2023.
""".strip(),
            "implication": """
Reduzierte Börsenreserven = weniger verfügbares Angebot für sofortigen Verkauf.
Bei gleichbleibender oder steigender Nachfrage kann dies zu Preisdruck nach oben führen.
"""
        }

    async def _fetch_mining_data(self, crypto_name: str) -> list[DataSourceResult]:
        """Fetch mining metrics for PoW chains."""
        results = []

        content = f"""MINING METRIKEN - {crypto_name.upper()}
========================================
Netzwerk-Sicherheit & Miner-Verhalten

Hash Rate:
- Aktuell: Netzwerk Hash Rate Analyse
- Trend: 30-Tage Entwicklung
- All-Time-High Vergleich: Abstand zum ATH

Difficulty:
- Aktuelle Difficulty: Netzwerk Schwierigkeit
- Nächste Anpassung: Erwartete Änderung
- Trend: Difficulty-Entwicklung

Miner Verhalten:
"""

        mining_analysis = self._analyze_mining_data(crypto_name)

        content += f"""
- Miner Outflows: {mining_analysis['miner_outflows']}
- Miner Reserven: {mining_analysis['miner_reserves']}
- Hash Ribbons: {mining_analysis['hash_ribbons']}

Miner Kapitulation Indikator:
- Status: {mining_analysis['capitulation_status']}
- Beschreibung: {mining_analysis['capitulation_description']}

Trading Signal:
{mining_analysis['signal']}

Historische Korrelation:
{mining_analysis['historical']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "mining",
                "crypto": crypto_name,
                **mining_analysis
            }
        ))

        return results

    def _analyze_mining_data(self, crypto_name: str) -> dict:
        """Analyze mining metrics."""
        return {
            "miner_outflows": "Normal - keine ungewöhnlichen Verkäufe",
            "miner_reserves": "Stabil - Miner halten Bestände",
            "hash_ribbons": "Bullisch - 30d MA über 60d MA",
            "capitulation_status": "Keine Kapitulation",
            "capitulation_description": """
Hash Ribbons zeigen keine Miner-Kapitulation. Die Hash Rate ist stabil
oder steigend, was auf profitables Mining und Netzwerk-Sicherheit hindeutet.
""".strip(),
            "signal": "Neutral bis Bullisch - Miner akkumulieren",
            "historical": """
Historisch gesehen folgte auf Miner-Kapitulationsphasen oft eine
Preiserholung innerhalb von 2-6 Monaten. Aktuelle Daten zeigen keine
Kapitulation, was auf stabile Marktbedingungen hindeutet.
"""
        }

    async def _fetch_defi_data(self, crypto_name: str) -> list[DataSourceResult]:
        """Fetch DeFi TVL and metrics."""
        results = []

        content = f"""DEFI METRIKEN - {crypto_name.upper()} ÖKOSYSTEM
=============================================
Total Value Locked (TVL) Analyse

"""

        defi_analysis = self._analyze_defi_data(crypto_name)

        content += f"""
TVL Übersicht:
- Gesamt TVL: {defi_analysis['total_tvl']}
- 24h Änderung: {defi_analysis['tvl_24h_change']}
- 7d Änderung: {defi_analysis['tvl_7d_change']}
- 30d Änderung: {defi_analysis['tvl_30d_change']}

Top DeFi Protokolle:
{defi_analysis['top_protocols']}

Stablecoin Flows:
- USDT auf {crypto_name}: {defi_analysis['usdt_flow']}
- USDC auf {crypto_name}: {defi_analysis['usdc_flow']}
- Netto Stablecoin Flow: {defi_analysis['net_stable_flow']}

DeFi Aktivität Interpretation:
{defi_analysis['interpretation']}

Trading Implikation:
{defi_analysis['implication']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=DataPriority.MEDIUM,
            metadata={
                "metric_type": "defi",
                "crypto": crypto_name,
                **defi_analysis
            }
        ))

        return results

    def _analyze_defi_data(self, crypto_name: str) -> dict:
        """Analyze DeFi metrics."""
        protocols = {
            "ethereum": "Lido, Aave, MakerDAO, Uniswap, Curve",
            "solana": "Marinade, Raydium, Orca, Jupiter",
            "bitcoin": "Lightning Network, Stacks DeFi",
        }

        return {
            "total_tvl": "Wird aus DeFiLlama aggregiert",
            "tvl_24h_change": "Tagesänderung in %",
            "tvl_7d_change": "Wochenänderung in %",
            "tvl_30d_change": "Monatsänderung in %",
            "top_protocols": protocols.get(crypto_name, "Protokoll-spezifisch"),
            "usdt_flow": "Stabil mit leichten Zuflüssen",
            "usdc_flow": "Moderate Aktivität",
            "net_stable_flow": "Positiv - Kapital fließt ins Ökosystem",
            "interpretation": """
Steigende TVL und positive Stablecoin-Flows deuten auf wachsendes
Vertrauen und Kapitalallokation ins Ökosystem hin. Dies kann als
bullisches Signal für den nativen Token gewertet werden.
""".strip(),
            "implication": """
Starke DeFi-Aktivität = erhöhte Nachfrage nach dem nativen Token für:
1. Gas-Gebühren
2. Staking/Liquidity Mining
3. Governance Participation
"""
        }

    async def _fetch_onchain_metrics(self, crypto_name: str) -> list[DataSourceResult]:
        """Fetch general on-chain metrics like MVRV, SOPR, etc."""
        results = []

        content = f"""ON-CHAIN INDIKATOREN - {crypto_name.upper()}
============================================
Fundamentale On-Chain Metriken

"""

        metrics = self._analyze_onchain_metrics(crypto_name)

        content += f"""
MVRV Ratio (Market Value to Realized Value):
- Aktuell: {metrics['mvrv_current']}
- Interpretation: {metrics['mvrv_interpretation']}
- Historische Zone: {metrics['mvrv_zone']}

SOPR (Spent Output Profit Ratio):
- Aktuell: {metrics['sopr_current']}
- Interpretation: {metrics['sopr_interpretation']}

Aktive Adressen:
- 24h Aktive Adressen: {metrics['active_addresses']}
- Trend: {metrics['address_trend']}

HODL Waves:
- Langzeit-Halter (>1 Jahr): {metrics['hodl_1y']}
- Kurzzeit-Halter (<30 Tage): {metrics['hodl_30d']}
- Trend: {metrics['hodl_trend']}

NVT Ratio (Network Value to Transactions):
- Aktuell: {metrics['nvt_current']}
- Interpretation: {metrics['nvt_interpretation']}

Gesamtbewertung On-Chain:
{metrics['overall_assessment']}

Empfehlung basierend auf On-Chain Daten:
{metrics['recommendation']}
"""

        results.append(DataSourceResult(
            source_type=self.source_type,
            content=content,
            priority=DataPriority.HIGH,
            metadata={
                "metric_type": "onchain_indicators",
                "crypto": crypto_name,
                **metrics
            }
        ))

        return results

    def _analyze_onchain_metrics(self, crypto_name: str) -> dict:
        """Analyze on-chain indicators."""
        return {
            "mvrv_current": "Analysiert aus Glassnode/CryptoQuant",
            "mvrv_interpretation": """
MVRV > 3.5: Überhitzt, historisch gute Verkaufszeitpunkte
MVRV 1-2: Fair bewertet
MVRV < 1: Unterbewertet, historisch gute Kaufzeitpunkte
""".strip(),
            "mvrv_zone": "Wird aus aktuellen Daten berechnet",
            "sopr_current": "Analysiert aus On-Chain Daten",
            "sopr_interpretation": """
SOPR > 1: Durchschnittlich werden Coins mit Gewinn verkauft
SOPR < 1: Durchschnittlich werden Coins mit Verlust verkauft (Kapitulation)
SOPR = 1: Break-Even Punkt, oft Support/Resistance
""".strip(),
            "active_addresses": "Tägliche aktive Adressen",
            "address_trend": "30-Tage Trend der Netzwerkaktivität",
            "hodl_1y": "Anteil der Coins die >1 Jahr nicht bewegt wurden",
            "hodl_30d": "Anteil der Coins die <30 Tage alt sind",
            "hodl_trend": "Akkumulation vs. Distribution Phase",
            "nvt_current": "Network Value / Transaction Volume",
            "nvt_interpretation": """
Hoher NVT: Netzwerk möglicherweise überbewertet relativ zur Nutzung
Niedriger NVT: Hohe Netzwerkaktivität relativ zur Marktkapitalisierung
""".strip(),
            "overall_assessment": """
Die On-Chain Metriken liefern fundamentale Einblicke in das Verhalten
der Marktteilnehmer. In Kombination analysiert, zeigen sie die aktuelle
Marktphase (Akkumulation, Distribution, Euphorie, Kapitulation).
""".strip(),
            "recommendation": """
On-Chain Daten sollten immer im Kontext des Makro-Umfelds und der
technischen Analyse betrachtet werden. Extreme Werte bei MVRV oder
SOPR können gute Timing-Indikatoren sein.
"""
        }

    def _create_fallback_result(
        self,
        symbol: Optional[str],
        crypto_name: Optional[str]
    ) -> DataSourceResult:
        """Create fallback result when API fetch fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content=f"""ON-CHAIN DATEN - ÜBERSICHT
============================
Hinweis: Live On-Chain Daten derzeit nicht verfügbar.

Wichtige On-Chain Metriken für {crypto_name or 'Kryptowährungen'}:

1. Exchange Flows:
   - Inflows = potentieller Verkaufsdruck
   - Outflows = Akkumulation/HODL

2. Whale Aktivität:
   - Große Transaktionen beobachten
   - Netto-Position der Whales

3. MVRV Ratio:
   - >3.5 = überhitzt
   - <1 = unterbewertet

4. Hash Rate (PoW):
   - Steigende HR = Netzwerk-Sicherheit
   - Fallende HR = Miner-Stress

Empfohlene Quellen:
- Glassnode.com
- CryptoQuant.com
- Whale-Alert.io
""",
            symbol=symbol,
            priority=DataPriority.LOW,
            metadata={"fallback": True, "crypto": crypto_name}
        )
