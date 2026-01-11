"""
Deep Analysis Service.

Vertiefte Analyse mit RAG + LLM Integration.
"""

import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger

from ..config import settings
from ..models.schemas import (
    TradingSetup,
    DeepAnalysisRequest,
    DeepAnalysisResponse,
    EntryExitLevels,
)
from .signal_aggregator import signal_aggregator
from .scoring_service import scoring_service


class DeepAnalysisService:
    """Führt vertiefte Analysen mit RAG und LLM durch."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialisiert den HTTP-Client mit längeren Timeouts."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.deep_analysis_timeout_seconds)
            )
        return self._client

    async def close(self):
        """Schliesst den HTTP-Client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def analyze(
        self,
        symbol: str,
        request: DeepAnalysisRequest,
        existing_setup: Optional[TradingSetup] = None,
    ) -> DeepAnalysisResponse:
        """
        Führt eine vertiefte Analyse durch.

        1. Holt/erstellt das Trading-Setup
        2. Fragt RAG für historischen Kontext
        3. Generiert LLM-Analyse

        Args:
            symbol: Trading-Symbol
            request: Analyse-Request
            existing_setup: Optional bereits vorhandenes Setup

        Returns:
            Vollständige DeepAnalysisResponse
        """
        start_time = time.time()

        # Setup holen oder erstellen
        if existing_setup:
            setup = existing_setup
        else:
            signals = await signal_aggregator.fetch_all_signals(
                symbol, request.timeframe
            )
            setup = scoring_service.create_setup(symbol, request.timeframe, signals)

        # Response initialisieren
        response = DeepAnalysisResponse(
            symbol=symbol,
            timeframe=request.timeframe,
            timestamp=datetime.now(timezone.utc),
            setup=setup,
        )

        client = await self._get_client()

        # RAG-Kontext holen
        if request.include_rag:
            rag_result = await self._fetch_rag_context(client, symbol, setup)
            if rag_result:
                response.similar_patterns = rag_result.get("similar_patterns")
                response.historical_context = rag_result.get("context")
                response.rag_sources_used = rag_result.get("sources_count", 0)

        # LLM-Analyse generieren
        if request.include_llm:
            llm_result = await self._generate_llm_analysis(client, symbol, setup, response)
            if llm_result:
                response.analysis_summary = llm_result.get("summary")
                response.risk_assessment = llm_result.get("risk_assessment")
                response.rationale = llm_result.get("rationale")
                response.llm_model = llm_result.get("model")

                # Entry/Exit Levels
                if llm_result.get("levels"):
                    response.entry_exit_levels = EntryExitLevels(**llm_result["levels"])

        response.analysis_duration_ms = (time.time() - start_time) * 1000

        return response

    async def _fetch_rag_context(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        setup: TradingSetup,
    ) -> Optional[dict]:
        """Holt historischen Kontext vom RAG Service."""
        try:
            # Query basierend auf Setup erstellen
            query = self._build_rag_query(symbol, setup)

            url = f"{settings.rag_service_url}/api/v1/rag/query"
            response = await client.post(
                url,
                json={
                    "query": query,
                    "symbol": symbol,
                    "limit": 5,
                    "include_metadata": True,
                }
            )

            if response.status_code != 200:
                logger.warning(f"RAG query failed: {response.status_code}")
                return None

            data = response.json()

            # Ergebnisse parsen
            results = data.get("results", [])
            context_parts = []
            similar_patterns = []

            for result in results:
                # Kontext-Text sammeln
                if result.get("content"):
                    context_parts.append(result["content"])

                # Pattern-Matches extrahieren
                if result.get("metadata", {}).get("pattern_type"):
                    similar_patterns.append({
                        "pattern": result["metadata"]["pattern_type"],
                        "date": result["metadata"].get("date"),
                        "outcome": result["metadata"].get("outcome"),
                        "similarity": result.get("score", 0),
                    })

            return {
                "similar_patterns": similar_patterns[:5] if similar_patterns else None,
                "context": "\n\n".join(context_parts[:3]) if context_parts else None,
                "sources_count": len(results),
            }

        except Exception as e:
            logger.warning(f"RAG fetch error: {e}")
            return None

    def _build_rag_query(self, symbol: str, setup: TradingSetup) -> str:
        """Erstellt eine optimierte RAG-Query basierend auf dem Setup."""
        parts = [
            f"Trading setup für {symbol}",
            f"Richtung: {setup.direction.value}",
        ]

        # Regime hinzufügen
        if setup.hmm_signal.available:
            parts.append(f"Regime: {setup.hmm_signal.regime.value}")

        # Patterns hinzufügen
        if setup.tcn_signal.patterns:
            parts.append(f"Patterns: {', '.join(setup.tcn_signal.patterns)}")

        if setup.candlestick_signal.patterns:
            parts.append(f"Candlestick: {', '.join(setup.candlestick_signal.patterns)}")

        # Key Drivers
        if setup.key_drivers:
            parts.append(f"Signale: {'; '.join(setup.key_drivers)}")

        return " | ".join(parts)

    async def _generate_llm_analysis(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        setup: TradingSetup,
        context: DeepAnalysisResponse,
    ) -> Optional[dict]:
        """Generiert LLM-Analyse basierend auf Setup und RAG-Kontext."""
        try:
            # Prompt für LLM erstellen
            prompt = self._build_llm_prompt(symbol, setup, context)

            url = f"{settings.llm_service_url}/api/v1/analyze"
            response = await client.post(
                url,
                json={
                    "symbol": symbol,
                    "timeframe": setup.timeframe,
                    "prompt": prompt,
                    "context": {
                        "direction": setup.direction.value,
                        "composite_score": setup.composite_score,
                        "confidence_level": setup.confidence_level.value,
                        "key_drivers": setup.key_drivers,
                        "regime": setup.hmm_signal.regime.value if setup.hmm_signal.available else None,
                        "historical_context": context.historical_context,
                    },
                    "include_levels": True,
                    "include_risk_assessment": True,
                }
            )

            if response.status_code != 200:
                logger.warning(f"LLM analyze failed: {response.status_code}")
                return None

            data = response.json()

            return {
                "summary": data.get("summary") or data.get("analysis"),
                "risk_assessment": data.get("risk_assessment"),
                "rationale": data.get("rationale"),
                "model": data.get("model", "unknown"),
                "levels": self._extract_levels(data),
            }

        except Exception as e:
            logger.warning(f"LLM analysis error: {e}")
            return None

    def _build_llm_prompt(
        self,
        symbol: str,
        setup: TradingSetup,
        context: DeepAnalysisResponse,
    ) -> str:
        """Erstellt den LLM-Prompt für die Analyse."""
        prompt_parts = [
            f"Analysiere das Trading-Setup für {symbol} ({setup.timeframe}).",
            "",
            f"## Signal-Score: {setup.composite_score:.1f}/100 ({setup.confidence_level.value})",
            f"## Richtung: {setup.direction.value}",
            f"## Signal-Alignment: {setup.signal_alignment.value}",
            "",
            "## Aktive Signale:",
        ]

        # NHITS
        if setup.nhits_signal.available:
            prompt_parts.append(
                f"- NHITS: {setup.nhits_signal.direction.value} "
                f"(Trend-Prob: {setup.nhits_signal.trend_probability:.0%})"
            )

        # HMM
        if setup.hmm_signal.available:
            prompt_parts.append(
                f"- HMM Regime: {setup.hmm_signal.regime.value} "
                f"({setup.hmm_signal.regime_probability:.0%})"
            )

        # TCN
        if setup.tcn_signal.patterns:
            prompt_parts.append(
                f"- TCN Patterns: {', '.join(setup.tcn_signal.patterns)} "
                f"({setup.tcn_signal.pattern_confidence:.0%})"
            )

        # Candlestick
        if setup.candlestick_signal.patterns:
            prompt_parts.append(
                f"- Candlestick: {', '.join(setup.candlestick_signal.patterns)} "
                f"({setup.candlestick_signal.pattern_strength:.0%})"
            )

        # Technical
        if setup.technical_signal.available:
            prompt_parts.append(
                f"- RSI: {setup.technical_signal.rsi_signal}, "
                f"MACD: {setup.technical_signal.macd_signal}"
            )

        # Historischer Kontext
        if context.historical_context:
            prompt_parts.extend([
                "",
                "## Historischer Kontext:",
                context.historical_context[:500],  # Limit
            ])

        prompt_parts.extend([
            "",
            "## Aufgabe:",
            "1. Bewerte das Setup und seine Stärken/Schwächen",
            "2. Gib eine Risiko-Einschätzung (low/medium/high)",
            "3. Schlage Entry-, Stop-Loss- und Take-Profit-Level vor",
            "4. Begründe deine Empfehlung",
        ])

        return "\n".join(prompt_parts)

    def _extract_levels(self, llm_response: dict) -> Optional[dict]:
        """Extrahiert Entry/Exit Levels aus der LLM-Antwort."""
        levels = llm_response.get("levels") or llm_response.get("entry_exit_levels")
        if not levels:
            return None

        return {
            "entry_price": levels.get("entry") or levels.get("entry_price"),
            "stop_loss": levels.get("stop_loss") or levels.get("sl"),
            "take_profit_1": levels.get("take_profit_1") or levels.get("tp1"),
            "take_profit_2": levels.get("take_profit_2") or levels.get("tp2"),
            "take_profit_3": levels.get("take_profit_3") or levels.get("tp3"),
            "risk_reward_ratio": levels.get("risk_reward") or levels.get("rr_ratio"),
        }


# Singleton-Instanz
deep_analysis_service = DeepAnalysisService()
