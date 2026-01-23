"""
Signal Aggregator Service.

Holt Signale von allen ML-Services parallel und aggregiert sie.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from loguru import logger

from ..config import settings
from ..models.schemas import (
    NHITSSignal,
    HMMSignal,
    TCNSignal,
    CandlestickSignal,
    TechnicalSignal,
    SignalDirection,
    MarketRegime,
)


class SignalAggregatorService:
    """Aggregiert Signale von allen ML-Services."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialisiert den HTTP-Client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.http_timeout_seconds)
            )
        return self._client

    async def close(self):
        """Schliesst den HTTP-Client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def fetch_all_signals(
        self,
        symbol: str,
        timeframe: str = "H1"
    ) -> dict[str, Any]:
        """
        Holt Signale von allen ML-Services parallel.

        Args:
            symbol: Trading-Symbol (z.B. BTCUSD)
            timeframe: Timeframe (z.B. H1, D1)

        Returns:
            Dict mit allen Signal-Objekten
        """
        client = await self._get_client()

        # Alle Requests parallel starten
        tasks = {
            "nhits": self._fetch_nhits(client, symbol, timeframe),
            "hmm": self._fetch_hmm(client, symbol, timeframe),
            "tcn": self._fetch_tcn(client, symbol, timeframe),
            "candlestick": self._fetch_candlestick(client, symbol, timeframe),
            "technical": self._fetch_technical(client, symbol, timeframe),
        }

        results = {}
        gathered = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )

        for key, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                logger.warning(f"Fehler beim Abrufen von {key} für {symbol}: {result}")
                results[key] = None
            else:
                results[key] = result

        return results

    async def _fetch_nhits(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        timeframe: str
    ) -> NHITSSignal:
        """Holt NHITS Preis-Prognose."""
        try:
            url = f"{settings.nhits_service_url}/api/v1/forecast/{symbol}"
            response = await client.get(url, params={"timeframe": timeframe})

            if response.status_code != 200:
                return NHITSSignal(available=False)

            data = response.json()

            # Richtung bestimmen
            trend_up = data.get("trend_up_probability", 0.5)
            if trend_up > 0.6:
                direction = SignalDirection.LONG
            elif trend_up < 0.4:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            return NHITSSignal(
                available=True,
                trend_probability=trend_up,
                direction=direction,
                forecast_change_1h=data.get("predicted_change_percent_1h"),
                forecast_change_24h=data.get("predicted_change_percent_24h"),
                model_confidence=data.get("model_confidence"),
            )
        except Exception as e:
            logger.debug(f"NHITS fetch error for {symbol}: {e}")
            return NHITSSignal(available=False)

    async def _fetch_hmm(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        timeframe: str
    ) -> HMMSignal:
        """Holt HMM Regime-Detection."""
        try:
            url = f"{settings.hmm_service_url}/api/v1/regime/detect/{symbol}"
            response = await client.get(url, params={"timeframe": timeframe})

            if response.status_code != 200:
                return HMMSignal(available=False)

            data = response.json()

            # Regime parsen
            regime_str = data.get("current_regime", "sideways").lower().replace(" ", "_")
            try:
                regime = MarketRegime(regime_str)
            except ValueError:
                regime = MarketRegime.SIDEWAYS

            # Alignment bestimmen
            alignment = "neutral"
            if regime in [MarketRegime.BULL_TREND]:
                alignment = "aligned"  # Bullish
            elif regime in [MarketRegime.BEAR_TREND]:
                alignment = "contrary"  # Bearish (für Long-Setups)

            return HMMSignal(
                available=True,
                regime=regime,
                regime_probability=data.get("regime_probability", 0.5),
                signal_score=data.get("signal_score", 50.0),
                regime_duration=data.get("regime_duration"),
                alignment=alignment,
            )
        except Exception as e:
            logger.debug(f"HMM fetch error for {symbol}: {e}")
            return HMMSignal(available=False)

    async def _fetch_tcn(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        timeframe: str
    ) -> TCNSignal:
        """Holt TCN Chart-Pattern Detection."""
        try:
            url = f"{settings.tcn_service_url}/api/v1/detect/{symbol}"
            response = await client.get(url, params={"timeframe": timeframe})

            if response.status_code != 200:
                return TCNSignal(available=False)

            data = response.json()
            patterns_data = data.get("patterns", [])

            if not patterns_data:
                return TCNSignal(available=True, patterns=[], pattern_confidence=0.0)

            # Patterns extrahieren
            patterns = [p.get("pattern_type", "unknown") for p in patterns_data]
            max_confidence = max(p.get("confidence", 0) for p in patterns_data) if patterns_data else 0

            # Richtung aus dem stärksten Pattern
            top_pattern = max(patterns_data, key=lambda x: x.get("confidence", 0)) if patterns_data else {}
            direction_str = top_pattern.get("direction", "neutral").lower()

            if direction_str == "bullish":
                direction = SignalDirection.LONG
            elif direction_str == "bearish":
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            return TCNSignal(
                available=True,
                patterns=patterns[:3],  # Top 3 Patterns
                pattern_confidence=max_confidence,
                direction=direction,
                price_target=top_pattern.get("price_target"),
            )
        except Exception as e:
            logger.debug(f"TCN fetch error for {symbol}: {e}")
            return TCNSignal(available=False)

    async def _fetch_candlestick(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        timeframe: str
    ) -> CandlestickSignal:
        """Holt Candlestick Pattern Detection."""
        try:
            url = f"{settings.candlestick_service_url}/api/v1/scan"
            response = await client.post(
                url,
                json={"symbols": [symbol], "timeframe": timeframe}
            )

            if response.status_code != 200:
                return CandlestickSignal(available=False)

            data = response.json()
            results = data.get("results", [])

            if not results:
                return CandlestickSignal(available=True, patterns=[], pattern_strength=0.0)

            # Erstes Resultat (unser Symbol)
            symbol_result = results[0] if results else {}
            patterns_data = symbol_result.get("patterns", [])

            if not patterns_data:
                return CandlestickSignal(available=True, patterns=[], pattern_strength=0.0)

            patterns = [p.get("pattern_type", "unknown") for p in patterns_data]
            max_strength = max(p.get("confidence", 0) for p in patterns_data) if patterns_data else 0

            # Richtung bestimmen
            bullish_count = sum(1 for p in patterns_data if p.get("direction", "").lower() == "bullish")
            bearish_count = sum(1 for p in patterns_data if p.get("direction", "").lower() == "bearish")

            if bullish_count > bearish_count:
                direction = SignalDirection.LONG
            elif bearish_count > bullish_count:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            return CandlestickSignal(
                available=True,
                patterns=patterns[:3],
                pattern_strength=max_strength,
                direction=direction,
            )
        except Exception as e:
            logger.debug(f"Candlestick fetch error for {symbol}: {e}")
            return CandlestickSignal(available=False)

    async def _fetch_technical(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        timeframe: str
    ) -> TechnicalSignal:
        """Holt technische Indikatoren vom Data Service."""
        try:
            url = f"{settings.data_service_url}/api/v1/db/market-snapshot/{symbol}"
            response = await client.get(url, params={"timeframe": timeframe})

            if response.status_code != 200:
                return TechnicalSignal(available=False)

            data = response.json()
            indicators = data.get("indicators", {})

            # RSI
            rsi = indicators.get("rsi", {}).get("value")
            rsi_signal = "neutral"
            if rsi is not None:
                if rsi < 30:
                    rsi_signal = "oversold"
                elif rsi > 70:
                    rsi_signal = "overbought"

            # MACD
            macd = indicators.get("macd", {})
            macd_main = macd.get("main", 0)
            macd_signal_val = macd.get("signal", 0)
            macd_signal = "neutral"
            if macd_main > macd_signal_val:
                macd_signal = "bullish"
            elif macd_main < macd_signal_val:
                macd_signal = "bearish"

            # Trend Alignment berechnen (wie viele Indikatoren bullish)
            bullish_signals = 0
            total_signals = 0

            if rsi is not None:
                total_signals += 1
                if rsi < 50:
                    bullish_signals += 0.5  # Neutral-ish
                elif rsi < 30:
                    bullish_signals += 1  # Oversold = potential bullish

            if macd_signal == "bullish":
                bullish_signals += 1
                total_signals += 1
            elif macd_signal == "bearish":
                total_signals += 1

            trend_alignment = bullish_signals / total_signals if total_signals > 0 else 0.5

            # Bollinger Bands Position
            bb = indicators.get("bollinger", {})
            bb_upper = bb.get("upper")
            bb_lower = bb.get("lower")
            current_price = data.get("price", {}).get("last")
            bb_position = None

            if all(v is not None for v in [bb_upper, bb_lower, current_price]) and bb_upper != bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)

            return TechnicalSignal(
                available=True,
                rsi=rsi,
                rsi_signal=rsi_signal,
                macd_signal=macd_signal,
                trend_alignment=trend_alignment,
                bb_position=bb_position,
            )
        except Exception as e:
            logger.debug(f"Technical fetch error for {symbol}: {e}")
            return TechnicalSignal(available=False)

    async def check_services_health(self) -> dict[str, bool]:
        """Prüft die Erreichbarkeit aller ML-Services."""
        client = await self._get_client()
        services = {
            "data": settings.data_service_url,
            "nhits": settings.nhits_service_url,
            "hmm": settings.hmm_service_url,
            "tcn": settings.tcn_service_url,
            "candlestick": settings.candlestick_service_url,
            "rag": settings.rag_service_url,
            "llm": settings.llm_service_url,
        }

        results = {}
        for name, url in services.items():
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                results[name] = response.status_code == 200
            except Exception:
                results[name] = False

        return results


# Singleton-Instanz
signal_aggregator = SignalAggregatorService()
