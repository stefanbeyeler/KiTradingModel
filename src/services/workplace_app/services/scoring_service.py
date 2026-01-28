"""
Scoring Service.

Berechnet gewichteten Composite-Score aus allen Signal-Quellen.
"""

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from ..config import settings
from ..models.schemas import (
    NHITSSignal,
    HMMSignal,
    TCNSignal,
    CandlestickSignal,
    TechnicalSignal,
    CNNLSTMSignal,
    TradingSetup,
    SignalDirection,
    ConfidenceLevel,
    SignalAlignment,
    MarketRegime,
)


class ScoringService:
    """Berechnet gewichteten Composite-Score aus allen Signalen."""

    def calculate_composite_score(
        self,
        nhits: NHITSSignal,
        hmm: HMMSignal,
        tcn: TCNSignal,
        candlestick: CandlestickSignal,
        technical: TechnicalSignal,
        cnn_lstm: Optional[CNNLSTMSignal] = None,
    ) -> tuple[float, SignalDirection, list[str], SignalAlignment]:
        """
        Berechnet gewichteten Score.

        Gewichte (konfigurierbar):
        - NHITS trend_probability: 25%
        - HMM regime_confidence: 20%
        - CNN-LSTM multi-task: 20%
        - TCN pattern_confidence: 15%
        - Candlestick pattern_strength: 10%
        - Technical alignment: 10%

        Returns:
            (composite_score, direction, key_drivers, alignment)
        """
        if cnn_lstm is None:
            cnn_lstm = CNNLSTMSignal()
        scores = []
        weights = []
        drivers = []
        directions = []

        # NHITS (30%)
        if nhits.available:
            # Trend-Probability auf 0-100 skalieren
            nhits_score = nhits.trend_probability * 100
            scores.append(nhits_score)
            weights.append(settings.nhits_weight)
            directions.append(nhits.direction)

            if nhits.trend_probability >= 0.65:
                pct = f"{nhits.trend_probability:.0%}"
                drivers.append(f"NHITS: {nhits.direction.value} ({pct})")

        # HMM (25%)
        if hmm.available:
            # Signal-Score bereits 0-100
            hmm_score = hmm.signal_score

            # Alignment-Bonus/Penalty
            if hmm.alignment == "aligned":
                hmm_score += settings.alignment_bonus
            elif hmm.alignment == "contrary":
                hmm_score += settings.alignment_penalty

            hmm_score = max(0, min(100, hmm_score))  # Clamp
            scores.append(hmm_score)
            weights.append(settings.hmm_weight)

            # Direction aus Regime ableiten
            if hmm.regime in [MarketRegime.BULL_TREND]:
                directions.append(SignalDirection.LONG)
            elif hmm.regime in [MarketRegime.BEAR_TREND]:
                directions.append(SignalDirection.SHORT)
            else:
                directions.append(SignalDirection.NEUTRAL)

            if hmm.regime_probability >= 0.7:
                drivers.append(f"HMM: {hmm.regime.value} ({hmm.regime_probability:.0%})")

        # TCN (20%)
        if tcn.available and tcn.patterns:
            tcn_score = tcn.pattern_confidence * 100
            scores.append(tcn_score)
            weights.append(settings.tcn_weight)
            directions.append(tcn.direction)

            if tcn.pattern_confidence >= 0.6:
                patterns_str = ", ".join(tcn.patterns[:2])
                drivers.append(f"TCN: {patterns_str}")

        # Candlestick (15%)
        if candlestick.available and candlestick.patterns:
            candle_score = candlestick.pattern_strength * 100
            scores.append(candle_score)
            weights.append(settings.candlestick_weight)
            directions.append(candlestick.direction)

            if candlestick.pattern_strength >= 0.6:
                patterns_str = ", ".join(candlestick.patterns[:2])
                drivers.append(f"Candle: {patterns_str}")

        # Technical (10%)
        if technical.available:
            tech_score = technical.trend_alignment * 100
            scores.append(tech_score)
            weights.append(settings.technical_weight)

            # Direction aus RSI/MACD
            if technical.macd_signal == "bullish":
                directions.append(SignalDirection.LONG)
            elif technical.macd_signal == "bearish":
                directions.append(SignalDirection.SHORT)
            else:
                directions.append(SignalDirection.NEUTRAL)

            if technical.rsi is not None:
                if technical.rsi_signal == "oversold":
                    drivers.append(f"RSI: oversold ({technical.rsi:.0f})")
                elif technical.rsi_signal == "overbought":
                    drivers.append(f"RSI: overbought ({technical.rsi:.0f})")

        # CNN-LSTM (20%)
        if cnn_lstm.available:
            # Kombinierter Score aus Price-Confidence und Regime-Probability
            cnn_lstm_score = (cnn_lstm.price_confidence * 0.6 + cnn_lstm.regime_probability * 0.4) * 100

            # Bonus für erkannte Patterns
            if cnn_lstm.patterns and cnn_lstm.pattern_confidence >= 0.5:
                cnn_lstm_score = min(100, cnn_lstm_score + cnn_lstm.pattern_confidence * 10)

            scores.append(cnn_lstm_score)
            weights.append(settings.cnn_lstm_weight)
            directions.append(cnn_lstm.price_direction)

            if cnn_lstm.price_confidence >= 0.65:
                pct = f"{cnn_lstm.price_confidence:.0%}"
                drivers.append(f"CNN-LSTM: {cnn_lstm.price_direction.value} ({pct})")

            if cnn_lstm.patterns and cnn_lstm.pattern_confidence >= 0.6:
                patterns_str = ", ".join(cnn_lstm.patterns[:2])
                drivers.append(f"CNN-LSTM patterns: {patterns_str}")

        # Composite Score berechnen
        if not scores:
            return 0.0, SignalDirection.NEUTRAL, [], SignalAlignment.MIXED

        total_weight = sum(weights)
        composite = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Richtung bestimmen (Mehrheitsentscheidung)
        direction = self._determine_direction(directions)

        # Signal Alignment bestimmen (mit Mindestanzahl)
        alignment = self._determine_alignment(directions)

        # Coverage-Penalty: Weniger verfügbare Signale = niedrigerer Score
        signals_count = len(scores)
        coverage_factor = self._calculate_coverage_factor(signals_count)
        composite = composite * coverage_factor

        # Alignment-basierter Bonus/Penalty auf Composite
        # Nur wenn genügend Signale für aussagekräftiges Alignment vorhanden
        if signals_count >= settings.min_signals_for_moderate_alignment:
            if alignment == SignalAlignment.STRONG:
                composite = min(100, composite * 1.1)  # 10% Bonus
            elif alignment == SignalAlignment.CONFLICTING:
                composite = composite * 0.9  # 10% Penalty

        return round(composite, 1), direction, drivers, alignment

    def _calculate_coverage_factor(self, signals_count: int) -> float:
        """
        Berechnet den Coverage-Faktor basierend auf verfügbaren Signalen.

        Weniger Signale = niedrigerer maximaler Score.
        Beispiel mit min_signals_for_full_score=4 und penalty=0.15:
        - 4+ Signale: 100% (Faktor 1.0)
        - 3 Signale: 85% (Faktor 0.85)
        - 2 Signale: 70% (Faktor 0.70)
        - 1 Signal: 55% (Faktor 0.55)
        """
        if signals_count >= settings.min_signals_for_full_score:
            return 1.0

        missing_signals = settings.min_signals_for_full_score - signals_count
        penalty = missing_signals * settings.coverage_penalty_factor
        factor = 1.0 - penalty

        # Mindestens min_coverage_factor behalten
        return max(settings.min_coverage_factor, factor)

    def _determine_direction(self, directions: list[SignalDirection]) -> SignalDirection:
        """Bestimmt die Gesamt-Richtung aus allen Signal-Richtungen.

        Verwendet relative Mehrheit: Die Richtung mit den meisten Stimmen gewinnt.
        NEUTRAL wird nur bei Gleichstand zwischen LONG und SHORT zurückgegeben.
        """
        if not directions:
            return SignalDirection.NEUTRAL

        long_count = sum(1 for d in directions if d == SignalDirection.LONG)
        short_count = sum(1 for d in directions if d == SignalDirection.SHORT)

        # Relative Mehrheit: Die Richtung mit mehr Stimmen gewinnt
        if long_count > short_count:
            return SignalDirection.LONG
        elif short_count > long_count:
            return SignalDirection.SHORT
        else:
            # Gleichstand zwischen long und short → neutral
            return SignalDirection.NEUTRAL

    def _determine_alignment(self, directions: list[SignalDirection]) -> SignalAlignment:
        """
        Bestimmt wie gut die Signale übereinstimmen.

        STRONG erfordert mindestens min_signals_for_strong_alignment (default: 3) Signale.
        MODERATE erfordert mindestens min_signals_for_moderate_alignment (default: 2) Signale.
        """
        if not directions:
            return SignalAlignment.MIXED

        # Nur nicht-neutrale Signale zählen
        non_neutral = [d for d in directions if d != SignalDirection.NEUTRAL]

        if not non_neutral:
            return SignalAlignment.MIXED

        long_count = sum(1 for d in non_neutral if d == SignalDirection.LONG)
        short_count = sum(1 for d in non_neutral if d == SignalDirection.SHORT)
        total = len(non_neutral)

        # STRONG: Alle in eine Richtung UND genügend Signale
        if long_count == total or short_count == total:
            if total >= settings.min_signals_for_strong_alignment:
                return SignalAlignment.STRONG
            elif total >= settings.min_signals_for_moderate_alignment:
                return SignalAlignment.MODERATE
            else:
                # Zu wenige Signale für aussagekräftiges Alignment
                return SignalAlignment.MIXED

        # Klare Mehrheit (>= 75%) mit genügend Signalen
        if total >= settings.min_signals_for_moderate_alignment:
            if long_count / total >= 0.75 or short_count / total >= 0.75:
                return SignalAlignment.MODERATE

        # Mix mit einer Tendenz
        if long_count > short_count or short_count > long_count:
            return SignalAlignment.MIXED

        # Gleichverteilt = Konflikt
        return SignalAlignment.CONFLICTING

    def get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Bestimmt die Konfidenz-Stufe aus dem Score."""
        if score >= settings.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif score >= settings.moderate_confidence_threshold:
            return ConfidenceLevel.MODERATE
        elif score >= settings.min_confidence_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.WEAK

    def create_setup(
        self,
        symbol: str,
        timeframe: str,
        signals: dict,
        current_price: Optional[float] = None
    ) -> TradingSetup:
        """
        Erstellt ein vollständiges TradingSetup aus den Signal-Daten.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            signals: Dict mit Signal-Objekten von SignalAggregator
            current_price: Optionaler aktueller Preis

        Returns:
            TradingSetup mit allen Daten
        """
        # Signals extrahieren (mit Defaults)
        nhits = signals.get("nhits") or NHITSSignal()
        hmm = signals.get("hmm") or HMMSignal()
        tcn = signals.get("tcn") or TCNSignal()
        candlestick = signals.get("candlestick") or CandlestickSignal()
        technical = signals.get("technical") or TechnicalSignal()
        cnn_lstm = signals.get("cnn_lstm") or CNNLSTMSignal()

        # Score berechnen
        composite_score, direction, key_drivers, alignment = self.calculate_composite_score(
            nhits, hmm, tcn, candlestick, technical, cnn_lstm
        )

        # Anzahl verfügbarer Signale
        signals_available = sum([
            nhits.available,
            hmm.available,
            tcn.available and bool(tcn.patterns),
            candlestick.available and bool(candlestick.patterns),
            technical.available,
            cnn_lstm.available,
        ])

        return TradingSetup(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            composite_score=composite_score,
            confidence_level=self.get_confidence_level(composite_score),
            nhits_signal=nhits,
            hmm_signal=hmm,
            tcn_signal=tcn,
            candlestick_signal=candlestick,
            technical_signal=technical,
            cnn_lstm_signal=cnn_lstm,
            signal_alignment=alignment,
            key_drivers=key_drivers,
            signals_available=signals_available,
            current_price=current_price,
        )


# Singleton-Instanz
scoring_service = ScoringService()
