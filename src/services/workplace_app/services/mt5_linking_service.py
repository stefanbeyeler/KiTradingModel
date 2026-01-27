"""MT5 Linking Service.

Automatische und manuelle Verknüpfung von MT5 Trades mit Trading-Setups.
Implementiert den Linking-Algorithmus basierend auf Symbol, Richtung und Zeit.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger

from ..config import settings
from ..models.schemas import (
    MT5Trade,
    MT5TradeSetupLink,
    TradingSetup,
    SignalDirection,
    MT5LinkType,
    MT5TradeType,
)
from .signal_aggregator import signal_aggregator
from .scoring_service import scoring_service
from .mt5_trade_service import mt5_trade_service


class MT5LinkingService:
    """
    Service für Trade-Setup-Verknüpfungen.

    Linking-Kriterien:
    1. Symbol muss exakt übereinstimmen
    2. Richtung muss passen (buy = long, sell = short)
    3. Zeit-Nähe: Setup innerhalb von LINK_WINDOW vor Trade-Entry
    4. Score-Schwelle: Setup-Score >= MIN_SCORE
    """

    # Konfiguration
    LINK_WINDOW_HOURS = 4  # Zeitfenster für Setup-Suche
    MIN_SCORE = 50.0  # Mindest-Score für Verknüpfung
    MIN_LINK_CONFIDENCE = 0.5  # Mindest-Konfidenz für Auto-Link

    def __init__(self):
        self._setup_cache: dict[str, list[TradingSetup]] = {}
        self._cache_ttl = timedelta(minutes=30)
        self._cache_timestamp: Optional[datetime] = None

    async def auto_link_trade(self, trade: MT5Trade) -> Optional[MT5TradeSetupLink]:
        """
        Verknüpft einen Trade automatisch mit dem passendsten Setup.

        Args:
            trade: Der zu verknüpfende Trade

        Returns:
            TradeSetupLink wenn Verknüpfung erfolgreich, sonst None
        """
        logger.debug(f"Auto-linking trade {trade.trade_id} ({trade.symbol})")

        # Setups für das Symbol suchen
        setups = await self._get_recent_setups(
            symbol=trade.symbol,
            since=trade.entry_time - timedelta(hours=self.LINK_WINDOW_HOURS),
            until=trade.entry_time,
        )

        if not setups:
            logger.debug(f"No setups found for {trade.symbol} in time window")
            return None

        # Trade-Richtung bestimmen
        trade_direction = (
            SignalDirection.LONG
            if trade.trade_type == MT5TradeType.BUY
            else SignalDirection.SHORT
        )

        # Passende Setups filtern
        matching_setups = []
        for setup in setups:
            # Richtung muss passen (oder neutral)
            if setup.direction not in (trade_direction, SignalDirection.NEUTRAL):
                continue

            # Score-Schwelle prüfen
            if setup.composite_score < self.MIN_SCORE:
                continue

            matching_setups.append(setup)

        if not matching_setups:
            logger.debug(f"No matching setups for trade direction {trade_direction.value}")
            return None

        # Beste Übereinstimmung finden
        best_setup, best_confidence = self._find_best_match(
            trade=trade,
            setups=matching_setups,
            trade_direction=trade_direction,
        )

        if best_confidence < self.MIN_LINK_CONFIDENCE:
            logger.debug(f"Best match confidence {best_confidence:.2f} below threshold")
            return None

        # Link erstellen
        link_data = self._create_link_data(
            trade=trade,
            setup=best_setup,
            link_type=MT5LinkType.AUTO,
            link_confidence=best_confidence,
        )

        # An Data Service senden
        link = await mt5_trade_service.create_trade_link(link_data)

        if link:
            logger.info(
                f"Auto-linked trade {trade.trade_id} to setup "
                f"({best_setup.timestamp}, score={best_setup.composite_score:.1f}, "
                f"confidence={best_confidence:.2f})"
            )

        return link

    async def manual_link_trade(
        self,
        trade_id: str,
        setup_timestamp: datetime,
        setup_timeframe: str = "H1",
        notes: Optional[str] = None,
    ) -> Optional[MT5TradeSetupLink]:
        """
        Verknüpft einen Trade manuell mit einem Setup zu einem bestimmten Zeitpunkt.

        Args:
            trade_id: ID des Trades
            setup_timestamp: Zeitstempel des gewünschten Setups
            setup_timeframe: Timeframe des Setups
            notes: Optionale Notizen

        Returns:
            TradeSetupLink wenn erfolgreich, sonst None
        """
        # Trade holen
        trade = await mt5_trade_service.get_trade(trade_id, include_link=False)
        if not trade:
            logger.warning(f"Trade {trade_id} not found for manual linking")
            return None

        # Setup für den Zeitpunkt generieren
        setup = await self._generate_setup_at_time(
            symbol=trade.symbol,
            timeframe=setup_timeframe,
            timestamp=setup_timestamp,
        )

        if not setup:
            logger.warning(f"Could not generate setup for {trade.symbol} at {setup_timestamp}")
            return None

        # Link erstellen
        link_data = self._create_link_data(
            trade=trade,
            setup=setup,
            link_type=MT5LinkType.MANUAL,
            link_confidence=1.0,  # Manuell = 100% Konfidenz
            notes=notes,
        )

        link = await mt5_trade_service.create_trade_link(link_data)

        if link:
            logger.info(f"Manually linked trade {trade_id} to setup at {setup_timestamp}")

        return link

    async def unlink_trade(self, trade_id: str) -> bool:
        """Entfernt die Setup-Verknüpfung von einem Trade."""
        success = await mt5_trade_service.delete_trade_link(trade_id)
        if success:
            logger.info(f"Unlinked trade {trade_id}")
        return success

    async def auto_link_new_trades(self) -> int:
        """
        Verknüpft alle neuen Trades automatisch mit Setups.

        Wird periodisch aufgerufen um neue Trades zu verarbeiten.

        Returns:
            Anzahl der erfolgreich verknüpften Trades
        """
        # Trades ohne Link holen (letzte 24 Stunden)
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        trades, _ = await mt5_trade_service.get_trades(
            since=since,
            limit=500,
            include_links=True,
        )

        linked_count = 0
        for trade in trades:
            # Nur Trades ohne bestehende Verknüpfung
            if trade.setup:
                continue

            link = await self.auto_link_trade(trade)
            if link:
                linked_count += 1

        if linked_count > 0:
            logger.info(f"Auto-linked {linked_count} trades to setups")

        return linked_count

    def _find_best_match(
        self,
        trade: MT5Trade,
        setups: list[TradingSetup],
        trade_direction: SignalDirection,
    ) -> tuple[TradingSetup, float]:
        """
        Findet das beste passende Setup für einen Trade.

        Konfidenz-Berechnung:
        - Zeit-Faktor (40%): 100% bei < 15 Min, 50% bei LINK_WINDOW
        - Score-Faktor (40%): Normalisierter Setup-Score
        - Richtungs-Faktor (20%): 100% bei Übereinstimmung, 50% bei neutral

        Returns:
            Tuple aus (bestes Setup, Konfidenz)
        """
        scored = []

        for setup in setups:
            # Zeit-Differenz in Stunden
            time_diff = (trade.entry_time - setup.timestamp).total_seconds() / 3600

            # Zeit-Faktor: Linear von 1.0 (0h) zu 0.5 (LINK_WINDOW)
            time_factor = max(0.5, 1.0 - (time_diff / self.LINK_WINDOW_HOURS) * 0.5)

            # Score-Faktor: Normalisiert auf 0-1
            score_factor = setup.composite_score / 100.0

            # Richtungs-Faktor
            if setup.direction == trade_direction:
                direction_factor = 1.0
            elif setup.direction == SignalDirection.NEUTRAL:
                direction_factor = 0.5
            else:
                direction_factor = 0.0  # Sollte nicht vorkommen nach Filterung

            # Gewichtete Konfidenz
            confidence = (
                time_factor * 0.4 +
                score_factor * 0.4 +
                direction_factor * 0.2
            )

            scored.append((setup, confidence))

        # Sortieren nach Konfidenz (absteigend)
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0] if scored else (setups[0], 0.0)

    def _create_link_data(
        self,
        trade: MT5Trade,
        setup: TradingSetup,
        link_type: MT5LinkType,
        link_confidence: float,
        notes: Optional[str] = None,
    ) -> dict:
        """Erstellt die Link-Daten für den Data Service."""
        return {
            "trade_id": trade.trade_id,
            "setup_symbol": setup.symbol,
            "setup_timeframe": setup.timeframe,
            "setup_timestamp": setup.timestamp.isoformat(),
            "setup_direction": setup.direction.value,
            "setup_score": setup.composite_score,
            "setup_confidence": setup.confidence_level.value,
            "nhits_direction": setup.nhits_signal.direction.value if setup.nhits_signal.available else None,
            "nhits_probability": setup.nhits_signal.trend_probability if setup.nhits_signal.available else None,
            "hmm_regime": setup.hmm_signal.regime.value if setup.hmm_signal.available else None,
            "hmm_score": setup.hmm_signal.signal_score if setup.hmm_signal.available else None,
            "tcn_patterns": setup.tcn_signal.patterns if setup.tcn_signal.available else None,
            "tcn_confidence": setup.tcn_signal.pattern_confidence if setup.tcn_signal.available else None,
            "candlestick_patterns": setup.candlestick_signal.patterns if setup.candlestick_signal.available else None,
            "candlestick_strength": setup.candlestick_signal.pattern_strength if setup.candlestick_signal.available else None,
            "link_type": link_type.value,
            "link_confidence": link_confidence,
            "notes": notes,
        }

    async def _get_recent_setups(
        self,
        symbol: str,
        since: datetime,
        until: datetime,
    ) -> list[TradingSetup]:
        """
        Holt kürzliche Setups für ein Symbol.

        Nutzt den Scanner-Cache oder generiert neue Setups.
        In Zukunft: Prediction History aus Data Service abfragen.
        """
        # Aktuell: Neues Setup generieren (keine Historie verfügbar)
        # TODO: Aus prediction_history abfragen wenn verfügbar

        try:
            # Aktuelles Setup holen (als Näherung)
            signals = await signal_aggregator.fetch_all_signals(symbol, "H1")
            setup = scoring_service.create_setup(symbol, "H1", signals)

            if setup:
                return [setup]

        except Exception as e:
            logger.warning(f"Failed to get setups for {symbol}: {e}")

        return []

    async def _generate_setup_at_time(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
    ) -> Optional[TradingSetup]:
        """
        Generiert ein Setup für einen bestimmten Zeitpunkt.

        Für manuelle Verknüpfung - nutzt historische Daten wenn möglich.
        """
        try:
            # Aktuell: Nutze aktuelles Setup als Näherung
            # TODO: Historische Signale abrufen wenn verfügbar
            signals = await signal_aggregator.fetch_all_signals(symbol, timeframe)
            setup = scoring_service.create_setup(symbol, timeframe, signals)

            if setup:
                # Timestamp überschreiben für manuelles Linking
                setup.timestamp = timestamp

            return setup

        except Exception as e:
            logger.error(f"Failed to generate setup for {symbol} at {timestamp}: {e}")
            return None


# Singleton-Instanz
mt5_linking_service = MT5LinkingService()
