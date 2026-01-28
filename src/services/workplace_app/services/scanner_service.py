"""
Scanner Service.

Background-Scanner für automatisches Scanning der Watchlist.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger

from ..config import settings
from ..models.schemas import (
    TradingSetup,
    ScanStatus,
    ScanStatusResponse,
    ScanTriggerResponse,
    SignalDirection,
)
from .signal_aggregator import signal_aggregator
from .scoring_service import scoring_service
from .watchlist_service import watchlist_service
from .setup_recorder_service import setup_recorder


class ScannerService:
    """Background-Scanner für Watchlist-Symbole."""

    def __init__(self):
        self._running = False
        self._status = ScanStatus.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._eval_task: Optional[asyncio.Task] = None
        self._results: dict[str, TradingSetup] = {}
        self._last_scan_time: Optional[datetime] = None
        self._current_symbol: Optional[str] = None
        self._symbols_scanned_total = 0
        self._errors_count = 0
        self._alerts_triggered = 0
        self._setups_recorded = 0
        self._scan_start_time: Optional[datetime] = None
        # Alert-State: Speichert letzten Alert-Zustand pro Symbol
        # Format: {symbol: {"alerted": bool, "direction": str, "score": float}}
        self._alert_state: dict[str, dict] = {}

    async def start(self):
        """Startet den Auto-Scanner."""
        if self._running:
            logger.info("Scanner läuft bereits")
            return

        self._running = True
        self._status = ScanStatus.RUNNING
        self._scan_start_time = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._scan_loop())
        self._eval_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Scanner gestartet (inkl. Evaluation-Job)")

    async def stop(self):
        """Stoppt den Auto-Scanner."""
        self._running = False
        self._status = ScanStatus.STOPPED

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass

        self._current_symbol = None
        logger.info("Scanner gestoppt")

    async def pause(self):
        """Pausiert den Scanner."""
        if self._running:
            self._status = ScanStatus.PAUSED
            logger.info("Scanner pausiert")

    async def resume(self):
        """Setzt den Scanner fort."""
        if self._status == ScanStatus.PAUSED:
            self._status = ScanStatus.RUNNING
            logger.info("Scanner fortgesetzt")

    async def _scan_loop(self):
        """Hauptschleife für periodisches Scanning."""
        while self._running:
            try:
                if self._status == ScanStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # Symbole von Watchlist holen
                symbols = await watchlist_service.get_symbols()

                if not symbols:
                    logger.debug("Keine Symbole in Watchlist")
                    await asyncio.sleep(settings.scan_interval_seconds)
                    continue

                logger.info(f"Starte Scan von {len(symbols)} Symbolen")
                scan_start = datetime.now(timezone.utc)

                for symbol in symbols:
                    if not self._running or self._status == ScanStatus.PAUSED:
                        break

                    self._current_symbol = symbol

                    try:
                        setup = await self._scan_symbol(symbol)
                        if setup:
                            self._results[symbol] = setup
                            self._symbols_scanned_total += 1

                            # Watchlist aktualisieren mit Multi-Timeframe Daten
                            await watchlist_service.update_scan_result(
                                symbol,
                                setup.composite_score,
                                setup.direction,
                                best_timeframe=setup.timeframe,
                                timeframe_scores=setup.timeframe_scores
                            )

                            # Alert prüfen
                            await self._check_alert(symbol, setup)

                            # Setup zur Prediction History aufzeichnen
                            # (nur bei relevanten Scores für Evaluation)
                            if setup.composite_score >= 50 and setup.direction != SignalDirection.NEUTRAL:
                                prediction_id = await setup_recorder.record_setup(setup)
                                if prediction_id:
                                    self._setups_recorded += 1

                    except Exception as e:
                        logger.error(f"Fehler beim Scannen von {symbol}: {e}")
                        self._errors_count += 1

                    # Kleine Pause zwischen Symbolen
                    await asyncio.sleep(0.5)

                self._current_symbol = None
                self._last_scan_time = datetime.now(timezone.utc)

                scan_duration = (self._last_scan_time - scan_start).total_seconds()
                logger.info(
                    f"Scan abgeschlossen: {len(symbols)} Symbole in {scan_duration:.1f}s"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler in Scan-Loop: {e}")
                self._errors_count += 1

            # Warten bis zum nächsten Scan
            await asyncio.sleep(settings.scan_interval_seconds)

    async def _evaluation_loop(self):
        """Background-Loop für periodische Evaluation vergangener Setups."""
        # Initiales Delay - warte bis Scanner warmgelaufen ist
        await asyncio.sleep(60)

        while self._running:
            try:
                if self._status != ScanStatus.PAUSED:
                    logger.debug("Starte Evaluation vergangener Setups")
                    stats = await setup_recorder.evaluate_pending_setups()

                    if stats.get("evaluated", 0) > 0:
                        logger.info(
                            f"Evaluation: {stats['correct']}/{stats['evaluated']} korrekt "
                            f"({stats['correct']/stats['evaluated']*100:.1f}%)"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler in Evaluation-Loop: {e}")

            # Evaluation alle 5 Minuten
            await asyncio.sleep(300)

    async def _scan_symbol(self, symbol: str) -> Optional[TradingSetup]:
        """Scannt ein einzelnes Symbol über die konfigurierten Timeframes."""
        try:
            # Timeframes aus Watchlist-Item laden (falls vorhanden)
            item = await watchlist_service.get(symbol)
            if item and item.timeframes:
                timeframes = item.timeframes
            else:
                # Fallback auf Standard-Timeframes
                timeframes = ["M5", "M15", "H1", "H4", "D1"]

            best_setup: Optional[TradingSetup] = None
            timeframe_scores: dict[str, float] = {}

            for timeframe in timeframes:
                try:
                    # Signale aggregieren für diesen Timeframe
                    signals = await signal_aggregator.fetch_all_signals(symbol, timeframe)

                    # Setup erstellen
                    setup = scoring_service.create_setup(symbol, timeframe, signals)

                    if setup:
                        timeframe_scores[timeframe] = setup.composite_score

                        # Bestes Setup merken
                        if best_setup is None or setup.composite_score > best_setup.composite_score:
                            best_setup = setup

                except Exception as e:
                    logger.debug(f"Scan-Fehler für {symbol}/{timeframe}: {e}")
                    continue

            # Timeframe-Scores im Setup speichern (falls vorhanden)
            if best_setup:
                # Speichere alle Scores für spätere Referenz
                best_setup.timeframe_scores = timeframe_scores

            return best_setup

        except Exception as e:
            logger.debug(f"Scan-Fehler für {symbol}: {e}")
            return None

    async def _check_alert(self, symbol: str, setup: TradingSetup):
        """
        Prüft ob ein Alert ausgelöst werden soll.

        Alert wird nur ausgelöst bei:
        1. Erstes Überschreiten der Schwelle (noch kein Alert für dieses Symbol)
        2. Richtungswechsel (z.B. von neutral zu long, oder von long zu short)
        3. Score fällt unter Schwelle und steigt wieder darüber
        """
        try:
            item = await watchlist_service.get(symbol)
            if not item:
                return

            current_direction = setup.direction.value
            current_score = setup.composite_score
            threshold = item.alert_threshold

            # Vorherigen Alert-State holen
            prev_state = self._alert_state.get(symbol, {})
            prev_alerted = prev_state.get("alerted", False)
            prev_direction = prev_state.get("direction", None)

            # Prüfen ob Score über Schwelle
            is_above_threshold = current_score >= threshold

            # Alert auslösen wenn:
            should_alert = False
            alert_reason = ""

            if is_above_threshold:
                if not prev_alerted:
                    # Fall 1: Erstes Überschreiten oder nach Reset
                    should_alert = True
                    alert_reason = "Schwelle überschritten"
                elif prev_direction and prev_direction != current_direction:
                    # Fall 2: Richtungswechsel während über Schwelle
                    should_alert = True
                    alert_reason = f"Richtungswechsel: {prev_direction} → {current_direction}"

            # State aktualisieren
            self._alert_state[symbol] = {
                "alerted": is_above_threshold,
                "direction": current_direction,
                "score": current_score
            }

            if should_alert:
                self._alerts_triggered += 1
                await watchlist_service.increment_alert_count(symbol)

                logger.info(
                    f"ALERT: {symbol} Score {current_score:.1f} "
                    f"(Threshold: {threshold}) - {current_direction} - {alert_reason}"
                )

                # TODO: Hier könnte Telegram/Push-Notification implementiert werden

        except Exception as e:
            logger.debug(f"Alert-Check Fehler für {symbol}: {e}")

    async def trigger_manual_scan(self) -> ScanTriggerResponse:
        """Löst einen manuellen Scan aus."""
        symbols = await watchlist_service.get_symbols()

        if not self._running:
            return ScanTriggerResponse(
                success=False,
                message="Scanner ist nicht gestartet",
                symbols_to_scan=len(symbols)
            )

        # Scan-Status zurücksetzen für sofortigen Re-Scan
        self._last_scan_time = None

        return ScanTriggerResponse(
            success=True,
            message="Manueller Scan wird beim nächsten Intervall ausgeführt",
            symbols_to_scan=len(symbols)
        )

    def get_top_setups(
        self,
        limit: int = 10,
        min_score: float = 0.0,
        direction: Optional[SignalDirection] = None
    ) -> list[TradingSetup]:
        """Gibt die Top-N Setups nach Score zurück."""
        # Filtern
        setups = list(self._results.values())

        if min_score > 0:
            setups = [s for s in setups if s.composite_score >= min_score]

        if direction:
            setups = [s for s in setups if s.direction == direction]

        # Sortieren nach Score (absteigend)
        setups.sort(key=lambda s: s.composite_score, reverse=True)

        return setups[:limit]

    async def get_top_setups_with_watchlist_fallback(
        self,
        limit: int = 10,
        min_score: float = 0.0,
        direction: Optional[SignalDirection] = None
    ) -> list[TradingSetup]:
        """
        Gibt die Top-N Setups zurück.

        Wenn keine Scanner-Ergebnisse vorhanden sind, werden Setups
        aus den Watchlist-Daten generiert.
        """
        # Erst Scanner-Ergebnisse prüfen
        setups = self.get_top_setups(limit=limit, min_score=min_score, direction=direction)

        if setups:
            return setups

        # Fallback: Setups aus Watchlist-Daten generieren
        logger.info("Keine Scanner-Ergebnisse, generiere Setups aus Watchlist-Daten")
        return await self._generate_setups_from_watchlist(limit, min_score, direction)

    async def _generate_setups_from_watchlist(
        self,
        limit: int,
        min_score: float,
        direction: Optional[SignalDirection]
    ) -> list[TradingSetup]:
        """Generiert TradingSetup-Objekte aus Watchlist-Daten."""
        from .watchlist_service import watchlist_service
        from ..models.schemas import (
            ConfidenceLevel,
            SignalAlignment,
        )

        items = await watchlist_service.get_all()
        setups = []

        for item in items:
            # Nur Items mit Score berücksichtigen
            if item.last_score is None or item.last_score <= 0:
                continue

            # Min-Score-Filter
            if item.last_score < min_score:
                continue

            # Direction-Filter
            item_direction = item.last_direction or SignalDirection.NEUTRAL
            if direction and item_direction != direction:
                continue

            # Confidence-Level bestimmen
            if item.last_score >= 75:
                confidence = ConfidenceLevel.HIGH
            elif item.last_score >= 60:
                confidence = ConfidenceLevel.MODERATE
            elif item.last_score >= 50:
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.WEAK

            setup = TradingSetup(
                symbol=item.symbol,
                timeframe=item.best_timeframe or item.timeframe or "H1",
                timestamp=item.last_scan or datetime.now(timezone.utc),
                direction=item_direction,
                composite_score=item.last_score,
                confidence_level=confidence,
                signal_alignment=SignalAlignment.MIXED,
                key_drivers=["Aus Watchlist-Cache"],
                signals_available=0,
                timeframe_scores=item.timeframe_scores,
            )
            setups.append(setup)

        # Sortieren nach Score (absteigend)
        setups.sort(key=lambda s: s.composite_score, reverse=True)

        return setups[:limit]

    def get_setup(self, symbol: str) -> Optional[TradingSetup]:
        """Gibt das Setup für ein Symbol zurück."""
        return self._results.get(symbol.upper())

    def get_status(self) -> ScanStatusResponse:
        """Gibt den aktuellen Scanner-Status zurück."""
        symbols = list(self._results.keys())

        # Nächster Scan berechnen
        next_scan = None
        if self._running and self._last_scan_time:
            next_scan = self._last_scan_time + timedelta(
                seconds=settings.scan_interval_seconds
            )

        return ScanStatusResponse(
            status=self._status,
            is_running=self._running,
            scan_interval_seconds=settings.scan_interval_seconds,
            last_scan_time=self._last_scan_time,
            next_scan_time=next_scan,
            symbols_in_queue=len(symbols),
            symbols_scanned_total=self._symbols_scanned_total,
            current_symbol=self._current_symbol,
            errors_count=self._errors_count,
            alerts_triggered=self._alerts_triggered,
            setups_recorded=self._setups_recorded,
        )

    @property
    def is_running(self) -> bool:
        """Prüft ob der Scanner läuft."""
        return self._running

    @property
    def results(self) -> dict[str, TradingSetup]:
        """Gibt alle Scan-Ergebnisse zurück."""
        return self._results


# Singleton-Instanz
scanner_service = ScannerService()
