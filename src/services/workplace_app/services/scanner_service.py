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


class ScannerService:
    """Background-Scanner für Watchlist-Symbole."""

    def __init__(self):
        self._running = False
        self._status = ScanStatus.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._results: dict[str, TradingSetup] = {}
        self._last_scan_time: Optional[datetime] = None
        self._current_symbol: Optional[str] = None
        self._symbols_scanned_total = 0
        self._errors_count = 0
        self._alerts_triggered = 0
        self._scan_start_time: Optional[datetime] = None

    async def start(self):
        """Startet den Auto-Scanner."""
        if self._running:
            logger.info("Scanner läuft bereits")
            return

        self._running = True
        self._status = ScanStatus.RUNNING
        self._scan_start_time = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._scan_loop())
        logger.info("Scanner gestartet")

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

                            # Watchlist aktualisieren
                            await watchlist_service.update_scan_result(
                                symbol,
                                setup.composite_score,
                                setup.direction
                            )

                            # Alert prüfen
                            await self._check_alert(symbol, setup)

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

    async def _scan_symbol(self, symbol: str) -> Optional[TradingSetup]:
        """Scannt ein einzelnes Symbol."""
        try:
            # Watchlist-Item für Timeframe holen
            item = await watchlist_service.get(symbol)
            timeframe = item.timeframe if item else "H1"

            # Signale aggregieren
            signals = await signal_aggregator.fetch_all_signals(symbol, timeframe)

            # Setup erstellen
            setup = scoring_service.create_setup(symbol, timeframe, signals)

            return setup

        except Exception as e:
            logger.debug(f"Scan-Fehler für {symbol}: {e}")
            return None

    async def _check_alert(self, symbol: str, setup: TradingSetup):
        """Prüft ob ein Alert ausgelöst werden soll."""
        try:
            item = await watchlist_service.get(symbol)
            if not item:
                return

            # Alert wenn Score über Schwelle
            if setup.composite_score >= item.alert_threshold:
                self._alerts_triggered += 1
                await watchlist_service.increment_alert_count(symbol)

                logger.info(
                    f"ALERT: {symbol} Score {setup.composite_score:.1f} "
                    f"(Threshold: {item.alert_threshold}) - {setup.direction.value}"
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
