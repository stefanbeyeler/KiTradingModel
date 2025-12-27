"""Pattern History Service - Speichert und verwaltet erkannte Candlestick-Patterns.

Dieser Service fuehrt periodische Scans durch und speichert die Ergebnisse
fuer die Anzeige im Frontend.
"""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from ..models.schemas import (
    PatternHistoryEntry,
    PatternScanRequest,
    Timeframe,
)
from .pattern_detection_service import candlestick_pattern_service


# Data directory
DATA_DIR = os.getenv("DATA_DIR", "/app/data")


class PatternHistoryService:
    """
    Service fuer die Verwaltung der Pattern-History.

    Speichert erkannte Patterns in einer JSON-Datei und fuehrt
    periodische Scans durch.
    """

    def __init__(self, history_file: str = None):
        if history_file is None:
            history_file = os.path.join(DATA_DIR, "pattern_history.json")
        self._history_file = Path(history_file)
        self._history: list[PatternHistoryEntry] = []
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))  # 5 Minuten
        self._max_history_entries = int(os.getenv("MAX_HISTORY_ENTRIES", "1000"))
        self._max_history_age_hours = int(os.getenv("MAX_HISTORY_AGE_HOURS", "24"))

        # Timeframes fuer den automatischen Scan
        self._scan_timeframes = [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.D1]

        # Lade bestehende History
        self._load_history()

        logger.info(f"PatternHistoryService initialized - {len(self._history)} entries loaded")

    def _load_history(self):
        """Lade History aus Datei."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r') as f:
                    data = json.load(f)
                    self._history = []
                    for entry in data:
                        # Migration: Alte Eintraege ohne detected_at
                        if 'detected_at' not in entry:
                            entry['detected_at'] = entry.get('timestamp', '')
                        self._history.append(PatternHistoryEntry(**entry))
                logger.info(f"Loaded {len(self._history)} pattern history entries")
        except Exception as e:
            logger.error(f"Failed to load pattern history: {e}")
            self._history = []

    def _save_history(self):
        """Speichere History in Datei."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, 'w') as f:
                json.dump([entry.to_dict() for entry in self._history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pattern history: {e}")

    def _cleanup_old_entries(self):
        """Entferne alte Eintraege."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_history_age_hours)
        cutoff_str = cutoff.isoformat()

        original_count = len(self._history)
        self._history = [
            entry for entry in self._history
            if entry.timestamp > cutoff_str
        ]

        # Begrenze auf max Eintraege
        if len(self._history) > self._max_history_entries:
            self._history = self._history[-self._max_history_entries:]

        removed = original_count - len(self._history)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old pattern entries")

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Gibt die Anzahl Minuten fuer einen Timeframe zurueck."""
        tf_minutes = {
            "M5": 5,
            "M15": 15,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
        }
        return tf_minutes.get(timeframe, 60)

    def _is_duplicate(self, symbol: str, timeframe: str, pattern_type: str, pattern_ts: str) -> bool:
        """
        Prueft ob ein Pattern bereits in der History existiert.

        Ein Pattern gilt als Duplikat wenn Symbol, Timeframe, Pattern-Typ
        uebereinstimmen und der Timestamp innerhalb des Timeframe-Intervalls liegt.
        Dies verhindert, dass ueberlappende Patterns (z.B. Three White Soldiers
        auf Kerzen 1-2-3 und 2-3-4) mehrfach gespeichert werden.
        """
        try:
            # Parse den neuen Pattern-Timestamp
            new_ts = datetime.fromisoformat(pattern_ts.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            # Fallback: exakter String-Vergleich
            for entry in self._history:
                if (entry.symbol == symbol and
                    entry.timeframe == timeframe and
                    entry.pattern_type == pattern_type):
                    if entry.id.endswith(pattern_ts):
                        return True
            return False

        # Zeitfenster basierend auf Timeframe (Patterns innerhalb einer Kerze = Duplikat)
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        tolerance = timedelta(minutes=timeframe_minutes)

        for entry in self._history:
            if (entry.symbol == symbol and
                entry.timeframe == timeframe and
                entry.pattern_type == pattern_type):
                try:
                    # Extrahiere Timestamp aus der ID (letzter Teil nach dem letzten _)
                    entry_ts_str = entry.id.split('_')[-1]
                    # Versuche verschiedene Formate
                    try:
                        entry_ts = datetime.fromisoformat(entry_ts_str.replace('Z', '+00:00'))
                    except ValueError:
                        # Timestamp koennte + statt +00:00 haben
                        entry_ts = datetime.fromisoformat(entry_ts_str)

                    # Pruefe ob innerhalb des Tolerance-Fensters
                    if abs((new_ts - entry_ts).total_seconds()) < tolerance.total_seconds():
                        return True
                except (ValueError, AttributeError, IndexError):
                    # Fallback: exakter Vergleich
                    if entry.id.endswith(pattern_ts):
                        return True

        return False

    async def _get_available_timeframes(self, symbol: str) -> list[Timeframe]:
        """Pruefe welche Timeframes fuer ein Symbol verfuegbar sind."""
        available = []

        for tf in self._scan_timeframes:
            try:
                # Try to fetch a small amount of data to check availability
                request = PatternScanRequest(
                    symbol=symbol,
                    timeframes=[tf],
                    lookback_candles=10,
                    min_confidence=0.9,  # High threshold to get quick response
                )
                response = await candlestick_pattern_service.scan_patterns(request)
                # Check if we got data
                tf_result = getattr(response.result, tf.value.lower(), None)
                if tf_result and tf_result.candles_analyzed >= 3:
                    available.append(tf)
            except Exception:
                continue

        return available

    async def scan_all_symbols(self) -> int:
        """
        Scanne alle Symbole nach Patterns.

        Returns:
            Anzahl der gefundenen Patterns
        """
        try:
            # Hole alle verfuegbaren Symbole
            symbols = await candlestick_pattern_service._get_symbol_names()

            if not symbols:
                logger.warning("No symbols available for pattern scanning")
                return 0

            logger.info(f"Scanning {len(symbols)} symbols for patterns...")

            new_patterns = 0
            scan_time = datetime.now(timezone.utc)

            for symbol in symbols:
                try:
                    # Preufe ob Daten fuer die Timeframes verfuegbar sind
                    available_timeframes = await self._get_available_timeframes(symbol)

                    if not available_timeframes:
                        continue

                    # Scanne nur verfuegbare Timeframes
                    request = PatternScanRequest(
                        symbol=symbol,
                        timeframes=available_timeframes,
                        lookback_candles=50,  # Weniger Kerzen fuer schnelleren Scan
                        min_confidence=0.6,
                        include_weak_patterns=False,
                    )

                    response = await candlestick_pattern_service.scan_patterns(request)

                    # Sammle alle Patterns aus allen Timeframes
                    all_patterns = (
                        response.result.m5.patterns +
                        response.result.m15.patterns +
                        response.result.h1.patterns +
                        response.result.h4.patterns +
                        response.result.d1.patterns
                    )

                    # Fuege nur neue/aktuelle Patterns hinzu (Duplikat-Pruefung)
                    for pattern in all_patterns:
                        # Erstelle eindeutige ID basierend auf Pattern-Timestamp, Symbol, Timeframe und Typ
                        pattern_ts = pattern.timestamp.isoformat() if pattern.timestamp else scan_time.isoformat()
                        unique_key = f"{symbol}_{pattern.timeframe.value}_{pattern.pattern_type.value}_{pattern_ts}"

                        # Pruefe ob dieses Pattern bereits existiert
                        if self._is_duplicate(symbol, pattern.timeframe.value, pattern.pattern_type.value, pattern_ts):
                            continue

                        entry = PatternHistoryEntry(
                            id=unique_key,
                            timestamp=pattern_ts,  # Pattern-Zeitpunkt (Kerzen-Close)
                            detected_at=scan_time.isoformat(),  # Scan-Zeitpunkt
                            symbol=symbol,
                            pattern_type=pattern.pattern_type.value,
                            category=pattern.category.value,
                            direction=pattern.direction.value,
                            strength=pattern.strength.value,
                            confidence=pattern.confidence,
                            timeframe=pattern.timeframe.value,
                            price_at_detection=pattern.price_at_detection,
                            description=pattern.description,
                            trading_implication=pattern.trading_implication,
                        )
                        self._history.append(entry)
                        new_patterns += 1

                except Exception as e:
                    logger.warning(f"Error scanning {symbol}: {e}")
                    continue

            # Cleanup und speichern
            self._cleanup_old_entries()
            self._save_history()

            logger.info(f"Pattern scan complete: {new_patterns} new patterns found")
            return new_patterns

        except Exception as e:
            logger.error(f"Error in pattern scan: {e}")
            return 0

    async def _scan_loop(self):
        """Hintergrund-Scan-Loop."""
        while self._running:
            try:
                await self.scan_all_symbols()
            except Exception as e:
                logger.error(f"Error in pattern scan loop: {e}")

            await asyncio.sleep(self._scan_interval)

    async def start(self):
        """Starte den periodischen Scan."""
        if self._running:
            return

        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info(f"Pattern scan started (interval: {self._scan_interval}s)")

    async def stop(self):
        """Stoppe den periodischen Scan."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Pattern scan stopped")

    def is_running(self) -> bool:
        """Check if the scan is running."""
        return self._running

    def get_history(
        self,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        category: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[dict]:
        """
        Hole Pattern-History mit optionalen Filtern.

        Args:
            symbol: Filter nach Symbol
            direction: Filter nach Richtung (bullish, bearish, neutral)
            category: Filter nach Kategorie (reversal, continuation, indecision)
            timeframe: Filter nach Timeframe (M15, H1, H4, D1)
            min_confidence: Minimale Konfidenz
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Liste der Pattern-Eintraege
        """
        results = []

        for entry in reversed(self._history):  # Neueste zuerst
            # Filter anwenden
            if symbol and entry.symbol != symbol:
                continue
            if direction and entry.direction != direction:
                continue
            if category and entry.category != category:
                continue
            if timeframe and entry.timeframe != timeframe:
                continue
            if entry.confidence < min_confidence:
                continue

            results.append(entry.to_dict())

            if len(results) >= limit:
                break

        return results

    def get_latest_by_symbol(self) -> dict[str, list[dict]]:
        """
        Hole die neuesten Patterns gruppiert nach Symbol.

        Returns:
            Dictionary mit Symbol als Key und Liste der Patterns als Value
        """
        by_symbol: dict[str, list[dict]] = {}

        for entry in reversed(self._history):
            if entry.symbol not in by_symbol:
                by_symbol[entry.symbol] = []

            # Max 5 Patterns pro Symbol
            if len(by_symbol[entry.symbol]) < 5:
                by_symbol[entry.symbol].append(entry.to_dict())

        return by_symbol

    def get_statistics(self) -> dict:
        """Hole Statistiken ueber die Pattern-History."""
        if not self._history:
            return {
                "total_patterns": 0,
                "symbols_with_patterns": 0,
                "by_direction": {},
                "by_category": {},
                "by_timeframe": {},
                "last_scan": None,
            }

        by_direction = {"bullish": 0, "bearish": 0, "neutral": 0}
        by_category = {"reversal": 0, "continuation": 0, "indecision": 0}
        by_timeframe = {"M5": 0, "M15": 0, "H1": 0, "H4": 0, "D1": 0}
        symbols = set()

        for entry in self._history:
            symbols.add(entry.symbol)
            by_direction[entry.direction] = by_direction.get(entry.direction, 0) + 1
            by_category[entry.category] = by_category.get(entry.category, 0) + 1
            by_timeframe[entry.timeframe] = by_timeframe.get(entry.timeframe, 0) + 1

        return {
            "total_patterns": len(self._history),
            "symbols_with_patterns": len(symbols),
            "by_direction": by_direction,
            "by_category": by_category,
            "by_timeframe": by_timeframe,
            "last_scan": self._history[-1].timestamp if self._history else None,
            "scan_running": self._running,
            "scan_interval_seconds": self._scan_interval,
        }

    def clear_history(self):
        """Loesche die gesamte History."""
        self._history = []
        self._save_history()
        logger.info("Pattern history cleared")


# Global singleton
pattern_history_service = PatternHistoryService()
