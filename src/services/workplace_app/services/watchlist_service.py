"""
Watchlist Service.

Verwaltet die konfigurierbare Watchlist mit JSON-Persistenz.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from ..config import settings
from ..models.schemas import (
    WatchlistItem,
    WatchlistResponse,
    WatchlistAddRequest,
    WatchlistUpdateRequest,
    SignalDirection,
)


class WatchlistService:
    """Verwaltet die Trading-Watchlist."""

    def __init__(self):
        self._watchlist: dict[str, WatchlistItem] = {}
        self._file_path = Path(settings.watchlist_file)
        self._loaded = False

    async def initialize(self):
        """Initialisiert die Watchlist (lädt von Disk oder erstellt Default)."""
        if self._loaded:
            return

        # Flag ZUERST setzen um Rekursion zu verhindern
        self._loaded = True

        await self._load_from_disk()

        # Falls leer, Default-Symbole hinzufügen
        if not self._watchlist:
            logger.info("Initialisiere Watchlist mit Default-Symbolen")
            for symbol in settings.default_symbols:
                await self._add_internal(WatchlistAddRequest(
                    symbol=symbol,
                    is_favorite=symbol in ["BTCUSD", "EURUSD"],
                    alert_threshold=settings.default_alert_threshold,
                ))

        logger.info(f"Watchlist initialisiert mit {len(self._watchlist)} Symbolen")

    async def _load_from_disk(self):
        """Lädt die Watchlist von der JSON-Datei."""
        try:
            if self._file_path.exists():
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item_data in data.get("items", []):
                    # Datetime-Felder konvertieren
                    for dt_field in ["last_scan", "added_at"]:
                        if item_data.get(dt_field):
                            item_data[dt_field] = datetime.fromisoformat(
                                item_data[dt_field].replace("Z", "+00:00")
                            )

                    item = WatchlistItem(**item_data)
                    self._watchlist[item.symbol] = item

                logger.info(f"Watchlist geladen: {len(self._watchlist)} Symbole")
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Watchlist: {e}")
            self._watchlist = {}

    async def _save_to_disk(self):
        """Speichert die Watchlist auf Disk."""
        try:
            # Verzeichnis erstellen falls nötig
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

            # Zu JSON serialisieren
            items_data = []
            for item in self._watchlist.values():
                item_dict = item.model_dump()
                # Datetime zu ISO-String
                for dt_field in ["last_scan", "added_at"]:
                    if item_dict.get(dt_field):
                        item_dict[dt_field] = item_dict[dt_field].isoformat()
                items_data.append(item_dict)

            data = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "items": items_data
            }

            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Watchlist gespeichert: {len(self._watchlist)} Symbole")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Watchlist: {e}")

    async def get_all(self) -> list[WatchlistItem]:
        """Gibt alle Watchlist-Items zurück."""
        await self.initialize()
        return list(self._watchlist.values())

    async def get(self, symbol: str) -> Optional[WatchlistItem]:
        """Gibt ein einzelnes Watchlist-Item zurück."""
        await self.initialize()
        return self._watchlist.get(symbol.upper())

    async def _add_internal(self, request: WatchlistAddRequest, save: bool = True) -> WatchlistItem:
        """Interne Methode zum Hinzufügen (ohne initialize-Aufruf)."""
        symbol = request.symbol.upper()

        if symbol in self._watchlist:
            logger.debug(f"Symbol {symbol} bereits in Watchlist")
            return self._watchlist[symbol]

        if len(self._watchlist) >= settings.max_watchlist_size:
            raise ValueError(f"Watchlist ist voll (max {settings.max_watchlist_size})")

        item = WatchlistItem(
            symbol=symbol,
            is_favorite=request.is_favorite,
            alert_threshold=request.alert_threshold,
            timeframe=request.timeframe,
            notes=request.notes,
            added_at=datetime.now(timezone.utc),
        )

        self._watchlist[symbol] = item

        if save:
            await self._save_to_disk()

        logger.info(f"Symbol {symbol} zur Watchlist hinzugefügt")
        return item

    async def add(self, request: WatchlistAddRequest) -> WatchlistItem:
        """Fügt ein Symbol zur Watchlist hinzu."""
        await self.initialize()
        return await self._add_internal(request)

    async def update(self, symbol: str, request: WatchlistUpdateRequest) -> Optional[WatchlistItem]:
        """Aktualisiert ein Watchlist-Item."""
        await self.initialize()

        symbol = symbol.upper()
        if symbol not in self._watchlist:
            return None

        item = self._watchlist[symbol]

        # Nur gesetzte Felder aktualisieren
        if request.is_favorite is not None:
            item.is_favorite = request.is_favorite
        if request.alert_threshold is not None:
            item.alert_threshold = request.alert_threshold
        if request.timeframe is not None:
            item.timeframe = request.timeframe
        if request.notes is not None:
            item.notes = request.notes

        self._watchlist[symbol] = item
        await self._save_to_disk()

        logger.info(f"Watchlist-Item {symbol} aktualisiert")
        return item

    async def remove(self, symbol: str) -> bool:
        """Entfernt ein Symbol aus der Watchlist."""
        await self.initialize()

        symbol = symbol.upper()
        if symbol not in self._watchlist:
            return False

        del self._watchlist[symbol]
        await self._save_to_disk()

        logger.info(f"Symbol {symbol} aus Watchlist entfernt")
        return True

    async def update_scan_result(
        self,
        symbol: str,
        score: float,
        direction: SignalDirection
    ):
        """Aktualisiert das Scan-Ergebnis für ein Symbol."""
        await self.initialize()

        symbol = symbol.upper()
        if symbol not in self._watchlist:
            return

        item = self._watchlist[symbol]
        item.last_score = score
        item.last_direction = direction
        item.last_scan = datetime.now(timezone.utc)

        self._watchlist[symbol] = item
        # Nicht bei jedem Scan speichern (Performance)
        # await self._save_to_disk()

    async def increment_alert_count(self, symbol: str):
        """Erhöht den Alert-Counter für ein Symbol."""
        await self.initialize()

        symbol = symbol.upper()
        if symbol not in self._watchlist:
            return

        item = self._watchlist[symbol]
        item.alerts_triggered += 1
        self._watchlist[symbol] = item
        await self._save_to_disk()

    async def get_response(self) -> WatchlistResponse:
        """Erstellt die vollständige Watchlist-Response."""
        await self.initialize()

        items = list(self._watchlist.values())
        favorites_count = sum(1 for item in items if item.is_favorite)

        # Letzter Scan über alle Items
        last_scans = [item.last_scan for item in items if item.last_scan]
        last_scan = max(last_scans) if last_scans else None

        return WatchlistResponse(
            items=items,
            total=len(items),
            favorites_count=favorites_count,
            last_scan=last_scan,
        )

    async def get_symbols(self, favorites_only: bool = False) -> list[str]:
        """Gibt die Symbol-Liste zurück."""
        await self.initialize()

        if favorites_only:
            return [s for s, item in self._watchlist.items() if item.is_favorite]
        return list(self._watchlist.keys())

    async def save(self):
        """Explizites Speichern (z.B. bei Shutdown)."""
        await self._save_to_disk()


# Singleton-Instanz
watchlist_service = WatchlistService()
