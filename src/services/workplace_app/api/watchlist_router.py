"""
Watchlist Router - Watchlist-Verwaltung.

CRUD-Operationen für die konfigurierbare Trading-Watchlist.
"""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.schemas import (
    WatchlistItem,
    WatchlistResponse,
    WatchlistAddRequest,
    WatchlistUpdateRequest,
)
from ..services.watchlist_service import watchlist_service

router = APIRouter()


@router.get(
    "/",
    response_model=WatchlistResponse,
    summary="Watchlist abrufen",
    description="Gibt die vollständige Watchlist mit allen Symbolen zurück."
)
async def get_watchlist(
    favorites_only: bool = Query(
        default=False,
        description="Nur Favoriten anzeigen"
    ),
):
    """
    Watchlist abrufen.

    Gibt alle Symbole in der Watchlist mit ihren Einstellungen,
    letzten Scores und Alert-Konfigurationen zurück.
    """
    try:
        response = await watchlist_service.get_response()

        # Filter wenn nur Favoriten
        if favorites_only:
            response.items = [item for item in response.items if item.is_favorite]
            response.total = len(response.items)

        return response

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Watchlist: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Abrufen der Watchlist: {str(e)}"
        )


@router.get(
    "/symbols",
    response_model=list[str],
    summary="Symbol-Liste",
    description="Gibt nur die Symbol-Namen zurück."
)
async def get_symbols(
    favorites_only: bool = Query(default=False, description="Nur Favoriten"),
):
    """Gibt die Symbol-Namen der Watchlist zurück."""
    return await watchlist_service.get_symbols(favorites_only=favorites_only)


@router.get(
    "/{symbol}",
    response_model=WatchlistItem,
    summary="Einzelnes Symbol",
    description="Gibt die Details eines Watchlist-Items zurück."
)
async def get_watchlist_item(symbol: str):
    """Details für ein einzelnes Watchlist-Item."""
    symbol = symbol.upper()
    item = await watchlist_service.get(symbol)

    if not item:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol} nicht in Watchlist"
        )

    return item


@router.post(
    "/",
    response_model=WatchlistItem,
    status_code=201,
    summary="Symbol hinzufügen",
    description="Fügt ein neues Symbol zur Watchlist hinzu."
)
async def add_to_watchlist(request: WatchlistAddRequest):
    """
    Symbol zur Watchlist hinzufügen.

    Konfigurierbare Optionen:
    - is_favorite: Als Favorit markieren
    - alert_threshold: Score-Schwelle für Alerts (0-100)
    - timeframe: Bevorzugter Timeframe für Scans
    - notes: Benutzer-Notizen
    """
    try:
        item = await watchlist_service.add(request)
        logger.info(f"Symbol {request.symbol} zur Watchlist hinzugefügt")
        return item

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Fehler beim Hinzufügen zu Watchlist: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Hinzufügen: {str(e)}"
        )


@router.put(
    "/{symbol}",
    response_model=WatchlistItem,
    summary="Symbol aktualisieren",
    description="Aktualisiert die Einstellungen eines Watchlist-Items."
)
async def update_watchlist_item(
    symbol: str,
    request: WatchlistUpdateRequest,
):
    """
    Watchlist-Item aktualisieren.

    Nur die angegebenen Felder werden aktualisiert.
    """
    symbol = symbol.upper()

    try:
        item = await watchlist_service.update(symbol, request)

        if not item:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} nicht in Watchlist"
            )

        logger.info(f"Watchlist-Item {symbol} aktualisiert")
        return item

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren von {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Aktualisieren: {str(e)}"
        )


@router.delete(
    "/{symbol}",
    status_code=204,
    summary="Symbol entfernen",
    description="Entfernt ein Symbol aus der Watchlist."
)
async def remove_from_watchlist(symbol: str):
    """Symbol aus der Watchlist entfernen."""
    symbol = symbol.upper()

    try:
        removed = await watchlist_service.remove(symbol)

        if not removed:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} nicht in Watchlist"
            )

        logger.info(f"Symbol {symbol} aus Watchlist entfernt")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Entfernen von {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Entfernen: {str(e)}"
        )


@router.post(
    "/{symbol}/favorite",
    response_model=WatchlistItem,
    summary="Als Favorit markieren",
    description="Toggled den Favorit-Status eines Symbols."
)
async def toggle_favorite(symbol: str):
    """Toggle Favorit-Status."""
    symbol = symbol.upper()

    item = await watchlist_service.get(symbol)
    if not item:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol} nicht in Watchlist"
        )

    # Toggle
    updated = await watchlist_service.update(
        symbol,
        WatchlistUpdateRequest(is_favorite=not item.is_favorite)
    )

    return updated
