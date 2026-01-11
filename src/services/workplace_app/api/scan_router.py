"""
Scan Router - Scanner-Kontrolle.

Endpoints zum Starten, Stoppen und Überwachen des Auto-Scanners.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models.schemas import (
    ScanStatusResponse,
    ScanTriggerResponse,
)
from ..services.scanner_service import scanner_service

router = APIRouter()


@router.get(
    "/status",
    response_model=ScanStatusResponse,
    summary="Scanner-Status",
    description="Gibt den aktuellen Status des Auto-Scanners zurück."
)
async def get_scan_status():
    """
    Scanner-Status abrufen.

    Zeigt an, ob der Scanner läuft, wann der letzte Scan war,
    wie viele Symbole gescannt wurden und wie viele Alerts
    ausgelöst wurden.
    """
    return scanner_service.get_status()


@router.post(
    "/start",
    response_model=ScanTriggerResponse,
    summary="Scanner starten",
    description="Startet den Auto-Scanner."
)
async def start_scanner():
    """
    Auto-Scanner starten.

    Der Scanner läuft im Hintergrund und scannt periodisch
    alle Symbole in der Watchlist.
    """
    try:
        if scanner_service.is_running:
            return ScanTriggerResponse(
                success=True,
                message="Scanner läuft bereits",
                symbols_to_scan=len(scanner_service.results),
            )

        await scanner_service.start()

        return ScanTriggerResponse(
            success=True,
            message="Scanner gestartet",
            symbols_to_scan=0,  # Wird beim ersten Scan gefüllt
        )

    except Exception as e:
        logger.error(f"Fehler beim Starten des Scanners: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Starten: {str(e)}"
        )


@router.post(
    "/stop",
    response_model=ScanTriggerResponse,
    summary="Scanner stoppen",
    description="Stoppt den Auto-Scanner."
)
async def stop_scanner():
    """Auto-Scanner stoppen."""
    try:
        if not scanner_service.is_running:
            return ScanTriggerResponse(
                success=True,
                message="Scanner war nicht aktiv",
                symbols_to_scan=0,
            )

        await scanner_service.stop()

        return ScanTriggerResponse(
            success=True,
            message="Scanner gestoppt",
            symbols_to_scan=0,
        )

    except Exception as e:
        logger.error(f"Fehler beim Stoppen des Scanners: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Stoppen: {str(e)}"
        )


@router.post(
    "/pause",
    response_model=ScanTriggerResponse,
    summary="Scanner pausieren",
    description="Pausiert den Auto-Scanner temporär."
)
async def pause_scanner():
    """Scanner pausieren (temporär)."""
    try:
        await scanner_service.pause()

        return ScanTriggerResponse(
            success=True,
            message="Scanner pausiert",
            symbols_to_scan=len(scanner_service.results),
        )

    except Exception as e:
        logger.error(f"Fehler beim Pausieren: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Pausieren: {str(e)}"
        )


@router.post(
    "/resume",
    response_model=ScanTriggerResponse,
    summary="Scanner fortsetzen",
    description="Setzt einen pausierten Scanner fort."
)
async def resume_scanner():
    """Pausierten Scanner fortsetzen."""
    try:
        await scanner_service.resume()

        return ScanTriggerResponse(
            success=True,
            message="Scanner fortgesetzt",
            symbols_to_scan=len(scanner_service.results),
        )

    except Exception as e:
        logger.error(f"Fehler beim Fortsetzen: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Fortsetzen: {str(e)}"
        )


@router.post(
    "/trigger",
    response_model=ScanTriggerResponse,
    summary="Manuellen Scan auslösen",
    description="Löst einen sofortigen Scan aller Watchlist-Symbole aus."
)
async def trigger_manual_scan():
    """
    Manuellen Scan auslösen.

    Setzt das Scan-Intervall zurück, sodass beim nächsten
    Durchlauf alle Symbole neu gescannt werden.
    """
    try:
        result = await scanner_service.trigger_manual_scan()
        return result

    except Exception as e:
        logger.error(f"Fehler beim manuellen Scan: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Auslösen: {str(e)}"
        )
