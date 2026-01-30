"""
Scan Router - Scanner-Kontrolle.

Endpoints zum Starten, Stoppen und Überwachen des Auto-Scanners.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.schemas import (
    ScanStatusResponse,
    ScanTriggerResponse,
)
from ..services.scanner_service import scanner_service
from ..services.setup_recorder_service import setup_recorder

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


@router.get(
    "/accuracy",
    summary="Setup-Trefferquote",
    description="Gibt die Trefferquote der aufgezeichneten Trading-Setups zurück."
)
async def get_accuracy_stats(
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    days: int = Query(30, ge=1, le=365, description="Zeitraum in Tagen")
):
    """
    Trefferquote der Trading-Setups abrufen.

    Zeigt wie viele der vorgeschlagenen Setups korrekt waren,
    basierend auf dem tatsächlichen Marktverlauf nach Ablauf des Zeithorizonts.
    """
    try:
        stats = await setup_recorder.get_accuracy_stats(symbol=symbol, days=days)

        return {
            "symbol": symbol,
            "period_days": days,
            "total_predictions": stats.get("total_predictions", 0),
            "evaluated_count": stats.get("evaluated_count", 0),
            "correct_count": stats.get("correct_count", 0),
            "accuracy_percent": stats.get("accuracy_percent"),
            "avg_confidence": stats.get("avg_confidence"),
            "service": "workplace",
        }

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Trefferquote: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler: {str(e)}"
        )


@router.post(
    "/evaluate",
    summary="Evaluation manuell auslösen",
    description="Löst die Evaluation ausstehender Setups manuell aus."
)
async def trigger_evaluation():
    """
    Manuelle Evaluation auslösen.

    Evaluiert alle Setups, deren Zeithorizont abgelaufen ist,
    gegen den tatsächlichen Marktverlauf.
    """
    try:
        stats = await setup_recorder.evaluate_pending_setups()

        return {
            "success": True,
            "evaluated": stats.get("evaluated", 0),
            "correct": stats.get("correct", 0),
            "incorrect": stats.get("incorrect", 0),
            "errors": stats.get("errors", 0),
            "accuracy_percent": (
                stats["correct"] / stats["evaluated"] * 100
                if stats.get("evaluated", 0) > 0 else None
            ),
        }

    except Exception as e:
        logger.error(f"Fehler bei manueller Evaluation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler: {str(e)}"
        )


@router.get(
    "/analytics",
    summary="Setup-Analytics",
    description="Detaillierte Auswertung der Trading-Setups nach verschiedenen Kriterien."
)
async def get_setup_analytics(
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    timeframe: Optional[str] = Query(None, description="Filter nach Timeframe"),
    days: int = Query(90, ge=1, le=365, description="Zeitraum in Tagen")
):
    """
    Detaillierte Analytics für Trading-Setups.

    Analysiert Erfolgsraten nach:
    - Symbol
    - Timeframe
    - Richtung (Long/Short)
    - Score-Bereichen
    - Signal-Alignment
    - Anzahl aktiver Signale
    - Einzelne Signal-Performance
    - Key Drivers
    """
    try:
        analytics = await setup_recorder.get_analytics(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
        )
        return analytics

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler: {str(e)}"
        )


@router.delete(
    "/cleanup-invalid",
    summary="Ungültige Setups bereinigen",
    description="Löscht Setups ohne Entry-Preis, die nicht evaluiert werden können."
)
async def cleanup_invalid_predictions():
    """
    Bereinigt Prediction History.

    Löscht alle Workplace-Setups die keinen Entry-Preis haben
    und daher nicht evaluiert werden können.
    """
    try:
        result = await setup_recorder.cleanup_invalid_predictions()

        return {
            "success": True,
            "checked": result.get("checked", 0),
            "deleted": result.get("deleted", 0),
            "message": f"{result.get('deleted', 0)} ungültige Setups gelöscht"
        }

    except Exception as e:
        logger.error(f"Fehler beim Bereinigen: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler: {str(e)}"
        )
