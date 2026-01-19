"""
Revalidation Router - API-Endpoints fuer Prediction-History, Feedback und Backtesting.

Ermoeglicht:
- Abruf und Verwaltung der Prediction-History
- Feedback-Submission und Revalidierung
- Backtesting gegen historische Daten
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from ..services.prediction_history_service import prediction_history_service
from ..services.prediction_feedback_service import prediction_feedback_service, REASON_CATEGORIES
from ..services.backtesting_service import backtesting_service
from ..services.claude_validator_service import claude_prediction_validator_service


router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class FeedbackSubmission(BaseModel):
    """Request fuer Feedback-Submission."""
    prediction_id: str = Field(..., description="ID der Prediction")
    feedback_type: str = Field(..., description="confirmed, corrected, rejected")
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    prediction_timestamp: str = Field(..., description="Zeitpunkt der Prediction")

    # Task-spezifisches Feedback (optional)
    price_feedback: Optional[dict] = Field(
        None,
        description="Feedback zur Preis-Vorhersage",
        json_schema_extra={
            "example": {
                "actual_direction": "bearish",
                "actual_change_percent": -2.5,
                "was_correct": False,
                "reason": "wrong_direction"
            }
        }
    )
    pattern_feedback: Optional[dict] = Field(
        None,
        description="Feedback zur Pattern-Klassifikation",
        json_schema_extra={
            "example": {
                "confirmed_patterns": ["double_top"],
                "missed_patterns": ["head_and_shoulders"],
                "false_patterns": ["ascending_triangle"],
                "reason": "false_positive_pattern"
            }
        }
    )
    regime_feedback: Optional[dict] = Field(
        None,
        description="Feedback zur Regime-Vorhersage",
        json_schema_extra={
            "example": {
                "actual_regime": "bear_trend",
                "was_correct": False,
                "reason": "wrong_regime"
            }
        }
    )

    # Begruendung
    reason_category: Optional[str] = Field(None, description="Begruendungskategorie")
    reason_text: Optional[str] = Field(None, description="Freitext-Begruendung")

    # OHLCV-Daten
    ohlcv_data: Optional[list] = Field(None, description="OHLCV-Daten zum Zeitpunkt")


class RevalidationRequest(BaseModel):
    """Request fuer Revalidierung."""
    feedback_id: str = Field(..., description="Feedback-ID")
    validation_result: str = Field(
        ...,
        description="Ergebnis: correct, still_wrong, now_correct"
    )
    notes: Optional[str] = Field(None, description="Optionale Notizen")
    corrected_data: Optional[dict] = Field(
        None,
        description="Neue Korrekturdaten wenn still_wrong"
    )


class BacktestRequest(BaseModel):
    """Request fuer manuellen Backtest."""
    symbol: Optional[str] = Field(None, description="Symbol-Filter")
    timeframe: Optional[str] = Field(None, description="Timeframe-Filter")
    max_backtests: int = Field(50, ge=1, le=500, description="Max. Anzahl Backtests")


# =============================================================================
# History Endpoints
# =============================================================================

@router.get("/history")
async def get_prediction_history(
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    timeframe: Optional[str] = Query(None, description="Filter nach Timeframe"),
    feedback_status: Optional[str] = Query(None, description="Filter nach Feedback-Status"),
    price_direction: Optional[str] = Query(None, description="Filter nach Preis-Richtung"),
    regime: Optional[str] = Query(None, description="Filter nach Regime"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimale Konfidenz"),
    date_from: Optional[str] = Query(None, description="Ab Datum (ISO format)"),
    date_to: Optional[str] = Query(None, description="Bis Datum (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Max. Ergebnisse")
):
    """
    Hole Prediction-History mit optionalen Filtern.

    Gibt alle gespeicherten CNN-LSTM Predictions zurueck, gefiltert nach
    verschiedenen Kriterien. Jede Prediction enthaelt:
    - Preis-Vorhersagen (Richtung, Forecasts)
    - Pattern-Klassifikationen
    - Regime-Vorhersage
    - Feedback-Status
    """
    try:
        history = prediction_history_service.get_history(
            symbol=symbol,
            timeframe=timeframe,
            feedback_status=feedback_status,
            price_direction=price_direction,
            regime=regime,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )

        return {
            "count": len(history),
            "predictions": history
        }

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/statistics")
async def get_history_statistics():
    """
    Hole Statistiken ueber die Prediction-History.

    Gibt aggregierte Statistiken zurueck:
    - Anzahl Predictions gesamt
    - Verteilung nach Richtung, Regime, Timeframe
    - Feedback-Status Verteilung
    - Durchschnittliche Konfidenz
    """
    try:
        stats = prediction_history_service.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{prediction_id}")
async def get_prediction_by_id(prediction_id: str):
    """
    Hole einzelne Prediction nach ID.

    Gibt die vollstaendigen Details einer spezifischen Prediction zurueck
    inklusive OHLCV-Snapshot falls vorhanden.
    """
    try:
        prediction = prediction_history_service.get_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction nicht gefunden")
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/scan")
async def trigger_prediction_scan():
    """
    Starte manuellen Prediction-Scan.

    Fuehrt Predictions fuer alle konfigurierten Symbole und Timeframes durch
    und speichert sie in der History.
    """
    try:
        new_predictions = await prediction_history_service.scan_and_store_predictions()

        return {
            "status": "completed",
            "new_predictions": new_predictions
        }

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/scan/status")
async def get_scan_status():
    """
    Hole Status des Auto-Scans.

    Gibt zurueck ob der periodische Scan laeuft und das Scan-Intervall.
    """
    try:
        stats = prediction_history_service.get_statistics()
        return {
            "running": stats.get("scan_running", False),
            "interval_seconds": stats.get("scan_interval_seconds", 300),
            "total_predictions": stats.get("total_predictions", 0),
            "last_prediction": stats.get("last_prediction")
        }

    except Exception as e:
        logger.error(f"Error getting scan status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/scan/start")
async def start_auto_scan():
    """Starte automatischen Prediction-Scan."""
    try:
        if prediction_history_service.is_running():
            return {"status": "already_running"}

        await prediction_history_service.start()
        return {"status": "started"}

    except Exception as e:
        logger.error(f"Error starting scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/scan/stop")
async def stop_auto_scan():
    """Stoppe automatischen Prediction-Scan."""
    try:
        if not prediction_history_service.is_running():
            return {"status": "not_running"}

        await prediction_history_service.stop()
        return {"status": "stopped"}

    except Exception as e:
        logger.error(f"Error stopping scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/clear")
async def clear_history():
    """
    Loesche gesamte Prediction-History.

    ACHTUNG: Diese Aktion kann nicht rueckgaengig gemacht werden.
    """
    try:
        prediction_history_service.clear_history()
        return {"status": "cleared"}

    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Feedback Endpoints
# =============================================================================

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """
    Speichere Nutzer-Feedback zu einer Prediction.

    Ermoeglicht die Korrektur oder Bestaetigung von Predictions fuer:
    - Preis-Vorhersagen (Richtung, Magnitude)
    - Pattern-Klassifikation (falsch-positiv, verpasst)
    - Regime-Erkennung (falsches Regime)

    Feedback-Typen:
    - **confirmed**: Prediction war korrekt
    - **corrected**: Prediction war teilweise falsch, Korrektur angegeben
    - **rejected**: Prediction war komplett falsch
    """
    try:
        result = prediction_feedback_service.submit_feedback(
            prediction_id=feedback.prediction_id,
            feedback_type=feedback.feedback_type,
            symbol=feedback.symbol,
            timeframe=feedback.timeframe,
            prediction_timestamp=feedback.prediction_timestamp,
            price_feedback=feedback.price_feedback,
            pattern_feedback=feedback.pattern_feedback,
            regime_feedback=feedback.regime_feedback,
            reason_category=feedback.reason_category,
            reason_text=feedback.reason_text,
            ohlcv_data=feedback.ohlcv_data
        )

        # Update auch den Status in der History
        if result.get("status") == "saved":
            prediction_history_service.update_feedback_status(
                feedback.prediction_id, feedback.feedback_type
            )

        return result

    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback")
async def get_feedback(
    prediction_id: Optional[str] = Query(None, description="Filter nach Prediction-ID"),
    feedback_type: Optional[str] = Query(None, description="Filter nach Feedback-Typ"),
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    limit: int = Query(100, ge=1, le=1000, description="Max. Ergebnisse")
):
    """
    Hole gespeichertes Feedback.

    Gibt alle Feedback-Eintraege zurueck, optional gefiltert.
    """
    try:
        feedback = prediction_feedback_service.get_feedback(
            prediction_id=prediction_id,
            feedback_type=feedback_type,
            symbol=symbol,
            limit=limit
        )

        return {
            "count": len(feedback),
            "feedback": feedback
        }

    except Exception as e:
        logger.error(f"Error getting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/statistics")
async def get_feedback_statistics():
    """
    Hole Feedback-Statistiken.

    Gibt aggregierte Statistiken ueber das Feedback zurueck:
    - Verteilung nach Feedback-Typ
    - Verteilung nach Task
    - Revalidierungs-Status
    """
    try:
        stats = prediction_feedback_service.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/reason-categories")
async def get_reason_categories(
    task: Optional[str] = Query(
        None,
        description="Filter nach Task (price, patterns, regime)"
    )
):
    """
    Hole verfuegbare Begruendungskategorien.

    Gibt alle vordefinierten Begruendungen zurueck, die beim Feedback
    verwendet werden koennen. Optional gefiltert nach Task.
    """
    return prediction_feedback_service.get_reason_categories(task)


@router.delete("/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """Loesche ein Feedback."""
    try:
        result = prediction_feedback_service.delete_feedback(feedback_id)
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Feedback nicht gefunden")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Revalidation Endpoints
# =============================================================================

@router.get("/feedback/pending-revalidation")
async def get_pending_revalidation():
    """
    Hole Predictions zur Revalidierung.

    Gibt alle korrigierten/abgelehnten Predictions zurueck, die nach
    einem Model-Training erneut geprueft werden sollten.

    Workflow:
    1. Nutzer gibt Feedback (corrected/rejected)
    2. Model wird trainiert
    3. /feedback/reset-revalidation wird aufgerufen
    4. Nutzer prueft Predictions erneut via diesen Endpoint
    5. Nutzer markiert als revalidiert via /feedback/revalidate
    """
    try:
        pending = prediction_feedback_service.get_pending_revalidation()

        return {
            "count": len(pending),
            "pending": pending,
            "message": f"{len(pending)} Predictions zur Revalidierung"
        }

    except Exception as e:
        logger.error(f"Error getting pending revalidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/revalidate")
async def mark_as_revalidated(request: RevalidationRequest):
    """
    Markiere Feedback als revalidiert.

    Wird nach erneuter Pruefung einer Prediction aufgerufen um das
    Ergebnis zu dokumentieren.

    Validation Results:
    - **correct**: Das neue Model macht jetzt korrekte Vorhersagen
    - **still_wrong**: Das Model macht immer noch Fehler
    - **now_correct**: Vorher abgelehntes ist jetzt korrekt
    """
    try:
        result = prediction_feedback_service.mark_revalidated(
            feedback_id=request.feedback_id,
            validation_result=request.validation_result,
            notes=request.notes,
            corrected_data=request.corrected_data
        )

        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Feedback nicht gefunden")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking revalidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/reset-revalidation")
async def reset_revalidation():
    """
    Setze Revalidierungs-Status zurueck.

    Wird nach einem neuen Model-Training aufgerufen um alle korrigierten/
    abgelehnten Predictions erneut zur Pruefung freizugeben.

    Dies ermoeglicht das Tracking der Model-Verbesserung ueber Zeit.
    """
    try:
        result = prediction_feedback_service.reset_revalidation()
        return result

    except Exception as e:
        logger.error(f"Error resetting revalidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/revalidation-statistics")
async def get_revalidation_statistics():
    """
    Hole detaillierte Revalidierungs-Statistiken.

    Zeigt wie viele korrigierte Predictions nach Training verbessert wurden:
    - Verbesserungsrate gesamt
    - Verbesserungsrate pro Task (Price, Patterns, Regime)
    """
    try:
        stats = prediction_feedback_service.get_revalidation_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting revalidation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Backtesting Endpoints
# =============================================================================

@router.post("/backtest/run")
async def run_backtests(request: BacktestRequest):
    """
    Fuehre Backtests fuer gespeicherte Predictions durch.

    Vergleicht Predictions automatisch mit den tatsaechlichen Marktdaten
    um die Model-Genauigkeit zu messen.

    Metriken:
    - Price Direction Accuracy
    - Price MAE (Mean Absolute Error)
    - Regime Accuracy
    - Pattern Success Rate
    """
    try:
        # Hole Predictions aus History
        predictions = prediction_history_service.get_history(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.max_backtests * 2  # Extra Buffer
        )

        if not predictions:
            return {
                "status": "no_predictions",
                "message": "Keine Predictions fuer Backtest verfuegbar"
            }

        result = await backtesting_service.run_backtests(
            predictions=predictions,
            max_backtests=request.max_backtests
        )

        return {
            "status": "completed",
            **result
        }

    except Exception as e:
        logger.error(f"Error running backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/results")
async def get_backtest_results(
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    timeframe: Optional[str] = Query(None, description="Filter nach Timeframe"),
    limit: int = Query(100, ge=1, le=1000, description="Max. Ergebnisse")
):
    """
    Hole Backtest-Ergebnisse.

    Gibt einzelne Backtest-Resultate zurueck mit Details zu:
    - Predicted vs Actual Direction
    - Price Errors (1h, 1d)
    - Regime Accuracy
    - Pattern Outcomes
    """
    try:
        results = backtesting_service.get_results(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

        return {
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/summary")
async def get_backtest_summary():
    """
    Hole Backtest-Zusammenfassung.

    Gibt aggregierte Metriken ueber alle Backtests zurueck:
    - Overall Accuracy (Direction, Regime)
    - Mean Absolute Error (Price)
    - Breakdown nach Symbol und Timeframe
    """
    try:
        summary = backtesting_service.get_summary()
        return summary

    except Exception as e:
        logger.error(f"Error getting backtest summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/clear")
async def clear_backtest_results():
    """Loesche alle Backtest-Ergebnisse."""
    try:
        backtesting_service.clear_results()
        return {"status": "cleared"}

    except Exception as e:
        logger.error(f"Error clearing backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Claude Vision Validation Endpoints
# =============================================================================

class ClaudeValidationRequest(BaseModel):
    """Request fuer Claude Vision Validierung."""
    prediction_id: str = Field(..., description="ID der Prediction")
    force: bool = Field(False, description="Validierung erzwingen auch wenn gecached")


@router.post("/validate/claude")
async def validate_with_claude(request: ClaudeValidationRequest):
    """
    Validiere eine Prediction mit Claude Vision API.

    Sendet einen generierten Chart an Claude zur visuellen Analyse:
    - Preis-Richtung plausibel?
    - Patterns erkennbar?
    - Regime korrekt?

    Erfordert konfiguriertes ANTHROPIC_API_KEY.
    """
    try:
        # Hole Prediction aus History
        prediction = prediction_history_service.get_by_id(request.prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction nicht gefunden")

        # Hole OHLCV-Daten
        ohlcv_data = prediction.get("ohlcv_snapshot", [])
        if not ohlcv_data:
            # Fallback: versuche Daten zu holen
            import httpx
            data_service_url = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{data_service_url}/api/v1/db/ohlcv/{prediction.get('symbol')}",
                    params={
                        "timeframe": prediction.get("timeframe"),
                        "limit": 50
                    }
                )
                if response.status_code == 200:
                    ohlcv_data = response.json().get("data", [])

        if not ohlcv_data:
            raise HTTPException(
                status_code=400,
                detail="Keine OHLCV-Daten fuer Validierung verfuegbar"
            )

        # Rekonstruiere Prediction-Format fuer Validator
        prediction_data = {
            "symbol": prediction.get("symbol"),
            "timeframe": prediction.get("timeframe"),
            "timestamp": prediction.get("timestamp"),
            "predictions": {
                "price": {
                    "current": prediction.get("price_current"),
                    "forecast_1h": prediction.get("price_forecast_1h"),
                    "forecast_1d": prediction.get("price_forecast_1d"),
                    "direction": prediction.get("price_direction"),
                    "confidence": prediction.get("price_confidence")
                },
                "patterns": prediction.get("patterns", []),
                "regime": {
                    "current": prediction.get("regime_current"),
                    "probability": prediction.get("regime_probability")
                }
            }
        }

        # Validiere mit Claude
        result = await claude_prediction_validator_service.validate_prediction(
            prediction_id=request.prediction_id,
            prediction=prediction_data,
            ohlcv_data=ohlcv_data,
            force=request.force
        )

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Claude validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/claude/history")
async def get_claude_validation_history(
    symbol: Optional[str] = Query(None, description="Filter nach Symbol"),
    limit: int = Query(50, ge=1, le=500, description="Max. Ergebnisse")
):
    """
    Hole Historie der Claude-Validierungen.

    Zeigt alle durchgefuehrten Claude Vision Validierungen mit:
    - Validierungs-Ergebnis pro Task
    - Begruendungen
    - Konfidenzwerte
    """
    try:
        history = claude_prediction_validator_service.get_validation_history(
            limit=limit,
            symbol=symbol
        )

        return {
            "count": len(history),
            "validations": history
        }

    except Exception as e:
        logger.error(f"Error getting Claude validation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/claude/statistics")
async def get_claude_validation_statistics():
    """
    Hole Statistiken der Claude-Validierungen.

    Gibt aggregierte Metriken zurueck:
    - Agreement Rate (gesamt und pro Task)
    - Durchschnittliche Konfidenz
    - Anzahl Validierungen
    """
    try:
        stats = claude_prediction_validator_service.get_validation_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting Claude validation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/claude/status")
async def get_claude_validator_status():
    """
    Hole Status des Claude Validators.

    Zeigt ob der Service konfiguriert und einsatzbereit ist.
    """
    try:
        status = claude_prediction_validator_service.get_status()
        return status

    except Exception as e:
        logger.error(f"Error getting Claude validator status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import os for environment variable access in validate endpoint
import os
