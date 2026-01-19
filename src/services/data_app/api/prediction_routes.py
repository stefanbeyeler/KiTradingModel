"""Prediction History API Routes.

This module provides endpoints for managing prediction history across all microservices:
- Store predictions from NHITS, TCN, HMM, Candlestick, CNN-LSTM, Workplace
- Retrieve prediction history with filtering
- Evaluate predictions against actual outcomes
- Get accuracy statistics
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from ..services.prediction_history_service import (
    prediction_history_service,
    PredictionCreate,
    PredictionEvaluation,
    PredictionResult,
    PredictionSummary,
    PredictionStats,
)


router = APIRouter(prefix="/predictions", tags=["15. Prediction History"])


# ==================== Request/Response Models ====================


class CreatePredictionRequest(BaseModel):
    """Request to create a new prediction."""

    service: str  # nhits, tcn, hmm, candlestick, cnn-lstm, workplace
    symbol: str
    timeframe: str
    prediction_type: str  # price, direction, pattern, regime, signal
    prediction: dict  # The actual prediction data
    confidence: Optional[float] = None
    target_time: Optional[str] = None  # ISO format datetime
    horizon: Optional[str] = None  # 1h, 4h, 1d, etc.
    model_version: Optional[str] = None
    model_params: Optional[dict] = None
    input_features: Optional[dict] = None
    triggered_by: str = "api"
    tags: Optional[list[str]] = None
    notes: Optional[str] = None


class EvaluatePredictionRequest(BaseModel):
    """Request to evaluate a prediction."""

    actual_outcome: dict
    is_correct: Optional[bool] = None
    accuracy_score: Optional[float] = None
    error_amount: Optional[float] = None


class PredictionListResponse(BaseModel):
    """Response for prediction list."""

    total: int
    predictions: list[PredictionSummary]


# ==================== Endpoints ====================


@router.post("/", response_model=dict)
async def create_prediction(request: CreatePredictionRequest):
    """
    Eine neue Vorhersage speichern.

    Speichert Vorhersagen von allen Microservices (NHITS, TCN, HMM, etc.)
    für spätere Evaluierung und Analyse.
    """
    from datetime import datetime

    target_time = None
    if request.target_time:
        try:
            target_time = datetime.fromisoformat(request.target_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid target_time format")

    data = PredictionCreate(
        service=request.service,
        symbol=request.symbol,
        timeframe=request.timeframe,
        prediction_type=request.prediction_type,
        prediction=request.prediction,
        confidence=request.confidence,
        target_time=target_time,
        horizon=request.horizon,
        model_version=request.model_version,
        model_params=request.model_params,
        input_features=request.input_features,
        triggered_by=request.triggered_by,
        tags=request.tags,
        notes=request.notes,
    )

    prediction_id = await prediction_history_service.create_prediction(data)

    if not prediction_id:
        raise HTTPException(
            status_code=503,
            detail="Prediction history service not available",
        )

    return {
        "status": "ok",
        "prediction_id": prediction_id,
        "message": f"Prediction stored for {request.service}/{request.symbol}",
    }


@router.put("/{prediction_id}/evaluate")
async def evaluate_prediction(prediction_id: str, request: EvaluatePredictionRequest):
    """
    Eine Vorhersage mit dem tatsächlichen Ergebnis evaluieren.

    Speichert das tatsächliche Ergebnis und berechnet die Genauigkeit.
    """
    evaluation = PredictionEvaluation(
        actual_outcome=request.actual_outcome,
        is_correct=request.is_correct,
        accuracy_score=request.accuracy_score,
        error_amount=request.error_amount,
    )

    success = await prediction_history_service.evaluate_prediction(prediction_id, evaluation)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {prediction_id} not found or evaluation failed",
        )

    return {
        "status": "ok",
        "prediction_id": prediction_id,
        "message": "Prediction evaluated",
    }


@router.get("/", response_model=list[PredictionSummary])
async def list_predictions(
    service: Optional[str] = Query(default=None, description="Filter nach Service (nhits, tcn, etc.)"),
    symbol: Optional[str] = Query(default=None, description="Filter nach Symbol"),
    timeframe: Optional[str] = Query(default=None, description="Filter nach Timeframe"),
    prediction_type: Optional[str] = Query(default=None, description="Filter nach Typ (price, direction, etc.)"),
    evaluated_only: bool = Query(default=False, description="Nur evaluierte Vorhersagen"),
    limit: int = Query(default=50, ge=1, le=500, description="Max. Anzahl Einträge"),
    offset: int = Query(default=0, ge=0, description="Offset für Paginierung"),
):
    """
    Vorhersagen auflisten.

    Gibt eine Liste aller Vorhersagen zurück, sortiert nach Datum (neueste zuerst).
    """
    predictions = await prediction_history_service.list_predictions(
        service=service,
        symbol=symbol,
        timeframe=timeframe,
        prediction_type=prediction_type,
        evaluated_only=evaluated_only,
        limit=limit,
        offset=offset,
    )
    return predictions


@router.get("/due-for-evaluation", response_model=list[PredictionSummary])
async def get_predictions_due_for_evaluation(
    service: Optional[str] = Query(default=None, description="Filter nach Service"),
    limit: int = Query(default=100, ge=1, le=500, description="Max. Anzahl"),
):
    """
    Vorhersagen abrufen, die zur Evaluierung fällig sind.

    Gibt Vorhersagen zurück, deren target_time erreicht wurde,
    aber noch nicht evaluiert wurden.
    """
    predictions = await prediction_history_service.get_predictions_due_for_evaluation(
        service=service,
        limit=limit,
    )
    return predictions


@router.get("/stats", response_model=PredictionStats)
async def get_prediction_stats(
    service: Optional[str] = Query(default=None, description="Filter nach Service"),
    symbol: Optional[str] = Query(default=None, description="Filter nach Symbol"),
):
    """
    Statistiken über Vorhersagen abrufen.

    Zeigt Gesamtzahlen, Evaluierungen und Genauigkeit.
    """
    stats = await prediction_history_service.get_stats(
        service=service,
        symbol=symbol,
    )
    return stats


@router.get("/{prediction_id}", response_model=PredictionResult)
async def get_prediction(prediction_id: str):
    """
    Details einer spezifischen Vorhersage abrufen.
    """
    prediction = await prediction_history_service.get_prediction(prediction_id)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {prediction_id} not found",
        )

    return prediction


@router.delete("/{prediction_id}")
async def delete_prediction(prediction_id: str):
    """
    Eine Vorhersage löschen.
    """
    success = await prediction_history_service.delete_prediction(prediction_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {prediction_id} not found",
        )

    return {
        "status": "ok",
        "message": f"Prediction {prediction_id} deleted",
    }


@router.post("/cleanup")
async def cleanup_old_predictions(
    days_to_keep: int = Query(default=90, ge=7, le=365, description="Tage zu behalten"),
):
    """
    Alte Vorhersagen bereinigen.

    Löscht Vorhersagen älter als die angegebene Anzahl Tage.
    """
    deleted = await prediction_history_service.cleanup_old_predictions(days_to_keep=days_to_keep)

    return {
        "status": "ok",
        "deleted_count": deleted,
        "message": f"{deleted} old predictions deleted",
    }
