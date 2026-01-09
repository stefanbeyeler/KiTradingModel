"""
Training Router für CNN-LSTM Training Service.

Endpoints für Training-Job-Verwaltung.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models.training_schemas import (
    TrainingHistoryResponse,
    TrainingRequest,
    TrainingStatus,
    TrainingStatusResponse,
)

router = APIRouter()


# =============================================================================
# Training Job Endpoints
# =============================================================================

@router.post("/train", response_model=TrainingStatusResponse, tags=["2. Training"])
async def start_training(request: TrainingRequest):
    """
    Startet einen neuen Training-Job.

    Trainiert das CNN-LSTM Modell mit den angegebenen Symbolen und Timeframes.

    - **symbols**: Liste der zu trainierenden Symbole
    - **timeframes**: Timeframes fuer Training (default: H1, D1)
    - **epochs**: Anzahl Trainings-Epochen
    - **batch_size**: Batch-Groesse
    - **learning_rate**: Lernrate
    - **price_weight**: Gewichtung des Preis-Tasks (0-1)
    - **pattern_weight**: Gewichtung des Pattern-Tasks (0-1)
    - **regime_weight**: Gewichtung des Regime-Tasks (0-1)
    """
    from ..services.training_service import training_service

    if training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    logger.info(f"Starting training for {len(request.symbols)} symbols")
    result = await training_service.start_training(request)

    return result


@router.get("/train/status", response_model=TrainingStatusResponse, tags=["2. Training"])
async def get_training_status():
    """
    Gibt den aktuellen Training-Status zurueck.

    Zeigt Fortschritt, Metriken und eventuelle Fehler.
    """
    from ..services.training_service import training_service

    return training_service.get_current_status()


@router.post("/train/cancel", response_model=TrainingStatusResponse, tags=["2. Training"])
async def cancel_training():
    """
    Bricht laufendes Training ab.

    Das Training wird nach der aktuellen Epoche beendet.
    """
    from ..services.training_service import training_service

    if not training_service.is_training():
        raise HTTPException(
            status_code=400,
            detail="No training in progress"
        )

    logger.info("Cancelling training")
    result = await training_service.cancel_training()

    return result


# =============================================================================
# Training History
# =============================================================================

@router.get("/train/history", response_model=TrainingHistoryResponse, tags=["2. Training"])
async def get_training_history():
    """
    Gibt die Training-Historie zurueck.

    Zeigt vergangene Training-Jobs mit Status und Ergebnissen.
    """
    from ..services.training_service import training_service

    history = training_service.get_history()

    successful = sum(1 for h in history if h.status == TrainingStatus.COMPLETED)
    failed = sum(1 for h in history if h.status == TrainingStatus.FAILED)

    return TrainingHistoryResponse(
        history=history,
        total_jobs=len(history),
        successful_jobs=successful,
        failed_jobs=failed
    )


# =============================================================================
# Quick Training Endpoints
# =============================================================================

@router.post("/train/quick", response_model=TrainingStatusResponse, tags=["2. Training"])
async def quick_training(
    symbols: list[str] = ["BTCUSD", "EURUSD"],
    timeframe: str = "H1",
    epochs: int = 50
):
    """
    Schnelles Training mit Default-Einstellungen.

    Vereinfachter Endpoint fuer schnelle Tests.
    """
    request = TrainingRequest(
        symbols=symbols,
        timeframes=[timeframe],
        epochs=epochs,
        batch_size=32,
        learning_rate=1e-4,
        early_stopping_patience=5
    )

    from ..services.training_service import training_service

    if training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    return await training_service.start_training(request)


@router.post("/train/all-symbols", response_model=TrainingStatusResponse, tags=["2. Training"])
async def train_all_symbols(
    timeframes: list[str] = ["H1", "D1"],
    epochs: int = 100
):
    """
    Trainiert mit allen verfuegbaren Symbolen.

    Holt Symbol-Liste vom Data Service und trainiert alle.
    """
    from ..services.data_pipeline import data_pipeline
    from ..services.training_service import training_service

    if training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    # Hole alle Symbole
    symbols = await data_pipeline.fetch_available_symbols()

    if not symbols:
        raise HTTPException(
            status_code=500,
            detail="Could not fetch symbol list"
        )

    # Limitiere auf max 50 Symbole
    symbols = symbols[:50]

    request = TrainingRequest(
        symbols=symbols,
        timeframes=timeframes,
        epochs=epochs
    )

    return await training_service.start_training(request)


# =============================================================================
# Watchdog Integration
# =============================================================================

@router.post("/train/watchdog", tags=["2. Training"])
async def watchdog_trigger(
    symbols: list[str],
    timeframes: list[str] = ["H1"],
    priority: str = "normal"
):
    """
    Watchdog-Trigger fuer orchestriertes Training.

    Wird vom Watchdog Service aufgerufen um Training zu starten.
    """
    from ..services.training_service import training_service

    if training_service.is_training():
        return {
            "accepted": False,
            "reason": "Training already in progress",
            "status": training_service.get_current_status().status.value
        }

    request = TrainingRequest(
        symbols=symbols,
        timeframes=timeframes,
        epochs=100,
        priority=priority
    )

    result = await training_service.start_training(request)

    return {
        "accepted": True,
        "job_id": result.job_id,
        "status": result.status.value
    }


@router.get("/train/ready", tags=["2. Training"])
async def is_ready_for_training():
    """
    Prueft ob Service bereit fuer Training ist.

    Wird vom Watchdog zur Verfuegbarkeitspruefung verwendet.
    """
    from ..services.training_service import training_service

    return {
        "ready": not training_service.is_training(),
        "current_status": training_service.get_current_status().status.value,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
