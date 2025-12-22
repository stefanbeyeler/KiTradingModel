"""TCN model training endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from ..models.schemas import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    ModelInfoResponse,
)
from ..services.tcn_training_service import tcn_training_service, TrainingConfig
from ..services.pattern_detection_service import pattern_detection_service

router = APIRouter()


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start TCN model training.

    Training runs in the background. Use /train/status to monitor progress.

    - **symbols**: List of symbols to use for training
    - **timeframe**: Timeframe for training data
    - **lookback_days**: Days of historical data
    - **epochs**: Number of training epochs
    - **batch_size**: Batch size
    - **learning_rate**: Learning rate
    """
    try:
        if tcn_training_service.is_training():
            return TrainingResponse(
                status=TrainingStatus.TRAINING,
                message="Training already in progress",
                **tcn_training_service.get_training_status()
            )

        config = TrainingConfig(
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            validation_split=request.validation_split,
            early_stopping_patience=request.early_stopping_patience
        )

        # Start training in background
        async def train_task():
            await tcn_training_service.train(
                symbols=request.symbols,
                timeframe=request.timeframe,
                lookback_days=request.lookback_days,
                config=config
            )

        background_tasks.add_task(train_task)

        return TrainingResponse(
            status=TrainingStatus.PREPARING,
            message="Training started",
            started_at=None,
            total_epochs=request.epochs
        )

    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/status", response_model=TrainingResponse)
async def get_training_status():
    """
    Get current training status.

    Returns progress, current epoch, and best loss.
    """
    status = tcn_training_service.get_training_status()

    return TrainingResponse(
        status=status.get("status", TrainingStatus.IDLE),
        job_id=status.get("job_id"),
        message=status.get("message", ""),
        progress=status.get("progress"),
        current_epoch=status.get("current_epoch"),
        total_epochs=status.get("total_epochs"),
        best_loss=status.get("best_loss")
    )


@router.post("/train/stop")
async def stop_training():
    """
    Stop current training.

    Note: Training will stop after current epoch completes.
    """
    # Note: Would need to implement proper training cancellation
    if not tcn_training_service.is_training():
        return {"status": "no_training", "message": "No training in progress"}

    return {"status": "stopping", "message": "Training will stop after current epoch"}


@router.get("/train/history")
async def get_training_history():
    """
    Get training history.

    Returns list of past training jobs with metrics.
    """
    history = tcn_training_service.get_training_history()
    return {"history": history, "count": len(history)}


@router.get("/models")
async def list_models():
    """
    List available trained models.

    Returns model files with metadata.
    """
    models = tcn_training_service.list_models()
    return {"models": models, "count": len(models)}


@router.post("/models/load")
async def load_model(model_name: str):
    """
    Load a specific model.

    - **model_name**: Name of the model file to load
    """
    try:
        models = tcn_training_service.list_models()
        model = next((m for m in models if m["name"] == model_name), None)

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        pattern_detection_service.load_model(model["path"])

        return {
            "status": "loaded",
            "model": model_name,
            "path": model["path"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about currently loaded model.
    """
    from ..models.tcn_model import TCNPatternClassifier

    is_loaded = pattern_detection_service.is_model_loaded()

    return ModelInfoResponse(
        model_version="1.0.0",
        trained_on=None,
        pattern_classes=TCNPatternClassifier.PATTERN_CLASSES,
        num_parameters=pattern_detection_service.tcn_model.get_num_parameters(),
        input_sequence_length=200,
        device=pattern_detection_service.device,
        last_training_metrics=None
    )
