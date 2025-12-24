"""TCN model training endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..models.schemas import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    ModelInfoResponse,
)
from ..services.tcn_training_service import tcn_training_service, TrainingConfig
from ..services.pattern_detection_service import pattern_detection_service
from ..services.auto_training_scheduler import auto_training_scheduler


class AutoTrainingConfigRequest(BaseModel):
    """Request to update auto-training configuration."""
    enabled: Optional[bool] = None
    interval: Optional[str] = Field(None, pattern="^(daily|weekly|monthly|manual)$")
    timeframes: Optional[List[str]] = None
    lookback_days: Optional[int] = Field(None, ge=30, le=1000)
    epochs: Optional[int] = Field(None, ge=10, le=1000)
    batch_size: Optional[int] = Field(None, ge=8, le=128)
    learning_rate: Optional[float] = Field(None, ge=1e-6, le=0.1)
    min_symbols: Optional[int] = Field(None, ge=1, le=100)


class AutoTrainingRunRequest(BaseModel):
    """Request to manually trigger auto-training."""
    timeframes: Optional[List[str]] = None

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
        best_loss=status.get("best_loss"),
        started_at=status.get("started_at"),
        samples_count=status.get("samples_count")
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


# =============================================================================
# Auto-Training Scheduler Endpoints
# =============================================================================

@router.get("/auto-training/status")
async def get_auto_training_status():
    """
    Get auto-training scheduler status.

    Returns current configuration, last/next run times, and running state.
    """
    return auto_training_scheduler.get_status()


@router.post("/auto-training/config")
async def update_auto_training_config(request: AutoTrainingConfigRequest):
    """
    Update auto-training configuration.

    - **enabled**: Enable/disable automatic training
    - **interval**: Schedule interval (daily, weekly, monthly, manual)
    - **timeframes**: List of timeframes to train
    - **lookback_days**: Days of historical data
    - **epochs**: Training epochs per model
    - **batch_size**: Training batch size
    - **learning_rate**: Learning rate
    - **min_symbols**: Minimum symbols required to start training
    """
    try:
        config = auto_training_scheduler.update_config(
            enabled=request.enabled,
            interval=request.interval,
            timeframes=request.timeframes,
            lookback_days=request.lookback_days,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            min_symbols=request.min_symbols
        )

        return {
            "status": "updated",
            "config": auto_training_scheduler.get_status()
        }

    except Exception as e:
        logger.error(f"Auto-training config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-training/run")
async def run_auto_training(
    background_tasks: BackgroundTasks,
    request: Optional[AutoTrainingRunRequest] = None
):
    """
    Manually trigger auto-training for all symbols.

    - **timeframes**: Optional list of timeframes (defaults to config timeframes)

    Training runs in the background for all available symbols.
    """
    if tcn_training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    timeframes = request.timeframes if request else None

    async def run_training():
        await auto_training_scheduler.run_training_for_all(timeframes)

    background_tasks.add_task(run_training)

    return {
        "status": "started",
        "message": "Auto-training started for all symbols",
        "timeframes": timeframes or auto_training_scheduler.config.timeframes
    }


@router.post("/auto-training/enable")
async def enable_auto_training():
    """Enable automatic training scheduler."""
    auto_training_scheduler.update_config(enabled=True)
    auto_training_scheduler.start()
    return {
        "status": "enabled",
        "next_run": auto_training_scheduler.config.next_run
    }


@router.post("/auto-training/disable")
async def disable_auto_training():
    """Disable automatic training scheduler."""
    auto_training_scheduler.update_config(enabled=False)
    auto_training_scheduler.stop()
    return {"status": "disabled"}


@router.post("/models/cleanup")
async def cleanup_models(keep_count: int = 3):
    """
    Manually cleanup old models.

    - **keep_count**: Number of recent models to keep (default: 3)

    Deletes older models and their history entries.
    """
    if keep_count < 1:
        raise HTTPException(status_code=400, detail="keep_count must be at least 1")

    result = tcn_training_service.cleanup_old_models(keep_count)
    return result


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """
    Delete a specific model.

    - **model_name**: Name of the model file to delete
    """
    import os

    models = tcn_training_service.list_models()
    model = next((m for m in models if m["name"] == model_name), None)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    try:
        os.remove(model["path"])
        logger.info(f"Deleted model: {model_name}")
        return {
            "status": "deleted",
            "model": model_name,
            "freed_mb": model["size_mb"]
        }
    except Exception as e:
        logger.error(f"Failed to delete model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
