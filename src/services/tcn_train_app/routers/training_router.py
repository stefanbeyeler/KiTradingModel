"""Training endpoints for TCN Training Service."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_service import training_service, TrainingConfig
from ..services.training_scheduler import training_scheduler

router = APIRouter()


class TrainRequest(BaseModel):
    """Training request model."""
    symbols: Optional[List[str]] = Field(
        None,
        description="Symbols to train on. If not provided, uses all available symbols."
    )
    timeframe: str = Field("1h", description="Timeframe for training data")
    lookback_days: int = Field(365, description="Days of historical data")
    epochs: int = Field(100, description="Training epochs")
    batch_size: int = Field(32, description="Batch size")
    learning_rate: float = Field(1e-4, description="Learning rate")


class SchedulerConfigRequest(BaseModel):
    """Scheduler configuration request."""
    enabled: Optional[bool] = None
    interval: Optional[str] = Field(None, description="daily, weekly, monthly, or manual")
    timeframes: Optional[List[str]] = None
    lookback_days: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    min_symbols: Optional[int] = None


@router.post("/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start model training.

    Training runs in background. Use /train/status to monitor progress.
    """
    if training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    # If no symbols provided, fetch all available
    symbols = request.symbols
    if not symbols:
        symbols = await training_scheduler._get_available_symbols()
        if not symbols:
            raise HTTPException(
                status_code=400,
                detail="No symbols available for training"
            )

    config = TrainingConfig(
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate
    )

    # Start training in background
    async def run_training():
        await training_service.train(
            symbols=symbols,
            timeframe=request.timeframe,
            lookback_days=request.lookback_days,
            config=config
        )

    background_tasks.add_task(run_training)

    return {
        "status": "started",
        "message": "Training started in background",
        "symbols_count": len(symbols),
        "timeframe": request.timeframe
    }


@router.get("/train/status")
async def get_training_status():
    """Get current training status and progress."""
    return training_service.get_training_status()


@router.get("/train/history")
async def get_training_history():
    """Get training history."""
    history = training_service.get_training_history()
    return {"history": history}


@router.get("/models")
async def list_models():
    """List available trained models."""
    models = training_service.list_models()
    return {"models": models, "count": len(models)}


@router.delete("/models/cleanup")
async def cleanup_models(keep_count: int = 3):
    """
    Cleanup old models.

    Args:
        keep_count: Number of recent models to keep
    """
    return training_service.cleanup_old_models(keep_count)


@router.get("/scheduler")
async def get_scheduler_status():
    """Get auto-training scheduler status."""
    return training_scheduler.get_status()


@router.put("/scheduler/config")
async def update_scheduler_config(config: SchedulerConfigRequest):
    """Update scheduler configuration."""
    updated = training_scheduler.update_config(
        enabled=config.enabled,
        interval=config.interval,
        timeframes=config.timeframes,
        lookback_days=config.lookback_days,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        min_symbols=config.min_symbols
    )

    return {
        "status": "updated",
        "config": {
            "enabled": updated.enabled,
            "interval": updated.interval.value,
            "timeframes": updated.timeframes,
            "lookback_days": updated.lookback_days,
            "epochs": updated.epochs,
            "batch_size": updated.batch_size,
            "learning_rate": updated.learning_rate,
            "min_symbols": updated.min_symbols,
            "next_run": updated.next_run
        }
    }


@router.post("/scheduler/run-now")
async def run_training_now(background_tasks: BackgroundTasks):
    """Trigger immediate training for all symbols."""
    if training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    async def run():
        await training_scheduler.run_training_for_all()

    background_tasks.add_task(run)

    return {
        "status": "started",
        "message": "Scheduled training started in background"
    }


@router.post("/train/stop")
async def stop_training():
    """Stop current training."""
    if not training_service.is_training():
        return {
            "status": "not_running",
            "message": "No training in progress"
        }

    # Signal training to stop
    training_service.request_stop()

    return {
        "status": "stopping",
        "message": "Training stop requested"
    }


# =============================================================================
# Auto-Training Aliases (for Frontend compatibility)
# Frontend expects /auto-training/* but backend has /scheduler/*
# =============================================================================

@router.get("/auto-training/status")
async def get_auto_training_status():
    """Get auto-training status (alias for /scheduler)."""
    return training_scheduler.get_status()


@router.post("/auto-training/enable")
async def enable_auto_training():
    """Enable auto-training."""
    training_scheduler.update_config(enabled=True)
    return {
        "status": "enabled",
        "message": "Auto-training enabled",
        "config": training_scheduler.get_status()
    }


@router.post("/auto-training/disable")
async def disable_auto_training():
    """Disable auto-training."""
    training_scheduler.update_config(enabled=False)
    return {
        "status": "disabled",
        "message": "Auto-training disabled",
        "config": training_scheduler.get_status()
    }


class AutoTrainingConfigRequest(BaseModel):
    """Auto-training configuration request."""
    interval: Optional[str] = None
    timeframes: Optional[List[str]] = None


@router.post("/auto-training/config")
async def update_auto_training_config(config: AutoTrainingConfigRequest):
    """Update auto-training configuration."""
    training_scheduler.update_config(
        interval=config.interval,
        timeframes=config.timeframes
    )
    return {
        "status": "updated",
        "config": training_scheduler.get_status()
    }


class AutoTrainingRunRequest(BaseModel):
    """Auto-training run request."""
    timeframes: Optional[List[str]] = None


@router.post("/auto-training/run")
async def run_auto_training(request: AutoTrainingRunRequest, background_tasks: BackgroundTasks):
    """Trigger immediate auto-training run."""
    if training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    async def run():
        await training_scheduler.run_training_for_all(timeframes=request.timeframes)

    background_tasks.add_task(run)

    return {
        "status": "started",
        "message": "Auto-training started in background"
    }
