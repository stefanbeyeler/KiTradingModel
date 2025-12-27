"""Training endpoints."""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_service import training_service, TrainingStatus
from ..services.training_scheduler import training_scheduler

router = APIRouter()


class TrainingRequest(BaseModel):
    """Request to start training."""
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to train on (default: all available)"
    )
    timeframes: Optional[List[str]] = Field(
        default=None,
        description="Timeframes to use (default: M15, H1, H4, D1)"
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=256,
        description="Batch size"
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0,
        le=0.1,
        description="Learning rate"
    )


class SchedulerConfigUpdate(BaseModel):
    """Update scheduler configuration."""
    enabled: Optional[bool] = None
    interval_hours: Optional[int] = Field(default=None, ge=1, le=168)
    training_hour_utc: Optional[int] = Field(default=None, ge=0, le=23)
    epochs: Optional[int] = Field(default=None, ge=10, le=1000)


@router.post("/train")
async def start_training(request: TrainingRequest):
    """
    Start a new training job.

    Trains a TCN model for candlestick pattern recognition.

    Parameters:
    - **symbols**: List of symbols to train on
    - **timeframes**: Timeframes to include
    - **epochs**: Number of training epochs
    - **batch_size**: Batch size for training
    - **learning_rate**: Learning rate
    """
    try:
        job = await training_service.start_training(
            symbols=request.symbols,
            timeframes=request.timeframes,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
        )

        return {
            "status": "started",
            "job": job.to_dict()
        }

    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/status")
async def get_training_status():
    """
    Get current training status.

    Returns the currently running job or information about recent jobs.
    """
    current = training_service.get_current_job()

    if current:
        return {
            "training": True,
            "current_job": current.to_dict()
        }

    # Get most recent job
    jobs = training_service.get_all_jobs(limit=1)

    return {
        "training": False,
        "last_job": jobs[0].to_dict() if jobs else None
    }


@router.get("/train/progress")
async def get_training_progress():
    """
    Get training progress for the current job.
    """
    current = training_service.get_current_job()

    if not current:
        return {
            "training": False,
            "message": "No training job is running"
        }

    return {
        "training": True,
        "job_id": current.job_id,
        "status": current.status.value,
        "progress": current.progress,
        "current_epoch": current.current_epoch,
        "total_epochs": current.epochs,
        "current_loss": current.current_loss,
        "best_loss": current.best_loss,
    }


@router.post("/train/{job_id}/cancel")
async def cancel_training(job_id: str):
    """
    Cancel a running training job.
    """
    success = await training_service.cancel_training(job_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or not running"
        )

    return {
        "status": "cancelled",
        "job_id": job_id
    }


@router.get("/train/jobs")
async def get_training_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None)
):
    """
    Get training job history.

    Parameters:
    - **limit**: Maximum number of jobs to return
    - **status**: Filter by status (pending, running, completed, failed, cancelled)
    """
    jobs = training_service.get_all_jobs(limit=limit)

    if status:
        try:
            status_filter = TrainingStatus(status.lower())
            jobs = [j for j in jobs if j.status == status_filter]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    return {
        "count": len(jobs),
        "jobs": [j.to_dict() for j in jobs]
    }


@router.get("/train/jobs/{job_id}")
async def get_training_job(job_id: str):
    """
    Get details for a specific training job.
    """
    job = training_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return job.to_dict()


@router.get("/model")
async def get_model_info():
    """
    Get information about the current model.
    """
    latest_model = training_service.get_latest_model()

    return {
        "model_available": latest_model is not None,
        "model_path": latest_model,
        "training_in_progress": training_service.is_training(),
    }


@router.get("/scheduler")
async def get_scheduler_config():
    """
    Get scheduler configuration.
    """
    return training_scheduler.get_config()


@router.post("/scheduler")
async def update_scheduler_config(config: SchedulerConfigUpdate):
    """
    Update scheduler configuration.

    Parameters:
    - **enabled**: Enable/disable scheduled training
    - **interval_hours**: Hours between training runs
    - **training_hour_utc**: Hour (UTC) to run training
    - **epochs**: Number of epochs for scheduled training
    """
    training_scheduler.update_config(
        enabled=config.enabled,
        interval_hours=config.interval_hours,
        training_hour=config.training_hour_utc,
        epochs=config.epochs,
    )

    return training_scheduler.get_config()


@router.post("/scheduler/start")
async def start_scheduler():
    """
    Start the training scheduler.
    """
    if training_scheduler.is_running():
        return {
            "status": "already_running",
            "config": training_scheduler.get_config()
        }

    await training_scheduler.start()

    return {
        "status": "started",
        "config": training_scheduler.get_config()
    }


@router.post("/scheduler/stop")
async def stop_scheduler():
    """
    Stop the training scheduler.
    """
    if not training_scheduler.is_running():
        return {
            "status": "not_running",
            "config": training_scheduler.get_config()
        }

    await training_scheduler.stop()

    return {
        "status": "stopped",
        "config": training_scheduler.get_config()
    }
