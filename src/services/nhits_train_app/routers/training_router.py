"""Training endpoints for NHITS Training Service."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_service import training_service, TrainingStatus

router = APIRouter(prefix="/train", tags=["NHITS Training"])


class TrainingRequest(BaseModel):
    """Request to start training."""
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to train (default: all available)"
    )
    timeframes: Optional[List[str]] = Field(
        default=["H1", "D1"],
        description="Timeframes to train"
    )
    force: bool = Field(
        default=False,
        description="Force retraining even if model exists"
    )


class TrainingResponse(BaseModel):
    """Response for training operations."""
    job_id: str
    status: str
    message: str
    total_models: int = 0


@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """
    Start a new NHITS training job.

    Trains models for specified symbols and timeframes.
    Training runs in background - use /status to check progress.
    """
    try:
        if training_service.is_training():
            current = training_service.get_current_job()
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {current.job_id}"
            )

        job = await training_service.start_training(
            symbols=request.symbols,
            timeframes=request.timeframes,
            force=request.force
        )

        return TrainingResponse(
            job_id=job.job_id,
            status=job.status.value,
            message="Training job started",
            total_models=job.total_models
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
async def cancel_training(job_id: Optional[str] = None):
    """Cancel a running training job."""
    if await training_service.cancel_training(job_id):
        return {"message": "Training cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=400, detail="No active training to cancel")


@router.get("/status")
async def get_training_status():
    """Get current training status."""
    return training_service.get_status()


@router.get("/jobs")
async def list_training_jobs(limit: int = 20):
    """List recent training jobs."""
    jobs = training_service.get_all_jobs(limit)
    return {
        "jobs": [job.to_dict() for job in jobs],
        "total": len(jobs)
    }


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get details of a specific training job."""
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@router.get("/progress")
async def get_training_progress():
    """Get detailed progress of current training."""
    current = training_service.get_current_job()

    if not current:
        return {
            "status": "idle",
            "message": "No training in progress"
        }

    return {
        "job_id": current.job_id,
        "status": current.status.value,
        "progress_percent": current.progress,
        "current_symbol": current.current_symbol,
        "current_timeframe": current.current_timeframe,
        "completed_models": current.completed_models,
        "total_models": current.total_models,
        "successful": current.successful,
        "failed": current.failed,
        "started_at": current.started_at
    }


@router.get("/models")
async def list_models():
    """List available trained models."""
    models = training_service.list_models()
    return {
        "models": models,
        "total": len(models)
    }
