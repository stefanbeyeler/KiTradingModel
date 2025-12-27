"""Training endpoints for HMM Training Service."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_service import training_service, TrainingStatus, ModelType

router = APIRouter(prefix="/train", tags=["HMM Training"])


class TrainingRequest(BaseModel):
    """Request to start training."""
    model_type: str = Field(
        default="both",
        description="Type of model to train: 'hmm', 'scorer', or 'both'"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to train (default: all available)"
    )
    timeframe: str = Field(
        default="1h",
        description="Timeframe for training data"
    )
    lookback_days: int = Field(
        default=365,
        description="Days of historical data to use"
    )


class TrainingResponse(BaseModel):
    """Response for training operations."""
    job_id: str
    status: str
    model_type: str
    message: str
    total_models: int = 0


@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """
    Start a new HMM/Scorer training job.

    Model Types:
    - **hmm**: Train HMM regime detection models for each symbol
    - **scorer**: Train LightGBM signal scorer across all symbols
    - **both**: Train both model types

    Training runs in background - use /status to check progress.
    """
    try:
        if training_service.is_training():
            current = training_service.get_current_job()
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {current.job_id}"
            )

        # Parse model type
        try:
            model_type = ModelType(request.model_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid model_type: {request.model_type}. Use 'hmm', 'scorer', or 'both'"
            )

        job = await training_service.start_training(
            model_type=model_type,
            symbols=request.symbols,
            timeframe=request.timeframe,
            lookback_days=request.lookback_days
        )

        return TrainingResponse(
            job_id=job.job_id,
            status=job.status.value,
            model_type=job.model_type.value,
            message="Training job started",
            total_models=job.total_models
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hmm")
async def train_hmm_only(symbols: Optional[List[str]] = None):
    """Start HMM-only training (shortcut)."""
    if training_service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    job = await training_service.start_training(
        model_type=ModelType.HMM,
        symbols=symbols
    )
    return {"job_id": job.job_id, "status": job.status.value, "message": "HMM training started"}


@router.post("/scorer")
async def train_scorer_only(symbols: Optional[List[str]] = None):
    """Start Scorer-only training (shortcut)."""
    if training_service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    job = await training_service.start_training(
        model_type=ModelType.SCORER,
        symbols=symbols
    )
    return {"job_id": job.job_id, "status": job.status.value, "message": "Scorer training started"}


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
        "model_type": current.model_type.value,
        "progress_percent": current.progress,
        "current_symbol": current.current_symbol,
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
