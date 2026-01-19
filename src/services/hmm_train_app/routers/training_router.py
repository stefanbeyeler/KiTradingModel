"""Training endpoints for HMM Training Service."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_service import training_service, TrainingStatus, ModelType

router = APIRouter(prefix="/train", tags=["HMM Training"])

# Create separate router for validation/versioning endpoints
validation_router = APIRouter(prefix="/models", tags=["Model Versioning"])


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


# =============================================================================
# Model Versioning & Validation Endpoints
# =============================================================================

class RollbackRequest(BaseModel):
    """Request to rollback a model."""
    model_type: str = Field(..., description="Model type: 'hmm' or 'scorer'")
    symbol: Optional[str] = Field(default=None, description="Symbol for HMM models")
    target_version: str = Field(..., description="Version ID to rollback to")
    reason: str = Field(default="Manual rollback", description="Reason for rollback")


class ForceDeployRequest(BaseModel):
    """Request to force deploy a model version."""
    model_type: str = Field(..., description="Model type: 'hmm' or 'scorer'")
    symbol: Optional[str] = Field(default=None, description="Symbol for HMM models")
    reason: str = Field(default="Manual deployment", description="Reason for deployment")


@validation_router.get("/versions")
async def list_model_versions(
    model_type: Optional[str] = Query(default=None, description="Filter by model type"),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results")
):
    """
    List all model versions with their status.

    Returns version history for trained models including:
    - Validation metrics
    - Deployment status (candidate, production, archived, rejected)
    - Training information
    """
    versions = training_service.get_model_versions(
        model_type=model_type,
        symbol=symbol,
        limit=limit
    )
    return {
        "versions": versions,
        "total": len(versions),
        "validation_enabled": training_service._validation_enabled
    }


@validation_router.get("/versions/{version_id}")
async def get_version_details(version_id: str):
    """Get detailed information about a specific version."""
    if not training_service._validation_enabled:
        raise HTTPException(status_code=400, detail="Validation not enabled")

    registry = training_service._get_registry()
    version_data = registry.get_version(version_id)

    if not version_data:
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")

    return {
        "version_id": version_id,
        "models": {k: v.to_dict() for k, v in version_data.items()}
    }


@validation_router.get("/production")
async def get_production_models():
    """
    Get all currently deployed production models.

    Returns the active model for each symbol/type combination.
    """
    production = training_service.get_production_models()
    return {
        "production_models": production,
        "total": len(production),
        "validation_enabled": training_service._validation_enabled
    }


@validation_router.post("/rollback")
async def rollback_model(request: RollbackRequest):
    """
    Manually rollback to a previous model version.

    This will:
    1. Update the production symlink to the target version
    2. Notify the inference service to reload the model
    3. Record the rollback in deployment history

    Use this when a deployed model is not performing well.
    """
    if not training_service._validation_enabled:
        raise HTTPException(status_code=400, detail="Validation not enabled")

    result = await training_service.rollback_model(
        model_type=request.model_type,
        symbol=request.symbol,
        target_version=request.target_version,
        reason=request.reason
    )

    if result.get("action") == "rolled_back":
        return {
            "status": "success",
            "message": f"Rolled back to version {request.target_version}",
            "decision": result
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Rollback failed: {result.get('reason', 'Unknown error')}"
        )


@validation_router.post("/force-deploy/{version_id}")
async def force_deploy_model(version_id: str, request: ForceDeployRequest):
    """
    Force deploy a model version (bypasses A/B validation).

    **Use with caution** - this skips quality validation.
    Only use when you're certain the model should be deployed.
    """
    if not training_service._validation_enabled:
        raise HTTPException(status_code=400, detail="Validation not enabled")

    result = await training_service.force_deploy_version(
        version_id=version_id,
        model_type=request.model_type,
        symbol=request.symbol,
        reason=request.reason
    )

    if result.get("action") == "deployed":
        return {
            "status": "success",
            "message": f"Force deployed version {version_id}",
            "decision": result
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Force deploy failed: {result.get('reason', 'Unknown error')}"
        )


@validation_router.get("/deployment-history")
async def get_deployment_history(
    limit: int = Query(default=50, ge=1, le=200, description="Max results")
):
    """
    Get deployment decision history.

    Shows all deployment decisions including:
    - Successful deployments
    - Rejections with reasons
    - Rollbacks
    - A/B comparison results
    """
    history = training_service.get_deployment_history(limit=limit)
    return {
        "history": history,
        "total": len(history)
    }


@validation_router.get("/comparison-history")
async def get_comparison_history(
    limit: int = Query(default=50, ge=1, le=200, description="Max results")
):
    """
    Get A/B comparison history.

    Shows detailed comparison results between candidate and production models.
    """
    if not training_service._validation_enabled:
        raise HTTPException(status_code=400, detail="Validation not enabled")

    rollback_svc = training_service._get_rollback_service()
    comparisons = rollback_svc.get_comparison_history(limit=limit)

    return {
        "comparisons": comparisons,
        "total": len(comparisons)
    }


@validation_router.get("/validation-metrics/{job_id}")
async def get_validation_metrics(job_id: str):
    """
    Get validation metrics for a specific training job.

    Returns metrics calculated during the validation pipeline.
    """
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return {
        "job_id": job_id,
        "validation_metrics": job.validation_metrics or {},
        "deployment_decisions": job.deployment_decisions or {},
        "deployed_count": job.deployed_count,
        "rejected_count": job.rejected_count,
        "version_id": job.version_id
    }


@validation_router.get("/stats")
async def get_registry_stats():
    """Get model registry statistics."""
    if not training_service._validation_enabled:
        return {
            "validation_enabled": False,
            "message": "Validation pipeline not enabled"
        }

    registry = training_service._get_registry()
    rollback_svc = training_service._get_rollback_service()

    return {
        "validation_enabled": True,
        "registry": registry.get_stats(),
        "deployment": rollback_svc.get_stats()
    }
