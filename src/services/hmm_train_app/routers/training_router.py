"""Training endpoints for HMM Training Service."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_service import training_service, TrainingStatus, ModelType
from ..services.scheduler_service import scheduler_service
from ..services.self_learning_service import self_learning_service

router = APIRouter(prefix="/train", tags=["HMM Training"])

# Create separate router for validation/versioning endpoints
validation_router = APIRouter(prefix="/models", tags=["Model Versioning"])

# Create router for scheduling endpoints
scheduler_router = APIRouter(prefix="/schedules", tags=["Training Schedules"])

# Create router for self-learning endpoints
self_learning_router = APIRouter(prefix="/self-learning", tags=["Self-Learning"])


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


@validation_router.get("/metrics-history")
async def get_metrics_history(
    model_type: str = Query(default="hmm", description="Model type: hmm or scorer"),
    limit: int = Query(default=50, ge=10, le=200)
):
    """
    Get historical metrics for tracking model improvement over time.

    Returns aggregated metrics from training jobs, grouped by day.
    Uses both deployment decisions and training history for complete data.
    Useful for visualizing how model performance evolves with each training cycle.
    """
    from collections import defaultdict
    from datetime import datetime

    job_metrics = defaultdict(lambda: {
        "accuracies": [],
        "f1_scores": [],
        "deployed": 0,
        "rejected": 0,
        "total": 0
    })

    # Source 1: Training history (contains validation_metrics directly)
    for job in training_service._jobs.values():
        if job.status.value != "completed":
            continue

        completed_at = job.completed_at
        if not completed_at:
            continue

        try:
            dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            date_key = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        # Extract metrics from validation_metrics
        val_metrics = job.validation_metrics or {}

        for key, metrics in val_metrics.items():
            # Check if this is the right model type
            is_hmm = key.startswith("hmm_")
            is_scorer = key == "scorer"

            if model_type == "hmm" and not is_hmm:
                continue
            if model_type == "scorer" and not is_scorer:
                continue

            if model_type == "hmm":
                accuracy = metrics.get("regime_accuracy", 0)
                f1 = metrics.get("regime_f1_weighted", 0)
            else:
                accuracy = metrics.get("accuracy", 0)
                f1 = metrics.get("f1_weighted", 0)

            if accuracy > 0:
                job_metrics[date_key]["accuracies"].append(accuracy)
            if f1 > 0:
                job_metrics[date_key]["f1_scores"].append(f1)

        # Count deployments from deployment_decisions
        deploy_decisions = job.deployment_decisions or {}
        for key, decision in deploy_decisions.items():
            is_hmm = key.startswith("hmm_")
            is_scorer = key == "scorer"

            if model_type == "hmm" and not is_hmm:
                continue
            if model_type == "scorer" and not is_scorer:
                continue

            job_metrics[date_key]["total"] += 1
            action = decision.get("action", "")
            if action == "deployed":
                job_metrics[date_key]["deployed"] += 1
            elif action in ("rejected", "pending_review"):
                job_metrics[date_key]["rejected"] += 1

    # Source 2: Deployment decisions (fallback for older data without training history)
    if training_service._validation_enabled:
        try:
            rollback_svc = training_service._get_rollback_service()
            history_objects = rollback_svc.get_deployment_history(limit=limit * 2)

            for entry_obj in history_objects:
                entry = entry_obj.to_dict() if hasattr(entry_obj, 'to_dict') else entry_obj

                if entry.get("model_type") != model_type:
                    continue

                comparison = entry.get("comparison_result") or {}
                candidate_metrics = comparison.get("candidate_metrics") or {}

                if not candidate_metrics:
                    continue

                timestamp = entry.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        date_key = dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        continue
                else:
                    continue

                # Only add if not already counted from training history
                if job_metrics[date_key]["total"] == 0:
                    if model_type == "hmm":
                        accuracy = candidate_metrics.get("regime_accuracy", 0)
                        f1 = candidate_metrics.get("regime_f1_weighted", 0)
                    else:
                        accuracy = candidate_metrics.get("accuracy", 0)
                        f1 = candidate_metrics.get("f1_weighted", 0)

                    if accuracy > 0:
                        job_metrics[date_key]["accuracies"].append(accuracy)
                    if f1 > 0:
                        job_metrics[date_key]["f1_scores"].append(f1)

                    job_metrics[date_key]["total"] += 1
                    if entry.get("action") == "deployed":
                        job_metrics[date_key]["deployed"] += 1
                    else:
                        job_metrics[date_key]["rejected"] += 1
        except Exception as e:
            logger.debug(f"Could not load deployment history: {e}")

    # Convert to list sorted by date
    result = []
    for date_key in sorted(job_metrics.keys()):
        metrics = job_metrics[date_key]
        if metrics["accuracies"] or metrics["total"] > 0:
            avg_accuracy = sum(metrics["accuracies"]) / len(metrics["accuracies"]) if metrics["accuracies"] else 0
            avg_f1 = sum(metrics["f1_scores"]) / len(metrics["f1_scores"]) if metrics["f1_scores"] else 0
            deploy_rate = metrics["deployed"] / metrics["total"] if metrics["total"] > 0 else 0

            result.append({
                "date": date_key,
                "avg_accuracy": round(avg_accuracy * 100, 1),
                "avg_f1": round(avg_f1 * 100, 1),
                "deployed": metrics["deployed"],
                "rejected": metrics["rejected"],
                "total": metrics["total"],
                "deploy_rate": round(deploy_rate * 100, 1)
            })

    return {
        "validation_enabled": training_service._validation_enabled,
        "model_type": model_type,
        "history": result[-limit:],  # Return last N entries
        "total_entries": len(result)
    }


# =============================================================================
# Training Schedule Endpoints
# =============================================================================

class ScheduleCreateRequest(BaseModel):
    """Request to create a training schedule."""
    interval: str = Field(
        default="daily",
        description="Schedule interval: 'hourly', 'daily', or 'weekly'"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to train (empty/null = all symbols)"
    )
    timeframe: str = Field(
        default="1h",
        description="Timeframe for training data"
    )
    lookback_days: int = Field(
        default=365,
        ge=30,
        le=730,
        description="Days of historical data to use"
    )
    train_hmm: bool = Field(
        default=True,
        description="Train HMM regime detection models"
    )
    train_scorer: bool = Field(
        default=True,
        description="Train LightGBM signal scorer"
    )
    custom_hour: int = Field(
        default=3,
        ge=0,
        le=23,
        description="Hour of day for daily/weekly schedules (UTC)"
    )
    custom_weekday: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Day of week for weekly schedule (0=Monday)"
    )
    enabled: bool = Field(
        default=True,
        description="Enable schedule immediately"
    )


class ScheduleUpdateRequest(BaseModel):
    """Request to update a training schedule."""
    interval: Optional[str] = Field(
        default=None,
        description="Schedule interval: 'hourly', 'daily', or 'weekly'"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to train (empty list = all symbols)"
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Timeframe for training data"
    )
    lookback_days: Optional[int] = Field(
        default=None,
        ge=30,
        le=730,
        description="Days of historical data to use"
    )
    train_hmm: Optional[bool] = Field(
        default=None,
        description="Train HMM regime detection models"
    )
    train_scorer: Optional[bool] = Field(
        default=None,
        description="Train LightGBM signal scorer"
    )
    custom_hour: Optional[int] = Field(
        default=None,
        ge=0,
        le=23,
        description="Hour of day for daily/weekly schedules (UTC)"
    )
    custom_weekday: Optional[int] = Field(
        default=None,
        ge=0,
        le=6,
        description="Day of week for weekly schedule (0=Monday)"
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="Enable/disable schedule"
    )


class ScheduleResponse(BaseModel):
    """Response for schedule operations."""
    schedule_id: str
    interval: str
    enabled: bool
    symbols: Optional[List[str]]
    timeframe: str
    lookback_days: int
    train_hmm: bool
    train_scorer: bool
    custom_hour: int
    custom_weekday: int
    next_run: Optional[str]
    last_run: Optional[str]
    last_status: Optional[str]


@scheduler_router.get("", summary="List all schedules")
async def list_schedules():
    """
    List all HMM training schedules.

    Returns all configured schedules with their status and next run time.
    """
    schedules = scheduler_service.get_all_schedules()
    return {
        "schedules": [s.to_dict() for s in schedules],
        "total": len(schedules),
        "scheduler_status": scheduler_service.get_status()
    }


@scheduler_router.post("", summary="Create schedule")
async def create_schedule(request: ScheduleCreateRequest):
    """
    Create a new HMM training schedule.

    Intervals:
    - **hourly**: Run every hour at minute 0
    - **daily**: Run once per day at specified hour (UTC)
    - **weekly**: Run once per week on specified weekday and hour (UTC)

    Symbols:
    - Leave empty or null to train ALL symbols from the Data Service
    - Provide a list to train only specific symbols
    """
    try:
        schedule = scheduler_service.create_schedule(
            interval=request.interval,
            symbols=request.symbols,
            timeframe=request.timeframe,
            lookback_days=request.lookback_days,
            train_hmm=request.train_hmm,
            train_scorer=request.train_scorer,
            custom_hour=request.custom_hour,
            custom_weekday=request.custom_weekday,
            enabled=request.enabled
        )

        return {
            "status": "created",
            "schedule": schedule.to_dict(),
            "message": f"Schedule created, next run: {schedule.next_run}"
        }

    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@scheduler_router.get("/{schedule_id}", summary="Get schedule details")
async def get_schedule(schedule_id: str):
    """Get details of a specific schedule."""
    schedule = scheduler_service.get_schedule(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_id}")
    return schedule.to_dict()


@scheduler_router.put("/{schedule_id}", summary="Update schedule")
async def update_schedule(schedule_id: str, request: ScheduleUpdateRequest):
    """
    Update an existing schedule.

    Only provided fields will be updated.
    """
    schedule = scheduler_service.update_schedule(
        schedule_id=schedule_id,
        interval=request.interval,
        symbols=request.symbols,
        timeframe=request.timeframe,
        lookback_days=request.lookback_days,
        train_hmm=request.train_hmm,
        train_scorer=request.train_scorer,
        custom_hour=request.custom_hour,
        custom_weekday=request.custom_weekday,
        enabled=request.enabled
    )

    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_id}")

    return {
        "status": "updated",
        "schedule": schedule.to_dict()
    }


@scheduler_router.delete("/{schedule_id}", summary="Delete schedule")
async def delete_schedule(schedule_id: str):
    """Delete a schedule."""
    if scheduler_service.delete_schedule(schedule_id):
        return {
            "status": "deleted",
            "schedule_id": schedule_id
        }
    else:
        raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_id}")


@scheduler_router.post("/{schedule_id}/toggle", summary="Enable/disable schedule")
async def toggle_schedule(
    schedule_id: str,
    enabled: bool = Query(..., description="Enable (true) or disable (false) the schedule")
):
    """Enable or disable a schedule."""
    schedule = scheduler_service.toggle_schedule(schedule_id, enabled)
    if not schedule:
        raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_id}")

    return {
        "status": "toggled",
        "schedule_id": schedule_id,
        "enabled": schedule.enabled,
        "next_run": schedule.next_run
    }


@scheduler_router.post("/{schedule_id}/trigger", summary="Trigger immediate run")
async def trigger_schedule(schedule_id: str):
    """
    Trigger a schedule to run immediately.

    The training will start in the next scheduler cycle (within 1 minute).
    """
    if not scheduler_service.trigger_now(schedule_id):
        raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_id}")

    return {
        "status": "triggered",
        "schedule_id": schedule_id,
        "message": "Training will start within 1 minute"
    }


@scheduler_router.get("/status/overview", summary="Get scheduler status")
async def get_scheduler_status():
    """
    Get scheduler service status.

    Returns:
    - Whether the scheduler is running
    - Total and active schedules count
    - Next scheduled run information
    """
    return scheduler_service.get_status()


# ==================== Self-Learning Endpoints ====================


class SelfLearningConfigRequest(BaseModel):
    """Request to update self-learning configuration."""
    enabled: Optional[bool] = Field(default=None, description="Enable/disable self-learning monitor")
    check_interval_seconds: Optional[int] = Field(default=None, ge=60, le=3600, description="Check interval (60-3600 seconds)")


@self_learning_router.get("/status", summary="Get self-learning status")
async def get_self_learning_status():
    """
    Get the current status of the self-learning monitor.

    Returns:
    - Whether self-learning is enabled
    - Monitor running state
    - Last accuracy check time and value
    - Last retrain trigger time
    """
    return self_learning_service.status


@self_learning_router.post("/start", summary="Start self-learning monitor")
async def start_self_learning():
    """
    Start the self-learning background monitor.

    The monitor periodically checks accuracy from the HMM Inference service
    and triggers re-training when accuracy drops below threshold.
    """
    await self_learning_service.start_monitor()
    return {
        "status": "started",
        "message": "Self-learning monitor started",
        "check_interval_seconds": self_learning_service._status.check_interval_seconds
    }


@self_learning_router.post("/stop", summary="Stop self-learning monitor")
async def stop_self_learning():
    """
    Stop the self-learning background monitor.
    """
    await self_learning_service.stop_monitor()
    return {
        "status": "stopped",
        "message": "Self-learning monitor stopped"
    }


@self_learning_router.post("/check", summary="Manual accuracy check")
async def manual_accuracy_check():
    """
    Manually trigger an accuracy check.

    Returns the current accuracy status and whether retrain is needed.
    """
    return await self_learning_service.manual_check()


@self_learning_router.post("/trigger-retrain", summary="Force retrain")
async def force_retrain():
    """
    Force an immediate retrain regardless of accuracy.

    Use this to manually trigger the self-learning loop.
    """
    return await self_learning_service.force_retrain()


@self_learning_router.post("/config", summary="Update configuration")
async def update_self_learning_config(request: SelfLearningConfigRequest):
    """
    Update self-learning configuration.

    Parameters:
    - **enabled**: Enable/disable self-learning
    - **check_interval_seconds**: Time between accuracy checks (60-3600 seconds)
    """
    return self_learning_service.set_config(
        enabled=request.enabled,
        check_interval_seconds=request.check_interval_seconds
    )


@self_learning_router.get("/config", summary="Get configuration")
async def get_self_learning_config():
    """
    Get current self-learning configuration.
    """
    return {
        "enabled": self_learning_service._status.enabled,
        "check_interval_seconds": self_learning_service._status.check_interval_seconds,
        "inference_service_url": self_learning_service._status.inference_service_url
    }
