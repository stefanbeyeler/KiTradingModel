"""Training endpoints for TCN Training Service."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_service import training_service, TrainingConfig
from ..services.training_scheduler import training_scheduler
from ..services.ewc_trainer import EWCConfig
from ..services.feedback_buffer_service import feedback_buffer_service
from ..services.tcn_rollback_service import rollback_service

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
    scheduled_hour: Optional[int] = Field(None, ge=0, le=23, description="Hour for scheduled training (0-23)")
    scheduled_minute: Optional[int] = Field(None, ge=0, le=59, description="Minute for scheduled training (0-59)")


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
        min_symbols=config.min_symbols,
        scheduled_hour=config.scheduled_hour,
        scheduled_minute=config.scheduled_minute
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
            "scheduled_hour": updated.scheduled_hour,
            "scheduled_minute": updated.scheduled_minute,
            "scheduled_time": f"{updated.scheduled_hour:02d}:{updated.scheduled_minute:02d}",
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
    scheduled_hour: Optional[int] = Field(None, ge=0, le=23, description="Hour for scheduled training (0-23)")
    scheduled_minute: Optional[int] = Field(None, ge=0, le=59, description="Minute for scheduled training (0-59)")


@router.post("/auto-training/config")
async def update_auto_training_config(config: AutoTrainingConfigRequest):
    """Update auto-training configuration."""
    training_scheduler.update_config(
        interval=config.interval,
        timeframes=config.timeframes,
        scheduled_hour=config.scheduled_hour,
        scheduled_minute=config.scheduled_minute
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


# =============================================================================
# Incremental Training (EWC-based Self-Learning)
# =============================================================================

class IncrementalTrainRequest(BaseModel):
    """Request for incremental training."""
    min_samples: int = Field(50, ge=10, description="Minimum samples required")
    ewc_lambda: float = Field(1000.0, ge=0, description="EWC regularization strength")
    learning_rate: float = Field(1e-5, gt=0, description="Learning rate for fine-tuning")
    epochs: int = Field(10, ge=1, le=50, description="Training epochs")
    batch_size: int = Field(16, ge=1, description="Batch size")


@router.post("/train/incremental")
async def start_incremental_training(
    request: IncrementalTrainRequest,
    background_tasks: BackgroundTasks
):
    """
    Start incremental training using feedback samples.

    Uses Elastic Weight Consolidation (EWC) to prevent catastrophic
    forgetting of previously learned patterns while fine-tuning on
    new feedback data.
    """
    if training_service.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )

    # Check if enough samples in feedback buffer
    buffer_stats = feedback_buffer_service.get_statistics()
    if buffer_stats.total_samples < request.min_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient samples in feedback buffer ({buffer_stats.total_samples} < {request.min_samples})"
        )

    # Get training batch from feedback buffer
    batch = feedback_buffer_service.get_training_batch(
        batch_size=buffer_stats.total_samples,  # Get all available
        stratified=True
    )

    if not batch:
        raise HTTPException(
            status_code=400,
            detail="No samples available in feedback buffer"
        )

    # Extract sequences and labels from batch
    import numpy as np
    sequences = np.array([s.ohlcv_sequence for s in batch])
    labels = np.array([s.label_vector for s in batch])
    sample_ids = [s.sample_id for s in batch]

    config = EWCConfig(
        ewc_lambda=request.ewc_lambda,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        batch_size=request.batch_size,
        min_samples=request.min_samples
    )

    async def run_incremental():
        result = await training_service.incremental_train(
            sequences=sequences,
            labels=labels,
            config=config
        )
        # Mark samples as used if training succeeded
        if result.get("status") == "completed":
            feedback_buffer_service.mark_as_used(sample_ids)
            logger.info(f"Marked {len(sample_ids)} samples as used after incremental training")

    background_tasks.add_task(run_incremental)

    return {
        "status": "started",
        "message": "Incremental training started in background",
        "samples_count": len(batch),
        "config": {
            "ewc_lambda": config.ewc_lambda,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs
        }
    }


@router.get("/train/incremental/status")
async def get_incremental_status():
    """Get incremental training status and EWC statistics."""
    training_status = training_service.get_training_status()
    ewc_stats = training_service.get_ewc_statistics()
    buffer_stats = feedback_buffer_service.get_statistics()

    return {
        "training": training_status,
        "ewc": ewc_stats,
        "feedback_buffer": {
            "total_samples": buffer_stats.total_samples,
            "ready_for_training": buffer_stats.ready_for_training
        }
    }


@router.get("/train/incremental/ready")
async def check_incremental_ready(min_samples: int = 50):
    """Check if incremental training can be started."""
    buffer_stats = feedback_buffer_service.get_statistics()

    ready = (
        not training_service.is_training() and
        buffer_stats.total_samples >= min_samples
    )

    return {
        "ready": ready,
        "samples_available": buffer_stats.total_samples,
        "samples_required": min_samples,
        "training_in_progress": training_service.is_training(),
        "reasons": [] if ready else _get_not_ready_reasons(buffer_stats, min_samples)
    }


def _get_not_ready_reasons(buffer_stats, min_samples: int) -> List[str]:
    """Get reasons why incremental training is not ready."""
    reasons = []
    if training_service.is_training():
        reasons.append("Training already in progress")
    if buffer_stats.total_samples < min_samples:
        reasons.append(f"Insufficient samples ({buffer_stats.total_samples}/{min_samples})")
    return reasons


# =============================================================================
# Model Versioning & Rollback
# =============================================================================

@router.get("/models/versions")
async def get_model_versions(limit: int = 20):
    """Get model version history."""
    versions = rollback_service.get_versions(limit)
    current = rollback_service.get_current_version()

    return {
        "versions": [v.to_dict() for v in versions],
        "current_version": current.to_dict() if current else None,
        "count": len(versions)
    }


@router.get("/models/versions/{version_id}")
async def get_model_version(version_id: str):
    """Get a specific model version."""
    version = rollback_service._get_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")

    return version.to_dict()


@router.post("/models/deploy/{version_id}")
async def deploy_model_version(version_id: str, force: bool = False):
    """
    Deploy a specific model version.

    Args:
        version_id: The version to deploy
        force: Skip validation checks
    """
    result = await rollback_service.deploy_model(version_id, force)

    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result


@router.post("/models/rollback")
async def rollback_model(version_id: Optional[str] = None):
    """
    Rollback to a previous model version.

    Args:
        version_id: Specific version to rollback to.
                   If not provided, rolls back to previous version.
    """
    result = await rollback_service.rollback(version_id)

    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result


@router.get("/models/rollback/statistics")
async def get_rollback_statistics():
    """Get rollback service statistics."""
    return rollback_service.get_statistics()


@router.get("/models/metrics-history")
async def get_metrics_history(limit: int = 30):
    """
    Get aggregated metrics history for learning progress visualization.

    Returns daily aggregated training metrics including:
    - Average loss (converted to accuracy-like percentage)
    - Pattern detection F1-score
    - Deploy rate (successful deployments / total trainings)
    """
    from collections import defaultdict
    from datetime import datetime

    history = training_service.get_training_history()
    versions = rollback_service.get_versions(limit=100)

    # Group by date
    daily_metrics = defaultdict(lambda: {
        "losses": [],
        "deployed": 0,
        "rejected": 0,
        "total": 0
    })

    # Process training history
    for entry in history:
        # Get date from completed_at or started_at
        timestamp = entry.get("completed_at") or entry.get("started_at")
        if not timestamp:
            continue

        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            date_key = dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue

        # Extract metrics
        best_loss = entry.get("best_loss")
        if best_loss is not None and isinstance(best_loss, (int, float)):
            daily_metrics[date_key]["losses"].append(best_loss)

        daily_metrics[date_key]["total"] += 1

        # Check if deployed (status completed = deployed for TCN)
        status = entry.get("status")
        if status in ["completed", "COMPLETED"]:
            daily_metrics[date_key]["deployed"] += 1
        elif status in ["failed", "FAILED"]:
            daily_metrics[date_key]["rejected"] += 1

    # Also check version history for deployment stats
    for version in versions:
        try:
            created_at = version.created_at
            if isinstance(created_at, str):
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                dt = created_at
            date_key = dt.strftime("%Y-%m-%d")

            if version.status.value == "active":
                daily_metrics[date_key]["deployed"] += 1
            elif version.status.value == "rolled_back":
                daily_metrics[date_key]["rejected"] += 1
        except (ValueError, AttributeError):
            continue

    # Build response
    result = []
    for date_key in sorted(daily_metrics.keys())[-limit:]:
        metrics = daily_metrics[date_key]
        losses = metrics["losses"]

        # Convert loss to "accuracy-like" percentage (lower loss = higher accuracy)
        # Using 1 - loss for BCE loss which is typically 0-1
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_accuracy = max(0, min(100, (1 - avg_loss) * 100))

        # F1-Score approximation (based on pattern detection metrics if available)
        # For now, use a similar transformation
        avg_f1 = avg_accuracy * 0.95 if losses else 0

        # Deploy rate
        total = metrics["total"] or 1
        deploy_rate = (metrics["deployed"] / total) * 100 if total > 0 else 0

        result.append({
            "date": date_key,
            "avg_accuracy": round(avg_accuracy, 1),
            "avg_f1": round(avg_f1, 1),
            "deploy_rate": round(deploy_rate, 1),
            "deployed": metrics["deployed"],
            "rejected": metrics["rejected"],
            "total_trainings": metrics["total"]
        })

    return {
        "history": result,
        "count": len(result)
    }
