"""Self-learning orchestrator endpoints for TCN Training Service."""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from ..services.self_learning_orchestrator import self_learning_orchestrator

router = APIRouter()


class OrchestratorConfigUpdate(BaseModel):
    """Configuration update request."""
    enabled: Optional[bool] = None
    check_interval_minutes: Optional[int] = Field(None, ge=5, le=1440)
    scheduled_training_enabled: Optional[bool] = None
    scheduled_hour: Optional[int] = Field(None, ge=0, le=23)
    scheduled_minute: Optional[int] = Field(None, ge=0, le=59)
    min_samples_for_training: Optional[int] = Field(None, ge=10)
    min_hours_between_training: Optional[int] = Field(None, ge=1)
    max_hours_between_training: Optional[int] = Field(None, ge=24)
    auto_deploy_on_improvement: Optional[bool] = None
    ewc_lambda: Optional[float] = Field(None, ge=0)
    incremental_learning_rate: Optional[float] = Field(None, gt=0, le=0.01)
    incremental_epochs: Optional[int] = Field(None, ge=1, le=100)


@router.get("/self-learning/status")
async def get_self_learning_status():
    """
    Get self-learning orchestrator status.

    Returns current state, training statistics, and configuration.
    """
    status = self_learning_orchestrator.get_status()
    config = self_learning_orchestrator.get_config()

    return {
        "status": status.to_dict(),
        "config": config.to_dict()
    }


@router.post("/self-learning/start")
async def start_self_learning():
    """Start the self-learning loop."""
    success = await self_learning_orchestrator.start_loop()

    if success:
        return {
            "status": "started",
            "message": "Self-learning loop started"
        }
    else:
        status = self_learning_orchestrator.get_status()
        if status.loop_running:
            return {
                "status": "already_running",
                "message": "Self-learning loop is already running"
            }
        else:
            return {
                "status": "disabled",
                "message": "Self-learning is disabled in configuration"
            }


@router.post("/self-learning/stop")
async def stop_self_learning():
    """Stop the self-learning loop."""
    success = await self_learning_orchestrator.stop_loop()

    if success:
        return {
            "status": "stopped",
            "message": "Self-learning loop stopped"
        }
    else:
        return {
            "status": "not_running",
            "message": "Self-learning loop was not running"
        }


@router.post("/self-learning/trigger")
async def trigger_training(full_retrain: bool = False):
    """
    Manually trigger training.

    Args:
        full_retrain: If True, trigger full retraining instead of incremental
    """
    result = await self_learning_orchestrator.trigger_manual_training(full_retrain)
    return result


@router.put("/self-learning/config")
async def update_config(config: OrchestratorConfigUpdate):
    """Update self-learning configuration."""
    updated = self_learning_orchestrator.update_config(
        enabled=config.enabled,
        check_interval_minutes=config.check_interval_minutes,
        scheduled_training_enabled=config.scheduled_training_enabled,
        scheduled_hour=config.scheduled_hour,
        scheduled_minute=config.scheduled_minute,
        min_samples_for_training=config.min_samples_for_training,
        min_hours_between_training=config.min_hours_between_training,
        max_hours_between_training=config.max_hours_between_training,
        auto_deploy_on_improvement=config.auto_deploy_on_improvement,
        ewc_lambda=config.ewc_lambda,
        incremental_learning_rate=config.incremental_learning_rate,
        incremental_epochs=config.incremental_epochs
    )

    return {
        "status": "updated",
        "config": updated.to_dict()
    }


@router.get("/self-learning/history")
async def get_training_history():
    """Get self-learning training history."""
    history = self_learning_orchestrator.get_history()
    return {
        "history": history,
        "count": len(history)
    }
