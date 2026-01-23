"""
Self-Learning Router for CNN-LSTM Training Service.

API endpoints for controlling the self-learning orchestrator.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from ..services.self_learning_orchestrator import self_learning_orchestrator
from ..services.feedback_buffer_service import feedback_buffer_service


router = APIRouter(prefix="/self-learning", tags=["4. Self-Learning"])


class ConfigUpdateRequest(BaseModel):
    """Request to update orchestrator configuration."""
    enabled: Optional[bool] = None
    check_interval_minutes: Optional[int] = Field(None, ge=5, le=1440)
    scheduled_training_enabled: Optional[bool] = None
    scheduled_hour: Optional[int] = Field(None, ge=0, le=23)
    scheduled_minute: Optional[int] = Field(None, ge=0, le=59)
    min_samples_for_training: Optional[int] = Field(None, ge=10, le=10000)
    min_hours_between_trainings: Optional[int] = Field(None, ge=1, le=168)
    max_hours_between_trainings: Optional[int] = Field(None, ge=24, le=720)
    incremental_epochs: Optional[int] = Field(None, ge=1, le=50)
    incremental_learning_rate: Optional[float] = Field(None, ge=1e-7, le=1e-3)
    ewc_lambda: Optional[float] = Field(None, ge=0, le=10000)
    auto_deploy_enabled: Optional[bool] = None
    min_accuracy_improvement: Optional[float] = Field(None, ge=-1.0, le=1.0)
    max_accuracy_regression: Optional[float] = Field(None, ge=0.0, le=1.0)


@router.get("/status", summary="Get self-learning status")
async def get_status():
    """
    Get the current status of the self-learning orchestrator.

    Returns state, loop status, training history, and configuration.
    """
    status_data = self_learning_orchestrator.get_status()

    # Add feedback buffer status
    buffer_stats = feedback_buffer_service.get_statistics()
    status_data["status"]["feedback_buffer_samples"] = buffer_stats.unused_samples
    status_data["status"]["buffer_ready"] = buffer_stats.ready_for_training

    return status_data


@router.post("/start", summary="Start self-learning loop")
async def start_loop():
    """Start the self-learning monitoring loop."""
    return await self_learning_orchestrator.start_loop()


@router.post("/stop", summary="Stop self-learning loop")
async def stop_loop():
    """Stop the self-learning monitoring loop."""
    return await self_learning_orchestrator.stop_loop()


@router.post("/trigger", summary="Trigger manual training")
async def trigger_training():
    """
    Manually trigger the incremental training process.

    Analyzes feedback buffer and trains on available samples.
    """
    return await self_learning_orchestrator.trigger_manual_training()


@router.put("/config", summary="Update configuration")
async def update_config(request: ConfigUpdateRequest):
    """
    Update the self-learning orchestrator configuration.

    Only non-null fields will be updated.
    """
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if not updates:
        return {"status": "no_changes", "config": self_learning_orchestrator.get_config().to_dict()}

    config = self_learning_orchestrator.update_config(updates)

    return {"status": "updated", "config": config.to_dict()}


@router.get("/config", summary="Get configuration")
async def get_config():
    """Get the current self-learning configuration."""
    return self_learning_orchestrator.get_config().to_dict()
