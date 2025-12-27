"""Training Orchestrator API Routes.

Central API for managing all ML model training across services:
- NHITS (Price Forecasts)
- TCN (Chart Patterns)
- HMM (Market Regimes)
- Candlestick (Candlestick Patterns)
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..services.training_orchestrator import (
    training_orchestrator,
    TrainingServiceType,
    JobPriority,
    JobStatus
)

router = APIRouter(prefix="/training", tags=["Training Orchestrator"])


# ============ Request/Response Models ============

class TrainingRequest(BaseModel):
    """Request to start a training job."""
    service: str = Field(
        ...,
        description="Service type: 'nhits', 'tcn', 'hmm', or 'candlestick'"
    )
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to train (default: all available)"
    )
    timeframes: Optional[List[str]] = Field(
        default=None,
        description="Timeframes to train (default: service-specific)"
    )
    priority: Optional[str] = Field(
        default="normal",
        description="Priority: 'low', 'normal', 'high', 'critical'"
    )
    config: Optional[dict] = Field(
        default=None,
        description="Additional service-specific configuration"
    )


class ScheduleRequest(BaseModel):
    """Request to create a training schedule."""
    service: str = Field(..., description="Service type")
    schedule: str = Field(
        default="daily",
        description="Schedule: 'hourly', 'daily', 'weekly'"
    )
    symbols: Optional[List[str]] = Field(default=None)
    timeframes: Optional[List[str]] = Field(default=None)


class BulkTrainingRequest(BaseModel):
    """Request to start training across multiple services."""
    services: List[str] = Field(
        ...,
        description="List of services to train"
    )
    symbols: Optional[List[str]] = Field(default=None)
    priority: str = Field(default="normal")


# ============ Helper Functions ============

def _parse_service_type(service: str) -> TrainingServiceType:
    """Parse service type from string."""
    try:
        return TrainingServiceType(service.lower())
    except ValueError:
        valid = [t.value for t in TrainingServiceType]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid service: {service}. Valid options: {valid}"
        )


def _parse_priority(priority: str) -> JobPriority:
    """Parse priority from string."""
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
        "critical": JobPriority.CRITICAL
    }
    if priority.lower() not in priority_map:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid priority: {priority}. Valid options: {list(priority_map.keys())}"
        )
    return priority_map[priority.lower()]


# ============ Orchestrator Status ============

@router.get("/status")
async def get_orchestrator_status():
    """
    Get Training Orchestrator status.

    Returns overview of:
    - Queue status
    - Running jobs
    - Configured services
    - Active schedules
    """
    return training_orchestrator.get_status()


@router.get("/services")
async def get_services_status():
    """
    Get status of all training services.

    Checks health of each training service container.
    """
    return await training_orchestrator.get_all_services_status()


@router.get("/services/{service}")
async def get_service_status(service: str):
    """Get status of a specific training service."""
    service_type = _parse_service_type(service)
    return await training_orchestrator.get_service_status(service_type)


# ============ Job Management ============

@router.post("/queue")
async def queue_training(request: TrainingRequest):
    """
    Queue a new training job.

    The job will be executed based on priority and service availability.

    ## Services

    - **nhits**: Price forecast models (H1, D1 timeframes)
    - **tcn**: Chart pattern recognition models
    - **hmm**: Market regime detection (HMM + LightGBM Scorer)
    - **candlestick**: Candlestick pattern recognition

    ## Priority Levels

    - **low**: Background training, runs when resources are free
    - **normal**: Standard priority (default)
    - **high**: Expedited training
    - **critical**: Immediate training, preempts normal jobs
    """
    try:
        service_type = _parse_service_type(request.service)
        priority = _parse_priority(request.priority)

        job = await training_orchestrator.queue_training(
            service_type=service_type,
            symbols=request.symbols,
            timeframes=request.timeframes,
            priority=priority,
            config=request.config
        )

        return {
            "message": "Training job queued",
            "job_id": job.job_id,
            "service": request.service,
            "priority": request.priority,
            "status": job.status.value
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/queue/bulk")
async def queue_bulk_training(request: BulkTrainingRequest):
    """
    Queue training jobs for multiple services.

    Useful for triggering a complete retraining cycle.
    """
    jobs = []
    errors = []

    priority = _parse_priority(request.priority)

    for service in request.services:
        try:
            service_type = _parse_service_type(service)
            job = await training_orchestrator.queue_training(
                service_type=service_type,
                symbols=request.symbols,
                priority=priority
            )
            jobs.append({"service": service, "job_id": job.job_id})
        except Exception as e:
            errors.append({"service": service, "error": str(e)})

    return {
        "message": f"Queued {len(jobs)} training jobs",
        "jobs": jobs,
        "errors": errors if errors else None
    }


@router.get("/queue")
async def get_queue():
    """Get all queued training jobs."""
    return {
        "queue": training_orchestrator.get_queue(),
        "total": len(training_orchestrator._queue)
    }


@router.get("/running")
async def get_running_jobs():
    """Get all currently running training jobs."""
    return {
        "running": training_orchestrator.get_running_jobs(),
        "total": len(training_orchestrator._running_jobs)
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get details of a specific training job."""
    job = training_orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running training job."""
    if await training_orchestrator.cancel_job(job_id):
        return {"message": "Job cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=404, detail=f"Job not found or already completed: {job_id}")


@router.get("/history")
async def get_history(limit: int = 50):
    """Get training job history."""
    return {
        "history": training_orchestrator.get_history(limit),
        "total": len(training_orchestrator._completed_jobs)
    }


# ============ Schedules ============

@router.get("/schedules")
async def get_schedules():
    """Get all training schedules."""
    return {
        "schedules": training_orchestrator.get_schedules(),
        "total": len(training_orchestrator._schedules)
    }


@router.post("/schedules")
async def create_schedule(request: ScheduleRequest):
    """
    Create a training schedule.

    ## Schedule Types

    - **hourly**: Train every hour
    - **daily**: Train once per day (recommended)
    - **weekly**: Train once per week
    """
    try:
        service_type = _parse_service_type(request.service)

        if request.schedule not in ["hourly", "daily", "weekly"]:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid schedule: {request.schedule}. Use 'hourly', 'daily', or 'weekly'"
            )

        schedule = training_orchestrator.add_schedule(
            service_type=service_type,
            cron_expression=request.schedule,
            symbols=request.symbols,
            timeframes=request.timeframes
        )

        return {
            "message": "Schedule created",
            "schedule_id": schedule.schedule_id,
            "service": request.service,
            "schedule": request.schedule,
            "next_run": schedule.next_run
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a training schedule."""
    if training_orchestrator.remove_schedule(schedule_id):
        return {"message": "Schedule deleted", "schedule_id": schedule_id}
    else:
        raise HTTPException(status_code=404, detail=f"Schedule not found: {schedule_id}")


# ============ Quick Actions ============

@router.post("/train-all")
async def train_all(priority: str = "normal", symbols: Optional[List[str]] = None):
    """
    Trigger training for all services.

    Queues training jobs for: NHITS, TCN, HMM, Candlestick
    """
    return await queue_bulk_training(BulkTrainingRequest(
        services=["nhits", "tcn", "hmm", "candlestick"],
        symbols=symbols,
        priority=priority
    ))


@router.post("/train/{service}")
async def quick_train(service: str, symbols: Optional[List[str]] = None):
    """Quick endpoint to start training for a specific service."""
    return await queue_training(TrainingRequest(
        service=service,
        symbols=symbols,
        priority="normal"
    ))
