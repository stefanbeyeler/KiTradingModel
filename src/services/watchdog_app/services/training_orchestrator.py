"""Training Orchestrator Service.

Centralized training coordination for all ML model training services.
Manages training schedules, priorities, and resource allocation.

Training Services:
    - NHITS-Train (Port 3012): Price forecast models
    - TCN-Train (Port 3013): Chart pattern models
    - HMM-Train (Port 3014): Market regime models
    - Candlestick-Train (Port 3016): Candlestick pattern models
    - CNN-LSTM-Train (Port 3017): Multi-task hybrid models (price, pattern, regime)
"""

import os
import json
import asyncio
import httpx
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from loguru import logger

from .resource_monitor import resource_monitor

# Import zentrale Microservices-Konfiguration
from src.config.microservices import microservices_config


class TrainingServiceType(str, Enum):
    """Types of training services."""
    NHITS = "nhits"
    TCN = "tcn"
    HMM = "hmm"
    CANDLESTICK = "candlestick"
    CNN_LSTM = "cnn-lstm"


class JobStatus(str, Enum):
    """Orchestrated job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TrainingServiceConfig:
    """Configuration for a training service."""
    name: str
    service_type: TrainingServiceType
    url: str
    port: int
    start_endpoint: str = "/api/v1/train/start"
    status_endpoint: str = "/api/v1/train/status"
    cancel_endpoint: str = "/api/v1/train/cancel"
    models_endpoint: str = "/api/v1/train/models"
    enabled: bool = True
    default_priority: JobPriority = JobPriority.NORMAL
    max_concurrent: int = 1
    default_symbols: List[str] = None


@dataclass
class TrainingJob:
    """A training job in the orchestrator queue."""
    job_id: str
    service_type: TrainingServiceType
    status: JobStatus
    priority: JobPriority
    created_at: str
    symbols: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    remote_job_id: Optional[str] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Validation pipeline fields (for HMM)
    validation_status: Optional[str] = None  # "pending", "passed", "partial", "failed"
    validation_metrics: Optional[Dict[str, Any]] = None
    deployment_decisions: Optional[Dict[str, Any]] = None
    deployed_count: int = 0
    rejected_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        d["service_type"] = self.service_type.value
        return d


@dataclass
class ScheduledTraining:
    """Scheduled training configuration."""
    schedule_id: str
    service_type: TrainingServiceType
    cron_expression: str  # Simplified: "daily", "weekly", "hourly"
    symbols: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None


class TrainingOrchestrator:
    """
    Centralized Training Orchestrator.

    Responsibilities:
    1. Queue and prioritize training jobs across all services
    2. Manage training schedules
    3. Monitor training progress
    4. Handle training failures and retries
    5. Resource coordination (one GPU training at a time)
    """

    # Service configurations (URLs aus zentraler Konfiguration)
    SERVICES: Dict[TrainingServiceType, TrainingServiceConfig] = {
        TrainingServiceType.NHITS: TrainingServiceConfig(
            name="NHITS Training",
            service_type=TrainingServiceType.NHITS,
            url=os.getenv("NHITS_TRAIN_URL", microservices_config.nhits_train_url),
            port=microservices_config.nhits_train_port,
            default_priority=JobPriority.NORMAL
        ),
        TrainingServiceType.TCN: TrainingServiceConfig(
            name="TCN Training",
            service_type=TrainingServiceType.TCN,
            url=os.getenv("TCN_TRAIN_URL", microservices_config.tcn_train_url),
            port=microservices_config.tcn_train_port,
            default_priority=JobPriority.NORMAL
        ),
        TrainingServiceType.HMM: TrainingServiceConfig(
            name="HMM Training",
            service_type=TrainingServiceType.HMM,
            url=os.getenv("HMM_TRAIN_URL", microservices_config.hmm_train_url),
            port=microservices_config.hmm_train_port,
            default_priority=JobPriority.LOW
        ),
        TrainingServiceType.CANDLESTICK: TrainingServiceConfig(
            name="Candlestick Training",
            service_type=TrainingServiceType.CANDLESTICK,
            url=os.getenv("CANDLESTICK_TRAIN_URL", microservices_config.candlestick_train_url),
            port=microservices_config.candlestick_train_port,
            default_priority=JobPriority.LOW
        ),
        TrainingServiceType.CNN_LSTM: TrainingServiceConfig(
            name="CNN-LSTM Training",
            service_type=TrainingServiceType.CNN_LSTM,
            url=os.getenv("CNN_LSTM_TRAIN_URL", microservices_config.cnn_lstm_train_url),
            port=microservices_config.cnn_lstm_train_port,
            default_priority=JobPriority.LOW
        ),
    }

    def __init__(self):
        self._queue: List[TrainingJob] = []
        self._running_jobs: Dict[str, TrainingJob] = {}
        self._completed_jobs: List[TrainingJob] = []
        self._schedules: Dict[str, ScheduledTraining] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._running = False
        self._paused = False  # Resource protection: pause new job starts
        self._worker_task: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None

        # Persistence
        self._state_file = Path(os.getenv("ORCHESTRATOR_STATE_FILE", "/app/data/training_orchestrator_state.json"))
        self._load_state()

        logger.info("TrainingOrchestrator initialized")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def close(self):
        """Cleanup resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _load_state(self):
        """Load persisted state."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)

                # Restore completed jobs
                for job_data in state.get("completed_jobs", [])[-100:]:
                    job_data["status"] = JobStatus(job_data["status"])
                    job_data["priority"] = JobPriority(job_data["priority"])
                    job_data["service_type"] = TrainingServiceType(job_data["service_type"])
                    self._completed_jobs.append(TrainingJob(**job_data))

                # Restore schedules
                for schedule_data in state.get("schedules", []):
                    schedule_data["service_type"] = TrainingServiceType(schedule_data["service_type"])
                    schedule = ScheduledTraining(**schedule_data)
                    self._schedules[schedule.schedule_id] = schedule

                logger.info(f"Loaded orchestrator state: {len(self._completed_jobs)} completed jobs, {len(self._schedules)} schedules")
        except Exception as e:
            logger.error(f"Failed to load orchestrator state: {e}")

    def _save_state(self):
        """Persist state to file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "completed_jobs": [job.to_dict() for job in self._completed_jobs[-100:]],
                "schedules": [
                    {
                        **asdict(s),
                        "service_type": s.service_type.value
                    }
                    for s in self._schedules.values()
                ]
            }

            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save orchestrator state: {e}")

    def _generate_job_id(self, service_type: TrainingServiceType) -> str:
        """Generate unique job ID."""
        return f"orch_{service_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    async def start(self):
        """Start the orchestrator background workers."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._job_worker())
        self._scheduler_task = asyncio.create_task(self._schedule_worker())
        logger.info("Training Orchestrator started")

    async def stop(self):
        """Stop the orchestrator."""
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        await self.close()
        self._save_state()
        logger.info("Training Orchestrator stopped")

    async def _job_worker(self):
        """Background worker that processes the job queue with resource awareness."""
        while self._running:
            try:
                # Check if we can start a new job
                if self._queue and len(self._running_jobs) < 2:  # Max 2 concurrent
                    # Check if orchestrator is paused (resource protection)
                    if self._paused:
                        logger.debug("Training orchestrator paused - waiting")
                        await asyncio.sleep(10)
                        continue

                    # Resource check before starting new training
                    can_train, reason = resource_monitor.can_start_training()
                    if not can_train:
                        logger.warning(f"Training delayed due to resource constraints: {reason}")
                        await asyncio.sleep(10)
                        continue

                    # Get highest priority job
                    self._queue.sort(key=lambda j: j.priority.value, reverse=True)
                    job = self._queue.pop(0)

                    # Check if service already has a running job
                    service_running = any(
                        j.service_type == job.service_type
                        for j in self._running_jobs.values()
                    )

                    if service_running:
                        # Put back in queue
                        self._queue.insert(0, job)
                    else:
                        # Final resource check before execution
                        can_train, reason = resource_monitor.can_start_training()
                        if can_train:
                            # Start the job
                            asyncio.create_task(self._execute_job(job))
                        else:
                            # Put back in queue and wait
                            self._queue.insert(0, job)
                            logger.info(f"Job {job.job_id} deferred: {reason}")

                # Update running jobs progress
                for job in list(self._running_jobs.values()):
                    await self._update_job_progress(job)

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job worker: {e}")
                await asyncio.sleep(10)

    async def _schedule_worker(self):
        """Background worker for scheduled training."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                for schedule in self._schedules.values():
                    if not schedule.enabled:
                        continue

                    if schedule.next_run:
                        next_run = datetime.fromisoformat(schedule.next_run)
                        if now >= next_run:
                            # Trigger scheduled training
                            await self.queue_training(
                                service_type=schedule.service_type,
                                symbols=schedule.symbols,
                                timeframes=schedule.timeframes,
                                priority=JobPriority.NORMAL
                            )
                            schedule.last_run = now.isoformat()
                            schedule.next_run = self._calculate_next_run(schedule.cron_expression, now).isoformat()
                            self._save_state()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in schedule worker: {e}")
                await asyncio.sleep(60)

    def _calculate_next_run(self, cron_expr: str, from_time: datetime) -> datetime:
        """Calculate next run time based on simplified cron expression."""
        if cron_expr == "hourly":
            return from_time + timedelta(hours=1)
        elif cron_expr == "daily":
            return from_time + timedelta(days=1)
        elif cron_expr == "weekly":
            return from_time + timedelta(weeks=1)
        else:
            return from_time + timedelta(days=1)

    async def _execute_job(self, job: TrainingJob):
        """Execute a training job on the remote service."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._running_jobs[job.job_id] = job

        try:
            service_config = self.SERVICES.get(job.service_type)
            if not service_config:
                raise ValueError(f"Unknown service type: {job.service_type}")

            client = await self._get_client()

            # Start training on remote service
            payload = {}
            if job.symbols:
                payload["symbols"] = job.symbols
            if job.timeframes:
                payload["timeframes"] = job.timeframes
            if job.config:
                payload.update(job.config)

            response = await client.post(
                f"{service_config.url}{service_config.start_endpoint}",
                json=payload,
                timeout=30.0
            )

            if response.status_code in [200, 201]:
                result = response.json()
                job.remote_job_id = result.get("job_id")
                logger.info(f"Started training job {job.job_id} -> remote {job.remote_job_id}")

                # Wait for completion
                await self._wait_for_completion(job, service_config)

            else:
                job.status = JobStatus.FAILED
                job.error = f"Failed to start training: {response.status_code}"

        except Exception as e:
            logger.error(f"Failed to execute job {job.job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)

        finally:
            job.completed_at = datetime.now(timezone.utc).isoformat()
            del self._running_jobs[job.job_id]
            self._completed_jobs.append(job)
            self._save_state()

    async def _wait_for_completion(self, job: TrainingJob, service_config: TrainingServiceConfig):
        """Wait for a training job to complete."""
        client = await self._get_client()
        max_wait_time = 3600 * 4  # 4 hours max
        start_time = datetime.now(timezone.utc)

        while True:
            try:
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > max_wait_time:
                    job.status = JobStatus.FAILED
                    job.error = "Timeout waiting for training completion"
                    break

                response = await client.get(
                    f"{service_config.url}{service_config.status_endpoint}",
                    timeout=30.0
                )

                if response.status_code == 200:
                    status = response.json()
                    current_job = status.get("current_job") or status

                    if not current_job:
                        # No current job - training may have completed quickly
                        job.status = JobStatus.COMPLETED
                        job.progress = 100
                        break

                    remote_status = current_job.get("status", "").lower()
                    job.progress = current_job.get("progress", 0)

                    if remote_status in ["completed", "done"]:
                        job.status = JobStatus.COMPLETED
                        job.result = current_job.get("results")

                        # Extract validation info for HMM training
                        if job.service_type == TrainingServiceType.HMM:
                            job.validation_metrics = current_job.get("validation_metrics")
                            job.deployment_decisions = current_job.get("deployment_decisions")
                            job.deployed_count = current_job.get("deployed_count", 0)
                            job.rejected_count = current_job.get("rejected_count", 0)

                            # Determine validation status
                            if job.deployed_count > 0 and job.rejected_count == 0:
                                job.validation_status = "passed"
                            elif job.deployed_count > 0 and job.rejected_count > 0:
                                job.validation_status = "partial"
                            elif job.rejected_count > 0:
                                job.validation_status = "failed"
                            else:
                                job.validation_status = "pending"

                            logger.info(
                                f"HMM training completed: {job.deployed_count} deployed, "
                                f"{job.rejected_count} rejected (validation: {job.validation_status})"
                            )
                        break
                    elif remote_status in ["failed", "error"]:
                        job.status = JobStatus.FAILED
                        job.error = current_job.get("error_message") or current_job.get("error")
                        break
                    elif remote_status in ["cancelled"]:
                        job.status = JobStatus.CANCELLED
                        break

                await asyncio.sleep(10)  # Poll every 10 seconds

            except Exception as e:
                logger.warning(f"Error polling job status: {e}")
                await asyncio.sleep(30)

    async def _update_job_progress(self, job: TrainingJob):
        """Update job progress from remote service."""
        try:
            service_config = self.SERVICES.get(job.service_type)
            if not service_config:
                return

            client = await self._get_client()
            response = await client.get(
                f"{service_config.url}{service_config.status_endpoint}",
                timeout=10.0
            )

            if response.status_code == 200:
                status = response.json()
                current_job = status.get("current_job") or status
                if current_job:
                    job.progress = current_job.get("progress", job.progress)

        except Exception:
            pass  # Ignore progress update errors

    async def queue_training(
        self,
        service_type: TrainingServiceType,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        priority: Optional[JobPriority] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> TrainingJob:
        """Queue a new training job."""
        service_config = self.SERVICES.get(service_type)
        if not service_config:
            raise ValueError(f"Unknown service type: {service_type}")

        job = TrainingJob(
            job_id=self._generate_job_id(service_type),
            service_type=service_type,
            status=JobStatus.QUEUED,
            priority=priority or service_config.default_priority,
            created_at=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self._queue.append(job)
        logger.info(f"Queued training job: {job.job_id} (type: {service_type.value}, priority: {job.priority.name})")

        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        # Check queue
        for i, job in enumerate(self._queue):
            if job.job_id == job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc).isoformat()
                self._queue.pop(i)
                self._completed_jobs.append(job)
                self._save_state()
                logger.info(f"Cancelled queued job: {job_id}")
                return True

        # Check running jobs
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            service_config = self.SERVICES.get(job.service_type)

            try:
                client = await self._get_client()
                await client.post(
                    f"{service_config.url}{service_config.cancel_endpoint}",
                    json={"job_id": job.remote_job_id},
                    timeout=30.0
                )
            except Exception as e:
                logger.warning(f"Failed to cancel remote job: {e}")

            job.status = JobStatus.CANCELLED
            logger.info(f"Cancelled running job: {job_id}")
            return True

        return False

    def add_schedule(
        self,
        service_type: TrainingServiceType,
        cron_expression: str = "daily",
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ) -> ScheduledTraining:
        """Add a training schedule."""
        schedule_id = f"schedule_{service_type.value}_{len(self._schedules)}"

        schedule = ScheduledTraining(
            schedule_id=schedule_id,
            service_type=service_type,
            cron_expression=cron_expression,
            symbols=symbols,
            timeframes=timeframes,
            enabled=True,
            next_run=self._calculate_next_run(cron_expression, datetime.now(timezone.utc)).isoformat()
        )

        self._schedules[schedule_id] = schedule
        self._save_state()

        logger.info(f"Added training schedule: {schedule_id}")
        return schedule

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a training schedule."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            self._save_state()
            logger.info(f"Removed training schedule: {schedule_id}")
            return True
        return False

    async def get_service_status(self, service_type: TrainingServiceType) -> Dict[str, Any]:
        """Get status of a training service."""
        service_config = self.SERVICES.get(service_type)
        if not service_config:
            return {"error": f"Unknown service: {service_type}"}

        try:
            client = await self._get_client()
            response = await client.get(
                f"{service_config.url}/health",
                timeout=10.0
            )

            if response.status_code == 200:
                health = response.json()
                return {
                    "service": service_config.name,
                    "url": service_config.url,
                    "healthy": True,
                    "training_in_progress": health.get("training_in_progress", False)
                }
            else:
                return {
                    "service": service_config.name,
                    "healthy": False,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "service": service_config.name,
                "healthy": False,
                "error": str(e)
            }

    async def get_all_services_status(self) -> Dict[str, Any]:
        """Get status of all training services."""
        results = {}
        for service_type in TrainingServiceType:
            results[service_type.value] = await self.get_service_status(service_type)
        return results

    def get_queue(self) -> List[Dict]:
        """Get queued jobs."""
        return [job.to_dict() for job in self._queue]

    def get_running_jobs(self) -> List[Dict]:
        """Get running jobs."""
        return [job.to_dict() for job in self._running_jobs.values()]

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get a specific job."""
        # Check queue
        for job in self._queue:
            if job.job_id == job_id:
                return job.to_dict()

        # Check running
        if job_id in self._running_jobs:
            return self._running_jobs[job_id].to_dict()

        # Check completed
        for job in self._completed_jobs:
            if job.job_id == job_id:
                return job.to_dict()

        return None

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get job history."""
        return [job.to_dict() for job in self._completed_jobs[-limit:]]

    def get_schedules(self) -> List[Dict]:
        """Get all schedules."""
        return [
            {
                **asdict(s),
                "service_type": s.service_type.value
            }
            for s in self._schedules.values()
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status with resource metrics."""
        # Get resource information
        resource_info = resource_monitor.to_dict()
        can_train, reason = resource_monitor.can_start_training()

        return {
            "running": self._running,
            "paused": self._paused,
            "queued_jobs": len(self._queue),
            "running_jobs": len(self._running_jobs),
            "completed_jobs": len(self._completed_jobs),
            "schedules": len(self._schedules),
            "resources": {
                "status": resource_info["status"],
                "cpu_percent": resource_info["cpu_percent"],
                "memory_percent": resource_info["memory_percent"],
                "can_start_training": can_train,
                "block_reason": reason if not can_train else None,
            },
            "services": {
                st.value: {
                    "name": cfg.name,
                    "url": cfg.url,
                    "enabled": cfg.enabled
                }
                for st, cfg in self.SERVICES.items()
            }
        }

    def pause(self) -> None:
        """Pause the orchestrator - no new jobs will start."""
        self._paused = True
        logger.warning("Training orchestrator PAUSED - no new jobs will start")

    def resume(self) -> None:
        """Resume the orchestrator - jobs can start again."""
        self._paused = False
        logger.info("Training orchestrator RESUMED")


# Global singleton
training_orchestrator = TrainingOrchestrator()
