"""NHITS Training Service for dedicated training container.

This service handles NHITS model training in a separate container from inference.
Training is CPU/GPU intensive and should not block forecast API requests.

Architecture:
    - NHITS Service (Port 3002): Inference only - fast forecast API responses
    - NHITS-Train Service (Port 3012): Training only - background model training
    - Shared volume: /app/data/models/nhits - trained models accessible by both

Training Flow:
    1. Training request received (via API or scheduled)
    2. Fetch training data via Data Gateway Service
    3. Train NHITS model (CPU/GPU intensive)
    4. Save model to shared volume
    5. Notify NHITS inference service to reload model
"""

import os
import json
import asyncio
import httpx
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import numpy as np
from loguru import logger

# Lazy PyTorch import for container startup
torch = None


def _load_torch():
    global torch
    if torch is None:
        try:
            import torch as t
            torch = t
            return True
        except ImportError:
            logger.warning("PyTorch not available")
            return False
    return True


class TrainingStatus(str, Enum):
    """Training job status."""
    IDLE = "idle"
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job information."""
    job_id: str
    status: TrainingStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    symbols: List[str] = None
    timeframes: List[str] = None
    epochs: int = 100
    progress: float = 0.0
    current_symbol: Optional[str] = None
    current_timeframe: Optional[str] = None
    completed_models: int = 0
    total_models: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    error_message: Optional[str] = None
    results: Dict[str, Any] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


# Service URLs
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
NHITS_SERVICE_URL = os.getenv("NHITS_SERVICE_URL", "http://trading-nhits:3002")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/nhits")


class NHITSTrainingService:
    """
    Dedicated NHITS Training Service.

    Runs in a separate container from the inference service to prevent
    training from blocking API requests for forecasts.
    """

    # Timeframe configurations (matches NHITS inference service)
    TIMEFRAME_CONFIGS = {
        "M15": {"input_size": 96, "horizon": 16, "step_minutes": 15},
        "H1": {"input_size": 168, "horizon": 24, "step_minutes": 60},
        "D1": {"input_size": 60, "horizon": 14, "step_minutes": 1440},
    }

    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._current_job: Optional[TrainingJob] = None
        self._training_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._model_path = Path(MODEL_DIR)
        self._model_path.mkdir(parents=True, exist_ok=True)
        self._stop_requested = False

        # Load training history
        self._history_file = self._model_path / "training_history.json"
        self._load_history()

        logger.info(f"NHITSTrainingService initialized (model_dir: {MODEL_DIR})")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _load_history(self):
        """Load training history from file."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r') as f:
                    data = json.load(f)
                    for job_data in data[-50:]:  # Keep last 50 jobs
                        job_data["status"] = TrainingStatus(job_data["status"])
                        job = TrainingJob(**job_data)
                        self._jobs[job.job_id] = job
                logger.info(f"Loaded {len(self._jobs)} training jobs from history")
        except Exception as e:
            logger.error(f"Failed to load training history: {e}")

    def _save_history(self):
        """Save training history to file."""
        try:
            jobs_list = [job.to_dict() for job in list(self._jobs.values())[-50:]]
            with open(self._history_file, 'w') as f:
                json.dump(jobs_list, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        return f"nhits_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    async def _fetch_training_data(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365
    ) -> List[Dict]:
        """Fetch training data from Data Service."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"{DATA_SERVICE_URL}/api/v1/training-data/{symbol}",
                params={"timeframe": timeframe, "days": days, "use_cache": True}
            )
            response.raise_for_status()
            result = response.json()
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Failed to fetch training data for {symbol}/{timeframe}: {e}")
            return []

    async def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols from Data Service."""
        client = await self._get_client()

        try:
            response = await client.get(f"{DATA_SERVICE_URL}/api/v1/managed-symbols")
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list):
                return [s.get("symbol", s) if isinstance(s, dict) else s for s in result]
            return result.get("symbols", [])
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    async def _notify_nhits_service(self, symbol: str, timeframe: str, model_path: str) -> bool:
        """Notify NHITS inference service that a new model is available."""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{NHITS_SERVICE_URL}/api/v1/model/reload",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "model_path": model_path
                },
                timeout=30.0
            )
            if response.status_code == 200:
                logger.info(f"NHITS service notified of new model: {symbol}/{timeframe}")
                return True
            else:
                logger.warning(f"NHITS notification failed: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not notify NHITS service: {e}")
            return False

    async def _train_model(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict],
        job: TrainingJob
    ) -> Dict[str, Any]:
        """Train a single NHITS model."""
        if not _load_torch():
            return {"success": False, "error": "PyTorch not available"}

        tf_config = self.TIMEFRAME_CONFIGS.get(timeframe.upper(), self.TIMEFRAME_CONFIGS["H1"])
        min_required = tf_config["input_size"] + tf_config["horizon"]

        if len(data) < min_required:
            return {
                "success": False,
                "error": f"Insufficient data: {len(data)} < {min_required}",
                "samples": len(data)
            }

        try:
            start_time = datetime.now(timezone.utc)

            # Extract close prices for training
            closes = []
            for row in data:
                close = row.get("close") or row.get("h1_close") or row.get("d1_close") or row.get("m15_close")
                if close:
                    closes.append(float(close))

            if len(closes) < min_required:
                return {
                    "success": False,
                    "error": f"Insufficient close prices: {len(closes)} < {min_required}",
                    "samples": len(closes)
                }

            closes = np.array(closes, dtype=np.float32)

            # Simple NHITS-like training (placeholder - in production use proper NHITS)
            # This demonstrates the training pattern
            input_size = tf_config["input_size"]
            horizon = tf_config["horizon"]

            # Create sequences
            X, y = [], []
            for i in range(len(closes) - input_size - horizon):
                X.append(closes[i:i + input_size])
                y.append(closes[i + input_size:i + input_size + horizon])

            X = np.array(X)
            y = np.array(y)

            # Simple linear model as placeholder
            # In production, this would be the actual NHITS model
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(X, y)

            # Save model
            import joblib
            model_filename = f"{symbol}_{timeframe}.pkl"
            model_path = self._model_path / model_filename
            joblib.dump({
                "model": model,
                "symbol": symbol,
                "timeframe": timeframe,
                "input_size": input_size,
                "horizon": horizon,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "samples": len(X)
            }, model_path)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Notify inference service
            await self._notify_nhits_service(symbol, timeframe, str(model_path))

            return {
                "success": True,
                "samples": len(X),
                "duration_seconds": duration,
                "model_path": str(model_path)
            }

        except Exception as e:
            logger.error(f"Training failed for {symbol}/{timeframe}: {e}")
            return {"success": False, "error": str(e)}

    async def _run_training(self, job: TrainingJob):
        """Execute the training job."""
        try:
            job.status = TrainingStatus.PREPARING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(f"Starting training job {job.job_id}")

            # Get symbols
            symbols = job.symbols
            if not symbols:
                symbols = await self._get_available_symbols()
                if not symbols:
                    raise RuntimeError("No symbols available for training")
                job.symbols = symbols[:20]  # Limit to 20 symbols

            timeframes = job.timeframes or ["H1", "D1"]
            job.total_models = len(job.symbols) * len(timeframes)
            job.results = {}

            job.status = TrainingStatus.TRAINING
            self._save_history()

            for symbol in job.symbols:
                if self._stop_requested or job.status == TrainingStatus.CANCELLED:
                    logger.info(f"Training job {job.job_id} cancelled")
                    job.status = TrainingStatus.CANCELLED
                    break

                for tf in timeframes:
                    if self._stop_requested:
                        break

                    model_key = f"{symbol}_{tf}"
                    job.current_symbol = symbol
                    job.current_timeframe = tf

                    try:
                        # Fetch data
                        data = await self._fetch_training_data(symbol, tf)

                        if not data:
                            job.results[model_key] = {
                                "success": False,
                                "error": "No training data available"
                            }
                            job.failed += 1
                        else:
                            # Train model
                            result = await self._train_model(symbol, tf, data, job)
                            job.results[model_key] = result

                            if result.get("success"):
                                job.successful += 1
                            else:
                                job.failed += 1

                    except Exception as e:
                        logger.error(f"Error training {model_key}: {e}")
                        job.results[model_key] = {"success": False, "error": str(e)}
                        job.failed += 1

                    job.completed_models += 1
                    job.progress = job.completed_models / job.total_models * 100

                    # Yield to event loop periodically
                    await asyncio.sleep(0)

            if job.status != TrainingStatus.CANCELLED:
                job.status = TrainingStatus.COMPLETED

            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(
                f"Training job {job.job_id} {job.status.value}: "
                f"{job.successful} successful, {job.failed} failed"
            )

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_history()
            logger.error(f"Training job {job.job_id} failed: {e}")

        finally:
            self._current_job = None
            self._stop_requested = False

    async def start_training(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        force: bool = False
    ) -> TrainingJob:
        """
        Start a new training job.

        Args:
            symbols: List of symbols to train (default: all available)
            timeframes: List of timeframes (default: ["H1", "D1"])
            force: Force retraining even if model exists

        Returns:
            TrainingJob with job information
        """
        if self._current_job and self._current_job.status == TrainingStatus.TRAINING:
            raise RuntimeError("A training job is already running")

        job = TrainingJob(
            job_id=self._generate_job_id(),
            status=TrainingStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            timeframes=timeframes or ["H1", "D1"],
            results={}
        )

        self._jobs[job.job_id] = job
        self._current_job = job
        self._stop_requested = False
        self._save_history()

        # Start training in background
        self._training_task = asyncio.create_task(self._run_training(job))

        logger.info(f"Created training job {job.job_id}")
        return job

    async def cancel_training(self, job_id: Optional[str] = None) -> bool:
        """Cancel a training job."""
        job = self._jobs.get(job_id) if job_id else self._current_job

        if not job:
            return False

        if job.status not in [TrainingStatus.PENDING, TrainingStatus.PREPARING, TrainingStatus.TRAINING]:
            return False

        self._stop_requested = True
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        self._save_history()

        logger.info(f"Cancelled training job {job.job_id}")
        return True

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get currently running job."""
        return self._current_job

    def get_all_jobs(self, limit: int = 20) -> List[TrainingJob]:
        """Get all training jobs."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        current = self._current_job

        return {
            "service": "nhits-train",
            "status": "training" if current and current.status == TrainingStatus.TRAINING else "idle",
            "current_job": current.to_dict() if current else None,
            "jobs_in_history": len(self._jobs),
            "model_directory": str(self._model_path),
            "pytorch_available": _load_torch()
        }

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._current_job is not None and self._current_job.status == TrainingStatus.TRAINING

    def list_models(self) -> List[Dict[str, Any]]:
        """List available trained models."""
        models = []

        if self._model_path.exists():
            for f in self._model_path.glob("*.pkl"):
                try:
                    stat = f.stat()
                    models.append({
                        "name": f.stem,
                        "path": str(f),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
                except Exception:
                    pass

        return sorted(models, key=lambda x: x["created"], reverse=True)


# Global singleton
training_service = NHITSTrainingService()
