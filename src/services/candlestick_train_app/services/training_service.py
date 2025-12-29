"""Candlestick Pattern Training Service.

Handles model training for candlestick pattern recognition using PyTorch TCN.
"""

import os
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from loguru import logger

# Try to import PyTorch (may not be available in all environments)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - training will be disabled")


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
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
    batch_size: int = 32
    learning_rate: float = 0.001
    progress: float = 0.0
    current_epoch: int = 0
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    error_message: Optional[str] = None
    model_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# Data Service URL
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
CANDLESTICK_SERVICE_URL = os.getenv("CANDLESTICK_SERVICE_URL", "http://trading-candlestick:3006")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

# Shared data directory (mounted from candlestick service)
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
FEEDBACK_FILE = DATA_DIR / "pattern_feedback.json"


class TrainingService:
    """
    Service for training candlestick pattern recognition models.

    Uses PyTorch TCN (Temporal Convolutional Network) for pattern classification.
    """

    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._current_job: Optional[TrainingJob] = None
        self._training_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._model_path = Path(MODEL_DIR)
        self._model_path.mkdir(parents=True, exist_ok=True)

        # Load training history
        self._history_file = self._model_path / "training_history.json"
        self._load_history()

        logger.info(f"TrainingService initialized (PyTorch available: {TORCH_AVAILABLE})")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
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
                    for job_data in data:
                        job = TrainingJob(**job_data)
                        self._jobs[job.job_id] = job
                logger.info(f"Loaded {len(self._jobs)} training jobs from history")
        except Exception as e:
            logger.error(f"Failed to load training history: {e}")

    def _save_history(self):
        """Save training history to file."""
        try:
            with open(self._history_file, 'w') as f:
                json.dump([job.to_dict() for job in self._jobs.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        return f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    async def _fetch_training_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        lookback: int = 500
    ) -> Dict[str, Any]:
        """Fetch training data from Data Service.

        Uses TwelveData API via Data Service for OHLCV data.
        Falls back to EasyInsight if TwelveData fails.
        """
        client = await self._get_client()
        training_data = {}

        # Map standard timeframes to TwelveData intervals
        tf_mapping = {
            "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1h", "H2": "2h", "H4": "4h",
            "D1": "1day", "W1": "1week", "MN": "1month",
        }

        for symbol in symbols:
            symbol_data = {}
            for tf in timeframes:
                try:
                    # Try TwelveData first (primary source)
                    interval = tf_mapping.get(tf, tf.lower())
                    url = f"{DATA_SERVICE_URL}/api/v1/twelvedata/time_series/{symbol}"
                    params = {
                        "interval": interval,
                        "outputsize": lookback,
                    }
                    response = await client.get(url, params=params)

                    if response.status_code == 200:
                        result = response.json()
                        # TwelveData returns data in "values" array
                        values = result.get("values", [])
                        if values:
                            symbol_data[tf] = values
                            logger.debug(f"Fetched {len(values)} candles for {symbol} {tf} from TwelveData")
                            continue

                    # Fallback to EasyInsight for H1 (only timeframe supported)
                    if tf == "H1":
                        url = f"{DATA_SERVICE_URL}/api/v1/easyinsight/ohlcv/{symbol}"
                        params = {"limit": lookback}
                        response = await client.get(url, params=params)
                        if response.status_code == 200:
                            result = response.json()
                            data = result.get("data", [])
                            if data:
                                symbol_data[tf] = data
                                logger.debug(f"Fetched {len(data)} candles for {symbol} {tf} from EasyInsight")
                                continue

                    logger.warning(f"No data available for {symbol} {tf}")

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol} {tf}: {e}")
                    continue

            if symbol_data:
                training_data[symbol] = symbol_data

        return training_data

    def _load_user_feedback(self) -> Dict[str, Any]:
        """Load user feedback for pattern corrections.

        Returns a dictionary with:
        - corrections: Map of pattern_id -> corrected pattern info
        - rejected_ids: Set of pattern IDs marked as false positives
        - confirmed_ids: Set of pattern IDs confirmed as correct
        - statistics: Summary of feedback data
        """
        feedback_data = {
            "corrections": {},
            "rejected_ids": set(),
            "confirmed_ids": set(),
            "statistics": {
                "total": 0,
                "confirmed": 0,
                "corrected": 0,
                "rejected": 0,
            }
        }

        if not FEEDBACK_FILE.exists():
            logger.info("No user feedback file found - training without corrections")
            return feedback_data

        try:
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                raw_feedback = json.load(f)

            for entry in raw_feedback:
                pattern_id = entry.get("id", "")
                feedback_type = entry.get("feedback_type", "")

                if feedback_type == "confirmed":
                    feedback_data["confirmed_ids"].add(pattern_id)
                    feedback_data["statistics"]["confirmed"] += 1

                elif feedback_type == "corrected":
                    feedback_data["corrections"][pattern_id] = {
                        "original": entry.get("original_pattern"),
                        "corrected": entry.get("corrected_pattern"),
                        "symbol": entry.get("symbol"),
                        "timeframe": entry.get("timeframe"),
                        "ohlc_data": entry.get("ohlc_data"),
                    }
                    feedback_data["statistics"]["corrected"] += 1

                elif feedback_type == "rejected":
                    feedback_data["rejected_ids"].add(pattern_id)
                    feedback_data["statistics"]["rejected"] += 1

                feedback_data["statistics"]["total"] += 1

            logger.info(
                f"Loaded user feedback: {feedback_data['statistics']['total']} total, "
                f"{feedback_data['statistics']['confirmed']} confirmed, "
                f"{feedback_data['statistics']['corrected']} corrected, "
                f"{feedback_data['statistics']['rejected']} rejected"
            )

        except Exception as e:
            logger.error(f"Failed to load user feedback: {e}")

        return feedback_data

    def _apply_feedback_to_labels(
        self,
        pattern_history: List[Dict],
        feedback: Dict[str, Any]
    ) -> List[Dict]:
        """Apply user feedback corrections to pattern labels.

        - Rejected patterns are excluded from training
        - Corrected patterns have their labels updated
        - Confirmed patterns are kept with boosted weight
        """
        corrected_patterns = []
        excluded_count = 0
        corrected_count = 0

        for pattern in pattern_history:
            pattern_id = pattern.get("id", "")

            # Skip rejected patterns (false positives)
            if pattern_id in feedback["rejected_ids"]:
                excluded_count += 1
                continue

            # Apply corrections
            if pattern_id in feedback["corrections"]:
                correction = feedback["corrections"][pattern_id]
                original_type = pattern.get("pattern_type")
                corrected_type = correction.get("corrected")

                if corrected_type and corrected_type != "no_pattern":
                    # Update pattern type to corrected value
                    pattern = pattern.copy()
                    pattern["pattern_type"] = corrected_type
                    pattern["original_pattern_type"] = original_type
                    pattern["is_corrected"] = True
                    pattern["training_weight"] = 2.0  # Higher weight for corrected samples
                    corrected_count += 1
                    corrected_patterns.append(pattern)
                # If corrected to "no_pattern", skip it
                continue

            # Boost weight for confirmed patterns
            if pattern_id in feedback["confirmed_ids"]:
                pattern = pattern.copy()
                pattern["is_confirmed"] = True
                pattern["training_weight"] = 1.5  # Slightly higher weight

            corrected_patterns.append(pattern)

        logger.info(
            f"Applied feedback: {excluded_count} excluded, "
            f"{corrected_count} corrected, "
            f"{len(corrected_patterns)} patterns for training"
        )

        return corrected_patterns

    async def _fetch_pattern_history(self, limit: int = 1000) -> List[Dict]:
        """Fetch pattern history from Candlestick Service for training labels."""
        try:
            client = await self._get_client()
            url = f"{CANDLESTICK_SERVICE_URL}/api/v1/history"
            params = {"limit": limit}

            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                patterns = data.get("patterns", [])
                logger.info(f"Fetched {len(patterns)} patterns from history")
                return patterns
            else:
                logger.warning(f"Failed to fetch pattern history: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching pattern history: {e}")
            return []

    async def _run_training(self, job: TrainingJob):
        """Execute the training job."""
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(f"Starting training job {job.job_id}")

            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available")

            # Fetch training data
            logger.info(f"Fetching training data for {len(job.symbols)} symbols...")
            training_data = await self._fetch_training_data(
                symbols=job.symbols,
                timeframes=job.timeframes,
                lookback=500
            )

            if not training_data:
                raise RuntimeError("No training data available")

            logger.info(f"Fetched data for {len(training_data)} symbols")

            # Load user feedback for pattern corrections
            logger.info("Loading user feedback for training corrections...")
            user_feedback = self._load_user_feedback()

            # Fetch pattern history from Candlestick Service
            pattern_history = await self._fetch_pattern_history()

            # Apply user feedback corrections to pattern labels
            if pattern_history:
                corrected_patterns = self._apply_feedback_to_labels(
                    pattern_history, user_feedback
                )
                logger.info(f"Training with {len(corrected_patterns)} patterns "
                           f"(feedback applied: {user_feedback['statistics']['total']} entries)")
            else:
                corrected_patterns = []
                logger.warning("No pattern history available for supervised training")

            # TODO: Actual TCN training implementation
            # The corrected_patterns list contains:
            # - pattern_type: The (possibly corrected) pattern label
            # - training_weight: Higher weight for confirmed/corrected patterns
            # - is_corrected: True if user corrected the label
            # - is_confirmed: True if user confirmed the pattern

            # Simulate training progress (placeholder for actual TCN training)
            # In production, this would be replaced with actual model training
            for epoch in range(job.epochs):
                if job.status == TrainingStatus.CANCELLED:
                    logger.info(f"Training job {job.job_id} cancelled")
                    return

                # Simulate epoch training
                await asyncio.sleep(0.1)  # Placeholder for actual training

                # Update progress
                job.current_epoch = epoch + 1
                job.progress = (epoch + 1) / job.epochs * 100
                job.current_loss = 1.0 / (epoch + 1)  # Simulated decreasing loss

                if job.best_loss is None or job.current_loss < job.best_loss:
                    job.best_loss = job.current_loss

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Job {job.job_id}: Epoch {epoch + 1}/{job.epochs}, "
                        f"Loss: {job.current_loss:.4f}"
                    )

            # Save model (placeholder)
            model_filename = f"candlestick_model_{job.job_id}.pt"
            job.model_path = str(self._model_path / model_filename)

            # Create placeholder model file with feedback statistics
            model_info = {
                "job_id": job.job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "symbols": job.symbols,
                "timeframes": job.timeframes,
                "epochs": job.epochs,
                "final_loss": job.current_loss,
                "training_patterns": len(corrected_patterns),
                "feedback_applied": {
                    "total": user_feedback["statistics"]["total"],
                    "confirmed": user_feedback["statistics"]["confirmed"],
                    "corrected": user_feedback["statistics"]["corrected"],
                    "rejected": user_feedback["statistics"]["rejected"],
                },
            }
            with open(job.model_path + ".json", 'w') as f:
                json.dump(model_info, f, indent=2)

            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(f"Training job {job.job_id} completed successfully")

            # Notify Candlestick Service to reload model (optional)
            try:
                client = await self._get_client()
                await client.post(
                    f"{CANDLESTICK_SERVICE_URL}/api/v1/model/reload",
                    json={"model_path": job.model_path}
                )
            except Exception as e:
                logger.warning(f"Failed to notify Candlestick Service: {e}")

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_history()
            logger.error(f"Training job {job.job_id} failed: {e}")

        finally:
            self._current_job = None

    async def start_training(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> TrainingJob:
        """
        Start a new training job.

        Args:
            symbols: List of symbols to train on (default: all available)
            timeframes: List of timeframes to use (default: M15, H1, H4, D1)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate

        Returns:
            TrainingJob with job information
        """
        if self._current_job and self._current_job.status == TrainingStatus.RUNNING:
            raise RuntimeError("A training job is already running")

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available - training disabled")

        # Get available symbols if not specified
        if symbols is None:
            try:
                client = await self._get_client()
                response = await client.get(f"{DATA_SERVICE_URL}/api/v1/symbols")
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list):
                    symbols = [s.get("symbol", s) if isinstance(s, dict) else s for s in result[:10]]
                else:
                    symbols = ["BTCUSD", "EURUSD", "XAUUSD"]
            except Exception:
                symbols = ["BTCUSD", "EURUSD", "XAUUSD"]

        if timeframes is None:
            timeframes = ["M15", "H1", "H4", "D1"]

        # Create job
        job = TrainingJob(
            job_id=self._generate_job_id(),
            status=TrainingStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            timeframes=timeframes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        self._jobs[job.job_id] = job
        self._current_job = job
        self._save_history()

        # Start training in background
        self._training_task = asyncio.create_task(self._run_training(job))

        logger.info(f"Created training job {job.job_id}")
        return job

    async def cancel_training(self, job_id: str) -> bool:
        """Cancel a running training job."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        if job.status != TrainingStatus.RUNNING:
            return False

        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        self._save_history()

        if self._training_task and not self._training_task.done():
            self._training_task.cancel()

        logger.info(f"Cancelled training job {job_id}")
        return True

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get currently running job."""
        return self._current_job

    def get_all_jobs(self, limit: int = 50) -> List[TrainingJob]:
        """Get all training jobs."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def get_latest_model(self) -> Optional[str]:
        """Get path to the latest trained model."""
        # Find most recent completed job with model
        completed_jobs = [
            j for j in self._jobs.values()
            if j.status == TrainingStatus.COMPLETED and j.model_path
        ]
        if not completed_jobs:
            return None

        completed_jobs.sort(key=lambda j: j.completed_at, reverse=True)
        return completed_jobs[0].model_path

    def is_training(self) -> bool:
        """Check if a training job is currently running."""
        return self._current_job is not None and self._current_job.status == TrainingStatus.RUNNING


# Global singleton
training_service = TrainingService()
