"""HMM Training Service for dedicated training container.

This service handles HMM (Hidden Markov Model) and LightGBM Signal Scorer
training in a separate container from inference.

Architecture:
    - HMM Service (Port 3004): Inference only - fast regime detection API
    - HMM-Train Service (Port 3014): Training only - background model training
    - Shared volume: /app/data/models/hmm - trained models accessible by both

Training Types:
    1. HMM Models: Per-symbol regime detection (Bull/Bear/Sideways/High Volatility)
    2. LightGBM Scorer: Signal scoring model (trained across all symbols)
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

# Lazy imports for ML libraries
hmmlearn_hmm = None
lightgbm = None
joblib = None


def _load_hmmlearn():
    global hmmlearn_hmm, joblib
    if hmmlearn_hmm is None:
        try:
            from hmmlearn import hmm as h
            import joblib as jl
            hmmlearn_hmm = h
            joblib = jl
            return True
        except ImportError:
            logger.warning("hmmlearn not installed")
            return False
    return True


def _load_lightgbm():
    global lightgbm
    if lightgbm is None:
        try:
            import lightgbm as lgb
            lightgbm = lgb
            return True
        except ImportError:
            logger.warning("lightgbm not installed")
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


class ModelType(str, Enum):
    """Type of model to train."""
    HMM = "hmm"
    SCORER = "scorer"
    BOTH = "both"


class MarketRegime(str, Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class TrainingJob:
    """Training job information."""
    job_id: str
    status: TrainingStatus
    model_type: ModelType
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    symbols: List[str] = None
    timeframe: str = "1h"
    lookback_days: int = 365
    progress: float = 0.0
    current_symbol: Optional[str] = None
    completed_models: int = 0
    total_models: int = 0
    successful: int = 0
    failed: int = 0
    error_message: Optional[str] = None
    results: Dict[str, Any] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        d["model_type"] = self.model_type.value
        return d


# Service URLs
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
HMM_SERVICE_URL = os.getenv("HMM_SERVICE_URL", "http://trading-hmm:3004")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/hmm")


class HMMTrainingService:
    """
    Dedicated HMM Training Service.

    Trains:
    1. HMM models for market regime detection (per symbol)
    2. LightGBM signal scorer (across all symbols)
    """

    REGIMES = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.SIDEWAYS,
        MarketRegime.HIGH_VOLATILITY,
    ]

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

        logger.info(f"HMMTrainingService initialized (model_dir: {MODEL_DIR})")

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
                    for job_data in data[-50:]:
                        job_data["status"] = TrainingStatus(job_data["status"])
                        job_data["model_type"] = ModelType(job_data["model_type"])
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
        return f"hmm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    async def _fetch_training_data(
        self,
        symbol: str,
        timeframe: str = "1h",
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
            logger.error(f"Failed to fetch training data for {symbol}: {e}")
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

    async def _notify_hmm_service(self, symbol: str, model_path: str, model_type: str) -> bool:
        """Notify HMM inference service that a new model is available."""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{HMM_SERVICE_URL}/api/v1/model/reload",
                json={
                    "symbol": symbol,
                    "model_path": model_path,
                    "model_type": model_type
                },
                timeout=30.0
            )
            if response.status_code == 200:
                logger.info(f"HMM service notified of new model: {symbol}/{model_type}")
                return True
            else:
                logger.warning(f"HMM notification failed: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not notify HMM service: {e}")
            return False

    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract features for HMM training."""
        # Log returns
        returns = np.diff(np.log(prices))

        # Rolling volatility (20 periods)
        volatility = np.array([
            np.std(returns[max(0, i-20):i+1])
            for i in range(len(returns))
        ])

        # Trend strength (deviation from SMA)
        sma_period = 20
        sma = np.convolve(prices, np.ones(sma_period)/sma_period, mode='valid')
        sma_padded = np.concatenate([np.full(sma_period - 1, sma[0]), sma])
        trend_strength = (prices - sma_padded) / (sma_padded + 1e-8)

        # Combine features
        features = np.column_stack([
            returns,
            volatility,
            trend_strength[1:]
        ])

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _map_states_to_regimes(self, model, features: np.ndarray) -> Dict[int, MarketRegime]:
        """Map HMM states to market regimes based on state characteristics."""
        means = model.means_
        regime_mapping = {}

        state_chars = []
        for i in range(model.n_components):
            return_mean = means[i, 0] if means.shape[1] > 0 else 0
            vol_mean = means[i, 1] if means.shape[1] > 1 else 0
            state_chars.append({
                'state': i,
                'return': return_mean,
                'volatility': vol_mean
            })

        by_return = sorted(state_chars, key=lambda x: x['return'])
        by_vol = sorted(state_chars, key=lambda x: x['volatility'])
        assigned = set()

        # Highest volatility = HIGH_VOLATILITY
        high_vol_state = by_vol[-1]['state']
        regime_mapping[high_vol_state] = MarketRegime.HIGH_VOLATILITY
        assigned.add(high_vol_state)

        # Highest returns = BULL_TREND
        for s in reversed(by_return):
            if s['state'] not in assigned:
                regime_mapping[s['state']] = MarketRegime.BULL_TREND
                assigned.add(s['state'])
                break

        # Lowest returns = BEAR_TREND
        for s in by_return:
            if s['state'] not in assigned:
                regime_mapping[s['state']] = MarketRegime.BEAR_TREND
                assigned.add(s['state'])
                break

        # Remaining = SIDEWAYS
        for i in range(model.n_components):
            if i not in assigned:
                regime_mapping[i] = MarketRegime.SIDEWAYS

        return regime_mapping

    async def _train_hmm_model(
        self,
        symbol: str,
        data: List[Dict],
        job: TrainingJob
    ) -> Dict[str, Any]:
        """Train a single HMM model for regime detection."""
        if not _load_hmmlearn():
            return {"success": False, "error": "hmmlearn not available"}

        try:
            # Extract close prices
            closes = []
            for row in data:
                close = row.get("close") or row.get("h1_close") or row.get("d1_close")
                if close:
                    closes.append(float(close))

            if len(closes) < 100:
                return {
                    "success": False,
                    "error": f"Insufficient data: {len(closes)} < 100",
                    "samples": len(closes)
                }

            prices = np.array(closes, dtype=np.float64)
            features = self._extract_features(prices)

            start_time = datetime.now(timezone.utc)

            # Train HMM
            model = hmmlearn_hmm.GaussianHMM(
                n_components=4,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            model.fit(features)

            # Map states to regimes
            regime_mapping = self._map_states_to_regimes(model, features)

            # Save model
            model_filename = f"hmm_{symbol}.pkl"
            model_path = self._model_path / model_filename
            joblib.dump({
                "model": model,
                "regime_mapping": {k: v.value for k, v in regime_mapping.items()},
                "symbol": symbol,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "samples": len(features)
            }, model_path)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Notify inference service
            await self._notify_hmm_service(symbol, str(model_path), "hmm")

            return {
                "success": True,
                "samples": len(features),
                "duration_seconds": duration,
                "model_path": str(model_path)
            }

        except Exception as e:
            logger.error(f"HMM training failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    async def _train_scorer_model(
        self,
        symbols: List[str],
        data_by_symbol: Dict[str, List[Dict]],
        job: TrainingJob
    ) -> Dict[str, Any]:
        """Train LightGBM signal scorer across all symbols."""
        if not _load_lightgbm():
            return {"success": False, "error": "lightgbm not available"}

        try:
            start_time = datetime.now(timezone.utc)

            # Prepare training data
            X_all = []
            y_all = []

            for symbol, data in data_by_symbol.items():
                if len(data) < 50:
                    continue

                for i in range(20, len(data) - 5):
                    row = data[i]

                    # Extract features
                    features = [
                        row.get("rsi", 50) or 50,
                        row.get("macd_main", 0) or 0,
                        row.get("adx_main", 25) or 25,
                        row.get("atr_pct_d1", 1) or 1,
                        row.get("strength_1d", 0) or 0,
                    ]

                    # Calculate target (future returns)
                    close_now = float(row.get("close") or row.get("h1_close") or 0)
                    close_future = float(data[i + 5].get("close") or data[i + 5].get("h1_close") or 0)

                    if close_now > 0 and close_future > 0:
                        future_return = (close_future - close_now) / close_now
                        # Convert to signal: 1 = buy, 0 = hold, -1 = sell
                        if future_return > 0.01:
                            y_all.append(1)
                        elif future_return < -0.01:
                            y_all.append(-1)
                        else:
                            y_all.append(0)
                        X_all.append(features)

            if len(X_all) < 100:
                return {
                    "success": False,
                    "error": f"Insufficient training samples: {len(X_all)}",
                    "samples": len(X_all)
                }

            X = np.array(X_all)
            y = np.array(y_all)

            # Train LightGBM
            train_data = lightgbm.Dataset(X, label=y + 1)  # Shift to 0, 1, 2
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "verbose": -1
            }
            model = lightgbm.train(params, train_data, num_boost_round=100)

            # Save model
            model_filename = "scorer_lightgbm.pkl"
            model_path = self._model_path / model_filename
            joblib.dump({
                "model": model,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "samples": len(X),
                "symbols_used": list(data_by_symbol.keys())
            }, model_path)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Notify inference service
            await self._notify_hmm_service("scorer", str(model_path), "scorer")

            return {
                "success": True,
                "samples": len(X),
                "duration_seconds": duration,
                "model_path": str(model_path),
                "symbols_used": len(data_by_symbol)
            }

        except Exception as e:
            logger.error(f"Scorer training failed: {e}")
            return {"success": False, "error": str(e)}

    async def _run_training(self, job: TrainingJob):
        """Execute the training job."""
        try:
            job.status = TrainingStatus.PREPARING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(f"Starting training job {job.job_id} (type: {job.model_type.value})")

            # Get symbols
            symbols = job.symbols
            if not symbols:
                symbols = await self._get_available_symbols()
                if not symbols:
                    raise RuntimeError("No symbols available for training")
                job.symbols = symbols[:30]  # Limit symbols

            job.total_models = len(job.symbols) if job.model_type != ModelType.SCORER else 1
            if job.model_type == ModelType.BOTH:
                job.total_models += 1  # Add scorer

            job.results = {}
            job.status = TrainingStatus.TRAINING
            self._save_history()

            data_by_symbol = {}

            # Train HMM models if requested
            if job.model_type in [ModelType.HMM, ModelType.BOTH]:
                for symbol in job.symbols:
                    if self._stop_requested:
                        job.status = TrainingStatus.CANCELLED
                        break

                    job.current_symbol = symbol

                    try:
                        data = await self._fetch_training_data(
                            symbol, job.timeframe, job.lookback_days
                        )
                        data_by_symbol[symbol] = data

                        if not data:
                            job.results[f"hmm_{symbol}"] = {
                                "success": False,
                                "error": "No training data"
                            }
                            job.failed += 1
                        else:
                            result = await self._train_hmm_model(symbol, data, job)
                            job.results[f"hmm_{symbol}"] = result

                            if result.get("success"):
                                job.successful += 1
                            else:
                                job.failed += 1

                    except Exception as e:
                        logger.error(f"Error training HMM for {symbol}: {e}")
                        job.results[f"hmm_{symbol}"] = {"success": False, "error": str(e)}
                        job.failed += 1

                    job.completed_models += 1
                    job.progress = job.completed_models / job.total_models * 100
                    await asyncio.sleep(0)

            # Train scorer if requested
            if job.model_type in [ModelType.SCORER, ModelType.BOTH] and not self._stop_requested:
                job.current_symbol = "scorer"

                # Fetch data for scorer if not already fetched
                if not data_by_symbol:
                    for symbol in job.symbols[:10]:  # Use 10 symbols for scorer
                        data = await self._fetch_training_data(
                            symbol, job.timeframe, job.lookback_days
                        )
                        if data:
                            data_by_symbol[symbol] = data

                if data_by_symbol:
                    result = await self._train_scorer_model(job.symbols, data_by_symbol, job)
                    job.results["scorer"] = result

                    if result.get("success"):
                        job.successful += 1
                    else:
                        job.failed += 1
                else:
                    job.results["scorer"] = {"success": False, "error": "No data available"}
                    job.failed += 1

                job.completed_models += 1
                job.progress = 100

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
        model_type: ModelType = ModelType.BOTH,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h",
        lookback_days: int = 365
    ) -> TrainingJob:
        """Start a new training job."""
        if self._current_job and self._current_job.status == TrainingStatus.TRAINING:
            raise RuntimeError("A training job is already running")

        job = TrainingJob(
            job_id=self._generate_job_id(),
            status=TrainingStatus.PENDING,
            model_type=model_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            timeframe=timeframe,
            lookback_days=lookback_days,
            results={}
        )

        self._jobs[job.job_id] = job
        self._current_job = job
        self._stop_requested = False
        self._save_history()

        # Start training in background
        self._training_task = asyncio.create_task(self._run_training(job))

        logger.info(f"Created training job {job.job_id} (type: {model_type.value})")
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
            "service": "hmm-train",
            "status": "training" if current and current.status == TrainingStatus.TRAINING else "idle",
            "current_job": current.to_dict() if current else None,
            "jobs_in_history": len(self._jobs),
            "model_directory": str(self._model_path),
            "hmmlearn_available": _load_hmmlearn(),
            "lightgbm_available": _load_lightgbm()
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
training_service = HMMTrainingService()
