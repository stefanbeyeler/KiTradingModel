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

Validation Pipeline (NEW):
    - After training, models are validated against held-out data
    - A/B comparison against current production models
    - Automatic deployment if quality improves or stays stable
    - Automatic rejection if quality regresses beyond threshold
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
    global lightgbm, joblib
    if lightgbm is None:
        try:
            import lightgbm as lgb
            import joblib as jl
            lightgbm = lgb
            joblib = jl
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
    # Validation pipeline fields
    version_id: Optional[str] = None
    validation_metrics: Dict[str, Any] = None
    deployment_decisions: Dict[str, Any] = None
    deployed_count: int = 0
    rejected_count: int = 0

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

# Validation settings
ENABLE_VALIDATION = os.getenv("HMM_ENABLE_VALIDATION", "true").lower() == "true"
VALIDATION_SPLIT = float(os.getenv("HMM_VALIDATION_SPLIT", "0.2"))


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

        # Initialize validation services (lazy load)
        self._model_registry = None
        self._validation_service = None
        self._rollback_service = None
        self._validation_enabled = ENABLE_VALIDATION

        logger.info(f"HMMTrainingService initialized (model_dir: {MODEL_DIR}, validation: {ENABLE_VALIDATION})")

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
        """Extract features for HMM training with improved numerical stability."""
        # Log returns
        returns = np.diff(np.log(prices + 1e-10))  # Add small epsilon to prevent log(0)

        # Rolling volatility (20 periods)
        volatility = np.array([
            np.std(returns[max(0, i-20):i+1]) if i >= 20 else np.std(returns[:i+1])
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

        # Clean NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize features for numerical stability
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Add small random noise to prevent singular covariance matrices
        # This helps with numerical stability during HMM training
        noise = np.random.randn(*features.shape) * 1e-6
        features = features + noise

        return features

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

            # Train HMM with improved parameters for numerical stability
            model = hmmlearn_hmm.GaussianHMM(
                n_components=4,
                covariance_type="diag",  # Diagonal covariance is more stable than full
                n_iter=200,  # More iterations for better convergence
                random_state=42,
                tol=1e-3,  # Convergence tolerance
                min_covar=1e-3,  # Minimum covariance to prevent singular matrices
                init_params="stmc",  # Initialize all parameters
                params="stmc"  # Update all parameters
            )
            model.fit(features)

            # Map states to regimes
            regime_mapping = self._map_states_to_regimes(model, features)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Determine save path based on validation mode
            if self._validation_enabled and job.version_id:
                # Save to version directory
                version_path = self._get_version_path(job.version_id)
                version_path.mkdir(parents=True, exist_ok=True)
                model_filename = f"hmm_{symbol}.pkl"
                model_path = version_path / model_filename
            else:
                # Legacy: save directly to model directory
                model_filename = f"hmm_{symbol}.pkl"
                model_path = self._model_path / model_filename

            # Save model
            joblib.dump({
                "model": model,
                "regime_mapping": {k: v.value for k, v in regime_mapping.items()},
                "symbol": symbol,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "samples": len(features),
                "version_id": job.version_id
            }, model_path)

            result = {
                "success": True,
                "samples": len(features),
                "duration_seconds": duration,
                "model_path": str(model_path)
            }

            # If validation is disabled, notify inference service directly
            if not self._validation_enabled:
                await self._notify_hmm_service(symbol, str(model_path), "hmm")

            return result

        except Exception as e:
            logger.error(f"HMM training failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    async def _train_scorer_model(
        self,
        symbols: List[str],
        data_by_symbol: Dict[str, List[Dict]],
        job: TrainingJob
    ) -> Dict[str, Any]:
        """Train LightGBM signal scorer across all symbols.

        Uses the shared LightGBMSignalScorer to ensure feature consistency
        between training and inference (18 features).
        """
        if not _load_lightgbm():
            return {"success": False, "error": "lightgbm not available"}

        try:
            # Import the shared scorer
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
            from src.shared.lightgbm_scorer import LightGBMSignalScorer

            start_time = datetime.now(timezone.utc)
            scorer = LightGBMSignalScorer()

            # Prepare training data using OHLCV arrays
            X_all = []
            y_all = []

            for symbol, data in data_by_symbol.items():
                if len(data) < 100:
                    logger.debug(f"Skipping {symbol}: insufficient data ({len(data)} rows)")
                    continue

                # Convert to OHLCV numpy array
                ohlcv = np.array([
                    [
                        float(row.get("open") or row.get("h1_open") or 0),
                        float(row.get("high") or row.get("h1_high") or 0),
                        float(row.get("low") or row.get("h1_low") or 0),
                        float(row.get("close") or row.get("h1_close") or 0),
                        float(row.get("volume") or row.get("h1_volume") or 0)
                    ]
                    for row in data
                ], dtype=np.float64)

                # Generate training samples with sliding window
                for i in range(60, len(ohlcv) - 20):
                    window = ohlcv[i-60:i]

                    # Create regime probabilities (estimated from price action)
                    closes = window[:, 3]
                    returns = np.diff(closes) / closes[:-1]
                    avg_return = np.mean(returns[-20:])
                    volatility = np.std(returns[-20:])

                    # Estimate regime probabilities
                    if avg_return > 0.001 and volatility < 0.03:
                        regime_probs = {"bull_trend": 0.6, "bear_trend": 0.1, "sideways": 0.2, "high_volatility": 0.1}
                    elif avg_return < -0.001 and volatility < 0.03:
                        regime_probs = {"bull_trend": 0.1, "bear_trend": 0.6, "sideways": 0.2, "high_volatility": 0.1}
                    elif volatility > 0.03:
                        regime_probs = {"bull_trend": 0.15, "bear_trend": 0.15, "sideways": 0.2, "high_volatility": 0.5}
                    else:
                        regime_probs = {"bull_trend": 0.2, "bear_trend": 0.2, "sideways": 0.5, "high_volatility": 0.1}

                    # Calculate features using shared scorer (18 features)
                    try:
                        features = scorer.calculate_features(window, regime_probs, regime_duration=10)
                    except Exception as e:
                        logger.debug(f"Feature calculation failed at {i}: {e}")
                        continue

                    # Calculate target (future returns -> signal score 0-100)
                    close_now = ohlcv[i, 3]
                    close_future = ohlcv[i + 20, 3]

                    if close_now > 0 and close_future > 0:
                        future_return = (close_future - close_now) / close_now

                        # Convert return to signal quality score (0-100)
                        # Positive returns = good long signal, negative = good short signal
                        if future_return > 0.02:
                            score = 80 + min(future_return * 500, 20)  # 80-100 for strong long
                        elif future_return > 0:
                            score = 50 + future_return * 1500  # 50-80 for weak long
                        elif future_return > -0.02:
                            score = 50 + future_return * 1500  # 20-50 for weak short
                        else:
                            score = max(0, 20 + future_return * 500)  # 0-20 for strong short

                        X_all.append(features)
                        y_all.append(score)

                logger.debug(f"Generated {len(X_all)} samples from {symbol}")

            if len(X_all) < 100:
                return {
                    "success": False,
                    "error": f"Insufficient training samples: {len(X_all)}",
                    "samples": len(X_all)
                }

            X = np.array(X_all)
            y = np.array(y_all)

            logger.info(f"Training scorer with {len(X)} samples, {X.shape[1]} features")

            # Split for validation
            n_samples = len(X)
            n_val = int(n_samples * 0.2)
            indices = np.random.permutation(n_samples)

            train_X = X[indices[n_val:]]
            train_y = y[indices[n_val:]]
            val_X = X[indices[:n_val]]
            val_y = y[indices[:n_val]]

            # Train using the shared scorer's fit method
            scorer.fit(train_X, train_y, eval_set=(val_X, val_y))

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Determine save path based on validation mode
            if self._validation_enabled and job.version_id:
                version_path = self._get_version_path(job.version_id)
                version_path.mkdir(parents=True, exist_ok=True)
                model_path = version_path / "scorer_lightgbm.pkl"
            else:
                model_path = self._model_path / "scorer_lightgbm.pkl"

            # Save model using scorer's save method
            scorer.save(str(model_path))

            result = {
                "success": True,
                "samples": len(X),
                "features": X.shape[1],
                "duration_seconds": duration,
                "model_path": str(model_path),
                "symbols_used": len(data_by_symbol),
                "is_fitted": scorer.is_fitted()
            }

            # If validation is disabled, notify inference service directly
            if not self._validation_enabled:
                await self._notify_hmm_service("scorer", str(model_path), "scorer")

            return result

        except Exception as e:
            logger.error(f"Scorer training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    async def _run_training(self, job: TrainingJob):
        """Execute the training job with optional validation pipeline."""
        try:
            job.status = TrainingStatus.PREPARING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(f"Starting training job {job.job_id} (type: {job.model_type.value})")

            # Create version for validation if enabled
            if self._validation_enabled:
                job.version_id = self._get_registry().create_version(job.job_id)
                logger.info(f"Created version {job.version_id} for validation")

            # Get symbols - use ALL available symbols from Data Service
            symbols = job.symbols
            if not symbols:
                symbols = await self._get_available_symbols()
                if not symbols:
                    raise RuntimeError("No symbols available for training")
                job.symbols = symbols  # Train ALL symbols

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
                # Fetch data for scorer if not already fetched
                if not data_by_symbol:
                    # Update total_models to reflect data fetching progress
                    job.total_models = len(job.symbols)
                    job.completed_models = 0

                    for idx, symbol in enumerate(job.symbols):  # Use ALL symbols for scorer
                        if self._stop_requested:
                            break
                        job.current_symbol = f"Daten: {symbol}"
                        data = await self._fetch_training_data(
                            symbol, job.timeframe, job.lookback_days
                        )
                        if data:
                            data_by_symbol[symbol] = data
                        job.completed_models = idx + 1
                        job.progress = job.completed_models / job.total_models * 50  # 0-50% for data
                        await asyncio.sleep(0)

                job.current_symbol = "scorer"
                job.progress = 50  # Data fetched, now training

                if data_by_symbol:
                    job.current_symbol = f"Training ({len(data_by_symbol)} Symbole)"
                    result = await self._train_scorer_model(job.symbols, data_by_symbol, job)
                    job.results["scorer"] = result

                    if result.get("success"):
                        job.successful += 1
                    else:
                        job.failed += 1
                else:
                    job.results["scorer"] = {"success": False, "error": "No data available"}
                    job.failed += 1

                job.current_symbol = "scorer"
                job.completed_models = job.total_models
                job.progress = 100

            # Run validation pipeline if enabled
            if self._validation_enabled and job.version_id and job.status != TrainingStatus.CANCELLED:
                await self._run_validation_pipeline(job, data_by_symbol)

            if job.status != TrainingStatus.CANCELLED:
                job.status = TrainingStatus.COMPLETED

            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(
                f"Training job {job.job_id} {job.status.value}: "
                f"{job.successful} successful, {job.failed} failed, "
                f"{job.deployed_count} deployed, {job.rejected_count} rejected"
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

    # =========================================================================
    # Validation Pipeline Methods
    # =========================================================================

    def _get_registry(self):
        """Lazy load model registry."""
        if self._model_registry is None:
            from .model_registry import model_registry
            self._model_registry = model_registry
        return self._model_registry

    def _get_validation_service(self):
        """Lazy load validation service."""
        if self._validation_service is None:
            from .validation_service import validation_service
            self._validation_service = validation_service
        return self._validation_service

    def _get_rollback_service(self):
        """Lazy load rollback service."""
        if self._rollback_service is None:
            from .rollback_service import rollback_service
            self._rollback_service = rollback_service
        return self._rollback_service

    def _get_version_path(self, version_id: str) -> Path:
        """Get the path for a version directory."""
        return self._model_path / "versions" / version_id

    async def _run_validation_pipeline(
        self,
        job: TrainingJob,
        data_by_symbol: Dict[str, List[Dict]]
    ):
        """
        Run validation and A/B comparison for trained models.

        For each successfully trained model:
        1. Register in model registry
        2. Run validation metrics
        3. Compare against production (A/B)
        4. Deploy or reject based on comparison
        """
        logger.info(f"Running validation pipeline for job {job.job_id}")

        registry = self._get_registry()
        rollback_svc = self._get_rollback_service()

        job.validation_metrics = {}
        job.deployment_decisions = {}
        job.deployed_count = 0
        job.rejected_count = 0

        # Process HMM models
        if job.model_type in [ModelType.HMM, ModelType.BOTH]:
            for symbol in job.symbols:
                result_key = f"hmm_{symbol}"
                result = job.results.get(result_key, {})

                if not result.get("success"):
                    continue

                try:
                    model_path = result.get("model_path")

                    # Register model in registry
                    registry.register_model(
                        version_id=job.version_id,
                        model_type="hmm",
                        symbol=symbol,
                        training_job_id=job.job_id,
                        samples_used=result.get("samples", 0),
                        training_duration=result.get("duration_seconds", 0),
                        timeframe=job.timeframe,
                        model_path=f"{job.version_id}/hmm_{symbol}.pkl"
                    )

                    # Run validation and deployment decision
                    decision = await rollback_svc.process_training_result(
                        version_id=job.version_id,
                        model_type="hmm",
                        symbol=symbol,
                        candidate_path=model_path,
                        timeframe=job.timeframe,
                        training_job_id=job.job_id
                    )

                    # Record results
                    job.deployment_decisions[result_key] = {
                        "action": decision.action,
                        "reason": decision.reason
                    }

                    if decision.action == "deployed":
                        job.deployed_count += 1
                    elif decision.action == "rejected":
                        job.rejected_count += 1

                    # Store validation metrics from comparison
                    if decision.comparison_result:
                        job.validation_metrics[result_key] = decision.comparison_result.get(
                            "candidate_metrics", {}
                        )

                    logger.info(f"HMM {symbol}: {decision.action} - {decision.reason}")

                except Exception as e:
                    logger.error(f"Validation failed for HMM {symbol}: {e}")
                    job.deployment_decisions[result_key] = {
                        "action": "error",
                        "reason": str(e)
                    }

        # Process scorer model
        if job.model_type in [ModelType.SCORER, ModelType.BOTH]:
            result = job.results.get("scorer", {})

            if result.get("success"):
                try:
                    model_path = result.get("model_path")

                    # Register model
                    registry.register_model(
                        version_id=job.version_id,
                        model_type="scorer",
                        symbol=None,
                        training_job_id=job.job_id,
                        samples_used=result.get("samples", 0),
                        training_duration=result.get("duration_seconds", 0),
                        timeframe=job.timeframe,
                        model_path=f"{job.version_id}/scorer_lightgbm.pkl"
                    )

                    # Run validation and deployment
                    decision = await rollback_svc.process_training_result(
                        version_id=job.version_id,
                        model_type="scorer",
                        symbol=None,
                        candidate_path=model_path,
                        timeframe=job.timeframe,
                        symbols_for_scorer=job.symbols,  # Use ALL symbols
                        training_job_id=job.job_id
                    )

                    job.deployment_decisions["scorer"] = {
                        "action": decision.action,
                        "reason": decision.reason
                    }

                    if decision.action == "deployed":
                        job.deployed_count += 1
                    elif decision.action == "rejected":
                        job.rejected_count += 1

                    if decision.comparison_result:
                        job.validation_metrics["scorer"] = decision.comparison_result.get(
                            "candidate_metrics", {}
                        )

                    logger.info(f"Scorer: {decision.action} - {decision.reason}")

                except Exception as e:
                    logger.error(f"Validation failed for scorer: {e}")
                    job.deployment_decisions["scorer"] = {
                        "action": "error",
                        "reason": str(e)
                    }

        # Cleanup old versions
        try:
            deleted = registry.cleanup_old_versions()
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old model versions")
        except Exception as e:
            logger.warning(f"Version cleanup failed: {e}")

        logger.info(
            f"Validation pipeline complete: {job.deployed_count} deployed, "
            f"{job.rejected_count} rejected"
        )

    # =========================================================================
    # Validation Query Methods
    # =========================================================================

    def get_model_versions(
        self,
        model_type: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get model version history."""
        if not self._validation_enabled:
            return []

        registry = self._get_registry()
        versions = registry.list_versions(model_type=model_type, symbol=symbol, limit=limit)
        return [v.to_dict() for v in versions]

    def get_production_models(self) -> Dict[str, Any]:
        """Get all current production models."""
        if not self._validation_enabled:
            return {}

        registry = self._get_registry()
        production = registry.get_production_versions()
        return {k: v.to_dict() for k, v in production.items()}

    def get_deployment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get deployment decision history."""
        if not self._validation_enabled:
            return []

        rollback_svc = self._get_rollback_service()
        decisions = rollback_svc.get_deployment_history(limit=limit)
        return [d.to_dict() for d in decisions]

    async def rollback_model(
        self,
        model_type: str,
        symbol: Optional[str],
        target_version: str,
        reason: str = "Manual rollback"
    ) -> Dict[str, Any]:
        """Manually rollback to a previous version."""
        if not self._validation_enabled:
            return {"success": False, "error": "Validation not enabled"}

        rollback_svc = self._get_rollback_service()
        decision = await rollback_svc.rollback_to_version(
            model_type=model_type,
            symbol=symbol,
            target_version=target_version,
            reason=reason
        )
        return decision.to_dict()

    async def force_deploy_version(
        self,
        version_id: str,
        model_type: str,
        symbol: Optional[str],
        reason: str = "Manual deployment"
    ) -> Dict[str, Any]:
        """Force deploy a model version (bypass validation)."""
        if not self._validation_enabled:
            return {"success": False, "error": "Validation not enabled"}

        rollback_svc = self._get_rollback_service()
        decision = await rollback_svc.force_deploy(
            version_id=version_id,
            model_type=model_type,
            symbol=symbol,
            reason=reason
        )
        return decision.to_dict()


# Global singleton
training_service = HMMTrainingService()
