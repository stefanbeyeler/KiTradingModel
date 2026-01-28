"""
Training Service für CNN-LSTM Multi-Task Model.

Verwaltet Training-Jobs, Model-Checkpointing und Historie.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..models.training_schemas import (
    TrainingConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainingRequest,
    TrainingStatus,
    TrainingStatusResponse,
    TrainingHistoryItem,
)
from .data_pipeline import data_pipeline
from .multi_task_loss import LossWeights, create_multi_task_loss

# Lazy import fuer PyTorch
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch


class CUDATrainingError(Exception):
    """Custom exception for CUDA errors during training."""
    pass


def _check_cuda_health() -> tuple[bool, str]:
    """
    Check if CUDA is healthy and usable for training.

    Returns:
        Tuple of (is_healthy: bool, message: str)
    """
    _ensure_torch()

    if not torch.cuda.is_available():
        return True, "CUDA not available, using CPU"

    try:
        # Test basic CUDA operations
        torch.cuda.synchronize()

        # Check memory availability
        free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # GB
        if free_memory < 0.5:  # Less than 500MB free
            return False, f"Insufficient GPU memory: {free_memory:.2f}GB free"

        return True, f"CUDA healthy, {free_memory:.2f}GB free"

    except RuntimeError as e:
        return False, f"CUDA error: {e}"


def _cleanup_gpu_memory():
    """Clean up GPU memory after training or on error."""
    _ensure_torch()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned up")
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")


# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/cnn-lstm")
CNN_LSTM_SERVICE_URL = os.getenv("CNN_LSTM_SERVICE_URL", "http://trading-cnn-lstm:3007")
HISTORY_FILE = "training_history.json"
MAX_HISTORY_ITEMS = 50


class TrainingService:
    """
    Training Service fuer CNN-LSTM Multi-Task Model.

    Features:
    - Asynchrones Training
    - Model Checkpointing
    - Early Stopping
    - Training History
    - Watchdog Integration
    """

    def __init__(self):
        self._current_job: Optional[dict] = None
        self._training_task: Optional[asyncio.Task] = None
        self._cancel_requested: bool = False
        self._history: list[TrainingHistoryItem] = []
        self._http_client = None

    async def _get_http_client(self):
        """Lazy initialization des HTTP-Clients."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def is_training(self) -> bool:
        """Prueft ob Training laeuft."""
        return self._current_job is not None

    def get_current_status(self) -> TrainingStatusResponse:
        """Gibt aktuellen Training-Status zurueck."""
        if self._current_job is None:
            return TrainingStatusResponse(status=TrainingStatus.IDLE)

        return TrainingStatusResponse(
            job_id=self._current_job.get("job_id"),
            status=TrainingStatus(self._current_job.get("status", "idle")),
            started_at=self._current_job.get("started_at"),
            progress=self._current_job.get("progress"),
            metrics=self._current_job.get("metrics"),
            error_message=self._current_job.get("error_message")
        )

    # =========================================================================
    # Training Job Management
    # =========================================================================

    async def start_training(self, request: TrainingRequest) -> TrainingStatusResponse:
        """
        Startet einen neuen Training-Job.

        Args:
            request: Training-Request mit Symbolen, Timeframes, etc.

        Returns:
            TrainingStatusResponse mit Job-Status
        """
        if self.is_training():
            return TrainingStatusResponse(
                status=TrainingStatus.PENDING,
                error_message="Training already in progress"
            )

        # Erstelle Job
        job_id = f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self._current_job = {
            "job_id": job_id,
            "status": TrainingStatus.PENDING.value,
            "started_at": datetime.now(timezone.utc),
            "request": request.model_dump(),
            "progress": TrainingProgress(
                current_epoch=0,
                total_epochs=request.epochs,
                symbols_completed=0,
                total_symbols=len(request.symbols)
            ).model_dump(),
            "metrics": None,
            "error_message": None
        }

        # Starte Training in Background Task
        self._cancel_requested = False
        self._training_task = asyncio.create_task(
            self._run_training(job_id, request)
        )

        # Update main module status
        try:
            from ..main import set_training_in_progress
            set_training_in_progress(True)
        except ImportError:
            pass

        logger.info(f"Started training job: {job_id}")

        return self.get_current_status()

    async def cancel_training(self) -> TrainingStatusResponse:
        """Bricht laufendes Training ab."""
        if not self.is_training():
            return TrainingStatusResponse(
                status=TrainingStatus.IDLE,
                error_message="No training in progress"
            )

        self._cancel_requested = True
        logger.info(f"Cancellation requested for job: {self._current_job['job_id']}")

        # Warte kurz auf Abbruch
        if self._training_task:
            try:
                await asyncio.wait_for(self._training_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._training_task.cancel()

        return self.get_current_status()

    # =========================================================================
    # Training Loop
    # =========================================================================

    async def _run_training(self, job_id: str, request: TrainingRequest):
        """
        Haupttraining-Loop mit robustem CUDA Error Handling.

        Args:
            job_id: Job-ID
            request: Training-Request
        """
        _ensure_torch()
        device = "cpu"  # Default fallback

        try:
            self._update_status(TrainingStatus.PREPARING)
            logger.info(f"Preparing training for {len(request.symbols)} symbols")

            # Check CUDA health before training
            cuda_healthy, cuda_msg = _check_cuda_health()
            if not cuda_healthy:
                logger.warning(f"CUDA not healthy: {cuda_msg}")
                self._current_job["error_message"] = f"CUDA check failed: {cuda_msg}"
                # Continue with CPU

            # Erstelle Model
            from ...cnn_lstm_app.models.cnn_lstm_model import (
                CNNLSTMConfig,
                create_cnn_lstm_model,
                save_model
            )

            config = CNNLSTMConfig(
                input_features=25,
                sequence_length=data_pipeline.get_sequence_length(request.timeframes[0])
            )
            model = create_cnn_lstm_model(config)

            # Determine device with CUDA validation
            if torch.cuda.is_available() and cuda_healthy:
                try:
                    # Clear GPU memory before loading model
                    torch.cuda.empty_cache()
                    device = "cuda"
                    model = model.to(device)
                    # Verify model is on GPU
                    torch.cuda.synchronize()
                    logger.info(f"Model created on GPU: {torch.cuda.get_device_name(0)}")
                except RuntimeError as cuda_err:
                    logger.warning(f"Failed to use CUDA, falling back to CPU: {cuda_err}")
                    device = "cpu"
                    model = model.to(device)
            else:
                device = "cpu"
                model = model.to(device)
                logger.info("Model created on CPU")

            # Loss und Optimizer
            loss_fn = create_multi_task_loss(LossWeights(
                price=request.price_weight,
                pattern=request.pattern_weight,
                regime=request.regime_weight
            ))

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=request.learning_rate,
                weight_decay=1e-5
            )

            # Scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=request.epochs, eta_min=1e-6
            )

            # Training Loop ueber alle Symbole und Timeframes
            best_loss = float('inf')
            patience_counter = 0

            for symbol_idx, symbol in enumerate(request.symbols):
                if self._cancel_requested:
                    break

                for timeframe in request.timeframes:
                    if self._cancel_requested:
                        break

                    logger.info(f"Training {symbol} {timeframe}")
                    self._update_progress(
                        current_symbol=symbol,
                        symbols_completed=symbol_idx
                    )

                    # Lade Daten
                    dataset = await data_pipeline.create_training_dataset(
                        symbol, timeframe, days=365
                    )

                    if dataset is None:
                        logger.warning(f"No data for {symbol} {timeframe}, skipping")
                        continue

                    # Erstelle DataLoader
                    train_loader, val_loader = self._create_dataloaders(
                        dataset, request.batch_size, request.validation_split
                    )

                    # Epochs
                    self._update_status(TrainingStatus.TRAINING)

                    for epoch in range(request.epochs):
                        if self._cancel_requested:
                            break

                        # Training Epoch
                        train_loss = await self._train_epoch(
                            model, train_loader, loss_fn, optimizer, device
                        )

                        # Validation
                        val_loss, val_metrics = await self._validate_epoch(
                            model, val_loader, loss_fn, device
                        )

                        scheduler.step()

                        # Update Progress
                        self._update_progress(
                            current_epoch=epoch + 1,
                            current_loss=train_loss.total,
                            best_loss=min(best_loss, val_loss.total)
                        )

                        self._update_metrics(TrainingMetrics(
                            total_loss=train_loss.total,
                            price_loss=train_loss.price,
                            pattern_loss=train_loss.pattern,
                            regime_loss=train_loss.regime,
                            val_total_loss=val_loss.total,
                            val_price_loss=val_loss.price,
                            val_pattern_loss=val_loss.pattern,
                            val_regime_loss=val_loss.regime,
                            price_direction_accuracy=train_loss.direction_accuracy
                        ))

                        # Early Stopping
                        if val_loss.total < best_loss:
                            best_loss = val_loss.total
                            patience_counter = 0

                            # Speichere bestes Modell
                            if request.save_best_only:
                                model_path = Path(MODEL_DIR) / f"{symbol}_{timeframe}_best.pt"
                                save_model(model, str(model_path), {
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "epoch": epoch,
                                    "loss": best_loss,
                                    "job_id": job_id
                                })
                        else:
                            patience_counter += 1

                        if patience_counter >= request.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break

                        # Log Progress
                        if (epoch + 1) % 10 == 0:
                            logger.info(
                                f"Epoch {epoch+1}/{request.epochs} - "
                                f"Train: {train_loss.total:.4f}, Val: {val_loss.total:.4f}"
                            )

            # Training abgeschlossen
            if self._cancel_requested:
                self._update_status(TrainingStatus.CANCELLED)
                logger.info(f"Training cancelled: {job_id}")
            else:
                # Speichere finales Modell
                self._update_status(TrainingStatus.SAVING)
                final_path = Path(MODEL_DIR) / f"cnn_lstm_{job_id}.pt"
                save_model(model, str(final_path), {
                    "job_id": job_id,
                    "symbols": request.symbols,
                    "timeframes": request.timeframes,
                    "final_loss": best_loss,
                    "epochs_completed": request.epochs
                })

                # Update latest symlink
                latest_path = Path(MODEL_DIR) / "latest.pt"
                if latest_path.exists():
                    latest_path.unlink()
                latest_path.symlink_to(final_path.name)

                self._update_status(TrainingStatus.COMPLETED)
                self._current_job["model_path"] = str(final_path)
                logger.info(f"Training completed: {job_id}")

                # Benachrichtige Inference Service
                await self._notify_inference_service()

        except RuntimeError as e:
            error_str = str(e).lower()
            is_cuda_error = any(pattern in error_str for pattern in [
                "cuda", "gpu", "illegal memory access", "out of memory",
                "device-side assert", "cublas", "cudnn"
            ])

            if is_cuda_error:
                logger.error(f"CUDA training error: {e}")
                self._current_job["error_message"] = f"CUDA Error: {e}"
                self._current_job["cuda_error"] = True
                # Attempt GPU recovery
                _cleanup_gpu_memory()
            else:
                logger.error(f"Training runtime error: {e}")
                self._current_job["error_message"] = str(e)

            self._update_status(TrainingStatus.FAILED)

        except Exception as e:
            logger.error(f"Training error: {e}")
            self._update_status(TrainingStatus.FAILED)
            self._current_job["error_message"] = str(e)

        finally:
            # Clean up GPU memory after training
            _cleanup_gpu_memory()

            # Speichere in Historie
            await self._save_to_history()

            # Reset Status
            try:
                from ..main import set_training_in_progress
                set_training_in_progress(False)
            except ImportError:
                pass

    async def _train_epoch(self, model, dataloader, loss_fn, optimizer, device):
        """Trainiert eine Epoche mit CUDA Error Handling."""
        model.train()
        total_loss = None
        batch_count = 0

        for batch in dataloader:
            try:
                features = batch['features'].to(device, non_blocking=True)
                targets = {
                    'price': batch['price'].to(device, non_blocking=True),
                    'patterns': batch['patterns'].to(device, non_blocking=True),
                    'regime': batch['regime'].to(device, non_blocking=True)
                }

                optimizer.zero_grad()
                predictions = model(features)
                loss, components = loss_fn(predictions, targets)

                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected in batch {batch_count}, skipping")
                    continue

                loss.backward()

                # Check for NaN in gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    logger.warning(f"NaN gradient detected in batch {batch_count}, skipping")
                    optimizer.zero_grad()
                    continue

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Periodic GPU sync to catch deferred errors
                if device == "cuda" and batch_count % 50 == 0:
                    torch.cuda.synchronize()

                if total_loss is None:
                    total_loss = components
                else:
                    # Averaging
                    total_loss.total = (total_loss.total + components.total) / 2
                    total_loss.price = (total_loss.price + components.price) / 2
                    total_loss.pattern = (total_loss.pattern + components.pattern) / 2
                    total_loss.regime = (total_loss.regime + components.regime) / 2

                batch_count += 1

            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str:
                    logger.error(f"GPU OOM in batch {batch_count}, attempting recovery")
                    # Clear cache and skip batch
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                elif "illegal memory access" in error_str or "cuda" in error_str:
                    logger.error(f"CUDA error in batch {batch_count}: {e}")
                    raise CUDATrainingError(f"CUDA error during training: {e}") from e
                else:
                    raise

        # Final sync after epoch
        if device == "cuda":
            torch.cuda.synchronize()

        return total_loss

    async def _validate_epoch(self, model, dataloader, loss_fn, device):
        """Validiert eine Epoche mit CUDA Error Handling."""
        model.eval()
        total_loss = None

        try:
            with torch.no_grad():
                for batch in dataloader:
                    features = batch['features'].to(device, non_blocking=True)
                    targets = {
                        'price': batch['price'].to(device, non_blocking=True),
                        'patterns': batch['patterns'].to(device, non_blocking=True),
                        'regime': batch['regime'].to(device, non_blocking=True)
                    }

                    predictions = model(features)
                    _, components = loss_fn(predictions, targets)

                    # Skip if NaN
                    if np.isnan(components.total):
                        continue

                    if total_loss is None:
                        total_loss = components
                    else:
                        total_loss.total = (total_loss.total + components.total) / 2
                        total_loss.price = (total_loss.price + components.price) / 2
                        total_loss.pattern = (total_loss.pattern + components.pattern) / 2
                        total_loss.regime = (total_loss.regime + components.regime) / 2

            # Sync after validation
            if device == "cuda":
                torch.cuda.synchronize()

        except RuntimeError as e:
            error_str = str(e).lower()
            if "cuda" in error_str or "illegal memory" in error_str:
                logger.error(f"CUDA error during validation: {e}")
                raise CUDATrainingError(f"CUDA error during validation: {e}") from e
            raise

        return total_loss, {}

    def _create_dataloaders(self, dataset: dict, batch_size: int, val_split: float):
        """Erstellt Train/Val DataLoaders."""
        _ensure_torch()
        from torch.utils.data import DataLoader, TensorDataset, random_split

        features = torch.FloatTensor(dataset['features'])
        price_labels = torch.FloatTensor(dataset['price_labels'])
        pattern_labels = torch.FloatTensor(dataset['pattern_labels'])
        regime_labels = torch.LongTensor(dataset['regime_labels'])

        # Custom Dataset
        class MultiTaskDataset(torch.utils.data.Dataset):
            def __init__(self, features, price, patterns, regime):
                self.features = features
                self.price = price
                self.patterns = patterns
                self.regime = regime

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return {
                    'features': self.features[idx],
                    'price': self.price[idx],
                    'patterns': self.patterns[idx],
                    'regime': self.regime[idx]
                }

        full_dataset = MultiTaskDataset(features, price_labels, pattern_labels, regime_labels)

        # Split
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    # =========================================================================
    # Status Updates
    # =========================================================================

    def _update_status(self, status: TrainingStatus):
        """Updated Job-Status."""
        if self._current_job:
            self._current_job["status"] = status.value
            if status == TrainingStatus.COMPLETED:
                self._current_job["completed_at"] = datetime.now(timezone.utc)

    def _update_progress(self, **kwargs):
        """Updated Progress."""
        if self._current_job and self._current_job.get("progress"):
            self._current_job["progress"].update(kwargs)

    def _update_metrics(self, metrics: TrainingMetrics):
        """Updated Metrics."""
        if self._current_job:
            self._current_job["metrics"] = metrics.model_dump()

    # =========================================================================
    # History Management
    # =========================================================================

    async def load_history(self):
        """Laedt Training-Historie aus Datei."""
        history_path = Path(MODEL_DIR) / HISTORY_FILE
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
                self._history = [TrainingHistoryItem(**item) for item in data]
                logger.info(f"Loaded {len(self._history)} history items")
            except Exception as e:
                logger.warning(f"Could not load history: {e}")
                self._history = []

    async def save_history(self):
        """Speichert Training-Historie."""
        history_path = Path(MODEL_DIR) / HISTORY_FILE
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump([item.model_dump() for item in self._history], f, default=str)
        except Exception as e:
            logger.error(f"Could not save history: {e}")

    async def _save_to_history(self):
        """Speichert aktuellen Job in Historie."""
        if self._current_job:
            job = self._current_job
            item = TrainingHistoryItem(
                job_id=job["job_id"],
                status=TrainingStatus(job["status"]),
                started_at=job["started_at"],
                completed_at=job.get("completed_at"),
                duration_seconds=(
                    (job.get("completed_at", datetime.now(timezone.utc)) - job["started_at"]).total_seconds()
                    if job.get("completed_at") else None
                ),
                symbols=job["request"]["symbols"],
                timeframes=job["request"]["timeframes"],
                epochs_completed=job["progress"].get("current_epoch", 0),
                final_loss=job["metrics"]["total_loss"] if job.get("metrics") else None,
                model_path=job.get("model_path"),
                error_message=job.get("error_message")
            )

            self._history.insert(0, item)
            self._history = self._history[:MAX_HISTORY_ITEMS]
            await self.save_history()

            self._current_job = None

    def get_history(self) -> list[TrainingHistoryItem]:
        """Gibt Training-Historie zurueck."""
        return self._history

    # =========================================================================
    # Incremental Training (Self-Learning)
    # =========================================================================

    async def incremental_train(
        self,
        samples: list[dict],
        config: dict,
    ) -> dict:
        """
        Perform incremental training with EWC regularization.

        Uses Elastic Weight Consolidation to prevent catastrophic forgetting
        while fine-tuning on new feedback samples.

        Args:
            samples: List of feedback samples with OHLCV context and labels
            config: Training configuration (epochs, learning_rate, ewc_lambda)

        Returns:
            Training result with metrics and validation info
        """
        _ensure_torch()

        if self.is_training():
            return {"status": "error", "message": "Training already in progress"}

        try:
            logger.info(f"Starting incremental training with {len(samples)} samples")

            # Load existing model
            from ...cnn_lstm_app.models.cnn_lstm_model import (
                CNNLSTMConfig,
                create_cnn_lstm_model,
                load_model,
                save_model
            )

            # Find latest model
            model_path = Path(MODEL_DIR) / "latest.pt"
            if not model_path.exists():
                return {"status": "error", "message": "No existing model to fine-tune"}

            # Load model
            model, metadata = load_model(str(model_path))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            # Store original parameters for EWC
            original_params = {
                name: param.clone()
                for name, param in model.named_parameters()
            }

            # Compute Fisher Information (simplified - use gradient magnitudes)
            fisher_info = {}
            model.eval()

            # Use validation data to compute Fisher
            # For simplicity, we use uniform weights here
            for name, param in model.named_parameters():
                fisher_info[name] = torch.ones_like(param)

            # Prepare training data from samples
            features_list = []
            price_labels = []
            pattern_labels = []
            regime_labels = []

            for sample in samples:
                if sample.get("ohlcv_context"):
                    # Convert OHLCV to features (simplified)
                    ohlcv = sample["ohlcv_context"]
                    if len(ohlcv) >= 50:  # Minimum sequence length
                        features = self._extract_features_from_ohlcv(ohlcv)
                        if features is not None:
                            features_list.append(features)

                            # Extract labels
                            if sample.get("price_label"):
                                price_labels.append([
                                    1.0 if sample["price_label"].get("direction") == "bullish" else 0.0,
                                    sample["price_label"].get("change_percent", 0.0)
                                ])
                            else:
                                price_labels.append([0.5, 0.0])

                            if sample.get("pattern_labels"):
                                pattern_vec = [0.0] * 16
                                for idx in sample["pattern_labels"]:
                                    if 0 <= idx < 16:
                                        pattern_vec[idx] = 1.0
                                pattern_labels.append(pattern_vec)
                            else:
                                pattern_labels.append([0.0] * 16)

                            if sample.get("regime_label") is not None:
                                regime_labels.append(sample["regime_label"])
                            else:
                                regime_labels.append(2)  # Default: sideways

            if len(features_list) < 10:
                return {"status": "skipped", "message": "Insufficient valid samples"}

            # Convert to tensors
            features_tensor = torch.FloatTensor(np.array(features_list)).to(device)
            price_tensor = torch.FloatTensor(np.array(price_labels)).to(device)
            pattern_tensor = torch.FloatTensor(np.array(pattern_labels)).to(device)
            regime_tensor = torch.LongTensor(np.array(regime_labels)).to(device)

            # Training setup
            epochs = config.get("epochs", 5)
            learning_rate = config.get("learning_rate", 1e-5)
            ewc_lambda = config.get("ewc_lambda", 1000.0)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-6
            )

            loss_fn = create_multi_task_loss(LossWeights(
                price=0.4, pattern=0.35, regime=0.25
            ))

            # Training loop
            model.train()
            best_loss = float('inf')
            metrics_history = []

            for epoch in range(epochs):
                optimizer.zero_grad()

                # Forward pass
                predictions = model(features_tensor)
                targets = {
                    'price': price_tensor,
                    'patterns': pattern_tensor,
                    'regime': regime_tensor
                }

                # Task loss
                task_loss, components = loss_fn(predictions, targets)

                # EWC loss
                ewc_loss = 0.0
                for name, param in model.named_parameters():
                    if name in fisher_info and name in original_params:
                        ewc_loss += (
                            fisher_info[name] * (param - original_params[name]) ** 2
                        ).sum()

                # Total loss
                total_loss = task_loss + (ewc_lambda / 2) * ewc_loss

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                metrics_history.append({
                    "epoch": epoch + 1,
                    "task_loss": components.total,
                    "ewc_loss": float(ewc_loss),
                    "total_loss": float(total_loss),
                })

                if components.total < best_loss:
                    best_loss = components.total

                logger.debug(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

            # Save updated model
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            new_model_path = Path(MODEL_DIR) / f"cnn_lstm_incremental_{timestamp}.pt"

            save_model(model, str(new_model_path), {
                "type": "incremental",
                "base_model": str(model_path),
                "samples_count": len(features_list),
                "epochs": epochs,
                "final_loss": float(best_loss),
                "ewc_lambda": ewc_lambda,
            })

            # Update latest symlink
            latest_path = Path(MODEL_DIR) / "latest.pt"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(new_model_path.name)

            logger.info(f"Incremental training completed: {new_model_path}")

            # Notify inference service
            await self._notify_inference_service()

            return {
                "status": "completed",
                "model_version": new_model_path.name,
                "samples_used": len(features_list),
                "epochs": epochs,
                "final_loss": float(best_loss),
                "metrics_history": metrics_history,
                "validation": {
                    "accuracy_change": 0.0,  # Would need comparison with baseline
                }
            }

        except Exception as e:
            logger.error(f"Incremental training failed: {e}")
            return {"status": "error", "message": str(e)}

    def _extract_features_from_ohlcv(self, ohlcv: list) -> np.ndarray | None:
        """Extract 25-dimensional features from OHLCV data."""
        try:
            if len(ohlcv) < 50:
                return None

            # Take last 50 candles
            data = ohlcv[-50:]

            # Extract basic OHLCV
            opens = np.array([c.get("open", c.get("o", 0)) for c in data])
            highs = np.array([c.get("high", c.get("h", 0)) for c in data])
            lows = np.array([c.get("low", c.get("l", 0)) for c in data])
            closes = np.array([c.get("close", c.get("c", 0)) for c in data])
            volumes = np.array([c.get("volume", c.get("v", 0)) for c in data])

            # Normalize
            price_mean = closes.mean()
            price_std = closes.std() + 1e-8
            volume_mean = volumes.mean() + 1e-8

            # Build feature matrix (50 x 25)
            features = np.zeros((50, 25))

            # OHLCV (normalized)
            features[:, 0] = (opens - price_mean) / price_std
            features[:, 1] = (highs - price_mean) / price_std
            features[:, 2] = (lows - price_mean) / price_std
            features[:, 3] = (closes - price_mean) / price_std
            features[:, 4] = volumes / volume_mean

            # Returns and volatility
            returns = np.diff(closes) / (closes[:-1] + 1e-8)
            features[1:, 5] = returns
            features[:, 6] = np.std(features[:, 5])  # Volatility proxy

            # Simple moving averages
            for i in range(20, 50):
                features[i, 7] = closes[i-20:i].mean()  # SMA20
                features[i, 8] = closes[i-12:i].mean()  # EMA12 proxy

            # Normalize SMAs
            features[:, 7] = (features[:, 7] - price_mean) / price_std
            features[:, 8] = (features[:, 8] - price_mean) / price_std

            # Fill remaining features with zeros (simplified)
            # In production, would compute RSI, MACD, BB, etc.

            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    # =========================================================================
    # Inference Service Notification
    # =========================================================================

    async def _notify_inference_service(self):
        """Benachrichtigt Inference Service ueber neues Modell."""
        try:
            client = await self._get_http_client()
            response = await client.post(
                f"{CNN_LSTM_SERVICE_URL}/api/v1/models/reload"
            )
            if response.status_code == 200:
                logger.info("Inference service notified of new model")
            else:
                logger.warning(f"Failed to notify inference service: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not notify inference service: {e}")

    async def close(self):
        """Schliesst Service und gibt GPU-Ressourcen frei."""
        # Clean up GPU resources
        _cleanup_gpu_memory()

        if self._http_client:
            await self._http_client.aclose()


# Singleton Instance
training_service = TrainingService()
