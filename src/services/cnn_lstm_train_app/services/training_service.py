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
        Haupttraining-Loop.

        Args:
            job_id: Job-ID
            request: Training-Request
        """
        _ensure_torch()

        try:
            self._update_status(TrainingStatus.PREPARING)
            logger.info(f"Preparing training for {len(request.symbols)} symbols")

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

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info(f"Model created on device: {device}")

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

        except Exception as e:
            logger.error(f"Training error: {e}")
            self._update_status(TrainingStatus.FAILED)
            self._current_job["error_message"] = str(e)

        finally:
            # Speichere in Historie
            await self._save_to_history()

            # Reset Status
            try:
                from ..main import set_training_in_progress
                set_training_in_progress(False)
            except ImportError:
                pass

    async def _train_epoch(self, model, dataloader, loss_fn, optimizer, device):
        """Trainiert eine Epoche."""
        model.train()
        total_loss = None

        for batch in dataloader:
            features = batch['features'].to(device)
            targets = {
                'price': batch['price'].to(device),
                'patterns': batch['patterns'].to(device),
                'regime': batch['regime'].to(device)
            }

            optimizer.zero_grad()
            predictions = model(features)
            loss, components = loss_fn(predictions, targets)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if total_loss is None:
                total_loss = components
            else:
                # Averaging
                total_loss.total = (total_loss.total + components.total) / 2
                total_loss.price = (total_loss.price + components.price) / 2
                total_loss.pattern = (total_loss.pattern + components.pattern) / 2
                total_loss.regime = (total_loss.regime + components.regime) / 2

        return total_loss

    async def _validate_epoch(self, model, dataloader, loss_fn, device):
        """Validiert eine Epoche."""
        model.eval()
        total_loss = None

        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(device)
                targets = {
                    'price': batch['price'].to(device),
                    'patterns': batch['patterns'].to(device),
                    'regime': batch['regime'].to(device)
                }

                predictions = model(features)
                _, components = loss_fn(predictions, targets)

                if total_loss is None:
                    total_loss = components
                else:
                    total_loss.total = (total_loss.total + components.total) / 2
                    total_loss.price = (total_loss.price + components.price) / 2
                    total_loss.pattern = (total_loss.pattern + components.pattern) / 2
                    total_loss.regime = (total_loss.regime + components.regime) / 2

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
        """Schliesst Service."""
        if self._http_client:
            await self._http_client.aclose()


# Singleton Instance
training_service = TrainingService()
