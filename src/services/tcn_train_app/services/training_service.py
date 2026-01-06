"""TCN model training service for dedicated training container."""

import os
import json
import asyncio
import httpx
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from loguru import logger

# Lazy imports for PyTorch
torch = None


def _load_torch():
    global torch
    if torch is None:
        try:
            import torch as t
            torch = t
            return True
        except ImportError:
            return False
    return True


class TrainingStatus(str, Enum):
    """Training status enum."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Training configuration."""
    sequence_length: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    weight_decay: float = 1e-5


class TCNTrainingService:
    """
    Dedicated training service for TCN pattern recognition.

    Runs in a separate container from the inference service.
    Writes trained models to shared volume and notifies inference service.
    """

    MODEL_DIR = os.getenv("TCN_MODEL_DIR", "data/models/tcn")
    HISTORY_FILE = "data/models/tcn/training_history.json"
    MAX_MODELS_TO_KEEP = 3
    LATEST_MODEL_LINK = "data/models/tcn/latest.pt"

    # TCN Pattern Classes
    PATTERN_CLASSES = [
        "head_and_shoulders", "inverse_head_and_shoulders",
        "double_top", "double_bottom",
        "triple_top", "triple_bottom",
        "ascending_triangle", "descending_triangle", "symmetrical_triangle",
        "bull_flag", "bear_flag",
        "cup_and_handle",
        "rising_wedge", "falling_wedge",
        "channel_up", "channel_down"
    ]

    def __init__(self, device: str = "cuda"):
        """Initialize training service."""
        self.device = device
        self._training_in_progress = False
        self._stop_requested = False
        self._current_job: Optional[Dict] = None
        self._training_history: List[Dict] = []
        self._model = None

        # TCN inference service URL for notification
        self._tcn_service_url = os.getenv(
            "TCN_SERVICE_URL",
            "http://trading-tcn:3003"
        )

        # Ensure model directory exists
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # Load existing training history
        self._load_history()

    def _load_history(self) -> None:
        """Load training history from JSON file."""
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r') as f:
                    self._training_history = json.load(f)
                logger.info(f"Loaded {len(self._training_history)} training history entries")
        except Exception as e:
            logger.warning(f"Could not load training history: {e}")
            self._training_history = []

    def _save_history(self) -> None:
        """Save training history to JSON file."""
        try:
            serializable_history = []
            for entry in self._training_history:
                clean_entry = {}
                for key, value in entry.items():
                    if isinstance(value, datetime):
                        clean_entry[key] = value.isoformat()
                    elif isinstance(value, TrainingStatus):
                        clean_entry[key] = value.value
                    elif isinstance(value, (np.floating, np.integer)):
                        clean_entry[key] = float(value) if isinstance(value, np.floating) else int(value)
                    else:
                        clean_entry[key] = value
                serializable_history.append(clean_entry)

            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(serializable_history, f, indent=2, default=str)
            logger.info(f"Saved {len(serializable_history)} training history entries")
        except Exception as e:
            logger.error(f"Could not save training history: {e}")

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._training_in_progress

    def request_stop(self) -> None:
        """Request training to stop."""
        if self._training_in_progress:
            self._stop_requested = True
            logger.info("Training stop requested")

    def is_stop_requested(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested

    def get_training_status(self) -> Dict:
        """Get current training status."""
        if not self._training_in_progress:
            return {
                "status": TrainingStatus.IDLE.value,
                "message": "No training in progress"
            }

        started_at = self._current_job.get("started_at") if self._current_job else None

        return {
            "status": TrainingStatus.TRAINING.value,
            "job_id": self._current_job.get("job_id") if self._current_job else None,
            "progress": self._current_job.get("progress", 0) if self._current_job else 0,
            "current_epoch": self._current_job.get("current_epoch", 0) if self._current_job else 0,
            "total_epochs": self._current_job.get("total_epochs", 0) if self._current_job else 0,
            "best_loss": self._current_job.get("best_loss") if self._current_job else None,
            "started_at": started_at.isoformat() if started_at else None,
            "samples_count": self._current_job.get("samples_count") if self._current_job else None
        }

    def _init_model(self):
        """Initialize TCN model for training."""
        if not _load_torch():
            raise RuntimeError("PyTorch not available")

        # Import model architecture
        from .tcn_model import TCNPatternModel

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        logger.info(f"TCN Training Service initialized on {device}")

        self._model = TCNPatternModel(
            num_inputs=5,  # OHLCV
            num_channels=[32, 64, 128],
            num_classes=len(self.PATTERN_CLASSES),
            kernel_size=3,
            dropout=0.2
        ).to(device)

        return device

    async def _notify_tcn_service(self, model_path: str) -> bool:
        """Notify TCN inference service that a new model is available."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._tcn_service_url}/api/v1/model/reload",
                    json={"model_path": model_path}
                )
                if response.status_code == 200:
                    logger.info(f"TCN service notified of new model: {model_path}")
                    return True
                else:
                    logger.warning(f"TCN service notification failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.warning(f"Could not notify TCN service: {e}")
            return False

    def _update_latest_link(self, model_path: str) -> None:
        """Update the 'latest.pt' symlink to point to new model."""
        try:
            # Remove existing link if present
            if os.path.exists(self.LATEST_MODEL_LINK):
                os.remove(self.LATEST_MODEL_LINK)

            # Create new symlink
            os.symlink(os.path.basename(model_path), self.LATEST_MODEL_LINK)
            logger.info(f"Updated latest model link to {model_path}")
        except Exception as e:
            logger.warning(f"Could not update latest model link: {e}")

    async def prepare_training_data(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        lookback_days: int = 365
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical OHLCV via Data Service."""
        from src.services.data_gateway_service import data_gateway
        from .pattern_classifier import PatternClassifier

        label_generator = PatternClassifier()
        all_sequences = []
        all_labels = []

        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")

                # Approximate candle count
                candles_per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}
                limit = lookback_days * candles_per_day.get(timeframe, 24)

                data = await data_gateway.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )

                if not data or len(data) < 200:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue

                # Map timeframe to field prefix
                tf_map = {"1h": "h1", "4h": "h1", "1d": "d1", "15m": "m15", "m15": "m15", "h1": "h1", "d1": "d1"}
                prefix = tf_map.get(timeframe.lower(), "h1")

                # Convert to numpy
                ohlcv = np.array([
                    [
                        d.get('open') or d.get(f'{prefix}_open', 0),
                        d.get('high') or d.get(f'{prefix}_high', 0),
                        d.get('low') or d.get(f'{prefix}_low', 0),
                        d.get('close') or d.get(f'{prefix}_close', 0),
                        d.get('volume', 0)
                    ]
                    for d in data
                ], dtype=np.float32)

                # Generate sequences and labels
                sequences, labels = self._generate_labeled_sequences(ohlcv, label_generator)
                all_sequences.extend(sequences)
                all_labels.extend(labels)

                logger.info(f"Generated {len(sequences)} samples from {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        if not all_sequences:
            raise ValueError("No training data generated")

        return np.array(all_sequences), np.array(all_labels)

    def _generate_labeled_sequences(
        self,
        ohlcv: np.ndarray,
        label_generator,
        sequence_length: int = 200,
        step: int = 20
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate labeled sequences using sliding window."""
        sequences = []
        labels = []
        num_classes = len(self.PATTERN_CLASSES)

        for i in range(0, len(ohlcv) - sequence_length, step):
            sequence = ohlcv[i:i + sequence_length]

            # Detect patterns using rule-based classifier
            patterns = label_generator.detect_all_patterns(sequence)

            # Create multi-label vector
            label = np.zeros(num_classes, dtype=np.float32)

            for pattern in patterns:
                try:
                    pattern_idx = self.PATTERN_CLASSES.index(pattern.pattern_type.value)
                    label[pattern_idx] = pattern.confidence
                except ValueError:
                    pass

            sequences.append(sequence)
            labels.append(label)

        return sequences, labels

    async def train(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        lookback_days: int = 365,
        config: Optional[TrainingConfig] = None
    ) -> Dict:
        """
        Train the TCN model.

        Args:
            symbols: Symbols for training
            timeframe: Timeframe
            lookback_days: Days of history
            config: Training configuration

        Returns:
            Training results
        """
        if self._training_in_progress:
            return {
                "status": TrainingStatus.FAILED.value,
                "message": "Training already in progress"
            }

        config = config or TrainingConfig()
        job_id = f"tcn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._training_in_progress = True
        self._stop_requested = False  # Reset stop flag
        self._current_job = {
            "job_id": job_id,
            "status": TrainingStatus.PREPARING,
            "started_at": datetime.now(),
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.epochs
        }

        try:
            logger.info(f"Starting training job {job_id}")

            # Initialize model
            device = self._init_model()

            # Prepare data
            sequences, labels = await self.prepare_training_data(
                symbols, timeframe, lookback_days
            )

            logger.info(f"Training with {len(sequences)} samples")
            self._current_job["samples_count"] = len(sequences)

            # Split data
            n_samples = len(sequences)
            n_val = int(n_samples * config.validation_split)
            indices = np.random.permutation(n_samples)

            train_indices = indices[n_val:]
            val_indices = indices[:n_val]

            train_sequences = sequences[train_indices]
            train_labels = labels[train_indices]
            val_sequences = sequences[val_indices]
            val_labels = labels[val_indices]

            # Training
            self._current_job["status"] = TrainingStatus.TRAINING

            result = await self._train_loop(
                train_sequences, train_labels,
                val_sequences, val_labels,
                config, device
            )

            # Save model
            model_path = os.path.join(self.MODEL_DIR, f"{job_id}.pt")
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'pattern_classes': self.PATTERN_CLASSES,
                'training_config': {
                    'epochs': config.epochs,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate
                },
                'metrics': result
            }, model_path)

            # Update latest link
            self._update_latest_link(model_path)

            # Update history
            self._current_job["status"] = TrainingStatus.COMPLETED
            self._current_job["best_loss"] = result["best_loss"]
            self._current_job["completed_at"] = datetime.now()
            self._current_job["model_path"] = model_path
            self._training_history.append(self._current_job.copy())
            self._save_history()

            # Cleanup old models
            cleanup_result = self.cleanup_old_models()
            if cleanup_result["deleted"] > 0:
                logger.info(
                    f"Cleaned up {cleanup_result['deleted']} old models, "
                    f"freed {cleanup_result['freed_mb']} MB"
                )

            # Notify TCN inference service
            await self._notify_tcn_service(model_path)

            logger.info(f"Training completed: {result}")

            return {
                "status": TrainingStatus.COMPLETED.value,
                "job_id": job_id,
                "message": "Training completed successfully",
                "metrics": result,
                "model_path": model_path
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._current_job["status"] = TrainingStatus.FAILED
            return {
                "status": TrainingStatus.FAILED.value,
                "job_id": job_id,
                "message": str(e)
            }

        finally:
            self._training_in_progress = False

    async def _train_loop(
        self,
        train_sequences: np.ndarray,
        train_labels: np.ndarray,
        val_sequences: np.ndarray,
        val_labels: np.ndarray,
        config: TrainingConfig,
        device
    ) -> Dict:
        """Main training loop with async yield for health checks."""
        model = self._model
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(config.epochs):
            # Check for stop request at epoch start
            if self._stop_requested:
                logger.info(f"Training stopped by user at epoch {epoch}")
                break

            # Training
            model.train()
            train_losses = []

            # Shuffle training data
            indices = np.random.permutation(len(train_sequences))

            for batch_num, i in enumerate(range(0, len(indices), config.batch_size)):
                # Yield to event loop every 10 batches for API responsiveness
                if batch_num % 10 == 0:
                    await asyncio.sleep(0)

                batch_indices = indices[i:i + config.batch_size]

                # Shape: (batch, sequence_length, channels)
                batch_x = torch.tensor(
                    train_sequences[batch_indices],
                    dtype=torch.float32
                ).to(device)

                # Transpose to (batch, channels, sequence_length) for TCN
                batch_x = batch_x.transpose(1, 2)

                batch_y = torch.tensor(
                    train_labels[batch_indices],
                    dtype=torch.float32
                ).to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_x = torch.tensor(val_sequences, dtype=torch.float32).to(device)
                val_x = val_x.transpose(1, 2)
                val_y = torch.tensor(val_labels, dtype=torch.float32).to(device)

                val_outputs = model(val_x)
                val_loss = criterion(val_outputs, val_y).item()

            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            # Update progress
            self._current_job["current_epoch"] = epoch + 1
            self._current_job["progress"] = (epoch + 1) / config.epochs
            self._current_job["best_loss"] = min(best_val_loss, val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Check for stop request
            if self._stop_requested:
                logger.info(f"Training stopped by user at epoch {epoch + 1}")
                break

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_loss": best_val_loss,
            "epochs_trained": len(history["train_loss"]),
            "history": history
        }

    def get_training_history(self) -> List[Dict]:
        """Get training history with sanitized values."""
        import math

        def sanitize_value(v):
            """Sanitize value for JSON serialization."""
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
            return v

        def sanitize_entry(entry: Dict) -> Dict:
            """Sanitize an entry for JSON serialization."""
            return {k: sanitize_value(v) for k, v in entry.items()}

        return [sanitize_entry(e) for e in self._training_history]

    def list_models(self) -> List[Dict]:
        """List available trained models."""
        models = []

        if os.path.exists(self.MODEL_DIR):
            for f in os.listdir(self.MODEL_DIR):
                if f.endswith('.pt') and f != 'latest.pt':
                    path = os.path.join(self.MODEL_DIR, f)
                    if os.path.isfile(path):
                        models.append({
                            "name": f,
                            "path": path,
                            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
                            "created": datetime.fromtimestamp(os.path.getctime(path)).isoformat()
                        })

        return sorted(models, key=lambda x: x["created"], reverse=True)

    def cleanup_old_models(self, keep_count: Optional[int] = None) -> Dict:
        """Remove old model files, keeping only the most recent ones."""
        keep_count = keep_count or self.MAX_MODELS_TO_KEEP
        models = self.list_models()

        if len(models) <= keep_count:
            return {
                "status": "no_cleanup_needed",
                "models_count": len(models),
                "kept": len(models),
                "deleted": 0,
                "deleted_models": []
            }

        # Sort by creation date (newest first)
        models_sorted = sorted(models, key=lambda m: m["created"], reverse=True)

        # Keep the newest models
        models_to_delete = models_sorted[keep_count:]

        deleted_models = []
        deleted_size_mb = 0

        for model in models_to_delete:
            try:
                os.remove(model["path"])
                deleted_models.append(model["name"])
                deleted_size_mb += model["size_mb"]
                logger.info(f"Deleted old model: {model['name']}")
            except Exception as e:
                logger.error(f"Failed to delete model {model['name']}: {e}")

        return {
            "status": "cleanup_completed",
            "models_count": len(models),
            "kept": keep_count,
            "deleted": len(deleted_models),
            "deleted_models": deleted_models,
            "freed_mb": round(deleted_size_mb, 2)
        }


# Singleton instance
training_service = TCNTrainingService()
