"""TCN model training service."""

from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import os
import numpy as np
from loguru import logger

from ..models.tcn_model import TCNPatternClassifier
from ..models.pattern_classifier import PatternClassifier
from ..models.schemas import TrainingStatus

# Lazy imports
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
    Service for training TCN pattern recognition model.

    Features:
    - Automatic label generation from rule-based classifier
    - GPU-accelerated training
    - Early stopping
    - Model checkpointing
    """

    MODEL_DIR = "data/models/tcn"

    def __init__(self, device: str = "cuda"):
        """
        Initialize the training service.

        Args:
            device: Device for training ('cuda' or 'cpu')
        """
        self.device = device
        self.model = TCNPatternClassifier(device=device)
        self.label_generator = PatternClassifier()
        self._training_in_progress = False
        self._current_job: Optional[Dict] = None
        self._training_history: List[Dict] = []

        # Ensure model directory exists
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._training_in_progress

    def get_training_status(self) -> Dict:
        """Get current training status."""
        if not self._training_in_progress:
            return {
                "status": TrainingStatus.IDLE,
                "message": "No training in progress"
            }

        return {
            "status": TrainingStatus.TRAINING,
            "job_id": self._current_job.get("job_id") if self._current_job else None,
            "progress": self._current_job.get("progress", 0) if self._current_job else 0,
            "current_epoch": self._current_job.get("current_epoch", 0) if self._current_job else 0,
            "total_epochs": self._current_job.get("total_epochs", 0) if self._current_job else 0,
            "best_loss": self._current_job.get("best_loss") if self._current_job else None
        }

    async def prepare_training_data(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        lookback_days: int = 365
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical OHLCV.

        Uses rule-based classifier to generate labels.

        Args:
            symbols: List of symbols
            timeframe: Timeframe
            lookback_days: Days of history

        Returns:
            Tuple of (sequences, labels)
        """
        from src.services.data_gateway_service import data_gateway

        all_sequences = []
        all_labels = []

        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")

                # Approximate candle count
                candles_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
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

                # Convert to numpy - handle both direct OHLC and prefixed fields
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
                sequences, labels = self._generate_labeled_sequences(ohlcv)
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
        sequence_length: int = 200,
        step: int = 20
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate labeled sequences using sliding window.

        Uses rule-based classifier to create labels.
        """
        sequences = []
        labels = []
        num_classes = len(TCNPatternClassifier.PATTERN_CLASSES)

        for i in range(0, len(ohlcv) - sequence_length, step):
            sequence = ohlcv[i:i + sequence_length]

            # Detect patterns using rule-based classifier
            patterns = self.label_generator.detect_all_patterns(sequence)

            # Create multi-label vector
            label = np.zeros(num_classes, dtype=np.float32)

            for pattern in patterns:
                try:
                    pattern_idx = TCNPatternClassifier.PATTERN_CLASSES.index(
                        pattern.pattern_type.value
                    )
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
                "status": TrainingStatus.FAILED,
                "message": "Training already in progress"
            }

        if not _load_torch():
            return {
                "status": TrainingStatus.FAILED,
                "message": "PyTorch not available"
            }

        config = config or TrainingConfig()
        job_id = f"tcn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._training_in_progress = True
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

            # Prepare data
            sequences, labels = await self.prepare_training_data(
                symbols, timeframe, lookback_days
            )

            logger.info(f"Training with {len(sequences)} samples")

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

            # Create model
            self.model = TCNPatternClassifier(device=self.device)
            self.model.load()

            if self.model._model is None:
                raise ValueError("Failed to initialize model")

            # Training
            self._current_job["status"] = TrainingStatus.TRAINING

            result = self._train_loop(
                train_sequences, train_labels,
                val_sequences, val_labels,
                config
            )

            # Save model
            model_path = os.path.join(self.MODEL_DIR, f"{job_id}.pt")
            self.model.save(model_path)

            self._current_job["status"] = TrainingStatus.COMPLETED
            self._current_job["best_loss"] = result["best_loss"]
            self._training_history.append(self._current_job.copy())

            logger.info(f"Training completed: {result}")

            return {
                "status": TrainingStatus.COMPLETED,
                "job_id": job_id,
                "message": "Training completed successfully",
                "metrics": result,
                "model_path": model_path
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._current_job["status"] = TrainingStatus.FAILED
            return {
                "status": TrainingStatus.FAILED,
                "job_id": job_id,
                "message": str(e)
            }

        finally:
            self._training_in_progress = False

    def _train_loop(
        self,
        train_sequences: np.ndarray,
        train_labels: np.ndarray,
        val_sequences: np.ndarray,
        val_labels: np.ndarray,
        config: TrainingConfig
    ) -> Dict:
        """
        Main training loop.
        """
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = self.model._model
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
            # Training
            model.train()
            train_losses = []

            # Shuffle training data
            indices = np.random.permutation(len(train_sequences))

            for i in range(0, len(indices), config.batch_size):
                batch_indices = indices[i:i + config.batch_size]

                # Shape: (batch, sequence_length, channels) - model handles transpose
                batch_x = torch.tensor(
                    train_sequences[batch_indices],
                    dtype=torch.float32
                ).to(device)

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
                # Shape: (batch, sequence_length, channels) - model handles transpose
                val_x = torch.tensor(val_sequences, dtype=torch.float32).to(device)
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
        """Get training history."""
        return self._training_history

    def list_models(self) -> List[Dict]:
        """List available trained models."""
        models = []

        if os.path.exists(self.MODEL_DIR):
            for f in os.listdir(self.MODEL_DIR):
                if f.endswith('.pt'):
                    path = os.path.join(self.MODEL_DIR, f)
                    models.append({
                        "name": f,
                        "path": path,
                        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(os.path.getctime(path)).isoformat()
                    })

        return models


# Singleton instance
tcn_training_service = TCNTrainingService()
