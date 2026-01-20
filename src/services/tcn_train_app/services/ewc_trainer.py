"""
Elastic Weight Consolidation (EWC) Trainer for Incremental Learning.

Implements EWC to prevent catastrophic forgetting when fine-tuning
the TCN model on new feedback samples.

EWC Loss = Task Loss + (λ/2) × Σ Fisher_i × (θ_i - θ*_i)²

Where:
- Fisher_i: Diagonal of Fisher Information Matrix (parameter importance)
- θ*_i: Optimal parameters from previous task
- λ: Regularization strength (default 1000)
"""

import os
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from loguru import logger

# Lazy import for PyTorch
torch = None
nn = None


def _load_torch():
    global torch, nn
    if torch is None:
        try:
            import torch as t
            import torch.nn as tnn
            torch = t
            nn = tnn
            return True
        except ImportError:
            return False
    return True


@dataclass
class EWCConfig:
    """EWC Training configuration."""
    # EWC specific
    ewc_lambda: float = 1000.0  # Regularization strength
    fisher_sample_size: int = 200  # Samples for Fisher computation

    # Training parameters (conservative for fine-tuning)
    learning_rate: float = 1e-5  # Very small LR for fine-tuning
    batch_size: int = 16
    epochs: int = 10  # Few epochs for incremental update
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    weight_decay: float = 1e-6

    # Min requirements
    min_samples: int = 50  # Minimum samples to start training


class EWCTrainer:
    """
    Elastic Weight Consolidation Trainer for incremental learning.

    Prevents catastrophic forgetting by adding a regularization term
    that penalizes changes to important parameters.
    """

    FISHER_FILE = "data/models/tcn/fisher_matrix.pt"
    OPTIMAL_PARAMS_FILE = "data/models/tcn/optimal_params.pt"

    def __init__(self, device: str = "cuda"):
        """Initialize EWC Trainer."""
        self.device = device
        self._fisher_matrix: Optional[Dict[str, "torch.Tensor"]] = None
        self._optimal_params: Optional[Dict[str, "torch.Tensor"]] = None
        self._is_initialized = False

    def _ensure_torch(self):
        """Ensure PyTorch is loaded."""
        if not _load_torch():
            raise RuntimeError("PyTorch not available")

    def _get_device(self) -> "torch.device":
        """Get the compute device."""
        self._ensure_torch()
        return torch.device(self.device if torch.cuda.is_available() else "cpu")

    def initialize_from_model(self, model: "nn.Module", dataloader=None) -> None:
        """
        Initialize EWC from a trained model.

        If dataloader is provided, computes Fisher Information Matrix.
        Otherwise, tries to load from saved files.

        Args:
            model: The trained model
            dataloader: Optional dataloader for Fisher computation
        """
        self._ensure_torch()
        device = self._get_device()

        # Store optimal parameters (deep copy)
        self._optimal_params = {
            name: param.clone().detach().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        if dataloader is not None:
            # Compute Fisher Information Matrix
            logger.info("Computing Fisher Information Matrix...")
            self._fisher_matrix = self._compute_fisher(model, dataloader, device)

            # Save for future use
            self._save_ewc_state()
        else:
            # Try to load existing Fisher matrix
            if os.path.exists(self.FISHER_FILE):
                self._load_ewc_state(device)
            else:
                # Initialize with uniform importance
                logger.warning("No Fisher matrix available, using uniform importance")
                self._fisher_matrix = {
                    name: torch.ones_like(param)
                    for name, param in self._optimal_params.items()
                }

        self._is_initialized = True
        logger.info("EWC Trainer initialized")

    def _compute_fisher(
        self,
        model: "nn.Module",
        dataloader,
        device: "torch.device",
        sample_size: int = 200
    ) -> Dict[str, "torch.Tensor"]:
        """
        Compute diagonal of Fisher Information Matrix.

        The Fisher matrix measures the importance of each parameter
        for the current task.

        F_i = E[(∂log p(y|x,θ) / ∂θ_i)²]

        For neural networks, this is approximated by the squared gradients.
        """
        self._ensure_torch()

        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        model.eval()
        criterion = nn.BCELoss()

        samples_processed = 0

        for batch_x, batch_y in dataloader:
            if samples_processed >= sample_size:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            model.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.clone().pow(2)

            samples_processed += len(batch_x)

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= max(samples_processed, 1)

        logger.info(f"Fisher matrix computed from {samples_processed} samples")
        return fisher

    def _save_ewc_state(self) -> None:
        """Save Fisher matrix and optimal parameters to disk."""
        self._ensure_torch()

        try:
            os.makedirs(os.path.dirname(self.FISHER_FILE), exist_ok=True)

            # Save Fisher matrix
            torch.save(self._fisher_matrix, self.FISHER_FILE)

            # Save optimal parameters
            torch.save(self._optimal_params, self.OPTIMAL_PARAMS_FILE)

            logger.info("EWC state saved to disk")
        except Exception as e:
            logger.error(f"Failed to save EWC state: {e}")

    def _load_ewc_state(self, device: "torch.device") -> None:
        """Load Fisher matrix and optimal parameters from disk."""
        self._ensure_torch()

        try:
            if os.path.exists(self.FISHER_FILE):
                self._fisher_matrix = torch.load(self.FISHER_FILE, map_location=device)
                logger.info("Loaded Fisher matrix from disk")

            if os.path.exists(self.OPTIMAL_PARAMS_FILE):
                self._optimal_params = torch.load(self.OPTIMAL_PARAMS_FILE, map_location=device)
                logger.info("Loaded optimal parameters from disk")
        except Exception as e:
            logger.error(f"Failed to load EWC state: {e}")

    def compute_ewc_loss(
        self,
        model: "nn.Module",
        ewc_lambda: float = 1000.0
    ) -> "torch.Tensor":
        """
        Compute EWC regularization loss.

        EWC Loss = (λ/2) × Σ Fisher_i × (θ_i - θ*_i)²

        Args:
            model: Current model
            ewc_lambda: Regularization strength

        Returns:
            EWC loss tensor
        """
        self._ensure_torch()

        if not self._is_initialized:
            return torch.tensor(0.0, requires_grad=True)

        ewc_loss = torch.tensor(0.0, device=self._get_device())

        for name, param in model.named_parameters():
            if param.requires_grad and name in self._fisher_matrix:
                # (θ - θ*)²
                diff = param - self._optimal_params[name]
                # Fisher_i × (θ - θ*)²
                ewc_loss += (self._fisher_matrix[name] * diff.pow(2)).sum()

        return (ewc_lambda / 2) * ewc_loss

    def update_ewc_state(
        self,
        model: "nn.Module",
        dataloader,
        blend_ratio: float = 0.5
    ) -> None:
        """
        Update EWC state after incremental training.

        Blends old Fisher matrix with new one to maintain knowledge
        of both old and new tasks.

        Args:
            model: Updated model
            dataloader: Data used for training
            blend_ratio: Weight of new Fisher (0.5 = equal blend)
        """
        self._ensure_torch()
        device = self._get_device()

        # Compute new Fisher matrix
        new_fisher = self._compute_fisher(model, dataloader, device)

        # Blend with old Fisher matrix
        if self._fisher_matrix is not None:
            for name in self._fisher_matrix:
                if name in new_fisher:
                    self._fisher_matrix[name] = (
                        (1 - blend_ratio) * self._fisher_matrix[name] +
                        blend_ratio * new_fisher[name]
                    )
        else:
            self._fisher_matrix = new_fisher

        # Update optimal parameters
        self._optimal_params = {
            name: param.clone().detach().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Save updated state
        self._save_ewc_state()
        logger.info("EWC state updated after incremental training")

    def get_parameter_importance(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most important parameters according to Fisher matrix.

        Args:
            top_k: Number of top parameters to return

        Returns:
            List of (parameter_name, importance_score) tuples
        """
        if self._fisher_matrix is None:
            return []

        importance = [
            (name, fisher.sum().item())
            for name, fisher in self._fisher_matrix.items()
        ]

        return sorted(importance, key=lambda x: x[1], reverse=True)[:top_k]

    def get_statistics(self) -> Dict:
        """Get EWC trainer statistics."""
        stats = {
            "is_initialized": self._is_initialized,
            "fisher_file_exists": os.path.exists(self.FISHER_FILE),
            "optimal_params_file_exists": os.path.exists(self.OPTIMAL_PARAMS_FILE)
        }

        if self._fisher_matrix is not None:
            stats["num_parameters"] = len(self._fisher_matrix)
            stats["top_important_params"] = [
                {"name": name, "importance": score}
                for name, score in self.get_parameter_importance(5)
            ]

        return stats


class IncrementalTrainer:
    """
    Wrapper class that combines EWC with training logic.

    Provides a simple interface for incremental training with
    automatic catastrophic forgetting prevention.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize incremental trainer."""
        self.device = device
        self.ewc = EWCTrainer(device)
        self._training_in_progress = False

    def _ensure_torch(self):
        """Ensure PyTorch is loaded."""
        if not _load_torch():
            raise RuntimeError("PyTorch not available")

    async def incremental_train(
        self,
        model: "nn.Module",
        sequences: np.ndarray,
        labels: np.ndarray,
        config: Optional[EWCConfig] = None,
        progress_callback=None
    ) -> Dict:
        """
        Perform incremental training with EWC regularization.

        Args:
            model: Model to fine-tune
            sequences: Training sequences (N, seq_len, features)
            labels: Training labels (N, num_classes)
            config: EWC configuration
            progress_callback: Optional callback(epoch, total, loss)

        Returns:
            Training results dictionary
        """
        import asyncio

        self._ensure_torch()
        config = config or EWCConfig()
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        if len(sequences) < config.min_samples:
            return {
                "status": "skipped",
                "reason": f"Insufficient samples ({len(sequences)} < {config.min_samples})"
            }

        self._training_in_progress = True
        start_time = datetime.now()

        try:
            # Prepare data
            n_samples = len(sequences)
            n_val = int(n_samples * config.validation_split)
            indices = np.random.permutation(n_samples)

            train_indices = indices[n_val:]
            val_indices = indices[:n_val]

            train_sequences = sequences[train_indices]
            train_labels = labels[train_indices]
            val_sequences = sequences[val_indices]
            val_labels = labels[val_indices]

            # Create dataloader for Fisher computation
            train_tensor_x = torch.tensor(train_sequences, dtype=torch.float32)
            train_tensor_y = torch.tensor(train_labels, dtype=torch.float32)

            # Transpose for TCN: (batch, seq, features) -> (batch, features, seq)
            train_tensor_x = train_tensor_x.transpose(1, 2)

            train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True
            )

            # Initialize EWC from current model if not already
            if not self.ewc._is_initialized:
                self.ewc.initialize_from_model(model, train_loader)

            # Move model to device
            model = model.to(device)
            model.train()

            # Setup optimizer with very small learning rate
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            criterion = nn.BCELoss()

            best_val_loss = float('inf')
            patience_counter = 0
            history = {"train_loss": [], "val_loss": [], "ewc_loss": []}

            # Validation data
            val_tensor_x = torch.tensor(val_sequences, dtype=torch.float32).transpose(1, 2).to(device)
            val_tensor_y = torch.tensor(val_labels, dtype=torch.float32).to(device)

            logger.info(f"Starting incremental training with {len(train_sequences)} samples")

            for epoch in range(config.epochs):
                model.train()
                epoch_losses = []
                epoch_ewc_losses = []

                for batch_x, batch_y in train_loader:
                    # Yield to event loop for responsiveness
                    await asyncio.sleep(0)

                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(batch_x)
                    task_loss = criterion(outputs, batch_y)

                    # EWC regularization
                    ewc_loss = self.ewc.compute_ewc_loss(model, config.ewc_lambda)

                    # Total loss
                    total_loss = task_loss + ewc_loss

                    # Backward pass
                    total_loss.backward()
                    optimizer.step()

                    epoch_losses.append(task_loss.item())
                    epoch_ewc_losses.append(ewc_loss.item())

                avg_train_loss = np.mean(epoch_losses)
                avg_ewc_loss = np.mean(epoch_ewc_losses)
                history["train_loss"].append(avg_train_loss)
                history["ewc_loss"].append(avg_ewc_loss)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_tensor_x)
                    val_loss = criterion(val_outputs, val_tensor_y).item()

                history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} - "
                    f"Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, EWC: {avg_ewc_loss:.4f}"
                )

                if progress_callback:
                    progress_callback(epoch + 1, config.epochs, val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            # Update EWC state with new knowledge
            self.ewc.update_ewc_state(model, train_loader)

            duration = (datetime.now() - start_time).total_seconds()

            return {
                "status": "completed",
                "epochs_trained": len(history["train_loss"]),
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "best_val_loss": best_val_loss,
                "avg_ewc_loss": np.mean(history["ewc_loss"]),
                "samples_used": len(train_sequences),
                "duration_seconds": duration,
                "history": history
            }

        except Exception as e:
            logger.error(f"Incremental training failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

        finally:
            self._training_in_progress = False

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._training_in_progress

    def get_statistics(self) -> Dict:
        """Get trainer statistics."""
        return {
            "is_training": self._training_in_progress,
            "ewc": self.ewc.get_statistics()
        }


# Singleton instance
ewc_trainer = EWCTrainer()
incremental_trainer = IncrementalTrainer()
