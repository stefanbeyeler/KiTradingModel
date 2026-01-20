"""
TCN Model Validation Service.

Validates model performance using held-out test data and
real-world outcome metrics.
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from loguru import logger

# Lazy import for PyTorch
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


class ValidationMetricType(str, Enum):
    """Types of validation metrics."""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    OUTCOME_ACCURACY = "outcome_accuracy"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class PatternMetrics:
    """Metrics for a single pattern type."""
    pattern_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0


@dataclass
class ValidationResult:
    """Complete validation result for a model."""
    model_path: str
    validation_time: datetime = field(default_factory=datetime.utcnow)

    # Overall metrics
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1_score: float = 0.0
    overall_accuracy: float = 0.0

    # Outcome-based metrics
    outcome_accuracy: float = 0.0  # % of patterns with favorable outcome
    profit_factor: float = 0.0  # Sum favorable / Sum adverse moves
    win_rate: float = 0.0  # % winning patterns
    avg_favorable_move: float = 0.0
    avg_adverse_move: float = 0.0

    # Confidence calibration
    confidence_calibration: float = 0.0  # How well confidence predicts outcome

    # Per-pattern metrics
    pattern_metrics: Dict[str, PatternMetrics] = field(default_factory=dict)

    # Sample counts
    total_samples: int = 0
    samples_with_outcomes: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "validation_time": self.validation_time.isoformat(),
            "overall": {
                "precision": round(self.overall_precision, 4),
                "recall": round(self.overall_recall, 4),
                "f1_score": round(self.overall_f1_score, 4),
                "accuracy": round(self.overall_accuracy, 4)
            },
            "outcome_based": {
                "outcome_accuracy": round(self.outcome_accuracy, 4),
                "profit_factor": round(self.profit_factor, 4),
                "win_rate": round(self.win_rate, 4),
                "avg_favorable_move": round(self.avg_favorable_move, 4),
                "avg_adverse_move": round(self.avg_adverse_move, 4)
            },
            "confidence_calibration": round(self.confidence_calibration, 4),
            "samples": {
                "total": self.total_samples,
                "with_outcomes": self.samples_with_outcomes
            },
            "pattern_metrics": {
                name: {
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1_score": round(m.f1_score, 4),
                    "tp": m.true_positives,
                    "fp": m.false_positives,
                    "fn": m.false_negatives
                }
                for name, m in self.pattern_metrics.items()
            }
        }


class TCNValidationService:
    """
    Service for validating TCN models.

    Computes multiple metrics including:
    - Classification metrics (precision, recall, F1)
    - Outcome-based metrics (win rate, profit factor)
    - Confidence calibration
    """

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

    # Confidence threshold for positive prediction
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, device: str = "cuda"):
        """Initialize validation service."""
        self.device = device

    def _ensure_torch(self):
        """Ensure PyTorch is loaded."""
        if not _load_torch():
            raise RuntimeError("PyTorch not available")

    def _load_model(self, model_path: str):
        """Load a model from path."""
        self._ensure_torch()
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        from .tcn_model import TCNPatternModel

        checkpoint = torch.load(model_path, map_location=device)

        model = TCNPatternModel(
            num_inputs=5,
            num_channels=[32, 64, 128],
            num_classes=len(self.PATTERN_CLASSES),
            kernel_size=3,
            dropout=0.2
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, device

    def validate_on_test_data(
        self,
        model_path: str,
        test_sequences: np.ndarray,
        test_labels: np.ndarray,
        threshold: float = 0.5
    ) -> ValidationResult:
        """
        Validate model on test data.

        Args:
            model_path: Path to model checkpoint
            test_sequences: Test sequences (N, seq_len, features)
            test_labels: Test labels (N, num_classes)
            threshold: Confidence threshold for positive prediction

        Returns:
            ValidationResult with computed metrics
        """
        self._ensure_torch()

        model, device = self._load_model(model_path)

        result = ValidationResult(model_path=model_path, total_samples=len(test_sequences))

        # Initialize pattern metrics
        for pattern in self.PATTERN_CLASSES:
            result.pattern_metrics[pattern] = PatternMetrics(pattern_type=pattern)

        # Get predictions
        with torch.no_grad():
            sequences_tensor = torch.tensor(test_sequences, dtype=torch.float32).to(device)
            # Transpose for TCN: (batch, seq, features) -> (batch, features, seq)
            sequences_tensor = sequences_tensor.transpose(1, 2)

            predictions = model(sequences_tensor).cpu().numpy()

        # Compute metrics for each pattern
        for pattern_idx, pattern_name in enumerate(self.PATTERN_CLASSES):
            y_true = test_labels[:, pattern_idx]
            y_pred_prob = predictions[:, pattern_idx]
            y_pred = (y_pred_prob >= threshold).astype(int)

            metrics = result.pattern_metrics[pattern_name]

            for i in range(len(y_true)):
                if y_pred[i] == 1 and y_true[i] >= threshold:
                    metrics.true_positives += 1
                elif y_pred[i] == 1 and y_true[i] < threshold:
                    metrics.false_positives += 1
                elif y_pred[i] == 0 and y_true[i] >= threshold:
                    metrics.false_negatives += 1
                else:
                    metrics.true_negatives += 1

        # Compute overall metrics
        self._compute_overall_metrics(result)

        return result

    def validate_with_outcomes(
        self,
        model_path: str,
        samples_with_outcomes: List[Dict],
        threshold: float = 0.5
    ) -> ValidationResult:
        """
        Validate model using samples with real outcome data.

        Args:
            model_path: Path to model checkpoint
            samples_with_outcomes: List of samples with outcome information
            threshold: Confidence threshold

        Returns:
            ValidationResult with outcome-based metrics
        """
        self._ensure_torch()

        if not samples_with_outcomes:
            return ValidationResult(model_path=model_path)

        model, device = self._load_model(model_path)

        result = ValidationResult(
            model_path=model_path,
            total_samples=len(samples_with_outcomes),
            samples_with_outcomes=len(samples_with_outcomes)
        )

        # Initialize pattern metrics
        for pattern in self.PATTERN_CLASSES:
            result.pattern_metrics[pattern] = PatternMetrics(pattern_type=pattern)

        # Collect outcome metrics
        favorable_moves = []
        adverse_moves = []
        confidence_outcome_pairs = []  # For calibration

        for sample in samples_with_outcomes:
            sequence = np.array(sample["ohlcv_sequence"], dtype=np.float32)
            outcome = sample.get("outcome", {})
            pattern_label = sample.get("pattern_type")
            label_confidence = sample.get("original_confidence", 0.5)

            # Get model prediction
            with torch.no_grad():
                seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                seq_tensor = seq_tensor.transpose(1, 2)
                pred = model(seq_tensor).cpu().numpy()[0]

            # Update pattern metrics
            if pattern_label and pattern_label in self.PATTERN_CLASSES:
                pattern_idx = self.PATTERN_CLASSES.index(pattern_label)
                predicted_conf = pred[pattern_idx]
                metrics = result.pattern_metrics[pattern_label]

                is_success = outcome.get("is_success", False)

                if predicted_conf >= threshold:
                    if is_success:
                        metrics.true_positives += 1
                    else:
                        metrics.false_positives += 1
                else:
                    if is_success:
                        metrics.false_negatives += 1
                    else:
                        metrics.true_negatives += 1

                # Collect for outcome metrics
                favorable = outcome.get("max_favorable_move", 0)
                adverse = outcome.get("max_adverse_move", 0)

                if favorable > 0:
                    favorable_moves.append(favorable)
                if adverse > 0:
                    adverse_moves.append(adverse)

                # For calibration: (predicted confidence, actual success rate)
                confidence_outcome_pairs.append((predicted_conf, 1.0 if is_success else 0.0))

        # Compute outcome metrics
        if favorable_moves or adverse_moves:
            result.avg_favorable_move = np.mean(favorable_moves) if favorable_moves else 0
            result.avg_adverse_move = np.mean(adverse_moves) if adverse_moves else 0

            sum_favorable = sum(favorable_moves)
            sum_adverse = sum(adverse_moves)
            result.profit_factor = sum_favorable / sum_adverse if sum_adverse > 0 else float('inf')

            # Win rate (favorable > adverse)
            wins = sum(1 for f, a in zip(favorable_moves, adverse_moves) if f > a)
            total_trades = len(favorable_moves)
            result.win_rate = wins / total_trades if total_trades > 0 else 0

        # Compute outcome accuracy
        successes = sum(1 for s in samples_with_outcomes if s.get("outcome", {}).get("is_success", False))
        result.outcome_accuracy = successes / len(samples_with_outcomes) if samples_with_outcomes else 0

        # Compute confidence calibration
        result.confidence_calibration = self._compute_calibration(confidence_outcome_pairs)

        # Compute overall metrics
        self._compute_overall_metrics(result)

        return result

    def _compute_overall_metrics(self, result: ValidationResult) -> None:
        """Compute overall metrics from per-pattern metrics."""
        total_tp = sum(m.true_positives for m in result.pattern_metrics.values())
        total_fp = sum(m.false_positives for m in result.pattern_metrics.values())
        total_fn = sum(m.false_negatives for m in result.pattern_metrics.values())
        total_tn = sum(m.true_negatives for m in result.pattern_metrics.values())

        # Micro-averaged metrics
        result.overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        result.overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        p, r = result.overall_precision, result.overall_recall
        result.overall_f1_score = 2 * p * r / (p + r) if (p + r) > 0 else 0

        total = total_tp + total_tn + total_fp + total_fn
        result.overall_accuracy = (total_tp + total_tn) / total if total > 0 else 0

    def _compute_calibration(self, confidence_outcome_pairs: List[Tuple[float, float]]) -> float:
        """
        Compute confidence calibration score.

        A well-calibrated model has predicted confidence close to actual success rate.
        Returns calibration error (lower is better).
        """
        if not confidence_outcome_pairs:
            return 0.0

        # Bin predictions by confidence level
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_data = {i: [] for i in range(len(bins) - 1)}

        for conf, outcome in confidence_outcome_pairs:
            for i in range(len(bins) - 1):
                if bins[i] <= conf < bins[i + 1]:
                    bin_data[i].append((conf, outcome))
                    break

        # Calculate Expected Calibration Error (ECE)
        total_samples = len(confidence_outcome_pairs)
        ece = 0.0

        for bin_idx, pairs in bin_data.items():
            if pairs:
                avg_confidence = np.mean([p[0] for p in pairs])
                avg_accuracy = np.mean([p[1] for p in pairs])
                weight = len(pairs) / total_samples
                ece += weight * abs(avg_confidence - avg_accuracy)

        # Convert to calibration score (1 - ECE, higher is better)
        return 1.0 - min(ece, 1.0)

    def compare_models(
        self,
        result_a: ValidationResult,
        result_b: ValidationResult
    ) -> Dict:
        """
        Compare two validation results.

        Args:
            result_a: Validation result for model A (usually production)
            result_b: Validation result for model B (usually candidate)

        Returns:
            Comparison dictionary with deltas
        """
        return {
            "model_a": result_a.model_path,
            "model_b": result_b.model_path,
            "deltas": {
                "precision": round(result_b.overall_precision - result_a.overall_precision, 4),
                "recall": round(result_b.overall_recall - result_a.overall_recall, 4),
                "f1_score": round(result_b.overall_f1_score - result_a.overall_f1_score, 4),
                "accuracy": round(result_b.overall_accuracy - result_a.overall_accuracy, 4),
                "outcome_accuracy": round(result_b.outcome_accuracy - result_a.outcome_accuracy, 4),
                "profit_factor": round(result_b.profit_factor - result_a.profit_factor, 4),
                "win_rate": round(result_b.win_rate - result_a.win_rate, 4),
                "confidence_calibration": round(result_b.confidence_calibration - result_a.confidence_calibration, 4)
            },
            "result_a": result_a.to_dict(),
            "result_b": result_b.to_dict()
        }


# Singleton instance
validation_service = TCNValidationService()
