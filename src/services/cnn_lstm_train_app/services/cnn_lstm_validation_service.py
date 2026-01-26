"""
CNN-LSTM Model Validation Service.

Validates multi-task model performance for:
- Price direction prediction
- Pattern classification (16 classes)
- Regime prediction (4 classes)
"""

import os
from typing import Dict, List, Optional, Tuple, Any
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


class ValidationRecommendation(str, Enum):
    """Deployment recommendation based on validation."""
    DEPLOY = "deploy"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


class ValidationMetricType(str, Enum):
    """Types of validation metrics."""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    MAE = "mean_absolute_error"
    DIRECTION_ACCURACY = "direction_accuracy"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    task_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    samples: int = 0

    # Task-specific metrics
    mae: Optional[float] = None  # For price
    direction_accuracy: Optional[float] = None  # For price

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "task_name": self.task_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "samples": self.samples
        }
        if self.mae is not None:
            result["mae"] = round(self.mae, 4)
        if self.direction_accuracy is not None:
            result["direction_accuracy"] = round(self.direction_accuracy, 4)
        return result


@dataclass
class ValidationResult:
    """Complete validation result for a CNN-LSTM model."""
    model_path: str
    validation_time: datetime = field(default_factory=datetime.utcnow)

    # Per-task metrics
    price_metrics: TaskMetrics = field(default_factory=lambda: TaskMetrics("price"))
    pattern_metrics: TaskMetrics = field(default_factory=lambda: TaskMetrics("patterns"))
    regime_metrics: TaskMetrics = field(default_factory=lambda: TaskMetrics("regime"))

    # Overall metrics (weighted average)
    overall_accuracy: float = 0.0
    overall_f1_score: float = 0.0

    # Confidence calibration
    confidence_calibration: float = 0.0

    # Sample counts
    total_samples: int = 0

    # Recommendation
    recommendation: ValidationRecommendation = ValidationRecommendation.MANUAL_REVIEW
    recommendation_reason: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "validation_time": self.validation_time.isoformat(),
            "tasks": {
                "price": self.price_metrics.to_dict(),
                "patterns": self.pattern_metrics.to_dict(),
                "regime": self.regime_metrics.to_dict()
            },
            "overall": {
                "accuracy": round(self.overall_accuracy, 4),
                "f1_score": round(self.overall_f1_score, 4),
                "confidence_calibration": round(self.confidence_calibration, 4)
            },
            "total_samples": self.total_samples,
            "recommendation": self.recommendation.value,
            "recommendation_reason": self.recommendation_reason
        }


@dataclass
class ABComparisonResult:
    """Result of A/B comparison between two models."""
    production_model: str
    candidate_model: str
    comparison_time: datetime = field(default_factory=datetime.utcnow)

    # Per-task deltas (candidate - production)
    price_delta: Dict = field(default_factory=dict)
    pattern_delta: Dict = field(default_factory=dict)
    regime_delta: Dict = field(default_factory=dict)

    # Overall deltas
    overall_accuracy_delta: float = 0.0
    overall_f1_delta: float = 0.0

    # Recommendation
    recommendation: ValidationRecommendation = ValidationRecommendation.MANUAL_REVIEW
    recommendation_reason: str = ""
    confidence: float = 0.0

    # Full results
    production_result: Optional[ValidationResult] = None
    candidate_result: Optional[ValidationResult] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "production_model": self.production_model,
            "candidate_model": self.candidate_model,
            "comparison_time": self.comparison_time.isoformat(),
            "deltas": {
                "price": self.price_delta,
                "patterns": self.pattern_delta,
                "regime": self.regime_delta,
                "overall_accuracy": round(self.overall_accuracy_delta, 4),
                "overall_f1": round(self.overall_f1_delta, 4)
            },
            "recommendation": self.recommendation.value,
            "recommendation_reason": self.recommendation_reason,
            "confidence": round(self.confidence, 4),
            "production_result": self.production_result.to_dict() if self.production_result else None,
            "candidate_result": self.candidate_result.to_dict() if self.candidate_result else None
        }


class CNNLSTMValidationService:
    """
    Service for validating CNN-LSTM multi-task models.

    Computes metrics for all three tasks:
    - Price: Direction accuracy, MAE
    - Patterns: Multi-label classification metrics (16 patterns)
    - Regime: Multi-class classification metrics (4 regimes)
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

    REGIME_CLASSES = ["bull_trend", "bear_trend", "sideways", "high_volatility"]

    # Task weights for overall score
    TASK_WEIGHTS = {
        "price": 0.40,
        "patterns": 0.35,
        "regime": 0.25
    }

    # Thresholds for recommendations
    MIN_IMPROVEMENT_FOR_DEPLOY = 0.02  # 2% improvement
    MAX_REGRESSION_FOR_DEPLOY = 0.05  # 5% max regression
    CRITICAL_REGRESSION_THRESHOLD = 0.10  # 10% triggers reject

    def __init__(self, device: str = "cuda"):
        """Initialize validation service."""
        self.device = device
        self._validation_history: List[ValidationResult] = []
        self._comparison_history: List[ABComparisonResult] = []

    def _ensure_torch(self):
        """Ensure PyTorch is loaded."""
        if not _load_torch():
            raise RuntimeError("PyTorch not available")

    def _load_model(self, model_path: str):
        """Load a model from path."""
        self._ensure_torch()
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # Import model architecture
        from .cnn_lstm_model import CNNLSTMModel

        # Get config from checkpoint or use defaults
        config = checkpoint.get("config", {})
        input_dim = config.get("input_dim", 25)

        model = CNNLSTMModel(
            input_dim=input_dim,
            num_patterns=len(self.PATTERN_CLASSES),
            num_regimes=len(self.REGIME_CLASSES)
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, device

    def validate_model(
        self,
        model_path: str,
        test_data: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> ValidationResult:
        """
        Validate a model on test data.

        Args:
            model_path: Path to model checkpoint
            test_data: List of samples with features, labels, and outcomes
            threshold: Confidence threshold for classification

        Returns:
            ValidationResult with computed metrics
        """
        self._ensure_torch()

        if not test_data:
            return ValidationResult(model_path=model_path)

        model, device = self._load_model(model_path)

        result = ValidationResult(
            model_path=model_path,
            total_samples=len(test_data)
        )

        # Collect predictions and labels
        price_predictions = []
        price_labels = []
        pattern_predictions = []
        pattern_labels = []
        regime_predictions = []
        regime_labels = []
        confidence_outcome_pairs = []

        for sample in test_data:
            features = np.array(sample.get("features", []), dtype=np.float32)
            if len(features) == 0:
                continue

            # Get model predictions
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                outputs = model(features_tensor)

            # Price predictions
            if "price" in outputs:
                price_pred = outputs["price"].cpu().numpy()[0]
                price_predictions.append(price_pred)

                if "price_label" in sample:
                    price_labels.append(sample["price_label"])

            # Pattern predictions
            if "patterns" in outputs:
                pattern_pred = torch.sigmoid(outputs["patterns"]).cpu().numpy()[0]
                pattern_predictions.append(pattern_pred)

                if "pattern_labels" in sample:
                    pattern_labels.append(sample["pattern_labels"])

            # Regime predictions
            if "regime" in outputs:
                regime_pred = torch.softmax(outputs["regime"], dim=-1).cpu().numpy()[0]
                regime_predictions.append(regime_pred)

                if "regime_label" in sample:
                    regime_labels.append(sample["regime_label"])

            # Confidence-outcome pairs for calibration
            if "outcome" in sample:
                avg_confidence = np.mean([
                    outputs.get("price_confidence", torch.tensor([0.5])).cpu().numpy().mean(),
                    outputs.get("pattern_confidence", torch.tensor([0.5])).cpu().numpy().mean() if "pattern_confidence" in outputs else 0.5,
                    outputs.get("regime_confidence", torch.tensor([0.5])).cpu().numpy().mean() if "regime_confidence" in outputs else 0.5
                ])
                is_success = sample["outcome"].get("is_success", False)
                confidence_outcome_pairs.append((avg_confidence, 1.0 if is_success else 0.0))

        # Compute price metrics
        if price_predictions and price_labels:
            result.price_metrics = self._compute_price_metrics(
                np.array(price_predictions),
                np.array(price_labels)
            )

        # Compute pattern metrics
        if pattern_predictions and pattern_labels:
            result.pattern_metrics = self._compute_pattern_metrics(
                np.array(pattern_predictions),
                np.array(pattern_labels),
                threshold
            )

        # Compute regime metrics
        if regime_predictions and regime_labels:
            result.regime_metrics = self._compute_regime_metrics(
                np.array(regime_predictions),
                np.array(regime_labels)
            )

        # Compute overall metrics (weighted average)
        result.overall_accuracy = (
            self.TASK_WEIGHTS["price"] * result.price_metrics.accuracy +
            self.TASK_WEIGHTS["patterns"] * result.pattern_metrics.accuracy +
            self.TASK_WEIGHTS["regime"] * result.regime_metrics.accuracy
        )

        result.overall_f1_score = (
            self.TASK_WEIGHTS["price"] * result.price_metrics.f1_score +
            self.TASK_WEIGHTS["patterns"] * result.pattern_metrics.f1_score +
            self.TASK_WEIGHTS["regime"] * result.regime_metrics.f1_score
        )

        # Compute confidence calibration
        if confidence_outcome_pairs:
            result.confidence_calibration = self._compute_calibration(confidence_outcome_pairs)

        # Determine recommendation
        self._determine_recommendation(result)

        # Store in history
        self._validation_history.append(result)

        return result

    def _compute_price_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> TaskMetrics:
        """Compute metrics for price prediction task."""
        metrics = TaskMetrics(task_name="price", samples=len(predictions))

        # Direction accuracy (bullish vs bearish)
        pred_direction = predictions[:, 0] > 0  # Assuming first output is direction/change
        true_direction = labels[:, 0] > 0

        direction_correct = (pred_direction == true_direction).sum()
        metrics.direction_accuracy = direction_correct / len(predictions) if len(predictions) > 0 else 0
        metrics.accuracy = metrics.direction_accuracy

        # MAE for price predictions
        if predictions.shape[1] > 1 and labels.shape[1] > 1:
            metrics.mae = np.mean(np.abs(predictions[:, 1:] - labels[:, 1:]))

        # For binary direction: compute precision, recall, f1
        tp = ((pred_direction == True) & (true_direction == True)).sum()
        fp = ((pred_direction == True) & (true_direction == False)).sum()
        fn = ((pred_direction == False) & (true_direction == True)).sum()

        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0

        return metrics

    def _compute_pattern_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float
    ) -> TaskMetrics:
        """Compute metrics for multi-label pattern classification."""
        metrics = TaskMetrics(task_name="patterns", samples=len(predictions))

        # Binarize predictions
        pred_binary = (predictions >= threshold).astype(int)
        label_binary = (labels >= threshold).astype(int)

        # Per-class metrics, then average
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(predictions.shape[1]):
            tp = ((pred_binary[:, i] == 1) & (label_binary[:, i] == 1)).sum()
            fp = ((pred_binary[:, i] == 1) & (label_binary[:, i] == 0)).sum()
            fn = ((pred_binary[:, i] == 0) & (label_binary[:, i] == 1)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)

        metrics.precision = np.mean(precisions)
        metrics.recall = np.mean(recalls)
        metrics.f1_score = np.mean(f1_scores)

        # Overall accuracy (exact match or sample-based)
        correct_samples = np.all(pred_binary == label_binary, axis=1).sum()
        metrics.accuracy = correct_samples / len(predictions) if len(predictions) > 0 else 0

        return metrics

    def _compute_regime_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> TaskMetrics:
        """Compute metrics for multi-class regime classification."""
        metrics = TaskMetrics(task_name="regime", samples=len(predictions))

        # Get predicted class (argmax)
        pred_class = np.argmax(predictions, axis=1)

        # Labels might be one-hot or class indices
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            true_class = np.argmax(labels, axis=1)
        else:
            true_class = labels.astype(int).flatten()

        # Overall accuracy
        metrics.accuracy = (pred_class == true_class).sum() / len(predictions) if len(predictions) > 0 else 0

        # Per-class precision/recall for macro-average
        precisions = []
        recalls = []

        for c in range(len(self.REGIME_CLASSES)):
            tp = ((pred_class == c) & (true_class == c)).sum()
            fp = ((pred_class == c) & (true_class != c)).sum()
            fn = ((pred_class != c) & (true_class == c)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions.append(p)
            recalls.append(r)

        metrics.precision = np.mean(precisions)
        metrics.recall = np.mean(recalls)
        metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0

        return metrics

    def _compute_calibration(self, confidence_outcome_pairs: List[Tuple[float, float]]) -> float:
        """Compute confidence calibration score (1 - ECE)."""
        if not confidence_outcome_pairs:
            return 0.0

        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_data = {i: [] for i in range(len(bins) - 1)}

        for conf, outcome in confidence_outcome_pairs:
            for i in range(len(bins) - 1):
                if bins[i] <= conf < bins[i + 1]:
                    bin_data[i].append((conf, outcome))
                    break

        total_samples = len(confidence_outcome_pairs)
        ece = 0.0

        for bin_idx, pairs in bin_data.items():
            if pairs:
                avg_confidence = np.mean([p[0] for p in pairs])
                avg_accuracy = np.mean([p[1] for p in pairs])
                weight = len(pairs) / total_samples
                ece += weight * abs(avg_confidence - avg_accuracy)

        return 1.0 - min(ece, 1.0)

    def _determine_recommendation(self, result: ValidationResult) -> None:
        """Determine deployment recommendation based on metrics."""
        # Check if metrics meet minimum thresholds
        min_accuracy = 0.55
        min_f1 = 0.50

        if result.overall_accuracy >= min_accuracy and result.overall_f1_score >= min_f1:
            result.recommendation = ValidationRecommendation.DEPLOY
            result.recommendation_reason = f"Metrics meet thresholds (accuracy: {result.overall_accuracy:.2%}, F1: {result.overall_f1_score:.2%})"
        elif result.overall_accuracy < min_accuracy * 0.8:
            result.recommendation = ValidationRecommendation.REJECT
            result.recommendation_reason = f"Accuracy too low: {result.overall_accuracy:.2%} (min: {min_accuracy:.0%})"
        else:
            result.recommendation = ValidationRecommendation.MANUAL_REVIEW
            result.recommendation_reason = "Metrics are borderline - manual review recommended"

    def compare_models(
        self,
        production_path: str,
        candidate_path: str,
        test_data: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> ABComparisonResult:
        """
        Compare candidate model against production model.

        Args:
            production_path: Path to production model
            candidate_path: Path to candidate model
            test_data: Test data for validation
            threshold: Confidence threshold

        Returns:
            ABComparisonResult with comparison metrics and recommendation
        """
        # Validate both models
        prod_result = self.validate_model(production_path, test_data, threshold)
        cand_result = self.validate_model(candidate_path, test_data, threshold)

        # Create comparison result
        comparison = ABComparisonResult(
            production_model=production_path,
            candidate_model=candidate_path,
            production_result=prod_result,
            candidate_result=cand_result
        )

        # Compute deltas
        comparison.price_delta = {
            "accuracy": cand_result.price_metrics.accuracy - prod_result.price_metrics.accuracy,
            "f1_score": cand_result.price_metrics.f1_score - prod_result.price_metrics.f1_score,
            "direction_accuracy": (cand_result.price_metrics.direction_accuracy or 0) - (prod_result.price_metrics.direction_accuracy or 0)
        }

        comparison.pattern_delta = {
            "accuracy": cand_result.pattern_metrics.accuracy - prod_result.pattern_metrics.accuracy,
            "f1_score": cand_result.pattern_metrics.f1_score - prod_result.pattern_metrics.f1_score,
            "precision": cand_result.pattern_metrics.precision - prod_result.pattern_metrics.precision,
            "recall": cand_result.pattern_metrics.recall - prod_result.pattern_metrics.recall
        }

        comparison.regime_delta = {
            "accuracy": cand_result.regime_metrics.accuracy - prod_result.regime_metrics.accuracy,
            "f1_score": cand_result.regime_metrics.f1_score - prod_result.regime_metrics.f1_score
        }

        comparison.overall_accuracy_delta = cand_result.overall_accuracy - prod_result.overall_accuracy
        comparison.overall_f1_delta = cand_result.overall_f1_score - prod_result.overall_f1_score

        # Determine recommendation
        self._determine_ab_recommendation(comparison)

        # Store in history
        self._comparison_history.append(comparison)

        return comparison

    def _determine_ab_recommendation(self, comparison: ABComparisonResult) -> None:
        """Determine deployment recommendation from A/B comparison."""
        acc_delta = comparison.overall_accuracy_delta
        f1_delta = comparison.overall_f1_delta

        # Check for critical regression in any task
        critical_regression = False
        regression_task = None

        for task, delta in [("price", comparison.price_delta),
                           ("patterns", comparison.pattern_delta),
                           ("regime", comparison.regime_delta)]:
            if delta.get("accuracy", 0) < -self.CRITICAL_REGRESSION_THRESHOLD:
                critical_regression = True
                regression_task = task
                break

        if critical_regression:
            comparison.recommendation = ValidationRecommendation.REJECT
            comparison.recommendation_reason = f"Critical regression in {regression_task} task ({delta.get('accuracy', 0):.2%})"
            comparison.confidence = 0.9
        elif acc_delta >= self.MIN_IMPROVEMENT_FOR_DEPLOY and f1_delta >= 0:
            comparison.recommendation = ValidationRecommendation.DEPLOY
            comparison.recommendation_reason = f"Clear improvement: accuracy +{acc_delta:.2%}, F1 +{f1_delta:.2%}"
            comparison.confidence = 0.85
        elif acc_delta >= -self.MAX_REGRESSION_FOR_DEPLOY and f1_delta >= -self.MAX_REGRESSION_FOR_DEPLOY:
            comparison.recommendation = ValidationRecommendation.DEPLOY
            comparison.recommendation_reason = f"Acceptable performance: accuracy {acc_delta:+.2%}, F1 {f1_delta:+.2%}"
            comparison.confidence = 0.70
        elif acc_delta < -self.MAX_REGRESSION_FOR_DEPLOY or f1_delta < -self.MAX_REGRESSION_FOR_DEPLOY:
            comparison.recommendation = ValidationRecommendation.REJECT
            comparison.recommendation_reason = f"Performance regression: accuracy {acc_delta:+.2%}, F1 {f1_delta:+.2%}"
            comparison.confidence = 0.75
        else:
            comparison.recommendation = ValidationRecommendation.MANUAL_REVIEW
            comparison.recommendation_reason = f"Mixed results: accuracy {acc_delta:+.2%}, F1 {f1_delta:+.2%}"
            comparison.confidence = 0.50

    def get_validation_history(self, limit: int = 20) -> List[Dict]:
        """Get recent validation history."""
        return [r.to_dict() for r in self._validation_history[-limit:]]

    def get_comparison_history(self, limit: int = 20) -> List[Dict]:
        """Get recent comparison history."""
        return [c.to_dict() for c in self._comparison_history[-limit:]]

    def get_statistics(self) -> Dict:
        """Get validation service statistics."""
        return {
            "total_validations": len(self._validation_history),
            "total_comparisons": len(self._comparison_history),
            "task_weights": self.TASK_WEIGHTS,
            "thresholds": {
                "min_improvement_for_deploy": self.MIN_IMPROVEMENT_FOR_DEPLOY,
                "max_regression_for_deploy": self.MAX_REGRESSION_FOR_DEPLOY,
                "critical_regression": self.CRITICAL_REGRESSION_THRESHOLD
            }
        }


# Singleton instance
validation_service = CNNLSTMValidationService()
