"""
TCN A/B Comparison Service.

Compares candidate models against production model to determine
if the candidate should be deployed.
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from loguru import logger

from .tcn_validation_service import validation_service, ValidationResult


class DeploymentRecommendation(str, Enum):
    """Deployment recommendations."""
    DEPLOY = "deploy"  # Clear improvement
    REJECT = "reject"  # Critical regression
    MANUAL_REVIEW = "manual_review"  # Mixed results


@dataclass
class ComparisonThresholds:
    """Thresholds for deployment decisions."""
    # Minimum improvement for auto-deploy
    min_improvement_precision: float = 0.02
    min_improvement_recall: float = 0.02
    min_improvement_f1: float = 0.02
    min_improvement_outcome_accuracy: float = 0.03

    # Maximum acceptable regression
    max_regression_precision: float = 0.05
    max_regression_recall: float = 0.05
    max_regression_f1: float = 0.05
    max_regression_outcome_accuracy: float = 0.05

    # Critical thresholds (auto-reject)
    critical_regression_f1: float = 0.10
    critical_regression_outcome_accuracy: float = 0.10

    # Minimum samples for reliable comparison
    min_samples_for_comparison: int = 50


@dataclass
class ABComparisonResult:
    """Result of A/B comparison between two models."""
    production_model: str
    candidate_model: str
    comparison_time: datetime = field(default_factory=datetime.utcnow)

    # Recommendation
    recommendation: DeploymentRecommendation = DeploymentRecommendation.MANUAL_REVIEW
    confidence: float = 0.0  # 0-1, how confident we are in recommendation
    reasons: List[str] = field(default_factory=list)

    # Metric deltas (candidate - production)
    delta_precision: float = 0.0
    delta_recall: float = 0.0
    delta_f1_score: float = 0.0
    delta_accuracy: float = 0.0
    delta_outcome_accuracy: float = 0.0
    delta_profit_factor: float = 0.0
    delta_win_rate: float = 0.0

    # Sample counts
    samples_used: int = 0

    # Full validation results
    production_result: Optional[ValidationResult] = None
    candidate_result: Optional[ValidationResult] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "production_model": self.production_model,
            "candidate_model": self.candidate_model,
            "comparison_time": self.comparison_time.isoformat(),
            "recommendation": self.recommendation.value,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "deltas": {
                "precision": round(self.delta_precision, 4),
                "recall": round(self.delta_recall, 4),
                "f1_score": round(self.delta_f1_score, 4),
                "accuracy": round(self.delta_accuracy, 4),
                "outcome_accuracy": round(self.delta_outcome_accuracy, 4),
                "profit_factor": round(self.delta_profit_factor, 4),
                "win_rate": round(self.delta_win_rate, 4)
            },
            "samples_used": self.samples_used,
            "production_metrics": self.production_result.to_dict() if self.production_result else None,
            "candidate_metrics": self.candidate_result.to_dict() if self.candidate_result else None
        }


class TCNABComparisonService:
    """
    A/B Comparison Service for TCN models.

    Compares a candidate model against production and determines
    whether the candidate should be deployed based on multiple metrics.
    """

    HISTORY_FILE = "data/models/tcn/ab_comparison_history.json"
    MAX_HISTORY_ENTRIES = 50

    def __init__(self):
        """Initialize A/B comparison service."""
        self._comparison_history: List[Dict] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load comparison history."""
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r') as f:
                    self._comparison_history = json.load(f)
                logger.info(f"Loaded {len(self._comparison_history)} A/B comparison entries")
        except Exception as e:
            logger.warning(f"Could not load A/B comparison history: {e}")
            self._comparison_history = []

    def _save_history(self) -> None:
        """Save comparison history."""
        try:
            os.makedirs(os.path.dirname(self.HISTORY_FILE), exist_ok=True)
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(self._comparison_history[-self.MAX_HISTORY_ENTRIES:], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save A/B comparison history: {e}")

    def compare_with_test_data(
        self,
        production_model_path: str,
        candidate_model_path: str,
        test_sequences: np.ndarray,
        test_labels: np.ndarray,
        thresholds: Optional[ComparisonThresholds] = None
    ) -> ABComparisonResult:
        """
        Compare models using test data.

        Args:
            production_model_path: Path to production model
            candidate_model_path: Path to candidate model
            test_sequences: Test sequences
            test_labels: Test labels
            thresholds: Comparison thresholds

        Returns:
            ABComparisonResult with recommendation
        """
        thresholds = thresholds or ComparisonThresholds()

        # Validate both models
        logger.info(f"Validating production model: {production_model_path}")
        prod_result = validation_service.validate_on_test_data(
            production_model_path, test_sequences, test_labels
        )

        logger.info(f"Validating candidate model: {candidate_model_path}")
        cand_result = validation_service.validate_on_test_data(
            candidate_model_path, test_sequences, test_labels
        )

        # Create comparison
        result = self._create_comparison(
            production_model_path,
            candidate_model_path,
            prod_result,
            cand_result,
            thresholds
        )

        # Save to history
        self._comparison_history.append(result.to_dict())
        self._save_history()

        return result

    def compare_with_outcomes(
        self,
        production_model_path: str,
        candidate_model_path: str,
        samples_with_outcomes: List[Dict],
        thresholds: Optional[ComparisonThresholds] = None
    ) -> ABComparisonResult:
        """
        Compare models using samples with real outcome data.

        Args:
            production_model_path: Path to production model
            candidate_model_path: Path to candidate model
            samples_with_outcomes: Samples with outcome information
            thresholds: Comparison thresholds

        Returns:
            ABComparisonResult with recommendation
        """
        thresholds = thresholds or ComparisonThresholds()

        if len(samples_with_outcomes) < thresholds.min_samples_for_comparison:
            logger.warning(
                f"Insufficient samples for comparison: {len(samples_with_outcomes)} < {thresholds.min_samples_for_comparison}"
            )
            return ABComparisonResult(
                production_model=production_model_path,
                candidate_model=candidate_model_path,
                recommendation=DeploymentRecommendation.MANUAL_REVIEW,
                confidence=0.0,
                reasons=[f"Insufficient samples ({len(samples_with_outcomes)})"],
                samples_used=len(samples_with_outcomes)
            )

        # Validate both models
        logger.info(f"Validating production model with outcomes: {production_model_path}")
        prod_result = validation_service.validate_with_outcomes(
            production_model_path, samples_with_outcomes
        )

        logger.info(f"Validating candidate model with outcomes: {candidate_model_path}")
        cand_result = validation_service.validate_with_outcomes(
            candidate_model_path, samples_with_outcomes
        )

        # Create comparison
        result = self._create_comparison(
            production_model_path,
            candidate_model_path,
            prod_result,
            cand_result,
            thresholds
        )

        # Save to history
        self._comparison_history.append(result.to_dict())
        self._save_history()

        return result

    def _create_comparison(
        self,
        production_path: str,
        candidate_path: str,
        prod_result: ValidationResult,
        cand_result: ValidationResult,
        thresholds: ComparisonThresholds
    ) -> ABComparisonResult:
        """Create comparison result from validation results."""
        result = ABComparisonResult(
            production_model=production_path,
            candidate_model=candidate_path,
            production_result=prod_result,
            candidate_result=cand_result,
            samples_used=prod_result.total_samples
        )

        # Calculate deltas
        result.delta_precision = cand_result.overall_precision - prod_result.overall_precision
        result.delta_recall = cand_result.overall_recall - prod_result.overall_recall
        result.delta_f1_score = cand_result.overall_f1_score - prod_result.overall_f1_score
        result.delta_accuracy = cand_result.overall_accuracy - prod_result.overall_accuracy
        result.delta_outcome_accuracy = cand_result.outcome_accuracy - prod_result.outcome_accuracy
        result.delta_profit_factor = cand_result.profit_factor - prod_result.profit_factor
        result.delta_win_rate = cand_result.win_rate - prod_result.win_rate

        # Determine recommendation
        result.recommendation, result.confidence, result.reasons = self._determine_recommendation(
            result, thresholds
        )

        logger.info(
            f"A/B Comparison: {result.recommendation.value} "
            f"(confidence: {result.confidence:.2f}) - {', '.join(result.reasons)}"
        )

        return result

    def _determine_recommendation(
        self,
        result: ABComparisonResult,
        thresholds: ComparisonThresholds
    ) -> tuple[DeploymentRecommendation, float, List[str]]:
        """Determine deployment recommendation based on deltas."""
        reasons = []
        improvements = 0
        regressions = 0
        critical_regressions = False

        # Check F1 Score (primary metric)
        if result.delta_f1_score <= -thresholds.critical_regression_f1:
            critical_regressions = True
            reasons.append(f"Critical F1 regression: {result.delta_f1_score:.4f}")
        elif result.delta_f1_score <= -thresholds.max_regression_f1:
            regressions += 2  # Weight F1 higher
            reasons.append(f"F1 regression: {result.delta_f1_score:.4f}")
        elif result.delta_f1_score >= thresholds.min_improvement_f1:
            improvements += 2
            reasons.append(f"F1 improvement: +{result.delta_f1_score:.4f}")

        # Check Outcome Accuracy
        if result.delta_outcome_accuracy <= -thresholds.critical_regression_outcome_accuracy:
            critical_regressions = True
            reasons.append(f"Critical outcome accuracy regression: {result.delta_outcome_accuracy:.4f}")
        elif result.delta_outcome_accuracy <= -thresholds.max_regression_outcome_accuracy:
            regressions += 2
            reasons.append(f"Outcome accuracy regression: {result.delta_outcome_accuracy:.4f}")
        elif result.delta_outcome_accuracy >= thresholds.min_improvement_outcome_accuracy:
            improvements += 2
            reasons.append(f"Outcome accuracy improvement: +{result.delta_outcome_accuracy:.4f}")

        # Check Precision
        if result.delta_precision <= -thresholds.max_regression_precision:
            regressions += 1
            reasons.append(f"Precision regression: {result.delta_precision:.4f}")
        elif result.delta_precision >= thresholds.min_improvement_precision:
            improvements += 1
            reasons.append(f"Precision improvement: +{result.delta_precision:.4f}")

        # Check Recall
        if result.delta_recall <= -thresholds.max_regression_recall:
            regressions += 1
            reasons.append(f"Recall regression: {result.delta_recall:.4f}")
        elif result.delta_recall >= thresholds.min_improvement_recall:
            improvements += 1
            reasons.append(f"Recall improvement: +{result.delta_recall:.4f}")

        # Check Win Rate (if available)
        if result.delta_win_rate > 0.02:
            improvements += 1
            reasons.append(f"Win rate improvement: +{result.delta_win_rate:.4f}")
        elif result.delta_win_rate < -0.03:
            regressions += 1
            reasons.append(f"Win rate regression: {result.delta_win_rate:.4f}")

        # Determine recommendation
        if critical_regressions:
            recommendation = DeploymentRecommendation.REJECT
            confidence = 0.95
        elif improvements >= 4 and regressions == 0:
            recommendation = DeploymentRecommendation.DEPLOY
            confidence = min(0.9, 0.5 + improvements * 0.1)
        elif improvements >= 2 and regressions == 0:
            recommendation = DeploymentRecommendation.DEPLOY
            confidence = min(0.8, 0.4 + improvements * 0.1)
        elif regressions >= 3:
            recommendation = DeploymentRecommendation.REJECT
            confidence = min(0.85, 0.5 + regressions * 0.1)
        else:
            recommendation = DeploymentRecommendation.MANUAL_REVIEW
            confidence = 0.5

        if not reasons:
            reasons.append("No significant changes detected")

        return recommendation, confidence, reasons

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get comparison history."""
        return self._comparison_history[-limit:]

    def get_statistics(self) -> Dict:
        """Get comparison statistics."""
        total = len(self._comparison_history)
        if total == 0:
            return {
                "total_comparisons": 0,
                "deploy_count": 0,
                "reject_count": 0,
                "manual_review_count": 0
            }

        deploy_count = sum(1 for h in self._comparison_history if h.get("recommendation") == "deploy")
        reject_count = sum(1 for h in self._comparison_history if h.get("recommendation") == "reject")
        manual_review_count = sum(1 for h in self._comparison_history if h.get("recommendation") == "manual_review")

        return {
            "total_comparisons": total,
            "deploy_count": deploy_count,
            "reject_count": reject_count,
            "manual_review_count": manual_review_count,
            "deploy_rate": deploy_count / total if total > 0 else 0,
            "reject_rate": reject_count / total if total > 0 else 0
        }


# Singleton instance
ab_comparison_service = TCNABComparisonService()
