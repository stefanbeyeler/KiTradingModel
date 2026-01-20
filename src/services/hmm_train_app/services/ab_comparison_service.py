"""A/B Comparison Service for model evaluation.

Compares new candidate models against current production models
using identical validation datasets.

Provides:
- Side-by-side metric comparison
- Improvement/regression calculation
- Deployment recommendation (deploy, reject, manual_review)
"""

import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from loguru import logger

from .model_registry import ModelRegistry, model_registry
from .validation_service import (
    ValidationService,
    validation_service,
    HMMValidationMetrics,
    ScorerValidationMetrics
)


class Recommendation(str, Enum):
    """Deployment recommendation."""
    DEPLOY = "deploy"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


@dataclass
class ABComparisonResult:
    """Result of A/B model comparison."""
    comparison_id: str
    timestamp: str

    model_type: str                    # "hmm" or "scorer"
    symbol: Optional[str] = None       # For HMM models

    # Model identifiers
    candidate_version: str = ""
    production_version: Optional[str] = None  # None if no production model

    # Metrics comparison
    candidate_metrics: Dict[str, float] = field(default_factory=dict)
    production_metrics: Optional[Dict[str, float]] = None

    # Improvement analysis
    improvements: Dict[str, float] = field(default_factory=dict)   # Metric -> % improvement
    regressions: Dict[str, float] = field(default_factory=dict)    # Metric -> % regression

    # Decision
    recommendation: Recommendation = Recommendation.DEPLOY
    recommendation_reason: str = ""

    # Thresholds used
    thresholds_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["recommendation"] = self.recommendation.value
        return d


class ABComparisonService:
    """
    Service for comparing candidate models against production.

    Uses configurable thresholds to make deployment recommendations:
    - DEPLOY: New model is better or within acceptable regression
    - REJECT: New model has critical regressions
    - MANUAL_REVIEW: Mixed results require human decision
    """

    # Default thresholds (can be overridden via environment)
    DEFAULT_THRESHOLDS = {
        # HMM thresholds - max allowed regression (negative = worse)
        "hmm_regime_accuracy_min": -0.03,       # Max 3% accuracy drop
        "hmm_regime_f1_weighted_min": -0.05,    # Max 5% F1 drop
        "hmm_log_likelihood_min": -0.10,        # Max 10% LL drop
        "hmm_confidence_calibration_min": -0.10,

        # Scorer thresholds
        "scorer_accuracy_min": -0.03,           # Max 3% accuracy drop
        "scorer_f1_weighted_min": -0.05,        # Max 5% F1 drop
        "scorer_profitable_signals_rate_min": 0.45,  # Minimum 45% profitable
        "scorer_mae_max_increase": 0.10,        # Max 10% MAE increase

        # General thresholds
        "improvement_threshold": 0.02,          # 2% improvement considered significant
        "first_model_min_accuracy": 0.50,       # Minimum accuracy for first deployment
    }

    def __init__(
        self,
        validation_svc: Optional[ValidationService] = None,
        registry: Optional[ModelRegistry] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        self._validation = validation_svc or validation_service
        self._registry = registry or model_registry
        self._thresholds = {**self.DEFAULT_THRESHOLDS}

        # Override with environment variables
        for key in self._thresholds:
            env_key = f"HMM_AB_{key.upper()}"
            env_val = os.getenv(env_key)
            if env_val:
                try:
                    self._thresholds[key] = float(env_val)
                except ValueError:
                    pass

        # Override with provided thresholds
        if thresholds:
            self._thresholds.update(thresholds)

        logger.info("ABComparisonService initialized")

    def _generate_comparison_id(self) -> str:
        """Generate unique comparison ID."""
        return f"cmp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # =========================================================================
    # HMM Comparison
    # =========================================================================

    async def compare_hmm_models(
        self,
        candidate_path: str,
        symbol: str,
        timeframe: str,
        candidate_version: str
    ) -> ABComparisonResult:
        """
        Compare candidate HMM model against production.

        Args:
            candidate_path: Path to candidate model file
            symbol: Symbol this model is for
            timeframe: Data timeframe
            candidate_version: Version ID of candidate

        Returns:
            ABComparisonResult with comparison details and recommendation
        """
        result = ABComparisonResult(
            comparison_id=self._generate_comparison_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_type="hmm",
            symbol=symbol,
            candidate_version=candidate_version,
            thresholds_used={k: v for k, v in self._thresholds.items() if k.startswith("hmm_")}
        )

        try:
            # Validate candidate model
            candidate_metrics = await self._validation.validate_hmm_model(
                candidate_path, symbol, timeframe, use_cached=True
            )
            result.candidate_metrics = candidate_metrics.to_flat_dict()

            # Check if validation succeeded
            if candidate_metrics.validation_samples == 0:
                result.recommendation = Recommendation.REJECT
                result.recommendation_reason = "Candidate validation failed - no samples"
                return result

            # Get production model
            production = self._registry.get_current_production("hmm", symbol)

            if production is None:
                # First model - check minimum quality
                result.production_version = None
                result.production_metrics = None

                min_acc = self._thresholds.get("first_model_min_accuracy", 0.50)
                if candidate_metrics.regime_accuracy >= min_acc:
                    result.recommendation = Recommendation.DEPLOY
                    result.recommendation_reason = f"First model for {symbol} - meets minimum accuracy ({candidate_metrics.regime_accuracy:.1%} >= {min_acc:.1%})"
                else:
                    result.recommendation = Recommendation.REJECT
                    result.recommendation_reason = f"First model below minimum accuracy ({candidate_metrics.regime_accuracy:.1%} < {min_acc:.1%})"

                return result

            # Validate production model
            production_path = self._registry.get_production_model_path("hmm", symbol)
            if production_path is None:
                result.recommendation = Recommendation.DEPLOY
                result.recommendation_reason = "Production model file not found - deploying candidate"
                return result

            production_metrics = await self._validation.validate_hmm_model(
                str(production_path), symbol, timeframe, use_cached=True
            )

            result.production_version = production.version_id
            result.production_metrics = production_metrics.to_flat_dict()

            # Calculate improvements and regressions
            improvements, regressions = self._calculate_changes(
                result.candidate_metrics,
                result.production_metrics,
                higher_is_better=["regime_accuracy", "regime_f1_weighted", "log_likelihood",
                                  "confidence_calibration", "regime_transition_accuracy"]
            )

            result.improvements = improvements
            result.regressions = regressions

            # Make recommendation
            result.recommendation, result.recommendation_reason = self._make_hmm_recommendation(
                candidate_metrics, production_metrics, improvements, regressions
            )

        except Exception as e:
            logger.error(f"HMM comparison failed: {e}")
            result.recommendation = Recommendation.MANUAL_REVIEW
            result.recommendation_reason = f"Comparison error: {str(e)}"

        return result

    def _make_hmm_recommendation(
        self,
        candidate: HMMValidationMetrics,
        production: HMMValidationMetrics,
        improvements: Dict[str, float],
        regressions: Dict[str, float]
    ) -> Tuple[Recommendation, str]:
        """Make deployment recommendation for HMM model."""

        # Check critical regressions
        critical_regressions = []

        # Accuracy check
        acc_threshold = self._thresholds.get("hmm_regime_accuracy_min", -0.03)
        acc_change = regressions.get("regime_accuracy", 0)
        if acc_change < acc_threshold:
            critical_regressions.append(
                f"accuracy: {acc_change:+.1%} (threshold: {acc_threshold:+.1%})"
            )

        # F1 check
        f1_threshold = self._thresholds.get("hmm_regime_f1_weighted_min", -0.05)
        f1_change = regressions.get("regime_f1_weighted", 0)
        if f1_change < f1_threshold:
            critical_regressions.append(
                f"f1_weighted: {f1_change:+.1%} (threshold: {f1_threshold:+.1%})"
            )

        # Log-likelihood check (relative)
        ll_threshold = self._thresholds.get("hmm_log_likelihood_min", -0.10)
        ll_change = regressions.get("log_likelihood", 0)
        if ll_change < ll_threshold:
            critical_regressions.append(
                f"log_likelihood: {ll_change:+.1%} (threshold: {ll_threshold:+.1%})"
            )

        # Critical regression -> REJECT
        if critical_regressions:
            return (
                Recommendation.REJECT,
                f"Critical regressions: {'; '.join(critical_regressions)}"
            )

        # Clear improvement -> DEPLOY
        improvement_threshold = self._thresholds.get("improvement_threshold", 0.02)
        significant_improvements = [
            metric for metric, change in improvements.items()
            if change >= improvement_threshold
        ]

        if len(significant_improvements) >= 2 and not regressions:
            return (
                Recommendation.DEPLOY,
                f"Significant improvements in: {', '.join(significant_improvements)}"
            )

        # Minor improvements or no change -> DEPLOY
        if not regressions or all(r >= -0.01 for r in regressions.values()):
            avg_change = 0
            all_changes = {**improvements, **{k: -v for k, v in regressions.items()}}
            if all_changes:
                avg_change = sum(all_changes.values()) / len(all_changes)

            if avg_change >= 0:
                return (
                    Recommendation.DEPLOY,
                    f"Overall improvement or stable (avg change: {avg_change:+.1%})"
                )

        # Mixed results -> MANUAL_REVIEW
        return (
            Recommendation.MANUAL_REVIEW,
            f"Mixed results: {len(improvements)} improvements, {len(regressions)} regressions"
        )

    # =========================================================================
    # Scorer Comparison
    # =========================================================================

    async def compare_scorer_models(
        self,
        candidate_path: str,
        symbols: List[str],
        timeframe: str,
        candidate_version: str
    ) -> ABComparisonResult:
        """
        Compare candidate scorer model against production.

        Args:
            candidate_path: Path to candidate scorer model
            symbols: Symbols to validate on
            timeframe: Data timeframe
            candidate_version: Version ID of candidate

        Returns:
            ABComparisonResult with comparison details
        """
        result = ABComparisonResult(
            comparison_id=self._generate_comparison_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_type="scorer",
            symbol=None,
            candidate_version=candidate_version,
            thresholds_used={k: v for k, v in self._thresholds.items() if k.startswith("scorer_")}
        )

        try:
            # Validate candidate
            candidate_metrics = await self._validation.validate_scorer_model(
                candidate_path, symbols, timeframe
            )
            result.candidate_metrics = candidate_metrics.to_flat_dict()

            if candidate_metrics.validation_samples == 0:
                result.recommendation = Recommendation.REJECT
                result.recommendation_reason = "Candidate validation failed - no samples"
                return result

            # Get production model
            production = self._registry.get_current_production("scorer", None)

            if production is None:
                # First scorer model - be lenient, just deploy if accuracy is reasonable
                result.production_version = None

                # For first model, only check basic accuracy (lower threshold)
                min_accuracy_first = self._thresholds.get("scorer_first_model_accuracy_min", 0.40)
                if candidate_metrics.accuracy >= min_accuracy_first:
                    result.recommendation = Recommendation.DEPLOY
                    result.recommendation_reason = (
                        f"First scorer model - accuracy ({candidate_metrics.accuracy:.1%}) "
                        f"meets minimum ({min_accuracy_first:.1%}), "
                        f"profitable rate: {candidate_metrics.profitable_signals_rate:.1%}"
                    )
                else:
                    result.recommendation = Recommendation.REJECT
                    result.recommendation_reason = (
                        f"First scorer below minimum accuracy "
                        f"({candidate_metrics.accuracy:.1%} < {min_accuracy_first:.1%})"
                    )

                return result

            # Validate production
            production_path = self._registry.get_production_model_path("scorer", None)
            if production_path is None:
                result.recommendation = Recommendation.DEPLOY
                result.recommendation_reason = "Production scorer not found - deploying candidate"
                return result

            production_metrics = await self._validation.validate_scorer_model(
                str(production_path), symbols, timeframe
            )

            result.production_version = production.version_id
            result.production_metrics = production_metrics.to_flat_dict()

            # Calculate changes
            # Note: For MAE/RMSE, lower is better
            improvements, regressions = self._calculate_changes(
                result.candidate_metrics,
                result.production_metrics,
                higher_is_better=["accuracy", "precision_weighted", "recall_weighted",
                                  "f1_weighted", "profitable_signals_rate", "avg_return_per_signal"],
                lower_is_better=["mae", "rmse"]
            )

            result.improvements = improvements
            result.regressions = regressions

            # Make recommendation
            result.recommendation, result.recommendation_reason = self._make_scorer_recommendation(
                candidate_metrics, production_metrics, improvements, regressions
            )

        except Exception as e:
            logger.error(f"Scorer comparison failed: {e}")
            result.recommendation = Recommendation.MANUAL_REVIEW
            result.recommendation_reason = f"Comparison error: {str(e)}"

        return result

    def _make_scorer_recommendation(
        self,
        candidate: ScorerValidationMetrics,
        production: ScorerValidationMetrics,
        improvements: Dict[str, float],
        regressions: Dict[str, float]
    ) -> Tuple[Recommendation, str]:
        """Make deployment recommendation for scorer model."""

        critical_regressions = []

        # Accuracy check
        acc_threshold = self._thresholds.get("scorer_accuracy_min", -0.03)
        acc_change = regressions.get("accuracy", 0)
        if acc_change < acc_threshold:
            critical_regressions.append(
                f"accuracy: {acc_change:+.1%} (threshold: {acc_threshold:+.1%})"
            )

        # Profitable rate check
        profitable_min = self._thresholds.get("scorer_profitable_signals_rate_min", 0.45)
        if candidate.profitable_signals_rate < profitable_min:
            critical_regressions.append(
                f"profitable_rate: {candidate.profitable_signals_rate:.1%} < {profitable_min:.1%} minimum"
            )

        # MAE increase check
        mae_threshold = self._thresholds.get("scorer_mae_max_increase", 0.10)
        mae_change = regressions.get("mae", 0)
        if mae_change < -mae_threshold:  # Remember: for MAE, negative change in "improvements" means MAE increased
            critical_regressions.append(
                f"mae: {-mae_change:+.1%} increase (max: {mae_threshold:+.1%})"
            )

        if critical_regressions:
            return (
                Recommendation.REJECT,
                f"Critical regressions: {'; '.join(critical_regressions)}"
            )

        # Check for improvements
        improvement_threshold = self._thresholds.get("improvement_threshold", 0.02)
        key_improvements = [
            metric for metric in ["accuracy", "f1_weighted", "profitable_signals_rate"]
            if improvements.get(metric, 0) >= improvement_threshold
        ]

        if len(key_improvements) >= 2:
            return (
                Recommendation.DEPLOY,
                f"Significant improvements in: {', '.join(key_improvements)}"
            )

        # No major regressions -> DEPLOY
        if not regressions or all(r >= -0.01 for r in regressions.values()):
            return (
                Recommendation.DEPLOY,
                "No significant regressions detected"
            )

        # Mixed results
        return (
            Recommendation.MANUAL_REVIEW,
            f"Mixed results: {len(improvements)} improvements, {len(regressions)} regressions"
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _calculate_changes(
        self,
        candidate_metrics: Dict[str, float],
        production_metrics: Dict[str, float],
        higher_is_better: Optional[List[str]] = None,
        lower_is_better: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate metric improvements and regressions.

        Returns:
            Tuple of (improvements, regressions) dictionaries
            Values are percentage changes (positive = better)
        """
        improvements = {}
        regressions = {}

        higher_is_better = higher_is_better or []
        lower_is_better = lower_is_better or []

        for metric in set(candidate_metrics.keys()) & set(production_metrics.keys()):
            cand_val = candidate_metrics[metric]
            prod_val = production_metrics[metric]

            # Skip if production value is 0 or very small
            if abs(prod_val) < 1e-10:
                if cand_val > 0 and metric in higher_is_better:
                    improvements[metric] = 1.0  # 100% improvement
                elif cand_val < 0 and metric in lower_is_better:
                    improvements[metric] = 1.0
                continue

            # Calculate relative change
            relative_change = (cand_val - prod_val) / abs(prod_val)

            # For "lower is better" metrics, flip the sign
            if metric in lower_is_better:
                relative_change = -relative_change

            # Categorize
            if relative_change >= 0:
                improvements[metric] = relative_change
            else:
                regressions[metric] = relative_change

        return improvements, regressions

    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return self._thresholds.copy()

    def update_thresholds(self, updates: Dict[str, float]):
        """Update thresholds."""
        self._thresholds.update(updates)
        logger.info(f"Updated A/B thresholds: {updates}")


# Global singleton
ab_comparison_service = ABComparisonService()
