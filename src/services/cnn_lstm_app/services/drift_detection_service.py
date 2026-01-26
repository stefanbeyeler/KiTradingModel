"""
Drift Detection Service for CNN-LSTM Multi-Task Model.

Monitors model performance and detects degradation through multiple drift types:
- Price Accuracy Drift: Price prediction accuracy drops
- Pattern Accuracy Drift: Pattern detection accuracy drops
- Regime Accuracy Drift: Regime prediction accuracy drops
- Confidence Drift: Confidence doesn't match outcomes
- Distribution Drift: Prediction distribution changes

Supports task-specific and overall drift detection with configurable thresholds.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import numpy as np
from loguru import logger


DATA_DIR = os.getenv("DATA_DIR", "/app/data")


class DriftSeverity(str, Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"          # < 10% drop, informational
    MEDIUM = "medium"    # 10-20% drop, alert
    HIGH = "high"        # 20-30% drop, incremental training recommended
    CRITICAL = "critical"  # > 30% drop, full retrain recommended


class DriftType(str, Enum):
    """Types of drift detected."""
    PRICE_ACCURACY = "price_accuracy"
    PATTERN_ACCURACY = "pattern_accuracy"
    REGIME_ACCURACY = "regime_accuracy"
    OVERALL_ACCURACY = "overall_accuracy"
    CONFIDENCE = "confidence"
    DISTRIBUTION = "distribution"


@dataclass
class DriftEvent:
    """A detected drift event."""
    drift_type: DriftType
    severity: DriftSeverity
    current_value: float
    baseline_value: float
    drop_percentage: float
    detected_at: str = ""
    task: Optional[str] = None
    message: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "current_value": round(self.current_value, 4),
            "baseline_value": round(self.baseline_value, 4),
            "drop_percentage": round(self.drop_percentage, 4),
            "detected_at": self.detected_at,
            "task": self.task,
            "message": self.message
        }


@dataclass
class DriftConfig:
    """Drift detection configuration."""
    # Threshold drops for each severity (as fractions)
    threshold_low: float = 0.05       # 5%
    threshold_medium: float = 0.10    # 10%
    threshold_high: float = 0.20      # 20%
    threshold_critical: float = 0.30  # 30%

    # Minimum samples for reliable detection
    min_samples_for_detection: int = 30

    # Baseline window (hours) for comparison
    baseline_window_hours: int = 168  # 7 days

    # Recent window (hours) for current performance
    recent_window_hours: int = 24  # 1 day

    # Confidence calibration threshold
    calibration_tolerance: float = 0.15  # 15% tolerance

    # Distribution change threshold
    distribution_change_threshold: float = 0.20  # 20% change

    # Task weights for overall severity
    price_weight: float = 0.40
    pattern_weight: float = 0.35
    regime_weight: float = 0.25

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DriftStatus:
    """Current drift status summary for multi-task model."""
    overall_severity: DriftSeverity = DriftSeverity.NONE

    # Task-specific drifts
    price_drift: Optional[DriftEvent] = None
    pattern_drift: Optional[DriftEvent] = None
    regime_drift: Optional[DriftEvent] = None

    # Overall drifts
    overall_accuracy_drift: Optional[DriftEvent] = None
    confidence_drift: Optional[DriftEvent] = None
    distribution_drift: Optional[DriftEvent] = None

    active_drifts: List[DriftEvent] = field(default_factory=list)
    last_check: Optional[str] = None
    recommendation: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "overall_severity": self.overall_severity.value,
            "price_drift": self.price_drift.to_dict() if self.price_drift else None,
            "pattern_drift": self.pattern_drift.to_dict() if self.pattern_drift else None,
            "regime_drift": self.regime_drift.to_dict() if self.regime_drift else None,
            "overall_accuracy_drift": self.overall_accuracy_drift.to_dict() if self.overall_accuracy_drift else None,
            "confidence_drift": self.confidence_drift.to_dict() if self.confidence_drift else None,
            "distribution_drift": self.distribution_drift.to_dict() if self.distribution_drift else None,
            "active_drifts_count": len(self.active_drifts),
            "last_check": self.last_check,
            "recommendation": self.recommendation
        }


class DriftDetectionService:
    """
    Service for detecting CNN-LSTM multi-task model performance drift.

    Monitors:
    - Price prediction accuracy (direction correct, magnitude error)
    - Pattern detection accuracy (predicted patterns vs actual)
    - Regime prediction accuracy (predicted regime vs actual)
    - Confidence calibration across all tasks
    - Prediction distribution stability
    """

    def __init__(self, history_file: str = None):
        if history_file is None:
            history_file = os.path.join(DATA_DIR, "cnn_lstm_drift_history.json")

        self._history_file = history_file
        self._max_observations = 1000

        self.config = DriftConfig()
        self._observations: deque = deque(maxlen=self._max_observations)
        self._drift_events: List[DriftEvent] = []
        self._baseline_metrics: Dict = {}
        self._current_status = DriftStatus()

        self._load_state()
        logger.info(f"DriftDetectionService initialized with {len(self._observations)} observations")

    def _load_state(self) -> None:
        """Load drift detection state from disk."""
        try:
            if os.path.exists(self._history_file):
                with open(self._history_file, 'r') as f:
                    data = json.load(f)
                    self._observations = deque(
                        data.get("observations", []),
                        maxlen=self._max_observations
                    )
                    self._baseline_metrics = data.get("baseline_metrics", {})
                    self._drift_events = [
                        DriftEvent(
                            drift_type=DriftType(e["drift_type"]),
                            severity=DriftSeverity(e["severity"]),
                            current_value=e["current_value"],
                            baseline_value=e["baseline_value"],
                            drop_percentage=e["drop_percentage"],
                            detected_at=e.get("detected_at", ""),
                            task=e.get("task"),
                            message=e.get("message", "")
                        )
                        for e in data.get("drift_events", [])
                    ]
        except Exception as e:
            logger.warning(f"Could not load drift state: {e}")

    def _save_state(self) -> None:
        """Save drift detection state to disk."""
        try:
            os.makedirs(os.path.dirname(self._history_file), exist_ok=True)
            with open(self._history_file, 'w') as f:
                json.dump({
                    "observations": list(self._observations),
                    "baseline_metrics": self._baseline_metrics,
                    "drift_events": [e.to_dict() for e in self._drift_events[-100:]]
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save drift state: {e}")

    def add_observation(
        self,
        prediction_id: str,
        symbol: str,
        timeframe: str,
        # Price task
        price_direction_correct: Optional[bool] = None,
        price_confidence: Optional[float] = None,
        price_magnitude_error: Optional[float] = None,
        # Pattern task
        pattern_correct: Optional[bool] = None,
        pattern_confidence: Optional[float] = None,
        patterns_predicted: Optional[List[str]] = None,
        # Regime task
        regime_correct: Optional[bool] = None,
        regime_confidence: Optional[float] = None,
        regime_predicted: Optional[str] = None,
        # Overall
        overall_accuracy: float = 0.0,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Add an observation for drift detection.

        Args:
            prediction_id: Unique prediction identifier
            symbol: Trading symbol
            timeframe: Timeframe
            price_direction_correct: Whether price direction was correct
            price_confidence: Confidence of price prediction
            price_magnitude_error: MAE of price prediction
            pattern_correct: Whether patterns were correctly detected
            pattern_confidence: Confidence of pattern prediction
            patterns_predicted: List of predicted patterns
            regime_correct: Whether regime was correctly predicted
            regime_confidence: Confidence of regime prediction
            regime_predicted: Predicted regime name
            overall_accuracy: Overall prediction accuracy (0-1)
            timestamp: Observation timestamp
        """
        observation = {
            "prediction_id": prediction_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "price_direction_correct": price_direction_correct,
            "price_confidence": price_confidence,
            "price_magnitude_error": price_magnitude_error,
            "pattern_correct": pattern_correct,
            "pattern_confidence": pattern_confidence,
            "patterns_predicted": patterns_predicted or [],
            "regime_correct": regime_correct,
            "regime_confidence": regime_confidence,
            "regime_predicted": regime_predicted,
            "overall_accuracy": overall_accuracy,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat()
        }
        self._observations.append(observation)

    def check_drift(self) -> DriftStatus:
        """
        Check for drift in model performance across all tasks.

        Returns:
            DriftStatus with current drift information
        """
        now = datetime.now(timezone.utc)
        status = DriftStatus(last_check=now.isoformat())

        if len(self._observations) < self.config.min_samples_for_detection:
            status.recommendation = (
                f"Insufficient data ({len(self._observations)}/{self.config.min_samples_for_detection} samples)"
            )
            self._current_status = status
            return status

        # Split observations into baseline and recent
        baseline_cutoff = now - timedelta(hours=self.config.baseline_window_hours)
        recent_cutoff = now - timedelta(hours=self.config.recent_window_hours)

        baseline_obs = [
            o for o in self._observations
            if self._parse_timestamp(o["timestamp"]) <= baseline_cutoff
        ]
        recent_obs = [
            o for o in self._observations
            if self._parse_timestamp(o["timestamp"]) >= recent_cutoff
        ]

        # Ensure we have enough baseline data
        if len(baseline_obs) < self.config.min_samples_for_detection // 2:
            baseline_obs = list(self._observations)[:-len(recent_obs)] if recent_obs else list(self._observations)

        if len(recent_obs) < self.config.min_samples_for_detection // 2:
            status.recommendation = f"Insufficient recent data ({len(recent_obs)} samples)"
            self._current_status = status
            return status

        # Check task-specific drifts
        status.price_drift = self._check_task_drift(
            baseline_obs, recent_obs, "price", "price_direction_correct"
        )
        if status.price_drift:
            status.active_drifts.append(status.price_drift)

        status.pattern_drift = self._check_task_drift(
            baseline_obs, recent_obs, "pattern", "pattern_correct"
        )
        if status.pattern_drift:
            status.active_drifts.append(status.pattern_drift)

        status.regime_drift = self._check_task_drift(
            baseline_obs, recent_obs, "regime", "regime_correct"
        )
        if status.regime_drift:
            status.active_drifts.append(status.regime_drift)

        # Check overall accuracy drift
        status.overall_accuracy_drift = self._check_overall_accuracy_drift(
            baseline_obs, recent_obs
        )
        if status.overall_accuracy_drift:
            status.active_drifts.append(status.overall_accuracy_drift)

        # Check confidence drift
        status.confidence_drift = self._check_confidence_drift(recent_obs)
        if status.confidence_drift:
            status.active_drifts.append(status.confidence_drift)

        # Check distribution drift
        status.distribution_drift = self._check_distribution_drift(
            baseline_obs, recent_obs
        )
        if status.distribution_drift:
            status.active_drifts.append(status.distribution_drift)

        # Determine overall severity based on weighted task severities
        status.overall_severity = self._compute_overall_severity(status)
        status.recommendation = self._get_recommendation(status.overall_severity)

        # Record significant drift events
        for drift in status.active_drifts:
            if drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                self._drift_events.append(drift)

        self._current_status = status
        self._save_state()

        return status

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse ISO timestamp to datetime."""
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            return datetime.now(timezone.utc)

    def _check_task_drift(
        self,
        baseline_obs: List[Dict],
        recent_obs: List[Dict],
        task_name: str,
        correct_field: str
    ) -> Optional[DriftEvent]:
        """Check for accuracy drift in a specific task."""
        # Filter observations that have this task evaluated
        baseline = [o for o in baseline_obs if o.get(correct_field) is not None]
        recent = [o for o in recent_obs if o.get(correct_field) is not None]

        if not baseline or not recent:
            return None

        baseline_accuracy = sum(1 for o in baseline if o[correct_field]) / len(baseline)
        recent_accuracy = sum(1 for o in recent if o[correct_field]) / len(recent)

        if baseline_accuracy == 0:
            return None

        drop = (baseline_accuracy - recent_accuracy) / baseline_accuracy

        if drop <= 0:
            return None  # No drop or improvement

        severity = self._get_severity(drop)

        if severity == DriftSeverity.NONE:
            return None

        drift_type = {
            "price": DriftType.PRICE_ACCURACY,
            "pattern": DriftType.PATTERN_ACCURACY,
            "regime": DriftType.REGIME_ACCURACY
        }.get(task_name, DriftType.OVERALL_ACCURACY)

        return DriftEvent(
            drift_type=drift_type,
            severity=severity,
            current_value=recent_accuracy,
            baseline_value=baseline_accuracy,
            drop_percentage=drop * 100,
            task=task_name,
            message=f"{task_name.title()} accuracy dropped from {baseline_accuracy:.2%} to {recent_accuracy:.2%}"
        )

    def _check_overall_accuracy_drift(
        self,
        baseline_obs: List[Dict],
        recent_obs: List[Dict]
    ) -> Optional[DriftEvent]:
        """Check for overall accuracy drift."""
        baseline_acc = [o.get("overall_accuracy", 0) for o in baseline_obs if o.get("overall_accuracy") is not None]
        recent_acc = [o.get("overall_accuracy", 0) for o in recent_obs if o.get("overall_accuracy") is not None]

        if not baseline_acc or not recent_acc:
            return None

        baseline_mean = np.mean(baseline_acc)
        recent_mean = np.mean(recent_acc)

        if baseline_mean == 0:
            return None

        drop = (baseline_mean - recent_mean) / baseline_mean

        if drop <= 0:
            return None

        severity = self._get_severity(drop)

        if severity == DriftSeverity.NONE:
            return None

        return DriftEvent(
            drift_type=DriftType.OVERALL_ACCURACY,
            severity=severity,
            current_value=recent_mean,
            baseline_value=baseline_mean,
            drop_percentage=drop * 100,
            message=f"Overall accuracy dropped from {baseline_mean:.2%} to {recent_mean:.2%}"
        )

    def _check_confidence_drift(self, recent_obs: List[Dict]) -> Optional[DriftEvent]:
        """Check for confidence calibration drift across all tasks."""
        calibration_errors = []

        # Check each task's confidence calibration
        for task, conf_field, correct_field in [
            ("price", "price_confidence", "price_direction_correct"),
            ("pattern", "pattern_confidence", "pattern_correct"),
            ("regime", "regime_confidence", "regime_correct")
        ]:
            task_obs = [
                o for o in recent_obs
                if o.get(conf_field) is not None and o.get(correct_field) is not None
            ]

            if len(task_obs) >= 10:
                # Group by confidence bins
                bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]

                for low, high in bins:
                    bin_obs = [o for o in task_obs if low <= o[conf_field] < high]
                    if len(bin_obs) >= 3:
                        expected = (low + high) / 2
                        actual = sum(1 for o in bin_obs if o[correct_field]) / len(bin_obs)
                        calibration_errors.append(abs(expected - actual))

        if not calibration_errors:
            return None

        avg_error = np.mean(calibration_errors)

        if avg_error <= self.config.calibration_tolerance:
            return None

        drop = avg_error / self.config.calibration_tolerance - 1
        severity = self._get_severity(drop)

        if severity == DriftSeverity.NONE:
            return None

        return DriftEvent(
            drift_type=DriftType.CONFIDENCE,
            severity=severity,
            current_value=1 - avg_error,
            baseline_value=1 - self.config.calibration_tolerance,
            drop_percentage=drop * 100,
            message=f"Confidence calibration error: {avg_error:.2%}"
        )

    def _check_distribution_drift(
        self,
        baseline_obs: List[Dict],
        recent_obs: List[Dict]
    ) -> Optional[DriftEvent]:
        """Check for prediction distribution drift (regime predictions)."""
        baseline_regimes = [o.get("regime_predicted") for o in baseline_obs if o.get("regime_predicted")]
        recent_regimes = [o.get("regime_predicted") for o in recent_obs if o.get("regime_predicted")]

        if not baseline_regimes or not recent_regimes:
            return None

        all_regimes = set(baseline_regimes) | set(recent_regimes)

        if not all_regimes:
            return None

        # Calculate distribution difference
        baseline_dist = {r: baseline_regimes.count(r) / len(baseline_regimes) for r in all_regimes}
        recent_dist = {r: recent_regimes.count(r) / len(recent_regimes) for r in all_regimes}

        total_diff = sum(
            abs(baseline_dist.get(r, 0) - recent_dist.get(r, 0))
            for r in all_regimes
        ) / 2

        if total_diff <= self.config.distribution_change_threshold:
            return None

        severity = self._get_severity(total_diff)

        if severity == DriftSeverity.NONE:
            return None

        return DriftEvent(
            drift_type=DriftType.DISTRIBUTION,
            severity=severity,
            current_value=1 - total_diff,
            baseline_value=1.0,
            drop_percentage=total_diff * 100,
            message=f"Regime prediction distribution changed by {total_diff:.2%}"
        )

    def _get_severity(self, drop: float) -> DriftSeverity:
        """Get severity level for a given drop percentage."""
        if drop >= self.config.threshold_critical:
            return DriftSeverity.CRITICAL
        elif drop >= self.config.threshold_high:
            return DriftSeverity.HIGH
        elif drop >= self.config.threshold_medium:
            return DriftSeverity.MEDIUM
        elif drop >= self.config.threshold_low:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def _compute_overall_severity(self, status: DriftStatus) -> DriftSeverity:
        """Compute overall severity from task-specific drifts."""
        severities = [d.severity for d in status.active_drifts]

        if DriftSeverity.CRITICAL in severities:
            return DriftSeverity.CRITICAL
        elif DriftSeverity.HIGH in severities:
            return DriftSeverity.HIGH
        elif DriftSeverity.MEDIUM in severities:
            return DriftSeverity.MEDIUM
        elif DriftSeverity.LOW in severities:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def _get_recommendation(self, severity: DriftSeverity) -> str:
        """Get recommendation based on severity."""
        recommendations = {
            DriftSeverity.CRITICAL: "Full retraining recommended immediately",
            DriftSeverity.HIGH: "Incremental training recommended",
            DriftSeverity.MEDIUM: "Monitor closely, consider training",
            DriftSeverity.LOW: "Minor drift detected, continue monitoring",
            DriftSeverity.NONE: "No significant drift detected"
        }
        return recommendations.get(severity, "Unknown severity")

    def get_status(self) -> DriftStatus:
        """Get current drift status."""
        return self._current_status

    def get_drift_history(self, limit: int = 20) -> List[Dict]:
        """Get recent drift events."""
        return [e.to_dict() for e in self._drift_events[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get drift detection statistics."""
        if not self._observations:
            return {
                "observations_count": 0,
                "drift_events_count": 0,
                "recent_drift_events": 0,
                "current_severity": DriftSeverity.NONE.value,
                "last_check": None,
                "recommendation": "No data collected yet",
                "task_accuracies": {},
                "overall_accuracy": 0.0,
                "severity_distribution": {s.value: 0 for s in DriftSeverity}
            }

        # Calculate task-specific accuracies
        task_accuracies = {}
        for task, correct_field in [
            ("price", "price_direction_correct"),
            ("pattern", "pattern_correct"),
            ("regime", "regime_correct")
        ]:
            task_obs = [o for o in self._observations if o.get(correct_field) is not None]
            if task_obs:
                task_accuracies[task] = sum(1 for o in task_obs if o[correct_field]) / len(task_obs)

        # Calculate overall accuracy
        overall_acc = [o.get("overall_accuracy", 0) for o in self._observations if o.get("overall_accuracy") is not None]
        avg_overall = np.mean(overall_acc) if overall_acc else 0.0

        recent_events = [
            e for e in self._drift_events
            if self._parse_timestamp(e.detected_at) >= datetime.now(timezone.utc) - timedelta(days=7)
        ]

        return {
            "observations_count": len(self._observations),
            "drift_events_count": len(self._drift_events),
            "recent_drift_events": len(recent_events),
            "current_severity": self._current_status.overall_severity.value,
            "last_check": self._current_status.last_check,
            "recommendation": self._current_status.recommendation,
            "task_accuracies": {k: round(v, 3) for k, v in task_accuracies.items()},
            "overall_accuracy": round(avg_overall, 3),
            "severity_distribution": {
                severity.value: sum(1 for e in self._drift_events if e.severity == severity)
                for severity in DriftSeverity
            }
        }

    def reset_baseline(self) -> None:
        """Reset baseline metrics from current observations."""
        if self._observations:
            # Calculate current accuracies as baseline
            task_baselines = {}
            for task, correct_field in [
                ("price", "price_direction_correct"),
                ("pattern", "pattern_correct"),
                ("regime", "regime_correct")
            ]:
                task_obs = [o for o in self._observations if o.get(correct_field) is not None]
                if task_obs:
                    task_baselines[task] = sum(1 for o in task_obs if o[correct_field]) / len(task_obs)

            overall_acc = [o.get("overall_accuracy", 0) for o in self._observations if o.get("overall_accuracy") is not None]

            self._baseline_metrics = {
                "task_accuracies": task_baselines,
                "overall_accuracy": np.mean(overall_acc) if overall_acc else 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "samples": len(self._observations)
            }
            self._save_state()
            logger.info("Baseline metrics reset")

    def update_config(
        self,
        threshold_medium: Optional[float] = None,
        threshold_high: Optional[float] = None,
        threshold_critical: Optional[float] = None,
        min_samples: Optional[int] = None,
        price_weight: Optional[float] = None,
        pattern_weight: Optional[float] = None,
        regime_weight: Optional[float] = None
    ) -> DriftConfig:
        """Update drift detection configuration."""
        if threshold_medium is not None:
            self.config.threshold_medium = threshold_medium
        if threshold_high is not None:
            self.config.threshold_high = threshold_high
        if threshold_critical is not None:
            self.config.threshold_critical = threshold_critical
        if min_samples is not None:
            self.config.min_samples_for_detection = min_samples
        if price_weight is not None:
            self.config.price_weight = price_weight
        if pattern_weight is not None:
            self.config.pattern_weight = pattern_weight
        if regime_weight is not None:
            self.config.regime_weight = regime_weight
        return self.config

    def clear_observations(self) -> int:
        """Clear all observations. Returns count cleared."""
        count = len(self._observations)
        self._observations.clear()
        self._save_state()
        return count


# Singleton instance
drift_detection_service = DriftDetectionService()
