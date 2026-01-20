"""
Drift Detection Service for TCN Model.

Monitors model performance and detects degradation through multiple drift types:
- Accuracy Drift: Outcome accuracy drops
- Confidence Drift: Confidence doesn't match outcomes
- Distribution Drift: Pattern distribution changes
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import numpy as np
from loguru import logger


class DriftSeverity(str, Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"  # < 10% drop, informational
    MEDIUM = "medium"  # 10-20% drop, alert
    HIGH = "high"  # 20-30% drop, incremental training recommended
    CRITICAL = "critical"  # > 30% drop, full retrain recommended


class DriftType(str, Enum):
    """Types of drift detected."""
    ACCURACY = "accuracy"  # Outcome accuracy drift
    CONFIDENCE = "confidence"  # Confidence calibration drift
    DISTRIBUTION = "distribution"  # Pattern distribution drift


@dataclass
class DriftEvent:
    """A detected drift event."""
    drift_type: DriftType
    severity: DriftSeverity
    current_value: float
    baseline_value: float
    drop_percentage: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    pattern_type: Optional[str] = None  # For per-pattern drift
    message: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "current_value": round(self.current_value, 4),
            "baseline_value": round(self.baseline_value, 4),
            "drop_percentage": round(self.drop_percentage, 4),
            "detected_at": self.detected_at.isoformat(),
            "pattern_type": self.pattern_type,
            "message": self.message
        }


@dataclass
class DriftConfig:
    """Drift detection configuration."""
    # Threshold drops for each severity (as fractions)
    threshold_low: float = 0.05  # 5%
    threshold_medium: float = 0.10  # 10%
    threshold_high: float = 0.20  # 20%
    threshold_critical: float = 0.30  # 30%

    # Minimum samples for reliable detection
    min_samples_for_detection: int = 30

    # Baseline window (hours) for comparison
    baseline_window_hours: int = 168  # 7 days

    # Recent window (hours) for current performance
    recent_window_hours: int = 24  # 1 day

    # Confidence calibration threshold
    calibration_tolerance: float = 0.15  # 15% tolerance

    # Pattern distribution change threshold
    distribution_change_threshold: float = 0.20  # 20% change in distribution


@dataclass
class DriftStatus:
    """Current drift status summary."""
    overall_severity: DriftSeverity = DriftSeverity.NONE
    accuracy_drift: Optional[DriftEvent] = None
    confidence_drift: Optional[DriftEvent] = None
    distribution_drift: Optional[DriftEvent] = None
    active_drifts: List[DriftEvent] = field(default_factory=list)
    last_check: Optional[datetime] = None
    recommendation: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "overall_severity": self.overall_severity.value,
            "accuracy_drift": self.accuracy_drift.to_dict() if self.accuracy_drift else None,
            "confidence_drift": self.confidence_drift.to_dict() if self.confidence_drift else None,
            "distribution_drift": self.distribution_drift.to_dict() if self.distribution_drift else None,
            "active_drifts_count": len(self.active_drifts),
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "recommendation": self.recommendation
        }


class DriftDetectionService:
    """
    Service for detecting model performance drift.

    Monitors outcome accuracy, confidence calibration, and pattern
    distribution to detect when the model needs retraining.
    """

    HISTORY_FILE = "data/tcn_drift_history.json"
    MAX_OBSERVATIONS = 1000  # Max stored observations

    def __init__(self):
        """Initialize drift detection service."""
        self.config = DriftConfig()
        self._observations: deque = deque(maxlen=self.MAX_OBSERVATIONS)
        self._drift_events: List[DriftEvent] = []
        self._baseline_metrics: Dict = {}
        self._current_status = DriftStatus()
        self._load_state()

    def _load_state(self) -> None:
        """Load drift detection state from disk."""
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    self._observations = deque(data.get("observations", []), maxlen=self.MAX_OBSERVATIONS)
                    self._baseline_metrics = data.get("baseline_metrics", {})
                    self._drift_events = [
                        DriftEvent(
                            drift_type=DriftType(e["drift_type"]),
                            severity=DriftSeverity(e["severity"]),
                            current_value=e["current_value"],
                            baseline_value=e["baseline_value"],
                            drop_percentage=e["drop_percentage"],
                            detected_at=datetime.fromisoformat(e["detected_at"]),
                            pattern_type=e.get("pattern_type"),
                            message=e.get("message", "")
                        )
                        for e in data.get("drift_events", [])
                    ]
                    logger.info(f"Loaded {len(self._observations)} drift observations")
        except Exception as e:
            logger.warning(f"Could not load drift state: {e}")

    def _save_state(self) -> None:
        """Save drift detection state to disk."""
        try:
            os.makedirs(os.path.dirname(self.HISTORY_FILE), exist_ok=True)
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump({
                    "observations": list(self._observations),
                    "baseline_metrics": self._baseline_metrics,
                    "drift_events": [e.to_dict() for e in self._drift_events[-100:]]  # Keep last 100
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save drift state: {e}")

    def add_observation(
        self,
        pattern_type: str,
        confidence: float,
        is_success: bool,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add an observation for drift detection.

        Args:
            pattern_type: Type of pattern detected
            confidence: Model's confidence score
            is_success: Whether the pattern outcome was successful
            timestamp: Observation timestamp
        """
        observation = {
            "pattern_type": pattern_type,
            "confidence": confidence,
            "is_success": is_success,
            "timestamp": (timestamp or datetime.utcnow()).isoformat()
        }
        self._observations.append(observation)

    def check_drift(self) -> DriftStatus:
        """
        Check for drift in model performance.

        Returns:
            DriftStatus with current drift information
        """
        now = datetime.utcnow()
        status = DriftStatus(last_check=now)

        if len(self._observations) < self.config.min_samples_for_detection:
            status.recommendation = f"Insufficient data ({len(self._observations)}/{self.config.min_samples_for_detection} samples)"
            self._current_status = status
            return status

        # Split observations into baseline and recent
        baseline_cutoff = now - timedelta(hours=self.config.baseline_window_hours)
        recent_cutoff = now - timedelta(hours=self.config.recent_window_hours)

        baseline_obs = [
            o for o in self._observations
            if datetime.fromisoformat(o["timestamp"]) <= baseline_cutoff
        ]
        recent_obs = [
            o for o in self._observations
            if datetime.fromisoformat(o["timestamp"]) >= recent_cutoff
        ]

        if len(baseline_obs) < self.config.min_samples_for_detection / 2:
            # Not enough baseline data, use all data as baseline
            baseline_obs = list(self._observations)[:-len(recent_obs)] if recent_obs else list(self._observations)

        if len(recent_obs) < self.config.min_samples_for_detection / 2:
            status.recommendation = f"Insufficient recent data ({len(recent_obs)} samples)"
            self._current_status = status
            return status

        # Check accuracy drift
        accuracy_drift = self._check_accuracy_drift(baseline_obs, recent_obs)
        if accuracy_drift:
            status.accuracy_drift = accuracy_drift
            status.active_drifts.append(accuracy_drift)

        # Check confidence drift
        confidence_drift = self._check_confidence_drift(recent_obs)
        if confidence_drift:
            status.confidence_drift = confidence_drift
            status.active_drifts.append(confidence_drift)

        # Check distribution drift
        distribution_drift = self._check_distribution_drift(baseline_obs, recent_obs)
        if distribution_drift:
            status.distribution_drift = distribution_drift
            status.active_drifts.append(distribution_drift)

        # Determine overall severity
        if status.active_drifts:
            severities = [d.severity for d in status.active_drifts]
            if DriftSeverity.CRITICAL in severities:
                status.overall_severity = DriftSeverity.CRITICAL
                status.recommendation = "Full retraining recommended"
            elif DriftSeverity.HIGH in severities:
                status.overall_severity = DriftSeverity.HIGH
                status.recommendation = "Incremental training recommended"
            elif DriftSeverity.MEDIUM in severities:
                status.overall_severity = DriftSeverity.MEDIUM
                status.recommendation = "Monitor closely, consider training"
            else:
                status.overall_severity = DriftSeverity.LOW
                status.recommendation = "Minor drift detected, monitoring"

            # Record drift events
            for drift in status.active_drifts:
                if drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    self._drift_events.append(drift)
        else:
            status.recommendation = "No significant drift detected"

        self._current_status = status
        self._save_state()

        return status

    def _check_accuracy_drift(
        self,
        baseline_obs: List[Dict],
        recent_obs: List[Dict]
    ) -> Optional[DriftEvent]:
        """Check for accuracy drift."""
        if not baseline_obs or not recent_obs:
            return None

        baseline_accuracy = sum(1 for o in baseline_obs if o["is_success"]) / len(baseline_obs)
        recent_accuracy = sum(1 for o in recent_obs if o["is_success"]) / len(recent_obs)

        if baseline_accuracy == 0:
            return None

        drop = (baseline_accuracy - recent_accuracy) / baseline_accuracy

        if drop <= 0:
            return None  # No drop, possibly improvement

        severity = self._get_severity(drop)

        if severity == DriftSeverity.NONE:
            return None

        return DriftEvent(
            drift_type=DriftType.ACCURACY,
            severity=severity,
            current_value=recent_accuracy,
            baseline_value=baseline_accuracy,
            drop_percentage=drop * 100,
            message=f"Accuracy dropped from {baseline_accuracy:.2%} to {recent_accuracy:.2%}"
        )

    def _check_confidence_drift(self, recent_obs: List[Dict]) -> Optional[DriftEvent]:
        """Check for confidence calibration drift."""
        if not recent_obs:
            return None

        # Group by confidence bins
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        calibration_errors = []

        for low, high in bins:
            bin_obs = [o for o in recent_obs if low <= o["confidence"] < high]
            if len(bin_obs) >= 5:  # Need at least 5 samples per bin
                expected_accuracy = (low + high) / 2  # Mid-point as expected
                actual_accuracy = sum(1 for o in bin_obs if o["is_success"]) / len(bin_obs)
                error = abs(expected_accuracy - actual_accuracy)
                calibration_errors.append(error)

        if not calibration_errors:
            return None

        avg_calibration_error = np.mean(calibration_errors)

        if avg_calibration_error <= self.config.calibration_tolerance:
            return None

        # Determine severity based on calibration error
        drop = avg_calibration_error / self.config.calibration_tolerance - 1
        severity = self._get_severity(drop)

        if severity == DriftSeverity.NONE:
            return None

        return DriftEvent(
            drift_type=DriftType.CONFIDENCE,
            severity=severity,
            current_value=1 - avg_calibration_error,  # Calibration score
            baseline_value=1 - self.config.calibration_tolerance,
            drop_percentage=drop * 100,
            message=f"Confidence calibration error: {avg_calibration_error:.2%}"
        )

    def _check_distribution_drift(
        self,
        baseline_obs: List[Dict],
        recent_obs: List[Dict]
    ) -> Optional[DriftEvent]:
        """Check for pattern distribution drift."""
        if not baseline_obs or not recent_obs:
            return None

        # Get pattern distributions
        baseline_patterns = [o["pattern_type"] for o in baseline_obs]
        recent_patterns = [o["pattern_type"] for o in recent_obs]

        all_patterns = set(baseline_patterns) | set(recent_patterns)

        if not all_patterns:
            return None

        # Calculate distribution difference (Jensen-Shannon divergence approximation)
        baseline_dist = {p: baseline_patterns.count(p) / len(baseline_patterns) for p in all_patterns}
        recent_dist = {p: recent_patterns.count(p) / len(recent_patterns) for p in all_patterns}

        total_diff = sum(abs(baseline_dist.get(p, 0) - recent_dist.get(p, 0)) for p in all_patterns) / 2

        if total_diff <= self.config.distribution_change_threshold:
            return None

        severity = self._get_severity(total_diff)

        if severity == DriftSeverity.NONE:
            return None

        return DriftEvent(
            drift_type=DriftType.DISTRIBUTION,
            severity=severity,
            current_value=1 - total_diff,  # Distribution similarity
            baseline_value=1.0,
            drop_percentage=total_diff * 100,
            message=f"Pattern distribution changed by {total_diff:.2%}"
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

    def get_status(self) -> DriftStatus:
        """Get current drift status."""
        return self._current_status

    def get_drift_history(self, limit: int = 20) -> List[Dict]:
        """Get recent drift events."""
        return [e.to_dict() for e in self._drift_events[-limit:]]

    def get_statistics(self) -> Dict:
        """Get drift detection statistics."""
        if not self._observations:
            return {
                "observations_count": 0,
                "drift_events_count": 0,
                "current_severity": DriftSeverity.NONE.value
            }

        recent_events = [
            e for e in self._drift_events
            if e.detected_at >= datetime.utcnow() - timedelta(days=7)
        ]

        return {
            "observations_count": len(self._observations),
            "drift_events_count": len(self._drift_events),
            "recent_drift_events": len(recent_events),
            "current_severity": self._current_status.overall_severity.value,
            "last_check": self._current_status.last_check.isoformat() if self._current_status.last_check else None,
            "recommendation": self._current_status.recommendation,
            "accuracy_baseline": self._calculate_overall_accuracy(),
            "severity_distribution": {
                severity.value: sum(1 for e in self._drift_events if e.severity == severity)
                for severity in DriftSeverity
            }
        }

    def _calculate_overall_accuracy(self) -> float:
        """Calculate overall accuracy from observations."""
        if not self._observations:
            return 0.0
        return sum(1 for o in self._observations if o["is_success"]) / len(self._observations)

    def reset_baseline(self) -> None:
        """Reset baseline metrics from current observations."""
        if self._observations:
            self._baseline_metrics = {
                "accuracy": self._calculate_overall_accuracy(),
                "timestamp": datetime.utcnow().isoformat(),
                "samples": len(self._observations)
            }
            self._save_state()
            logger.info("Baseline metrics reset")

    def update_config(
        self,
        threshold_medium: Optional[float] = None,
        threshold_high: Optional[float] = None,
        threshold_critical: Optional[float] = None,
        min_samples: Optional[int] = None
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
        return self.config


# Singleton instance
drift_detection_service = DriftDetectionService()
