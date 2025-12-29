"""
Feedback Analyzer Service for Candlestick Pattern Improvement.

This service analyzes Claude validation feedback to:
1. Extract structured feedback categories from Claude's reasoning
2. Aggregate feedback statistics by pattern type and reason
3. Generate parameter adjustment recommendations
4. Track improvement over time
"""

import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger


# Feedback category patterns - regex patterns to extract from Claude's reasoning
FEEDBACK_CATEGORY_PATTERNS = {
    # Body-related issues
    "body_too_large": [
        r"body.{0,20}(too|is|appears?).{0,10}(large|big|thick)",
        r"body.{0,20}(should be|needs to be).{0,10}(smaller|thinner)",
        r"(large|big|thick).{0,10}body",
        r"body.{0,10}(exceed|over|more than).{0,10}\d+%",
    ],
    "body_too_small": [
        r"body.{0,20}(too|is|appears?).{0,10}(small|thin|tiny)",
        r"body.{0,20}(should be|needs to be).{0,10}(larger|bigger)",
        r"(small|tiny|minimal).{0,10}body",
        r"not enough.{0,10}body",
    ],

    # Shadow-related issues
    "upper_shadow_too_short": [
        r"upper.{0,10}shadow.{0,20}(too|is).{0,10}(short|small|minimal)",
        r"(insufficient|lacking|missing).{0,10}upper.{0,10}shadow",
        r"upper.{0,10}wick.{0,20}(too|is).{0,10}(short|small)",
    ],
    "upper_shadow_too_long": [
        r"upper.{0,10}shadow.{0,20}(too|is).{0,10}(long|large|big)",
        r"upper.{0,10}wick.{0,20}(too|is).{0,10}(long|large)",
    ],
    "lower_shadow_too_short": [
        r"lower.{0,10}shadow.{0,20}(too|is).{0,10}(short|small|minimal)",
        r"(insufficient|lacking|missing).{0,10}lower.{0,10}shadow",
        r"lower.{0,10}wick.{0,20}(too|is).{0,10}(short|small)",
    ],
    "lower_shadow_too_long": [
        r"lower.{0,10}shadow.{0,20}(too|is).{0,10}(long|large|big)",
        r"lower.{0,10}wick.{0,20}(too|is).{0,10}(long|large)",
    ],

    # Engulfing-related issues
    "not_fully_engulfing": [
        r"not.{0,20}(fully|completely).{0,10}engulf",
        r"(does not|doesn't).{0,10}engulf",
        r"fail.{0,20}engulf",
        r"engulfing.{0,20}not.{0,10}(complete|full)",
    ],
    "engulfing_too_small": [
        r"engulfing.{0,20}candle.{0,20}(too|is).{0,10}(small|weak)",
        r"(second|current).{0,10}candle.{0,20}not.{0,10}(large|big) enough",
        r"size.{0,10}ratio.{0,10}(insufficient|too small)",
    ],

    # Trend context issues
    "wrong_trend_context": [
        r"(wrong|incorrect|inappropriate).{0,10}(trend|context)",
        r"no.{0,10}(prior|preceding).{0,10}(uptrend|downtrend)",
        r"(reversal|continuation).{0,10}pattern.{0,20}(without|missing).{0,10}trend",
        r"context.{0,20}not.{0,10}appropriate",
    ],
    "no_prior_trend": [
        r"no.{0,10}(clear|visible|prior).{0,10}trend",
        r"(lacking|missing|absent).{0,10}(uptrend|downtrend)",
        r"sideways.{0,10}(market|movement)",
    ],

    # Gap-related issues
    "missing_gap": [
        r"(missing|no|absent).{0,10}gap",
        r"gap.{0,10}(required|needed|expected)",
        r"(should|needs to).{0,10}gap",
    ],

    # General issues
    "false_positive": [
        r"(false|incorrect).{0,10}positive",
        r"not.{0,20}(valid|genuine|true).{0,10}(pattern|formation)",
        r"(misidentified|incorrectly identified)",
        r"(does not|doesn't).{0,10}meet.{0,10}(criteria|definition)",
    ],
    "wrong_pattern_type": [
        r"(looks more like|appears to be|actually|should be).{0,20}(doji|hammer|engulfing|star|harami)",
        r"(this is|i see).{0,10}(a|an).{0,10}(doji|hammer|engulfing|star|harami)",
        r"misclassified.{0,10}as",
    ],

    # Proportion issues
    "shadow_body_ratio_wrong": [
        r"shadow.{0,10}(to|vs|compared).{0,10}body.{0,10}ratio",
        r"ratio.{0,20}(shadow|wick).{0,20}body",
        r"(shadow|wick).{0,20}(at least|minimum).{0,10}\d+x",
    ],
}


@dataclass
class FeedbackCategory:
    """A categorized piece of feedback from Claude."""
    category: str
    pattern_type: str
    symbol: str
    timeframe: str
    reasoning: str
    confidence: float
    timestamp: str
    validation_id: str


@dataclass
class FeedbackRecommendation:
    """A recommendation based on aggregated feedback."""
    pattern: str
    parameter: str
    current_value: float
    recommended_value: float
    reason: str
    feedback_count: int
    confidence: float
    priority: str  # "high", "medium", "low"
    impact_estimate: str  # Description of expected impact

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class FeedbackAnalyzerService:
    """
    Analyzes Claude validation feedback to improve pattern detection.

    This service:
    1. Extracts structured categories from Claude's reasoning text
    2. Aggregates statistics by pattern type and feedback category
    3. Generates parameter adjustment recommendations
    4. Tracks improvements after adjustments
    """

    def __init__(self):
        self._feedback_categories: List[FeedbackCategory] = []
        self._adjustment_history: List[Dict] = []

        # Compile regex patterns for efficiency
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for category, patterns in FEEDBACK_CATEGORY_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        logger.info("Feedback Analyzer Service initialized")

    def extract_feedback_categories(
        self,
        reasoning: str,
        pattern_type: str,
        symbol: str = "",
        timeframe: str = "",
        validation_id: str = "",
        confidence: float = 0.0
    ) -> List[FeedbackCategory]:
        """
        Extract structured feedback categories from Claude's reasoning text.

        Args:
            reasoning: Claude's reasoning text
            pattern_type: The claimed pattern type
            symbol: Trading symbol
            timeframe: Chart timeframe
            validation_id: ID of the validation
            confidence: Claude's confidence in the rejection

        Returns:
            List of FeedbackCategory objects
        """
        categories = []
        reasoning_lower = reasoning.lower()

        for category, compiled_patterns in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(reasoning_lower):
                    fb = FeedbackCategory(
                        category=category,
                        pattern_type=pattern_type.lower(),
                        symbol=symbol,
                        timeframe=timeframe,
                        reasoning=reasoning[:500],  # Truncate
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        validation_id=validation_id
                    )
                    categories.append(fb)
                    break  # Only one match per category

        return categories

    def analyze_validation_history(
        self,
        validation_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze validation history to extract feedback statistics.

        Args:
            validation_history: List of Claude validation results

        Returns:
            Aggregated feedback statistics
        """
        # Filter rejections
        rejections = [
            v for v in validation_history
            if not v.get("claude_agrees", True)
        ]

        # Statistics
        by_pattern_and_reason: Dict[str, int] = {}
        by_pattern: Dict[str, Dict[str, int]] = {}
        by_reason: Dict[str, int] = {}
        recent_rejections: List[Dict] = []

        for rejection in rejections:
            pattern_type = rejection.get("pattern_type", "unknown")
            reasoning = rejection.get("claude_reasoning", "")
            validation_id = rejection.get("pattern_id", "")
            symbol = rejection.get("symbol", "")
            timeframe = rejection.get("timeframe", "")
            confidence = rejection.get("claude_confidence", 0.5)

            # Extract categories
            categories = self.extract_feedback_categories(
                reasoning=reasoning,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                validation_id=validation_id,
                confidence=confidence
            )

            # Store for tracking
            self._feedback_categories.extend(categories)

            # Aggregate
            for cat in categories:
                key = f"{pattern_type}:{cat.category}"
                by_pattern_and_reason[key] = by_pattern_and_reason.get(key, 0) + 1

                if pattern_type not in by_pattern:
                    by_pattern[pattern_type] = {}
                by_pattern[pattern_type][cat.category] = by_pattern[pattern_type].get(cat.category, 0) + 1

                by_reason[cat.category] = by_reason.get(cat.category, 0) + 1

            # Track recent
            recent_rejections.append({
                "pattern_type": pattern_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "reasoning": reasoning[:200],
                "categories": [c.category for c in categories],
                "timestamp": rejection.get("validation_timestamp", "")
            })

        return {
            "total_rejections": len(rejections),
            "by_pattern_and_reason": by_pattern_and_reason,
            "by_pattern": by_pattern,
            "by_reason": by_reason,
            "recent_rejections": recent_rejections[-20:]  # Last 20
        }

    def generate_recommendations(
        self,
        feedback_stats: Dict[str, Any],
        rule_config_service: Any  # Avoid circular import
    ) -> List[FeedbackRecommendation]:
        """
        Generate parameter adjustment recommendations based on feedback.

        Args:
            feedback_stats: Aggregated feedback statistics
            rule_config_service: The RuleConfigService to get current params

        Returns:
            List of recommendations
        """
        recommendations = []

        # Mapping from feedback category to parameter adjustments
        CATEGORY_TO_PARAMS = {
            "body_too_large": {
                "params": ["body_to_range_ratio", "body_to_avg_ratio", "body_max_ratio"],
                "direction": "decrease",
                "step": 0.02,
                "impact": "Strengerer Body-Check führt zu weniger False Positives"
            },
            "body_too_small": {
                "params": ["body_min_ratio", "prev_body_min_ratio", "curr_body_min_ratio"],
                "direction": "increase",
                "step": 0.05,
                "impact": "Lockererer Body-Check akzeptiert mehr kleine Bodies"
            },
            "upper_shadow_too_short": {
                "params": ["upper_shadow_min_ratio"],
                "direction": "decrease",
                "step": 0.05,
                "impact": "Reduzierte Anforderung an Upper Shadow Länge"
            },
            "upper_shadow_too_long": {
                "params": ["upper_shadow_max_ratio"],
                "direction": "decrease",
                "step": 0.02,
                "impact": "Strengere Begrenzung der Upper Shadow"
            },
            "lower_shadow_too_short": {
                "params": ["lower_shadow_min_ratio"],
                "direction": "decrease",
                "step": 0.05,
                "impact": "Reduzierte Anforderung an Lower Shadow Länge"
            },
            "lower_shadow_too_long": {
                "params": ["lower_shadow_max_ratio"],
                "direction": "decrease",
                "step": 0.02,
                "impact": "Strengere Begrenzung der Lower Shadow"
            },
            "not_fully_engulfing": {
                "params": ["size_ratio_min"],
                "direction": "increase",
                "step": 0.1,
                "impact": "Strengeres Engulfing-Verhältnis"
            },
            "engulfing_too_small": {
                "params": ["size_ratio_min", "curr_body_min_avg"],
                "direction": "increase",
                "step": 0.15,
                "impact": "Größere Engulfing-Kerze erforderlich"
            },
            "wrong_trend_context": {
                "params": ["uptrend_threshold", "downtrend_threshold"],
                "direction": "increase_abs",
                "step": 0.5,
                "impact": "Strengere Trend-Erkennung"
            },
            "no_prior_trend": {
                "params": ["lookback_candles"],
                "direction": "increase",
                "step": 2,
                "impact": "Längerer Lookback für Trend-Bestimmung"
            },
            "missing_gap": {
                "params": ["gap_tolerance"],
                "direction": "decrease",
                "step": 0.001,
                "impact": "Strengere Gap-Anforderung"
            },
            "false_positive": {
                "params": ["body_max_ratio", "body_to_range_ratio"],
                "direction": "decrease",
                "step": 0.03,
                "impact": "Allgemein strengere Kriterien"
            },
            "shadow_body_ratio_wrong": {
                "params": ["shadow_to_body_min"],
                "direction": "increase",
                "step": 0.25,
                "impact": "Strengeres Shadow-to-Body Verhältnis"
            },
        }

        by_pattern_and_reason = feedback_stats.get("by_pattern_and_reason", {})

        for key, count in sorted(by_pattern_and_reason.items(), key=lambda x: -x[1]):
            if count < 2:  # Minimum threshold
                continue

            parts = key.split(":", 1)
            if len(parts) != 2:
                continue

            pattern, reason = parts

            if reason not in CATEGORY_TO_PARAMS:
                continue

            adjustment_info = CATEGORY_TO_PARAMS[reason]

            try:
                pattern_params = rule_config_service.get_pattern_params(pattern)
            except Exception:
                continue

            for param_name in adjustment_info["params"]:
                current_value = pattern_params.get(param_name)
                if current_value is None:
                    continue

                # Calculate new value
                step = adjustment_info["step"]
                direction = adjustment_info["direction"]

                # Scale step by feedback count
                multiplier = min(3, count // 2)  # Cap at 3x
                scaled_step = step * max(1, multiplier)

                if direction == "decrease":
                    new_value = max(0.01, current_value - scaled_step)
                elif direction == "increase":
                    new_value = current_value + scaled_step
                elif direction == "increase_abs":
                    if current_value >= 0:
                        new_value = current_value + scaled_step
                    else:
                        new_value = current_value - scaled_step
                else:
                    continue

                # Round to reasonable precision
                new_value = round(new_value, 4)

                if abs(new_value - current_value) < 0.001:
                    continue  # No significant change

                # Calculate confidence based on feedback count
                confidence = min(0.95, 0.4 + count * 0.1)

                # Determine priority
                if count >= 5:
                    priority = "high"
                elif count >= 3:
                    priority = "medium"
                else:
                    priority = "low"

                recommendations.append(FeedbackRecommendation(
                    pattern=pattern,
                    parameter=param_name,
                    current_value=current_value,
                    recommended_value=new_value,
                    reason=reason,
                    feedback_count=count,
                    confidence=confidence,
                    priority=priority,
                    impact_estimate=adjustment_info["impact"]
                ))

        # Sort by feedback count (most important first)
        recommendations.sort(key=lambda x: (-x.feedback_count, -x.confidence))

        return recommendations[:20]  # Return top 20

    def track_adjustment(
        self,
        pattern: str,
        parameter: str,
        old_value: float,
        new_value: float,
        reason: str,
        feedback_count: int
    ):
        """Track a parameter adjustment for impact analysis."""
        self._adjustment_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pattern": pattern,
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "feedback_count": feedback_count,
            "pre_rejection_count": 0,  # Will be filled with actual data
            "post_rejection_count": None,  # Measured later
        })

    def measure_impact(
        self,
        validation_history: List[Dict],
        adjustment_timestamp: str
    ) -> Dict[str, Any]:
        """
        Measure the impact of parameter adjustments by comparing
        rejection rates before and after the adjustment.

        Args:
            validation_history: All validation results
            adjustment_timestamp: When the adjustment was made

        Returns:
            Impact metrics
        """
        before = []
        after = []

        for v in validation_history:
            ts = v.get("validation_timestamp", "")
            if ts < adjustment_timestamp:
                before.append(v)
            else:
                after.append(v)

        def calc_rejection_rate(validations):
            if not validations:
                return 0.0
            rejections = sum(1 for v in validations if not v.get("claude_agrees", True))
            return round(rejections / len(validations) * 100, 1)

        before_rate = calc_rejection_rate(before)
        after_rate = calc_rejection_rate(after)

        improvement = before_rate - after_rate if before_rate > 0 else 0

        return {
            "before": {
                "total": len(before),
                "rejection_rate": before_rate
            },
            "after": {
                "total": len(after),
                "rejection_rate": after_rate
            },
            "improvement_pct": round(improvement, 1),
            "improved": improvement > 0
        }

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of all analyzed feedback."""
        return {
            "total_feedback_items": len(self._feedback_categories),
            "unique_patterns": len(set(f.pattern_type for f in self._feedback_categories)),
            "unique_categories": len(set(f.category for f in self._feedback_categories)),
            "adjustments_made": len(self._adjustment_history),
            "recent_adjustments": self._adjustment_history[-10:]
        }


# Global singleton
feedback_analyzer_service = FeedbackAnalyzerService()
