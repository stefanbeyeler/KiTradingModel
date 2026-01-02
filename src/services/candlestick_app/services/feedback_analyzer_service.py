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
# Supports both English and German (Claude often responds in German)
FEEDBACK_CATEGORY_PATTERNS = {
    # Body-related issues (English + German)
    "body_too_large": [
        # English
        r"body.{0,20}(too|is|appears?).{0,10}(large|big|thick)",
        r"body.{0,20}(should be|needs to be).{0,10}(smaller|thinner)",
        r"(large|big|thick).{0,10}body",
        r"body.{0,10}(exceed|over|more than).{0,10}\d+%",
        # German
        r"body.?ratio.{0,20}(beträgt|ist).{0,10}\d+",
        r"körper.{0,20}(zu|ist).{0,10}(groß|gross|dick)",
        r"(zu|viel zu).{0,10}(groß|gross).{0,10}(körper|body)",
        r"\d+%.{0,20}(über|deutlich über|weit über).{0,20}(grenzwert|kriterium|limit)",
        r"(body|körper).{0,20}(über|überschreitet).{0,20}\d+%",
    ],
    "body_too_small": [
        # English
        r"body.{0,20}(too|is|appears?).{0,10}(small|thin|tiny)",
        r"body.{0,20}(should be|needs to be).{0,10}(larger|bigger)",
        r"(small|tiny|minimal).{0,10}body",
        r"not enough.{0,10}body",
        # German
        r"körper.{0,20}(zu|ist).{0,10}(klein|dünn)",
        r"(zu|viel zu).{0,10}(klein|dünn).{0,10}(körper|body)",
        r"(body|körper).{0,20}(weniger als|unter).{0,20}\d+%",
    ],

    # Shadow-related issues (English + German)
    "upper_shadow_too_short": [
        # English
        r"upper.{0,10}shadow.{0,20}(too|is).{0,10}(short|small|minimal)",
        r"(insufficient|lacking|missing).{0,10}upper.{0,10}shadow",
        r"upper.{0,10}wick.{0,20}(too|is).{0,10}(short|small)",
        # German
        r"(oberer|ober).{0,10}(schatten|docht).{0,20}(zu|ist).{0,10}(kurz|klein)",
        r"(fehlender|kein).{0,10}(oberer|ober).{0,10}(schatten|docht)",
    ],
    "upper_shadow_too_long": [
        # English
        r"upper.{0,10}shadow.{0,20}(too|is).{0,10}(long|large|big)",
        r"upper.{0,10}wick.{0,20}(too|is).{0,10}(long|large)",
        # German
        r"(oberer|ober).{0,10}(schatten|docht).{0,20}(zu|ist).{0,10}(lang|gross)",
    ],
    "lower_shadow_too_short": [
        # English
        r"lower.{0,10}shadow.{0,20}(too|is).{0,10}(short|small|minimal)",
        r"(insufficient|lacking|missing).{0,10}lower.{0,10}shadow",
        r"lower.{0,10}wick.{0,20}(too|is).{0,10}(short|small)",
        # German
        r"(unterer|unter).{0,10}(schatten|docht).{0,20}(zu|ist).{0,10}(kurz|klein)",
        r"(fehlender|kein).{0,10}(unterer|unter).{0,10}(schatten|docht)",
    ],
    "lower_shadow_too_long": [
        # English
        r"lower.{0,10}shadow.{0,20}(too|is).{0,10}(long|large|big)",
        r"lower.{0,10}wick.{0,20}(too|is).{0,10}(long|large)",
        # German
        r"(unterer|unter).{0,10}(schatten|docht).{0,20}(zu|ist).{0,10}(lang|gross)",
    ],

    # Engulfing-related issues (English + German)
    "not_fully_engulfing": [
        # English
        r"not.{0,20}(fully|completely).{0,10}engulf",
        r"(does not|doesn't).{0,10}engulf",
        r"fail.{0,20}engulf",
        r"engulfing.{0,20}not.{0,10}(complete|full)",
        # German
        r"(umhüllt|umschliesst).{0,10}nicht.{0,10}(vollständig|komplett)",
        r"nicht.{0,10}(vollständig|komplett).{0,10}(umhüllt|umschlossen)",
        r"kein.{0,10}(gültiges|echtes).{0,10}engulfing",
    ],
    "engulfing_too_small": [
        # English
        r"engulfing.{0,20}candle.{0,20}(too|is).{0,10}(small|weak)",
        r"(second|current).{0,10}candle.{0,20}not.{0,10}(large|big) enough",
        r"size.{0,10}ratio.{0,10}(insufficient|too small)",
        # German
        r"(aktuelle|zweite).{0,10}kerze.{0,20}(zu|ist).{0,10}(klein|schwach)",
    ],

    # Trend context issues (English + German)
    "wrong_trend_context": [
        # English
        r"(wrong|incorrect|inappropriate).{0,10}(trend|context)",
        r"no.{0,10}(prior|preceding).{0,10}(uptrend|downtrend)",
        r"(reversal|continuation).{0,10}pattern.{0,20}(without|missing).{0,10}trend",
        r"context.{0,20}not.{0,10}appropriate",
        # German
        r"(falscher|falsches|kein).{0,10}(trend|kontext)",
        r"(trend|kontext).{0,20}(nicht|fehlt)",
    ],
    "no_prior_trend": [
        # English
        r"no.{0,10}(clear|visible|prior).{0,10}trend",
        r"(lacking|missing|absent).{0,10}(uptrend|downtrend)",
        r"sideways.{0,10}(market|movement)",
        # German
        r"(kein|fehlender).{0,10}(vorheriger|klarer).{0,10}trend",
        r"seitwärts.{0,10}(markt|bewegung)",
    ],

    # Gap-related issues (English + German)
    "missing_gap": [
        # English
        r"(missing|no|absent).{0,10}gap",
        r"gap.{0,10}(required|needed|expected)",
        r"(should|needs to).{0,10}gap",
        # German
        r"(fehlende|kein).{0,10}(gap|lücke)",
        r"(gap|lücke).{0,10}(fehlt|erforderlich)",
    ],

    # General issues (English + German)
    "false_positive": [
        # English
        r"(false|incorrect).{0,10}positive",
        r"not.{0,20}(valid|genuine|true).{0,10}(pattern|formation)",
        r"(misidentified|incorrectly identified)",
        r"(does not|doesn't).{0,10}meet.{0,10}(criteria|definition)",
        # German
        r"(kein|nicht).{0,10}(gültiges|echtes|valides).{0,10}(muster|pattern)",
        r"(falsch|inkorrekt).{0,10}(identifiziert|erkannt)",
        r"(erfüllt|entspricht).{0,10}nicht.{0,10}(kriterien|definition)",
        r"nicht.{0,10}(vollständig|korrekt).{0,10}erfüllt",
    ],
    "wrong_pattern_type": [
        # English
        r"(looks more like|appears to be|actually|should be).{0,20}(doji|hammer|engulfing|star|harami)",
        r"(this is|i see).{0,10}(a|an).{0,10}(doji|hammer|engulfing|star|harami)",
        r"misclassified.{0,10}as",
        # German
        r"(sieht aus wie|scheint|ist eher).{0,20}(doji|hammer|engulfing|star|harami)",
        r"(falsch klassifiziert|fehlklassifiziert)",
    ],

    # Proportion issues (English + German)
    "shadow_body_ratio_wrong": [
        # English
        r"shadow.{0,10}(to|vs|compared).{0,10}body.{0,10}ratio",
        r"ratio.{0,20}(shadow|wick).{0,20}body",
        r"(shadow|wick).{0,20}(at least|minimum).{0,10}\d+x",
        # German
        r"(schatten|docht).{0,10}(zu|verhältnis).{0,10}(körper|body)",
        r"verhältnis.{0,20}(schatten|docht).{0,20}(körper|body)",
    ],

    # Doji-specific issues (German patterns for common doji rejections)
    "doji_body_too_large": [
        r"body.?ratio.{0,20}\d+%.{0,20}(über|deutlich über|weit über)",
        r"(doji|grenzwert).{0,20}5%",
        r"echter doji.{0,10}(erfordert|benötigt)",
        r"body.?ratio.{0,10}beträgt.{0,10}\d+%",
        r"automatische ablehnung",
    ],

    # Multi-candle pattern issues (German)
    "wrong_candle_direction": [
        r"kerze.{0,10}\d.{0,10}(ist|sollte).{0,10}(bullish|bearish|grün|rot)",
        r"(bullish|bearish).{0,10}statt.{0,10}(bullish|bearish)",
        r"(grün|rot).{0,10}statt.{0,10}(grün|rot)",
        r"alle.{0,10}(drei|3).{0,10}kerzen.{0,10}(bullish|bearish)",
    ],

    # Harami issues (German)
    "harami_not_contained": [
        r"(nicht|liegt nicht).{0,10}(vollständig|komplett).{0,10}(innerhalb|in|enthalten)",
        r"harami.{0,10}(bedingung|kriterium).{0,10}(nicht|fehlt)",
        r"kerze.{0,10}2.{0,10}(innerhalb|in).{0,10}kerze.{0,10}1",
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
