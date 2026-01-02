"""Rule Optimizer Service - Claude Vision-based Pattern Rule Optimization.

This service performs a one-time comprehensive analysis of pattern detection rules
using Claude Vision API to evaluate detected patterns and optimize rule parameters.

Workflow:
1. Collect sample patterns from live market data
2. Validate each pattern using Claude Vision
3. Analyze validation results to identify systematic issues
4. Generate optimized rule parameters
5. Apply optimizations to the rule configuration
"""

import asyncio
import json
import statistics
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from .claude_validator_service import claude_validator_service, ClaudeValidationResult, ValidationStatus
from .pattern_detection_service import candlestick_pattern_service
from .rule_config_service import rule_config_service, DEFAULT_RULE_PARAMS
from ..models.schemas import PatternScanRequest, Timeframe


class PatternValidity(str, Enum):
    """Pattern validity classification from Claude."""
    VALID = "valid"
    INVALID = "invalid"
    BORDERLINE = "borderline"
    ERROR = "error"


@dataclass
class PatternSample:
    """A single pattern sample for analysis."""
    pattern_type: str
    symbol: str
    timeframe: str
    timestamp: str
    rule_confidence: float
    ohlcv_data: List[Dict]

    # Unique ID for this sample
    sample_id: str = field(default_factory=lambda: "")

    # Claude validation results
    claude_valid: Optional[PatternValidity] = None
    claude_confidence: Optional[float] = None
    claude_reasoning: Optional[str] = None
    claude_issues: List[str] = field(default_factory=list)
    claude_status: Optional[str] = None  # validated, rejected, error

    # Chart image (base64)
    chart_image: Optional[str] = None

    # Calculated metrics for the pattern
    body_ratio: Optional[float] = None
    upper_shadow_ratio: Optional[float] = None
    lower_shadow_ratio: Optional[float] = None
    engulfing_ratio: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dict, excluding large data like chart_image for API responses."""
        result = asdict(self)
        # Don't include ohlcv_data and chart_image in regular API responses
        result.pop('ohlcv_data', None)
        return result

    def to_summary_dict(self) -> Dict:
        """Convert to summary dict including chart_image for detailed view."""
        return {
            'sample_id': self.sample_id,
            'pattern_type': self.pattern_type,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp,
            'rule_confidence': self.rule_confidence,
            'claude_valid': self.claude_valid.value if self.claude_valid else None,
            'claude_confidence': self.claude_confidence,
            'claude_reasoning': self.claude_reasoning,
            'claude_status': self.claude_status,
            'chart_image': self.chart_image,
            'body_ratio': self.body_ratio,
            'upper_shadow_ratio': self.upper_shadow_ratio,
            'lower_shadow_ratio': self.lower_shadow_ratio,
        }


@dataclass
class PatternTypeAnalysis:
    """Analysis results for a specific pattern type."""
    pattern_type: str
    total_samples: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    borderline_count: int = 0
    error_count: int = 0

    # Issue frequency
    issues: Dict[str, int] = field(default_factory=dict)

    # Parameter statistics from invalid patterns
    invalid_body_ratios: List[float] = field(default_factory=list)
    invalid_shadow_ratios: List[float] = field(default_factory=list)

    # Parameter statistics from valid patterns
    valid_body_ratios: List[float] = field(default_factory=list)
    valid_shadow_ratios: List[float] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.valid_count / self.total_samples

    @property
    def false_positive_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.invalid_count / self.total_samples

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["validity_rate"] = self.validity_rate
        result["false_positive_rate"] = self.false_positive_rate
        return result


@dataclass
class OptimizationResult:
    """Result of a parameter optimization."""
    pattern_type: str
    parameter: str
    current_value: float
    recommended_value: float
    confidence: float
    reasoning: str
    sample_count: int
    improvement_estimate: float  # Expected reduction in false positives


@dataclass
class OptimizationSession:
    """A complete optimization session."""
    session_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"

    # Configuration
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    samples_per_pattern: int = 10

    # Results
    total_patterns_collected: int = 0
    total_patterns_validated: int = 0
    pattern_analyses: Dict[str, Dict] = field(default_factory=dict)
    recommendations: List[Dict] = field(default_factory=list)

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class RuleOptimizerService:
    """Service for optimizing pattern detection rules using Claude Vision."""

    def __init__(self):
        self.sessions: Dict[str, OptimizationSession] = {}
        self.current_session: Optional[OptimizationSession] = None
        self.samples: Dict[str, List[PatternSample]] = {}  # pattern_type -> samples
        self.analyses: Dict[str, PatternTypeAnalysis] = {}

        # Progress tracking
        self.progress: Dict[str, Any] = {
            "phase": "idle",
            "progress": 0,
            "message": "",
            "current_pattern": None,
            "patterns_collected": 0,
            "patterns_validated": 0,
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get current optimization progress."""
        return self.progress.copy()

    async def start_optimization(
        self,
        symbols: List[str],
        timeframes: List[str],
        samples_per_pattern: int = 10,
        pattern_types: Optional[List[str]] = None
    ) -> str:
        """
        Start a new optimization session.

        Args:
            symbols: List of symbols to scan
            timeframes: List of timeframes to scan
            samples_per_pattern: Number of samples to collect per pattern type
            pattern_types: Specific pattern types to optimize (None = all)

        Returns:
            Session ID
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]

        session = OptimizationSession(
            session_id=session_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            timeframes=timeframes,
            samples_per_pattern=samples_per_pattern,
        )

        self.sessions[session_id] = session
        self.current_session = session
        self.samples = {}
        self.analyses = {}

        # Start optimization in background
        asyncio.create_task(self._run_optimization(session, pattern_types))

        return session_id

    async def _run_optimization(
        self,
        session: OptimizationSession,
        pattern_types: Optional[List[str]] = None
    ):
        """Run the complete optimization workflow."""
        try:
            # Phase 1: Collect pattern samples
            self.progress = {
                "phase": "collecting",
                "progress": 0,
                "message": "Sammle Pattern-Samples...",
                "current_pattern": None,
                "patterns_collected": 0,
                "patterns_validated": 0,
            }

            await self._collect_samples(session, pattern_types)

            # Phase 2: Validate patterns with Claude Vision
            self.progress["phase"] = "validating"
            self.progress["message"] = "Validiere Patterns mit Claude Vision..."

            await self._validate_samples(session)

            # Phase 3: Analyze results
            self.progress["phase"] = "analyzing"
            self.progress["message"] = "Analysiere Validierungsergebnisse..."
            self.progress["progress"] = 80

            await self._analyze_results(session)

            # Phase 4: Generate recommendations
            self.progress["phase"] = "recommending"
            self.progress["message"] = "Generiere Optimierungsvorschläge..."
            self.progress["progress"] = 90

            recommendations = self._generate_recommendations()
            session.recommendations = [r.__dict__ if hasattr(r, '__dict__') else r for r in recommendations]

            # Complete
            session.status = "completed"
            session.completed_at = datetime.now(timezone.utc).isoformat()

            self.progress["phase"] = "completed"
            self.progress["progress"] = 100
            self.progress["message"] = f"Optimierung abgeschlossen. {len(recommendations)} Empfehlungen generiert."

            logger.info(f"Optimization session {session.session_id} completed with {len(recommendations)} recommendations")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            session.status = "failed"
            session.errors.append(str(e))
            self.progress["phase"] = "error"
            self.progress["message"] = f"Fehler: {str(e)}"

    async def _collect_samples(
        self,
        session: OptimizationSession,
        pattern_types: Optional[List[str]] = None
    ):
        """Collect pattern samples from market data."""
        all_pattern_types = [
            # Single candle patterns
            "doji", "dragonfly_doji", "gravestone_doji",
            "hammer", "inverted_hammer", "shooting_star", "hanging_man",
            "spinning_top",
            "bullish_belt_hold", "bearish_belt_hold",
            # Two candle patterns
            "bullish_engulfing", "bearish_engulfing",
            "bullish_harami", "bearish_harami", "harami_cross",
            "piercing_line", "dark_cloud_cover",
            "bullish_counterattack", "bearish_counterattack",
            # Three candle patterns
            "morning_star", "evening_star",
            "three_white_soldiers", "three_black_crows",
            "three_inside_up", "three_inside_down",
            "bullish_abandoned_baby", "bearish_abandoned_baby",
            "tower_bottom", "tower_top",
            "advance_block",
            # Island reversal patterns (multi-candle)
            "bearish_island", "bullish_island",
        ]

        target_patterns = pattern_types or all_pattern_types
        total_scans = len(session.symbols) * len(session.timeframes)
        scans_done = 0

        for symbol in session.symbols:
            for tf_str in session.timeframes:
                try:
                    # Convert timeframe string to enum
                    tf = Timeframe(tf_str.upper())

                    request = PatternScanRequest(
                        symbol=symbol,
                        timeframes=[tf],
                        min_confidence=0.3,  # Low threshold to catch edge cases
                        lookback_candles=100,
                        include_weak_patterns=True,
                    )

                    response = await candlestick_pattern_service.scan_patterns(request)

                    # Extract patterns from response
                    tf_result = getattr(response.result, tf_str.lower(), None)
                    if tf_result and tf_result.patterns:
                        for pattern in tf_result.patterns:
                            pt = pattern.pattern_type.value.lower()

                            if pt not in target_patterns:
                                continue

                            if pt not in self.samples:
                                self.samples[pt] = []

                            # Skip if we have enough samples
                            if len(self.samples[pt]) >= session.samples_per_pattern:
                                continue

                            # Fetch OHLCV data for this pattern
                            chart_data = await candlestick_pattern_service.get_pattern_chart_data(
                                symbol=symbol,
                                timeframe=tf_str,
                                pattern_timestamp=pattern.timestamp,
                                candles_before=10,
                                candles_after=3,
                            )

                            if chart_data.get("candles"):
                                # Generate unique sample ID
                                sample_id = f"{pt}_{symbol}_{tf_str}_{str(uuid.uuid4())[:8]}"

                                sample = PatternSample(
                                    pattern_type=pt,
                                    symbol=symbol,
                                    timeframe=tf_str,
                                    timestamp=pattern.timestamp.isoformat(),
                                    rule_confidence=pattern.confidence,
                                    ohlcv_data=chart_data["candles"],
                                    sample_id=sample_id,
                                )

                                # Calculate metrics
                                self._calculate_sample_metrics(sample)

                                self.samples[pt].append(sample)
                                session.total_patterns_collected += 1
                                self.progress["patterns_collected"] = session.total_patterns_collected

                except Exception as e:
                    logger.warning(f"Error scanning {symbol}/{tf_str}: {e}")
                    session.errors.append(f"Scan error {symbol}/{tf_str}: {str(e)}")

                scans_done += 1
                self.progress["progress"] = int((scans_done / total_scans) * 40)  # 0-40% for collection

        logger.info(f"Collected {session.total_patterns_collected} pattern samples across {len(self.samples)} pattern types")

    def _calculate_sample_metrics(self, sample: PatternSample):
        """Calculate pattern metrics from OHLCV data."""
        candles = sample.ohlcv_data
        if not candles:
            return

        # Find the pattern candle (marked with is_pattern=True)
        pattern_candle = None
        for c in candles:
            if c.get("is_pattern"):
                pattern_candle = c
                break

        if not pattern_candle:
            pattern_candle = candles[-1] if candles else None

        if not pattern_candle:
            return

        o = pattern_candle.get("open", 0)
        h = pattern_candle.get("high", 0)
        l = pattern_candle.get("low", 0)
        c = pattern_candle.get("close", 0)

        total_range = h - l
        if total_range <= 0:
            return

        body_size = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        sample.body_ratio = body_size / total_range
        sample.upper_shadow_ratio = upper_shadow / total_range
        sample.lower_shadow_ratio = lower_shadow / total_range

        # Calculate engulfing ratio for engulfing patterns
        if "engulfing" in sample.pattern_type:
            # Find previous candle
            pattern_idx = next((i for i, c in enumerate(candles) if c.get("is_pattern")), len(candles) - 1)
            if pattern_idx > 0:
                prev_candle = candles[pattern_idx - 1]
                prev_body = abs(prev_candle.get("close", 0) - prev_candle.get("open", 0))
                if prev_body > 0:
                    sample.engulfing_ratio = body_size / prev_body

    async def _validate_samples(self, session: OptimizationSession):
        """Validate collected samples using Claude Vision."""
        total_samples = sum(len(samples) for samples in self.samples.values())
        validated = 0

        for pattern_type, samples in self.samples.items():
            self.progress["current_pattern"] = pattern_type

            for sample in samples:
                try:
                    # Use Claude validator service with pattern_id
                    # CRITICAL: Pass pattern_timestamp so Claude analyzes the correct candle!
                    result = await claude_validator_service.validate_pattern(
                        pattern_id=sample.sample_id,
                        pattern_type=sample.pattern_type,
                        symbol=sample.symbol,
                        timeframe=sample.timeframe,
                        ohlcv_data=sample.ohlcv_data,
                        pattern_timestamp=sample.timestamp,  # Critical for correct candle selection
                    )

                    # Store the chart image from validation result
                    if hasattr(result, 'chart_image_base64') and result.chart_image_base64:
                        sample.chart_image = result.chart_image_base64

                    # Store the status for display
                    sample.claude_status = result.status.value if result.status else None

                    # Map validation result to our classification
                    # WICHTIG: Status-basierte Klassifikation hat Priorität
                    if result.status == ValidationStatus.ERROR:
                        sample.claude_valid = PatternValidity.ERROR
                        logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): ERROR status")
                    elif result.status == ValidationStatus.REJECTED:
                        # Claude hat das Pattern abgelehnt - es ist ein False Positive
                        # HINWEIS: Bei REJECTED ist agrees oft None, daher prüfen wir hier nur den Status
                        sample.claude_valid = PatternValidity.INVALID
                        logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): REJECTED -> INVALID")
                    elif result.claude_agrees is False:
                        # Claude stimmt nicht zu, aber Status ist nicht REJECTED (z.B. VALIDATED mit agrees=False)
                        sample.claude_valid = PatternValidity.INVALID
                        logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): claude_agrees=False -> INVALID")
                    elif result.status == ValidationStatus.VALIDATED:
                        # Claude hat validiert - prüfe Confidence
                        if result.claude_agrees is True or result.claude_confidence >= 0.6:
                            sample.claude_valid = PatternValidity.VALID
                            logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): VALIDATED (conf={result.claude_confidence}) -> VALID")
                        elif result.claude_confidence >= 0.4:
                            sample.claude_valid = PatternValidity.BORDERLINE
                            logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): VALIDATED (conf={result.claude_confidence}) -> BORDERLINE")
                        else:
                            # Niedrige Confidence bei VALIDATED = grenzwertig
                            sample.claude_valid = PatternValidity.BORDERLINE
                            logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): VALIDATED low conf -> BORDERLINE")
                    elif result.status == ValidationStatus.SKIPPED:
                        # Übersprungen = Error zählen
                        sample.claude_valid = PatternValidity.ERROR
                        logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): SKIPPED -> ERROR")
                    else:
                        # Fallback basierend auf Confidence
                        if result.claude_confidence >= 0.6:
                            sample.claude_valid = PatternValidity.VALID
                        elif result.claude_confidence >= 0.4:
                            sample.claude_valid = PatternValidity.BORDERLINE
                        else:
                            sample.claude_valid = PatternValidity.INVALID
                        logger.info(f"Pattern {sample.pattern_type} ({sample.symbol}/{sample.timeframe}): Fallback (conf={result.claude_confidence}) -> {sample.claude_valid}")

                    sample.claude_confidence = result.claude_confidence
                    sample.claude_reasoning = result.claude_reasoning

                    # Extract issues from reasoning
                    if result.claude_reasoning:
                        sample.claude_issues = self._extract_issues(result.claude_reasoning)

                    session.total_patterns_validated += 1

                except Exception as e:
                    logger.warning(f"Validation error for {pattern_type}: {e}")
                    sample.claude_valid = PatternValidity.ERROR
                    sample.claude_status = "error"
                    sample.claude_issues = [str(e)]

                validated += 1
                self.progress["patterns_validated"] = validated
                self.progress["progress"] = 40 + int((validated / total_samples) * 40)  # 40-80%

                # Rate limiting - don't overwhelm the API
                await asyncio.sleep(0.5)

        logger.info(f"Validated {validated} pattern samples")

    def _extract_issues(self, reasoning: str) -> List[str]:
        """Extract specific issues from Claude's reasoning."""
        issues = []
        reasoning_lower = reasoning.lower()

        issue_keywords = {
            "body_too_large": ["body too large", "body is too large", "körper zu groß"],
            "body_too_small": ["body too small", "body is too small", "körper zu klein"],
            "upper_shadow_too_short": ["upper shadow too short", "oberer schatten zu kurz"],
            "upper_shadow_too_long": ["upper shadow too long", "oberer schatten zu lang"],
            "lower_shadow_too_short": ["lower shadow too short", "unterer schatten zu kurz"],
            "lower_shadow_too_long": ["lower shadow too long", "unterer schatten zu lang"],
            "not_fully_engulfing": ["not fully engulfing", "doesn't fully engulf", "nicht vollständig"],
            "wrong_trend_context": ["wrong trend", "incorrect trend", "falscher trend"],
            "missing_gap": ["missing gap", "no gap", "kein gap", "fehlendes gap"],
            "shadows_not_balanced": ["shadows not balanced", "unbalanced shadows"],
            "not_a_doji": ["not a doji", "not doji", "kein doji"],
            "no_prior_trend": ["no prior trend", "kein vorheriger trend"],
        }

        for issue_code, keywords in issue_keywords.items():
            for keyword in keywords:
                if keyword in reasoning_lower:
                    issues.append(issue_code)
                    break

        return issues

    async def _analyze_results(self, session: OptimizationSession):
        """Analyze validation results to identify patterns."""
        for pattern_type, samples in self.samples.items():
            analysis = PatternTypeAnalysis(pattern_type=pattern_type)

            for sample in samples:
                analysis.total_samples += 1

                if sample.claude_valid == PatternValidity.VALID:
                    analysis.valid_count += 1
                    if sample.body_ratio is not None:
                        analysis.valid_body_ratios.append(sample.body_ratio)
                    if sample.upper_shadow_ratio is not None:
                        analysis.valid_shadow_ratios.append(sample.upper_shadow_ratio)

                elif sample.claude_valid == PatternValidity.INVALID:
                    analysis.invalid_count += 1
                    if sample.body_ratio is not None:
                        analysis.invalid_body_ratios.append(sample.body_ratio)
                    if sample.upper_shadow_ratio is not None:
                        analysis.invalid_shadow_ratios.append(sample.upper_shadow_ratio)

                    # Count issues
                    for issue in sample.claude_issues:
                        analysis.issues[issue] = analysis.issues.get(issue, 0) + 1

                elif sample.claude_valid == PatternValidity.BORDERLINE:
                    analysis.borderline_count += 1
                else:
                    analysis.error_count += 1

            self.analyses[pattern_type] = analysis
            session.pattern_analyses[pattern_type] = analysis.to_dict()

        logger.info(f"Analyzed {len(self.analyses)} pattern types")

    def _generate_recommendations(self) -> List[OptimizationResult]:
        """Generate parameter adjustment recommendations based on analysis."""
        recommendations = []

        for pattern_type, analysis in self.analyses.items():
            # Skip if not enough data
            if analysis.total_samples < 3:
                continue

            # Skip if already good (>80% valid)
            if analysis.validity_rate > 0.8:
                continue

            current_params = rule_config_service.get_pattern_params(pattern_type)

            # Analyze body ratio issues
            if analysis.valid_body_ratios and analysis.invalid_body_ratios:
                valid_avg = statistics.mean(analysis.valid_body_ratios)
                invalid_avg = statistics.mean(analysis.invalid_body_ratios)

                # If invalid patterns have larger bodies, we need stricter body limits
                if invalid_avg > valid_avg * 1.2:  # Invalid bodies are 20%+ larger
                    for param in ["body_to_range_ratio", "body_max_ratio"]:
                        if param in current_params:
                            current = current_params[param]
                            # Recommend value between valid average and current
                            recommended = (valid_avg + current) / 2
                            recommended = max(0.05, min(0.5, round(recommended, 3)))

                            if abs(recommended - current) > 0.02:
                                recommendations.append(OptimizationResult(
                                    pattern_type=pattern_type,
                                    parameter=param,
                                    current_value=current,
                                    recommended_value=recommended,
                                    confidence=min(0.9, 0.5 + analysis.total_samples * 0.05),
                                    reasoning=f"Invalid patterns haben im Schnitt größere Bodies ({invalid_avg:.2f}) als valide ({valid_avg:.2f})",
                                    sample_count=analysis.total_samples,
                                    improvement_estimate=analysis.false_positive_rate * 0.5,
                                ))

            # Analyze shadow ratio issues
            if analysis.valid_shadow_ratios and analysis.invalid_shadow_ratios:
                valid_avg = statistics.mean(analysis.valid_shadow_ratios)
                invalid_avg = statistics.mean(analysis.invalid_shadow_ratios)

                for param in ["upper_shadow_min_ratio", "lower_shadow_min_ratio"]:
                    if param in current_params:
                        current = current_params[param]

                        # If valid patterns have longer shadows, increase minimum
                        if valid_avg > invalid_avg * 1.2:
                            recommended = (valid_avg + current) / 2
                            recommended = max(0.1, min(0.8, round(recommended, 3)))

                            if recommended > current:
                                recommendations.append(OptimizationResult(
                                    pattern_type=pattern_type,
                                    parameter=param,
                                    current_value=current,
                                    recommended_value=recommended,
                                    confidence=min(0.9, 0.5 + analysis.total_samples * 0.05),
                                    reasoning=f"Valide Patterns haben längere Schatten ({valid_avg:.2f}) - erhöhe Minimum",
                                    sample_count=analysis.total_samples,
                                    improvement_estimate=analysis.false_positive_rate * 0.4,
                                ))

            # Analyze specific issues
            for issue, count in analysis.issues.items():
                if count < 2:  # Need at least 2 occurrences
                    continue

                issue_ratio = count / analysis.total_samples

                if issue == "body_too_large" and "body_to_range_ratio" in current_params:
                    current = current_params["body_to_range_ratio"]
                    recommended = current * 0.8  # Reduce by 20%
                    recommendations.append(OptimizationResult(
                        pattern_type=pattern_type,
                        parameter="body_to_range_ratio",
                        current_value=current,
                        recommended_value=round(recommended, 3),
                        confidence=min(0.9, issue_ratio + 0.3),
                        reasoning=f"{count} Patterns wurden als 'Body zu groß' markiert ({issue_ratio*100:.0f}%)",
                        sample_count=count,
                        improvement_estimate=issue_ratio * 0.7,
                    ))

                elif issue == "not_fully_engulfing" and "size_ratio_min" in current_params:
                    current = current_params["size_ratio_min"]
                    recommended = current * 1.2  # Increase by 20%
                    recommendations.append(OptimizationResult(
                        pattern_type=pattern_type,
                        parameter="size_ratio_min",
                        current_value=current,
                        recommended_value=round(recommended, 2),
                        confidence=min(0.9, issue_ratio + 0.3),
                        reasoning=f"{count} Patterns wurden als 'nicht vollständig engulfing' markiert",
                        sample_count=count,
                        improvement_estimate=issue_ratio * 0.6,
                    ))

                elif issue == "missing_gap" and "gap_tolerance" in current_params:
                    current = current_params.get("gap_tolerance", 0.998)
                    recommended = current - 0.002  # Stricter gap requirement
                    recommendations.append(OptimizationResult(
                        pattern_type=pattern_type,
                        parameter="gap_tolerance",
                        current_value=current,
                        recommended_value=round(recommended, 4),
                        confidence=min(0.9, issue_ratio + 0.3),
                        reasoning=f"{count} Patterns hatten kein echtes Gap",
                        sample_count=count,
                        improvement_estimate=issue_ratio * 0.5,
                    ))

        # Sort by improvement estimate
        recommendations.sort(key=lambda r: r.improvement_estimate, reverse=True)

        return recommendations[:30]  # Top 30 recommendations

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details."""
        session = self.sessions.get(session_id)
        if session:
            return session.to_dict()
        return None

    def get_recommendations(self) -> List[Dict]:
        """Get current recommendations."""
        if self.current_session:
            return self.current_session.recommendations
        return []

    def get_analyses(self) -> Dict[str, Dict]:
        """Get pattern type analyses."""
        return {pt: a.to_dict() for pt, a in self.analyses.items()}

    async def apply_recommendations(
        self,
        recommendation_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Apply selected recommendations to the rule configuration.

        Args:
            recommendation_indices: List of indices to apply (None = all)

        Returns:
            Summary of applied changes
        """
        if not self.current_session or not self.current_session.recommendations:
            return {"error": "No recommendations available"}

        recommendations = self.current_session.recommendations

        if recommendation_indices is None:
            to_apply = recommendations
        else:
            to_apply = [recommendations[i] for i in recommendation_indices if i < len(recommendations)]

        applied = []
        for rec in to_apply:
            try:
                success = rule_config_service.set_param(
                    pattern=rec["pattern_type"],
                    param_name=rec["parameter"],
                    value=rec["recommended_value"],
                    reason=f"claude_optimizer:{rec['reasoning'][:50]}",
                    feedback_count=rec["sample_count"],
                )

                if success:
                    applied.append({
                        "pattern": rec["pattern_type"],
                        "parameter": rec["parameter"],
                        "old_value": rec["current_value"],
                        "new_value": rec["recommended_value"],
                    })

            except Exception as e:
                logger.error(f"Failed to apply recommendation: {e}")

        return {
            "applied_count": len(applied),
            "total_recommendations": len(to_apply),
            "changes": applied,
        }

    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset all rule parameters to defaults."""
        rule_config_service.reset_all()
        return {"status": "reset", "message": "Alle Parameter auf Standardwerte zurückgesetzt"}

    def get_validated_samples(self) -> List[Dict]:
        """Get all validated samples with their details including chart images."""
        all_samples = []
        for pattern_type, samples in self.samples.items():
            for sample in samples:
                all_samples.append(sample.to_summary_dict())
        return all_samples

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict]:
        """Get a specific sample by its ID."""
        for pattern_type, samples in self.samples.items():
            for sample in samples:
                if sample.sample_id == sample_id:
                    return sample.to_summary_dict()
        return None


# Global singleton
rule_optimizer_service = RuleOptimizerService()
