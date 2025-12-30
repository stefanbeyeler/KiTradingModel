"""
Auto-Optimization Service for Candlestick Pattern Detection.

This service provides fully automated optimization of pattern detection rules
by integrating Claude Vision Validator feedback into the re-validation process.

Features:
1. Automatic Claude validation after new patterns are detected
2. Automatic application of recommendations when confidence threshold is met
3. Integration with the re-validation process
4. Configurable thresholds and controls
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from loguru import logger

from .claude_validator_service import claude_validator_service, ValidationStatus
from .feedback_analyzer_service import feedback_analyzer_service
from .rule_config_service import rule_config_service
from .pattern_history_service import pattern_history_service


@dataclass
class AutoOptimizationConfig:
    """Configuration for auto-optimization behavior."""
    # Enable/disable switches
    enabled: bool = True
    auto_validate_new_patterns: bool = True  # Validate new patterns automatically
    auto_apply_recommendations: bool = False  # Apply recommendations automatically
    auto_revalidate_after_adjustment: bool = True  # Re-validate after parameter changes

    # Thresholds
    validation_sample_rate: float = 0.3  # Validate 30% of new patterns
    min_confidence_for_auto_apply: float = 0.85  # Minimum confidence to auto-apply
    min_feedback_count_for_auto_apply: int = 5  # Minimum feedback count to auto-apply
    max_auto_adjustments_per_hour: int = 3  # Limit auto-adjustments

    # Batch settings
    batch_size: int = 5  # Patterns to validate per batch
    batch_interval_seconds: int = 60  # Interval between batches

    def to_dict(self) -> Dict:
        return asdict(self)


class AutoOptimizationService:
    """
    Service for automated optimization of pattern detection rules.

    This service:
    1. Hooks into pattern_history_service to validate new patterns
    2. Automatically applies recommendations when confidence is high enough
    3. Tracks optimization history and impact
    """

    def __init__(self):
        self.config = AutoOptimizationConfig(
            enabled=os.getenv("AUTO_OPTIMIZATION_ENABLED", "true").lower() == "true",
            auto_validate_new_patterns=os.getenv("AUTO_VALIDATE_NEW_PATTERNS", "true").lower() == "true",
            auto_apply_recommendations=os.getenv("AUTO_APPLY_RECOMMENDATIONS", "false").lower() == "true",
            auto_revalidate_after_adjustment=os.getenv("AUTO_REVALIDATE_AFTER_ADJUSTMENT", "true").lower() == "true",
            validation_sample_rate=float(os.getenv("VALIDATION_SAMPLE_RATE", "0.3")),
            min_confidence_for_auto_apply=float(os.getenv("MIN_CONFIDENCE_FOR_AUTO_APPLY", "0.85")),
            min_feedback_count_for_auto_apply=int(os.getenv("MIN_FEEDBACK_COUNT_FOR_AUTO_APPLY", "5")),
            max_auto_adjustments_per_hour=int(os.getenv("MAX_AUTO_ADJUSTMENTS_PER_HOUR", "3")),
        )

        # Tracking
        self._pending_validations: List[Dict] = []
        self._auto_adjustments_this_hour: int = 0
        self._last_hour_reset: datetime = datetime.now(timezone.utc)
        self._optimization_history: List[Dict] = []

        # Background task
        self._running = False
        self._validation_task: Optional[asyncio.Task] = None

        logger.info(f"AutoOptimizationService initialized - enabled: {self.config.enabled}")

    def _reset_hourly_counter_if_needed(self):
        """Reset the hourly adjustment counter if an hour has passed."""
        now = datetime.now(timezone.utc)
        if (now - self._last_hour_reset).total_seconds() >= 3600:
            self._auto_adjustments_this_hour = 0
            self._last_hour_reset = now

    async def queue_patterns_for_validation(self, patterns: List[Dict]) -> int:
        """
        Queue new patterns for automatic Claude validation.

        Args:
            patterns: List of pattern dictionaries from pattern_history_service

        Returns:
            Number of patterns queued
        """
        if not self.config.enabled or not self.config.auto_validate_new_patterns:
            return 0

        import random

        queued = 0
        for pattern in patterns:
            # Apply sampling rate
            if random.random() > self.config.validation_sample_rate:
                continue

            # Check if already pending
            pattern_id = pattern.get("id", "")
            if any(p.get("id") == pattern_id for p in self._pending_validations):
                continue

            self._pending_validations.append(pattern)
            queued += 1

        if queued > 0:
            logger.debug(f"Queued {queued} patterns for automatic validation")

        return queued

    async def _process_validation_batch(self) -> int:
        """
        Process a batch of pending validations.

        Returns:
            Number of patterns validated
        """
        if not self._pending_validations:
            return 0

        # Get batch
        batch = self._pending_validations[:self.config.batch_size]
        self._pending_validations = self._pending_validations[self.config.batch_size:]

        validated = 0
        for pattern in batch:
            try:
                result = await self._validate_pattern(pattern)
                if result:
                    validated += 1

                    # If rejected, check for auto-apply opportunities
                    if not result.get("claude_agrees", True):
                        await self._process_rejection_feedback(result)

            except Exception as e:
                logger.error(f"Error validating pattern {pattern.get('id')}: {e}")

        return validated

    async def _validate_pattern(self, pattern: Dict) -> Optional[Dict]:
        """
        Validate a single pattern using Claude.

        Args:
            pattern: Pattern dictionary

        Returns:
            Validation result dictionary
        """
        try:
            from ..services.pattern_detection_service import candlestick_pattern_service

            symbol = pattern.get("symbol", "")
            timeframe = pattern.get("timeframe", "H1")
            pattern_id = pattern.get("id", "")
            pattern_type = pattern.get("pattern_type", "")

            # Get OHLCV data for this pattern
            ohlcv_data = await candlestick_pattern_service._fetch_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=30
            )

            if not ohlcv_data:
                logger.warning(f"No OHLCV data for pattern {pattern_id}")
                return None

            # Validate with Claude
            result = await claude_validator_service.validate_pattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                ohlcv_data=ohlcv_data
            )

            logger.info(
                f"Auto-validated {pattern_type} ({symbol}/{timeframe}): "
                f"agrees={result.claude_agrees}, confidence={result.claude_confidence:.2f}"
            )

            return result.to_dict()

        except Exception as e:
            logger.error(f"Error in auto-validation: {e}")
            return None

    async def _process_rejection_feedback(self, validation_result: Dict):
        """
        Process a rejection and potentially auto-apply recommendations.

        Args:
            validation_result: The Claude validation result
        """
        try:
            # Get all validation history including new rejection
            history = claude_validator_service.get_validation_history(limit=500)

            # Analyze feedback
            feedback_stats = feedback_analyzer_service.analyze_validation_history(history)

            # Generate recommendations
            recommendations = feedback_analyzer_service.generate_recommendations(
                feedback_stats=feedback_stats,
                rule_config_service=rule_config_service
            )

            # Check if we should auto-apply any recommendations
            if self.config.auto_apply_recommendations:
                await self._auto_apply_high_confidence_recommendations(recommendations)

        except Exception as e:
            logger.error(f"Error processing rejection feedback: {e}")

    async def _auto_apply_high_confidence_recommendations(
        self,
        recommendations: List[Any]
    ) -> int:
        """
        Automatically apply recommendations that meet the confidence threshold.

        Args:
            recommendations: List of FeedbackRecommendation objects

        Returns:
            Number of recommendations applied
        """
        self._reset_hourly_counter_if_needed()

        applied = 0
        for rec in recommendations:
            # Check hourly limit
            if self._auto_adjustments_this_hour >= self.config.max_auto_adjustments_per_hour:
                logger.info("Hourly auto-adjustment limit reached")
                break

            # Check confidence threshold
            if rec.confidence < self.config.min_confidence_for_auto_apply:
                continue

            # Check feedback count threshold
            if rec.feedback_count < self.config.min_feedback_count_for_auto_apply:
                continue

            # Check for extreme changes (max 50% for auto-apply)
            current = rec.current_value
            new = rec.recommended_value
            if current > 0:
                change_pct = abs(new - current) / current
                if change_pct > 0.5:
                    logger.info(
                        f"Skipping auto-apply for {rec.pattern}.{rec.parameter}: "
                        f"change too large ({change_pct:.1%})"
                    )
                    continue

            try:
                # Apply the recommendation
                success = rule_config_service.set_param(
                    pattern=rec.pattern,
                    param_name=rec.parameter,
                    value=rec.recommended_value,
                    reason=f"auto_optimization:{rec.reason}",
                    feedback_count=rec.feedback_count
                )

                if success:
                    applied += 1
                    self._auto_adjustments_this_hour += 1

                    # Record in history
                    self._optimization_history.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "action": "auto_apply",
                        "pattern": rec.pattern,
                        "parameter": rec.parameter,
                        "old_value": rec.current_value,
                        "new_value": rec.recommended_value,
                        "reason": rec.reason,
                        "confidence": rec.confidence,
                        "feedback_count": rec.feedback_count
                    })

                    logger.info(
                        f"Auto-applied recommendation: {rec.pattern}.{rec.parameter} "
                        f"{rec.current_value} -> {rec.recommended_value} "
                        f"(confidence: {rec.confidence:.2f}, count: {rec.feedback_count})"
                    )

            except Exception as e:
                logger.error(f"Failed to auto-apply recommendation: {e}")

        return applied

    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """
        Run a complete optimization cycle:
        1. Validate pending patterns
        2. Analyze feedback
        3. Apply recommendations if configured

        Returns:
            Cycle results
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patterns_validated": 0,
            "rejections_found": 0,
            "recommendations_applied": 0,
            "pending_validations": len(self._pending_validations)
        }

        if not self.config.enabled:
            results["status"] = "disabled"
            return results

        try:
            # 1. Process pending validations
            results["patterns_validated"] = await self._process_validation_batch()

            # 2. Get validation statistics
            stats = claude_validator_service.get_validation_statistics()
            results["rejections_found"] = stats.get("rejections", 0)

            # 3. If auto-apply is enabled, process recommendations
            if self.config.auto_apply_recommendations:
                history = claude_validator_service.get_validation_history(limit=500)
                feedback_stats = feedback_analyzer_service.analyze_validation_history(history)
                recommendations = feedback_analyzer_service.generate_recommendations(
                    feedback_stats=feedback_stats,
                    rule_config_service=rule_config_service
                )
                results["recommendations_applied"] = await self._auto_apply_high_confidence_recommendations(
                    recommendations
                )

            results["status"] = "success"

        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
            results["status"] = "error"
            results["error"] = str(e)

        return results

    async def _optimization_loop(self):
        """Background loop for continuous optimization."""
        while self._running:
            try:
                await self.run_optimization_cycle()
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")

            await asyncio.sleep(self.config.batch_interval_seconds)

    async def start(self):
        """Start the auto-optimization service."""
        if self._running:
            return

        if not self.config.enabled:
            logger.info("Auto-optimization is disabled")
            return

        self._running = True
        self._validation_task = asyncio.create_task(self._optimization_loop())
        logger.info("Auto-optimization service started")

    async def stop(self):
        """Stop the auto-optimization service."""
        self._running = False
        if self._validation_task:
            self._validation_task.cancel()
            try:
                await self._validation_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-optimization service stopped")

    def update_config(self, **kwargs) -> Dict:
        """
        Update configuration settings.

        Args:
            **kwargs: Configuration key-value pairs

        Returns:
            Updated configuration
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Auto-optimization config updated: {key}={value}")

        return self.config.to_dict()

    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "running": self._running,
            "config": self.config.to_dict(),
            "pending_validations": len(self._pending_validations),
            "auto_adjustments_this_hour": self._auto_adjustments_this_hour,
            "optimization_history_count": len(self._optimization_history),
            "claude_validator_status": claude_validator_service.get_status()
        }

    def get_optimization_history(self, limit: int = 50) -> List[Dict]:
        """Get recent optimization history."""
        return list(reversed(self._optimization_history[-limit:]))

    async def trigger_revalidation_with_claude(
        self,
        pattern_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Trigger Claude re-validation for specific patterns.

        This is used after user feedback to verify if patterns are now
        being detected correctly with updated parameters.

        Args:
            pattern_ids: List of pattern IDs to revalidate

        Returns:
            Revalidation results
        """
        results = {
            "requested": len(pattern_ids),
            "validated": 0,
            "agreed": 0,
            "rejected": 0,
            "errors": 0,
            "details": []
        }

        for pattern_id in pattern_ids:
            try:
                # Find pattern in history
                history = pattern_history_service.get_history(limit=1000)
                pattern = next(
                    (p for p in history if p.get("id") == pattern_id),
                    None
                )

                if not pattern:
                    results["errors"] += 1
                    results["details"].append({
                        "pattern_id": pattern_id,
                        "status": "not_found"
                    })
                    continue

                # Validate with Claude
                validation = await self._validate_pattern(pattern)

                if validation:
                    results["validated"] += 1
                    if validation.get("claude_agrees", False):
                        results["agreed"] += 1
                    else:
                        results["rejected"] += 1

                    results["details"].append({
                        "pattern_id": pattern_id,
                        "status": "validated",
                        "agrees": validation.get("claude_agrees"),
                        "confidence": validation.get("claude_confidence"),
                        "reasoning": validation.get("claude_reasoning", "")[:200]
                    })
                else:
                    results["errors"] += 1
                    results["details"].append({
                        "pattern_id": pattern_id,
                        "status": "error"
                    })

            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "pattern_id": pattern_id,
                    "status": "error",
                    "error": str(e)
                })

        return results


# Global singleton
auto_optimization_service = AutoOptimizationService()


# Hook into pattern_history_service scan
async def on_patterns_detected(new_patterns: List[Dict]):
    """
    Callback when new patterns are detected by pattern_history_service.

    This function should be called after scan_all_symbols() completes.
    """
    if auto_optimization_service.config.enabled:
        await auto_optimization_service.queue_patterns_for_validation(new_patterns)
