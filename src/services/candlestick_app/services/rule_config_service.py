"""Rule Configuration Service for adaptive pattern detection.

This service manages pattern detection rule parameters that can be
adjusted based on user feedback. Parameters are stored in a JSON file
and can be updated through the API.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from loguru import logger


# Default rule parameters (matching the hardcoded values in pattern_detection_service.py)
DEFAULT_RULE_PARAMS = {
    # Doji parameters
    "doji": {
        "body_to_range_ratio": 0.1,  # Body must be < 10% of range
        "body_to_avg_ratio": 0.1,    # Body must be < 10% of average
    },
    # Dragonfly Doji
    "dragonfly_doji": {
        "lower_shadow_min_ratio": 0.6,  # Lower shadow > 60% of range
        "upper_shadow_max_ratio": 0.1,  # Upper shadow < 10% of range
    },
    # Gravestone Doji
    "gravestone_doji": {
        "upper_shadow_min_ratio": 0.6,  # Upper shadow > 60% of range
        "lower_shadow_max_ratio": 0.1,  # Lower shadow < 10% of range
    },
    # Hammer
    "hammer": {
        "body_max_ratio": 0.35,         # Body < 35% of range
        "lower_shadow_min_ratio": 0.5,  # Lower shadow > 50% of range
        "upper_shadow_max_ratio": 0.15, # Upper shadow < 15% of range
        "shadow_to_body_min": 2.0,      # Lower shadow >= 2x body
    },
    # Inverted Hammer
    "inverted_hammer": {
        "body_max_ratio": 0.35,         # Body < 35% of range
        "upper_shadow_min_ratio": 0.5,  # Upper shadow > 50% of range
        "lower_shadow_max_ratio": 0.15, # Lower shadow < 15% of range
        "shadow_to_body_min": 2.0,      # Upper shadow >= 2x body
    },
    # Shooting Star / Hanging Man
    "shooting_star": {
        "gap_tolerance": 0.998,  # Open >= prev_close * 0.998
    },
    "hanging_man": {
        "gap_tolerance": 0.998,
    },
    # Spinning Top
    "spinning_top": {
        "body_max_ratio": 0.3,           # Body < 30% of range
        "shadow_min_ratio": 0.25,        # Both shadows > 25% of range
        "shadow_balance_max_diff": 0.2,  # Shadow difference < 20%
    },
    # Engulfing
    "bullish_engulfing": {
        "min_pct_move": 0.25,        # Min 0.25% price move
        "prev_body_min_ratio": 0.25, # Prev body >= 25% of its range
        "curr_body_min_ratio": 0.4,  # Current body >= 40% of its range
        "prev_body_min_avg": 0.5,    # Prev body >= 50% of average
        "curr_body_min_avg": 1.0,    # Current body >= 100% of average
        "size_ratio_min": 1.5,       # Current >= 1.5x previous
    },
    "bearish_engulfing": {
        "min_pct_move": 0.25,
        "prev_body_min_ratio": 0.25,
        "curr_body_min_ratio": 0.4,
        "prev_body_min_avg": 0.5,
        "curr_body_min_avg": 1.0,
        "size_ratio_min": 1.5,
    },
    # Harami
    "bullish_harami": {
        "body_size_max_ratio": 0.5,  # Current body < 50% of previous
    },
    "bearish_harami": {
        "body_size_max_ratio": 0.5,
    },
    # Harami Cross
    "harami_cross": {
        "prev_body_min_avg": 0.8,  # Previous body >= 80% of average
    },
    # Morning Star / Evening Star (strict rules for valid patterns)
    "morning_star": {
        "first_body_min_avg": 0.8,   # First candle >= 80% of average (large bearish)
        "second_body_max_avg": 0.3,  # Second candle <= 30% of average (small/Doji)
        "third_body_min_avg": 0.8,   # Third candle >= 80% of average (large bullish)
    },
    "evening_star": {
        "first_body_min_avg": 0.8,   # First candle >= 80% of average (large bullish)
        "second_body_max_avg": 0.3,  # Second candle <= 30% of average (small/Doji)
        "third_body_min_avg": 0.8,   # Third candle >= 80% of average (large bearish)
    },
    # Three White Soldiers / Three Black Crows
    "three_white_soldiers": {
        "body_min_avg": 0.5,          # Each body >= 50% of average
        "max_shadow_ratio": 0.3,      # Shadow < 30% of body
    },
    "three_black_crows": {
        "body_min_avg": 0.5,
        "max_shadow_ratio": 0.3,
    },
    # Trend detection
    "trend_detection": {
        "lookback_candles": 10,
        "uptrend_threshold": 1.0,    # +1% for uptrend
        "downtrend_threshold": -1.0, # -1% for downtrend
    },
    # Belt Hold - Einzelkerze mit langem Körper, fast ohne Schatten
    "bullish_belt_hold": {
        "body_min_ratio": 0.75,      # Body >= 75% of range (long body)
        "lower_shadow_max_ratio": 0.05,  # Lower shadow < 5% of range (opens at low)
        "upper_shadow_max_ratio": 0.15,  # Upper shadow < 15% of range
    },
    "bearish_belt_hold": {
        "body_min_ratio": 0.75,      # Body >= 75% of range (long body)
        "upper_shadow_max_ratio": 0.05,  # Upper shadow < 5% of range (opens at high)
        "lower_shadow_max_ratio": 0.15,  # Lower shadow < 15% of range
    },
    # Counterattack - 2 Kerzen, schliessen auf gleichem Niveau
    "bullish_counterattack": {
        "close_tolerance_pct": 0.05,  # Close within 0.05% of each other
        "body_min_ratio": 0.5,        # Both candles have significant bodies
        "gap_min_pct": 0.2,           # Gap down of at least 0.2%
    },
    "bearish_counterattack": {
        "close_tolerance_pct": 0.05,
        "body_min_ratio": 0.5,
        "gap_min_pct": 0.2,           # Gap up of at least 0.2%
    },
    # Three Inside Up/Down - Harami + Bestätigungskerze
    "three_inside_up": {
        "harami_body_max_ratio": 0.5,  # Second candle < 50% of first
        "confirm_close_above_first": True,  # Third closes above first's body
    },
    "three_inside_down": {
        "harami_body_max_ratio": 0.5,
        "confirm_close_below_first": True,
    },
    # Abandoned Baby - Doji mit Gaps auf beiden Seiten
    "bullish_abandoned_baby": {
        "doji_body_max_ratio": 0.1,  # Middle candle is Doji (< 10% body)
        "first_body_min_ratio": 0.5,  # First candle has significant body
        "third_body_min_ratio": 0.5,  # Third candle has significant body
    },
    "bearish_abandoned_baby": {
        "doji_body_max_ratio": 0.1,
        "first_body_min_ratio": 0.5,
        "third_body_min_ratio": 0.5,
    },
    # Tower patterns - Multi-Kerzen mit kleinen inneren Kerzen
    "tower_bottom": {
        "outer_body_min_ratio": 0.6,  # Outer candles have large bodies
        "inner_body_max_ratio": 0.4,  # Inner candles are small
        "min_inner_candles": 2,       # At least 2 inner candles
        "max_inner_candles": 10,      # Up to 10 inner candles
    },
    "tower_top": {
        "outer_body_min_ratio": 0.6,
        "inner_body_max_ratio": 0.4,
        "min_inner_candles": 2,
        "max_inner_candles": 10,
    },
}


@dataclass
class RuleAdjustmentRecord:
    """Record of a rule parameter adjustment."""
    timestamp: str
    pattern: str
    parameter: str
    old_value: float
    new_value: float
    reason: str
    feedback_count: int


class RuleConfigService:
    """Service for managing adaptive pattern detection rules."""

    def __init__(self, config_path: str = "/app/data/rule_config.json"):
        self.config_path = Path(config_path)
        self.params: Dict[str, Dict[str, float]] = {}
        self.adjustment_history: List[Dict] = []
        self._load_config()

    def _load_config(self):
        """Load configuration from file or use defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.params = data.get("params", {})
                    self.adjustment_history = data.get("history", [])
                    logger.info(f"Loaded rule config from {self.config_path}")

                    # Ensure all parameter changes are in history
                    self._ensure_history_complete()
            else:
                # Use defaults
                self.params = DEFAULT_RULE_PARAMS.copy()
                self._save_config()
                logger.info("Created default rule config")
        except Exception as e:
            logger.error(f"Failed to load rule config: {e}")
            self.params = DEFAULT_RULE_PARAMS.copy()

    def _ensure_history_complete(self):
        """
        Ensure all parameter differences from defaults are recorded in history.

        This catches parameters that were set directly in the config file
        without going through set_param().
        """
        # Build a set of already-recorded changes (pattern:param combinations)
        recorded_changes = set()
        for record in self.adjustment_history:
            pattern = record.get("pattern", "")
            param = record.get("parameter", "")
            if pattern and param:
                recorded_changes.add(f"{pattern}:{param}")

        # Find differences between current params and defaults
        missing_records = []
        for pattern, params in self.params.items():
            defaults = DEFAULT_RULE_PARAMS.get(pattern, {})

            for param_name, current_value in params.items():
                default_value = defaults.get(param_name)
                key = f"{pattern}:{param_name}"

                # If value differs from default and not in history, add it
                if default_value is not None and current_value != default_value:
                    if key not in recorded_changes:
                        missing_records.append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "pattern": pattern,
                            "parameter": param_name,
                            "old_value": default_value,
                            "new_value": current_value,
                            "reason": "initial_config",
                            "feedback_count": 0
                        })
                        logger.info(
                            f"Added missing history for {pattern}.{param_name}: "
                            f"{default_value} -> {current_value}"
                        )

        # Add missing records at the beginning of history
        if missing_records:
            self.adjustment_history = missing_records + self.adjustment_history
            self._save_config()
            logger.info(f"Added {len(missing_records)} missing history records")

    def _save_config(self):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "params": self.params,
                    "history": self.adjustment_history[-100:],  # Keep last 100
                    "updated_at": datetime.utcnow().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save rule config: {e}")

    def get_param(self, pattern: str, param_name: str) -> Optional[float]:
        """Get a specific rule parameter."""
        pattern_lower = pattern.lower()

        # First check custom params
        if pattern_lower in self.params:
            value = self.params[pattern_lower].get(param_name)
            if value is not None:
                return value

        # Fallback to defaults (also when param not in custom params)
        if pattern_lower in DEFAULT_RULE_PARAMS:
            return DEFAULT_RULE_PARAMS[pattern_lower].get(param_name)

        return None

    def get_pattern_params(self, pattern: str) -> Dict[str, float]:
        """Get all parameters for a pattern."""
        pattern_lower = pattern.lower()
        # Merge defaults with current params
        defaults = DEFAULT_RULE_PARAMS.get(pattern_lower, {})
        current = self.params.get(pattern_lower, {})
        return {**defaults, **current}

    def get_all_params(self) -> Dict[str, Dict[str, float]]:
        """Get all current parameters."""
        # Merge defaults with custom params
        result = {}
        for pattern in DEFAULT_RULE_PARAMS:
            result[pattern] = self.get_pattern_params(pattern)
        # Add any custom patterns not in defaults
        for pattern in self.params:
            if pattern not in result:
                result[pattern] = self.params[pattern]
        return result

    def set_param(
        self,
        pattern: str,
        param_name: str,
        value: float,
        reason: str = "",
        feedback_count: int = 0
    ) -> bool:
        """Set a specific rule parameter."""
        pattern_lower = pattern.lower()

        # Initialize pattern if not exists
        if pattern_lower not in self.params:
            self.params[pattern_lower] = {}

        # Get old value for history
        old_value = self.get_param(pattern_lower, param_name)

        # Set new value
        self.params[pattern_lower][param_name] = value

        # Record adjustment
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "pattern": pattern_lower,
            "parameter": param_name,
            "old_value": old_value,
            "new_value": value,
            "reason": reason,
            "feedback_count": feedback_count
        }
        self.adjustment_history.append(record)

        # Save
        self._save_config()

        logger.info(
            f"Rule param adjusted: {pattern_lower}.{param_name} = {old_value} -> {value} "
            f"(reason: {reason}, feedback_count: {feedback_count})"
        )

        return True

    def reset_param(self, pattern: str, param_name: str) -> bool:
        """Reset a parameter to its default value."""
        pattern_lower = pattern.lower()

        if pattern_lower in DEFAULT_RULE_PARAMS:
            default_value = DEFAULT_RULE_PARAMS[pattern_lower].get(param_name)
            if default_value is not None:
                return self.set_param(
                    pattern_lower,
                    param_name,
                    default_value,
                    reason="reset_to_default"
                )
        return False

    def reset_all(self) -> bool:
        """Reset all parameters to defaults."""
        self.params = DEFAULT_RULE_PARAMS.copy()
        self.adjustment_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "pattern": "all",
            "parameter": "all",
            "old_value": None,
            "new_value": None,
            "reason": "reset_all_to_default",
            "feedback_count": 0
        })
        self._save_config()
        logger.info("All rule parameters reset to defaults")
        return True

    def factory_reset(self) -> Dict[str, Any]:
        """
        Reset everything to factory defaults.

        Deletes all data files and resets to a clean state:
        - rule_config.json (reset to defaults)
        - pattern_feedback.json (deleted)
        - pattern_history.json (deleted)
        - claude_validations.json (deleted)
        - pending_validations.json (deleted)
        - backups/ directory (deleted)
        - In-Memory data from all services (cleared)

        Returns:
            Result of factory reset operation
        """
        import shutil

        data_dir = self.config_path.parent
        deleted_files = []
        memory_cleared = []
        errors = []

        # Files to delete
        files_to_delete = [
            "pattern_feedback.json",
            "pattern_history.json",
            "claude_validations.json",
            "pending_validations.json",
        ]

        # Delete individual files
        for filename in files_to_delete:
            file_path = data_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_files.append(filename)
                    logger.info(f"Factory reset: Deleted {filename}")
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
                    logger.error(f"Factory reset: Failed to delete {filename}: {e}")

        # Delete backups directory
        backup_dir = data_dir / "backups"
        if backup_dir.exists():
            try:
                shutil.rmtree(backup_dir)
                deleted_files.append("backups/")
                logger.info("Factory reset: Deleted backups directory")
            except Exception as e:
                errors.append(f"backups/: {str(e)}")
                logger.error(f"Factory reset: Failed to delete backups: {e}")

        # Reset rule_config.json to defaults
        try:
            self.params = {}  # Will be populated from defaults
            self.adjustment_history = [{
                "timestamp": datetime.utcnow().isoformat(),
                "pattern": "all",
                "parameter": "all",
                "old_value": None,
                "new_value": None,
                "reason": "factory_reset",
                "feedback_count": 0
            }]
            self._save_config()
            deleted_files.append("rule_config.json (reset)")
            logger.info("Factory reset: Reset rule_config.json to defaults")
        except Exception as e:
            errors.append(f"rule_config.json: {str(e)}")
            logger.error(f"Factory reset: Failed to reset rule_config.json: {e}")

        # Clear In-Memory data from all services
        try:
            from .pattern_history_service import pattern_history_service
            result = pattern_history_service.clear_memory()
            memory_cleared.append(f"pattern_history_service (was_running: {result.get('was_running', False)})")
            logger.info("Factory reset: Cleared pattern_history_service memory")
        except Exception as e:
            errors.append(f"pattern_history_service memory: {str(e)}")
            logger.error(f"Factory reset: Failed to clear pattern_history_service memory: {e}")

        try:
            from .auto_optimization_service import auto_optimization_service
            result = auto_optimization_service.clear_memory()
            memory_cleared.append(
                f"auto_optimization_service (pending: {result.get('pending_validations_cleared', 0)}, "
                f"history: {result.get('optimization_history_cleared', 0)})"
            )
            logger.info("Factory reset: Cleared auto_optimization_service memory")
        except Exception as e:
            errors.append(f"auto_optimization_service memory: {str(e)}")
            logger.error(f"Factory reset: Failed to clear auto_optimization_service memory: {e}")

        try:
            from .claude_validator_service import claude_validator_service
            result = claude_validator_service.clear_memory()
            memory_cleared.append(
                f"claude_validator_service (history: {result.get('validation_history_cleared', 0)}, "
                f"cache: {result.get('validation_cache_cleared', 0)})"
            )
            logger.info("Factory reset: Cleared claude_validator_service memory")
        except Exception as e:
            errors.append(f"claude_validator_service memory: {str(e)}")
            logger.error(f"Factory reset: Failed to clear claude_validator_service memory: {e}")

        if errors:
            return {
                "success": False,
                "deleted": deleted_files,
                "memory_cleared": memory_cleared,
                "errors": errors,
                "message": "Factory reset completed with errors"
            }

        return {
            "success": True,
            "deleted": deleted_files,
            "memory_cleared": memory_cleared,
            "message": "Factory reset completed successfully. All data and in-memory state have been cleared."
        }

    def get_adjustment_history(self, limit: int = 50) -> List[Dict]:
        """Get recent parameter adjustment history, sorted newest first."""
        # Sort by timestamp descending (newest first)
        sorted_history = sorted(
            self.adjustment_history,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return sorted_history[:limit]

    def generate_recommendations(
        self,
        feedback_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter adjustment recommendations based on feedback.

        Args:
            feedback_stats: Statistics from feedback/reason-statistics endpoint

        Returns:
            List of recommended adjustments
        """
        recommendations = []

        by_pattern_and_reason = feedback_stats.get("by_pattern_and_reason", {})

        # Mapping from reason categories to parameter adjustments
        ADJUSTMENT_MAP = {
            "body_too_large": {
                "params": ["body_to_range_ratio", "body_to_avg_ratio", "body_max_ratio"],
                "direction": "decrease",
                "step": 0.02,  # Decrease by 2%
            },
            "body_too_small": {
                "params": ["body_min_ratio", "prev_body_min_ratio", "curr_body_min_ratio"],
                "direction": "increase",
                "step": 0.05,
            },
            "upper_shadow_too_short": {
                "params": ["upper_shadow_min_ratio"],
                "direction": "decrease",
                "step": 0.05,
            },
            "upper_shadow_too_long": {
                "params": ["upper_shadow_max_ratio"],
                "direction": "decrease",
                "step": 0.02,
            },
            "lower_shadow_too_short": {
                "params": ["lower_shadow_min_ratio"],
                "direction": "decrease",
                "step": 0.05,
            },
            "lower_shadow_too_long": {
                "params": ["lower_shadow_max_ratio"],
                "direction": "decrease",
                "step": 0.02,
            },
            "not_fully_engulfing": {
                "params": ["size_ratio_min"],
                "direction": "increase",
                "step": 0.1,
            },
            "engulfing_too_small": {
                "params": ["size_ratio_min", "min_pct_move"],
                "direction": "increase",
                "step": 0.1,
            },
            "wrong_trend_context": {
                "params": ["uptrend_threshold", "downtrend_threshold"],
                "direction": "increase_abs",  # Increase absolute value
                "step": 0.5,
            },
            "no_prior_trend": {
                "params": ["lookback_candles"],
                "direction": "increase",
                "step": 2,
            },
            "missing_gap": {
                "params": ["gap_tolerance"],
                "direction": "decrease",
                "step": 0.001,
            },
            "false_positive": {
                "params": ["body_max_ratio", "body_to_range_ratio"],
                "direction": "decrease",
                "step": 0.02,
            },
        }

        # Analyze each pattern:reason combination
        for key, count in sorted(by_pattern_and_reason.items(), key=lambda x: -x[1]):
            if count < 3:  # Minimum threshold
                continue

            parts = key.split(":", 1)
            if len(parts) != 2:
                continue

            pattern, reason = parts

            if reason not in ADJUSTMENT_MAP:
                continue

            adjustment_info = ADJUSTMENT_MAP[reason]
            pattern_params = self.get_pattern_params(pattern)

            for param_name in adjustment_info["params"]:
                current_value = pattern_params.get(param_name)
                if current_value is None:
                    continue

                # Calculate recommended new value
                step = adjustment_info["step"]
                direction = adjustment_info["direction"]

                if direction == "decrease":
                    new_value = max(0.01, current_value - step * (count // 3))
                elif direction == "increase":
                    new_value = current_value + step * (count // 3)
                elif direction == "increase_abs":
                    new_value = current_value + step * (count // 3) if current_value >= 0 else current_value - step * (count // 3)
                else:
                    continue

                # Round to reasonable precision
                new_value = round(new_value, 4)

                if new_value != current_value:
                    recommendations.append({
                        "pattern": pattern,
                        "parameter": param_name,
                        "current_value": current_value,
                        "recommended_value": new_value,
                        "reason": reason,
                        "feedback_count": count,
                        "confidence": min(0.9, 0.5 + count * 0.05),  # Higher count = more confidence
                        "priority": "high" if count >= 5 else "medium" if count >= 3 else "low"
                    })

        # Sort by feedback count (most important first)
        recommendations.sort(key=lambda x: -x["feedback_count"])

        return recommendations[:20]  # Return top 20

    def apply_recommendation(
        self,
        pattern: str,
        param_name: str,
        new_value: float,
        reason: str,
        feedback_count: int
    ) -> bool:
        """Apply a single recommendation."""
        return self.set_param(
            pattern=pattern,
            param_name=param_name,
            value=new_value,
            reason=f"feedback_recommendation:{reason}",
            feedback_count=feedback_count
        )

    # ==================== Backup/Restore ====================

    def create_backup(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a backup of current rule configuration.

        Args:
            name: Optional backup name/description

        Returns:
            Backup data including metadata
        """
        backup_name = name or f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        backup_data = {
            "name": backup_name,
            "created_at": datetime.utcnow().isoformat(),
            "params": self.params.copy(),
            "history": self.adjustment_history.copy(),
            "version": "1.0"
        }

        # Save backup to file
        backup_dir = self.config_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_filename = f"rule_config_{backup_name}.json"
        backup_path = backup_dir / backup_filename

        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Created backup: {backup_path}")

            return {
                "success": True,
                "name": backup_name,
                "path": str(backup_path),
                "created_at": backup_data["created_at"],
                "param_count": sum(len(p) for p in self.params.values()),
                "history_count": len(self.adjustment_history)
            }
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.

        Returns:
            List of backup metadata
        """
        backup_dir = self.config_path.parent / "backups"
        backups = []

        if not backup_dir.exists():
            return backups

        for backup_file in sorted(backup_dir.glob("rule_config_*.json"), reverse=True):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                backups.append({
                    "name": data.get("name", backup_file.stem),
                    "filename": backup_file.name,
                    "created_at": data.get("created_at", ""),
                    "param_count": sum(len(p) for p in data.get("params", {}).values()),
                    "history_count": len(data.get("history", [])),
                    "size_bytes": backup_file.stat().st_size
                })
            except Exception as e:
                logger.warning(f"Failed to read backup {backup_file}: {e}")

        return backups

    def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Restore rule configuration from a backup.

        Args:
            backup_name: Name or filename of the backup to restore

        Returns:
            Result of restore operation
        """
        backup_dir = self.config_path.parent / "backups"

        # Find the backup file
        backup_path = None
        for pattern in [
            backup_dir / f"rule_config_{backup_name}.json",
            backup_dir / backup_name,
            backup_dir / f"{backup_name}.json"
        ]:
            if pattern.exists():
                backup_path = pattern
                break

        if not backup_path:
            return {
                "success": False,
                "error": f"Backup not found: {backup_name}"
            }

        try:
            # Read backup
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Store current config as auto-backup before restore
            self.create_backup(name="auto_before_restore")

            # Restore params and history
            self.params = backup_data.get("params", {})
            self.adjustment_history = backup_data.get("history", [])

            # Add restore record to history
            self.adjustment_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "pattern": "all",
                "parameter": "all",
                "old_value": None,
                "new_value": None,
                "reason": f"restored_from_backup:{backup_data.get('name', backup_name)}",
                "feedback_count": 0
            })

            # Save restored config
            self._save_config()

            logger.info(f"Restored from backup: {backup_path}")

            return {
                "success": True,
                "restored_from": backup_data.get("name", backup_name),
                "backup_created_at": backup_data.get("created_at", ""),
                "param_count": sum(len(p) for p in self.params.values()),
                "history_count": len(self.adjustment_history)
            }

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def delete_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Delete a backup file.

        Args:
            backup_name: Name or filename of the backup to delete

        Returns:
            Result of delete operation
        """
        backup_dir = self.config_path.parent / "backups"

        # Find the backup file
        backup_path = None
        for pattern in [
            backup_dir / f"rule_config_{backup_name}.json",
            backup_dir / backup_name,
            backup_dir / f"{backup_name}.json"
        ]:
            if pattern.exists():
                backup_path = pattern
                break

        if not backup_path:
            return {
                "success": False,
                "error": f"Backup not found: {backup_name}"
            }

        try:
            backup_path.unlink()
            logger.info(f"Deleted backup: {backup_path}")
            return {
                "success": True,
                "deleted": backup_name
            }
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Get a specific backup's data for export.

        Args:
            backup_name: Name or filename of the backup

        Returns:
            Backup data or error
        """
        backup_dir = self.config_path.parent / "backups"

        # Find the backup file
        backup_path = None
        for pattern in [
            backup_dir / f"rule_config_{backup_name}.json",
            backup_dir / backup_name,
            backup_dir / f"{backup_name}.json"
        ]:
            if pattern.exists():
                backup_path = pattern
                break

        if not backup_path:
            return {
                "success": False,
                "error": f"Backup not found: {backup_name}"
            }

        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                "success": True,
                "data": data
            }
        except Exception as e:
            logger.error(f"Failed to read backup: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def export_config(self) -> Dict[str, Any]:
        """
        Export current configuration as downloadable data.

        Returns:
            Complete configuration data
        """
        return {
            "exported_at": datetime.utcnow().isoformat(),
            "params": self.params,
            "history": self.adjustment_history,
            "defaults": DEFAULT_RULE_PARAMS,
            "version": "1.0"
        }

    def import_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import configuration from uploaded data.

        Args:
            config_data: Configuration data to import

        Returns:
            Result of import operation
        """
        try:
            # Validate structure
            if "params" not in config_data:
                return {
                    "success": False,
                    "error": "Invalid config: missing 'params' field"
                }

            # Create backup before import
            self.create_backup(name="auto_before_import")

            # Import params
            self.params = config_data.get("params", {})

            # Optionally import history
            if "history" in config_data:
                self.adjustment_history = config_data["history"]

            # Add import record
            self.adjustment_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "pattern": "all",
                "parameter": "all",
                "old_value": None,
                "new_value": None,
                "reason": "imported_config",
                "feedback_count": 0
            })

            self._save_config()

            logger.info("Imported configuration successfully")

            return {
                "success": True,
                "param_count": sum(len(p) for p in self.params.values()),
                "history_count": len(self.adjustment_history)
            }

        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global singleton
rule_config_service = RuleConfigService()
