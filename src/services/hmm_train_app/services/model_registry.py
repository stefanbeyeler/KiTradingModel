"""Model Registry for HMM models with versioning support.

Handles:
- Version management for HMM and Scorer models
- Metadata storage and retrieval
- Production/candidate model management
- Cleanup of old versions
- Symlink management for current production models

Directory Structure:
    /app/data/models/hmm/
    ├── registry.json           # Master registry
    ├── versions/
    │   ├── v_20240115_143052/  # Version directory
    │   │   ├── hmm_BTCUSD.pkl
    │   │   ├── hmm_EURUSD.pkl
    │   │   └── scorer_lightgbm.pkl
    │   └── v_20240116_091530/
    │       └── ...
    ├── current/                # Symlinks to production versions
    │   ├── hmm_BTCUSD.pkl -> ../versions/v_xxx/hmm_BTCUSD.pkl
    │   └── scorer_lightgbm.pkl -> ../versions/v_xxx/scorer_lightgbm.pkl
    └── validation_data/        # Cached validation datasets
        └── BTCUSD_H1_val.npz
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
from loguru import logger


class ModelStatus(str, Enum):
    """Model version status."""
    CANDIDATE = "candidate"      # Newly trained, awaiting validation
    PRODUCTION = "production"    # Currently deployed
    ARCHIVED = "archived"        # Previously used, kept for history
    REJECTED = "rejected"        # Failed validation, not deployed


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_type: str                     # "hmm" or "scorer"
    symbol: Optional[str]               # For HMM: symbol, for Scorer: None
    created_at: str                     # ISO timestamp
    training_job_id: str                # Link to training job

    # Training info
    samples_used: int = 0
    training_duration_seconds: float = 0.0
    timeframe: str = "H1"

    # Validation metrics
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Status
    status: ModelStatus = ModelStatus.CANDIDATE
    promoted_at: Optional[str] = None   # When promoted to production
    rejected_at: Optional[str] = None   # When rejected
    rejected_reason: Optional[str] = None

    # Files
    model_path: str = ""                # Relative path from versions/

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        data = data.copy()
        if "status" in data:
            data["status"] = ModelStatus(data["status"])
        return cls(**data)


@dataclass
class RegistryState:
    """Complete registry state."""
    schema_version: str = "1.0"
    last_updated: str = ""
    versions: Dict[str, Dict[str, ModelVersion]] = field(default_factory=dict)  # version_id -> model_key -> ModelVersion
    production: Dict[str, str] = field(default_factory=dict)  # model_key -> version_id
    cleanup_policy: Dict[str, int] = field(default_factory=lambda: {"keep_versions": 5, "keep_days": 30})


class ModelRegistry:
    """
    Central registry for managing HMM model versions.

    Provides:
    - Version creation and tracking
    - Production/candidate model management
    - Symlink-based current model access
    - Automatic cleanup of old versions
    """

    REGISTRY_FILE = "registry.json"
    VERSIONS_DIR = "versions"
    CURRENT_DIR = "current"
    VALIDATION_DIR = "validation_data"

    def __init__(self, base_path: str = "/app/data/models/hmm"):
        self._base_path = Path(base_path)
        self._versions_path = self._base_path / self.VERSIONS_DIR
        self._current_path = self._base_path / self.CURRENT_DIR
        self._validation_path = self._base_path / self.VALIDATION_DIR
        self._registry_file = self._base_path / self.REGISTRY_FILE

        # Ensure directories exist
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._versions_path.mkdir(exist_ok=True)
        self._current_path.mkdir(exist_ok=True)
        self._validation_path.mkdir(exist_ok=True)

        # Load or initialize registry
        self._state = self._load_registry()

        logger.info(f"ModelRegistry initialized (base: {base_path})")

    def _load_registry(self) -> RegistryState:
        """Load registry from disk or create new."""
        try:
            if self._registry_file.exists():
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)

                state = RegistryState(
                    schema_version=data.get("schema_version", "1.0"),
                    last_updated=data.get("last_updated", ""),
                    production=data.get("production", {}),
                    cleanup_policy=data.get("cleanup_policy", {"keep_versions": 5, "keep_days": 30})
                )

                # Load versions
                for version_id, models in data.get("versions", {}).items():
                    state.versions[version_id] = {}
                    for model_key, model_data in models.items():
                        state.versions[version_id][model_key] = ModelVersion.from_dict(model_data)

                logger.info(f"Loaded registry with {len(state.versions)} versions")
                return state

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

        return RegistryState(last_updated=datetime.now(timezone.utc).isoformat())

    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {
                "schema_version": self._state.schema_version,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "versions": {},
                "production": self._state.production,
                "cleanup_policy": self._state.cleanup_policy
            }

            for version_id, models in self._state.versions.items():
                data["versions"][version_id] = {
                    model_key: mv.to_dict() for model_key, mv in models.items()
                }

            with open(self._registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _get_model_key(self, model_type: str, symbol: Optional[str]) -> str:
        """Generate model key for registry lookup."""
        if model_type == "hmm" and symbol:
            return f"hmm_{symbol}"
        elif model_type == "scorer":
            return "scorer"
        else:
            return f"{model_type}_{symbol or 'global'}"

    def _get_model_filename(self, model_type: str, symbol: Optional[str]) -> str:
        """Generate model filename."""
        if model_type == "hmm" and symbol:
            return f"hmm_{symbol}.pkl"
        elif model_type == "scorer":
            return "scorer_lightgbm.pkl"
        else:
            return f"{model_type}_{symbol or 'global'}.pkl"

    # =========================================================================
    # Version Management
    # =========================================================================

    def create_version(self, job_id: str) -> str:
        """
        Create a new version directory for a training job.

        Args:
            job_id: Training job ID

        Returns:
            version_id: Unique version identifier
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        version_id = f"v_{timestamp}"

        # Create version directory
        version_path = self._versions_path / version_id
        version_path.mkdir(parents=True, exist_ok=True)

        # Initialize version entry
        self._state.versions[version_id] = {}
        self._save_registry()

        logger.info(f"Created version {version_id} for job {job_id}")
        return version_id

    def get_version_path(self, version_id: str) -> Path:
        """Get the directory path for a version."""
        return self._versions_path / version_id

    def register_model(
        self,
        version_id: str,
        model_type: str,
        symbol: Optional[str],
        training_job_id: str,
        samples_used: int = 0,
        training_duration: float = 0.0,
        timeframe: str = "H1",
        metrics: Optional[Dict[str, float]] = None,
        model_path: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a trained model in the registry.

        Args:
            version_id: Version this model belongs to
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models, None for scorer
            training_job_id: Associated training job
            samples_used: Number of training samples
            training_duration: Training time in seconds
            timeframe: Training timeframe
            metrics: Validation metrics dictionary
            model_path: Path to model file (relative to version dir)

        Returns:
            ModelVersion: Registered model metadata
        """
        model_key = self._get_model_key(model_type, symbol)

        if version_id not in self._state.versions:
            self._state.versions[version_id] = {}

        # Determine model path
        if model_path is None:
            filename = self._get_model_filename(model_type, symbol)
            model_path = f"{version_id}/{filename}"

        model_version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            symbol=symbol,
            created_at=datetime.now(timezone.utc).isoformat(),
            training_job_id=training_job_id,
            samples_used=samples_used,
            training_duration_seconds=training_duration,
            timeframe=timeframe,
            validation_metrics=metrics or {},
            status=ModelStatus.CANDIDATE,
            model_path=model_path
        )

        self._state.versions[version_id][model_key] = model_version
        self._save_registry()

        logger.info(f"Registered model {model_key} in version {version_id}")
        return model_version

    def get_version(self, version_id: str) -> Optional[Dict[str, ModelVersion]]:
        """Get all models for a version."""
        return self._state.versions.get(version_id)

    def get_model(self, version_id: str, model_type: str, symbol: Optional[str]) -> Optional[ModelVersion]:
        """Get specific model from a version."""
        model_key = self._get_model_key(model_type, symbol)
        version = self._state.versions.get(version_id, {})
        return version.get(model_key)

    def get_current_production(self, model_type: str, symbol: Optional[str]) -> Optional[ModelVersion]:
        """
        Get the current production model.

        Args:
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models

        Returns:
            ModelVersion if production model exists, None otherwise
        """
        model_key = self._get_model_key(model_type, symbol)
        version_id = self._state.production.get(model_key)

        if version_id and version_id in self._state.versions:
            return self._state.versions[version_id].get(model_key)

        return None

    def get_production_model_path(self, model_type: str, symbol: Optional[str]) -> Optional[Path]:
        """Get the file path for current production model."""
        production = self.get_current_production(model_type, symbol)
        if production:
            return self._versions_path / production.model_path

        # Fallback: check current/ directory symlink
        filename = self._get_model_filename(model_type, symbol)
        current_path = self._current_path / filename
        if current_path.exists():
            return current_path.resolve()

        return None

    # =========================================================================
    # Promotion / Rollback
    # =========================================================================

    def promote_to_production(
        self,
        version_id: str,
        model_type: str,
        symbol: Optional[str]
    ) -> bool:
        """
        Promote a candidate model to production.

        Updates symlinks in current/ directory.

        Args:
            version_id: Version to promote
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models

        Returns:
            True if successful
        """
        model_key = self._get_model_key(model_type, symbol)

        # Validate version exists
        if version_id not in self._state.versions:
            logger.error(f"Version {version_id} not found")
            return False

        model_version = self._state.versions[version_id].get(model_key)
        if not model_version:
            logger.error(f"Model {model_key} not found in version {version_id}")
            return False

        # Archive current production
        old_version_id = self._state.production.get(model_key)
        if old_version_id and old_version_id in self._state.versions:
            old_model = self._state.versions[old_version_id].get(model_key)
            if old_model:
                old_model.status = ModelStatus.ARCHIVED

        # Update status
        model_version.status = ModelStatus.PRODUCTION
        model_version.promoted_at = datetime.now(timezone.utc).isoformat()

        # Update production pointer
        self._state.production[model_key] = version_id

        # Update symlink
        self._update_symlink(model_type, symbol, version_id)

        self._save_registry()

        logger.info(f"Promoted {model_key} from version {version_id} to production")
        return True

    def _update_symlink(self, model_type: str, symbol: Optional[str], version_id: str):
        """Update symlink in current/ to point to version model."""
        filename = self._get_model_filename(model_type, symbol)
        symlink_path = self._current_path / filename
        target_path = self._versions_path / version_id / filename

        # Remove existing symlink or file
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        # Create relative symlink
        relative_target = os.path.relpath(target_path, self._current_path)
        symlink_path.symlink_to(relative_target)

        logger.debug(f"Updated symlink: {symlink_path} -> {relative_target}")

    def rollback_to_version(
        self,
        model_type: str,
        symbol: Optional[str],
        target_version: str,
        reason: str = "Manual rollback"
    ) -> bool:
        """
        Rollback to a previous version.

        Args:
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models
            target_version: Version to rollback to
            reason: Reason for rollback

        Returns:
            True if successful
        """
        model_key = self._get_model_key(model_type, symbol)

        # Validate target version
        if target_version not in self._state.versions:
            logger.error(f"Target version {target_version} not found")
            return False

        target_model = self._state.versions[target_version].get(model_key)
        if not target_model:
            logger.error(f"Model {model_key} not found in version {target_version}")
            return False

        # Demote current production
        current_version_id = self._state.production.get(model_key)
        if current_version_id and current_version_id in self._state.versions:
            current_model = self._state.versions[current_version_id].get(model_key)
            if current_model:
                current_model.status = ModelStatus.ARCHIVED

        # Promote target
        target_model.status = ModelStatus.PRODUCTION
        target_model.promoted_at = datetime.now(timezone.utc).isoformat()
        self._state.production[model_key] = target_version

        # Update symlink
        self._update_symlink(model_type, symbol, target_version)

        self._save_registry()

        logger.info(f"Rolled back {model_key} to version {target_version}: {reason}")
        return True

    def reject_version(
        self,
        version_id: str,
        model_type: str,
        symbol: Optional[str],
        reason: str
    ) -> bool:
        """
        Mark a model version as rejected.

        Args:
            version_id: Version to reject
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models
            reason: Rejection reason

        Returns:
            True if successful
        """
        model_key = self._get_model_key(model_type, symbol)

        if version_id not in self._state.versions:
            return False

        model_version = self._state.versions[version_id].get(model_key)
        if not model_version:
            return False

        model_version.status = ModelStatus.REJECTED
        model_version.rejected_at = datetime.now(timezone.utc).isoformat()
        model_version.rejected_reason = reason

        self._save_registry()

        logger.warning(f"Rejected {model_key} version {version_id}: {reason}")
        return True

    def update_metrics(
        self,
        version_id: str,
        model_type: str,
        symbol: Optional[str],
        metrics: Dict[str, float]
    ) -> bool:
        """Update validation metrics for a model."""
        model_key = self._get_model_key(model_type, symbol)

        if version_id not in self._state.versions:
            return False

        model_version = self._state.versions[version_id].get(model_key)
        if not model_version:
            return False

        model_version.validation_metrics.update(metrics)
        self._save_registry()

        return True

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_old_versions(self, keep_last_n: Optional[int] = None) -> int:
        """
        Clean up old model versions.

        Keeps:
        - All production models
        - Last N versions per model
        - Versions younger than cleanup_policy.keep_days

        Args:
            keep_last_n: Override for number of versions to keep

        Returns:
            Number of versions deleted
        """
        keep_n = keep_last_n or self._state.cleanup_policy.get("keep_versions", 5)
        keep_days = self._state.cleanup_policy.get("keep_days", 30)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=keep_days)

        # Get production versions (never delete)
        production_versions = set(self._state.production.values())

        # Group versions by model_key
        versions_by_model: Dict[str, List[tuple]] = {}  # model_key -> [(version_id, created_at)]

        for version_id, models in self._state.versions.items():
            for model_key, model_version in models.items():
                if model_key not in versions_by_model:
                    versions_by_model[model_key] = []
                versions_by_model[model_key].append((version_id, model_version.created_at))

        # Determine versions to delete
        versions_to_delete = set()

        for model_key, versions in versions_by_model.items():
            # Sort by creation date (newest first)
            versions.sort(key=lambda x: x[1], reverse=True)

            for i, (version_id, created_at) in enumerate(versions):
                # Skip production versions
                if version_id in production_versions:
                    continue

                # Keep last N
                if i < keep_n:
                    continue

                # Keep recent versions
                try:
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if created > cutoff_date:
                        continue
                except:
                    pass

                versions_to_delete.add(version_id)

        # Delete versions
        deleted = 0
        for version_id in versions_to_delete:
            if self._delete_version(version_id):
                deleted += 1

        if deleted > 0:
            self._save_registry()
            logger.info(f"Cleaned up {deleted} old versions")

        return deleted

    def _delete_version(self, version_id: str) -> bool:
        """Delete a version and its files."""
        try:
            # Remove from registry
            if version_id in self._state.versions:
                del self._state.versions[version_id]

            # Remove files
            version_path = self._versions_path / version_id
            if version_path.exists():
                shutil.rmtree(version_path)

            return True
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    # =========================================================================
    # Queries
    # =========================================================================

    def list_versions(
        self,
        model_type: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        limit: int = 20
    ) -> List[ModelVersion]:
        """
        List model versions with optional filters.

        Args:
            model_type: Filter by model type
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum results

        Returns:
            List of ModelVersion sorted by creation date (newest first)
        """
        results = []

        for version_id, models in self._state.versions.items():
            for model_key, model_version in models.items():
                # Apply filters
                if model_type and model_version.model_type != model_type:
                    continue
                if symbol is not None and model_version.symbol != symbol:
                    continue
                if status and model_version.status != status:
                    continue

                results.append(model_version)

        # Sort by creation date
        results.sort(key=lambda x: x.created_at, reverse=True)

        return results[:limit]

    def get_production_versions(self) -> Dict[str, ModelVersion]:
        """Get all currently deployed production models."""
        result = {}

        for model_key, version_id in self._state.production.items():
            if version_id in self._state.versions:
                model = self._state.versions[version_id].get(model_key)
                if model:
                    result[model_key] = model

        return result

    def get_version_history(self, model_type: str, symbol: Optional[str], limit: int = 10) -> List[ModelVersion]:
        """Get version history for a specific model."""
        model_key = self._get_model_key(model_type, symbol)

        results = []
        for version_id, models in self._state.versions.items():
            if model_key in models:
                results.append(models[model_key])

        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_models = sum(len(models) for models in self._state.versions.values())

        status_counts = {status.value: 0 for status in ModelStatus}
        for models in self._state.versions.values():
            for model in models.values():
                status_counts[model.status.value] += 1

        return {
            "total_versions": len(self._state.versions),
            "total_models": total_models,
            "production_models": len(self._state.production),
            "status_counts": status_counts,
            "cleanup_policy": self._state.cleanup_policy
        }


# Global singleton
model_registry = ModelRegistry()
