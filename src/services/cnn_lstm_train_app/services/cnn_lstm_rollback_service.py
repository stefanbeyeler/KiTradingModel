"""
CNN-LSTM Rollback Service.

Provides safe model deployment with rollback capability:
- Version tracking of all deployed models
- Quick rollback to previous versions
- Automatic rollback on critical drift
- Hot-reload notification to inference service
"""

import os
import json
import shutil
import httpx
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class DeploymentStatus(str, Enum):
    """Status of a model deployment."""
    ACTIVE = "active"
    PREVIOUS = "previous"
    CANDIDATE = "candidate"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class RollbackReason(str, Enum):
    """Reason for rollback."""
    MANUAL = "manual"
    CRITICAL_DRIFT = "critical_drift"
    VALIDATION_FAILED = "validation_failed"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR = "error"


@dataclass
class ModelVersion:
    """A tracked model version."""
    version_id: str
    model_path: str
    training_type: str  # "full" or "incremental"
    status: DeploymentStatus = DeploymentStatus.CANDIDATE
    created_at: str = ""
    deployed_at: Optional[str] = None
    rolled_back_at: Optional[str] = None
    rollback_reason: Optional[str] = None
    metrics: Dict = field(default_factory=dict)
    validation_result: Optional[Dict] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_path": self.model_path,
            "training_type": self.training_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "deployed_at": self.deployed_at,
            "rolled_back_at": self.rolled_back_at,
            "rollback_reason": self.rollback_reason,
            "metrics": self.metrics,
            "validation_result": self.validation_result,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            model_path=data["model_path"],
            training_type=data.get("training_type", "full"),
            status=DeploymentStatus(data.get("status", "candidate")),
            created_at=data.get("created_at", ""),
            deployed_at=data.get("deployed_at"),
            rolled_back_at=data.get("rolled_back_at"),
            rollback_reason=data.get("rollback_reason"),
            metrics=data.get("metrics", {}),
            validation_result=data.get("validation_result"),
            notes=data.get("notes", "")
        )


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    timestamp: str
    from_version: str
    to_version: str
    reason: RollbackReason
    triggered_by: str  # "automatic" or "manual"
    success: bool
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason.value,
            "triggered_by": self.triggered_by,
            "success": self.success,
            "notes": self.notes
        }


class CNNLSTMRollbackService:
    """
    Service for model versioning and rollback.

    Features:
    - Track all deployed model versions
    - Quick rollback to any previous version
    - Automatic rollback on critical drift detection
    - Safe deployment with validation
    - Hot-reload notification to inference service
    """

    VERSION_FILE = "data/models/cnn-lstm/version_history.json"
    MODEL_DIR = "data/models/cnn-lstm"
    LATEST_MODEL_LINK = "data/models/cnn-lstm/latest.pt"
    BACKUP_DIR = "data/models/cnn-lstm/backups"
    MAX_VERSIONS_TO_KEEP = 10
    MAX_ROLLBACK_HISTORY = 50

    def __init__(self):
        """Initialize rollback service."""
        self._versions: List[ModelVersion] = []
        self._rollback_history: List[RollbackEvent] = []
        self._current_version: Optional[str] = None
        self._cnn_lstm_service_url = os.getenv(
            "CNN_LSTM_SERVICE_URL",
            "http://trading-cnn-lstm:3007"
        )

        # Ensure directories exist
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)

        self._load_versions()

    def _load_versions(self) -> None:
        """Load version history from disk."""
        try:
            if os.path.exists(self.VERSION_FILE):
                with open(self.VERSION_FILE, 'r') as f:
                    data = json.load(f)
                    self._versions = [
                        ModelVersion.from_dict(v)
                        for v in data.get("versions", [])
                    ]
                    self._current_version = data.get("current_version")
                    self._rollback_history = [
                        RollbackEvent(
                            timestamp=e["timestamp"],
                            from_version=e["from_version"],
                            to_version=e["to_version"],
                            reason=RollbackReason(e.get("reason", "manual")),
                            triggered_by=e.get("triggered_by", "manual"),
                            success=e.get("success", True),
                            notes=e.get("notes", "")
                        )
                        for e in data.get("rollback_history", [])
                    ]

                logger.info(f"Loaded {len(self._versions)} model versions, {len(self._rollback_history)} rollback events")
        except Exception as e:
            logger.warning(f"Could not load version history: {e}")

    def _save_versions(self) -> None:
        """Save version history to disk."""
        try:
            os.makedirs(os.path.dirname(self.VERSION_FILE), exist_ok=True)
            with open(self.VERSION_FILE, 'w') as f:
                json.dump({
                    "versions": [v.to_dict() for v in self._versions],
                    "current_version": self._current_version,
                    "rollback_history": [e.to_dict() for e in self._rollback_history[-self.MAX_ROLLBACK_HISTORY:]],
                    "updated_at": datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save version history: {e}")

    def register_model(
        self,
        model_path: str,
        training_type: str = "full",
        metrics: Optional[Dict] = None,
        validation_result: Optional[Dict] = None,
        notes: str = ""
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_path: Path to the model file
            training_type: "full" or "incremental"
            metrics: Training metrics
            validation_result: Validation results
            notes: Optional notes

        Returns:
            The created model version
        """
        filename = os.path.basename(model_path)
        version_id = filename.replace(".pt", "")

        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            training_type=training_type,
            status=DeploymentStatus.CANDIDATE,
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
            validation_result=validation_result,
            notes=notes
        )

        self._versions.append(version)
        self._cleanup_old_versions()
        self._save_versions()

        logger.info(f"Registered new model version: {version_id}")
        return version

    async def deploy_model(
        self,
        version_id: str,
        force: bool = False
    ) -> Dict:
        """
        Deploy a model version.

        Args:
            version_id: The version to deploy
            force: Skip validation checks

        Returns:
            Deployment result
        """
        version = self._get_version(version_id)
        if not version:
            return {
                "status": "failed",
                "message": f"Version {version_id} not found"
            }

        if not os.path.exists(version.model_path):
            return {
                "status": "failed",
                "message": f"Model file not found: {version.model_path}"
            }

        # Backup current model if exists
        previous_version_id = self._current_version
        if self._current_version:
            current = self._get_version(self._current_version)
            if current:
                await self._backup_model(current)
                current.status = DeploymentStatus.PREVIOUS

        # Update symlink
        try:
            if os.path.exists(self.LATEST_MODEL_LINK):
                os.remove(self.LATEST_MODEL_LINK)

            rel_path = os.path.relpath(version.model_path, os.path.dirname(self.LATEST_MODEL_LINK))
            os.symlink(rel_path, self.LATEST_MODEL_LINK)
        except Exception as e:
            logger.error(f"Failed to update symlink: {e}")
            return {
                "status": "failed",
                "message": f"Failed to update symlink: {e}"
            }

        # Update version status
        version.status = DeploymentStatus.ACTIVE
        version.deployed_at = datetime.utcnow().isoformat()
        self._current_version = version_id

        # Notify inference service
        reload_success = await self._notify_inference_service(version.model_path)

        self._save_versions()

        logger.info(f"Deployed model version: {version_id}")

        return {
            "status": "deployed",
            "version_id": version_id,
            "model_path": version.model_path,
            "inference_reloaded": reload_success,
            "previous_version": previous_version_id
        }

    async def rollback(
        self,
        target_version_id: Optional[str] = None,
        reason: RollbackReason = RollbackReason.MANUAL,
        triggered_by: str = "manual",
        notes: str = ""
    ) -> Dict:
        """
        Rollback to a previous model version.

        Args:
            target_version_id: Specific version to rollback to.
                              If None, rolls back to previous version.
            reason: Reason for the rollback
            triggered_by: "manual" or "automatic"
            notes: Optional notes

        Returns:
            Rollback result
        """
        if target_version_id:
            target = self._get_version(target_version_id)
        else:
            target = self._get_previous_version()

        if not target:
            return {
                "status": "failed",
                "message": "No previous version available for rollback"
            }

        from_version = self._current_version

        # Mark current version as rolled back
        if self._current_version:
            current = self._get_version(self._current_version)
            if current:
                current.status = DeploymentStatus.ROLLED_BACK
                current.rolled_back_at = datetime.utcnow().isoformat()
                current.rollback_reason = reason.value

        # Deploy the target version
        result = await self.deploy_model(target.version_id)

        success = result.get("status") == "deployed"

        # Record rollback event
        event = RollbackEvent(
            timestamp=datetime.utcnow().isoformat(),
            from_version=from_version or "unknown",
            to_version=target.version_id,
            reason=reason,
            triggered_by=triggered_by,
            success=success,
            notes=notes
        )
        self._rollback_history.append(event)
        self._save_versions()

        if success:
            result["rollback_from"] = from_version
            result["rollback_reason"] = reason.value
            logger.info(f"Rolled back from {from_version} to {target.version_id} (reason: {reason.value})")

        return result

    async def auto_rollback_on_drift(self, drift_severity: str, drift_details: Dict) -> Dict:
        """
        Automatically rollback if drift is critical.

        Args:
            drift_severity: Severity level (none, low, medium, high, critical)
            drift_details: Details about the detected drift

        Returns:
            Rollback result or skip message
        """
        if drift_severity != "critical":
            return {
                "status": "skipped",
                "message": f"Drift severity '{drift_severity}' does not require automatic rollback"
            }

        logger.warning(f"Critical drift detected - triggering automatic rollback")

        return await self.rollback(
            target_version_id=None,  # Rollback to previous
            reason=RollbackReason.CRITICAL_DRIFT,
            triggered_by="automatic",
            notes=f"Auto-rollback due to critical drift: {drift_details.get('recommendation', 'Unknown')}"
        )

    async def _backup_model(self, version: ModelVersion) -> bool:
        """Backup a model to the backup directory."""
        try:
            backup_path = os.path.join(
                self.BACKUP_DIR,
                f"{version.version_id}_backup.pt"
            )

            if os.path.exists(version.model_path) and not os.path.exists(backup_path):
                shutil.copy2(version.model_path, backup_path)
                logger.debug(f"Backed up model: {version.version_id}")

            return True
        except Exception as e:
            logger.warning(f"Could not backup model: {e}")
            return False

    async def _notify_inference_service(self, model_path: str) -> bool:
        """Notify CNN-LSTM inference service to reload the model."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._cnn_lstm_service_url}/api/v1/model/reload",
                    json={"model_path": model_path}
                )

                if response.status_code == 200:
                    logger.info("CNN-LSTM inference service notified successfully")
                    return True
                else:
                    logger.warning(f"CNN-LSTM service notification failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.warning(f"Could not notify CNN-LSTM service: {e}")
            return False

    def _get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a version by ID."""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None

    def _get_previous_version(self) -> Optional[ModelVersion]:
        """Get the previous active version."""
        deployed = [
            v for v in self._versions
            if v.status in [DeploymentStatus.PREVIOUS, DeploymentStatus.ROLLED_BACK]
            and v.deployed_at
        ]
        deployed.sort(key=lambda x: x.deployed_at or "", reverse=True)

        return deployed[0] if deployed else None

    def _cleanup_old_versions(self) -> None:
        """Cleanup old versions beyond the limit."""
        if len(self._versions) <= self.MAX_VERSIONS_TO_KEEP:
            return

        self._versions.sort(key=lambda x: x.created_at)

        to_keep = []
        to_remove = []

        for v in self._versions:
            if v.status in [DeploymentStatus.ACTIVE, DeploymentStatus.PREVIOUS]:
                to_keep.append(v)
            elif len(to_keep) + len([x for x in self._versions if x not in to_remove]) > self.MAX_VERSIONS_TO_KEEP:
                to_remove.append(v)
            else:
                to_keep.append(v)

        for v in to_remove:
            try:
                if os.path.exists(v.model_path):
                    os.remove(v.model_path)
                    logger.debug(f"Removed old model: {v.version_id}")
            except Exception as e:
                logger.warning(f"Could not remove model file: {e}")

        self._versions = [v for v in self._versions if v not in to_remove]

    def get_versions(self, limit: int = 20) -> List[Dict]:
        """Get model versions."""
        versions = sorted(self._versions, key=lambda x: x.created_at, reverse=True)
        return [v.to_dict() for v in versions[:limit]]

    def get_current_version(self) -> Optional[Dict]:
        """Get the currently deployed version."""
        if self._current_version:
            version = self._get_version(self._current_version)
            return version.to_dict() if version else None
        return None

    def get_rollback_history(self, limit: int = 20) -> List[Dict]:
        """Get rollback history."""
        return [e.to_dict() for e in self._rollback_history[-limit:]]

    def get_statistics(self) -> Dict:
        """Get rollback service statistics."""
        total = len(self._versions)
        active = sum(1 for v in self._versions if v.status == DeploymentStatus.ACTIVE)
        rolled_back = sum(1 for v in self._versions if v.status == DeploymentStatus.ROLLED_BACK)
        total_rollbacks = len(self._rollback_history)
        auto_rollbacks = sum(1 for e in self._rollback_history if e.triggered_by == "automatic")

        return {
            "total_versions": total,
            "active_versions": active,
            "rolled_back_count": rolled_back,
            "current_version": self._current_version,
            "max_versions_kept": self.MAX_VERSIONS_TO_KEEP,
            "backup_directory": self.BACKUP_DIR,
            "rollback_statistics": {
                "total_rollbacks": total_rollbacks,
                "automatic_rollbacks": auto_rollbacks,
                "manual_rollbacks": total_rollbacks - auto_rollbacks
            }
        }

    def discover_existing_models(self) -> int:
        """Discover and register existing model files not yet tracked."""
        registered = 0
        existing_ids = {v.version_id for v in self._versions}

        if os.path.exists(self.MODEL_DIR):
            for filename in os.listdir(self.MODEL_DIR):
                if filename.endswith('.pt') and filename != 'latest.pt':
                    version_id = filename.replace('.pt', '')

                    if version_id not in existing_ids:
                        model_path = os.path.join(self.MODEL_DIR, filename)
                        if os.path.isfile(model_path):
                            training_type = "incremental" if "_incr_" in filename else "full"

                            created_at = datetime.fromtimestamp(
                                os.path.getctime(model_path)
                            ).isoformat()

                            version = ModelVersion(
                                version_id=version_id,
                                model_path=model_path,
                                training_type=training_type,
                                status=DeploymentStatus.CANDIDATE,
                                created_at=created_at,
                                notes="Automatically discovered"
                            )

                            self._versions.append(version)
                            registered += 1
                            logger.info(f"Discovered existing model: {version_id}")

        if registered > 0:
            if os.path.exists(self.LATEST_MODEL_LINK):
                try:
                    linked_path = os.path.realpath(self.LATEST_MODEL_LINK)
                    linked_name = os.path.basename(linked_path).replace('.pt', '')

                    for v in self._versions:
                        if v.version_id == linked_name:
                            v.status = DeploymentStatus.ACTIVE
                            v.deployed_at = datetime.utcnow().isoformat()
                            self._current_version = linked_name
                            logger.info(f"Marked {linked_name} as active (linked by latest.pt)")
                            break
                except Exception as e:
                    logger.warning(f"Could not determine active model: {e}")

            self._save_versions()

        return registered


# Singleton instance
rollback_service = CNNLSTMRollbackService()

# Auto-discover existing models on startup
_discovered = rollback_service.discover_existing_models()
if _discovered > 0:
    logger.info(f"Auto-discovered {_discovered} existing CNN-LSTM model(s)")
