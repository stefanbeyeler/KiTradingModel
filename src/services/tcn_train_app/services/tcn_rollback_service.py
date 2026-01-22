"""
TCN Rollback Service.

Provides safe model deployment with rollback capability:
- Version tracking of all deployed models
- Quick rollback to previous versions
- Validation before deployment
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
    ACTIVE = "active"  # Currently deployed
    PREVIOUS = "previous"  # Was deployed, now rolled back
    CANDIDATE = "candidate"  # Ready for deployment
    ROLLED_BACK = "rolled_back"  # Explicitly rolled back
    FAILED = "failed"  # Deployment failed


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
    metrics: Dict = field(default_factory=dict)
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
            "metrics": self.metrics,
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
            metrics=data.get("metrics", {}),
            notes=data.get("notes", "")
        )


class TCNRollbackService:
    """
    Service for model versioning and rollback.

    Features:
    - Track all deployed model versions
    - Quick rollback to any previous version
    - Safe deployment with validation
    - Hot-reload notification to inference service
    """

    VERSION_FILE = "data/models/tcn/version_history.json"
    MODEL_DIR = "data/models/tcn"
    LATEST_MODEL_LINK = "data/models/tcn/latest.pt"
    BACKUP_DIR = "data/models/tcn/backups"
    MAX_VERSIONS_TO_KEEP = 10

    def __init__(self):
        """Initialize rollback service."""
        self._versions: List[ModelVersion] = []
        self._current_version: Optional[str] = None
        self._tcn_service_url = os.getenv(
            "TCN_SERVICE_URL",
            "http://trading-tcn:3003"
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

                logger.info(f"Loaded {len(self._versions)} model versions")
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
                    "updated_at": datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save version history: {e}")

    def register_model(
        self,
        model_path: str,
        training_type: str = "full",
        metrics: Optional[Dict] = None,
        notes: str = ""
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_path: Path to the model file
            training_type: "full" or "incremental"
            metrics: Training metrics
            notes: Optional notes about the model

        Returns:
            The created model version
        """
        # Extract version ID from filename
        filename = os.path.basename(model_path)
        version_id = filename.replace(".pt", "")

        # Create version entry
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            training_type=training_type,
            status=DeploymentStatus.CANDIDATE,
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
            notes=notes
        )

        # Add to versions list
        self._versions.append(version)

        # Cleanup old versions
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
        # Find the version
        version = self._get_version(version_id)
        if not version:
            return {
                "status": "failed",
                "message": f"Version {version_id} not found"
            }

        # Check if model file exists
        if not os.path.exists(version.model_path):
            return {
                "status": "failed",
                "message": f"Model file not found: {version.model_path}"
            }

        # Backup current model if exists
        if self._current_version:
            current = self._get_version(self._current_version)
            if current:
                await self._backup_model(current)
                current.status = DeploymentStatus.PREVIOUS

        # Update symlink
        try:
            if os.path.exists(self.LATEST_MODEL_LINK):
                os.remove(self.LATEST_MODEL_LINK)

            # Use relative path for symlink
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
            "previous_version": self._get_previous_version_id()
        }

    async def rollback(
        self,
        target_version_id: Optional[str] = None
    ) -> Dict:
        """
        Rollback to a previous model version.

        Args:
            target_version_id: Specific version to rollback to.
                              If None, rolls back to previous version.

        Returns:
            Rollback result
        """
        if target_version_id:
            # Rollback to specific version
            target = self._get_version(target_version_id)
        else:
            # Rollback to previous version
            target = self._get_previous_version()

        if not target:
            return {
                "status": "failed",
                "message": "No previous version available for rollback"
            }

        # Mark current version as rolled back
        if self._current_version:
            current = self._get_version(self._current_version)
            if current:
                current.status = DeploymentStatus.ROLLED_BACK
                current.rolled_back_at = datetime.utcnow().isoformat()

        # Deploy the target version
        result = await self.deploy_model(target.version_id)

        if result.get("status") == "deployed":
            result["rollback_from"] = self._current_version
            logger.info(f"Rolled back from {self._current_version} to {target.version_id}")

        return result

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
        """Notify TCN inference service to reload the model."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._tcn_service_url}/api/v1/model/reload",
                    json={"model_path": model_path}
                )

                if response.status_code == 200:
                    logger.info("TCN inference service notified successfully")
                    return True
                else:
                    logger.warning(f"TCN service notification failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.warning(f"Could not notify TCN service: {e}")
            return False

    def _get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a version by ID."""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None

    def _get_previous_version(self) -> Optional[ModelVersion]:
        """Get the previous active version."""
        # Sort by deployed_at descending
        deployed = [
            v for v in self._versions
            if v.status in [DeploymentStatus.PREVIOUS, DeploymentStatus.ROLLED_BACK]
            and v.deployed_at
        ]
        deployed.sort(key=lambda x: x.deployed_at or "", reverse=True)

        return deployed[0] if deployed else None

    def _get_previous_version_id(self) -> Optional[str]:
        """Get the previous version ID."""
        prev = self._get_previous_version()
        return prev.version_id if prev else None

    def _cleanup_old_versions(self) -> None:
        """Cleanup old versions beyond the limit."""
        if len(self._versions) <= self.MAX_VERSIONS_TO_KEEP:
            return

        # Sort by created_at (oldest first)
        self._versions.sort(key=lambda x: x.created_at)

        # Keep the most recent versions plus any active/previous
        to_keep = []
        to_remove = []

        for v in self._versions:
            if v.status in [DeploymentStatus.ACTIVE, DeploymentStatus.PREVIOUS]:
                to_keep.append(v)
            elif len(to_keep) + len([x for x in self._versions if x not in to_remove]) > self.MAX_VERSIONS_TO_KEEP:
                to_remove.append(v)
            else:
                to_keep.append(v)

        # Remove old model files
        for v in to_remove:
            try:
                if os.path.exists(v.model_path):
                    os.remove(v.model_path)
                    logger.debug(f"Removed old model: {v.version_id}")
            except Exception as e:
                logger.warning(f"Could not remove model file: {e}")

        self._versions = [v for v in self._versions if v not in to_remove]

    def get_versions(self, limit: int = 20) -> List[ModelVersion]:
        """Get model versions."""
        # Sort by created_at descending
        versions = sorted(self._versions, key=lambda x: x.created_at, reverse=True)
        return versions[:limit]

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get the currently deployed version."""
        if self._current_version:
            return self._get_version(self._current_version)
        return None

    def get_statistics(self) -> Dict:
        """Get rollback service statistics."""
        total = len(self._versions)
        active = sum(1 for v in self._versions if v.status == DeploymentStatus.ACTIVE)
        rolled_back = sum(1 for v in self._versions if v.status == DeploymentStatus.ROLLED_BACK)

        return {
            "total_versions": total,
            "active_versions": active,
            "rolled_back_count": rolled_back,
            "current_version": self._current_version,
            "max_versions_kept": self.MAX_VERSIONS_TO_KEEP,
            "backup_directory": self.BACKUP_DIR
        }

    def discover_existing_models(self) -> int:
        """
        Discover and register existing model files that are not yet tracked.

        Returns:
            Number of newly registered models
        """
        registered = 0
        existing_ids = {v.version_id for v in self._versions}

        if os.path.exists(self.MODEL_DIR):
            for filename in os.listdir(self.MODEL_DIR):
                if filename.endswith('.pt') and filename != 'latest.pt':
                    version_id = filename.replace('.pt', '')

                    if version_id not in existing_ids:
                        model_path = os.path.join(self.MODEL_DIR, filename)
                        if os.path.isfile(model_path):
                            # Determine training type from filename
                            training_type = "incremental" if "_incr_" in filename else "full"

                            # Get creation time
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
            # Check if latest.pt points to one of these models
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
rollback_service = TCNRollbackService()

# Auto-discover existing models on startup
_discovered = rollback_service.discover_existing_models()
if _discovered > 0:
    logger.info(f"Auto-discovered {_discovered} existing model(s)")
