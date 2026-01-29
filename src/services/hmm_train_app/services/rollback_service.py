"""Rollback Service for model deployment decisions.

Handles:
- Automatic deployment of validated models
- Automatic rejection of underperforming models
- Manual rollback to previous versions
- Notification of inference service and watchdog
"""

import os
import json
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from src.config.microservices import microservices_config
from .model_registry import ModelRegistry, ModelVersion, ModelStatus, model_registry
from .validation_service import ValidationService, validation_service
from .ab_comparison_service import (
    ABComparisonService,
    ABComparisonResult,
    Recommendation,
    ab_comparison_service
)


@dataclass
class DeploymentDecision:
    """Decision record for model deployment."""
    decision_id: str
    timestamp: str

    model_type: str                    # "hmm" or "scorer"
    symbol: Optional[str]              # For HMM models
    version_id: str

    action: str                        # "deployed", "rejected", "rolled_back", "pending_review"
    reason: str

    # Comparison details
    comparison_result: Optional[Dict[str, Any]] = None

    # Notification status
    inference_notified: bool = False
    watchdog_notified: bool = False

    # Previous production (for rollback reference)
    previous_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Service URLs
HMM_SERVICE_URL = os.getenv("HMM_SERVICE_URL", microservices_config.hmm_service_url)
WATCHDOG_URL = os.getenv("WATCHDOG_URL", microservices_config.watchdog_service_url)
DECISION_HISTORY_FILE = os.getenv(
    "DECISION_HISTORY_FILE",
    "/app/data/models/hmm/deployment_decisions.json"
)


class RollbackService:
    """
    Service for managing model deployment and rollback.

    Integrates with:
    - Model Registry: Version and production management
    - A/B Comparison Service: Quality validation
    - HMM Inference Service: Model reload notifications
    - Watchdog Service: Status reporting
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        ab_service: Optional[ABComparisonService] = None,
        validation_svc: Optional[ValidationService] = None,
        hmm_service_url: str = HMM_SERVICE_URL,
        watchdog_url: str = WATCHDOG_URL
    ):
        self._registry = registry or model_registry
        self._ab_service = ab_service or ab_comparison_service
        self._validation = validation_svc or validation_service
        self._hmm_url = hmm_service_url
        self._watchdog_url = watchdog_url

        self._http_client: Optional[httpx.AsyncClient] = None
        self._history_file = Path(DECISION_HISTORY_FILE)
        self._decisions: List[DeploymentDecision] = []

        self._load_history()
        logger.info("RollbackService initialized")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _load_history(self):
        """Load deployment decision history."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r') as f:
                    data = json.load(f)

                for item in data[-100:]:  # Keep last 100 decisions
                    self._decisions.append(DeploymentDecision(**item))

                logger.info(f"Loaded {len(self._decisions)} deployment decisions")
        except Exception as e:
            logger.error(f"Failed to load decision history: {e}")

    def _save_history(self):
        """Save deployment decision history."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)

            data = [d.to_dict() for d in self._decisions[-100:]]
            with open(self._history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save decision history: {e}")

    def _generate_decision_id(self) -> str:
        """Generate unique decision ID."""
        return f"dec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    def _save_decision(self, decision: DeploymentDecision):
        """Save a decision to history."""
        self._decisions.append(decision)
        self._save_history()

    # =========================================================================
    # Main Decision Flow
    # =========================================================================

    async def process_training_result(
        self,
        version_id: str,
        model_type: str,
        symbol: Optional[str],
        candidate_path: str,
        timeframe: str,
        symbols_for_scorer: Optional[List[str]] = None,
        training_job_id: Optional[str] = None
    ) -> DeploymentDecision:
        """
        Process a training result and make deployment decision.

        This is the main entry point after training completes.

        Args:
            version_id: Version ID of the newly trained model
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models, None for scorer
            candidate_path: Path to trained model file
            timeframe: Training timeframe
            symbols_for_scorer: Symbols to use for scorer validation
            training_job_id: Associated training job ID

        Returns:
            DeploymentDecision with action taken
        """
        decision = DeploymentDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            symbol=symbol,
            version_id=version_id,
            action="pending",
            reason=""
        )

        try:
            # Get current production for rollback reference
            current_production = self._registry.get_current_production(model_type, symbol)
            if current_production:
                decision.previous_version = current_production.version_id

            # Run A/B comparison
            if model_type == "hmm":
                comparison = await self._ab_service.compare_hmm_models(
                    candidate_path, symbol, timeframe, version_id
                )
            else:
                symbols = symbols_for_scorer or []
                comparison = await self._ab_service.compare_scorer_models(
                    candidate_path, symbols, timeframe, version_id
                )

            decision.comparison_result = comparison.to_dict()

            # Process recommendation
            if comparison.recommendation == Recommendation.DEPLOY:
                # Promote to production
                success = self._registry.promote_to_production(version_id, model_type, symbol)

                if success:
                    # Notify inference service
                    decision.inference_notified = await self.notify_inference_service(
                        model_type, symbol, candidate_path
                    )
                    decision.action = "deployed"
                    decision.reason = comparison.recommendation_reason
                else:
                    decision.action = "failed"
                    decision.reason = "Registry promotion failed"

            elif comparison.recommendation == Recommendation.REJECT:
                # Mark as rejected in registry
                self._registry.reject_version(
                    version_id, model_type, symbol,
                    comparison.recommendation_reason
                )
                decision.action = "rejected"
                decision.reason = comparison.recommendation_reason

            else:  # MANUAL_REVIEW
                decision.action = "pending_review"
                decision.reason = comparison.recommendation_reason

            # Notify watchdog
            decision.watchdog_notified = await self.notify_watchdog(decision)

        except Exception as e:
            logger.error(f"Deployment decision failed: {e}")
            decision.action = "error"
            decision.reason = str(e)

        # Save decision
        self._save_decision(decision)

        logger.info(
            f"Deployment decision for {model_type}/{symbol or 'global'}: "
            f"{decision.action} - {decision.reason}"
        )

        return decision

    # =========================================================================
    # Manual Operations
    # =========================================================================

    async def rollback_to_version(
        self,
        model_type: str,
        symbol: Optional[str],
        target_version: str,
        reason: str = "Manual rollback"
    ) -> DeploymentDecision:
        """
        Manually rollback to a previous version.

        Args:
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models
            target_version: Version ID to rollback to
            reason: Reason for rollback

        Returns:
            DeploymentDecision with rollback result
        """
        decision = DeploymentDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            symbol=symbol,
            version_id=target_version,
            action="pending",
            reason=reason
        )

        try:
            # Get current production
            current = self._registry.get_current_production(model_type, symbol)
            if current:
                decision.previous_version = current.version_id

            # Perform rollback in registry
            success = self._registry.rollback_to_version(
                model_type, symbol, target_version, reason
            )

            if success:
                # Get the model path
                target_model = self._registry.get_model(target_version, model_type, symbol)
                if target_model:
                    model_path = str(self._registry._versions_path / target_model.model_path)

                    # Notify inference service
                    decision.inference_notified = await self.notify_inference_service(
                        model_type, symbol, model_path
                    )

                decision.action = "rolled_back"
                decision.reason = reason
            else:
                decision.action = "failed"
                decision.reason = f"Rollback failed - version {target_version} not found"

            # Notify watchdog
            decision.watchdog_notified = await self.notify_watchdog(decision)

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            decision.action = "error"
            decision.reason = str(e)

        self._save_decision(decision)
        return decision

    async def force_deploy(
        self,
        version_id: str,
        model_type: str,
        symbol: Optional[str],
        reason: str = "Manual deployment"
    ) -> DeploymentDecision:
        """
        Force deploy a model version (bypass A/B comparison).

        Use with caution - this skips quality validation.

        Args:
            version_id: Version to deploy
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models
            reason: Reason for forced deployment

        Returns:
            DeploymentDecision with deployment result
        """
        decision = DeploymentDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            symbol=symbol,
            version_id=version_id,
            action="pending",
            reason=f"FORCE DEPLOY: {reason}"
        )

        try:
            # Get current production
            current = self._registry.get_current_production(model_type, symbol)
            if current:
                decision.previous_version = current.version_id

            # Promote to production
            success = self._registry.promote_to_production(version_id, model_type, symbol)

            if success:
                # Get model path
                model = self._registry.get_model(version_id, model_type, symbol)
                if model:
                    model_path = str(self._registry._versions_path / model.model_path)

                    decision.inference_notified = await self.notify_inference_service(
                        model_type, symbol, model_path
                    )

                decision.action = "deployed"
            else:
                decision.action = "failed"
                decision.reason = f"Force deploy failed - version {version_id} not found"

            decision.watchdog_notified = await self.notify_watchdog(decision)

        except Exception as e:
            logger.error(f"Force deploy failed: {e}")
            decision.action = "error"
            decision.reason = str(e)

        self._save_decision(decision)

        logger.warning(f"Force deployed {model_type}/{symbol or 'global'} version {version_id}")
        return decision

    # =========================================================================
    # Notifications
    # =========================================================================

    async def notify_inference_service(
        self,
        model_type: str,
        symbol: Optional[str],
        model_path: str
    ) -> bool:
        """
        Notify HMM inference service of model update.

        Args:
            model_type: "hmm" or "scorer"
            symbol: Symbol for HMM models
            model_path: Path to new model file

        Returns:
            True if notification succeeded
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self._hmm_url}/api/v1/model/reload",
                json={
                    "model_type": model_type,
                    "symbol": symbol,
                    "model_path": model_path
                },
                timeout=30.0
            )

            if response.status_code == 200:
                logger.info(f"Inference service notified: {model_type}/{symbol or 'global'}")
                return True
            else:
                logger.warning(f"Inference notification failed: {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"Could not notify inference service: {e}")
            return False

    async def notify_watchdog(self, decision: DeploymentDecision) -> bool:
        """
        Notify watchdog service of deployment decision.

        Args:
            decision: Deployment decision to report

        Returns:
            True if notification succeeded
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self._watchdog_url}/api/v1/training/deployment-update",
                json={
                    "decision_id": decision.decision_id,
                    "model_type": decision.model_type,
                    "symbol": decision.symbol,
                    "version_id": decision.version_id,
                    "action": decision.action,
                    "reason": decision.reason,
                    "timestamp": decision.timestamp
                },
                timeout=10.0
            )

            if response.status_code in [200, 404]:  # 404 = endpoint not yet implemented
                return True
            else:
                logger.debug(f"Watchdog notification returned: {response.status_code}")
                return False

        except Exception as e:
            logger.debug(f"Could not notify watchdog: {e}")
            return False

    # =========================================================================
    # Queries
    # =========================================================================

    def get_deployment_history(
        self,
        model_type: Optional[str] = None,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 50
    ) -> List[DeploymentDecision]:
        """
        Get deployment decision history.

        Args:
            model_type: Filter by model type
            symbol: Filter by symbol
            action: Filter by action (deployed, rejected, etc.)
            limit: Maximum results

        Returns:
            List of decisions (newest first)
        """
        results = []

        for decision in reversed(self._decisions):
            # Apply filters
            if model_type and decision.model_type != model_type:
                continue
            if symbol is not None and decision.symbol != symbol:
                continue
            if action and decision.action != action:
                continue

            results.append(decision)

            if len(results) >= limit:
                break

        return results

    def get_latest_decision(
        self,
        model_type: str,
        symbol: Optional[str]
    ) -> Optional[DeploymentDecision]:
        """Get the most recent decision for a model."""
        for decision in reversed(self._decisions):
            if decision.model_type == model_type:
                if model_type == "scorer" or decision.symbol == symbol:
                    return decision
        return None

    def get_comparison_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get A/B comparison history from decisions."""
        comparisons = []

        for decision in reversed(self._decisions):
            if decision.comparison_result:
                comparisons.append({
                    "decision_id": decision.decision_id,
                    "timestamp": decision.timestamp,
                    "model_type": decision.model_type,
                    "symbol": decision.symbol,
                    "comparison": decision.comparison_result
                })

            if len(comparisons) >= limit:
                break

        return comparisons

    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        stats = {
            "total_decisions": len(self._decisions),
            "actions": {},
            "by_model_type": {}
        }

        for decision in self._decisions:
            # Count by action
            stats["actions"][decision.action] = stats["actions"].get(decision.action, 0) + 1

            # Count by model type
            mt = decision.model_type
            if mt not in stats["by_model_type"]:
                stats["by_model_type"][mt] = {"total": 0, "deployed": 0, "rejected": 0}

            stats["by_model_type"][mt]["total"] += 1
            if decision.action == "deployed":
                stats["by_model_type"][mt]["deployed"] += 1
            elif decision.action == "rejected":
                stats["by_model_type"][mt]["rejected"] += 1

        return stats


# Global singleton
rollback_service = RollbackService()
