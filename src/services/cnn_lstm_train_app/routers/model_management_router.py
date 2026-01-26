"""
Model Management Router for CNN-LSTM Training Service.

API endpoints for:
- A/B Model Validation & Comparison
- Model Versioning & Rollback
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from loguru import logger

from ..services.cnn_lstm_validation_service import validation_service
from ..services.cnn_lstm_rollback_service import rollback_service, RollbackReason


router = APIRouter(prefix="/model", tags=["5. Model Management"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ValidateModelRequest(BaseModel):
    """Request to validate a model."""
    model_path: str = Field(..., description="Path to model file")
    test_data_source: str = Field(
        default="feedback_buffer",
        description="Source of test data: feedback_buffer, file, or inline"
    )
    test_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Inline test data if source is 'inline'"
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CompareModelsRequest(BaseModel):
    """Request to compare two models."""
    production_path: str = Field(..., description="Path to production model")
    candidate_path: str = Field(..., description="Path to candidate model")
    test_data_source: str = Field(default="feedback_buffer")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class DeployModelRequest(BaseModel):
    """Request to deploy a model version."""
    version_id: str = Field(..., description="Version ID to deploy")
    force: bool = Field(default=False, description="Skip validation checks")


class RollbackRequest(BaseModel):
    """Request to rollback to a previous version."""
    target_version_id: Optional[str] = Field(
        None,
        description="Specific version to rollback to. If None, rollback to previous."
    )
    reason: str = Field(
        default="manual",
        description="Reason for rollback: manual, critical_drift, validation_failed, performance_degradation, error"
    )
    notes: Optional[str] = Field(None, description="Optional notes")


class RegisterModelRequest(BaseModel):
    """Request to register a new model version."""
    model_path: str = Field(..., description="Path to model file")
    training_type: str = Field(default="full", description="full or incremental")
    metrics: Optional[Dict] = Field(None, description="Training metrics")
    notes: Optional[str] = Field(None, description="Optional notes")


# =============================================================================
# Validation Endpoints
# =============================================================================

@router.post("/validate", summary="Validate a model")
async def validate_model(request: ValidateModelRequest):
    """
    Validate a model on test data.

    Computes metrics for all three tasks (price, patterns, regime)
    and provides a deployment recommendation.
    """
    try:
        # Get test data
        test_data = []

        if request.test_data_source == "inline" and request.test_data:
            test_data = request.test_data
        elif request.test_data_source == "feedback_buffer":
            # Get samples from feedback buffer
            from ..services.feedback_buffer_service import feedback_buffer_service
            batch = feedback_buffer_service.get_training_batch(min_samples=50)
            test_data = [
                {
                    "features": s.get("ohlcv_context", []),
                    "price_label": [s.get("price_direction_correct", 0)],
                    "pattern_labels": s.get("pattern_predictions", []),
                    "regime_label": s.get("regime_prediction", 0),
                    "outcome": {"is_success": s.get("overall_success", False)}
                }
                for s in batch.get("samples", [])
            ]

        if not test_data:
            return {
                "status": "no_data",
                "message": "No test data available for validation"
            }

        result = validation_service.validate_model(
            model_path=request.model_path,
            test_data=test_data,
            threshold=request.threshold
        )

        return {
            "status": "completed",
            "result": result.to_dict()
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", summary="Compare two models (A/B test)")
async def compare_models(request: CompareModelsRequest):
    """
    Compare a candidate model against the production model.

    Computes per-task deltas and provides a deployment recommendation
    based on the comparison results.
    """
    try:
        # Get test data
        test_data = []

        if request.test_data_source == "feedback_buffer":
            from ..services.feedback_buffer_service import feedback_buffer_service
            batch = feedback_buffer_service.get_training_batch(min_samples=50)
            test_data = [
                {
                    "features": s.get("ohlcv_context", []),
                    "price_label": [s.get("price_direction_correct", 0)],
                    "pattern_labels": s.get("pattern_predictions", []),
                    "regime_label": s.get("regime_prediction", 0),
                    "outcome": {"is_success": s.get("overall_success", False)}
                }
                for s in batch.get("samples", [])
            ]

        if not test_data:
            return {
                "status": "no_data",
                "message": "No test data available for comparison"
            }

        result = validation_service.compare_models(
            production_path=request.production_path,
            candidate_path=request.candidate_path,
            test_data=test_data,
            threshold=request.threshold
        )

        return {
            "status": "completed",
            "comparison": result.to_dict()
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation/history", summary="Get validation history")
async def get_validation_history(
    limit: int = Query(default=20, ge=1, le=100)
):
    """Get recent validation results."""
    return {
        "history": validation_service.get_validation_history(limit),
        "statistics": validation_service.get_statistics()
    }


@router.get("/comparison/history", summary="Get comparison history")
async def get_comparison_history(
    limit: int = Query(default=20, ge=1, le=100)
):
    """Get recent A/B comparison results."""
    return {
        "history": validation_service.get_comparison_history(limit)
    }


# =============================================================================
# Version Management Endpoints
# =============================================================================

@router.post("/register", summary="Register a new model version")
async def register_model(request: RegisterModelRequest):
    """
    Register a new model version for tracking.

    The model becomes a candidate for deployment.
    """
    try:
        version = rollback_service.register_model(
            model_path=request.model_path,
            training_type=request.training_type,
            metrics=request.metrics,
            notes=request.notes or ""
        )

        return {
            "status": "registered",
            "version": version.to_dict()
        }

    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions", summary="Get all model versions")
async def get_versions(
    limit: int = Query(default=20, ge=1, le=100)
):
    """Get tracked model versions."""
    return {
        "versions": rollback_service.get_versions(limit),
        "current_version": rollback_service.get_current_version(),
        "statistics": rollback_service.get_statistics()
    }


@router.get("/versions/current", summary="Get current deployed version")
async def get_current_version():
    """Get the currently deployed model version."""
    current = rollback_service.get_current_version()
    if not current:
        return {
            "status": "no_active_version",
            "message": "No model is currently deployed"
        }

    return {
        "status": "active",
        "version": current
    }


# =============================================================================
# Deployment Endpoints
# =============================================================================

@router.post("/deploy", summary="Deploy a model version")
async def deploy_model(request: DeployModelRequest):
    """
    Deploy a registered model version.

    Updates the symlink and notifies the inference service to reload.
    """
    try:
        result = await rollback_service.deploy_model(
            version_id=request.version_id,
            force=request.force
        )

        if result.get("status") == "failed":
            raise HTTPException(status_code=400, detail=result.get("message"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback", summary="Rollback to a previous version")
async def rollback(request: RollbackRequest):
    """
    Rollback to a previous model version.

    If no target version is specified, rolls back to the most recent
    previous version.
    """
    try:
        # Parse reason
        try:
            reason = RollbackReason(request.reason)
        except ValueError:
            reason = RollbackReason.MANUAL

        result = await rollback_service.rollback(
            target_version_id=request.target_version_id,
            reason=reason,
            triggered_by="manual",
            notes=request.notes or ""
        )

        if result.get("status") == "failed":
            raise HTTPException(status_code=400, detail=result.get("message"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rollback/history", summary="Get rollback history")
async def get_rollback_history(
    limit: int = Query(default=20, ge=1, le=100)
):
    """Get rollback event history."""
    return {
        "history": rollback_service.get_rollback_history(limit),
        "statistics": rollback_service.get_statistics().get("rollback_statistics", {})
    }


@router.post("/discover", summary="Discover existing models")
async def discover_models():
    """
    Discover and register existing model files that are not yet tracked.

    Useful after manual model deployments or migrations.
    """
    try:
        discovered = rollback_service.discover_existing_models()

        return {
            "status": "completed",
            "discovered_count": discovered,
            "total_versions": len(rollback_service._versions)
        }

    except Exception as e:
        logger.error(f"Error discovering models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
