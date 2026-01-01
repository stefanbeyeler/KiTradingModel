"""Rule Optimizer API Router.

Provides endpoints for Claude Vision-based pattern rule optimization.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from ..services.rule_optimizer_service import rule_optimizer_service
from ..services.rule_config_service import rule_config_service


router = APIRouter(
    prefix="/api/v1/rule-optimizer",
    tags=["5. Rule Optimizer"],
)


# === Request/Response Models ===

class StartOptimizationRequest(BaseModel):
    """Request to start an optimization session."""
    symbols: List[str] = Field(
        default=["BTCUSD", "EURUSD", "XAUUSD", "GER40"],
        description="Symbols to scan for patterns"
    )
    timeframes: List[str] = Field(
        default=["H1", "H4", "D1"],
        description="Timeframes to scan"
    )
    samples_per_pattern: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of samples to collect per pattern type"
    )
    pattern_types: Optional[List[str]] = Field(
        default=None,
        description="Specific pattern types to optimize (None = all)"
    )


class ApplyRecommendationsRequest(BaseModel):
    """Request to apply selected recommendations."""
    indices: Optional[List[int]] = Field(
        default=None,
        description="Indices of recommendations to apply (None = all)"
    )


class SetParameterRequest(BaseModel):
    """Request to manually set a parameter."""
    pattern: str = Field(..., description="Pattern type (e.g., 'hammer')")
    parameter: str = Field(..., description="Parameter name")
    value: float = Field(..., description="New value")
    reason: str = Field(default="manual_adjustment", description="Reason for change")


# === Endpoints ===

@router.post("/start", summary="Start optimization session")
async def start_optimization(request: StartOptimizationRequest):
    """
    Start a new rule optimization session.

    This will:
    1. Scan the specified symbols/timeframes for patterns
    2. Validate each pattern using Claude Vision API
    3. Analyze results to identify systematic issues
    4. Generate parameter optimization recommendations

    The process runs in the background - use /progress to monitor status.
    """
    try:
        session_id = await rule_optimizer_service.start_optimization(
            symbols=request.symbols,
            timeframes=request.timeframes,
            samples_per_pattern=request.samples_per_pattern,
            pattern_types=request.pattern_types,
        )

        return {
            "status": "started",
            "session_id": session_id,
            "message": "Optimierung gestartet. Verwende /progress zum Überwachen.",
            "config": {
                "symbols": request.symbols,
                "timeframes": request.timeframes,
                "samples_per_pattern": request.samples_per_pattern,
            }
        }

    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress", summary="Get optimization progress")
async def get_progress():
    """Get current optimization progress."""
    return rule_optimizer_service.get_progress()


@router.get("/session/{session_id}", summary="Get session details")
async def get_session(session_id: str):
    """Get details of a specific optimization session."""
    session = rule_optimizer_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/recommendations", summary="Get optimization recommendations")
async def get_recommendations():
    """
    Get generated optimization recommendations.

    Returns a list of parameter changes recommended based on Claude Vision analysis.
    """
    recommendations = rule_optimizer_service.get_recommendations()
    analyses = rule_optimizer_service.get_analyses()

    return {
        "recommendations": recommendations,
        "pattern_analyses": analyses,
        "total_recommendations": len(recommendations),
    }


@router.post("/apply", summary="Apply recommendations")
async def apply_recommendations(request: ApplyRecommendationsRequest):
    """
    Apply selected recommendations to the rule configuration.

    This will update the rule_config.json with the recommended parameter values.
    """
    try:
        result = await rule_optimizer_service.apply_recommendations(
            recommendation_indices=request.indices
        )
        return result

    except Exception as e:
        logger.error(f"Failed to apply recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-params", summary="Get current rule parameters")
async def get_current_params(pattern: Optional[str] = None):
    """
    Get current rule parameters.

    Args:
        pattern: Optional pattern type to filter (e.g., 'hammer')
    """
    if pattern:
        params = rule_config_service.get_pattern_params(pattern)
        return {
            "pattern": pattern,
            "params": params,
        }
    else:
        return {
            "params": rule_config_service.get_all_params(),
        }


@router.post("/set-param", summary="Manually set a parameter")
async def set_parameter(request: SetParameterRequest):
    """Manually set a specific rule parameter."""
    try:
        success = rule_config_service.set_param(
            pattern=request.pattern,
            param_name=request.parameter,
            value=request.value,
            reason=request.reason,
        )

        if success:
            return {
                "status": "success",
                "pattern": request.pattern,
                "parameter": request.parameter,
                "value": request.value,
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set parameter")

    except Exception as e:
        logger.error(f"Failed to set parameter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", summary="Reset parameters to defaults")
async def reset_to_defaults():
    """Reset all rule parameters to their default values."""
    result = rule_optimizer_service.reset_to_defaults()
    return result


@router.post("/factory-reset", summary="Reset to factory defaults")
async def factory_reset():
    """
    Reset everything to factory defaults.

    This will DELETE all data files and reset to a clean state:
    - rule_config.json (reset to defaults)
    - pattern_feedback.json (deleted)
    - pattern_history.json (deleted)
    - claude_validations.json (deleted)
    - pending_validations.json (deleted)
    - backups/ directory (deleted)

    WARNING: This action is irreversible! All feedback, history, and validations will be lost.
    """
    result = rule_config_service.factory_reset()

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Factory reset failed"))

    return result


@router.get("/history", summary="Get parameter adjustment history")
async def get_adjustment_history(limit: int = Query(default=50, ge=1, le=200)):
    """Get history of parameter adjustments."""
    history = rule_config_service.get_adjustment_history(limit=limit)
    return {
        "history": history,
        "total": len(history),
    }


@router.get("/compare", summary="Compare current vs default parameters")
async def compare_parameters():
    """
    Compare current parameters with defaults.

    Shows which parameters have been modified and their differences.
    """
    from ..services.rule_config_service import DEFAULT_RULE_PARAMS

    current = rule_config_service.get_all_params()
    differences = []

    for pattern, params in current.items():
        defaults = DEFAULT_RULE_PARAMS.get(pattern, {})

        for param_name, current_value in params.items():
            default_value = defaults.get(param_name)

            if default_value is not None and current_value != default_value:
                differences.append({
                    "pattern": pattern,
                    "parameter": param_name,
                    "default": default_value,
                    "current": current_value,
                    "change_pct": round((current_value - default_value) / default_value * 100, 1) if default_value != 0 else None,
                })

    return {
        "total_differences": len(differences),
        "differences": differences,
    }


@router.get("/pattern-stats", summary="Get pattern detection statistics")
async def get_pattern_stats():
    """
    Get statistics about pattern detection performance.

    This shows the analysis results from the last optimization session.
    """
    analyses = rule_optimizer_service.get_analyses()

    summary = {
        "pattern_count": len(analyses),
        "total_samples": sum(a.get("total_samples", 0) for a in analyses.values()),
        "patterns": {},
    }

    for pattern_type, analysis in analyses.items():
        summary["patterns"][pattern_type] = {
            "total_samples": analysis.get("total_samples", 0),
            "valid_count": analysis.get("valid_count", 0),
            "invalid_count": analysis.get("invalid_count", 0),
            "validity_rate": analysis.get("validity_rate", 0),
            "false_positive_rate": analysis.get("false_positive_rate", 0),
            "top_issues": sorted(
                analysis.get("issues", {}).items(),
                key=lambda x: -x[1]
            )[:5],
        }

    return summary


@router.get("/validated-samples", summary="Get all validated pattern samples")
async def get_validated_samples():
    """
    Get all validated pattern samples with their details.

    Returns samples including chart images, Claude's validation result,
    and reasoning for each pattern that was validated during optimization.
    """
    samples = rule_optimizer_service.get_validated_samples()
    return {
        "samples": samples,
        "total": len(samples),
    }


@router.get("/sample/{sample_id}", summary="Get a specific sample by ID")
async def get_sample(sample_id: str):
    """Get a specific validated sample by its ID."""
    sample = rule_optimizer_service.get_sample_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample


# ==================== Backup/Restore Endpoints ====================

class CreateBackupRequest(BaseModel):
    """Request to create a backup."""
    name: Optional[str] = Field(
        default=None,
        description="Optional backup name. Auto-generated if not provided."
    )


class RestoreBackupRequest(BaseModel):
    """Request to restore a backup."""
    backup_name: str = Field(..., description="Name or filename of the backup to restore")


class ImportConfigRequest(BaseModel):
    """Request to import configuration."""
    params: dict = Field(..., description="Parameter configuration to import")
    history: Optional[list] = Field(default=None, description="Optional history to import")


@router.post("/backup", summary="Create configuration backup")
async def create_backup(request: CreateBackupRequest = CreateBackupRequest()):
    """
    Create a backup of the current rule configuration.

    The backup includes all parameters and adjustment history.
    An automatic backup is also created before restore/import operations.
    """
    result = rule_config_service.create_backup(name=request.name)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Backup failed"))

    return result


@router.get("/backups", summary="List all backups")
async def list_backups():
    """
    List all available configuration backups.

    Returns backup metadata including name, creation date, and size.
    """
    backups = rule_config_service.list_backups()
    return {
        "backups": backups,
        "total": len(backups)
    }


@router.post("/restore", summary="Restore from backup")
async def restore_backup(request: RestoreBackupRequest):
    """
    Restore rule configuration from a backup.

    An automatic backup of the current state is created before restoring.
    """
    result = rule_config_service.restore_backup(backup_name=request.backup_name)

    if not result.get("success"):
        raise HTTPException(
            status_code=404 if "not found" in result.get("error", "") else 500,
            detail=result.get("error", "Restore failed")
        )

    return result


@router.delete("/backup/{backup_name}", summary="Delete a backup")
async def delete_backup(backup_name: str):
    """Delete a specific backup file."""
    result = rule_config_service.delete_backup(backup_name)

    if not result.get("success"):
        raise HTTPException(
            status_code=404 if "not found" in result.get("error", "") else 500,
            detail=result.get("error", "Delete failed")
        )

    return result


@router.get("/backup/{backup_name}/export", summary="Export a specific backup")
async def export_backup(backup_name: str):
    """
    Export a specific backup as JSON for download.

    Returns the complete backup data including params and history.
    """
    result = rule_config_service.get_backup(backup_name)

    if not result.get("success"):
        raise HTTPException(
            status_code=404 if "not found" in result.get("error", "") else 500,
            detail=result.get("error", "Backup not found")
        )

    return result["data"]


@router.get("/export", summary="Export configuration")
async def export_config():
    """
    Export the current configuration as JSON.

    The exported data can be saved locally and imported later.
    Includes parameters, history, and default values for reference.
    """
    return rule_config_service.export_config()


@router.post("/import", summary="Import configuration")
async def import_config(request: ImportConfigRequest):
    """
    Import configuration from uploaded JSON.

    An automatic backup is created before importing.
    The request must include a 'params' field with the configuration.
    """
    config_data = {
        "params": request.params,
        "history": request.history
    }

    result = rule_config_service.import_config(config_data)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Import failed"))

    return result
