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
