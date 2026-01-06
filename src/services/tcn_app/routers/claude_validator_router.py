"""Claude Vision Validator API Endpoints for TCN Chart Patterns.

Provides endpoints for external pattern validation using Claude AI.
"""

import os
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..services.claude_validator_service import (
    tcn_claude_validator_service,
    ValidationStatus,
    TCNClaudeValidationResult
)
from ..services.tcn_pattern_history_service import tcn_pattern_history_service

router = APIRouter()


class TCNValidationRequest(BaseModel):
    """Request to validate a specific TCN pattern."""
    pattern_id: str = Field(..., description="ID of the pattern to validate")
    force: bool = Field(default=False, description="Force re-validation even if cached")


class TCNValidationResponse(BaseModel):
    """Response for a validation request."""
    pattern_id: str
    pattern_type: str
    symbol: str
    timeframe: str
    claude_agrees: bool
    claude_confidence: float
    claude_pattern_type: Optional[str]
    claude_reasoning: str
    visual_quality_score: float
    market_context_score: float
    status: str
    validation_timestamp: str


@router.get("/status")
async def get_claude_validator_status():
    """
    Get Claude validator service status.

    Returns information about the validator configuration and availability.
    """
    return tcn_claude_validator_service.get_status()


@router.post("/validate")
async def validate_pattern(request: TCNValidationRequest):
    """
    Validate a single TCN pattern using Claude Vision API.

    This endpoint renders a chart image of the pattern and sends it to
    Claude for visual analysis. Claude will assess whether the pattern
    is correctly identified.

    The pattern must exist in the TCN pattern history with stored OHLCV data.

    **Note**: Requires ANTHROPIC_API_KEY environment variable to be set.
    """
    try:
        # Get pattern from history
        history = tcn_pattern_history_service.get_all_patterns()
        pattern = next((p for p in history if p.get("id") == request.pattern_id), None)

        if not pattern:
            raise HTTPException(
                status_code=404,
                detail=f"Pattern {request.pattern_id} not found in TCN pattern history"
            )

        # Get OHLCV data from pattern
        ohlcv_data = pattern.get("ohlcv_data", [])
        if not ohlcv_data:
            raise HTTPException(
                status_code=400,
                detail=f"Pattern {request.pattern_id} has no stored OHLCV data. "
                       "Only patterns with stored OHLCV data can be validated."
            )

        # Extract pattern details
        symbol = pattern.get("symbol", "")
        timeframe = pattern.get("timeframe", "H1")
        pattern_type = pattern.get("pattern_type", "unknown")
        direction = pattern.get("direction", "neutral")
        confidence = pattern.get("confidence", 0.5)
        pattern_start_time = pattern.get("pattern_start_time")
        pattern_end_time = pattern.get("pattern_end_time")
        pattern_points = pattern.get("pattern_points", [])

        # Validate with Claude
        result = await tcn_claude_validator_service.validate_pattern(
            pattern_id=request.pattern_id,
            pattern_type=pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            ohlcv_data=ohlcv_data,
            pattern_points=pattern_points,
            pattern_start_time=pattern_start_time,
            pattern_end_time=pattern_end_time,
            force=request.force
        )

        return {
            "status": "success",
            "validation": result.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TCN validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_validation_history(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    pattern_type: Optional[str] = Query(default=None, description="Filter by pattern type"),
    status: Optional[str] = Query(
        default=None,
        description="Filter by status (validated, rejected, error, skipped)"
    )
):
    """
    Get Claude validation history for TCN patterns.

    Returns historical validation results with optional filtering.
    """
    try:
        status_filter = None
        if status:
            try:
                status_filter = ValidationStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        history = tcn_claude_validator_service.get_validation_history(
            limit=limit,
            symbol=symbol,
            pattern_type=pattern_type,
            status=status_filter
        )

        return {
            "count": len(history),
            "history": history
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting TCN validation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_validation_statistics():
    """
    Get Claude validation statistics for TCN patterns.

    Returns aggregated statistics about pattern validations including:
    - Total validations performed
    - Agreement/rejection rates
    - Average quality scores
    - Breakdown by pattern type and symbol
    """
    try:
        stats = tcn_claude_validator_service.get_validation_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting TCN validation statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart-preview/{pattern_id}")
async def get_chart_preview(pattern_id: str):
    """
    Get a chart preview for a TCN pattern without running validation.

    Returns the base64-encoded chart image that would be sent to Claude.
    Useful for debugging and previewing the chart rendering.
    """
    try:
        # Get pattern from history
        history = tcn_pattern_history_service.get_all_patterns()
        pattern = next((p for p in history if p.get("id") == pattern_id), None)

        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        ohlcv_data = pattern.get("ohlcv_data", [])
        if not ohlcv_data:
            raise HTTPException(
                status_code=400,
                detail=f"Pattern {pattern_id} has no stored OHLCV data"
            )

        # Render chart
        pattern_type = pattern.get("pattern_type", "unknown")
        direction = pattern.get("direction", "neutral")
        pattern_points = pattern.get("pattern_points", [])
        pattern_start_time = pattern.get("pattern_start_time")
        pattern_end_time = pattern.get("pattern_end_time")

        chart_renderer = tcn_claude_validator_service.chart_renderer
        chart_base64 = chart_renderer.render_tcn_pattern_chart(
            ohlcv_data=ohlcv_data,
            pattern_type=pattern_type,
            pattern_points=pattern_points,
            direction=direction,
            pattern_start_time=pattern_start_time,
            pattern_end_time=pattern_end_time
        )

        if not chart_base64:
            raise HTTPException(
                status_code=500,
                detail="Failed to render chart (matplotlib may not be available)"
            )

        return {
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "symbol": pattern.get("symbol"),
            "timeframe": pattern.get("timeframe"),
            "direction": direction,
            "chart_image": f"data:image/png;base64,{chart_base64}",
            "candles_shown": len(ohlcv_data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating TCN chart preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/problematic-patterns")
async def get_problematic_patterns(
    min_rejections: int = Query(default=2, ge=1, description="Minimum rejections to include"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results")
):
    """
    Get TCN patterns that Claude frequently rejects.

    Identifies patterns that may need rule adjustments based on
    Claude's assessments. Useful for improving detection accuracy.
    """
    try:
        stats = tcn_claude_validator_service.get_validation_statistics()
        by_pattern = stats.get("by_pattern", {})

        problematic = []
        for pattern_type, data in by_pattern.items():
            total = data.get("total", 0)
            agreed = data.get("agreed", 0)
            rejected = total - agreed

            if rejected >= min_rejections:
                rejection_rate = (rejected / total * 100) if total > 0 else 0
                problematic.append({
                    "pattern_type": pattern_type,
                    "total_validations": total,
                    "agreed": agreed,
                    "rejected": rejected,
                    "rejection_rate": round(rejection_rate, 1)
                })

        # Sort by rejection rate
        problematic.sort(key=lambda x: x["rejection_rate"], reverse=True)

        return {
            "count": len(problematic[:limit]),
            "patterns": problematic[:limit],
            "message": f"{len(problematic)} TCN patterns with >= {min_rejections} rejections"
        }

    except Exception as e:
        logger.error(f"Error getting problematic TCN patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-statistics")
async def reset_validation_statistics():
    """
    Reset Claude validation statistics for TCN patterns.

    Clears the validation history and cache, which resets:
    - All validation statistics
    - Agreement/rejection rates
    - All by-pattern and by-symbol statistics
    """
    try:
        result = tcn_claude_validator_service.clear_history()

        logger.info(f"TCN Claude validation statistics reset: {result}")

        return {
            "status": "success",
            "message": "TCN Claude Validierungsstatistik zurückgesetzt",
            **result
        }

    except Exception as e:
        logger.error(f"Error resetting TCN validation statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
