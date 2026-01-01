"""Claude Vision Validator API Endpoints.

Provides endpoints for external pattern validation using Claude AI.
"""

import os
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger
import httpx

from ..services.claude_validator_service import (
    claude_validator_service,
    ValidationStatus,
    ClaudeValidationResult
)
from ..services.pattern_history_service import pattern_history_service
from ..services.feedback_analyzer_service import feedback_analyzer_service
from ..services.rule_config_service import rule_config_service

router = APIRouter()

# Data Service URL for fetching OHLCV data
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")

# Timeframe mapping for Data Service
TIMEFRAME_TO_INTERVAL = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1day", "W1": "1week", "MN": "1month"
}


class ValidationRequest(BaseModel):
    """Request to validate a specific pattern."""
    pattern_id: str = Field(..., description="ID of the pattern to validate")
    force: bool = Field(default=False, description="Force re-validation even if cached")


class BatchValidationRequest(BaseModel):
    """Request to validate multiple patterns."""
    pattern_ids: Optional[List[str]] = Field(
        default=None,
        description="List of pattern IDs to validate (if None, validates recent pending patterns)"
    )
    max_patterns: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of patterns to validate"
    )
    symbol: Optional[str] = Field(default=None, description="Filter by symbol")
    timeframe: Optional[str] = Field(default=None, description="Filter by timeframe")


class ValidationResponse(BaseModel):
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
    return claude_validator_service.get_status()


async def fetch_ohlcv_data(
    symbol: str,
    timeframe: str,
    limit: int = 50,
    pattern_timestamp: str = None
) -> List[dict]:
    """
    Fetch OHLCV data from Data Service.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe (M5, H1, etc.)
        limit: Number of candles to fetch
        pattern_timestamp: If provided, fetch data ending at this timestamp
                          so the pattern candle is the last one in the data
    """
    try:
        interval = TIMEFRAME_TO_INTERVAL.get(timeframe.upper(), "1h")
        url = f"{DATA_SERVICE_URL}/api/v1/twelvedata/time_series/{symbol}"

        params = {
            "interval": interval,
            "outputsize": limit + 20  # Fetch extra to ensure we have enough after filtering
        }

        # If we have a pattern timestamp, add end_date to get historical data
        if pattern_timestamp:
            # TwelveData expects end_date in format YYYY-MM-DD HH:MM:SS
            params["end_date"] = pattern_timestamp.replace("T", " ").split("+")[0].split(".")[0]
            logger.info(f"Fetching OHLCV data ending at {params['end_date']}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                candles = data.get("values", data.get("data", []))

                # Ensure newest last for chart rendering
                if candles and len(candles) > 1:
                    # Check if data is in descending order (newest first)
                    first_ts = candles[0].get("datetime", candles[0].get("timestamp", ""))
                    last_ts = candles[-1].get("datetime", candles[-1].get("timestamp", ""))
                    if first_ts > last_ts:
                        candles = list(reversed(candles))

                # If pattern_timestamp provided, find and trim to that candle
                if pattern_timestamp and candles:
                    pattern_ts_clean = pattern_timestamp.replace("T", " ").split("+")[0].split(".")[0]
                    # Find the index of the pattern candle
                    pattern_idx = None
                    for i, candle in enumerate(candles):
                        candle_ts = candle.get("datetime", candle.get("timestamp", ""))
                        # Compare timestamps (handle different formats)
                        candle_ts_clean = candle_ts.replace("T", " ").split("+")[0].split(".")[0]
                        if candle_ts_clean == pattern_ts_clean:
                            pattern_idx = i
                            break

                    if pattern_idx is not None:
                        # Return candles up to and including the pattern candle
                        # Take 'limit' candles ending at pattern_idx
                        start_idx = max(0, pattern_idx - limit + 1)
                        candles = candles[start_idx:pattern_idx + 1]
                        logger.info(f"Trimmed to {len(candles)} candles ending at pattern timestamp")
                    else:
                        logger.warning(f"Pattern timestamp {pattern_ts_clean} not found in data")
                        # Fall back to last 'limit' candles
                        candles = candles[-limit:]

                return candles[-limit:]  # Ensure we return at most 'limit' candles
            else:
                logger.warning(f"Failed to fetch OHLCV: {response.status_code}")
                return []

    except Exception as e:
        logger.error(f"Error fetching OHLCV: {e}")
        return []


@router.post("/validate")
async def validate_pattern(request: ValidationRequest):
    """
    Validate a single pattern using Claude Vision API.

    This endpoint renders a chart image of the pattern and sends it to
    Claude for visual analysis. Claude will assess whether the pattern
    is correctly identified.

    **Note**: Requires ANTHROPIC_API_KEY environment variable to be set.
    """
    try:
        # Get pattern from history
        history = pattern_history_service.get_history(limit=1000)
        pattern = next((p for p in history if p.get("id") == request.pattern_id), None)

        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {request.pattern_id} not found")

        # Get OHLCV data for this pattern
        ohlcv_data = pattern.get("ohlc_data", [])
        if not ohlcv_data:
            # Fetch from Data Service - use pattern timestamp to get correct candle
            symbol = pattern.get("symbol", "")
            timeframe = pattern.get("timeframe", "H1")
            pattern_timestamp = pattern.get("timestamp", "")

            ohlcv_data = await fetch_ohlcv_data(
                symbol,
                timeframe,
                limit=50,
                pattern_timestamp=pattern_timestamp
            )

            if not ohlcv_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not fetch OHLCV data for {symbol}/{timeframe}"
                )

        # Validate with Claude
        result = await claude_validator_service.validate_pattern(
            pattern_id=request.pattern_id,
            pattern_type=pattern.get("pattern_type", "unknown"),
            symbol=pattern.get("symbol", ""),
            timeframe=pattern.get("timeframe", ""),
            ohlcv_data=ohlcv_data,
            force=request.force
        )

        return {
            "status": "success",
            "validation": result.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-batch")
async def validate_patterns_batch(
    request: BatchValidationRequest,
    background_tasks: BackgroundTasks
):
    """
    Validate multiple patterns using Claude Vision API.

    Patterns are validated sequentially with rate limiting to respect
    API constraints. For large batches, consider using the background
    validation endpoint.

    Parameters:
    - **pattern_ids**: Specific pattern IDs to validate
    - **max_patterns**: Maximum number to validate (default: 10)
    - **symbol**: Optional symbol filter
    - **timeframe**: Optional timeframe filter
    """
    try:
        # Get patterns to validate
        if request.pattern_ids:
            history = pattern_history_service.get_history(limit=1000)
            patterns = [p for p in history if p.get("id") in request.pattern_ids]
        else:
            # Get recent patterns without Claude validation
            patterns = pattern_history_service.get_history(
                symbol=request.symbol.upper() if request.symbol else None,
                timeframe=request.timeframe.upper() if request.timeframe else None,
                limit=request.max_patterns
            )

        if not patterns:
            return {
                "status": "no_patterns",
                "message": "No patterns found matching criteria",
                "validated": 0
            }

        # Build OHLCV data map - fetch from Data Service if not present
        ohlcv_data_map = {}
        for p in patterns:
            pattern_id = p.get("id", "")
            if p.get("ohlc_data"):
                ohlcv_data_map[pattern_id] = p.get("ohlc_data")
            else:
                # Fetch from Data Service - use pattern timestamp for correct candle
                symbol = p.get("symbol", "")
                timeframe = p.get("timeframe", "H1")
                pattern_timestamp = p.get("timestamp", "")
                if symbol:
                    ohlcv_data = await fetch_ohlcv_data(
                        symbol,
                        timeframe,
                        limit=50,
                        pattern_timestamp=pattern_timestamp
                    )
                    if ohlcv_data:
                        ohlcv_data_map[pattern_id] = ohlcv_data
                    else:
                        logger.warning(f"Could not fetch OHLCV for {symbol}/{timeframe}")

        # Validate batch
        results = await claude_validator_service.validate_patterns_batch(
            patterns=patterns,
            ohlcv_data_map=ohlcv_data_map,
            max_validations=request.max_patterns
        )

        return {
            "status": "success",
            "validated": len(results),
            "results": [r.to_dict() for r in results],
            "summary": {
                "total": len(results),
                "agreed": sum(1 for r in results if r.claude_agrees),
                "rejected": sum(1 for r in results if not r.claude_agrees),
                "errors": sum(1 for r in results if r.status == ValidationStatus.ERROR)
            }
        }

    except Exception as e:
        logger.error(f"Batch validation error: {e}")
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
    Get Claude validation history.

    Returns historical validation results with optional filtering.
    """
    try:
        status_filter = None
        if status:
            try:
                status_filter = ValidationStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        history = claude_validator_service.get_validation_history(
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
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_validation_statistics():
    """
    Get Claude validation statistics.

    Returns aggregated statistics about pattern validations including:
    - Total validations performed
    - Agreement/rejection rates
    - Average quality scores
    - Breakdown by pattern type and symbol
    """
    try:
        stats = claude_validator_service.get_validation_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart-preview/{pattern_id}")
async def get_chart_preview(pattern_id: str):
    """
    Get a chart preview for a pattern without running validation.

    Returns the base64-encoded chart image that would be sent to Claude.
    Useful for debugging and previewing the chart rendering.
    """
    try:
        # Get pattern from history
        history = pattern_history_service.get_history(limit=1000)
        pattern = next((p for p in history if p.get("id") == pattern_id), None)

        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        ohlcv_data = pattern.get("ohlc_data", [])
        if not ohlcv_data:
            # Fetch from Data Service - use pattern timestamp for correct candle
            symbol = pattern.get("symbol", "")
            timeframe = pattern.get("timeframe", "H1")
            pattern_timestamp = pattern.get("timestamp", "")

            ohlcv_data = await fetch_ohlcv_data(
                symbol,
                timeframe,
                limit=50,
                pattern_timestamp=pattern_timestamp
            )

            if not ohlcv_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not fetch OHLCV data for {symbol}/{timeframe}"
                )

        # Render chart with direction-based colors for consistency with Frontend
        pattern_type = pattern.get("pattern_type", "unknown")
        chart_renderer = claude_validator_service.chart_renderer

        # Determine candle count and direction
        pattern_candles = claude_validator_service._get_pattern_candle_count(pattern_type)
        direction = claude_validator_service._get_pattern_direction(pattern_type)

        chart_base64 = chart_renderer.render_pattern_chart(
            ohlcv_data=ohlcv_data,
            pattern_type=pattern_type,
            pattern_candles=pattern_candles,
            context_before=5,
            context_after=5,
            direction=direction
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
            "chart_image": f"data:image/png;base64,{chart_base64}",
            "candles_shown": len(ohlcv_data[-20:]) if ohlcv_data else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating chart preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/problematic-patterns")
async def get_problematic_patterns(
    min_rejections: int = Query(default=2, ge=1, description="Minimum rejections to include"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results")
):
    """
    Get patterns that Claude frequently rejects.

    Identifies patterns that may need rule adjustments based on
    Claude's assessments. Useful for improving detection accuracy.
    """
    try:
        stats = claude_validator_service.get_validation_statistics()
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
            "message": f"{len(problematic)} patterns with >= {min_rejections} rejections"
        }

    except Exception as e:
        logger.error(f"Error getting problematic patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FEEDBACK ANALYSIS ENDPOINTS
# ============================================================================

class ApplyRecommendationRequest(BaseModel):
    """Request to apply a parameter recommendation."""
    pattern: str = Field(..., description="Pattern type (e.g., 'hammer')")
    parameter: str = Field(..., description="Parameter name (e.g., 'body_max_ratio')")
    new_value: float = Field(..., description="New value to set")
    reason: str = Field(..., description="Feedback reason category")
    feedback_count: int = Field(default=1, ge=1, description="Number of feedback items")


@router.get("/feedback-analysis")
async def get_feedback_analysis():
    """
    Analyze Claude validation feedback to identify improvement opportunities.

    This endpoint:
    1. Analyzes all rejection reasoning from Claude validations
    2. Extracts structured feedback categories (e.g., "body_too_large", "wrong_trend_context")
    3. Aggregates statistics by pattern type and reason

    Use the results to understand why patterns are being rejected.
    """
    try:
        # Get validation history
        history = claude_validator_service.get_validation_history(limit=500)

        # Analyze feedback
        analysis = feedback_analyzer_service.analyze_validation_history(history)

        return {
            "status": "success",
            "analysis": analysis,
            "summary": feedback_analyzer_service.get_feedback_summary()
        }

    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations():
    """
    Generate parameter adjustment recommendations based on Claude feedback.

    This endpoint analyzes rejection patterns and generates specific
    recommendations for adjusting detection parameters to reduce
    false positives.

    **Example Response:**
    ```json
    {
        "recommendations": [
            {
                "pattern": "hammer",
                "parameter": "body_max_ratio",
                "current_value": 0.35,
                "recommended_value": 0.28,
                "reason": "body_too_large",
                "feedback_count": 5,
                "confidence": 0.9,
                "priority": "high",
                "impact_estimate": "Strengerer Body-Check führt zu weniger False Positives"
            }
        ]
    }
    ```
    """
    try:
        # Get validation history
        history = claude_validator_service.get_validation_history(limit=500)

        # Analyze feedback
        feedback_stats = feedback_analyzer_service.analyze_validation_history(history)

        # Generate recommendations
        recommendations = feedback_analyzer_service.generate_recommendations(
            feedback_stats=feedback_stats,
            rule_config_service=rule_config_service
        )

        return {
            "status": "success",
            "count": len(recommendations),
            "recommendations": [r.to_dict() for r in recommendations],
            "feedback_summary": {
                "total_rejections": feedback_stats.get("total_rejections", 0),
                "patterns_analyzed": len(feedback_stats.get("by_pattern", {})),
                "reason_categories": len(feedback_stats.get("by_reason", {}))
            }
        }

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-recommendation")
async def apply_recommendation(request: ApplyRecommendationRequest):
    """
    Apply a single parameter recommendation.

    This adjusts the detection parameters based on Claude's feedback
    to improve pattern recognition accuracy.

    **Safety:**
    - Changes are logged with reason and feedback count
    - Original values can be restored via /rules/reset endpoint
    """
    try:
        # Get current value
        current_value = rule_config_service.get_param(request.pattern, request.parameter)

        if current_value is None:
            raise HTTPException(
                status_code=400,
                detail=f"Parameter {request.parameter} not found for pattern {request.pattern}"
            )

        # Safety check - don't allow extreme changes (max 2x or 0.5x)
        # This allows doubling or halving values but prevents more extreme changes
        change_pct = abs(request.new_value - current_value) / max(current_value, 0.001)
        max_change = 1.0  # Allow up to 100% change (doubling/halving)
        if change_pct > max_change:
            raise HTTPException(
                status_code=400,
                detail=f"Change too large ({change_pct*100:.0f}%). Max {max_change*100:.0f}% change allowed per adjustment."
            )

        # Apply the change
        success = rule_config_service.set_param(
            pattern=request.pattern,
            param_name=request.parameter,
            value=request.new_value,
            reason=f"claude_feedback:{request.reason}",
            feedback_count=request.feedback_count
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply recommendation")

        # Track for impact analysis
        feedback_analyzer_service.track_adjustment(
            pattern=request.pattern,
            parameter=request.parameter,
            old_value=current_value,
            new_value=request.new_value,
            reason=request.reason,
            feedback_count=request.feedback_count
        )

        logger.info(
            f"Applied recommendation: {request.pattern}.{request.parameter} "
            f"{current_value} -> {request.new_value} (reason: {request.reason})"
        )

        return {
            "status": "success",
            "pattern": request.pattern,
            "parameter": request.parameter,
            "old_value": current_value,
            "new_value": request.new_value,
            "reason": request.reason,
            "message": f"Parameter adjusted successfully. New patterns will use the updated value."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-rules")
async def get_current_rules():
    """
    Get all current detection rule parameters.

    Returns the current configuration for all pattern types,
    including any adjustments made via feedback recommendations.
    """
    try:
        params = rule_config_service.get_all_params()
        history = rule_config_service.get_adjustment_history(limit=20)

        return {
            "params": params,
            "recent_adjustments": history,
            "total_patterns": len(params)
        }

    except Exception as e:
        logger.error(f"Error getting current rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/impact-analysis")
async def get_impact_analysis():
    """
    Analyze the impact of parameter adjustments.

    Compares rejection rates before and after adjustments
    to measure improvement.
    """
    try:
        # Get all adjustments
        adjustments = rule_config_service.get_adjustment_history(limit=50)
        validation_history = claude_validator_service.get_validation_history(limit=1000)

        impacts = []
        for adj in adjustments:
            if not adj.get("timestamp"):
                continue

            impact = feedback_analyzer_service.measure_impact(
                validation_history=validation_history,
                adjustment_timestamp=adj.get("timestamp", "")
            )

            impacts.append({
                "adjustment": adj,
                "impact": impact
            })

        # Calculate overall improvement
        total_improvements = sum(1 for i in impacts if i["impact"].get("improved", False))

        return {
            "status": "success",
            "total_adjustments": len(adjustments),
            "improvements": total_improvements,
            "success_rate": round(total_improvements / len(adjustments) * 100, 1) if adjustments else 0,
            "details": impacts[-10:]  # Last 10
        }

    except Exception as e:
        logger.error(f"Error analyzing impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))
