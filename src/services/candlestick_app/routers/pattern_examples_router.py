"""Pattern example chart endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field
import base64

from ..services.pattern_example_service import pattern_example_service


router = APIRouter()


class PatternExampleInfo(BaseModel):
    """Information about a pattern example."""
    pattern_type: str = Field(..., description="Pattern type identifier")
    direction: str = Field(..., description="Pattern direction: bullish, bearish, neutral")
    candles: int = Field(..., description="Number of candles in the pattern")
    context: str = Field(..., description="Required trend context: uptrend, downtrend, any")


class PatternExampleResponse(BaseModel):
    """Response containing a pattern example chart."""
    pattern_type: str = Field(..., description="Pattern type identifier")
    image_base64: str = Field(..., description="Base64 encoded PNG image")
    info: PatternExampleInfo = Field(..., description="Pattern metadata")


class AllPatternsResponse(BaseModel):
    """Response containing all pattern examples."""
    count: int = Field(..., description="Number of patterns")
    patterns: List[PatternExampleInfo] = Field(..., description="List of available patterns")


@router.get("/list", response_model=AllPatternsResponse)
async def list_pattern_examples():
    """
    List all available pattern example types.

    Returns pattern types with their metadata (direction, candle count, context).
    """
    pattern_types = pattern_example_service.get_available_patterns()
    patterns = []

    for pt in pattern_types:
        info = pattern_example_service.get_pattern_info(pt)
        if info:
            patterns.append(PatternExampleInfo(
                pattern_type=pt,
                direction=info.get("direction", "neutral"),
                candles=info.get("candles", 1),
                context=info.get("context", "any")
            ))

    return AllPatternsResponse(
        count=len(patterns),
        patterns=patterns
    )


@router.get("/chart/{pattern_type}")
async def get_pattern_example_chart(
    pattern_type: str,
    compact: bool = Query(default=False, description="Smaller chart for thumbnails"),
    show_labels: bool = Query(default=True, description="Show pattern labels and annotations"),
    format: str = Query(default="json", description="Response format: json or image")
):
    """
    Get an example chart for a specific candlestick pattern.

    Returns a synthetically generated "textbook" example of the pattern
    with proper trend context and highlighting.

    Parameters:
    - **pattern_type**: Pattern name (e.g., hammer, bullish_engulfing, morning_star)
    - **compact**: Generate smaller thumbnail-sized chart
    - **show_labels**: Include pattern name and annotations
    - **format**: 'json' returns base64, 'image' returns raw PNG
    """
    # Validate pattern type
    info = pattern_example_service.get_pattern_info(pattern_type)
    if not info:
        available = pattern_example_service.get_available_patterns()
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pattern type: {pattern_type}. Available: {', '.join(available)}"
        )

    # Render chart
    image_base64 = pattern_example_service.render_example_chart(
        pattern_type=pattern_type,
        show_labels=show_labels,
        compact=compact
    )

    if not image_base64:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render chart for pattern: {pattern_type}"
        )

    # Return as raw image if requested
    if format.lower() == "image":
        image_bytes = base64.b64decode(image_base64)
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={pattern_type}_example.png"}
        )

    # Return JSON response
    return PatternExampleResponse(
        pattern_type=pattern_type,
        image_base64=image_base64,
        info=PatternExampleInfo(
            pattern_type=pattern_type,
            direction=info.get("direction", "neutral"),
            candles=info.get("candles", 1),
            context=info.get("context", "any")
        )
    )


@router.get("/all")
async def get_all_pattern_examples(
    compact: bool = Query(default=True, description="Generate compact thumbnails")
):
    """
    Generate example charts for all available patterns.

    Returns a dictionary mapping pattern types to their base64 images.
    This is useful for generating a pattern gallery or documentation.

    Note: This may take a few seconds as it generates 32+ charts.
    """
    try:
        examples = pattern_example_service.get_all_examples(compact=compact)

        # Add metadata
        result = {}
        for pattern_type, image in examples.items():
            info = pattern_example_service.get_pattern_info(pattern_type)
            result[pattern_type] = {
                "image_base64": image,
                "direction": info.get("direction", "neutral") if info else "neutral",
                "candles": info.get("candles", 1) if info else 1,
                "context": info.get("context", "any") if info else "any"
            }

        return {
            "count": len(result),
            "patterns": result
        }

    except Exception as e:
        logger.error(f"Failed to generate all examples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gallery")
async def get_pattern_gallery(
    category: Optional[str] = Query(default=None, description="Filter by direction: bullish, bearish, neutral"),
    candles: Optional[int] = Query(default=None, ge=1, le=5, description="Filter by candle count")
):
    """
    Get a gallery of pattern examples with filtering options.

    Parameters:
    - **category**: Filter by direction (bullish, bearish, neutral)
    - **candles**: Filter by number of candles in pattern (1-5)
    """
    all_patterns = pattern_example_service.get_available_patterns()

    # Apply filters
    filtered = []
    for pt in all_patterns:
        info = pattern_example_service.get_pattern_info(pt)
        if not info:
            continue

        # Filter by category/direction
        if category and info.get("direction") != category.lower():
            continue

        # Filter by candle count
        if candles and info.get("candles") != candles:
            continue

        # Generate compact chart
        image = pattern_example_service.render_example_chart(pt, show_labels=True, compact=True)
        if image:
            filtered.append({
                "pattern_type": pt,
                "display_name": pt.replace("_", " ").title(),
                "direction": info.get("direction", "neutral"),
                "candles": info.get("candles", 1),
                "context": info.get("context", "any"),
                "image_base64": image
            })

    # Group by direction
    grouped = {
        "bullish": [],
        "bearish": [],
        "neutral": []
    }
    for p in filtered:
        direction = p["direction"]
        if direction in grouped:
            grouped[direction].append(p)

    return {
        "total": len(filtered),
        "filters": {
            "category": category,
            "candles": candles
        },
        "patterns": filtered,
        "by_direction": {
            k: len(v) for k, v in grouped.items()
        }
    }
