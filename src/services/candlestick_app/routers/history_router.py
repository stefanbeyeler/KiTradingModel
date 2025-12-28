"""Pattern history endpoints."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from loguru import logger

from ..services.pattern_history_service import pattern_history_service

router = APIRouter()

# Feedback storage path
FEEDBACK_FILE = Path("/app/data/pattern_feedback.json")


class PatternFeedback(BaseModel):
    """Pattern correction feedback from user."""
    pattern_id: str
    original_pattern: str
    feedback_type: str  # confirmed, corrected, rejected
    corrected_pattern: str
    symbol: str
    timeframe: str
    timestamp: str
    ohlc_data: Optional[list] = None


@router.get("/history")
async def get_pattern_history(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    direction: Optional[str] = Query(default=None, description="Filter by direction (bullish, bearish, neutral)"),
    category: Optional[str] = Query(default=None, description="Filter by category (reversal, continuation, indecision)"),
    timeframe: Optional[str] = Query(default=None, description="Filter by timeframe (M5, M15, H1, H4, D1)"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results")
):
    """
    Get pattern history with optional filters.

    Returns detected patterns from the history, sorted by most recent first.

    Parameters:
    - **symbol**: Filter by trading symbol
    - **direction**: Filter by signal direction (bullish, bearish, neutral)
    - **category**: Filter by pattern category (reversal, continuation, indecision)
    - **timeframe**: Filter by timeframe (M5, M15, H1, H4, D1)
    - **min_confidence**: Minimum confidence threshold
    - **limit**: Maximum number of results
    """
    try:
        history = pattern_history_service.get_history(
            symbol=symbol.upper() if symbol else None,
            direction=direction.lower() if direction else None,
            category=category.lower() if category else None,
            timeframe=timeframe.upper() if timeframe else None,
            min_confidence=min_confidence,
            limit=limit,
        )

        return {
            "count": len(history),
            "patterns": history
        }

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_symbol_history(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    Get pattern history for a specific symbol.

    Returns all detected patterns for the symbol, sorted by most recent first.
    """
    try:
        history = pattern_history_service.get_history(
            symbol=symbol.upper(),
            limit=limit,
        )

        return {
            "symbol": symbol.upper(),
            "count": len(history),
            "patterns": history
        }

    except Exception as e:
        logger.error(f"Error getting symbol history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/by-symbol")
async def get_history_by_symbol():
    """
    Get latest patterns grouped by symbol.

    Returns up to 5 most recent patterns per symbol.
    """
    try:
        by_symbol = pattern_history_service.get_latest_by_symbol()

        return {
            "symbols_count": len(by_symbol),
            "by_symbol": by_symbol
        }

    except Exception as e:
        logger.error(f"Error getting history by symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/statistics")
async def get_history_statistics():
    """
    Get statistics about detected patterns.

    Returns aggregated statistics including:
    - Total patterns
    - Patterns by direction
    - Patterns by category
    - Patterns by timeframe
    - Scan status
    """
    try:
        stats = pattern_history_service.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/scan")
async def trigger_scan():
    """
    Trigger an immediate pattern scan.

    Scans all available symbols for patterns and updates the history.
    """
    try:
        new_patterns = await pattern_history_service.scan_all_symbols()

        return {
            "status": "completed",
            "new_patterns_found": new_patterns
        }

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/clear")
async def clear_history():
    """
    Clear all pattern history.

    Warning: This action cannot be undone.
    """
    try:
        pattern_history_service.clear_history()

        return {
            "status": "cleared",
            "message": "Pattern history has been cleared"
        }

    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan/status")
async def get_scan_status():
    """
    Get the current auto-scan status.

    Returns whether the periodic scan is running and its interval.
    """
    try:
        stats = pattern_history_service.get_statistics()

        return {
            "running": stats.get("scan_running", False),
            "interval_seconds": stats.get("scan_interval_seconds", 300),
            "total_patterns": stats.get("total_patterns", 0),
            "last_scan": stats.get("last_scan"),
        }

    except Exception as e:
        logger.error(f"Error getting scan status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan/start")
async def start_auto_scan():
    """
    Start the automatic pattern scanning.

    Begins periodic scanning of all symbols for patterns.
    """
    try:
        if pattern_history_service.is_running():
            return {
                "status": "already_running",
                "message": "Auto-scan is already running"
            }

        await pattern_history_service.start()

        return {
            "status": "started",
            "message": "Auto-scan has been started"
        }

    except Exception as e:
        logger.error(f"Error starting scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan/stop")
async def stop_auto_scan():
    """
    Stop the automatic pattern scanning.
    """
    try:
        if not pattern_history_service.is_running():
            return {
                "status": "not_running",
                "message": "Auto-scan is not running"
            }

        await pattern_history_service.stop()

        return {
            "status": "stopped",
            "message": "Auto-scan has been stopped"
        }

    except Exception as e:
        logger.error(f"Error stopping scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Feedback Endpoints ====================


@router.post("/feedback")
async def submit_pattern_feedback(feedback: PatternFeedback):
    """
    Submit user feedback for pattern correction.

    Stores feedback for use in model training to improve detection accuracy.

    Feedback types:
    - **confirmed**: Pattern was correctly identified
    - **corrected**: Pattern was wrong, user provides correct pattern type
    - **rejected**: No pattern exists (false positive)
    """
    try:
        # Load existing feedback
        feedback_data = []
        if FEEDBACK_FILE.exists():
            try:
                with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                feedback_data = []

        # Add new feedback
        entry = {
            "id": feedback.pattern_id,
            "original_pattern": feedback.original_pattern,
            "feedback_type": feedback.feedback_type,
            "corrected_pattern": feedback.corrected_pattern,
            "symbol": feedback.symbol,
            "timeframe": feedback.timeframe,
            "pattern_timestamp": feedback.timestamp,
            "feedback_timestamp": datetime.utcnow().isoformat(),
            "ohlc_data": feedback.ohlc_data,
        }
        feedback_data.append(entry)

        # Ensure directory exists
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Save feedback
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Pattern feedback saved: {feedback.feedback_type} for {feedback.original_pattern} -> {feedback.corrected_pattern}")

        return {
            "status": "saved",
            "message": "Feedback gespeichert",
            "total_feedback_count": len(feedback_data)
        }

    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback")
async def get_pattern_feedback(
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    Get all stored pattern feedback.

    Returns feedback entries for review and training data export.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {"count": 0, "feedback": []}

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        # Return most recent first
        feedback_data = sorted(
            feedback_data,
            key=lambda x: x.get('feedback_timestamp', ''),
            reverse=True
        )[:limit]

        return {
            "count": len(feedback_data),
            "feedback": feedback_data
        }

    except Exception as e:
        logger.error(f"Error loading feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/statistics")
async def get_feedback_statistics():
    """
    Get statistics about collected feedback.

    Returns counts by feedback type and pattern to assess training data quality.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "total": 0,
                "by_type": {},
                "by_pattern": {},
                "correction_rate": 0.0
            }

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        # Aggregate statistics
        by_type = {}
        by_pattern = {}
        corrections = 0

        for entry in feedback_data:
            ft = entry.get('feedback_type', 'unknown')
            by_type[ft] = by_type.get(ft, 0) + 1

            pattern = entry.get('original_pattern', 'unknown')
            by_pattern[pattern] = by_pattern.get(pattern, 0) + 1

            if ft in ('corrected', 'rejected'):
                corrections += 1

        total = len(feedback_data)
        correction_rate = (corrections / total * 100) if total > 0 else 0.0

        return {
            "total": total,
            "by_type": by_type,
            "by_pattern": by_pattern,
            "correction_rate": round(correction_rate, 1)
        }

    except Exception as e:
        logger.error(f"Error calculating feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
