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


class ReasonCategory(str):
    """
    Vordefinierte Begründungskategorien für Feedback.

    Diese Kategorien sind an die Regel-Parameter gebunden und ermöglichen
    automatische Regelwerk-Anpassungen basierend auf Nutzer-Feedback.
    """
    pass


# Vordefinierte Begründungskategorien (für Validierung und UI)
REASON_CATEGORIES = {
    # Körper-bezogene Gründe
    "body_too_large": {
        "label": "Körper zu gross",
        "description": "Der Kerzenkörper ist zu gross für dieses Pattern",
        "affects_param": "body_ratio",
        "adjustment": "decrease_threshold",
        "patterns": ["doji", "dragonfly_doji", "gravestone_doji", "spinning_top", "hammer", "inverted_hammer"]
    },
    "body_too_small": {
        "label": "Körper zu klein",
        "description": "Der Kerzenkörper ist zu klein für dieses Pattern",
        "affects_param": "body_ratio",
        "adjustment": "increase_threshold",
        "patterns": ["engulfing", "three_white_soldiers", "three_black_crows"]
    },

    # Schatten-bezogene Gründe
    "upper_shadow_too_short": {
        "label": "Oberer Schatten zu kurz",
        "description": "Der obere Schatten ist zu kurz für dieses Pattern",
        "affects_param": "upper_shadow_ratio",
        "adjustment": "increase_threshold",
        "patterns": ["shooting_star", "inverted_hammer", "gravestone_doji"]
    },
    "upper_shadow_too_long": {
        "label": "Oberer Schatten zu lang",
        "description": "Der obere Schatten ist zu lang für dieses Pattern",
        "affects_param": "upper_shadow_ratio",
        "adjustment": "decrease_threshold",
        "patterns": ["hammer", "dragonfly_doji"]
    },
    "lower_shadow_too_short": {
        "label": "Unterer Schatten zu kurz",
        "description": "Der untere Schatten ist zu kurz für dieses Pattern",
        "affects_param": "lower_shadow_ratio",
        "adjustment": "increase_threshold",
        "patterns": ["hammer", "hanging_man", "dragonfly_doji"]
    },
    "lower_shadow_too_long": {
        "label": "Unterer Schatten zu lang",
        "description": "Der untere Schatten ist zu lang für dieses Pattern",
        "affects_param": "lower_shadow_ratio",
        "adjustment": "decrease_threshold",
        "patterns": ["shooting_star", "gravestone_doji"]
    },

    # Engulfing-spezifische Gründe
    "not_fully_engulfing": {
        "label": "Nicht vollständig umschliessend",
        "description": "Die Kerze umschliesst die vorherige nicht vollständig",
        "affects_param": "engulf_threshold",
        "adjustment": "increase_strictness",
        "patterns": ["bullish_engulfing", "bearish_engulfing"]
    },
    "engulfing_too_small": {
        "label": "Engulfing-Kerze zu klein",
        "description": "Die engulfing Kerze ist nicht gross genug im Vergleich zur vorherigen",
        "affects_param": "engulf_size_ratio",
        "adjustment": "increase_threshold",
        "patterns": ["bullish_engulfing", "bearish_engulfing"]
    },

    # Trend-Kontext-Gründe
    "wrong_trend_context": {
        "label": "Falscher Trend-Kontext",
        "description": "Das Pattern erscheint im falschen Trend-Kontext",
        "affects_param": "trend_detection",
        "adjustment": "adjust_trend_sensitivity",
        "patterns": ["hammer", "shooting_star", "morning_star", "evening_star", "hanging_man"]
    },
    "no_prior_trend": {
        "label": "Kein vorheriger Trend",
        "description": "Vor dem Pattern fehlt ein klarer Trend",
        "affects_param": "trend_lookback",
        "adjustment": "increase_lookback",
        "patterns": ["hammer", "shooting_star", "engulfing"]
    },

    # Gap-bezogene Gründe
    "missing_gap": {
        "label": "Fehlendes Gap",
        "description": "Das erforderliche Gap fehlt oder ist zu klein",
        "affects_param": "gap_threshold",
        "adjustment": "decrease_threshold",
        "patterns": ["morning_star", "evening_star", "shooting_star", "hanging_man"]
    },

    # Volumen-bezogene Gründe
    "volume_too_low": {
        "label": "Volumen zu niedrig",
        "description": "Das Volumen ist zu niedrig für eine valide Bestätigung",
        "affects_param": "volume_threshold",
        "adjustment": "context_only",
        "patterns": ["all"]
    },

    # Allgemeine Gründe
    "false_positive": {
        "label": "Falsch-positiv",
        "description": "Das Pattern wurde erkannt, ist aber keines (allgemeiner Grund)",
        "affects_param": "confidence_threshold",
        "adjustment": "increase_threshold",
        "patterns": ["all"]
    },
    "other": {
        "label": "Anderer Grund",
        "description": "Ein anderer, nicht aufgelisteter Grund",
        "affects_param": None,
        "adjustment": None,
        "patterns": ["all"]
    }
}


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
    # Neue Felder für strukturierte Begründungen
    reason_category: Optional[str] = None  # z.B. "body_too_large", "wrong_trend_context"
    reason_text: Optional[str] = None  # Freitext für zusätzliche Details


def _load_feedback_status_map() -> dict:
    """Load feedback data and create a lookup map by pattern_id."""
    feedback_map = {}
    try:
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
                for entry in feedback_data:
                    pattern_id = entry.get("id") or entry.get("pattern_id")
                    if pattern_id:
                        feedback_map[pattern_id] = {
                            "feedback_status": entry.get("feedback_type", "unknown"),
                            "corrected_pattern": entry.get("corrected_pattern"),
                            "reason_category": entry.get("reason_category"),
                            "reason_text": entry.get("reason_text"),
                            "feedback_timestamp": entry.get("feedback_timestamp"),
                        }
    except Exception as e:
        logger.warning(f"Failed to load feedback map: {e}")
    return feedback_map


@router.get("/history")
async def get_pattern_history(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    direction: Optional[str] = Query(default=None, description="Filter by direction (bullish, bearish, neutral)"),
    category: Optional[str] = Query(default=None, description="Filter by category (reversal, continuation, indecision)"),
    timeframe: Optional[str] = Query(default=None, description="Filter by timeframe (M5, M15, H1, H4, D1)"),
    feedback_status: Optional[str] = Query(
        default=None,
        description="Filter by feedback status (pending, confirmed, corrected, rejected)"
    ),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results")
):
    """
    Get pattern history with optional filters.

    Returns detected patterns from the history, sorted by most recent first.
    Each pattern is enriched with its feedback status if available.

    Parameters:
    - **symbol**: Filter by trading symbol
    - **direction**: Filter by signal direction (bullish, bearish, neutral)
    - **category**: Filter by pattern category (reversal, continuation, indecision)
    - **timeframe**: Filter by timeframe (M5, M15, H1, H4, D1)
    - **feedback_status**: Filter by feedback status (pending, confirmed, corrected, rejected)
    - **min_confidence**: Minimum confidence threshold
    - **limit**: Maximum number of results
    """
    try:
        # Get base history
        # When filtering by feedback_status, we need to fetch all patterns first
        # because feedback status is stored separately
        fetch_limit = 1000 if feedback_status else limit
        history = pattern_history_service.get_history(
            symbol=symbol.upper() if symbol else None,
            direction=direction.lower() if direction else None,
            category=category.lower() if category else None,
            timeframe=timeframe.upper() if timeframe else None,
            min_confidence=min_confidence,
            limit=fetch_limit,
        )

        # Load feedback status map
        feedback_map = _load_feedback_status_map()

        # Enrich patterns with feedback status
        enriched_history = []
        for pattern in history:
            pattern_id = pattern.get("id")
            if pattern_id and pattern_id in feedback_map:
                pattern["feedback_status"] = feedback_map[pattern_id]["feedback_status"]
                pattern["feedback_details"] = feedback_map[pattern_id]
            else:
                pattern["feedback_status"] = "pending"
                pattern["feedback_details"] = None

            # Apply feedback_status filter if specified
            if feedback_status:
                if pattern["feedback_status"] != feedback_status.lower():
                    continue

            enriched_history.append(pattern)

            if len(enriched_history) >= limit:
                break

        return {
            "count": len(enriched_history),
            "patterns": enriched_history
        }

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WICHTIG: Spezifische Routen MÜSSEN vor der generischen {symbol} Route kommen!
# FastAPI matched Routen in der Reihenfolge der Definition.


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


# Generische {symbol} Route MUSS nach den spezifischen Routen kommen!
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
            # Neue Felder für strukturierte Begründungen
            "reason_category": feedback.reason_category,
            "reason_text": feedback.reason_text,
        }

        # Validiere reason_category wenn angegeben
        if feedback.reason_category and feedback.reason_category not in REASON_CATEGORIES:
            logger.warning(f"Unknown reason category: {feedback.reason_category}")
            # Trotzdem speichern, aber als "other" klassifizieren
            entry["reason_category_valid"] = False
        else:
            entry["reason_category_valid"] = True

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


@router.get("/feedback/reason-categories")
async def get_reason_categories(pattern: Optional[str] = Query(default=None)):
    """
    Get available reason categories for feedback.

    Returns all predefined reason categories that can be used when
    correcting or rejecting a pattern. Each category is linked to
    specific rule parameters that can be automatically adjusted.

    Parameters:
    - **pattern**: Optional pattern type to filter relevant categories
    """
    if pattern:
        # Filter categories relevant for this pattern
        pattern_lower = pattern.lower()
        filtered = {}
        for key, info in REASON_CATEGORIES.items():
            patterns_list = info.get("patterns", [])
            if "all" in patterns_list or pattern_lower in patterns_list:
                filtered[key] = {
                    "label": info["label"],
                    "description": info["description"],
                }
        return {
            "pattern": pattern,
            "categories": filtered,
            "count": len(filtered)
        }

    # Return all categories
    all_categories = {}
    for key, info in REASON_CATEGORIES.items():
        all_categories[key] = {
            "label": info["label"],
            "description": info["description"],
            "applicable_patterns": info.get("patterns", [])
        }

    return {
        "categories": all_categories,
        "count": len(all_categories)
    }


@router.get("/feedback/reason-statistics")
async def get_reason_statistics():
    """
    Get statistics about feedback reasons.

    Analyzes the distribution of reason categories to identify
    common issues with the pattern detection rules.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "total_with_reasons": 0,
                "by_category": {},
                "by_pattern_and_reason": {},
                "adjustment_recommendations": []
            }

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        # Aggregate by reason category
        by_category = {}
        by_pattern_and_reason = {}
        total_with_reasons = 0

        for entry in feedback_data:
            reason = entry.get('reason_category')
            pattern = entry.get('original_pattern', 'unknown')

            if reason:
                total_with_reasons += 1
                by_category[reason] = by_category.get(reason, 0) + 1

                # Track by pattern and reason combination
                key = f"{pattern}:{reason}"
                by_pattern_and_reason[key] = by_pattern_and_reason.get(key, 0) + 1

        # Generate adjustment recommendations based on frequent issues
        recommendations = []
        for key, count in sorted(by_pattern_and_reason.items(), key=lambda x: -x[1]):
            if count >= 3:  # Threshold for recommendation
                pattern, reason = key.split(":", 1)
                if reason in REASON_CATEGORIES:
                    cat_info = REASON_CATEGORIES[reason]
                    if cat_info.get("affects_param"):
                        recommendations.append({
                            "pattern": pattern,
                            "reason": reason,
                            "reason_label": cat_info["label"],
                            "count": count,
                            "affected_param": cat_info["affects_param"],
                            "suggested_adjustment": cat_info["adjustment"],
                            "priority": "high" if count >= 5 else "medium"
                        })

        return {
            "total_with_reasons": total_with_reasons,
            "by_category": by_category,
            "by_pattern_and_reason": by_pattern_and_reason,
            "adjustment_recommendations": recommendations[:10]  # Top 10
        }

    except Exception as e:
        logger.error(f"Error calculating reason stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Re-Validation Endpoints ====================


class RevalidationRequest(BaseModel):
    """Request to mark a feedback entry as revalidated."""
    feedback_id: str
    validation_result: str  # "correct", "still_wrong", "now_correct"
    notes: Optional[str] = None
    corrected_pattern: Optional[str] = None  # New corrected pattern if still_wrong


@router.get("/feedback/pending-revalidation")
async def get_pending_revalidation():
    """
    Get corrected/rejected patterns that haven't been revalidated after training.

    Returns patterns that were marked as incorrect by users and should be
    checked again after model retraining to verify improvement.

    Use this after training a new model to review previously problematic patterns.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "count": 0,
                "pending": [],
                "message": "Keine Feedback-Daten vorhanden"
            }

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        # Filter for corrected/rejected patterns that haven't been revalidated
        pending = []
        for entry in feedback_data:
            feedback_type = entry.get('feedback_type', '')
            # Only include corrections and rejections (not confirmations)
            if feedback_type in ('corrected', 'rejected'):
                # Check if already revalidated
                if not entry.get('revalidated', False):
                    pending.append({
                        "id": entry.get("id", ""),
                        "original_pattern": entry.get("original_pattern", ""),
                        "corrected_pattern": entry.get("corrected_pattern", ""),
                        "feedback_type": feedback_type,
                        "symbol": entry.get("symbol", ""),
                        "timeframe": entry.get("timeframe", ""),
                        "pattern_timestamp": entry.get("pattern_timestamp", ""),
                        "feedback_timestamp": entry.get("feedback_timestamp", ""),
                        "ohlc_data": entry.get("ohlc_data"),
                        # Begründungen hinzufügen
                        "reason_category": entry.get("reason_category"),
                        "reason_text": entry.get("reason_text"),
                    })

        # Sort by feedback timestamp (most recent first)
        pending = sorted(
            pending,
            key=lambda x: x.get('feedback_timestamp', ''),
            reverse=True
        )

        return {
            "count": len(pending),
            "pending": pending,
            "message": f"{len(pending)} Patterns zur Re-Validierung verfügbar"
        }

    except Exception as e:
        logger.error(f"Error getting pending revalidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/revalidate")
async def mark_as_revalidated(request: RevalidationRequest):
    """
    Mark a feedback entry as revalidated after model retraining.

    Validation results:
    - **correct**: The new model now correctly identifies this pattern
    - **still_wrong**: The model still misidentifies this pattern
    - **now_correct**: Previously rejected pattern is now correctly not detected

    This helps track model improvement over time.
    """
    try:
        if not FEEDBACK_FILE.exists():
            raise HTTPException(status_code=404, detail="Keine Feedback-Daten vorhanden")

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        # Find and update the entry
        found = False
        for entry in feedback_data:
            if entry.get("id") == request.feedback_id:
                entry["revalidated"] = True
                entry["revalidation_result"] = request.validation_result
                entry["revalidation_timestamp"] = datetime.utcnow().isoformat()
                if request.notes:
                    entry["revalidation_notes"] = request.notes
                # Store new correction if provided (when still_wrong)
                if request.corrected_pattern:
                    entry["revalidation_corrected_pattern"] = request.corrected_pattern
                    # Also update the main corrected_pattern for future training
                    entry["corrected_pattern"] = request.corrected_pattern
                    # Change feedback type to corrected if it was rejected
                    if entry.get("feedback_type") == "rejected":
                        entry["feedback_type"] = "corrected"
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail=f"Feedback-Eintrag {request.feedback_id} nicht gefunden")

        # Save updated feedback
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Revalidation marked: {request.feedback_id} -> {request.validation_result}")

        return {
            "status": "success",
            "message": f"Re-Validierung gespeichert: {request.validation_result}",
            "feedback_id": request.feedback_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking revalidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/reset-revalidation")
async def reset_revalidation_status():
    """
    Reset revalidation status for all feedback entries.

    Call this after training a new model to make all previously
    corrected/rejected patterns available for re-checking.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "status": "success",
                "reset_count": 0,
                "message": "Keine Feedback-Daten vorhanden"
            }

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        reset_count = 0
        for entry in feedback_data:
            if entry.get('revalidated', False):
                entry['revalidated'] = False
                # Keep historical revalidation data
                if 'revalidation_history' not in entry:
                    entry['revalidation_history'] = []
                if entry.get('revalidation_result'):
                    entry['revalidation_history'].append({
                        "result": entry.get('revalidation_result'),
                        "timestamp": entry.get('revalidation_timestamp'),
                        "notes": entry.get('revalidation_notes'),
                    })
                # Clear current revalidation
                entry.pop('revalidation_result', None)
                entry.pop('revalidation_timestamp', None)
                entry.pop('revalidation_notes', None)
                reset_count += 1

        # Save updated feedback
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Reset revalidation status for {reset_count} entries")

        return {
            "status": "success",
            "reset_count": reset_count,
            "message": f"{reset_count} Einträge für Re-Validierung zurückgesetzt"
        }

    except Exception as e:
        logger.error(f"Error resetting revalidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/revalidation-statistics")
async def get_revalidation_statistics():
    """
    Get statistics about revalidation results.

    Shows how many corrected patterns are now correctly identified
    after model retraining - useful for tracking model improvement.
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "total_feedback": 0,
                "pending_revalidation": 0,
                "revalidated": 0,
                "results": {},
                "improvement_rate": 0.0
            }

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        # Count statistics
        total_corrections = 0  # corrected + rejected
        pending = 0
        revalidated = 0
        results = {}

        for entry in feedback_data:
            feedback_type = entry.get('feedback_type', '')
            if feedback_type in ('corrected', 'rejected'):
                total_corrections += 1
                if entry.get('revalidated', False):
                    revalidated += 1
                    result = entry.get('revalidation_result', 'unknown')
                    results[result] = results.get(result, 0) + 1
                else:
                    pending += 1

        # Calculate improvement rate (patterns now correctly identified)
        improved = results.get('correct', 0) + results.get('now_correct', 0)
        improvement_rate = (improved / revalidated * 100) if revalidated > 0 else 0.0

        # Find last training and revalidation timestamps
        last_training = None
        last_revalidation = None

        for entry in feedback_data:
            # Check for revalidation timestamp
            reval_ts = entry.get('revalidation_timestamp')
            if reval_ts:
                if not last_revalidation or reval_ts > last_revalidation:
                    last_revalidation = reval_ts

        # Check for last training from training service (if available)
        try:
            from ..services.pattern_detection_service import candlestick_pattern_service
            # Try to get model info which may contain last training date
            model_info = getattr(candlestick_pattern_service, '_last_model_load', None)
            if model_info:
                last_training = model_info
        except Exception:
            pass

        return {
            "total_feedback": len(feedback_data),
            "total_corrections": total_corrections,
            "pending_revalidation": pending,
            "revalidated": revalidated,
            "results": results,
            "improvement_rate": round(improvement_rate, 1),
            "message": f"Verbesserungsrate: {improvement_rate:.1f}% ({improved}/{revalidated} Patterns)",
            "last_training": last_training,
            "last_revalidation": last_revalidation
        }

    except Exception as e:
        logger.error(f"Error getting revalidation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Rule Configuration Endpoints ====================


class RuleParamUpdate(BaseModel):
    """Request to update a rule parameter."""
    pattern: str
    parameter: str
    value: float
    reason: Optional[str] = None


class ApplyRecommendationRequest(BaseModel):
    """Request to apply a parameter recommendation."""
    pattern: str
    parameter: str
    new_value: float
    reason: str
    feedback_count: int


@router.get("/rules/params")
async def get_rule_params(pattern: Optional[str] = Query(default=None)):
    """
    Get current rule parameters.

    Returns all configurable detection parameters that can be adjusted
    based on user feedback.

    Parameters:
    - **pattern**: Optional pattern type to get specific parameters
    """
    from ..services.rule_config_service import rule_config_service

    if pattern:
        params = rule_config_service.get_pattern_params(pattern)
        return {
            "pattern": pattern,
            "params": params
        }

    return {
        "params": rule_config_service.get_all_params()
    }


@router.post("/rules/params")
async def update_rule_param(update: RuleParamUpdate):
    """
    Update a specific rule parameter.

    Allows manual adjustment of detection thresholds.
    Changes are persisted and affect future pattern detection.
    """
    from ..services.rule_config_service import rule_config_service

    success = rule_config_service.set_param(
        pattern=update.pattern,
        param_name=update.parameter,
        value=update.value,
        reason=update.reason or "manual_update"
    )

    if success:
        return {
            "status": "success",
            "message": f"Parameter {update.pattern}.{update.parameter} auf {update.value} gesetzt",
            "current_params": rule_config_service.get_pattern_params(update.pattern)
        }
    else:
        raise HTTPException(status_code=400, detail="Parameter konnte nicht gesetzt werden")


@router.post("/rules/reset")
async def reset_rule_params(pattern: Optional[str] = Query(default=None)):
    """
    Reset rule parameters to defaults.

    Parameters:
    - **pattern**: Specific pattern to reset, or all if not specified
    """
    from ..services.rule_config_service import rule_config_service

    if pattern:
        from ..services.rule_config_service import DEFAULT_RULE_PARAMS
        if pattern.lower() in DEFAULT_RULE_PARAMS:
            for param_name in DEFAULT_RULE_PARAMS[pattern.lower()]:
                rule_config_service.reset_param(pattern, param_name)
            return {
                "status": "success",
                "message": f"Parameter für {pattern} zurückgesetzt",
                "params": rule_config_service.get_pattern_params(pattern)
            }
        else:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern} nicht gefunden")

    # Reset all
    rule_config_service.reset_all()
    return {
        "status": "success",
        "message": "Alle Parameter auf Standardwerte zurückgesetzt"
    }


@router.get("/rules/history")
async def get_rule_history(limit: int = Query(default=50, ge=1, le=200)):
    """
    Get history of rule parameter adjustments.

    Shows when and why parameters were changed.
    """
    from ..services.rule_config_service import rule_config_service

    history = rule_config_service.get_adjustment_history(limit=limit)
    return {
        "count": len(history),
        "history": history
    }


@router.get("/rules/recommendations")
async def get_rule_recommendations():
    """
    Get parameter adjustment recommendations based on feedback.

    Analyzes user feedback patterns to suggest threshold adjustments
    that could improve detection accuracy.
    """
    from ..services.rule_config_service import rule_config_service

    # First get feedback statistics
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "recommendations": [],
                "message": "Keine Feedback-Daten für Empfehlungen vorhanden"
            }

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)

        # Build reason statistics
        by_pattern_and_reason = {}
        for entry in feedback_data:
            reason = entry.get('reason_category')
            pattern = entry.get('original_pattern', 'unknown')

            if reason:
                key = f"{pattern}:{reason}"
                by_pattern_and_reason[key] = by_pattern_and_reason.get(key, 0) + 1

        feedback_stats = {
            "by_pattern_and_reason": by_pattern_and_reason
        }

        recommendations = rule_config_service.generate_recommendations(feedback_stats)

        return {
            "recommendations": recommendations,
            "total_feedback_entries": len(feedback_data),
            "entries_with_reasons": sum(by_pattern_and_reason.values()),
            "message": f"{len(recommendations)} Anpassungsempfehlungen basierend auf Feedback"
        }

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rules/apply-recommendation")
async def apply_rule_recommendation(request: ApplyRecommendationRequest):
    """
    Apply a specific parameter recommendation.

    Takes a recommendation from /rules/recommendations and applies it.
    """
    from ..services.rule_config_service import rule_config_service

    success = rule_config_service.apply_recommendation(
        pattern=request.pattern,
        param_name=request.parameter,
        new_value=request.new_value,
        reason=request.reason,
        feedback_count=request.feedback_count
    )

    if success:
        return {
            "status": "success",
            "message": f"Empfehlung angewendet: {request.pattern}.{request.parameter} = {request.new_value}",
            "current_params": rule_config_service.get_pattern_params(request.pattern)
        }
    else:
        raise HTTPException(status_code=400, detail="Empfehlung konnte nicht angewendet werden")
