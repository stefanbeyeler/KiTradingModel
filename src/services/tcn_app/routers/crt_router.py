"""CRT (Candle Range Theory) API Routes.

Provides endpoints for CRT detection, signals, and analysis.
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from ..services.crt import (
    crt_signal_service,
    session_manager,
    range_tracker,
    CRTState,
)

router = APIRouter()

UTC_TZ = timezone.utc


# ============================================================================
# Pydantic Models
# ============================================================================

class CRTStatusResponse(BaseModel):
    """Response model for CRT status."""
    symbol: str
    has_active_range: bool
    state: Optional[str] = None
    range: Optional[dict] = None
    session_info: dict
    next_key_session: dict


class CRTSignalResponse(BaseModel):
    """Response model for CRT signal."""
    symbol: str
    has_signal: bool
    signal: Optional[dict] = None
    message: str


class CRTAnalysisResponse(BaseModel):
    """Response model for full CRT analysis."""
    symbol: str
    timestamp: str
    session_info: dict
    crt_status: dict
    signal: Optional[dict] = None
    service_alignment: Optional[dict] = None


class CRTScanResponse(BaseModel):
    """Response model for CRT scan."""
    timestamp: str
    symbols_scanned: int
    signals_found: int
    signals: List[dict]
    min_confidence: float


class SessionInfoResponse(BaseModel):
    """Response model for session info."""
    current_time_utc: str
    current_time_est: str
    current_session: str
    is_key_session_hour: bool
    h4_candle: dict
    next_key_session: dict
    is_weekend: bool


class RangeCreateRequest(BaseModel):
    """Request model for creating a CRT range."""
    symbol: str = Field(..., description="Trading symbol")
    open_price: float = Field(..., description="H4 candle open")
    high_price: float = Field(..., description="H4 candle high")
    low_price: float = Field(..., description="H4 candle low")
    close_price: float = Field(..., description="H4 candle close")
    volume: float = Field(default=0, description="H4 candle volume")
    candle_start: Optional[str] = Field(default=None, description="Candle start time (ISO format)")


class PriceUpdateRequest(BaseModel):
    """Request model for price update."""
    symbol: str = Field(..., description="Trading symbol")
    price: float = Field(..., description="Current price")
    is_candle_close: bool = Field(default=False, description="Is this a candle close?")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/crt/session", response_model=SessionInfoResponse)
async def get_session_info():
    """
    Get current trading session information.

    Returns EST time, current session type, and H4 candle boundaries.
    Key sessions: 1 AM EST (London Pre), 5 AM EST (London), 9 AM EST (NY).
    """
    info = session_manager.get_session_info()
    return SessionInfoResponse(**info)


@router.get("/crt/status/{symbol}", response_model=CRTStatusResponse)
async def get_crt_status(symbol: str):
    """
    Get CRT status for a symbol.

    Returns active range information, current state, and session details.
    """
    symbol = symbol.upper()
    crt_range = range_tracker.get_active_range(symbol)
    session_info = session_manager.get_session_info()

    return CRTStatusResponse(
        symbol=symbol,
        has_active_range=crt_range is not None,
        state=crt_range.state.value if crt_range else None,
        range=crt_range.to_dict() if crt_range else None,
        session_info=session_info,
        next_key_session=session_manager.get_next_key_session(),
    )


@router.get("/crt/signal/{symbol}", response_model=CRTSignalResponse)
async def get_crt_signal(
    symbol: str,
    include_service_integration: bool = Query(
        default=True,
        description="Include HMM/NHITS/TCN alignment checks"
    ),
):
    """
    Get active CRT signal for a symbol.

    Returns trading signal with entry, stop loss, take profit, and confidence scores.
    """
    symbol = symbol.upper()
    crt_range = range_tracker.get_active_range(symbol)

    if not crt_range:
        return CRTSignalResponse(
            symbol=symbol,
            has_signal=False,
            signal=None,
            message="No active CRT range for this symbol",
        )

    if not crt_range.has_signal:
        return CRTSignalResponse(
            symbol=symbol,
            has_signal=False,
            signal=None,
            message=f"Range active but no signal yet (state: {crt_range.state.value})",
        )

    # Generate full signal with service integration
    try:
        analysis = await crt_signal_service.analyze_symbol(
            symbol, include_service_integration
        )

        if analysis.get("signal"):
            return CRTSignalResponse(
                symbol=symbol,
                has_signal=True,
                signal=analysis["signal"],
                message="CRT signal active",
            )
        else:
            return CRTSignalResponse(
                symbol=symbol,
                has_signal=False,
                signal=None,
                message="Signal generation failed",
            )

    except Exception as e:
        logger.error(f"Error generating CRT signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crt/analyze", response_model=CRTAnalysisResponse)
async def analyze_symbol(
    symbol: str = Query(..., description="Trading symbol"),
    include_service_integration: bool = Query(
        default=True,
        description="Include HMM/NHITS/TCN alignment"
    ),
):
    """
    Perform full CRT analysis for a symbol.

    Returns complete analysis including session info, range status,
    potential signal, and service alignments.
    """
    symbol = symbol.upper()

    try:
        analysis = await crt_signal_service.analyze_symbol(
            symbol, include_service_integration
        )

        return CRTAnalysisResponse(
            symbol=analysis["symbol"],
            timestamp=analysis["timestamp"],
            session_info=analysis["session_info"],
            crt_status=analysis["crt_status"],
            signal=analysis.get("signal"),
            service_alignment=analysis.get("service_alignment"),
        )

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crt/scan", response_model=CRTScanResponse)
async def scan_for_signals(
    symbols: Optional[str] = Query(
        default=None,
        description="Comma-separated list of symbols (empty = all active ranges)"
    ),
    min_confidence: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    ),
):
    """
    Scan symbols for CRT signals.

    Returns all signals above the confidence threshold.
    """
    # Determine symbols to scan
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        # Scan all active ranges
        active_ranges = range_tracker.get_all_active_ranges()
        symbol_list = list(active_ranges.keys())

    if not symbol_list:
        return CRTScanResponse(
            timestamp=datetime.now(UTC_TZ).isoformat(),
            symbols_scanned=0,
            signals_found=0,
            signals=[],
            min_confidence=min_confidence,
        )

    try:
        signals = await crt_signal_service.scan_symbols(symbol_list, min_confidence)

        return CRTScanResponse(
            timestamp=datetime.now(UTC_TZ).isoformat(),
            symbols_scanned=len(symbol_list),
            signals_found=len(signals),
            signals=[s.to_dict() for s in signals],
            min_confidence=min_confidence,
        )

    except Exception as e:
        logger.error(f"Error scanning symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crt/range/create")
async def create_crt_range(request: RangeCreateRequest):
    """
    Manually create a CRT range from H4 candle data.

    Use this when you have H4 candle data and want to start tracking a range.
    """
    symbol = request.symbol.upper()

    # Parse candle start time
    if request.candle_start:
        try:
            candle_start = datetime.fromisoformat(
                request.candle_start.replace("Z", "+00:00")
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid candle_start format. Use ISO 8601."
            )
    else:
        candle_start = session_manager.get_current_h4_candle()["start_utc"]

    # Check if this is a key session candle
    from src.utils.session_utils import get_session_for_h4_candle
    session_type = get_session_for_h4_candle(candle_start)

    if not session_type:
        return {
            "success": False,
            "message": "This H4 candle is not a key session candle (1/5/9 AM EST)",
            "candle_start": candle_start.isoformat(),
        }

    # Create range
    crt_range = range_tracker.create_range(
        symbol=symbol,
        session_type=session_type,
        candle_start=candle_start,
        candle_end=candle_start.replace(hour=candle_start.hour + 4),
        open_price=request.open_price,
        high_price=request.high_price,
        low_price=request.low_price,
        close_price=request.close_price,
        volume=request.volume,
    )

    if crt_range:
        return {
            "success": True,
            "message": "CRT range created",
            "range": crt_range.to_dict(),
        }
    else:
        return {
            "success": False,
            "message": "Failed to create range (may be invalid size)",
        }


@router.post("/crt/price/update")
async def update_price(request: PriceUpdateRequest):
    """
    Update CRT range with new price data.

    This triggers the state machine to check for purge/re-entry.
    """
    symbol = request.symbol.upper()

    crt_range = range_tracker.update_price(
        symbol=symbol,
        current_price=request.price,
        is_candle_close=request.is_candle_close,
    )

    if not crt_range:
        return {
            "symbol": symbol,
            "updated": False,
            "message": "No active range for this symbol",
        }

    result = {
        "symbol": symbol,
        "updated": True,
        "state": crt_range.state.value,
        "range": crt_range.to_dict(),
    }

    # Check if we just got a signal
    if crt_range.has_signal:
        result["signal_generated"] = True
        result["signal_direction"] = crt_range.signal_direction

    return result


@router.delete("/crt/range/{symbol}")
async def invalidate_range(
    symbol: str,
    reason: str = Query(default="Manual invalidation", description="Reason for invalidation"),
):
    """
    Manually invalidate a CRT range.

    Use this when you want to cancel an active range.
    """
    symbol = symbol.upper()

    success = range_tracker.invalidate_range(symbol, reason)

    return {
        "symbol": symbol,
        "invalidated": success,
        "reason": reason if success else "No active range found",
    }


@router.get("/crt/ranges")
async def get_all_ranges():
    """
    Get all active CRT ranges.

    Returns a summary of all tracked ranges and their states.
    """
    active_ranges = range_tracker.get_all_active_ranges()

    return {
        "timestamp": datetime.now(UTC_TZ).isoformat(),
        "total_active": len(active_ranges),
        "ranges": {
            symbol: crt_range.to_dict()
            for symbol, crt_range in active_ranges.items()
        },
        "summary": {
            "waiting_for_purge": sum(
                1 for r in active_ranges.values()
                if r.state == CRTState.RANGE_DEFINED
            ),
            "purge_detected": sum(
                1 for r in active_ranges.values()
                if r.state in [CRTState.PURGE_ABOVE, CRTState.PURGE_BELOW]
            ),
            "signals_active": sum(
                1 for r in active_ranges.values()
                if r.has_signal
            ),
        }
    }


@router.get("/crt/history/{symbol}")
async def get_range_history(
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100, description="Max entries to return"),
):
    """
    Get historical CRT ranges for a symbol.

    Returns past ranges including outcomes.
    """
    symbol = symbol.upper()
    history = range_tracker.get_history(symbol, limit)

    return {
        "symbol": symbol,
        "count": len(history),
        "history": history,
    }


@router.get("/crt/config")
async def get_crt_config():
    """
    Get CRT service configuration.

    Returns all configurable parameters and their current values.
    """
    return {
        "range_tracker": {
            "max_range_age_hours": range_tracker.MAX_RANGE_AGE_HOURS,
            "min_range_percent": range_tracker.MIN_RANGE_PERCENT,
            "max_range_percent": range_tracker.MAX_RANGE_PERCENT,
            "purge_min_percent": range_tracker.PURGE_MIN_PERCENT,
        },
        "signal_service": {
            "weights": crt_signal_service.WEIGHTS,
            "min_confidence_threshold": crt_signal_service.MIN_CONFIDENCE_THRESHOLD,
            "default_max_risk_percent": crt_signal_service.DEFAULT_MAX_RISK_PERCENT,
        },
        "key_sessions": {
            "1_am_est": "London Pre-Market",
            "5_am_est": "London Open",
            "9_am_est": "New York Open",
        },
        "h4_boundaries_utc": [0, 4, 8, 12, 16, 20],
    }


@router.get("/crt/health")
async def crt_health():
    """
    CRT service health check.
    """
    return {
        "status": "healthy",
        "service": "CRT Detection",
        "timestamp": datetime.now(UTC_TZ).isoformat(),
        "active_ranges": len(range_tracker.get_all_active_ranges()),
        "current_session": session_manager.get_current_session().value,
    }
