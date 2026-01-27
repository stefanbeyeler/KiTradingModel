"""MT5 Trade API Routes.

Provides endpoints for MT5 terminal management, trade recording, and setup linking.
This is the persistence layer - all database operations go through here.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from ..services.mt5_trade_history_service import (
    MT5Terminal,
    MT5TerminalCreate,
    MT5TerminalUpdate,
    MT5Trade,
    MT5TradeCreate,
    MT5TradeUpdate,
    MT5TradeSetupLink,
    MT5TradeSetupLinkCreate,
    mt5_trade_history_service,
)

router = APIRouter(prefix="/api/v1/mt5", tags=["7. MT5 Trades"])


# =============================================================================
# Response Models
# =============================================================================


class TerminalListResponse(BaseModel):
    """Response for terminal list."""

    terminals: list[MT5Terminal]
    total: int


class TradeListResponse(BaseModel):
    """Response for trade list."""

    trades: list[MT5Trade]
    total: int
    stats: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""

    service: str = "mt5"
    status: str
    message: str


# =============================================================================
# Terminal Endpoints
# =============================================================================


@router.post("/terminals", response_model=MT5Terminal)
async def create_terminal(data: MT5TerminalCreate):
    """
    Register a new MT5 terminal.

    Creates a terminal entry and generates an API key for agent authentication.
    The API key is returned only once - store it securely for the MT5 agent.
    """
    terminal = await mt5_trade_history_service.create_terminal(data)
    if not terminal:
        raise HTTPException(status_code=500, detail="Failed to create terminal")

    logger.info(f"Terminal created: {terminal.name} (Account: {terminal.account_number})")
    return terminal


@router.get("/terminals", response_model=TerminalListResponse)
async def list_terminals(
    active_only: bool = Query(default=True, description="Only active terminals"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
):
    """
    List all registered MT5 terminals.

    Returns terminals with their connection status based on last heartbeat.
    """
    terminals = await mt5_trade_history_service.list_terminals(
        active_only=active_only, limit=limit
    )

    # Mask API keys for security (show only last 8 chars)
    for terminal in terminals:
        if terminal.api_key:
            terminal.api_key = f"...{terminal.api_key[-8:]}"

    return TerminalListResponse(terminals=terminals, total=len(terminals))


@router.get("/terminals/{terminal_id}", response_model=MT5Terminal)
async def get_terminal(terminal_id: str):
    """Get a specific terminal by ID."""
    terminal = await mt5_trade_history_service.get_terminal(terminal_id)
    if not terminal:
        raise HTTPException(status_code=404, detail="Terminal not found")

    # Mask API key
    if terminal.api_key:
        terminal.api_key = f"...{terminal.api_key[-8:]}"

    return terminal


@router.put("/terminals/{terminal_id}", response_model=MT5Terminal)
async def update_terminal(terminal_id: str, data: MT5TerminalUpdate):
    """Update a terminal's configuration."""
    terminal = await mt5_trade_history_service.update_terminal(terminal_id, data)
    if not terminal:
        raise HTTPException(status_code=404, detail="Terminal not found")

    # Mask API key
    if terminal.api_key:
        terminal.api_key = f"...{terminal.api_key[-8:]}"

    return terminal


@router.delete("/terminals/{terminal_id}")
async def delete_terminal(terminal_id: str):
    """
    Deactivate a terminal (soft delete).

    The terminal and its trades are preserved but marked as inactive.
    """
    success = await mt5_trade_history_service.delete_terminal(terminal_id)
    if not success:
        raise HTTPException(status_code=404, detail="Terminal not found")

    return {"success": True, "message": "Terminal deactivated"}


@router.post("/terminals/{terminal_id}/heartbeat")
async def terminal_heartbeat(terminal_id: str):
    """
    Update terminal heartbeat.

    Called by the MT5 agent to indicate the terminal is still connected.
    """
    success = await mt5_trade_history_service.heartbeat(terminal_id)
    if not success:
        raise HTTPException(status_code=404, detail="Terminal not found")

    return {"success": True, "timestamp": datetime.utcnow().isoformat()}


@router.post("/terminals/{terminal_id}/regenerate-key", response_model=MT5Terminal)
async def regenerate_api_key(terminal_id: str):
    """
    Regenerate the API key for a terminal.

    Use this if the key was compromised. The old key becomes invalid immediately.
    """
    import secrets

    terminal = await mt5_trade_history_service.get_terminal(terminal_id)
    if not terminal:
        raise HTTPException(status_code=404, detail="Terminal not found")

    # Generate new key and update
    new_key = secrets.token_urlsafe(32)

    # Direct database update for API key
    pool = mt5_trade_history_service._pool if hasattr(mt5_trade_history_service, "_pool") else None
    if not pool:
        from ...timescaledb_service import timescaledb_service

        pool = timescaledb_service._pool

    if pool:
        import uuid as uuid_lib

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE mt5_terminals SET api_key = $1, updated_at = NOW() WHERE terminal_id = $2",
                new_key,
                uuid_lib.UUID(terminal_id),
            )

    # Return terminal with new key (full key shown only this once)
    terminal.api_key = new_key
    logger.warning(f"API key regenerated for terminal: {terminal_id}")
    return terminal


# =============================================================================
# Trade Endpoints
# =============================================================================


@router.post("/trades", response_model=dict)
async def record_trade(
    data: MT5TradeCreate,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Record or update a trade from MT5.

    Called by the MT5 agent when a trade is opened or modified.
    Uses upsert logic - same terminal+ticket combination updates existing trade.

    Authentication via X-API-Key header (optional but recommended).
    """
    # Validate API key if provided
    if x_api_key:
        terminal = await mt5_trade_history_service.get_terminal_by_api_key(x_api_key)
        if not terminal:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if terminal.terminal_id != data.terminal_id:
            raise HTTPException(status_code=403, detail="API key does not match terminal")

    trade_id = await mt5_trade_history_service.upsert_trade(data)
    if not trade_id:
        raise HTTPException(status_code=500, detail="Failed to record trade")

    return {"success": True, "trade_id": trade_id}


@router.put("/trades/{trade_id}", response_model=MT5Trade)
async def update_trade(
    trade_id: str,
    data: MT5TradeUpdate,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Update a trade (close, modify SL/TP).

    Called by the MT5 agent when a trade is closed or modified.
    """
    trade = await mt5_trade_history_service.update_trade(trade_id, data)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return trade


@router.get("/trades", response_model=TradeListResponse)
async def list_trades(
    terminal_id: Optional[str] = Query(None, description="Filter by terminal"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status (open, closed)"),
    since: Optional[datetime] = Query(None, description="Trades since this time"),
    until: Optional[datetime] = Query(None, description="Trades until this time"),
    include_stats: bool = Query(default=False, description="Include trade statistics"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """
    List trades with optional filtering.

    Supports filtering by terminal, symbol, status, and time range.
    """
    trades = await mt5_trade_history_service.list_trades(
        terminal_id=terminal_id,
        symbol=symbol,
        status=status,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )

    stats = None
    if include_stats:
        stats = await mt5_trade_history_service.get_trade_stats(
            terminal_id=terminal_id,
            symbol=symbol,
            since=since,
        )

    return TradeListResponse(trades=trades, total=len(trades), stats=stats)


@router.get("/trades/{trade_id}", response_model=MT5Trade)
async def get_trade(trade_id: str):
    """Get a specific trade by ID."""
    trade = await mt5_trade_history_service.get_trade(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return trade


@router.get("/trades/stats/summary")
async def get_trade_stats(
    terminal_id: Optional[str] = Query(None, description="Filter by terminal"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    since: Optional[datetime] = Query(None, description="Stats since this time"),
):
    """
    Get trade statistics summary.

    Returns win rate, profit factor, total P&L, and other metrics.
    """
    stats = await mt5_trade_history_service.get_trade_stats(
        terminal_id=terminal_id,
        symbol=symbol,
        since=since,
    )

    return stats


# =============================================================================
# Trade Setup Link Endpoints
# =============================================================================


@router.post("/trades/{trade_id}/link", response_model=MT5TradeSetupLink)
async def create_trade_link(trade_id: str, data: MT5TradeSetupLinkCreate):
    """
    Link a trade to a trading setup.

    Creates or updates the link between a trade and a setup.
    Used for tracking whether trades followed recommendations.
    """
    # Ensure trade_id matches
    if data.trade_id != trade_id:
        data.trade_id = trade_id

    link = await mt5_trade_history_service.create_link(data)
    if not link:
        raise HTTPException(status_code=500, detail="Failed to create link")

    return link


@router.get("/trades/{trade_id}/link", response_model=MT5TradeSetupLink)
async def get_trade_link(trade_id: str):
    """Get the setup link for a trade."""
    link = await mt5_trade_history_service.get_link(trade_id)
    if not link:
        raise HTTPException(status_code=404, detail="No link found for this trade")

    return link


@router.delete("/trades/{trade_id}/link")
async def delete_trade_link(trade_id: str):
    """Remove the setup link from a trade."""
    success = await mt5_trade_history_service.delete_link(trade_id)
    if not success:
        raise HTTPException(status_code=404, detail="No link found for this trade")

    return {"success": True, "message": "Link removed"}


@router.post("/trades/{trade_id}/link/evaluate")
async def evaluate_trade_link(
    trade_id: str,
    followed_recommendation: bool = Query(
        ..., description="Did the trade follow the setup recommendation?"
    ),
    outcome_vs_prediction: str = Query(
        ..., description="Outcome: correct, incorrect, partial"
    ),
):
    """
    Evaluate a trade-setup link after the trade closes.

    Records whether the trade followed the recommendation and if the prediction was correct.
    """
    if outcome_vs_prediction not in ("correct", "incorrect", "partial"):
        raise HTTPException(
            status_code=400,
            detail="outcome_vs_prediction must be: correct, incorrect, or partial",
        )

    success = await mt5_trade_history_service.evaluate_link(
        trade_id=trade_id,
        followed_recommendation=followed_recommendation,
        outcome_vs_prediction=outcome_vs_prediction,
    )

    if not success:
        raise HTTPException(status_code=404, detail="No link found for this trade")

    return {"success": True, "message": "Link evaluated"}


@router.get("/links/stats")
async def get_link_stats():
    """
    Get statistics about trade-setup links.

    Returns accuracy metrics for setup predictions.
    """
    stats = await mt5_trade_history_service.get_link_stats()
    return stats


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check MT5 trade service health."""
    initialized = await mt5_trade_history_service.initialize()

    if initialized:
        return HealthResponse(
            service="mt5",
            status="healthy",
            message="MT5 trade history service is operational",
        )
    else:
        return HealthResponse(
            service="mt5",
            status="degraded",
            message="TimescaleDB not available - MT5 features disabled",
        )
