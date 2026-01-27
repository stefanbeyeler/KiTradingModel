"""MT5 Trade History Service.

Provides persistent storage and retrieval of MT5 trades, terminals, and setup links.
Supports multi-terminal management and trade-setup linking for performance analysis.
"""

import json
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from ...timescaledb_service import timescaledb_service


# =============================================================================
# Terminal Schemas
# =============================================================================


class MT5TerminalCreate(BaseModel):
    """Request model for creating a terminal."""

    name: str = Field(..., max_length=100, description="Terminal display name")
    account_number: int = Field(..., description="MT5 account number")
    broker_name: Optional[str] = Field(None, max_length=100)
    server: Optional[str] = Field(None, max_length=100)
    account_type: str = Field(default="real", pattern="^(real|demo|contest)$")
    currency: str = Field(default="USD", max_length=10)
    leverage: Optional[int] = None
    metadata: Optional[dict] = None


class MT5TerminalUpdate(BaseModel):
    """Request model for updating a terminal."""

    name: Optional[str] = Field(None, max_length=100)
    broker_name: Optional[str] = Field(None, max_length=100)
    server: Optional[str] = Field(None, max_length=100)
    account_type: Optional[str] = Field(None, pattern="^(real|demo|contest)$")
    currency: Optional[str] = Field(None, max_length=10)
    leverage: Optional[int] = None
    is_active: Optional[bool] = None
    metadata: Optional[dict] = None


class MT5Terminal(BaseModel):
    """Complete terminal result."""

    terminal_id: str
    name: str
    account_number: int
    broker_name: Optional[str] = None
    server: Optional[str] = None
    account_type: str = "real"
    currency: str = "USD"
    leverage: Optional[int] = None
    api_key: Optional[str] = None
    is_active: bool = True
    last_heartbeat: Optional[datetime] = None
    metadata: Optional[dict] = None
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Trade Schemas
# =============================================================================


class MT5TradeCreate(BaseModel):
    """Request model for creating/updating a trade."""

    terminal_id: str = Field(..., description="Terminal UUID")
    ticket: int = Field(..., description="MT5 order ticket")
    position_id: Optional[int] = None
    symbol: str = Field(..., max_length=20)
    trade_type: str = Field(..., pattern="^(buy|sell)$")
    entry_time: datetime
    entry_price: float
    volume: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic_number: Optional[int] = None
    comment: Optional[str] = Field(None, max_length=200)
    timeframe: Optional[str] = Field(None, max_length=10)


class MT5TradeUpdate(BaseModel):
    """Request model for updating a trade (close, SL/TP change)."""

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit: Optional[float] = None
    profit_pips: Optional[float] = None
    commission: Optional[float] = None
    swap: Optional[float] = None
    status: Optional[str] = Field(None, pattern="^(open|closed|cancelled)$")
    close_reason: Optional[str] = Field(None, max_length=50)


class MT5Trade(BaseModel):
    """Complete trade result."""

    trade_id: str
    terminal_id: str
    ticket: int
    position_id: Optional[int] = None
    symbol: str
    trade_type: str
    entry_time: datetime
    entry_price: float
    volume: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit: Optional[float] = None
    profit_pips: Optional[float] = None
    commission: Optional[float] = None
    swap: Optional[float] = None
    status: str = "open"
    close_reason: Optional[str] = None
    magic_number: Optional[int] = None
    comment: Optional[str] = None
    timeframe: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Trade Setup Link Schemas
# =============================================================================


class MT5TradeSetupLinkCreate(BaseModel):
    """Request model for linking a trade to a setup."""

    trade_id: str
    setup_symbol: str = Field(..., max_length=20)
    setup_timeframe: str = Field(..., max_length=10)
    setup_timestamp: datetime
    setup_direction: str = Field(..., pattern="^(long|short|neutral)$")
    setup_score: float = Field(..., ge=0.0, le=100.0)
    setup_confidence: Optional[str] = None
    nhits_direction: Optional[str] = None
    nhits_probability: Optional[float] = None
    hmm_regime: Optional[str] = None
    hmm_score: Optional[float] = None
    tcn_patterns: Optional[list[str]] = None
    tcn_confidence: Optional[float] = None
    candlestick_patterns: Optional[list[str]] = None
    candlestick_strength: Optional[float] = None
    link_type: str = Field(default="auto", pattern="^(auto|manual)$")
    link_confidence: Optional[float] = None
    notes: Optional[str] = None


class MT5TradeSetupLink(BaseModel):
    """Complete trade-setup link result."""

    link_id: str
    trade_id: str
    setup_symbol: str
    setup_timeframe: str
    setup_timestamp: datetime
    setup_direction: str
    setup_score: float
    setup_confidence: Optional[str] = None
    nhits_direction: Optional[str] = None
    nhits_probability: Optional[float] = None
    hmm_regime: Optional[str] = None
    hmm_score: Optional[float] = None
    tcn_patterns: Optional[list[str]] = None
    tcn_confidence: Optional[float] = None
    candlestick_patterns: Optional[list[str]] = None
    candlestick_strength: Optional[float] = None
    link_type: str = "auto"
    link_confidence: Optional[float] = None
    notes: Optional[str] = None
    followed_recommendation: Optional[bool] = None
    outcome_vs_prediction: Optional[str] = None
    created_at: datetime


# =============================================================================
# Service Implementation
# =============================================================================


class MT5TradeHistoryService:
    """Service for managing MT5 trade history, terminals, and setup links."""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the service and ensure tables exist."""
        if self._initialized:
            return True

        if not timescaledb_service.is_available:
            logger.warning("TimescaleDB not available, MT5 trade history disabled")
            return False

        try:
            await self._ensure_tables_exist()
            self._initialized = True
            logger.info("MT5TradeHistoryService initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MT5TradeHistoryService: {e}")
            return False

    async def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist."""
        if not timescaledb_service._pool:
            await timescaledb_service.initialize()

        pool = timescaledb_service._pool
        if not pool:
            raise RuntimeError("No database pool available")

        async with pool.acquire() as conn:
            # Check and create mt5_terminals table
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'mt5_terminals'
                )
            """)

            if not exists:
                logger.info("Creating mt5_terminals table...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS mt5_terminals (
                        id                  SERIAL PRIMARY KEY,
                        terminal_id         UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
                        name                VARCHAR(100) NOT NULL,
                        account_number      BIGINT NOT NULL,
                        broker_name         VARCHAR(100),
                        server              VARCHAR(100),
                        account_type        VARCHAR(20) DEFAULT 'real',
                        currency            VARCHAR(10) DEFAULT 'USD',
                        leverage            INT,
                        api_key             VARCHAR(64),
                        is_active           BOOLEAN DEFAULT true,
                        last_heartbeat      TIMESTAMPTZ,
                        metadata            JSONB,
                        created_at          TIMESTAMPTZ DEFAULT NOW(),
                        updated_at          TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(account_number, server)
                    );

                    CREATE INDEX IF NOT EXISTS idx_mt5_terminals_active
                        ON mt5_terminals (is_active);
                    CREATE INDEX IF NOT EXISTS idx_mt5_terminals_account
                        ON mt5_terminals (account_number);
                """)
                logger.info("mt5_terminals table created")

            # Check and create mt5_trades table
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'mt5_trades'
                )
            """)

            if not exists:
                logger.info("Creating mt5_trades table...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS mt5_trades (
                        id                  SERIAL PRIMARY KEY,
                        trade_id            UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
                        terminal_id         UUID NOT NULL,
                        ticket              BIGINT NOT NULL,
                        position_id         BIGINT,
                        symbol              VARCHAR(20) NOT NULL,
                        trade_type          VARCHAR(10) NOT NULL,
                        entry_time          TIMESTAMPTZ NOT NULL,
                        entry_price         DECIMAL(20, 8) NOT NULL,
                        volume              DECIMAL(20, 8) NOT NULL,
                        exit_time           TIMESTAMPTZ,
                        exit_price          DECIMAL(20, 8),
                        stop_loss           DECIMAL(20, 8),
                        take_profit         DECIMAL(20, 8),
                        profit              DECIMAL(20, 8),
                        profit_pips         DECIMAL(10, 2),
                        commission          DECIMAL(10, 4),
                        swap                DECIMAL(10, 4),
                        status              VARCHAR(20) DEFAULT 'open',
                        close_reason        VARCHAR(50),
                        magic_number        BIGINT,
                        comment             VARCHAR(200),
                        timeframe           VARCHAR(10),
                        created_at          TIMESTAMPTZ DEFAULT NOW(),
                        updated_at          TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(terminal_id, ticket)
                    );

                    CREATE INDEX IF NOT EXISTS idx_mt5_trades_terminal
                        ON mt5_trades (terminal_id);
                    CREATE INDEX IF NOT EXISTS idx_mt5_trades_symbol
                        ON mt5_trades (symbol);
                    CREATE INDEX IF NOT EXISTS idx_mt5_trades_status
                        ON mt5_trades (status);
                    CREATE INDEX IF NOT EXISTS idx_mt5_trades_entry
                        ON mt5_trades (entry_time DESC);
                """)
                logger.info("mt5_trades table created")

            # Check and create mt5_trade_setup_links table
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'mt5_trade_setup_links'
                )
            """)

            if not exists:
                logger.info("Creating mt5_trade_setup_links table...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS mt5_trade_setup_links (
                        id                      SERIAL PRIMARY KEY,
                        link_id                 UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
                        trade_id                UUID NOT NULL,
                        setup_symbol            VARCHAR(20) NOT NULL,
                        setup_timeframe         VARCHAR(10) NOT NULL,
                        setup_timestamp         TIMESTAMPTZ NOT NULL,
                        setup_direction         VARCHAR(10) NOT NULL,
                        setup_score             DECIMAL(5, 2) NOT NULL,
                        setup_confidence        VARCHAR(20),
                        nhits_direction         VARCHAR(10),
                        nhits_probability       DECIMAL(5, 4),
                        hmm_regime              VARCHAR(20),
                        hmm_score               DECIMAL(5, 2),
                        tcn_patterns            TEXT[],
                        tcn_confidence          DECIMAL(5, 4),
                        candlestick_patterns    TEXT[],
                        candlestick_strength    DECIMAL(5, 4),
                        link_type               VARCHAR(20) DEFAULT 'auto',
                        link_confidence         DECIMAL(5, 4),
                        notes                   TEXT,
                        followed_recommendation BOOLEAN,
                        outcome_vs_prediction   VARCHAR(20),
                        created_at              TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(trade_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_trade_setup_links_trade
                        ON mt5_trade_setup_links (trade_id);
                    CREATE INDEX IF NOT EXISTS idx_trade_setup_links_symbol
                        ON mt5_trade_setup_links (setup_symbol);
                    CREATE INDEX IF NOT EXISTS idx_trade_setup_links_timestamp
                        ON mt5_trade_setup_links (setup_timestamp DESC);
                """)
                logger.info("mt5_trade_setup_links table created")

    # =========================================================================
    # Terminal Operations
    # =========================================================================

    async def create_terminal(self, data: MT5TerminalCreate) -> Optional[MT5Terminal]:
        """Create a new terminal and return it with generated API key."""
        if not await self.initialize():
            return None

        try:
            # Generate secure API key for agent authentication
            api_key = secrets.token_urlsafe(32)

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO mt5_terminals (
                        name, account_number, broker_name, server,
                        account_type, currency, leverage, api_key, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                    RETURNING terminal_id, name, account_number, broker_name, server,
                              account_type, currency, leverage, api_key, is_active,
                              last_heartbeat, metadata, created_at, updated_at
                    """,
                    data.name,
                    data.account_number,
                    data.broker_name,
                    data.server,
                    data.account_type,
                    data.currency,
                    data.leverage,
                    api_key,
                    json.dumps(data.metadata) if data.metadata else None,
                )

                logger.info(f"Created terminal: {row['terminal_id']} ({data.name})")
                return self._row_to_terminal(row)
        except Exception as e:
            logger.error(f"Failed to create terminal: {e}")
            return None

    async def get_terminal(self, terminal_id: str) -> Optional[MT5Terminal]:
        """Get a terminal by ID."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT terminal_id, name, account_number, broker_name, server,
                           account_type, currency, leverage, api_key, is_active,
                           last_heartbeat, metadata, created_at, updated_at
                    FROM mt5_terminals
                    WHERE terminal_id = $1
                    """,
                    uuid.UUID(terminal_id),
                )

                if not row:
                    return None
                return self._row_to_terminal(row)
        except Exception as e:
            logger.error(f"Failed to get terminal: {e}")
            return None

    async def get_terminal_by_api_key(self, api_key: str) -> Optional[MT5Terminal]:
        """Get a terminal by API key (for agent authentication)."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT terminal_id, name, account_number, broker_name, server,
                           account_type, currency, leverage, api_key, is_active,
                           last_heartbeat, metadata, created_at, updated_at
                    FROM mt5_terminals
                    WHERE api_key = $1 AND is_active = true
                    """,
                    api_key,
                )

                if not row:
                    return None
                return self._row_to_terminal(row)
        except Exception as e:
            logger.error(f"Failed to get terminal by API key: {e}")
            return None

    async def list_terminals(
        self, active_only: bool = True, limit: int = 50
    ) -> list[MT5Terminal]:
        """List all terminals."""
        if not await self.initialize():
            return []

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT terminal_id, name, account_number, broker_name, server,
                           account_type, currency, leverage, api_key, is_active,
                           last_heartbeat, metadata, created_at, updated_at
                    FROM mt5_terminals
                """
                if active_only:
                    query += " WHERE is_active = true"
                query += f" ORDER BY created_at DESC LIMIT {limit}"

                rows = await conn.fetch(query)
                return [self._row_to_terminal(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list terminals: {e}")
            return []

    async def update_terminal(
        self, terminal_id: str, data: MT5TerminalUpdate
    ) -> Optional[MT5Terminal]:
        """Update a terminal."""
        if not await self.initialize():
            return None

        try:
            # Build dynamic UPDATE query
            updates = []
            params = []
            param_idx = 1

            if data.name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(data.name)
                param_idx += 1
            if data.broker_name is not None:
                updates.append(f"broker_name = ${param_idx}")
                params.append(data.broker_name)
                param_idx += 1
            if data.server is not None:
                updates.append(f"server = ${param_idx}")
                params.append(data.server)
                param_idx += 1
            if data.account_type is not None:
                updates.append(f"account_type = ${param_idx}")
                params.append(data.account_type)
                param_idx += 1
            if data.currency is not None:
                updates.append(f"currency = ${param_idx}")
                params.append(data.currency)
                param_idx += 1
            if data.leverage is not None:
                updates.append(f"leverage = ${param_idx}")
                params.append(data.leverage)
                param_idx += 1
            if data.is_active is not None:
                updates.append(f"is_active = ${param_idx}")
                params.append(data.is_active)
                param_idx += 1
            if data.metadata is not None:
                updates.append(f"metadata = ${param_idx}::jsonb")
                params.append(json.dumps(data.metadata))
                param_idx += 1

            if not updates:
                return await self.get_terminal(terminal_id)

            updates.append("updated_at = NOW()")
            params.append(uuid.UUID(terminal_id))

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    UPDATE mt5_terminals
                    SET {', '.join(updates)}
                    WHERE terminal_id = ${param_idx}
                    RETURNING terminal_id, name, account_number, broker_name, server,
                              account_type, currency, leverage, api_key, is_active,
                              last_heartbeat, metadata, created_at, updated_at
                    """,
                    *params,
                )

                if not row:
                    return None
                logger.debug(f"Updated terminal: {terminal_id}")
                return self._row_to_terminal(row)
        except Exception as e:
            logger.error(f"Failed to update terminal: {e}")
            return None

    async def heartbeat(self, terminal_id: str) -> bool:
        """Update terminal heartbeat timestamp."""
        if not await self.initialize():
            return False

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE mt5_terminals
                    SET last_heartbeat = NOW(), updated_at = NOW()
                    WHERE terminal_id = $1
                    """,
                    uuid.UUID(terminal_id),
                )
                return result.split()[-1] != "0"
        except Exception as e:
            logger.error(f"Failed to update heartbeat: {e}")
            return False

    async def delete_terminal(self, terminal_id: str) -> bool:
        """Deactivate a terminal (soft delete)."""
        if not await self.initialize():
            return False

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE mt5_terminals
                    SET is_active = false, updated_at = NOW()
                    WHERE terminal_id = $1
                    """,
                    uuid.UUID(terminal_id),
                )
                return result.split()[-1] != "0"
        except Exception as e:
            logger.error(f"Failed to deactivate terminal: {e}")
            return False

    def _row_to_terminal(self, row) -> MT5Terminal:
        """Convert database row to MT5Terminal model."""
        return MT5Terminal(
            terminal_id=str(row["terminal_id"]),
            name=row["name"],
            account_number=row["account_number"],
            broker_name=row["broker_name"],
            server=row["server"],
            account_type=row["account_type"],
            currency=row["currency"],
            leverage=row["leverage"],
            api_key=row["api_key"],
            is_active=row["is_active"],
            last_heartbeat=row["last_heartbeat"],
            metadata=row["metadata"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def upsert_trade(self, data: MT5TradeCreate) -> Optional[str]:
        """Create or update a trade. Returns trade_id."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                trade_id = await conn.fetchval(
                    """
                    INSERT INTO mt5_trades (
                        terminal_id, ticket, position_id, symbol, trade_type,
                        entry_time, entry_price, volume, stop_loss, take_profit,
                        magic_number, comment, timeframe
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (terminal_id, ticket)
                    DO UPDATE SET
                        position_id = EXCLUDED.position_id,
                        stop_loss = EXCLUDED.stop_loss,
                        take_profit = EXCLUDED.take_profit,
                        updated_at = NOW()
                    RETURNING trade_id
                    """,
                    uuid.UUID(data.terminal_id),
                    data.ticket,
                    data.position_id,
                    data.symbol.upper(),
                    data.trade_type.lower(),
                    data.entry_time,
                    Decimal(str(data.entry_price)),
                    Decimal(str(data.volume)),
                    Decimal(str(data.stop_loss)) if data.stop_loss else None,
                    Decimal(str(data.take_profit)) if data.take_profit else None,
                    data.magic_number,
                    data.comment,
                    data.timeframe,
                )

                logger.debug(f"Upserted trade: {trade_id} (ticket={data.ticket})")
                return str(trade_id)
        except Exception as e:
            logger.error(f"Failed to upsert trade: {e}")
            return None

    async def update_trade(
        self, trade_id: str, data: MT5TradeUpdate
    ) -> Optional[MT5Trade]:
        """Update a trade (close, modify SL/TP)."""
        if not await self.initialize():
            return None

        try:
            updates = []
            params = []
            param_idx = 1

            if data.exit_time is not None:
                updates.append(f"exit_time = ${param_idx}")
                params.append(data.exit_time)
                param_idx += 1
            if data.exit_price is not None:
                updates.append(f"exit_price = ${param_idx}")
                params.append(Decimal(str(data.exit_price)))
                param_idx += 1
            if data.stop_loss is not None:
                updates.append(f"stop_loss = ${param_idx}")
                params.append(Decimal(str(data.stop_loss)))
                param_idx += 1
            if data.take_profit is not None:
                updates.append(f"take_profit = ${param_idx}")
                params.append(Decimal(str(data.take_profit)))
                param_idx += 1
            if data.profit is not None:
                updates.append(f"profit = ${param_idx}")
                params.append(Decimal(str(data.profit)))
                param_idx += 1
            if data.profit_pips is not None:
                updates.append(f"profit_pips = ${param_idx}")
                params.append(Decimal(str(data.profit_pips)))
                param_idx += 1
            if data.commission is not None:
                updates.append(f"commission = ${param_idx}")
                params.append(Decimal(str(data.commission)))
                param_idx += 1
            if data.swap is not None:
                updates.append(f"swap = ${param_idx}")
                params.append(Decimal(str(data.swap)))
                param_idx += 1
            if data.status is not None:
                updates.append(f"status = ${param_idx}")
                params.append(data.status)
                param_idx += 1
            if data.close_reason is not None:
                updates.append(f"close_reason = ${param_idx}")
                params.append(data.close_reason)
                param_idx += 1

            if not updates:
                return await self.get_trade(trade_id)

            updates.append("updated_at = NOW()")
            params.append(uuid.UUID(trade_id))

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    UPDATE mt5_trades
                    SET {', '.join(updates)}
                    WHERE trade_id = ${param_idx}
                    RETURNING *
                    """,
                    *params,
                )

                if not row:
                    return None
                logger.debug(f"Updated trade: {trade_id}")
                return self._row_to_trade(row)
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")
            return None

    async def get_trade(self, trade_id: str) -> Optional[MT5Trade]:
        """Get a trade by ID."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM mt5_trades WHERE trade_id = $1",
                    uuid.UUID(trade_id),
                )

                if not row:
                    return None
                return self._row_to_trade(row)
        except Exception as e:
            logger.error(f"Failed to get trade: {e}")
            return None

    async def list_trades(
        self,
        terminal_id: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MT5Trade]:
        """List trades with optional filtering."""
        if not await self.initialize():
            return []

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = "SELECT * FROM mt5_trades WHERE 1=1"
                params = []
                param_idx = 1

                if terminal_id:
                    query += f" AND terminal_id = ${param_idx}"
                    params.append(uuid.UUID(terminal_id))
                    param_idx += 1
                if symbol:
                    query += f" AND symbol = ${param_idx}"
                    params.append(symbol.upper())
                    param_idx += 1
                if status:
                    query += f" AND status = ${param_idx}"
                    params.append(status)
                    param_idx += 1
                if since:
                    query += f" AND entry_time >= ${param_idx}"
                    params.append(since)
                    param_idx += 1
                if until:
                    query += f" AND entry_time <= ${param_idx}"
                    params.append(until)
                    param_idx += 1

                query += f" ORDER BY entry_time DESC LIMIT {limit} OFFSET {offset}"

                rows = await conn.fetch(query, *params)
                return [self._row_to_trade(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list trades: {e}")
            return []

    async def get_trade_stats(
        self,
        terminal_id: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> dict:
        """Get trade statistics."""
        if not await self.initialize():
            return {}

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE status = 'open') as open_trades,
                        COUNT(*) FILTER (WHERE status = 'closed') as closed_trades,
                        COUNT(*) FILTER (WHERE status = 'closed' AND profit > 0) as winning_trades,
                        COUNT(*) FILTER (WHERE status = 'closed' AND profit < 0) as losing_trades,
                        COALESCE(SUM(profit) FILTER (WHERE status = 'closed' AND profit > 0), 0) as total_profit,
                        COALESCE(SUM(profit) FILTER (WHERE status = 'closed' AND profit < 0), 0) as total_loss,
                        COALESCE(SUM(profit) FILTER (WHERE status = 'closed'), 0) as net_profit,
                        COALESCE(SUM(commission), 0) as total_commission,
                        COALESCE(SUM(swap), 0) as total_swap
                    FROM mt5_trades
                    WHERE 1=1
                """
                params = []
                param_idx = 1

                if terminal_id:
                    query += f" AND terminal_id = ${param_idx}"
                    params.append(uuid.UUID(terminal_id))
                    param_idx += 1
                if symbol:
                    query += f" AND symbol = ${param_idx}"
                    params.append(symbol.upper())
                    param_idx += 1
                if since:
                    query += f" AND entry_time >= ${param_idx}"
                    params.append(since)
                    param_idx += 1

                row = await conn.fetchrow(query, *params)

                closed = row["closed_trades"] or 0
                winning = row["winning_trades"] or 0
                losing = row["losing_trades"] or 0
                total_profit = float(row["total_profit"] or 0)
                total_loss = abs(float(row["total_loss"] or 0))

                return {
                    "total_trades": row["total_trades"] or 0,
                    "open_trades": row["open_trades"] or 0,
                    "closed_trades": closed,
                    "winning_trades": winning,
                    "losing_trades": losing,
                    "win_rate": round(winning / closed * 100, 2) if closed > 0 else 0.0,
                    "total_profit": total_profit,
                    "total_loss": total_loss,
                    "net_profit": float(row["net_profit"] or 0),
                    "profit_factor": (
                        round(total_profit / total_loss, 2) if total_loss > 0 else 0.0
                    ),
                    "total_commission": float(row["total_commission"] or 0),
                    "total_swap": float(row["total_swap"] or 0),
                }
        except Exception as e:
            logger.error(f"Failed to get trade stats: {e}")
            return {}

    def _row_to_trade(self, row) -> MT5Trade:
        """Convert database row to MT5Trade model."""
        return MT5Trade(
            trade_id=str(row["trade_id"]),
            terminal_id=str(row["terminal_id"]),
            ticket=row["ticket"],
            position_id=row["position_id"],
            symbol=row["symbol"],
            trade_type=row["trade_type"],
            entry_time=row["entry_time"],
            entry_price=float(row["entry_price"]),
            volume=float(row["volume"]),
            exit_time=row["exit_time"],
            exit_price=float(row["exit_price"]) if row["exit_price"] else None,
            stop_loss=float(row["stop_loss"]) if row["stop_loss"] else None,
            take_profit=float(row["take_profit"]) if row["take_profit"] else None,
            profit=float(row["profit"]) if row["profit"] else None,
            profit_pips=float(row["profit_pips"]) if row["profit_pips"] else None,
            commission=float(row["commission"]) if row["commission"] else None,
            swap=float(row["swap"]) if row["swap"] else None,
            status=row["status"],
            close_reason=row["close_reason"],
            magic_number=row["magic_number"],
            comment=row["comment"],
            timeframe=row["timeframe"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # Trade Setup Link Operations
    # =========================================================================

    async def create_link(
        self, data: MT5TradeSetupLinkCreate
    ) -> Optional[MT5TradeSetupLink]:
        """Create a trade-setup link."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO mt5_trade_setup_links (
                        trade_id, setup_symbol, setup_timeframe, setup_timestamp,
                        setup_direction, setup_score, setup_confidence,
                        nhits_direction, nhits_probability, hmm_regime, hmm_score,
                        tcn_patterns, tcn_confidence, candlestick_patterns,
                        candlestick_strength, link_type, link_confidence, notes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        setup_symbol = EXCLUDED.setup_symbol,
                        setup_timeframe = EXCLUDED.setup_timeframe,
                        setup_timestamp = EXCLUDED.setup_timestamp,
                        setup_direction = EXCLUDED.setup_direction,
                        setup_score = EXCLUDED.setup_score,
                        setup_confidence = EXCLUDED.setup_confidence,
                        nhits_direction = EXCLUDED.nhits_direction,
                        nhits_probability = EXCLUDED.nhits_probability,
                        hmm_regime = EXCLUDED.hmm_regime,
                        hmm_score = EXCLUDED.hmm_score,
                        tcn_patterns = EXCLUDED.tcn_patterns,
                        tcn_confidence = EXCLUDED.tcn_confidence,
                        candlestick_patterns = EXCLUDED.candlestick_patterns,
                        candlestick_strength = EXCLUDED.candlestick_strength,
                        link_type = EXCLUDED.link_type,
                        link_confidence = EXCLUDED.link_confidence,
                        notes = EXCLUDED.notes
                    RETURNING *
                    """,
                    uuid.UUID(data.trade_id),
                    data.setup_symbol.upper(),
                    data.setup_timeframe,
                    data.setup_timestamp,
                    data.setup_direction,
                    Decimal(str(data.setup_score)),
                    data.setup_confidence,
                    data.nhits_direction,
                    Decimal(str(data.nhits_probability)) if data.nhits_probability else None,
                    data.hmm_regime,
                    Decimal(str(data.hmm_score)) if data.hmm_score else None,
                    data.tcn_patterns,
                    Decimal(str(data.tcn_confidence)) if data.tcn_confidence else None,
                    data.candlestick_patterns,
                    Decimal(str(data.candlestick_strength)) if data.candlestick_strength else None,
                    data.link_type,
                    Decimal(str(data.link_confidence)) if data.link_confidence else None,
                    data.notes,
                )

                logger.debug(f"Created/updated link for trade: {data.trade_id}")
                return self._row_to_link(row)
        except Exception as e:
            logger.error(f"Failed to create link: {e}")
            return None

    async def get_link(self, trade_id: str) -> Optional[MT5TradeSetupLink]:
        """Get the setup link for a trade."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM mt5_trade_setup_links WHERE trade_id = $1",
                    uuid.UUID(trade_id),
                )

                if not row:
                    return None
                return self._row_to_link(row)
        except Exception as e:
            logger.error(f"Failed to get link: {e}")
            return None

    async def delete_link(self, trade_id: str) -> bool:
        """Delete a trade-setup link."""
        if not await self.initialize():
            return False

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM mt5_trade_setup_links WHERE trade_id = $1",
                    uuid.UUID(trade_id),
                )
                return result.split()[-1] != "0"
        except Exception as e:
            logger.error(f"Failed to delete link: {e}")
            return False

    async def evaluate_link(
        self,
        trade_id: str,
        followed_recommendation: bool,
        outcome_vs_prediction: str,
    ) -> bool:
        """Evaluate a trade-setup link after trade closes."""
        if not await self.initialize():
            return False

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE mt5_trade_setup_links
                    SET followed_recommendation = $2,
                        outcome_vs_prediction = $3
                    WHERE trade_id = $1
                    """,
                    uuid.UUID(trade_id),
                    followed_recommendation,
                    outcome_vs_prediction,
                )
                return result.split()[-1] != "0"
        except Exception as e:
            logger.error(f"Failed to evaluate link: {e}")
            return False

    async def get_link_stats(self) -> dict:
        """Get statistics about trade-setup links."""
        if not await self.initialize():
            return {}

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_links,
                        COUNT(*) FILTER (WHERE link_type = 'auto') as auto_links,
                        COUNT(*) FILTER (WHERE link_type = 'manual') as manual_links,
                        COUNT(*) FILTER (WHERE followed_recommendation = true) as followed_count,
                        COUNT(*) FILTER (WHERE followed_recommendation = false) as not_followed_count,
                        COUNT(*) FILTER (WHERE outcome_vs_prediction = 'correct') as correct_predictions,
                        COUNT(*) FILTER (WHERE outcome_vs_prediction = 'incorrect') as incorrect_predictions,
                        COUNT(*) FILTER (WHERE outcome_vs_prediction IS NOT NULL) as evaluated_count,
                        ROUND(AVG(setup_score)::DECIMAL, 2) as avg_setup_score,
                        ROUND(AVG(link_confidence)::DECIMAL, 4) as avg_link_confidence
                    FROM mt5_trade_setup_links
                """)

                evaluated = row["evaluated_count"] or 0
                correct = row["correct_predictions"] or 0

                return {
                    "total_links": row["total_links"] or 0,
                    "auto_links": row["auto_links"] or 0,
                    "manual_links": row["manual_links"] or 0,
                    "followed_count": row["followed_count"] or 0,
                    "not_followed_count": row["not_followed_count"] or 0,
                    "correct_predictions": correct,
                    "incorrect_predictions": row["incorrect_predictions"] or 0,
                    "evaluated_count": evaluated,
                    "prediction_accuracy": (
                        round(correct / evaluated * 100, 2) if evaluated > 0 else 0.0
                    ),
                    "avg_setup_score": float(row["avg_setup_score"] or 0),
                    "avg_link_confidence": float(row["avg_link_confidence"] or 0),
                }
        except Exception as e:
            logger.error(f"Failed to get link stats: {e}")
            return {}

    def _row_to_link(self, row) -> MT5TradeSetupLink:
        """Convert database row to MT5TradeSetupLink model."""
        return MT5TradeSetupLink(
            link_id=str(row["link_id"]),
            trade_id=str(row["trade_id"]),
            setup_symbol=row["setup_symbol"],
            setup_timeframe=row["setup_timeframe"],
            setup_timestamp=row["setup_timestamp"],
            setup_direction=row["setup_direction"],
            setup_score=float(row["setup_score"]),
            setup_confidence=row["setup_confidence"],
            nhits_direction=row["nhits_direction"],
            nhits_probability=float(row["nhits_probability"]) if row["nhits_probability"] else None,
            hmm_regime=row["hmm_regime"],
            hmm_score=float(row["hmm_score"]) if row["hmm_score"] else None,
            tcn_patterns=row["tcn_patterns"],
            tcn_confidence=float(row["tcn_confidence"]) if row["tcn_confidence"] else None,
            candlestick_patterns=row["candlestick_patterns"],
            candlestick_strength=float(row["candlestick_strength"]) if row["candlestick_strength"] else None,
            link_type=row["link_type"],
            link_confidence=float(row["link_confidence"]) if row["link_confidence"] else None,
            notes=row["notes"],
            followed_recommendation=row["followed_recommendation"],
            outcome_vs_prediction=row["outcome_vs_prediction"],
            created_at=row["created_at"],
        )


# Singleton instance
mt5_trade_history_service = MT5TradeHistoryService()
