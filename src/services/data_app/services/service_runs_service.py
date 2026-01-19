"""Service Runs History Service.

Provides persistent storage and retrieval of service run history from all microservices.
Tracks validations, predictions, trainings, analyses, and scans.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from ...timescaledb_service import timescaledb_service


class ServiceRunCreate(BaseModel):
    """Request model for creating a service run."""

    service: str  # data, nhits, tcn, hmm, candlestick, cnn-lstm, rag, llm, workplace, watchdog
    run_type: str  # validation, prediction, training, analysis, scan, health_check
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    triggered_by: str = "manual"  # manual, scheduled, api, auto-scan
    input_params: Optional[dict] = None


class ServiceRunUpdate(BaseModel):
    """Request model for updating a service run."""

    status: str  # running, completed, failed, aborted
    total_items: Optional[int] = None
    items_ok: Optional[int] = None
    items_warning: Optional[int] = None
    items_error: Optional[int] = None
    success_rate: Optional[float] = None
    results: Optional[dict] = None
    metrics: Optional[dict] = None
    error_details: Optional[dict] = None


class ServiceRunResult(BaseModel):
    """Complete service run result."""

    run_id: str
    service: str
    run_type: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: str
    triggered_by: str
    total_items: int = 0
    items_ok: int = 0
    items_warning: int = 0
    items_error: int = 0
    success_rate: Optional[float] = None
    input_params: Optional[dict] = None
    results: Optional[dict] = None
    metrics: Optional[dict] = None
    error_details: Optional[dict] = None


class ServiceRunSummary(BaseModel):
    """Summary of a service run for list view."""

    run_id: str
    service: str
    run_type: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: str
    triggered_by: str
    total_items: int = 0
    items_ok: int = 0
    items_error: int = 0
    success_rate: Optional[float] = None


class ServiceRunStats(BaseModel):
    """Statistics for service runs."""

    service: Optional[str] = None
    run_type: Optional[str] = None
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    running_runs: int = 0
    avg_success_rate: Optional[float] = None
    avg_duration_ms: Optional[float] = None


class ServiceRunsService:
    """Service for managing service run history across all microservices."""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the service and ensure tables exist."""
        if self._initialized:
            return True

        if not timescaledb_service.is_available:
            logger.warning("TimescaleDB not available, service runs history disabled")
            return False

        try:
            await self._ensure_tables_exist()
            self._initialized = True
            logger.info("ServiceRunsService initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ServiceRunsService: {e}")
            return False

    async def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist."""
        if not timescaledb_service._pool:
            await timescaledb_service.initialize()

        pool = timescaledb_service._pool
        if not pool:
            raise RuntimeError("No database pool available")

        async with pool.acquire() as conn:
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'service_runs'
                )
            """)

            if not exists:
                logger.info("Creating service_runs table...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS service_runs (
                        id              SERIAL PRIMARY KEY,
                        run_id          UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
                        service         VARCHAR(50) NOT NULL,
                        run_type        VARCHAR(50) NOT NULL,
                        symbol          VARCHAR(20),
                        timeframe       VARCHAR(10),
                        started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        completed_at    TIMESTAMPTZ,
                        duration_ms     INTEGER,
                        status          VARCHAR(20) NOT NULL DEFAULT 'running',
                        triggered_by    VARCHAR(50) DEFAULT 'manual',
                        total_items     INTEGER DEFAULT 0,
                        items_ok        INTEGER DEFAULT 0,
                        items_warning   INTEGER DEFAULT 0,
                        items_error     INTEGER DEFAULT 0,
                        success_rate    DECIMAL(5, 2),
                        input_params    JSONB,
                        results         JSONB,
                        metrics         JSONB,
                        error_details   JSONB,
                        created_at      TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_service_runs_service
                        ON service_runs (service);
                    CREATE INDEX IF NOT EXISTS idx_service_runs_type
                        ON service_runs (run_type);
                    CREATE INDEX IF NOT EXISTS idx_service_runs_started
                        ON service_runs (started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_service_runs_status
                        ON service_runs (status);
                    CREATE INDEX IF NOT EXISTS idx_service_runs_symbol
                        ON service_runs (symbol) WHERE symbol IS NOT NULL;
                    CREATE INDEX IF NOT EXISTS idx_service_runs_service_type
                        ON service_runs (service, run_type);
                """)
                logger.info("service_runs table created")

    async def create_run(self, data: ServiceRunCreate) -> Optional[str]:
        """Create a new service run entry and return its run_id."""
        if not await self.initialize():
            return None

        try:
            import json

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                run_id = await conn.fetchval(
                    """
                    INSERT INTO service_runs (
                        service, run_type, symbol, timeframe,
                        triggered_by, input_params, status
                    ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, 'running')
                    RETURNING run_id
                    """,
                    data.service,
                    data.run_type,
                    data.symbol,
                    data.timeframe,
                    data.triggered_by,
                    json.dumps(data.input_params) if data.input_params else None,
                )

                logger.debug(f"Created service run: {run_id} ({data.service}/{data.run_type})")
                return str(run_id)
        except Exception as e:
            logger.error(f"Failed to create service run: {e}")
            return None

    async def update_run(self, run_id: str, update: ServiceRunUpdate) -> bool:
        """Update a service run with results."""
        if not await self.initialize():
            return False

        try:
            import json

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                # Calculate duration if completed
                completed_at = None
                duration_ms = None
                if update.status in ("completed", "failed", "aborted"):
                    completed_at = datetime.now(timezone.utc)
                    # Get started_at to calculate duration
                    started_at = await conn.fetchval(
                        "SELECT started_at FROM service_runs WHERE run_id = $1",
                        uuid.UUID(run_id)
                    )
                    if started_at:
                        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

                result = await conn.execute(
                    """
                    UPDATE service_runs
                    SET status = $2,
                        completed_at = COALESCE($3, completed_at),
                        duration_ms = COALESCE($4, duration_ms),
                        total_items = COALESCE($5, total_items),
                        items_ok = COALESCE($6, items_ok),
                        items_warning = COALESCE($7, items_warning),
                        items_error = COALESCE($8, items_error),
                        success_rate = COALESCE($9, success_rate),
                        results = COALESCE($10::jsonb, results),
                        metrics = COALESCE($11::jsonb, metrics),
                        error_details = COALESCE($12::jsonb, error_details)
                    WHERE run_id = $1
                    """,
                    uuid.UUID(run_id),
                    update.status,
                    completed_at,
                    duration_ms,
                    update.total_items,
                    update.items_ok,
                    update.items_warning,
                    update.items_error,
                    Decimal(str(update.success_rate)) if update.success_rate is not None else None,
                    json.dumps(update.results) if update.results else None,
                    json.dumps(update.metrics) if update.metrics else None,
                    json.dumps(update.error_details) if update.error_details else None,
                )

                success = result.split()[-1] != "0"
                if success:
                    logger.debug(f"Updated service run: {run_id} -> {update.status}")
                return success
        except Exception as e:
            logger.error(f"Failed to update service run: {e}")
            return False

    async def get_run(self, run_id: str) -> Optional[ServiceRunResult]:
        """Get a specific service run by ID."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT run_id, service, run_type, symbol, timeframe,
                           started_at, completed_at, duration_ms, status,
                           triggered_by, total_items, items_ok, items_warning,
                           items_error, success_rate, input_params, results,
                           metrics, error_details
                    FROM service_runs
                    WHERE run_id = $1
                    """,
                    uuid.UUID(run_id),
                )

                if not row:
                    return None

                return ServiceRunResult(
                    run_id=str(row["run_id"]),
                    service=row["service"],
                    run_type=row["run_type"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    duration_ms=row["duration_ms"],
                    status=row["status"],
                    triggered_by=row["triggered_by"],
                    total_items=row["total_items"] or 0,
                    items_ok=row["items_ok"] or 0,
                    items_warning=row["items_warning"] or 0,
                    items_error=row["items_error"] or 0,
                    success_rate=float(row["success_rate"]) if row["success_rate"] else None,
                    input_params=row["input_params"],
                    results=row["results"],
                    metrics=row["metrics"],
                    error_details=row["error_details"],
                )
        except Exception as e:
            logger.error(f"Failed to get service run: {e}")
            return None

    async def list_runs(
        self,
        service: Optional[str] = None,
        run_type: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ServiceRunSummary]:
        """List service runs with optional filtering."""
        if not await self.initialize():
            return []

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT run_id, service, run_type, symbol, timeframe,
                           started_at, completed_at, duration_ms, status,
                           triggered_by, total_items, items_ok, items_error, success_rate
                    FROM service_runs
                    WHERE 1=1
                """
                params = []
                param_idx = 1

                if service:
                    query += f" AND service = ${param_idx}"
                    params.append(service)
                    param_idx += 1

                if run_type:
                    query += f" AND run_type = ${param_idx}"
                    params.append(run_type)
                    param_idx += 1

                if symbol:
                    query += f" AND symbol = ${param_idx}"
                    params.append(symbol)
                    param_idx += 1

                if status:
                    query += f" AND status = ${param_idx}"
                    params.append(status)
                    param_idx += 1

                query += f" ORDER BY started_at DESC LIMIT {limit} OFFSET {offset}"

                rows = await conn.fetch(query, *params)

                return [
                    ServiceRunSummary(
                        run_id=str(row["run_id"]),
                        service=row["service"],
                        run_type=row["run_type"],
                        symbol=row["symbol"],
                        timeframe=row["timeframe"],
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        duration_ms=row["duration_ms"],
                        status=row["status"],
                        triggered_by=row["triggered_by"],
                        total_items=row["total_items"] or 0,
                        items_ok=row["items_ok"] or 0,
                        items_error=row["items_error"] or 0,
                        success_rate=float(row["success_rate"]) if row["success_rate"] else None,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to list service runs: {e}")
            return []

    async def get_stats(
        self,
        service: Optional[str] = None,
        run_type: Optional[str] = None,
    ) -> ServiceRunStats:
        """Get service run statistics."""
        if not await self.initialize():
            return ServiceRunStats()

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT
                        COUNT(*) as total_runs,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed_runs,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
                        COUNT(*) FILTER (WHERE status = 'running') as running_runs,
                        ROUND(AVG(success_rate) FILTER (WHERE success_rate IS NOT NULL)::DECIMAL, 2) as avg_success_rate,
                        ROUND(AVG(duration_ms) FILTER (WHERE duration_ms IS NOT NULL)::DECIMAL, 0) as avg_duration_ms
                    FROM service_runs
                    WHERE 1=1
                """
                params = []
                param_idx = 1

                if service:
                    query += f" AND service = ${param_idx}"
                    params.append(service)
                    param_idx += 1

                if run_type:
                    query += f" AND run_type = ${param_idx}"
                    params.append(run_type)
                    param_idx += 1

                row = await conn.fetchrow(query, *params)

                return ServiceRunStats(
                    service=service,
                    run_type=run_type,
                    total_runs=row["total_runs"] or 0,
                    completed_runs=row["completed_runs"] or 0,
                    failed_runs=row["failed_runs"] or 0,
                    running_runs=row["running_runs"] or 0,
                    avg_success_rate=float(row["avg_success_rate"]) if row["avg_success_rate"] else None,
                    avg_duration_ms=float(row["avg_duration_ms"]) if row["avg_duration_ms"] else None,
                )
        except Exception as e:
            logger.error(f"Failed to get service run stats: {e}")
            return ServiceRunStats()

    async def get_stats_by_service(self) -> dict[str, ServiceRunStats]:
        """Get statistics grouped by service."""
        if not await self.initialize():
            return {}

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        service,
                        COUNT(*) as total_runs,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed_runs,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
                        COUNT(*) FILTER (WHERE status = 'running') as running_runs,
                        ROUND(AVG(success_rate) FILTER (WHERE success_rate IS NOT NULL)::DECIMAL, 2) as avg_success_rate,
                        ROUND(AVG(duration_ms) FILTER (WHERE duration_ms IS NOT NULL)::DECIMAL, 0) as avg_duration_ms
                    FROM service_runs
                    GROUP BY service
                    ORDER BY service
                """)

                return {
                    row["service"]: ServiceRunStats(
                        service=row["service"],
                        total_runs=row["total_runs"] or 0,
                        completed_runs=row["completed_runs"] or 0,
                        failed_runs=row["failed_runs"] or 0,
                        running_runs=row["running_runs"] or 0,
                        avg_success_rate=float(row["avg_success_rate"]) if row["avg_success_rate"] else None,
                        avg_duration_ms=float(row["avg_duration_ms"]) if row["avg_duration_ms"] else None,
                    )
                    for row in rows
                }
        except Exception as e:
            logger.error(f"Failed to get stats by service: {e}")
            return {}

    async def delete_run(self, run_id: str) -> bool:
        """Delete a service run."""
        if not await self.initialize():
            return False

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM service_runs WHERE run_id = $1",
                    uuid.UUID(run_id),
                )
                return result.split()[-1] != "0"
        except Exception as e:
            logger.error(f"Failed to delete service run: {e}")
            return False

    async def cleanup_old_runs(self, keep_count: int = 100, service: Optional[str] = None) -> int:
        """Delete old service runs, keeping the most recent keep_count per service/run_type."""
        if not await self.initialize():
            return 0

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                # Get distinct service/run_type combinations
                filter_query = ""
                params = []
                if service:
                    filter_query = " WHERE service = $1"
                    params.append(service)

                combos = await conn.fetch(
                    f"SELECT DISTINCT service, run_type FROM service_runs{filter_query}",
                    *params
                )

                total_deleted = 0
                for combo in combos:
                    svc = combo["service"]
                    rt = combo["run_type"]

                    # Delete old runs for this combo
                    result = await conn.execute(
                        """
                        DELETE FROM service_runs
                        WHERE id IN (
                            SELECT id FROM service_runs
                            WHERE service = $1 AND run_type = $2
                            ORDER BY started_at DESC
                            OFFSET $3
                        )
                        """,
                        svc, rt, keep_count
                    )
                    deleted = int(result.split()[-1])
                    total_deleted += deleted

                if total_deleted > 0:
                    logger.info(f"Cleaned up {total_deleted} old service runs")
                return total_deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old service runs: {e}")
            return 0

    async def count_runs(
        self,
        service: Optional[str] = None,
        run_type: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """Count service runs with optional filtering."""
        if not await self.initialize():
            return 0

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = "SELECT COUNT(*) FROM service_runs WHERE 1=1"
                params = []
                param_idx = 1

                if service:
                    query += f" AND service = ${param_idx}"
                    params.append(service)
                    param_idx += 1

                if run_type:
                    query += f" AND run_type = ${param_idx}"
                    params.append(run_type)
                    param_idx += 1

                if symbol:
                    query += f" AND symbol = ${param_idx}"
                    params.append(symbol)
                    param_idx += 1

                if status:
                    query += f" AND status = ${param_idx}"
                    params.append(status)
                    param_idx += 1

                return await conn.fetchval(query, *params) or 0
        except Exception as e:
            logger.error(f"Failed to count service runs: {e}")
            return 0


# Singleton instance
service_runs_service = ServiceRunsService()
