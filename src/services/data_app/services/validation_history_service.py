"""Validation History Service.

Provides persistent storage and retrieval of validation run results.
Uses TimescaleDB for storing validation history.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from ...timescaledb_service import timescaledb_service


class ValidationComponentResult(BaseModel):
    """Result for a single validation component."""

    component: str
    status: str = "pending"  # pending, running, ok, warning, error
    tests_ok: int = 0
    tests_warning: int = 0
    tests_error: int = 0
    details: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ValidationError(BaseModel):
    """Individual validation error."""

    component: str
    error_type: str  # connection, data_quality, timeout, api_error
    severity: str = "error"  # warning, error
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    message: str
    details: Optional[dict] = None


class ValidationRunCreate(BaseModel):
    """Request model for creating a validation run."""

    triggered_by: str = "manual"
    results: Optional[dict] = None
    error_details: Optional[list] = None


class ValidationRunResult(BaseModel):
    """Complete validation run result."""

    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    triggered_by: str = "manual"
    total_tests: int = 0
    tests_ok: int = 0
    tests_warning: int = 0
    tests_error: int = 0
    success_rate: Optional[float] = None
    results: Optional[dict] = None
    error_details: Optional[list] = None


class ValidationRunSummary(BaseModel):
    """Summary of a validation run for list view."""

    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    triggered_by: str
    total_tests: int
    tests_ok: int
    tests_warning: int
    tests_error: int
    success_rate: Optional[float] = None


class ValidationHistoryService:
    """Service for managing validation run history."""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the service and ensure tables exist."""
        if self._initialized:
            return True

        if not timescaledb_service.is_available:
            logger.warning("TimescaleDB not available, validation history disabled")
            return False

        try:
            await self._ensure_tables_exist()
            self._initialized = True
            logger.info("ValidationHistoryService initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ValidationHistoryService: {e}")
            return False

    async def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist."""
        if not timescaledb_service._pool:
            await timescaledb_service.initialize()

        pool = timescaledb_service._pool
        if not pool:
            raise RuntimeError("No database pool available")

        async with pool.acquire() as conn:
            # Check if tables exist
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'validation_runs'
                )
            """)

            if not exists:
                logger.info("Creating validation history tables...")
                # Create tables inline if migration hasn't been run
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS validation_runs (
                        id              SERIAL PRIMARY KEY,
                        run_id          UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
                        started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        completed_at    TIMESTAMPTZ,
                        status          VARCHAR(20) NOT NULL DEFAULT 'running',
                        triggered_by    VARCHAR(50) DEFAULT 'manual',
                        total_tests     INTEGER DEFAULT 0,
                        tests_ok        INTEGER DEFAULT 0,
                        tests_warning   INTEGER DEFAULT 0,
                        tests_error     INTEGER DEFAULT 0,
                        success_rate    DECIMAL(5, 2),
                        results         JSONB,
                        error_details   JSONB,
                        created_at      TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_validation_runs_started_at
                        ON validation_runs (started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_validation_runs_status
                        ON validation_runs (status);
                """)
                logger.info("Validation history tables created")

    async def create_run(self, triggered_by: str = "manual") -> Optional[str]:
        """Create a new validation run and return its run_id."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                run_id = await conn.fetchval("""
                    INSERT INTO validation_runs (triggered_by, status)
                    VALUES ($1, 'running')
                    RETURNING run_id
                """, triggered_by)

                logger.info(f"Created validation run: {run_id}")
                return str(run_id)
        except Exception as e:
            logger.error(f"Failed to create validation run: {e}")
            return None

    async def complete_run(
        self,
        run_id: str,
        status: str,
        results: dict,
        error_details: Optional[list] = None,
    ) -> bool:
        """Complete a validation run with results."""
        if not await self.initialize():
            return False

        try:
            # Calculate totals
            total_ok = 0
            total_warning = 0
            total_error = 0

            for component_results in results.values():
                if isinstance(component_results, dict):
                    total_ok += component_results.get("ok", 0)
                    total_warning += component_results.get("warning", 0)
                    total_error += component_results.get("error", 0)

            total_tests = total_ok + total_warning + total_error
            success_rate = (total_ok / total_tests * 100) if total_tests > 0 else 0

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                import json

                await conn.execute(
                    """
                    UPDATE validation_runs
                    SET completed_at = NOW(),
                        status = $2,
                        total_tests = $3,
                        tests_ok = $4,
                        tests_warning = $5,
                        tests_error = $6,
                        success_rate = $7,
                        results = $8::jsonb,
                        error_details = $9::jsonb
                    WHERE run_id = $1
                    """,
                    uuid.UUID(run_id),
                    status,
                    total_tests,
                    total_ok,
                    total_warning,
                    total_error,
                    Decimal(str(round(success_rate, 2))),
                    json.dumps(results),
                    json.dumps(error_details) if error_details else None,
                )

                logger.info(
                    f"Completed validation run {run_id}: "
                    f"{total_ok} OK, {total_warning} warnings, {total_error} errors"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to complete validation run: {e}")
            return False

    async def get_run(self, run_id: str) -> Optional[ValidationRunResult]:
        """Get a specific validation run by ID."""
        if not await self.initialize():
            return None

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT run_id, started_at, completed_at, status, triggered_by,
                           total_tests, tests_ok, tests_warning, tests_error,
                           success_rate, results, error_details
                    FROM validation_runs
                    WHERE run_id = $1
                    """,
                    uuid.UUID(run_id),
                )

                if not row:
                    return None

                return ValidationRunResult(
                    run_id=str(row["run_id"]),
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    status=row["status"],
                    triggered_by=row["triggered_by"],
                    total_tests=row["total_tests"] or 0,
                    tests_ok=row["tests_ok"] or 0,
                    tests_warning=row["tests_warning"] or 0,
                    tests_error=row["tests_error"] or 0,
                    success_rate=float(row["success_rate"]) if row["success_rate"] else None,
                    results=row["results"],
                    error_details=row["error_details"],
                )
        except Exception as e:
            logger.error(f"Failed to get validation run: {e}")
            return None

    async def list_runs(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> list[ValidationRunSummary]:
        """List validation runs with optional filtering."""
        if not await self.initialize():
            return []

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT run_id, started_at, completed_at, status, triggered_by,
                           total_tests, tests_ok, tests_warning, tests_error, success_rate
                    FROM validation_runs
                """
                params = []

                if status:
                    query += " WHERE status = $1"
                    params.append(status)

                query += " ORDER BY started_at DESC"
                query += f" LIMIT {limit} OFFSET {offset}"

                rows = await conn.fetch(query, *params)

                return [
                    ValidationRunSummary(
                        run_id=str(row["run_id"]),
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        status=row["status"],
                        triggered_by=row["triggered_by"],
                        total_tests=row["total_tests"] or 0,
                        tests_ok=row["tests_ok"] or 0,
                        tests_warning=row["tests_warning"] or 0,
                        tests_error=row["tests_error"] or 0,
                        success_rate=float(row["success_rate"]) if row["success_rate"] else None,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to list validation runs: {e}")
            return []

    async def delete_run(self, run_id: str) -> bool:
        """Delete a validation run."""
        if not await self.initialize():
            return False

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM validation_runs WHERE run_id = $1",
                    uuid.UUID(run_id),
                )
                deleted = result.split()[-1] != "0"
                if deleted:
                    logger.info(f"Deleted validation run: {run_id}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to delete validation run: {e}")
            return False

    async def cleanup_old_runs(self, keep_count: int = 100) -> int:
        """Delete old validation runs, keeping the most recent ones."""
        if not await self.initialize():
            return 0

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                # Get IDs to delete
                old_ids = await conn.fetch(
                    """
                    SELECT id FROM validation_runs
                    ORDER BY started_at DESC
                    OFFSET $1
                    """,
                    keep_count,
                )

                if not old_ids:
                    return 0

                ids_to_delete = [row["id"] for row in old_ids]
                result = await conn.execute(
                    "DELETE FROM validation_runs WHERE id = ANY($1)",
                    ids_to_delete,
                )

                deleted_count = int(result.split()[-1])
                logger.info(f"Cleaned up {deleted_count} old validation runs")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old validation runs: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get validation history statistics."""
        if not await self.initialize():
            return {"available": False}

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_runs,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed_runs,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
                        AVG(success_rate) FILTER (WHERE success_rate IS NOT NULL) as avg_success_rate,
                        MAX(started_at) as last_run_at,
                        MIN(started_at) as first_run_at
                    FROM validation_runs
                """)

                return {
                    "available": True,
                    "total_runs": row["total_runs"],
                    "completed_runs": row["completed_runs"],
                    "failed_runs": row["failed_runs"],
                    "avg_success_rate": float(row["avg_success_rate"]) if row["avg_success_rate"] else None,
                    "last_run_at": row["last_run_at"].isoformat() if row["last_run_at"] else None,
                    "first_run_at": row["first_run_at"].isoformat() if row["first_run_at"] else None,
                }
        except Exception as e:
            logger.error(f"Failed to get validation stats: {e}")
            return {"available": False, "error": str(e)}


# Singleton instance
validation_history_service = ValidationHistoryService()
