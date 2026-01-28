"""Prediction History Service.

Provides persistent storage and retrieval of predictions from all microservices.
Supports evaluation tracking for accuracy analysis.
"""

import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from ...timescaledb_service import timescaledb_service


class PredictionCreate(BaseModel):
    """Request model for creating a prediction."""

    service: str  # nhits, tcn, hmm, candlestick, cnn-lstm, workplace
    symbol: str
    timeframe: str
    prediction_type: str  # price, direction, pattern, regime, signal
    prediction: dict  # The actual prediction data
    confidence: Optional[float] = None
    target_time: Optional[datetime] = None
    horizon: Optional[str] = None  # 1h, 4h, 1d, etc.
    model_version: Optional[str] = None
    model_params: Optional[dict] = None
    input_features: Optional[dict] = None
    triggered_by: str = "api"
    tags: Optional[list[str]] = None
    notes: Optional[str] = None


class PredictionEvaluation(BaseModel):
    """Request model for evaluating a prediction."""

    actual_outcome: dict
    is_correct: Optional[bool] = None
    accuracy_score: Optional[float] = None
    error_amount: Optional[float] = None


class PredictionResult(BaseModel):
    """Complete prediction result."""

    prediction_id: str
    service: str
    symbol: str
    timeframe: str
    prediction_type: str
    prediction: dict
    confidence: Optional[float] = None
    predicted_at: datetime
    target_time: Optional[datetime] = None
    horizon: Optional[str] = None
    model_version: Optional[str] = None
    actual_outcome: Optional[dict] = None
    evaluated_at: Optional[datetime] = None
    is_correct: Optional[bool] = None
    accuracy_score: Optional[float] = None
    triggered_by: str = "api"
    tags: Optional[list[str]] = None


class PredictionSummary(BaseModel):
    """Summary of a prediction for list view."""

    prediction_id: str
    service: str
    symbol: str
    timeframe: str
    prediction_type: str
    prediction: Optional[dict] = None  # Full prediction data including entry_price
    confidence: Optional[float] = None
    predicted_at: datetime
    target_time: Optional[datetime] = None
    is_correct: Optional[bool] = None
    evaluated_at: Optional[datetime] = None


class PredictionStats(BaseModel):
    """Statistics for predictions."""

    service: Optional[str] = None
    symbol: Optional[str] = None
    total_predictions: int = 0
    evaluated_count: int = 0
    correct_count: int = 0
    accuracy_percent: Optional[float] = None
    avg_confidence: Optional[float] = None


class PredictionHistoryService:
    """Service for managing prediction history across all microservices."""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the service and ensure tables exist."""
        if self._initialized:
            return True

        if not timescaledb_service.is_available:
            logger.warning("TimescaleDB not available, prediction history disabled")
            return False

        try:
            await self._ensure_tables_exist()
            self._initialized = True
            logger.info("PredictionHistoryService initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PredictionHistoryService: {e}")
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
                    WHERE table_name = 'prediction_history'
                )
            """)

            if not exists:
                logger.info("Creating prediction_history table...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        id              SERIAL PRIMARY KEY,
                        prediction_id   UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
                        service         VARCHAR(50) NOT NULL,
                        symbol          VARCHAR(20) NOT NULL,
                        timeframe       VARCHAR(10) NOT NULL,
                        predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        target_time     TIMESTAMPTZ,
                        horizon         VARCHAR(20),
                        prediction_type VARCHAR(50) NOT NULL,
                        prediction      JSONB NOT NULL,
                        confidence      DECIMAL(5, 4),
                        model_version   VARCHAR(50),
                        model_params    JSONB,
                        input_features  JSONB,
                        actual_outcome  JSONB,
                        evaluated_at    TIMESTAMPTZ,
                        is_correct      BOOLEAN,
                        accuracy_score  DECIMAL(5, 4),
                        error_amount    DECIMAL(20, 8),
                        triggered_by    VARCHAR(50) DEFAULT 'api',
                        tags            TEXT[],
                        notes           TEXT,
                        created_at      TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_prediction_service
                        ON prediction_history (service);
                    CREATE INDEX IF NOT EXISTS idx_prediction_symbol
                        ON prediction_history (symbol);
                    CREATE INDEX IF NOT EXISTS idx_prediction_time
                        ON prediction_history (predicted_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_prediction_service_symbol
                        ON prediction_history (service, symbol);
                """)
                logger.info("prediction_history table created")

    async def create_prediction(self, data: PredictionCreate) -> Optional[str]:
        """Create a new prediction entry and return its prediction_id."""
        if not await self.initialize():
            return None

        try:
            import json

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                prediction_id = await conn.fetchval(
                    """
                    INSERT INTO prediction_history (
                        service, symbol, timeframe, prediction_type, prediction,
                        confidence, target_time, horizon, model_version, model_params,
                        input_features, triggered_by, tags, notes
                    ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10::jsonb, $11::jsonb, $12, $13, $14)
                    RETURNING prediction_id
                    """,
                    data.service,
                    data.symbol,
                    data.timeframe,
                    data.prediction_type,
                    json.dumps(data.prediction),
                    Decimal(str(data.confidence)) if data.confidence else None,
                    data.target_time,
                    data.horizon,
                    data.model_version,
                    json.dumps(data.model_params) if data.model_params else None,
                    json.dumps(data.input_features) if data.input_features else None,
                    data.triggered_by,
                    data.tags,
                    data.notes,
                )

                logger.debug(f"Created prediction: {prediction_id} ({data.service}/{data.symbol})")
                return str(prediction_id)
        except Exception as e:
            logger.error(f"Failed to create prediction: {e}")
            return None

    async def evaluate_prediction(
        self,
        prediction_id: str,
        evaluation: PredictionEvaluation,
    ) -> bool:
        """Evaluate a prediction with actual outcome."""
        if not await self.initialize():
            return False

        try:
            import json

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE prediction_history
                    SET actual_outcome = $2::jsonb,
                        evaluated_at = NOW(),
                        is_correct = $3,
                        accuracy_score = $4,
                        error_amount = $5
                    WHERE prediction_id = $1
                    """,
                    uuid.UUID(prediction_id),
                    json.dumps(evaluation.actual_outcome),
                    evaluation.is_correct,
                    Decimal(str(evaluation.accuracy_score)) if evaluation.accuracy_score else None,
                    Decimal(str(evaluation.error_amount)) if evaluation.error_amount else None,
                )

                success = result.split()[-1] != "0"
                if success:
                    logger.debug(f"Evaluated prediction: {prediction_id}")
                return success
        except Exception as e:
            logger.error(f"Failed to evaluate prediction: {e}")
            return False

    async def get_prediction(self, prediction_id: str) -> Optional[PredictionResult]:
        """Get a specific prediction by ID."""
        if not await self.initialize():
            return None

        try:
            import json as json_module

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT prediction_id, service, symbol, timeframe, prediction_type,
                           prediction, confidence, predicted_at, target_time, horizon,
                           model_version, actual_outcome, evaluated_at, is_correct,
                           accuracy_score, triggered_by, tags
                    FROM prediction_history
                    WHERE prediction_id = $1
                    """,
                    uuid.UUID(prediction_id),
                )

                if not row:
                    return None

                # Parse JSONB fields if returned as strings
                prediction_data = row["prediction"]
                if isinstance(prediction_data, str):
                    prediction_data = json_module.loads(prediction_data)

                actual_outcome = row["actual_outcome"]
                if isinstance(actual_outcome, str):
                    actual_outcome = json_module.loads(actual_outcome)

                return PredictionResult(
                    prediction_id=str(row["prediction_id"]),
                    service=row["service"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    prediction_type=row["prediction_type"],
                    prediction=prediction_data,
                    confidence=float(row["confidence"]) if row["confidence"] else None,
                    predicted_at=row["predicted_at"],
                    target_time=row["target_time"],
                    horizon=row["horizon"],
                    model_version=row["model_version"],
                    actual_outcome=actual_outcome,
                    evaluated_at=row["evaluated_at"],
                    is_correct=row["is_correct"],
                    accuracy_score=float(row["accuracy_score"]) if row["accuracy_score"] else None,
                    triggered_by=row["triggered_by"],
                    tags=row["tags"],
                )
        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            return None

    async def list_predictions(
        self,
        service: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        prediction_type: Optional[str] = None,
        evaluated_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PredictionSummary]:
        """List predictions with optional filtering."""
        if not await self.initialize():
            return []

        try:
            import json as json_module

            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT prediction_id, service, symbol, timeframe, prediction_type,
                           prediction, confidence, predicted_at, target_time, is_correct, evaluated_at
                    FROM prediction_history
                    WHERE 1=1
                """
                params = []
                param_idx = 1

                if service:
                    query += f" AND service = ${param_idx}"
                    params.append(service)
                    param_idx += 1

                if symbol:
                    query += f" AND symbol = ${param_idx}"
                    params.append(symbol)
                    param_idx += 1

                if timeframe:
                    query += f" AND timeframe = ${param_idx}"
                    params.append(timeframe)
                    param_idx += 1

                if prediction_type:
                    query += f" AND prediction_type = ${param_idx}"
                    params.append(prediction_type)
                    param_idx += 1

                if evaluated_only:
                    query += " AND evaluated_at IS NOT NULL"

                query += f" ORDER BY predicted_at DESC LIMIT {limit} OFFSET {offset}"

                rows = await conn.fetch(query, *params)

                results = []
                for row in rows:
                    # Parse JSONB prediction field
                    prediction_data = row["prediction"]
                    if isinstance(prediction_data, str):
                        prediction_data = json_module.loads(prediction_data)

                    results.append(PredictionSummary(
                        prediction_id=str(row["prediction_id"]),
                        service=row["service"],
                        symbol=row["symbol"],
                        timeframe=row["timeframe"],
                        prediction_type=row["prediction_type"],
                        prediction=prediction_data,
                        confidence=float(row["confidence"]) if row["confidence"] else None,
                        predicted_at=row["predicted_at"],
                        target_time=row["target_time"],
                        is_correct=row["is_correct"],
                        evaluated_at=row["evaluated_at"],
                    ))
                return results
        except Exception as e:
            logger.error(f"Failed to list predictions: {e}")
            return []

    async def get_stats(
        self,
        service: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> PredictionStats:
        """Get prediction statistics."""
        if not await self.initialize():
            return PredictionStats()

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT
                        COUNT(*) as total_predictions,
                        COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL) as evaluated_count,
                        COUNT(*) FILTER (WHERE is_correct = true) as correct_count,
                        ROUND(AVG(confidence)::DECIMAL, 4) as avg_confidence
                    FROM prediction_history
                    WHERE 1=1
                """
                params = []
                param_idx = 1

                if service:
                    query += f" AND service = ${param_idx}"
                    params.append(service)
                    param_idx += 1

                if symbol:
                    query += f" AND symbol = ${param_idx}"
                    params.append(symbol)
                    param_idx += 1

                row = await conn.fetchrow(query, *params)

                evaluated = row["evaluated_count"] or 0
                correct = row["correct_count"] or 0
                accuracy = (correct / evaluated * 100) if evaluated > 0 else None

                return PredictionStats(
                    service=service,
                    symbol=symbol,
                    total_predictions=row["total_predictions"] or 0,
                    evaluated_count=evaluated,
                    correct_count=correct,
                    accuracy_percent=round(accuracy, 2) if accuracy else None,
                    avg_confidence=float(row["avg_confidence"]) if row["avg_confidence"] else None,
                )
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return PredictionStats()

    async def get_predictions_due_for_evaluation(
        self,
        service: Optional[str] = None,
        limit: int = 100,
    ) -> list[PredictionSummary]:
        """Get predictions that are due for evaluation (target_time passed)."""
        if not await self.initialize():
            return []

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                query = """
                    SELECT prediction_id, service, symbol, timeframe, prediction_type,
                           confidence, predicted_at, target_time, is_correct, evaluated_at
                    FROM prediction_history
                    WHERE evaluated_at IS NULL
                      AND target_time IS NOT NULL
                      AND target_time <= NOW()
                """
                params = []

                if service:
                    query += " AND service = $1"
                    params.append(service)

                query += f" ORDER BY target_time ASC LIMIT {limit}"

                rows = await conn.fetch(query, *params)

                return [
                    PredictionSummary(
                        prediction_id=str(row["prediction_id"]),
                        service=row["service"],
                        symbol=row["symbol"],
                        timeframe=row["timeframe"],
                        prediction_type=row["prediction_type"],
                        confidence=float(row["confidence"]) if row["confidence"] else None,
                        predicted_at=row["predicted_at"],
                        target_time=row["target_time"],
                        is_correct=row["is_correct"],
                        evaluated_at=row["evaluated_at"],
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get predictions due for evaluation: {e}")
            return []

    async def delete_prediction(self, prediction_id: str) -> bool:
        """Delete a prediction."""
        if not await self.initialize():
            return False

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM prediction_history WHERE prediction_id = $1",
                    uuid.UUID(prediction_id),
                )
                return result.split()[-1] != "0"
        except Exception as e:
            logger.error(f"Failed to delete prediction: {e}")
            return False

    async def cleanup_old_predictions(self, days_to_keep: int = 90) -> int:
        """Delete predictions older than specified days."""
        if not await self.initialize():
            return 0

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM prediction_history
                    WHERE predicted_at < NOW() - ($1 || ' days')::INTERVAL
                    """,
                    str(days_to_keep),
                )
                deleted = int(result.split()[-1])
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old predictions")
                return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old predictions: {e}")
            return 0

    async def cleanup_invalid_predictions(self, service: str = "workplace") -> dict:
        """
        Delete predictions without entry_price that cannot be evaluated.

        Args:
            service: Service to cleanup (default: workplace)

        Returns:
            Statistics dictionary with deleted count
        """
        if not await self.initialize():
            return {"deleted": 0, "error": "Service not initialized"}

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                # Delete predictions where prediction is null OR entry_price is null/missing
                result = await conn.execute(
                    """
                    DELETE FROM prediction_history
                    WHERE service = $1
                      AND (
                          prediction IS NULL
                          OR prediction->>'entry_price' IS NULL
                          OR prediction->>'entry_price' = 'null'
                          OR prediction->>'entry_price' = ''
                      )
                    """,
                    service,
                )
                deleted = int(result.split()[-1])
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} invalid predictions for {service}")
                return {"deleted": deleted, "service": service}
        except Exception as e:
            logger.error(f"Failed to cleanup invalid predictions: {e}")
            return {"deleted": 0, "error": str(e)}

    async def reset_stale_evaluations(self, service: str = "workplace") -> dict:
        """
        Reset evaluations that used stale price data (price_change_percent = 0).

        This allows predictions to be re-evaluated with fresh price data.

        Args:
            service: Service to reset (default: workplace)

        Returns:
            Statistics dictionary with reset count
        """
        if not await self.initialize():
            return {"reset": 0, "error": "Service not initialized"}

        try:
            pool = timescaledb_service._pool
            async with pool.acquire() as conn:
                # Reset evaluations where price_change_percent is 0 (stale data)
                result = await conn.execute(
                    """
                    UPDATE prediction_history
                    SET evaluated_at = NULL,
                        is_correct = NULL,
                        accuracy_score = NULL,
                        error_amount = NULL,
                        actual_outcome = NULL
                    WHERE service = $1
                      AND evaluated_at IS NOT NULL
                      AND (
                          actual_outcome->>'price_change_percent' = '0'
                          OR actual_outcome->>'price_change_percent' = '0.0'
                          OR (actual_outcome IS NOT NULL AND actual_outcome->>'price_change_percent' IS NULL)
                      )
                    """,
                    service,
                )
                reset_count = int(result.split()[-1])
                if reset_count > 0:
                    logger.info(f"Reset {reset_count} stale evaluations for {service}")
                return {"reset": reset_count, "service": service}
        except Exception as e:
            logger.error(f"Failed to reset stale evaluations: {e}")
            return {"reset": 0, "error": str(e)}


# Singleton instance
prediction_history_service = PredictionHistoryService()
