"""TimescaleDB Service for persistent data storage.

This service provides the ONLY interface to TimescaleDB.
All database operations must go through this service.

Architecture:
- asyncpg for async PostgreSQL connections
- Connection pooling for performance
- Automatic hypertable management
- Upsert operations for conflict handling
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional
import json

from loguru import logger

# asyncpg import with fallback
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not installed - TimescaleDB features disabled")

from ..config.settings import settings
from ..config.timeframes import Timeframe, normalize_timeframe
from ..config.database import get_ohlcv_table_name, SUPPORTED_TIMEFRAMES


class TimescaleDBService:
    """
    Service for TimescaleDB connection and queries.

    IMPORTANT: This service is the ONLY component that
    creates direct database connections.
    """

    def __init__(self):
        self._pool: Optional["asyncpg.Pool"] = None
        self._initialized = False
        self._dsn = settings.timescale_dsn

    @property
    def is_available(self) -> bool:
        """Check if TimescaleDB is available and enabled."""
        return (
            ASYNCPG_AVAILABLE
            and settings.timescale_enabled
            and bool(settings.timescale_password)
        )

    async def initialize(self) -> bool:
        """Initialize connection pool.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not self.is_available:
            logger.warning("TimescaleDB not available or not enabled")
            return False

        if self._initialized and self._pool:
            return True

        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=settings.timescale_pool_min,
                max_size=settings.timescale_pool_max,
                command_timeout=60,
                statement_cache_size=100,
            )
            self._initialized = True
            logger.info(
                f"TimescaleDB connection pool created: "
                f"{settings.timescale_host}:{settings.timescale_port}/{settings.timescale_database}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB pool: {e}")
            self._initialized = False
            return False

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("TimescaleDB connection pool closed")

    @asynccontextmanager
    async def connection(self):
        """Context manager for database connection."""
        if not self._initialized:
            await self.initialize()

        if not self._pool:
            raise RuntimeError("TimescaleDB pool not initialized")

        async with self._pool.acquire() as conn:
            yield conn

    async def health_check(self) -> dict:
        """Check database health.

        Returns:
            Health status dictionary.
        """
        if not self.is_available:
            return {
                "status": "disabled",
                "available": False,
                "error": "asyncpg not installed or TimescaleDB disabled",
            }

        try:
            async with self.connection() as conn:
                result = await conn.fetchval("SELECT 1")
                version = await conn.fetchval("SELECT version()")
                return {
                    "status": "healthy",
                    "available": True,
                    "host": settings.timescale_host,
                    "database": settings.timescale_database,
                    "version": version[:50] if version else None,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
            }

    # ==================== OHLCV Methods ====================

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """
        Get OHLCV data from TimescaleDB.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            timeframe: Timeframe ('M1', 'H1', 'D1', etc.)
            limit: Maximum number of data points
            start_time: Start time (optional)
            end_time: End time (optional)

        Returns:
            List of OHLCV dictionaries
        """
        if not self.is_available:
            return []

        tf = normalize_timeframe(timeframe)
        table = get_ohlcv_table_name(tf.value)

        query = f"""
            SELECT
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                source
            FROM {table}
            WHERE symbol = $1
        """
        params: list[Any] = [symbol]
        param_idx = 2

        if start_time:
            query += f" AND timestamp >= ${param_idx}"
            params.append(start_time)
            param_idx += 1

        if end_time:
            query += f" AND timestamp <= ${param_idx}"
            params.append(end_time)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        try:
            async with self.connection() as conn:
                rows = await conn.fetch(query, *params)

            return [
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "symbol": row["symbol"],
                    "timeframe": tf.value,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]) if row["volume"] else None,
                    "source": row["source"],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}/{tf.value}: {e}")
            return []

    async def upsert_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Insert or update OHLCV data (Upsert).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: List of OHLCV dictionaries with timestamp, open, high, low, close, volume
            source: Data source identifier

        Returns:
            Number of rows inserted/updated
        """
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)
        table = get_ohlcv_table_name(tf.value)

        query = f"""
            INSERT INTO {table}
                (timestamp, symbol, open, high, low, close, volume, source, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (timestamp, symbol)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source,
                updated_at = NOW()
        """

        try:
            async with self.connection() as conn:
                async with conn.transaction():
                    for row in data:
                        # Parse timestamp if string
                        ts = row.get("timestamp") or row.get("datetime") or row.get("snapshot_time")
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        await conn.execute(
                            query,
                            ts,
                            symbol,
                            float(row.get("open", 0)),
                            float(row.get("high", 0)),
                            float(row.get("low", 0)),
                            float(row.get("close", 0)),
                            float(row.get("volume", 0)) if row.get("volume") else None,
                            source,
                        )

            # Update freshness tracking
            await self._update_freshness(symbol, tf.value, "ohlcv", len(data), source)

            logger.debug(f"Upserted {len(data)} OHLCV rows for {symbol}/{tf.value}")
            return len(data)

        except Exception as e:
            logger.error(f"Failed to upsert OHLCV for {symbol}/{tf.value}: {e}")
            return 0

    async def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[datetime]:
        """Get the latest timestamp for symbol/timeframe."""
        if not self.is_available:
            return None

        tf = normalize_timeframe(timeframe)
        table = get_ohlcv_table_name(tf.value)

        query = f"""
            SELECT MAX(timestamp) as latest
            FROM {table}
            WHERE symbol = $1
        """

        try:
            async with self.connection() as conn:
                row = await conn.fetchrow(query, symbol)
            return row["latest"] if row else None
        except Exception as e:
            logger.error(f"Failed to get latest timestamp for {symbol}/{tf.value}: {e}")
            return None

    # ==================== Indicators Methods (JSONB table) ====================

    async def get_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get technical indicators from JSONB table."""
        if not self.is_available:
            return []

        tf = normalize_timeframe(timeframe)

        query = """
            SELECT timestamp, values, parameters, source
            FROM indicators
            WHERE symbol = $1
              AND timeframe = $2
              AND indicator_name = $3
            ORDER BY timestamp DESC
            LIMIT $4
        """

        try:
            async with self.connection() as conn:
                rows = await conn.fetch(query, symbol, tf.value, indicator_name, limit)

            results = []
            for row in rows:
                values = row["values"]
                if isinstance(values, str):
                    values = json.loads(values)
                result = {
                    "timestamp": row["timestamp"].isoformat(),
                    "symbol": symbol,
                    "timeframe": tf.value,
                    "indicator": indicator_name,
                    "parameters": row["parameters"],
                    "source": row["source"],
                }
                result.update(values)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Failed to get indicators for {symbol}/{tf.value}: {e}")
            return []

    async def upsert_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        data: list[dict[str, Any]],
        parameters: dict[str, Any],
        source: str,
    ) -> int:
        """Insert or update indicators in JSONB table."""
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)

        query = """
            INSERT INTO indicators
                (timestamp, symbol, timeframe, indicator_name, values, parameters, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (timestamp, symbol, timeframe, indicator_name)
            DO UPDATE SET
                values = EXCLUDED.values,
                parameters = EXCLUDED.parameters,
                source = EXCLUDED.source
        """

        try:
            async with self.connection() as conn:
                async with conn.transaction():
                    for row in data:
                        # Extract timestamp
                        timestamp = row.pop("timestamp", None) or row.pop("datetime", None)
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                        # Remaining fields become JSONB values
                        values = json.dumps(row)

                        await conn.execute(
                            query,
                            timestamp,
                            symbol,
                            tf.value,
                            indicator_name,
                            values,
                            json.dumps(parameters),
                            source,
                        )

            await self._update_freshness(
                symbol, tf.value, f"indicator_{indicator_name.lower()}", len(data), source
            )

            return len(data)

        except Exception as e:
            logger.error(f"Failed to upsert indicators for {symbol}/{tf.value}: {e}")
            return 0

    # ==================== Optimized Indicator Tables ====================

    async def get_momentum_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        indicators: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Get momentum indicators from optimized table."""
        if not self.is_available:
            return []

        tf = normalize_timeframe(timeframe)

        # Select specific columns or all
        if indicators:
            columns = ", ".join(indicators)
        else:
            columns = """
                rsi_14, rsi_7, rsi_21, stoch_rsi, connors_rsi,
                stoch_k, stoch_d,
                macd_line, macd_signal, macd_histogram,
                cci, williams_r, roc, momentum,
                adx, plus_di, minus_di, mfi
            """

        query = f"""
            SELECT timestamp, {columns}, source
            FROM indicators_momentum
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        try:
            async with self.connection() as conn:
                rows = await conn.fetch(query, symbol, tf.value, limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get momentum indicators: {e}")
            return []

    async def get_volatility_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get volatility indicators from optimized table."""
        if not self.is_available:
            return []

        tf = normalize_timeframe(timeframe)

        query = """
            SELECT
                timestamp,
                bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b,
                atr_14, atr_7, natr, true_range,
                kc_upper, kc_middle, kc_lower,
                dc_upper, dc_middle, dc_lower,
                source
            FROM indicators_volatility
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        try:
            async with self.connection() as conn:
                rows = await conn.fetch(query, symbol, tf.value, limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get volatility indicators: {e}")
            return []

    async def get_trend_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get trend indicators from optimized table."""
        if not self.is_available:
            return []

        tf = normalize_timeframe(timeframe)

        query = """
            SELECT
                timestamp,
                ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
                ichimoku_senkou_b, ichimoku_chikou,
                supertrend, supertrend_direction,
                psar, psar_direction,
                aroon_up, aroon_down, aroon_oscillator,
                linreg_slope, linreg_intercept, linreg_r_squared,
                ht_trendmode,
                source
            FROM indicators_trend
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        try:
            async with self.connection() as conn:
                rows = await conn.fetch(query, symbol, tf.value, limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get trend indicators: {e}")
            return []

    async def get_ma_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get moving average indicators from optimized table."""
        if not self.is_available:
            return []

        tf = normalize_timeframe(timeframe)

        query = """
            SELECT
                timestamp,
                sma_20, sma_50, sma_200,
                ema_12, ema_26, ema_50, ema_200,
                wma_20, dema_20, tema_20,
                vwap,
                source
            FROM indicators_ma
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        try:
            async with self.connection() as conn:
                rows = await conn.fetch(query, symbol, tf.value, limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get MA indicators: {e}")
            return []

    async def get_all_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get all indicators from all optimized tables.

        Returns:
            Dict with categories as keys and indicator lists as values.
        """
        if not self.is_available:
            return {}

        tf = normalize_timeframe(timeframe)
        results = {}

        try:
            async with self.connection() as conn:
                # MA Indicators
                ma_query = """
                    SELECT * FROM indicators_ma
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC LIMIT $3
                """
                results["moving_averages"] = [
                    dict(r) for r in await conn.fetch(ma_query, symbol, tf.value, limit)
                ]

                # Momentum
                mom_query = """
                    SELECT * FROM indicators_momentum
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC LIMIT $3
                """
                results["momentum"] = [
                    dict(r) for r in await conn.fetch(mom_query, symbol, tf.value, limit)
                ]

                # Volatility
                vol_query = """
                    SELECT * FROM indicators_volatility
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC LIMIT $3
                """
                results["volatility"] = [
                    dict(r) for r in await conn.fetch(vol_query, symbol, tf.value, limit)
                ]

                # Trend
                trend_query = """
                    SELECT * FROM indicators_trend
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC LIMIT $3
                """
                results["trend"] = [
                    dict(r) for r in await conn.fetch(trend_query, symbol, tf.value, limit)
                ]

                # Volume
                vol_ind_query = """
                    SELECT * FROM indicators_volume
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC LIMIT $3
                """
                results["volume"] = [
                    dict(r) for r in await conn.fetch(vol_ind_query, symbol, tf.value, limit)
                ]

                # Levels
                levels_query = """
                    SELECT * FROM indicators_levels
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC LIMIT $3
                """
                results["levels"] = [
                    dict(r) for r in await conn.fetch(levels_query, symbol, tf.value, limit)
                ]

                # Generic JSONB indicators table (TwelveData + EasyInsight)
                generic_query = """
                    SELECT indicator_name, timestamp, values, source
                    FROM indicators
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """
                generic_rows = await conn.fetch(generic_query, symbol, tf.value, limit * 10)

                # Group by indicator name
                generic_indicators = {}
                for row in generic_rows:
                    ind_name = row["indicator_name"]
                    if ind_name not in generic_indicators:
                        generic_indicators[ind_name] = []

                    values = row["values"]
                    if isinstance(values, str):
                        values = json.loads(values)

                    entry = {
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                        "source": row["source"],
                    }
                    entry.update(values if isinstance(values, dict) else {ind_name: values})
                    generic_indicators[ind_name].append(entry)

                results["generic"] = generic_indicators

            return results

        except Exception as e:
            logger.error(f"Failed to get all indicators for {symbol}/{tf.value}: {e}")
            return {}

    # ==================== Optimized Indicator Upserts ====================

    async def upsert_momentum_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Upsert momentum indicators into optimized table.

        Supported columns: rsi_14, rsi_7, rsi_21, stoch_rsi, connors_rsi,
        stoch_k, stoch_d, macd_line, macd_signal, macd_histogram,
        cci, williams_r, roc, momentum, adx, plus_di, minus_di, mfi

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (H1, D1, etc.)
            data: List of indicator dictionaries with timestamp and values
            source: Data source identifier

        Returns:
            Number of rows inserted/updated
        """
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)

        # Valid columns for this table
        valid_columns = {
            "rsi_14", "rsi_7", "rsi_21", "stoch_rsi", "connors_rsi",
            "stoch_k", "stoch_d", "macd_line", "macd_signal", "macd_histogram",
            "cci", "williams_r", "roc", "momentum", "adx", "plus_di", "minus_di", "mfi"
        }

        try:
            async with self.connection() as conn:
                count = 0
                async with conn.transaction():
                    for row in data:
                        # Extract timestamp
                        ts = row.get("timestamp") or row.get("datetime")
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        if not ts:
                            continue

                        # Build dynamic column list and values
                        columns = ["timestamp", "symbol", "timeframe", "source", "created_at"]
                        values = [ts, symbol, tf.value, source, datetime.now(timezone.utc)]
                        placeholders = ["$1", "$2", "$3", "$4", "$5"]
                        update_parts = []
                        idx = 6

                        for col in valid_columns:
                            if col in row and row[col] is not None:
                                columns.append(col)
                                values.append(float(row[col]))
                                placeholders.append(f"${idx}")
                                update_parts.append(f"{col} = EXCLUDED.{col}")
                                idx += 1

                        if len(columns) <= 5:
                            # No indicator values, skip
                            continue

                        query = f"""
                            INSERT INTO indicators_momentum ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT (timestamp, symbol, timeframe)
                            DO UPDATE SET {', '.join(update_parts)}, source = EXCLUDED.source
                        """

                        await conn.execute(query, *values)
                        count += 1

                await self._update_freshness(symbol, tf.value, "indicators_momentum", count, source)
                logger.debug(f"Upserted {count} momentum indicators for {symbol}/{tf.value}")
                return count

        except Exception as e:
            logger.error(f"Failed to upsert momentum indicators for {symbol}/{tf.value}: {e}")
            return 0

    async def upsert_volatility_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Upsert volatility indicators into optimized table.

        Supported columns: bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b,
        atr_14, atr_7, natr, true_range, kc_upper, kc_middle, kc_lower,
        dc_upper, dc_middle, dc_lower

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: List of indicator dictionaries
            source: Data source identifier

        Returns:
            Number of rows inserted/updated
        """
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)

        valid_columns = {
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent_b",
            "atr_14", "atr_7", "natr", "true_range",
            "kc_upper", "kc_middle", "kc_lower",
            "dc_upper", "dc_middle", "dc_lower"
        }

        try:
            async with self.connection() as conn:
                count = 0
                async with conn.transaction():
                    for row in data:
                        ts = row.get("timestamp") or row.get("datetime")
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        if not ts:
                            continue

                        columns = ["timestamp", "symbol", "timeframe", "source", "created_at"]
                        values = [ts, symbol, tf.value, source, datetime.now(timezone.utc)]
                        placeholders = ["$1", "$2", "$3", "$4", "$5"]
                        update_parts = []
                        idx = 6

                        for col in valid_columns:
                            if col in row and row[col] is not None:
                                columns.append(col)
                                values.append(float(row[col]))
                                placeholders.append(f"${idx}")
                                update_parts.append(f"{col} = EXCLUDED.{col}")
                                idx += 1

                        if len(columns) <= 5:
                            continue

                        query = f"""
                            INSERT INTO indicators_volatility ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT (timestamp, symbol, timeframe)
                            DO UPDATE SET {', '.join(update_parts)}, source = EXCLUDED.source
                        """

                        await conn.execute(query, *values)
                        count += 1

                await self._update_freshness(symbol, tf.value, "indicators_volatility", count, source)
                logger.debug(f"Upserted {count} volatility indicators for {symbol}/{tf.value}")
                return count

        except Exception as e:
            logger.error(f"Failed to upsert volatility indicators for {symbol}/{tf.value}: {e}")
            return 0

    async def upsert_trend_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Upsert trend indicators into optimized table.

        Supported columns: ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
        ichimoku_senkou_b, ichimoku_chikou, supertrend, supertrend_direction,
        psar, psar_direction, aroon_up, aroon_down, aroon_oscillator,
        linreg_slope, linreg_intercept, linreg_r_squared, ht_trendmode

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: List of indicator dictionaries
            source: Data source identifier

        Returns:
            Number of rows inserted/updated
        """
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)

        valid_columns = {
            "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
            "ichimoku_senkou_b", "ichimoku_chikou",
            "supertrend", "supertrend_direction",
            "psar", "psar_direction",
            "aroon_up", "aroon_down", "aroon_oscillator",
            "linreg_slope", "linreg_intercept", "linreg_r_squared",
            "ht_trendmode"
        }

        # Integer columns
        int_columns = {"supertrend_direction", "psar_direction", "ht_trendmode"}

        try:
            async with self.connection() as conn:
                count = 0
                async with conn.transaction():
                    for row in data:
                        ts = row.get("timestamp") or row.get("datetime")
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        if not ts:
                            continue

                        columns = ["timestamp", "symbol", "timeframe", "source", "created_at"]
                        values = [ts, symbol, tf.value, source, datetime.now(timezone.utc)]
                        placeholders = ["$1", "$2", "$3", "$4", "$5"]
                        update_parts = []
                        idx = 6

                        for col in valid_columns:
                            if col in row and row[col] is not None:
                                columns.append(col)
                                if col in int_columns:
                                    values.append(int(row[col]))
                                else:
                                    values.append(float(row[col]))
                                placeholders.append(f"${idx}")
                                update_parts.append(f"{col} = EXCLUDED.{col}")
                                idx += 1

                        if len(columns) <= 5:
                            continue

                        query = f"""
                            INSERT INTO indicators_trend ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT (timestamp, symbol, timeframe)
                            DO UPDATE SET {', '.join(update_parts)}, source = EXCLUDED.source
                        """

                        await conn.execute(query, *values)
                        count += 1

                await self._update_freshness(symbol, tf.value, "indicators_trend", count, source)
                logger.debug(f"Upserted {count} trend indicators for {symbol}/{tf.value}")
                return count

        except Exception as e:
            logger.error(f"Failed to upsert trend indicators for {symbol}/{tf.value}: {e}")
            return 0

    async def upsert_ma_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Upsert moving average indicators into optimized table.

        Supported columns: sma_20, sma_50, sma_200, ema_12, ema_26, ema_50,
        ema_200, wma_20, dema_20, tema_20, vwap

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: List of indicator dictionaries
            source: Data source identifier

        Returns:
            Number of rows inserted/updated
        """
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)

        valid_columns = {
            "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26", "ema_50", "ema_200",
            "wma_20", "dema_20", "tema_20", "vwap"
        }

        try:
            async with self.connection() as conn:
                count = 0
                async with conn.transaction():
                    for row in data:
                        ts = row.get("timestamp") or row.get("datetime")
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        if not ts:
                            continue

                        columns = ["timestamp", "symbol", "timeframe", "source", "created_at"]
                        values = [ts, symbol, tf.value, source, datetime.now(timezone.utc)]
                        placeholders = ["$1", "$2", "$3", "$4", "$5"]
                        update_parts = []
                        idx = 6

                        for col in valid_columns:
                            if col in row and row[col] is not None:
                                columns.append(col)
                                values.append(float(row[col]))
                                placeholders.append(f"${idx}")
                                update_parts.append(f"{col} = EXCLUDED.{col}")
                                idx += 1

                        if len(columns) <= 5:
                            continue

                        query = f"""
                            INSERT INTO indicators_ma ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT (timestamp, symbol, timeframe)
                            DO UPDATE SET {', '.join(update_parts)}, source = EXCLUDED.source
                        """

                        await conn.execute(query, *values)
                        count += 1

                await self._update_freshness(symbol, tf.value, "indicators_ma", count, source)
                logger.debug(f"Upserted {count} MA indicators for {symbol}/{tf.value}")
                return count

        except Exception as e:
            logger.error(f"Failed to upsert MA indicators for {symbol}/{tf.value}: {e}")
            return 0

    async def upsert_volume_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Upsert volume indicators into optimized table.

        Supported columns: obv, ad_line, ad_oscillator, chaikin_mf,
        volume_sma_20, volume_ratio

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: List of indicator dictionaries
            source: Data source identifier

        Returns:
            Number of rows inserted/updated
        """
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)

        valid_columns = {
            "obv", "ad_line", "ad_oscillator", "chaikin_mf",
            "volume_sma_20", "volume_ratio"
        }

        try:
            async with self.connection() as conn:
                count = 0
                async with conn.transaction():
                    for row in data:
                        ts = row.get("timestamp") or row.get("datetime")
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        if not ts:
                            continue

                        columns = ["timestamp", "symbol", "timeframe", "source", "created_at"]
                        values = [ts, symbol, tf.value, source, datetime.now(timezone.utc)]
                        placeholders = ["$1", "$2", "$3", "$4", "$5"]
                        update_parts = []
                        idx = 6

                        for col in valid_columns:
                            if col in row and row[col] is not None:
                                columns.append(col)
                                values.append(float(row[col]))
                                placeholders.append(f"${idx}")
                                update_parts.append(f"{col} = EXCLUDED.{col}")
                                idx += 1

                        if len(columns) <= 5:
                            continue

                        query = f"""
                            INSERT INTO indicators_volume ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT (timestamp, symbol, timeframe)
                            DO UPDATE SET {', '.join(update_parts)}, source = EXCLUDED.source
                        """

                        await conn.execute(query, *values)
                        count += 1

                await self._update_freshness(symbol, tf.value, "indicators_volume", count, source)
                logger.debug(f"Upserted {count} volume indicators for {symbol}/{tf.value}")
                return count

        except Exception as e:
            logger.error(f"Failed to upsert volume indicators for {symbol}/{tf.value}: {e}")
            return 0

    async def upsert_levels_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: list[dict[str, Any]],
        source: str,
    ) -> int:
        """
        Upsert pivot/level indicators into optimized table.

        Supported columns: pivot, r1, r2, r3, s1, s2, s3,
        fib_r1, fib_r2, fib_r3, fib_s1, fib_s2, fib_s3,
        cam_r1, cam_r2, cam_r3, cam_r4, cam_s1, cam_s2, cam_s3, cam_s4

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: List of indicator dictionaries
            source: Data source identifier

        Returns:
            Number of rows inserted/updated
        """
        if not data or not self.is_available:
            return 0

        tf = normalize_timeframe(timeframe)

        valid_columns = {
            "pivot", "r1", "r2", "r3", "s1", "s2", "s3",
            "fib_r1", "fib_r2", "fib_r3", "fib_s1", "fib_s2", "fib_s3",
            "cam_r1", "cam_r2", "cam_r3", "cam_r4",
            "cam_s1", "cam_s2", "cam_s3", "cam_s4"
        }

        try:
            async with self.connection() as conn:
                count = 0
                async with conn.transaction():
                    for row in data:
                        ts = row.get("timestamp") or row.get("datetime")
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                        if not ts:
                            continue

                        columns = ["timestamp", "symbol", "timeframe", "source", "created_at"]
                        values = [ts, symbol, tf.value, source, datetime.now(timezone.utc)]
                        placeholders = ["$1", "$2", "$3", "$4", "$5"]
                        update_parts = []
                        idx = 6

                        for col in valid_columns:
                            if col in row and row[col] is not None:
                                columns.append(col)
                                values.append(float(row[col]))
                                placeholders.append(f"${idx}")
                                update_parts.append(f"{col} = EXCLUDED.{col}")
                                idx += 1

                        if len(columns) <= 5:
                            continue

                        query = f"""
                            INSERT INTO indicators_levels ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                            ON CONFLICT (timestamp, symbol, timeframe)
                            DO UPDATE SET {', '.join(update_parts)}, source = EXCLUDED.source
                        """

                        await conn.execute(query, *values)
                        count += 1

                await self._update_freshness(symbol, tf.value, "indicators_levels", count, source)
                logger.debug(f"Upserted {count} level indicators for {symbol}/{tf.value}")
                return count

        except Exception as e:
            logger.error(f"Failed to upsert level indicators for {symbol}/{tf.value}: {e}")
            return 0

    # ==================== Freshness Tracking ====================

    async def _update_freshness(
        self,
        symbol: str,
        timeframe: str,
        data_type: str,
        record_count: int,
        source: str,
    ) -> None:
        """Update data freshness tracking."""
        if not self.is_available:
            return

        query = """
            INSERT INTO data_freshness
                (symbol, timeframe, data_type, last_updated, last_timestamp, record_count, source)
            VALUES ($1, $2, $3, NOW(), NOW(), $4, $5)
            ON CONFLICT (symbol, timeframe, data_type)
            DO UPDATE SET
                last_updated = NOW(),
                record_count = data_freshness.record_count + EXCLUDED.record_count,
                source = EXCLUDED.source
        """

        try:
            async with self.connection() as conn:
                await conn.execute(query, symbol, timeframe, data_type, record_count, source)
        except Exception as e:
            logger.warning(f"Failed to update freshness tracking: {e}")

    async def get_freshness(
        self,
        symbol: str,
        timeframe: str,
        data_type: str = "ohlcv",
    ) -> Optional[dict[str, Any]]:
        """Get freshness status for symbol/timeframe."""
        if not self.is_available:
            return None

        query = """
            SELECT last_updated, last_timestamp, record_count, source
            FROM data_freshness
            WHERE symbol = $1 AND timeframe = $2 AND data_type = $3
        """

        try:
            async with self.connection() as conn:
                row = await conn.fetchrow(query, symbol, timeframe, data_type)

            if not row:
                return None

            return {
                "last_updated": row["last_updated"].isoformat(),
                "last_timestamp": row["last_timestamp"].isoformat(),
                "record_count": row["record_count"],
                "source": row["source"],
            }
        except Exception as e:
            logger.error(f"Failed to get freshness for {symbol}/{timeframe}: {e}")
            return None

    # ==================== Statistics ====================

    async def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        if not self.is_available:
            return {"available": False}

        try:
            async with self.connection() as conn:
                # Hypertable sizes
                size_query = """
                    SELECT hypertable_name, total_bytes
                    FROM timescaledb_information.hypertable_size_info
                    ORDER BY total_bytes DESC
                    LIMIT 20
                """
                try:
                    sizes = await conn.fetch(size_query)
                    table_sizes = {row["hypertable_name"]: row["total_bytes"] for row in sizes}
                except Exception:
                    table_sizes = {}

                # Row counts per timeframe
                row_counts = {}
                for tf in SUPPORTED_TIMEFRAMES:
                    table = get_ohlcv_table_name(tf)
                    try:
                        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                        row_counts[tf] = count
                    except Exception:
                        row_counts[tf] = 0

                return {
                    "available": True,
                    "table_sizes": table_sizes,
                    "row_counts": row_counts,
                    "pool_size": self._pool.get_size() if self._pool else 0,
                    "pool_free": self._pool.get_idle_size() if self._pool else 0,
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"available": True, "error": str(e)}


# Singleton instance
timescaledb_service = TimescaleDBService()
