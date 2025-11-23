"""Service for continuous synchronization of TimescaleDB data to RAG."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import asyncpg
from loguru import logger

from ..config import settings
from ..models.trading_data import TimeSeriesData
from .rag_service import RAGService


class TimescaleDBSyncService:
    """Service for continuously syncing TimescaleDB data to the RAG system."""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self._pool: Optional[asyncpg.Pool] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sync_time: Optional[datetime] = None
        self._sync_count = 0

    async def connect(self):
        """Connect to TimescaleDB."""
        try:
            self._pool = await asyncpg.create_pool(
                host=settings.timescaledb_host,
                port=settings.timescaledb_port,
                database=settings.timescaledb_database,
                user=settings.timescaledb_user,
                password=settings.timescaledb_password,
                min_size=2,
                max_size=10
            )
            logger.info(
                f"Connected to TimescaleDB at {settings.timescaledb_host}:{settings.timescaledb_port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    async def disconnect(self):
        """Disconnect from TimescaleDB."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from TimescaleDB")

    async def start(self):
        """Start the continuous sync process."""
        if self._running:
            logger.warning("Sync service is already running")
            return

        if not self._pool:
            await self.connect()

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            f"Started RAG sync service (interval: {settings.rag_sync_interval_seconds}s)"
        )

    async def stop(self):
        """Stop the continuous sync process."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stopped RAG sync service")

    async def _sync_loop(self):
        """Main sync loop that runs continuously."""
        while self._running:
            try:
                await self._perform_sync()
                await asyncio.sleep(settings.rag_sync_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)  # Wait before retry on error

    async def _perform_sync(self):
        """Perform a single sync operation."""
        if not self._pool:
            logger.warning("No database connection for sync")
            return

        try:
            # Determine time range for sync
            if self._last_sync_time:
                start_time = self._last_sync_time
            else:
                # On first run, sync last 7 days
                start_time = datetime.now() - timedelta(days=7)

            end_time = datetime.now()

            # Get available symbols
            symbols = await self._get_symbols()

            if not symbols:
                logger.warning("No symbols found in TimescaleDB")
                return

            total_synced = 0
            for symbol in symbols:
                synced = await self._sync_symbol_data(symbol, start_time, end_time)
                total_synced += synced

            self._last_sync_time = end_time
            self._sync_count += 1

            if total_synced > 0:
                # Persist RAG database after sync
                await self.rag_service.persist()
                logger.info(
                    f"Sync #{self._sync_count} completed: {total_synced} documents added"
                )
            else:
                logger.debug(f"Sync #{self._sync_count} completed: no new data")

        except Exception as e:
            logger.error(f"Error during sync: {e}")
            raise

    async def _get_symbols(self) -> list[str]:
        """Get list of available symbols from TimescaleDB."""
        async with self._pool.acquire() as conn:
            # Query the EasyInsight symbol table directly
            try:
                rows = await conn.fetch(
                    "SELECT DISTINCT symbol FROM symbol WHERE symbol IS NOT NULL LIMIT 100"
                )
                if rows:
                    symbols = [row["symbol"] for row in rows if row["symbol"]]
                    logger.info(f"Found {len(symbols)} symbols in database")
                    return symbols
            except Exception as e:
                logger.error(f"Error querying symbols: {e}")

            return []

    async def _sync_symbol_data(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> int:
        """Sync data for a specific symbol from EasyInsight symbol table."""
        synced_count = 0

        async with self._pool.acquire() as conn:
            # Query EasyInsight symbol table with all relevant data
            query = """
                SELECT
                    data_timestamp,
                    symbol,
                    category,
                    bid,
                    ask,
                    spread,
                    d1_open,
                    d1_high,
                    d1_low,
                    d1_close,
                    h1_open,
                    h1_high,
                    h1_low,
                    h1_close,
                    rsi14price_close,
                    macd12269price_close_main_line,
                    macd12269price_close_signal_line,
                    adx14_main_line,
                    adx14_plusdi_line,
                    adx14_minusdi_line,
                    bb200200price_close_base_line,
                    bb200200price_close_upper_band,
                    bb200200price_close_lower_band,
                    atr_d1,
                    cci14price_typical,
                    sto533mode_smasto_lowhigh_main_line,
                    sto533mode_smasto_lowhigh_signal_line,
                    ichimoku92652_tenkansen_line,
                    ichimoku92652_kijunsen_line,
                    strength_4h,
                    strength_1d,
                    strength_1w
                FROM symbol
                WHERE symbol = $1
                AND data_timestamp > $2
                AND data_timestamp <= $3
                ORDER BY data_timestamp DESC
                LIMIT $4
            """

            try:
                rows = await conn.fetch(
                    query, symbol, start_time, end_time, settings.rag_sync_batch_size
                )

                if not rows:
                    return 0

                for row in rows:
                    # Extract values with null handling
                    d1_open = float(row["d1_open"]) if row["d1_open"] else 0
                    d1_high = float(row["d1_high"]) if row["d1_high"] else 0
                    d1_low = float(row["d1_low"]) if row["d1_low"] else 0
                    d1_close = float(row["d1_close"]) if row["d1_close"] else 0
                    bid = float(row["bid"]) if row["bid"] else 0
                    ask = float(row["ask"]) if row["ask"] else 0
                    spread = float(row["spread"]) if row["spread"] else 0

                    # Calculate metrics
                    price_change = (
                        (d1_close - d1_open) / d1_open * 100
                    ) if d1_open > 0 else 0

                    volatility = (
                        (d1_high - d1_low) / d1_low * 100
                    ) if d1_low > 0 else 0

                    # Technical indicators
                    rsi = float(row["rsi14price_close"]) if row["rsi14price_close"] else None
                    macd_main = float(row["macd12269price_close_main_line"]) if row["macd12269price_close_main_line"] else None
                    macd_signal = float(row["macd12269price_close_signal_line"]) if row["macd12269price_close_signal_line"] else None
                    adx = float(row["adx14_main_line"]) if row["adx14_main_line"] else None
                    atr = float(row["atr_d1"]) if row["atr_d1"] else None
                    cci = float(row["cci14price_typical"]) if row["cci14price_typical"] else None
                    stoch_k = float(row["sto533mode_smasto_lowhigh_main_line"]) if row["sto533mode_smasto_lowhigh_main_line"] else None
                    stoch_d = float(row["sto533mode_smasto_lowhigh_signal_line"]) if row["sto533mode_smasto_lowhigh_signal_line"] else None

                    # Bollinger Bands
                    bb_middle = float(row["bb200200price_close_base_line"]) if row["bb200200price_close_base_line"] else None
                    bb_upper = float(row["bb200200price_close_upper_band"]) if row["bb200200price_close_upper_band"] else None
                    bb_lower = float(row["bb200200price_close_lower_band"]) if row["bb200200price_close_lower_band"] else None

                    # Determine trend based on indicators
                    trend = "neutral"
                    if rsi:
                        if rsi > 70:
                            trend = "overbought"
                        elif rsi < 30:
                            trend = "oversold"
                        elif price_change > 0:
                            trend = "bullish"
                        elif price_change < 0:
                            trend = "bearish"

                    # Create comprehensive document content
                    content = f"""
Marktdaten - {symbol}
Zeitstempel: {row['data_timestamp'].isoformat()}
Kategorie: {row['category'] or 'N/A'}

Preisdaten:
- Bid: {bid:.5f}
- Ask: {ask:.5f}
- Spread: {spread:.5f}
- D1 Open: {d1_open:.5f}
- D1 High: {d1_high:.5f}
- D1 Low: {d1_low:.5f}
- D1 Close: {d1_close:.5f}

Performance:
- Tagesänderung: {price_change:.2f}%
- Volatilität: {volatility:.2f}%
- Trend: {trend}

Technische Indikatoren:
- RSI (14): {f'{rsi:.2f}' if rsi else 'N/A'}
- MACD: {f'{macd_main:.5f}' if macd_main else 'N/A'} / Signal: {f'{macd_signal:.5f}' if macd_signal else 'N/A'}
- ADX (14): {f'{adx:.2f}' if adx else 'N/A'}
- ATR (D1): {f'{atr:.5f}' if atr else 'N/A'}
- CCI (14): {f'{cci:.2f}' if cci else 'N/A'}
- Stochastik: %K={f'{stoch_k:.2f}' if stoch_k else 'N/A'}, %D={f'{stoch_d:.2f}' if stoch_d else 'N/A'}
- Bollinger Bands: Upper={f'{bb_upper:.5f}' if bb_upper else 'N/A'}, Middle={f'{bb_middle:.5f}' if bb_middle else 'N/A'}, Lower={f'{bb_lower:.5f}' if bb_lower else 'N/A'}

Signale:
- RSI Signal: {'Überkauft' if rsi and rsi > 70 else 'Überverkauft' if rsi and rsi < 30 else 'Neutral' if rsi else 'N/A'}
- MACD Signal: {'Bullish' if macd_main and macd_signal and macd_main > macd_signal else 'Bearish' if macd_main and macd_signal else 'N/A'}
- Trend-Stärke (ADX): {'Stark' if adx and adx > 25 else 'Schwach' if adx else 'N/A'}
"""

                    # Build metadata
                    metadata = {
                        "timestamp": row["data_timestamp"].isoformat(),
                        "category": row["category"],
                        "bid": bid,
                        "ask": ask,
                        "spread": spread,
                        "d1_open": d1_open,
                        "d1_high": d1_high,
                        "d1_low": d1_low,
                        "d1_close": d1_close,
                        "price_change": price_change,
                        "volatility": volatility,
                        "trend": trend
                    }

                    # Add indicators to metadata if available
                    if rsi:
                        metadata["rsi"] = rsi
                    if macd_main:
                        metadata["macd_main"] = macd_main
                    if macd_signal:
                        metadata["macd_signal"] = macd_signal
                    if adx:
                        metadata["adx"] = adx
                    if atr:
                        metadata["atr"] = atr

                    # Add to RAG
                    await self.rag_service.add_custom_document(
                        content=content,
                        document_type="market_data",
                        symbol=symbol,
                        metadata=metadata
                    )
                    synced_count += 1

            except Exception as e:
                logger.error(f"Error syncing {symbol}: {e}")

        return synced_count

    async def manual_sync(self, days_back: int = 7) -> int:
        """Manually trigger a sync for the specified number of days."""
        if not self._pool:
            await self.connect()

        start_time = datetime.now() - timedelta(days=days_back)
        end_time = datetime.now()

        symbols = await self._get_symbols()
        total_synced = 0

        for symbol in symbols:
            synced = await self._sync_symbol_data(symbol, start_time, end_time)
            total_synced += synced

        if total_synced > 0:
            await self.rag_service.persist()

        logger.info(f"Manual sync completed: {total_synced} documents added")
        return total_synced

    def get_status(self) -> dict:
        """Get the current status of the sync service."""
        return {
            "running": self._running,
            "connected": self._pool is not None,
            "last_sync_time": (
                self._last_sync_time.isoformat() if self._last_sync_time else None
            ),
            "sync_count": self._sync_count,
            "sync_interval_seconds": settings.rag_sync_interval_seconds
        }
