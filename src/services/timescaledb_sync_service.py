"""Service for continuous synchronization of EasyInsight API data to RAG.

This service fetches market data from the EasyInsight API (not direct database)
and stores it in the RAG knowledge base for LLM analysis.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
import httpx
from loguru import logger

from ..config import settings
from .rag_service import RAGService


class TimescaleDBSyncService:
    """Service for syncing EasyInsight API data to the RAG system.

    Note: Despite the name, this service uses the EasyInsight REST API
    exclusively and does not connect directly to any database.
    """

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sync_time: Optional[datetime] = None
        self._sync_count = 0
        self._api_url = settings.easyinsight_api_url

    async def start(self):
        """Start the continuous sync process."""
        if self._running:
            logger.warning("Sync service is already running")
            return

        # Verify API connectivity
        if not await self._check_api_connectivity():
            raise Exception(f"Cannot connect to EasyInsight API at {self._api_url}")

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            f"Started RAG sync service (API: {self._api_url}, interval: {settings.rag_sync_interval_seconds}s)"
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

    async def _check_api_connectivity(self) -> bool:
        """Check if the EasyInsight API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._api_url}/symbols")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"API connectivity check failed: {e}")
            return False

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
        try:
            # Determine time range for sync
            if self._last_sync_time:
                start_time = self._last_sync_time
            else:
                # On first run, sync last 7 days
                start_time = datetime.now(timezone.utc) - timedelta(days=7)

            end_time = datetime.now(timezone.utc)

            # Get available symbols from API
            symbols = await self._get_symbols()

            if not symbols:
                logger.warning("No symbols found from EasyInsight API")
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
        """Get list of available symbols from EasyInsight API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._api_url}/symbols")
                response.raise_for_status()

                data = response.json()
                # API returns list of dicts with 'symbol', 'category', 'count', etc.
                symbols = [item.get('symbol') for item in data if item.get('symbol')]

                logger.info(f"Found {len(symbols)} symbols from EasyInsight API")
                return symbols

        except Exception as e:
            logger.error(f"Error fetching symbols from EasyInsight API: {e}")
            return []

    async def _sync_symbol_data(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> int:
        """Sync data for a specific symbol from EasyInsight API."""
        synced_count = 0
        batch_size = getattr(settings, 'rag_sync_batch_size', 100)

        try:
            # Request data from API
            limit = batch_size * 2

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self._api_url}/symbol-data-full/{symbol}",
                    params={"limit": limit}
                )
                response.raise_for_status()

                data = response.json()
                rows = data.get('data', [])

                if not rows:
                    return 0

                # Process only records within the time range
                for row in rows:
                    try:
                        # Parse timestamp
                        timestamp_str = row.get('snapshot_time')
                        if not timestamp_str:
                            continue

                        # Parse ISO format timestamp
                        data_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                        # Skip if outside time range
                        if data_timestamp <= start_time or data_timestamp > end_time:
                            continue

                        # Limit processing to batch size
                        if synced_count >= batch_size:
                            break

                        # Extract values with null handling
                        d1_open = float(row.get("d1_open") or 0)
                        d1_high = float(row.get("d1_high") or 0)
                        d1_low = float(row.get("d1_low") or 0)
                        d1_close = float(row.get("d1_close") or 0)
                        bid = float(row.get("bid") or 0)
                        ask = float(row.get("ask") or 0)
                        spread = float(row.get("spread") or 0)

                        # H1 (hourly) data
                        h1_open = float(row.get("h1_open") or 0)
                        h1_high = float(row.get("h1_high") or 0)
                        h1_low = float(row.get("h1_low") or 0)
                        h1_close = float(row.get("h1_close") or 0)

                        # M15 (15-minute) data
                        m15_open = float(row.get("m15_open") or 0)
                        m15_high = float(row.get("m15_high") or 0)
                        m15_low = float(row.get("m15_low") or 0)
                        m15_close = float(row.get("m15_close") or 0)

                        # Calculate metrics
                        price_change = ((d1_close - d1_open) / d1_open * 100) if d1_open > 0 else 0
                        volatility = ((d1_high - d1_low) / d1_low * 100) if d1_low > 0 else 0
                        h1_price_change = ((h1_close - h1_open) / h1_open * 100) if h1_open > 0 else 0
                        h1_volatility = ((h1_high - h1_low) / h1_low * 100) if h1_low > 0 else 0
                        m15_price_change = ((m15_close - m15_open) / m15_open * 100) if m15_open > 0 else 0
                        m15_volatility = ((m15_high - m15_low) / m15_low * 100) if m15_low > 0 else 0

                        # Technical indicators
                        rsi = float(row.get("rsi") or 0) if row.get("rsi") else None
                        macd_main = float(row.get("macd_main") or 0) if row.get("macd_main") else None
                        macd_signal = float(row.get("macd_signal") or 0) if row.get("macd_signal") else None
                        adx = float(row.get("adx_main") or 0) if row.get("adx_main") else None
                        adx_plus_di = float(row.get("adx_plusdi") or 0) if row.get("adx_plusdi") else None
                        adx_minus_di = float(row.get("adx_minusdi") or 0) if row.get("adx_minusdi") else None
                        atr = float(row.get("atr_d1") or 0) if row.get("atr_d1") else None
                        range_d1 = float(row.get("range_d1") or 0) if row.get("range_d1") else None
                        cci = float(row.get("cci") or 0) if row.get("cci") else None
                        ma100 = float(row.get("ma_100") or 0) if row.get("ma_100") else None
                        stoch_k = float(row.get("sto_main") or 0) if row.get("sto_main") else None
                        stoch_d = float(row.get("sto_signal") or 0) if row.get("sto_signal") else None

                        # Bollinger Bands
                        bb_middle = float(row.get("bb_base") or 0) if row.get("bb_base") else None
                        bb_upper = float(row.get("bb_upper") or 0) if row.get("bb_upper") else None
                        bb_lower = float(row.get("bb_lower") or 0) if row.get("bb_lower") else None

                        # Ichimoku indicators
                        ichimoku_tenkan = float(row.get("ichimoku_tenkan") or 0) if row.get("ichimoku_tenkan") else None
                        ichimoku_kijun = float(row.get("ichimoku_kijun") or 0) if row.get("ichimoku_kijun") else None
                        ichimoku_senkou_a = float(row.get("ichimoku_senkoua") or 0) if row.get("ichimoku_senkoua") else None
                        ichimoku_senkou_b = float(row.get("ichimoku_senkoub") or 0) if row.get("ichimoku_senkoub") else None
                        ichimoku_chikou = float(row.get("ichimoku_chikou") or 0) if row.get("ichimoku_chikou") else None

                        # Strength indicators
                        strength_4h = float(row.get("strength_4h") or 0) if row.get("strength_4h") else None
                        strength_1d = float(row.get("strength_1d") or 0) if row.get("strength_1d") else None
                        strength_1w = float(row.get("strength_1w") or 0) if row.get("strength_1w") else None

                        # Determine trend
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

                        # Signal interpretations
                        adx_direction = "N/A"
                        if adx_plus_di and adx_minus_di:
                            adx_direction = "Bullish (+DI > -DI)" if adx_plus_di > adx_minus_di else "Bearish (-DI > +DI)"

                        ichimoku_signal = "N/A"
                        if ichimoku_tenkan and ichimoku_kijun:
                            ichimoku_signal = "Bullish (Tenkan > Kijun)" if ichimoku_tenkan > ichimoku_kijun else "Bearish (Tenkan < Kijun)"

                        ichimoku_cloud = "N/A"
                        if ichimoku_senkou_a and ichimoku_senkou_b:
                            ichimoku_cloud = "Bullish Cloud (Senkou A > B)" if ichimoku_senkou_a > ichimoku_senkou_b else "Bearish Cloud (Senkou A < B)"

                        ma100_signal = "N/A"
                        if ma100 and d1_close:
                            ma100_signal = "Bullish (Preis über MA100)" if d1_close > ma100 else "Bearish (Preis unter MA100)"

                        # Create document content
                        content = f"""
Marktdaten - {symbol}
Zeitstempel: {data_timestamp.isoformat()}
Kategorie: {row.get('category', 'N/A')}

Preisdaten (Aktuell):
- Bid: {bid:.5f}
- Ask: {ask:.5f}
- Spread: {spread:.5f}

Tages-Daten (D1):
- Open: {d1_open:.5f}
- High: {d1_high:.5f}
- Low: {d1_low:.5f}
- Close: {d1_close:.5f}
- Änderung: {price_change:.2f}%
- Volatilität: {volatility:.2f}%
- Range: {f'{range_d1:.5f}' if range_d1 else 'N/A'}

Stunden-Daten (H1):
- Open: {h1_open:.5f}
- High: {h1_high:.5f}
- Low: {h1_low:.5f}
- Close: {h1_close:.5f}
- Änderung: {h1_price_change:.2f}%
- Volatilität: {h1_volatility:.2f}%

15-Minuten-Daten (M15):
- Open: {m15_open:.5f}
- High: {m15_high:.5f}
- Low: {m15_low:.5f}
- Close: {m15_close:.5f}
- Änderung: {m15_price_change:.2f}%
- Volatilität: {m15_volatility:.2f}%

Trend-Analyse:
- Trend: {trend}

Technische Indikatoren:
- RSI (14): {f'{rsi:.2f}' if rsi else 'N/A'}
- MACD: {f'{macd_main:.5f}' if macd_main else 'N/A'} / Signal: {f'{macd_signal:.5f}' if macd_signal else 'N/A'}
- ADX (14): {f'{adx:.2f}' if adx else 'N/A'} | +DI: {f'{adx_plus_di:.2f}' if adx_plus_di else 'N/A'} | -DI: {f'{adx_minus_di:.2f}' if adx_minus_di else 'N/A'}
- ATR (D1): {f'{atr:.5f}' if atr else 'N/A'}
- CCI (14): {f'{cci:.2f}' if cci else 'N/A'}
- Stochastik: %K={f'{stoch_k:.2f}' if stoch_k else 'N/A'}, %D={f'{stoch_d:.2f}' if stoch_d else 'N/A'}
- Bollinger Bands: Upper={f'{bb_upper:.5f}' if bb_upper else 'N/A'}, Middle={f'{bb_middle:.5f}' if bb_middle else 'N/A'}, Lower={f'{bb_lower:.5f}' if bb_lower else 'N/A'}
- SMA (100): {f'{ma100:.5f}' if ma100 else 'N/A'}

Ichimoku Cloud:
- Tenkan-sen: {f'{ichimoku_tenkan:.5f}' if ichimoku_tenkan else 'N/A'}
- Kijun-sen: {f'{ichimoku_kijun:.5f}' if ichimoku_kijun else 'N/A'}
- Senkou Span A: {f'{ichimoku_senkou_a:.5f}' if ichimoku_senkou_a else 'N/A'}
- Senkou Span B: {f'{ichimoku_senkou_b:.5f}' if ichimoku_senkou_b else 'N/A'}
- Chikou Span: {f'{ichimoku_chikou:.5f}' if ichimoku_chikou else 'N/A'}
- TK Signal: {ichimoku_signal}
- Cloud Signal: {ichimoku_cloud}

Stärke-Indikatoren:
- Stärke 4H: {f'{strength_4h:.2f}' if strength_4h else 'N/A'}
- Stärke 1D: {f'{strength_1d:.2f}' if strength_1d else 'N/A'}
- Stärke 1W: {f'{strength_1w:.2f}' if strength_1w else 'N/A'}

Signale:
- RSI Signal: {'Überkauft' if rsi and rsi > 70 else 'Überverkauft' if rsi and rsi < 30 else 'Neutral' if rsi else 'N/A'}
- MACD Signal: {'Bullish' if macd_main and macd_signal and macd_main > macd_signal else 'Bearish' if macd_main and macd_signal else 'N/A'}
- Trend-Stärke (ADX): {'Stark' if adx and adx > 25 else 'Schwach' if adx else 'N/A'}
- MA100 Signal: {ma100_signal}
- ADX Richtung: {adx_direction}
- Ichimoku Signal: {ichimoku_signal}
- Stochastik Signal: {'Überkauft' if stoch_k and stoch_k > 80 else 'Überverkauft' if stoch_k and stoch_k < 20 else 'Neutral' if stoch_k else 'N/A'}
"""

                        # Build metadata
                        metadata = {
                            "timestamp": data_timestamp.isoformat(),
                            "category": row.get("category", "N/A"),
                            "bid": bid,
                            "ask": ask,
                            "spread": spread,
                            "d1_open": d1_open,
                            "d1_high": d1_high,
                            "d1_low": d1_low,
                            "d1_close": d1_close,
                            "price_change": price_change,
                            "volatility": volatility,
                            "h1_open": h1_open,
                            "h1_high": h1_high,
                            "h1_low": h1_low,
                            "h1_close": h1_close,
                            "h1_price_change": h1_price_change,
                            "h1_volatility": h1_volatility,
                            "m15_open": m15_open,
                            "m15_high": m15_high,
                            "m15_low": m15_low,
                            "m15_close": m15_close,
                            "m15_price_change": m15_price_change,
                            "m15_volatility": m15_volatility,
                            "trend": trend
                        }

                        # Add optional indicators
                        for key, val in [
                            ("rsi", rsi), ("macd_main", macd_main), ("macd_signal", macd_signal),
                            ("adx", adx), ("adx_plus_di", adx_plus_di), ("adx_minus_di", adx_minus_di),
                            ("atr", atr), ("range_d1", range_d1), ("cci", cci), ("ma100", ma100),
                            ("stoch_k", stoch_k), ("stoch_d", stoch_d),
                            ("bb_upper", bb_upper), ("bb_middle", bb_middle), ("bb_lower", bb_lower),
                            ("ichimoku_tenkan", ichimoku_tenkan), ("ichimoku_kijun", ichimoku_kijun),
                            ("ichimoku_senkou_a", ichimoku_senkou_a), ("ichimoku_senkou_b", ichimoku_senkou_b),
                            ("ichimoku_chikou", ichimoku_chikou),
                            ("strength_4h", strength_4h), ("strength_1d", strength_1d), ("strength_1w", strength_1w)
                        ]:
                            if val is not None:
                                metadata[key] = val

                        # Add to RAG
                        await self.rag_service.add_custom_document(
                            content=content,
                            document_type="market_data",
                            symbol=symbol,
                            metadata=metadata
                        )
                        synced_count += 1

                    except Exception as row_error:
                        logger.error(f"Error processing row for {symbol}: {row_error}")
                        continue

        except Exception as e:
            logger.error(f"Error syncing {symbol}: {e}")

        return synced_count

    async def manual_sync(self, days_back: int = 7) -> int:
        """Manually trigger a sync for the specified number of days."""
        # Verify API connectivity
        if not await self._check_api_connectivity():
            raise Exception(f"Cannot connect to EasyInsight API at {self._api_url}")

        start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
        end_time = datetime.now(timezone.utc)

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
            "api_url": self._api_url,
            "last_sync_time": (
                self._last_sync_time.isoformat() if self._last_sync_time else None
            ),
            "sync_count": self._sync_count,
            "sync_interval_seconds": settings.rag_sync_interval_seconds
        }
