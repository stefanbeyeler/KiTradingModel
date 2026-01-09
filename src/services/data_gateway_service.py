"""Data Gateway Service - Zentraler Zugriffspunkt für externe Daten.

Dieser Service ist das einzige Gateway für den Zugriff auf externe Datenquellen:
- TwelveData API (primär für OHLC-Daten aller Timeframes)
- EasyInsight API (Fallback + zusätzliche Indikatoren)
- Yahoo Finance (2. Fallback)

Alle anderen Services (NHITS, RAG, LLM, Analysis) MÜSSEN diesen Service
für externe Datenzugriffe verwenden.

Datenfluss (3-Layer-Caching):
1. Redis Cache → 2. TimescaleDB → 3. Externe APIs

Caching:
- Redis als primärer Cache (verteilt, persistent)
- TimescaleDB als persistente Speicherung (wenn verfügbar)
- In-Memory Fallback wenn Redis nicht verfügbar
- Kategorisierte TTL-Werte für verschiedene Datentypen

Timeframes:
- Alle Timeframes werden beim Laden standardisiert (siehe src/config/timeframes.py)
- Standard-Format: M1, M5, M15, M30, H1, H4, D1, W1, MN
- Downstream-Services erhalten immer konsistente Timeframe-Bezeichnungen

Siehe: DEVELOPMENT_GUIDELINES.md - Datenzugriff-Architektur
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Any
import httpx
from loguru import logger

from ..config import settings
from ..config.timeframes import (
    Timeframe,
    normalize_timeframe,
    normalize_timeframe_safe,
    to_twelvedata,
    to_easyinsight,
    TIMEFRAME_TO_TWELVEDATA,
)
from .cache_service import cache_service, CacheCategory
from .data_repository import data_repository


class DataGatewayService:
    """
    Zentraler Gateway-Service für alle externen Datenzugriffe.

    Dieser Service implementiert:
    - TwelveData als primäre Quelle für OHLC-Daten (alle Timeframes)
    - EasyInsight als Fallback und für zusätzliche Indikatoren
    - Yahoo Finance als 2. Fallback
    - Caching für häufig abgerufene Daten
    - Retry-Logik und Fehlerbehandlung
    """

    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        self._easyinsight_url = settings.easyinsight_api_url
        base_url = getattr(settings, 'data_service_url', 'http://trading-data:3001')
        # Ensure /api/v1 suffix
        self._data_service_url = base_url.rstrip('/') + '/api/v1' if not base_url.endswith('/api/v1') else base_url
        # Cache Service wird verwendet - kein lokaler Cache mehr nötig
        self._cache_initialized = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Close HTTP client, cache connection, and repository."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
        await cache_service.disconnect()
        await data_repository.close()

    async def _ensure_cache_connected(self):
        """Ensure cache service and repository are connected."""
        if not self._cache_initialized:
            await cache_service.connect()
            await data_repository.initialize()
            self._cache_initialized = True

    # ==================== Symbol Management ====================

    async def get_available_symbols(self) -> list[dict]:
        """
        Get list of available trading symbols from EasyInsight API.

        Returns:
            List of symbol dictionaries with 'symbol', 'category', 'count', etc.
        """
        await self._ensure_cache_connected()

        # Check Redis cache first
        cached = await cache_service.get(CacheCategory.SYMBOLS, "list")
        if cached:
            logger.debug(f"Returning {len(cached)} symbols from Redis cache")
            return cached

        try:
            client = await self._get_client()
            response = await client.get(f"{self._easyinsight_url}/symbols")
            response.raise_for_status()

            data = response.json()
            # Cache with SYMBOLS TTL (3600s)
            await cache_service.set(CacheCategory.SYMBOLS, data, "list")
            logger.info(f"Fetched and cached {len(data)} symbols from EasyInsight API")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch symbols from EasyInsight API: {e}")
            raise

    async def get_symbol_names(self) -> list[str]:
        """
        Get list of symbol names only.

        Returns:
            List of symbol strings (e.g., ['BTCUSD', 'EURUSD', ...])
        """
        symbols = await self.get_available_symbols()
        return sorted([s.get('symbol') for s in symbols if s.get('symbol')])

    # ==================== Market Data - Latest ====================

    async def get_latest_market_data(self, symbol: str) -> Optional[dict]:
        """
        Get the latest market data snapshot for a symbol.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)

        Returns:
            Dictionary with latest OHLCV, indicators, and price data
        """
        await self._ensure_cache_connected()

        # Check Redis cache first (MARKET_DATA TTL: 60s)
        cached = await cache_service.get(CacheCategory.MARKET_DATA, symbol, "latest")
        if cached:
            return cached

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._easyinsight_url}/symbol-latest-full/{symbol}"
            )
            response.raise_for_status()

            data = response.json()
            if data:
                await cache_service.set(CacheCategory.MARKET_DATA, data, symbol, "latest")
                logger.debug(f"Fetched latest market data for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch latest market data for {symbol}: {e}")
            return None

    # ==================== Market Data - Historical ====================

    async def get_historical_data(
        self,
        symbol: str,
        limit: int = 500,
        timeframe: str = "H1",
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Get historical OHLCV data for a symbol.

        Uses 3-layer caching: Redis → TimescaleDB → External APIs

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            limit: Number of data points to fetch
            timeframe: Timeframe in beliebigem Format (wird automatisch normalisiert)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            List of OHLCV dictionaries with standardized timeframe field
        """
        await self._ensure_cache_connected()

        # Normalize timeframe to standard format
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
        tf_str = tf.value  # Standard format: H1, D1, etc.

        # 1. Try Repository (Redis + TimescaleDB)
        data, source = await data_repository.get_ohlcv(
            symbol=symbol,
            timeframe=tf_str,
            limit=limit,
            force_refresh=force_refresh,
        )

        if data:
            logger.debug(f"Repository returned {len(data)} rows from {source} for {symbol}/{tf_str}")
            return data

        # 2. Fetch from external API
        try:
            client = await self._get_client()
            # Use internal training-data endpoint
            response = await client.get(
                f"{self._data_service_url}/training-data/{symbol}",
                params={"limit": limit, "timeframe": tf_str}
            )
            response.raise_for_status()

            result = response.json()
            rows = result.get('data', [])

            # Standardize timeframe in response data
            rows = self._standardize_timeframe_in_data(rows, tf)

            # 3. Save to repository (TimescaleDB + Redis)
            if rows:
                await data_repository.save_ohlcv(
                    symbol=symbol,
                    timeframe=tf_str,
                    data=rows,
                    source="easyinsight",
                )

            logger.debug(f"Fetched {len(rows)} historical data points for {symbol} {tf_str}")
            return rows

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return []

    def _standardize_timeframe_in_data(self, rows: list[dict], tf: Timeframe) -> list[dict]:
        """
        Standardize timeframe field in data rows.

        Ensures all rows have consistent 'timeframe' field with standard format.

        Args:
            rows: List of data dictionaries
            tf: Normalized Timeframe enum

        Returns:
            List with standardized timeframe field
        """
        tf_str = tf.value
        ei_prefix = to_easyinsight(tf)  # e.g., "h1", "d1"

        for row in rows:
            # Add/update standardized timeframe field
            row["timeframe"] = tf_str

            # Ensure OHLC fields use standard naming
            # Map from EasyInsight prefix format (h1_open) to standard (open)
            if f"{ei_prefix}_open" in row and "open" not in row:
                row["open"] = row[f"{ei_prefix}_open"]
                row["high"] = row.get(f"{ei_prefix}_high", row.get("high", 0))
                row["low"] = row.get(f"{ei_prefix}_low", row.get("low", 0))
                row["close"] = row.get(f"{ei_prefix}_close", row.get("close", 0))
                row["volume"] = row.get(f"{ei_prefix}_volume", row.get("volume", 0))

        return rows

    async def get_historical_data_with_fallback(
        self,
        symbol: str,
        limit: int = 500,
        timeframe: str = "H1",
        force_refresh: bool = False,
    ) -> tuple[list[dict], str]:
        """
        Get historical data with TwelveData as primary source.

        Data flow: 1. Repository (Cache+DB) → 2. TwelveData → 3. EasyInsight

        TwelveData is used exclusively for pattern analysis to ensure consistent
        OHLC data across all timeframes. Timeframe is automatically normalized
        to the standard format.

        Supported timeframes (beliebiges Format wird akzeptiert):
            - M1, 1m, 1min: 1 minute
            - M5, 5m, 5min: 5 minutes
            - M15, 15m, 15min: 15 minutes
            - M30, 30m, 30min: 30 minutes
            - H1, 1h, 1hour: 1 hour
            - H4, 4h: 4 hours
            - D1, 1d, 1day, daily: 1 day
            - W1, 1wk, 1week, weekly: 1 week
            - MN, 1mo, 1month, monthly: 1 month

        Args:
            symbol: Trading symbol
            limit: Number of data points
            timeframe: Timeframe in beliebigem Format (wird automatisch normalisiert)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            Tuple of (data_list, source) where source is 'cache', 'db', 'twelvedata' or 'easyinsight'
            Data contains standardized 'timeframe' field in format: M1, H1, D1, etc.
        """
        await self._ensure_cache_connected()

        # Normalize timeframe to standard format
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
        tf_str = tf.value

        # 1. Try Repository (Redis + TimescaleDB)
        if not force_refresh:
            data, source = await data_repository.get_ohlcv(
                symbol=symbol,
                timeframe=tf_str,
                limit=limit,
            )
            if data:
                logger.debug(f"Repository returned {len(data)} rows from {source}")
                return data, source

        # 2. Use TwelveData as primary external source for all timeframes (pattern analysis)
        td_data = await self._get_twelvedata_candles(symbol, tf_str, limit)
        if td_data:
            # Ensure standardized timeframe in data
            td_data = self._standardize_timeframe_in_data(td_data, tf)
            # Save to repository
            await data_repository.save_ohlcv(
                symbol=symbol,
                timeframe=tf_str,
                data=td_data,
                source="twelvedata",
            )
            return td_data, "twelvedata"

        # 3. Fallback to EasyInsight only if TwelveData fails
        logger.warning(f"TwelveData failed for {symbol} {tf_str}, trying EasyInsight fallback")
        data = await self.get_historical_data(symbol, limit, tf_str, force_refresh=True)
        return data, "easyinsight"

    def _convert_twelvedata_to_easyinsight(
        self,
        td_values: list[dict],
        symbol: str,
        timeframe: str
    ) -> list[dict]:
        """
        Convert TwelveData format to standardized format.

        Args:
            td_values: Raw TwelveData response values
            symbol: Trading symbol
            timeframe: Timeframe (already normalized to standard format)

        Returns:
            List of standardized OHLCV dictionaries
        """
        converted = []

        # Normalize timeframe to get consistent naming
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
        tf_str = tf.value  # Standard format: H1, D1, etc.
        ei_prefix = to_easyinsight(tf)  # e.g., "h1", "d1"

        for row in td_values:
            try:
                # TwelveData format: datetime, open, high, low, close, volume
                open_val = float(row.get("open", 0))
                high_val = float(row.get("high", 0))
                low_val = float(row.get("low", 0))
                close_val = float(row.get("close", 0))
                volume_val = float(row.get("volume", 0)) if row.get("volume") else 0

                entry = {
                    "symbol": symbol,
                    "timeframe": tf_str,  # Standardized timeframe field
                    "snapshot_time": row.get("datetime"),
                    # Standard OHLCV fields (for downstream services)
                    "open": open_val,
                    "high": high_val,
                    "low": low_val,
                    "close": close_val,
                    "volume": volume_val,
                    # EasyInsight-compatible prefixed fields
                    f"{ei_prefix}_open": open_val,
                    f"{ei_prefix}_high": high_val,
                    f"{ei_prefix}_low": low_val,
                    f"{ei_prefix}_close": close_val,
                }
                converted.append(entry)
            except Exception as e:
                logger.warning(f"Failed to convert TwelveData row: {e}")
                continue

        return converted

    async def _get_twelvedata_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> list[dict]:
        """
        Get candle data via Data Service (which proxies to TwelveData).

        ARCHITEKTUR-KONFORM: Alle externen Datenzugriffe gehen über den Data Service.
        Der Data Service (Port 3001) ist das einzige Gateway für TwelveData/EasyInsight.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            timeframe: Timeframe in beliebigem Format (wird normalisiert)
            limit: Number of candles to fetch (max 5000)

        Returns:
            List of candle dictionaries with standardized format
        """
        try:
            # Normalize and convert timeframe to TwelveData format
            tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
            td_interval = to_twelvedata(tf)

            # TwelveData max outputsize is 5000
            fetch_limit = min(limit, 5000)

            # Fetch via Data Service HTTP endpoint (architekturkonform)
            client = await self._get_client()
            url = f"{self._data_service_url}/twelvedata/time_series/{symbol}"
            params = {
                "interval": td_interval,
                "outputsize": fetch_limit
            }

            logger.debug(f"Fetching OHLC from Data Service: {url} params={params}")
            response = await client.get(url, params=params)

            if response.status_code == 200:
                td_data = response.json()

                # Check for unsupported symbol or error
                if td_data.get("unsupported") or td_data.get("error"):
                    logger.warning(f"Data Service: {symbol} not supported by TwelveData")
                    return []

                if td_data.get("values"):
                    # Convert to standardized format
                    converted = self._convert_twelvedata_to_easyinsight(
                        td_data["values"],
                        symbol,
                        tf.value  # Pass normalized timeframe
                    )
                    logger.info(f"Data Service returned {len(converted)} {tf.value} candles for {symbol}")
                    return converted
            else:
                logger.warning(f"Data Service returned {response.status_code} for {symbol}")

        except Exception as e:
            logger.error(f"Failed to get candles via Data Service for {symbol} {timeframe}: {e}")

        return []

    # ==================== Training Data (for NHITS) ====================

    async def get_training_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        min_points: int = 500
    ) -> tuple[list[dict], str]:
        """
        Get training data for NHITS model with automatic fallback.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN)
            min_points: Minimum required data points

        Returns:
            Tuple of (data_list, source)
        """
        return await self.get_historical_data_with_fallback(
            symbol=symbol,
            limit=min_points,
            timeframe=timeframe
        )

    # ==================== Symbol Info ====================

    async def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get detailed information about a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol metadata
        """
        symbols = await self.get_available_symbols()
        for s in symbols:
            if s.get('symbol') == symbol.upper():
                return s
        return None

    # ==================== Bulk Data ====================

    async def get_all_latest_market_data(self) -> list[dict]:
        """
        Get latest market data for all symbols.

        Returns:
            List of dictionaries with OHLCV and indicator data for all symbols
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._easyinsight_url}/symbol-data-full",
                params={"limit": 1}  # Get only latest snapshot for each symbol
            )
            response.raise_for_status()

            data = response.json()
            rows = data.get('data', [])
            logger.debug(f"Fetched latest data for {len(rows)} symbols")
            return rows

        except Exception as e:
            logger.error(f"Failed to fetch all latest market data: {e}")
            return []

    async def get_easyinsight_historical(
        self,
        symbol: str,
        timeframe: str = "H1",
        limit: int = 500,
    ) -> list[dict]:
        """
        Get historical data with indicators from EasyInsight API.

        EasyInsight provides OHLCV data along with pre-calculated indicators:
        - rsi, macd_main, macd_signal, cci
        - adx_main, adx_plusdi, adx_minusdi
        - atr_d1, atr_pct_d1
        - bb_base, bb_lower, bb_upper
        - sto_main, sto_signal
        - ichimoku_* (tenkan, kijun, senkoua, senkoub, chikou)
        - ma_10, strength_1d, strength_1w, strength_4h

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            timeframe: Timeframe (default H1 - EasyInsight provides best indicator data for H1)
            limit: Number of data points

        Returns:
            List of OHLCV dictionaries with indicator data
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._easyinsight_url}/symbol-data-full",
                params={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                }
            )
            response.raise_for_status()

            data = response.json()
            rows = data.get('data', [])

            if rows:
                logger.debug(f"Fetched {len(rows)} EasyInsight rows with indicators for {symbol}/{timeframe}")

            return rows

        except Exception as e:
            logger.error(f"Failed to fetch EasyInsight historical data for {symbol}: {e}")
            return []

    # ==================== API Health ====================

    async def check_easyinsight_health(self) -> bool:
        """Check if EasyInsight API is accessible."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._easyinsight_url}/symbols",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"EasyInsight API health check failed: {e}")
            return False

    def get_status(self) -> dict:
        """Get gateway service status."""
        cache_stats = cache_service.get_stats()
        return {
            "easyinsight_url": self._easyinsight_url,
            "client_active": self._http_client is not None and not self._http_client.is_closed,
            "cache": cache_stats
        }

    async def get_cache_health(self) -> dict:
        """Get cache health status."""
        await self._ensure_cache_connected()
        return await cache_service.health_check()

    async def clear_cache(self, category: Optional[CacheCategory] = None) -> int:
        """
        Clear cache entries.

        Args:
            category: Optional category to clear. If None, clears all.

        Returns:
            Number of entries cleared.
        """
        await self._ensure_cache_connected()
        if category:
            return await cache_service.clear_category(category)
        return await cache_service.clear_all()

    # ==================== TwelveData Indicators ====================

    async def get_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "1h",
        outputsize: int = 100,
        **kwargs,
    ) -> dict:
        """
        Get technical indicator data for a symbol via TwelveData.

        This method provides access to TwelveData's technical indicators
        when EasyInsight data is insufficient or unavailable. Interval is
        automatically normalized to the standard format.

        Args:
            symbol: Trading symbol (will be converted to TwelveData format)
            indicator: Indicator name (rsi, macd, bbands, stoch, adx, etc.)
            interval: Time interval in beliebigem Format (wird normalisiert)
            outputsize: Number of data points
            **kwargs: Additional indicator-specific parameters

        Returns:
            Dictionary with indicator values and standardized timeframe field
        """
        try:
            from .twelvedata_service import twelvedata_service

            # Normalize interval to standard format and convert to TwelveData
            tf = normalize_timeframe_safe(interval, Timeframe.H1)
            td_interval = to_twelvedata(tf)

            # Fetch indicator from TwelveData
            # Symbol conversion (BTCUSD -> BTC/USD) handled internally by TwelveDataService
            result = await twelvedata_service.get_technical_indicators(
                symbol=symbol,
                interval=td_interval,
                indicator=indicator,
                outputsize=outputsize,
                **kwargs,
            )

            # Add standardized timeframe to result
            if "error" not in result:
                result["timeframe"] = tf.value  # Standardized timeframe

            return result

        except Exception as e:
            logger.error(f"Failed to get indicator {indicator} for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "indicator": indicator}

    async def get_rsi(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get RSI indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="rsi",
            interval=interval,
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_macd(
        self,
        symbol: str,
        interval: str = "1h",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        outputsize: int = 100,
    ) -> dict:
        """Get MACD indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="macd",
            interval=interval,
            outputsize=outputsize,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    async def get_bollinger_bands(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 20,
        sd: float = 2.0,
        outputsize: int = 100,
    ) -> dict:
        """Get Bollinger Bands indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="bbands",
            interval=interval,
            outputsize=outputsize,
            time_period=time_period,
            sd=sd,
        )

    async def get_stochastic(
        self,
        symbol: str,
        interval: str = "1h",
        fast_k_period: int = 14,
        slow_k_period: int = 3,
        slow_d_period: int = 3,
        outputsize: int = 100,
    ) -> dict:
        """Get Stochastic Oscillator indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="stoch",
            interval=interval,
            outputsize=outputsize,
            fast_k_period=fast_k_period,
            slow_k_period=slow_k_period,
            slow_d_period=slow_d_period,
        )

    async def get_adx(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get ADX indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="adx",
            interval=interval,
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_atr(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get ATR indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="atr",
            interval=interval,
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_ichimoku(
        self,
        symbol: str,
        interval: str = "1h",
        outputsize: int = 100,
    ) -> dict:
        """Get Ichimoku Cloud indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="ichimoku",
            interval=interval,
            outputsize=outputsize,
        )

    async def get_supertrend(
        self,
        symbol: str,
        interval: str = "1h",
        period: int = 10,
        multiplier: float = 3.0,
        outputsize: int = 100,
    ) -> dict:
        """Get Supertrend indicator via TwelveData."""
        return await self.get_indicator(
            symbol=symbol,
            indicator="supertrend",
            interval=interval,
            outputsize=outputsize,
            period=period,
            multiplier=multiplier,
        )

    async def get_multiple_indicators(
        self,
        symbol: str,
        indicators: list[str],
        interval: str = "1h",
        outputsize: int = 100,
    ) -> dict:
        """
        Get multiple technical indicators for a symbol via TwelveData.

        Note: Each indicator requires a separate API call with rate limiting.
        Interval is automatically normalized to the standard format.

        Args:
            symbol: Trading symbol
            indicators: List of indicator names
            interval: Time interval in beliebigem Format (wird normalisiert)
            outputsize: Number of data points

        Returns:
            Dictionary with results for each indicator and standardized timeframe
        """
        try:
            from .twelvedata_service import twelvedata_service

            # Normalize interval to standard format and convert to TwelveData
            tf = normalize_timeframe_safe(interval, Timeframe.H1)
            td_interval = to_twelvedata(tf)

            # Symbol conversion (BTCUSD -> BTC/USD) handled internally by TwelveDataService
            result = await twelvedata_service.get_multiple_indicators(
                symbol=symbol,
                indicators=indicators,
                interval=td_interval,
                outputsize=outputsize,
            )

            result["timeframe"] = tf.value  # Standardized timeframe
            return result

        except Exception as e:
            logger.error(f"Failed to get multiple indicators for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    async def get_complete_analysis(
        self,
        symbol: str,
        interval: str = "1h",
        outputsize: int = 100,
    ) -> dict:
        """
        Get complete technical analysis with all major indicators via TwelveData.

        Includes: RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR, CCI, OBV

        Note: This makes 8 API calls with rate limiting.

        Args:
            symbol: Trading symbol
            interval: Time interval
            outputsize: Number of data points

        Returns:
            Dictionary with all major indicator values
        """
        indicators = ["rsi", "macd", "bbands", "stoch", "adx", "atr", "cci", "obv"]
        return await self.get_multiple_indicators(
            symbol=symbol,
            indicators=indicators,
            interval=interval,
            outputsize=outputsize,
        )


# Global singleton instance
data_gateway = DataGatewayService()
