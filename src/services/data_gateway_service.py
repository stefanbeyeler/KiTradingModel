"""Data Gateway Service - Zentraler Zugriffspunkt für externe Daten.

Dieser Service ist das einzige Gateway für den Zugriff auf externe Datenquellen:
- EasyInsight API (TimescaleDB)
- TwelveData API (Fallback)

Alle anderen Services (NHITS, RAG, LLM, Analysis) MÜSSEN diesen Service
für externe Datenzugriffe verwenden.

Siehe: DEVELOPMENT_GUIDELINES.md - Datenzugriff-Architektur
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Any
import httpx
from loguru import logger

from ..config import settings


class DataGatewayService:
    """
    Zentraler Gateway-Service für alle externen Datenzugriffe.

    Dieser Service implementiert:
    - Einheitlichen Zugriff auf EasyInsight API
    - Fallback zu TwelveData bei Fehlern
    - Caching für häufig abgerufene Daten
    - Retry-Logik und Fehlerbehandlung
    """

    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        self._easyinsight_url = settings.easyinsight_api_url
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl_seconds = 60  # 1 Minute Cache

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            cached_time, value = self._cache[key]
            if datetime.now() - cached_time < timedelta(seconds=self._cache_ttl_seconds):
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Set cached value."""
        self._cache[key] = (datetime.now(), value)

    # ==================== Symbol Management ====================

    async def get_available_symbols(self) -> list[dict]:
        """
        Get list of available trading symbols from EasyInsight API.

        Returns:
            List of symbol dictionaries with 'symbol', 'category', 'count', etc.
        """
        cache_key = "symbols_list"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            client = await self._get_client()
            response = await client.get(f"{self._easyinsight_url}/symbols")
            response.raise_for_status()

            data = response.json()
            self._set_cached(cache_key, data)
            logger.debug(f"Fetched {len(data)} symbols from EasyInsight API")
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
        cache_key = f"latest_{symbol}"
        cached = self._get_cached(cache_key)
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
                self._set_cached(cache_key, data)
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
        timeframe: str = "H1"
    ) -> list[dict]:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            limit: Number of data points to fetch
            timeframe: Timeframe (M15, H1, D1)

        Returns:
            List of OHLCV dictionaries
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._easyinsight_url}/symbol-data-full/{symbol}",
                params={"limit": limit}
            )
            response.raise_for_status()

            data = response.json()
            rows = data.get('data', [])
            logger.debug(f"Fetched {len(rows)} historical data points for {symbol}")
            return rows

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return []

    async def get_historical_data_with_fallback(
        self,
        symbol: str,
        limit: int = 500,
        timeframe: str = "H1"
    ) -> tuple[list[dict], str]:
        """
        Get historical data with TwelveData fallback.

        Args:
            symbol: Trading symbol
            limit: Number of data points
            timeframe: Timeframe

        Returns:
            Tuple of (data_list, source) where source is 'easyinsight' or 'twelvedata'
        """
        # Try EasyInsight first
        data = await self.get_historical_data(symbol, limit, timeframe)
        if data and len(data) >= limit * 0.8:  # At least 80% of requested data
            return data, "easyinsight"

        # Fallback to TwelveData
        logger.warning(
            f"EasyInsight returned insufficient data for {symbol} "
            f"({len(data)}/{limit}), trying TwelveData fallback"
        )

        try:
            from .twelvedata_service import twelvedata_service
            from ..services.symbol_service import symbol_service

            # Get TwelveData symbol format
            managed_symbol = await symbol_service.get_symbol(symbol)
            td_symbol = None
            if managed_symbol and managed_symbol.twelvedata_symbol:
                td_symbol = managed_symbol.twelvedata_symbol
            else:
                # Generate TwelveData symbol format
                td_symbol = symbol_service._generate_twelvedata_symbol(
                    symbol,
                    managed_symbol.category if managed_symbol else None
                )

            if not td_symbol:
                logger.warning(f"No TwelveData symbol mapping for {symbol}")
                return data, "easyinsight"

            # Map timeframe
            td_interval_map = {
                "M15": "15min",
                "H1": "1h",
                "H4": "4h",
                "D1": "1day"
            }
            td_interval = td_interval_map.get(timeframe.upper(), "1h")

            # Fetch from TwelveData
            td_data = await twelvedata_service.get_time_series(
                symbol=td_symbol,
                interval=td_interval,
                outputsize=limit
            )

            if td_data and td_data.get("values"):
                # Convert TwelveData format to EasyInsight format
                converted = self._convert_twelvedata_to_easyinsight(
                    td_data["values"],
                    symbol,
                    timeframe
                )
                logger.info(f"TwelveData fallback returned {len(converted)} data points for {symbol}")
                return converted, "twelvedata"

        except Exception as e:
            logger.error(f"TwelveData fallback failed for {symbol}: {e}")

        return data, "easyinsight"

    def _convert_twelvedata_to_easyinsight(
        self,
        td_values: list[dict],
        symbol: str,
        timeframe: str
    ) -> list[dict]:
        """Convert TwelveData format to EasyInsight format."""
        converted = []
        tf_lower = timeframe.lower()

        for row in td_values:
            try:
                # TwelveData format: datetime, open, high, low, close, volume
                entry = {
                    "symbol": symbol,
                    "snapshot_time": row.get("datetime"),
                    f"{tf_lower}_open": float(row.get("open", 0)),
                    f"{tf_lower}_high": float(row.get("high", 0)),
                    f"{tf_lower}_low": float(row.get("low", 0)),
                    f"{tf_lower}_close": float(row.get("close", 0)),
                }
                # Also populate standard OHLC fields
                entry["h1_open"] = entry.get(f"{tf_lower}_open", 0)
                entry["h1_high"] = entry.get(f"{tf_lower}_high", 0)
                entry["h1_low"] = entry.get(f"{tf_lower}_low", 0)
                entry["h1_close"] = entry.get(f"{tf_lower}_close", 0)
                converted.append(entry)
            except Exception as e:
                logger.warning(f"Failed to convert TwelveData row: {e}")
                continue

        return converted

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
            timeframe: Timeframe (M15, H1, D1)
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
        return {
            "easyinsight_url": self._easyinsight_url,
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "client_active": self._http_client is not None and not self._http_client.is_closed
        }

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
        when EasyInsight data is insufficient or unavailable.

        Args:
            symbol: Trading symbol (will be converted to TwelveData format)
            indicator: Indicator name (rsi, macd, bbands, stoch, adx, etc.)
            interval: Time interval ('1min', '5min', '15min', '1h', '4h', '1day')
            outputsize: Number of data points
            **kwargs: Additional indicator-specific parameters

        Returns:
            Dictionary with indicator values or error
        """
        try:
            from .twelvedata_service import twelvedata_service
            from .symbol_service import symbol_service

            # Get TwelveData symbol format
            managed_symbol = await symbol_service.get_symbol(symbol)
            td_symbol = None
            if managed_symbol and managed_symbol.twelvedata_symbol:
                td_symbol = managed_symbol.twelvedata_symbol
            else:
                # Generate TwelveData symbol format
                td_symbol = symbol_service._generate_twelvedata_symbol(
                    symbol,
                    managed_symbol.category if managed_symbol else None
                )

            if not td_symbol:
                return {"error": f"No TwelveData symbol mapping for {symbol}"}

            # Map interval formats
            td_interval_map = {
                "M15": "15min", "15min": "15min",
                "H1": "1h", "1h": "1h",
                "H4": "4h", "4h": "4h",
                "D1": "1day", "1day": "1day",
            }
            td_interval = td_interval_map.get(interval.upper(), interval)

            # Fetch indicator from TwelveData
            result = await twelvedata_service.get_technical_indicators(
                symbol=td_symbol,
                interval=td_interval,
                indicator=indicator,
                outputsize=outputsize,
                **kwargs,
            )

            # Add original symbol to result
            if "error" not in result:
                result["original_symbol"] = symbol
                result["twelvedata_symbol"] = td_symbol

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

        Args:
            symbol: Trading symbol
            indicators: List of indicator names
            interval: Time interval
            outputsize: Number of data points

        Returns:
            Dictionary with results for each indicator
        """
        try:
            from .twelvedata_service import twelvedata_service
            from .symbol_service import symbol_service

            # Get TwelveData symbol format
            managed_symbol = await symbol_service.get_symbol(symbol)
            td_symbol = None
            if managed_symbol and managed_symbol.twelvedata_symbol:
                td_symbol = managed_symbol.twelvedata_symbol
            else:
                td_symbol = symbol_service._generate_twelvedata_symbol(
                    symbol,
                    managed_symbol.category if managed_symbol else None
                )

            if not td_symbol:
                return {"error": f"No TwelveData symbol mapping for {symbol}"}

            # Map interval
            td_interval_map = {
                "M15": "15min", "H1": "1h", "H4": "4h", "D1": "1day"
            }
            td_interval = td_interval_map.get(interval.upper(), interval)

            result = await twelvedata_service.get_multiple_indicators(
                symbol=td_symbol,
                indicators=indicators,
                interval=td_interval,
                outputsize=outputsize,
            )

            result["original_symbol"] = symbol
            result["twelvedata_symbol"] = td_symbol
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
