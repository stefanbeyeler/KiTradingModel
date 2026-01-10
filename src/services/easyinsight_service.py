"""EasyInsight API Service - TwelveData-kompatible API für Marktdaten und Indikatoren.

Dieser Service nutzt die neuen TwelveData-kompatiblen EasyInsight API-Endpoints:
- /time_series/{symbol} - OHLCV-Daten für alle Timeframes
- /rsi/{symbol}, /macd/{symbol}, etc. - Technische Indikatoren
- /indicators/{symbol} - Multiple Indikatoren in einem Request
- /quote/{symbol}, /price/{symbol} - Echtzeit-Daten

Siehe: docs/EASYINSIGHT_API_MIGRATION_GUIDE.md

WICHTIG: Timeframes werden vom Data Gateway Service normalisiert und zum
TwelveData-Format konvertiert (siehe src/config/timeframes.py).
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, Any
from loguru import logger
import httpx

from ..config import settings
from .cache_service import cache_service, CacheCategory, TIMEFRAME_TTL


class EasyInsightService:
    """Service for accessing EasyInsight API with TwelveData-compatible endpoints."""

    # Supported technical indicators (matching TwelveData)
    SUPPORTED_INDICATORS = [
        # Momentum Indicators
        "rsi", "macd", "stoch", "cci", "adx",
        # Volatility Indicators
        "bbands", "atr",
        # Trend Indicators
        "ema", "sma", "ichimoku",
        # EasyInsight proprietary
        "strength",
    ]

    def __init__(self):
        self._api_url: str = settings.easyinsight_api_url
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cache_initialized: bool = False
        self._total_calls: int = 0
        self._cache_hits: int = 0

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

    async def _ensure_cache_connected(self):
        """Ensure cache service is connected."""
        if not self._cache_initialized:
            await cache_service.connect()
            self._cache_initialized = True

    def is_available(self) -> bool:
        """Check if EasyInsight service is available."""
        return bool(self._api_url)

    # ==================== OHLCV Data ====================

    async def get_time_series(
        self,
        symbol: str,
        interval: str = "1h",
        outputsize: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bypass_cache: bool = False,
    ) -> dict:
        """
        Get time series (OHLCV) data for a symbol.

        Uses the new TwelveData-compatible /time_series endpoint.

        Args:
            symbol: The symbol to get data for (e.g., 'BTCUSD', 'EURUSD')
            interval: Time interval (1min, 5min, 15min, 30min, 1h, 4h, 1day, 1week, 1month)
            outputsize: Number of data points (max 5000)
            start_date: Start date (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
            end_date: End date (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
            bypass_cache: If True, skip cache and fetch fresh data

        Returns:
            Dictionary with 'meta', 'values', and 'status' keys
        """
        await self._ensure_cache_connected()

        # Normalize symbol for cache key
        cache_symbol = symbol.upper().replace("/", "")

        # Build cache params
        cache_params = {"interval": interval, "outputsize": outputsize}
        if start_date:
            cache_params["start_date"] = start_date
        if end_date:
            cache_params["end_date"] = end_date

        # Check cache first (unless bypassed)
        if not bypass_cache and not start_date and not end_date:
            cached = await cache_service.get(
                CacheCategory.OHLCV, cache_symbol, interval, params=cache_params
            )
            if cached:
                self._cache_hits += 1
                if "meta" in cached:
                    cached["meta"]["from_cache"] = True
                logger.debug(f"EasyInsight cache hit for {symbol}/{interval}")
                return cached

        try:
            client = await self._get_client()
            self._total_calls += 1

            # Build request parameters
            params = {
                "interval": interval,
                "outputsize": outputsize,
            }
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            response = await client.get(
                f"{self._api_url}/time_series/{symbol}",
                params=params
            )

            if response.status_code != 200:
                error_msg = f"EasyInsight API error: {response.status_code}"
                logger.warning(error_msg)
                return {"meta": {"symbol": symbol}, "values": [], "status": "error", "error": error_msg}

            data = response.json()

            # Check for API error
            if data.get("status") == "error":
                return data

            # Convert string values to floats for consistency
            values = data.get("values", [])
            converted_values = []
            for row in values:
                converted_row = {
                    "datetime": row.get("datetime"),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)) if row.get("volume") else 0,
                }
                converted_values.append(converted_row)

            result = {
                "meta": {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "type": "Time Series",
                    "from_cache": False,
                    **(data.get("meta", {}))
                },
                "values": converted_values,
                "status": "ok"
            }

            # Cache the result (only if not using date filters)
            if not start_date and not end_date and converted_values:
                ttl = TIMEFRAME_TTL.get(interval, 300)
                await cache_service.set(
                    CacheCategory.OHLCV, result, cache_symbol, interval,
                    params=cache_params, ttl=ttl
                )
                logger.debug(f"Cached {len(converted_values)} OHLCV points for {symbol}/{interval}")

            logger.info(f"EasyInsight: Retrieved {len(converted_values)} data points for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to get time series from EasyInsight for {symbol}: {e}")
            return {"meta": {"symbol": symbol}, "values": [], "status": "error", "error": str(e)}

    # ==================== Technical Indicators ====================

    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "1h",
        outputsize: int = 100,
        **kwargs,
    ) -> dict:
        """
        Get technical indicator data for a symbol.

        Uses the new TwelveData-compatible indicator endpoints.

        Args:
            symbol: The symbol to analyze
            indicator: Indicator name (rsi, macd, bbands, stoch, adx, atr, etc.)
            interval: Time interval
            outputsize: Number of data points
            **kwargs: Additional indicator-specific parameters

        Returns:
            Dictionary with indicator values
        """
        await self._ensure_cache_connected()

        indicator_lower = indicator.lower()
        if indicator_lower not in self.SUPPORTED_INDICATORS:
            return {
                "error": f"Unsupported indicator: {indicator}. Supported: {self.SUPPORTED_INDICATORS}",
                "indicator": indicator,
                "symbol": symbol,
                "status": "error"
            }

        try:
            client = await self._get_client()
            self._total_calls += 1

            # Build request parameters
            params = {
                "interval": interval,
                "outputsize": outputsize,
                **kwargs
            }

            response = await client.get(
                f"{self._api_url}/{indicator_lower}/{symbol}",
                params=params
            )

            if response.status_code != 200:
                error_msg = f"EasyInsight API error: {response.status_code}"
                return {"indicator": indicator, "symbol": symbol, "status": "error", "error": error_msg}

            data = response.json()

            # Check for API error
            if data.get("status") == "error":
                return data

            # Standardize response format
            result = {
                "indicator": indicator.upper(),
                "symbol": symbol.upper(),
                "interval": interval,
                "values": data.get("values", []),
                "meta": data.get("meta", {}),
                "status": "ok"
            }

            logger.info(f"EasyInsight: Retrieved {indicator.upper()} for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to get {indicator} from EasyInsight for {symbol}: {e}")
            return {"indicator": indicator, "symbol": symbol, "status": "error", "error": str(e)}

    # ==================== Specific Indicator Methods ====================

    async def get_rsi(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get RSI (Relative Strength Index) indicator."""
        return await self.get_technical_indicator(
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
        """Get MACD (Moving Average Convergence Divergence) indicator."""
        return await self.get_technical_indicator(
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
        """Get Bollinger Bands indicator."""
        return await self.get_technical_indicator(
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
        """Get Stochastic Oscillator indicator."""
        return await self.get_technical_indicator(
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
        """Get ADX (Average Directional Index) indicator."""
        return await self.get_technical_indicator(
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
        """Get ATR (Average True Range) indicator."""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="atr",
            interval=interval,
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_ema(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 20,
        outputsize: int = 100,
    ) -> dict:
        """Get EMA (Exponential Moving Average) indicator."""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="ema",
            interval=interval,
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_sma(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 20,
        outputsize: int = 100,
    ) -> dict:
        """Get SMA (Simple Moving Average) indicator."""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="sma",
            interval=interval,
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_cci(
        self,
        symbol: str,
        interval: str = "1h",
        time_period: int = 20,
        outputsize: int = 100,
    ) -> dict:
        """Get CCI (Commodity Channel Index) indicator."""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="cci",
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
        """Get Ichimoku Cloud indicator."""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="ichimoku",
            interval=interval,
            outputsize=outputsize,
        )

    async def get_strength(
        self,
        symbol: str,
        interval: str = "1h",
        outputsize: int = 100,
    ) -> dict:
        """Get EasyInsight proprietary Multi-Timeframe Strength indicator."""
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator="strength",
            interval=interval,
            outputsize=outputsize,
        )

    # ==================== Multiple Indicators ====================

    async def get_multiple_indicators(
        self,
        symbol: str,
        indicators: list[str],
        interval: str = "1h",
        outputsize: int = 100,
    ) -> dict:
        """
        Get multiple technical indicators for a symbol in one request.

        Uses the TwelveData-compatible /indicators endpoint.

        Args:
            symbol: The symbol to analyze
            indicators: List of indicator names (e.g., ['rsi', 'macd', 'bbands'])
            interval: Time interval
            outputsize: Number of data points

        Returns:
            Dictionary with results for each indicator
        """
        await self._ensure_cache_connected()

        try:
            client = await self._get_client()
            self._total_calls += 1

            params = {
                "interval": interval,
                "indicators": ",".join(indicators),
                "outputsize": outputsize,
            }

            response = await client.get(
                f"{self._api_url}/indicators/{symbol}",
                params=params
            )

            if response.status_code != 200:
                # Fallback: fetch indicators individually
                logger.warning(f"Batch indicators failed, fetching individually")
                return await self._fetch_indicators_individually(
                    symbol, indicators, interval, outputsize
                )

            data = response.json()

            if data.get("status") == "error":
                # Fallback: fetch indicators individually
                return await self._fetch_indicators_individually(
                    symbol, indicators, interval, outputsize
                )

            result = {
                "symbol": symbol.upper(),
                "interval": interval,
                "indicators": data.get("indicators", {}),
                "errors": data.get("errors", []),
                "status": "ok"
            }

            logger.info(f"EasyInsight: Retrieved {len(result['indicators'])} indicators for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to get multiple indicators from EasyInsight for {symbol}: {e}")
            # Fallback: fetch indicators individually
            return await self._fetch_indicators_individually(
                symbol, indicators, interval, outputsize
            )

    async def _fetch_indicators_individually(
        self,
        symbol: str,
        indicators: list[str],
        interval: str,
        outputsize: int,
    ) -> dict:
        """Fallback: fetch indicators individually if batch endpoint fails."""
        result = {
            "symbol": symbol.upper(),
            "interval": interval,
            "indicators": {},
            "errors": [],
            "status": "ok"
        }

        for indicator in indicators:
            try:
                data = await self.get_technical_indicator(
                    symbol=symbol,
                    indicator=indicator,
                    interval=interval,
                    outputsize=outputsize,
                )
                if data.get("status") == "ok":
                    result["indicators"][indicator.upper()] = data.get("values", [])
                else:
                    result["errors"].append({indicator: data.get("error", "Unknown error")})
            except Exception as e:
                result["errors"].append({indicator: str(e)})

        return result

    # ==================== Quote / Price ====================

    async def get_quote(self, symbol: str) -> dict:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: The symbol to get quote for

        Returns:
            Dictionary with quote data
        """
        try:
            client = await self._get_client()
            self._total_calls += 1

            response = await client.get(f"{self._api_url}/quote/{symbol}")

            if response.status_code != 200:
                return {"symbol": symbol, "status": "error", "error": f"HTTP {response.status_code}"}

            data = response.json()
            logger.debug(f"EasyInsight: Retrieved quote for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to get quote from EasyInsight for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    async def get_price(self, symbol: str) -> dict:
        """
        Get current price for a symbol (lightweight endpoint).

        Args:
            symbol: The symbol to get price for

        Returns:
            Dictionary with current price
        """
        try:
            client = await self._get_client()
            self._total_calls += 1

            response = await client.get(f"{self._api_url}/price/{symbol}")

            if response.status_code != 200:
                return {"symbol": symbol, "status": "error", "error": f"HTTP {response.status_code}"}

            data = response.json()
            logger.debug(f"EasyInsight: Retrieved price for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to get price from EasyInsight for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    # ==================== Symbols ====================

    async def get_symbols(self, symbol_type: Optional[str] = None) -> dict:
        """
        Get list of available symbols.

        Args:
            symbol_type: Filter by type (forex, crypto, stock, index, commodity)

        Returns:
            Dictionary with symbol list
        """
        try:
            client = await self._get_client()
            self._total_calls += 1

            params = {}
            if symbol_type:
                params["type"] = symbol_type

            response = await client.get(f"{self._api_url}/symbols", params=params)

            if response.status_code != 200:
                return {"status": "error", "error": f"HTTP {response.status_code}"}

            data = response.json()
            logger.info(f"EasyInsight: Retrieved {data.get('count', len(data.get('data', [])))} symbols")
            return data

        except Exception as e:
            logger.error(f"Failed to get symbols from EasyInsight: {e}")
            return {"status": "error", "error": str(e)}

    # ==================== Status ====================

    async def get_status(self) -> dict:
        """
        Get EasyInsight API status.

        Returns:
            Dictionary with API status information
        """
        try:
            client = await self._get_client()

            response = await client.get(f"{self._api_url}/status", timeout=5.0)

            if response.status_code != 200:
                return {
                    "service": "EasyInsight",
                    "available": False,
                    "error": f"HTTP {response.status_code}"
                }

            data = response.json()
            return {
                "service": "EasyInsight",
                "available": True,
                "api_url": self._api_url,
                **data
            }

        except Exception as e:
            return {
                "service": "EasyInsight",
                "available": False,
                "api_url": self._api_url,
                "error": str(e)
            }

    async def health_check(self) -> bool:
        """Check if EasyInsight API is accessible."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._api_url}/status", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def get_service_stats(self) -> dict:
        """Get service statistics."""
        return {
            "service": "EasyInsight",
            "available": self.is_available(),
            "api_url": self._api_url,
            "session_stats": {
                "total_calls": self._total_calls,
                "cache_hits": self._cache_hits,
            },
            "supported_indicators": self.SUPPORTED_INDICATORS,
        }


# Global service instance
easyinsight_service = EasyInsightService()
