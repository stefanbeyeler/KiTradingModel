"""Twelve Data API Service - Access to real-time and historical market data.

Includes rate limiting to respect API limits (8 calls/minute for free tier).
"""

import asyncio
import time
from datetime import datetime
from typing import Optional
from loguru import logger
import httpx

try:
    from twelvedata import TDClient
    TWELVEDATA_AVAILABLE = True
except ImportError:
    TWELVEDATA_AVAILABLE = False
    TDClient = None

from ..config import settings


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 8):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum API calls allowed per minute (default: 8 for free tier)
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute  # Minimum seconds between calls
        self._call_times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """
        Acquire permission to make an API call.

        Returns:
            Number of seconds waited before acquiring.
        """
        async with self._lock:
            now = time.time()

            # Remove calls older than 1 minute
            cutoff = now - 60.0
            self._call_times = [t for t in self._call_times if t > cutoff]

            wait_time = 0.0

            # If we've hit the limit, wait until the oldest call expires
            if len(self._call_times) >= self.calls_per_minute:
                oldest_call = self._call_times[0]
                wait_time = (oldest_call + 60.0) - now + 0.1  # Add 100ms buffer

                if wait_time > 0:
                    logger.info(
                        f"Rate limit reached ({len(self._call_times)}/{self.calls_per_minute} calls). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)

                    # Recalculate after waiting
                    now = time.time()
                    cutoff = now - 60.0
                    self._call_times = [t for t in self._call_times if t > cutoff]

            # Record this call
            self._call_times.append(now)
            return wait_time

    def get_status(self) -> dict:
        """Get current rate limiter status."""
        now = time.time()
        cutoff = now - 60.0
        recent_calls = [t for t in self._call_times if t > cutoff]

        return {
            "calls_per_minute_limit": self.calls_per_minute,
            "calls_in_last_minute": len(recent_calls),
            "calls_remaining": max(0, self.calls_per_minute - len(recent_calls)),
            "next_slot_in_seconds": (
                max(0, (recent_calls[0] + 60.0 - now)) if len(recent_calls) >= self.calls_per_minute else 0
            ),
        }


class TwelveDataService:
    """Service for accessing Twelve Data API for market data with rate limiting."""

    def __init__(self):
        self._api_key: str = settings.twelvedata_api_key
        self._client: Optional[TDClient] = None
        self._initialized: bool = False
        # Rate limiter: 8 calls/minute for free tier, configurable via settings
        calls_per_minute = getattr(settings, 'twelvedata_rate_limit', 8)
        self._rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)
        self._total_calls: int = 0
        self._total_wait_time: float = 0.0

    def _get_client(self) -> Optional[TDClient]:
        """Get or create Twelve Data client."""
        if not TWELVEDATA_AVAILABLE:
            logger.warning("twelvedata package not installed. Install with: pip install twelvedata")
            return None

        if self._client is None:
            self._client = TDClient(apikey=self._api_key)
            self._initialized = True
            logger.info("Twelve Data client initialized")

        return self._client

    def is_available(self) -> bool:
        """Check if Twelve Data service is available."""
        return TWELVEDATA_AVAILABLE and bool(self._api_key)

    async def get_stock_list(
        self,
        exchange: Optional[str] = None,
        country: Optional[str] = None,
        symbol_type: str = "Common Stock",
    ) -> list[dict]:
        """
        Get list of available stocks.

        Args:
            exchange: Filter by exchange (e.g., 'NYSE', 'NASDAQ')
            country: Filter by country (e.g., 'United States', 'Germany')
            symbol_type: Type of symbol (default: 'Common Stock')

        Returns:
            List of stock dictionaries with symbol, name, currency, exchange, etc.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            params = {"type": symbol_type}
            if exchange:
                params["exchange"] = exchange
            if country:
                params["country"] = country

            stocks = client.get_stocks_list(**params).as_json()
            logger.info(f"Retrieved {len(stocks)} stocks from Twelve Data")
            return stocks
        except Exception as e:
            logger.error(f"Failed to get stock list: {e}")
            return []

    async def get_forex_pairs(self) -> list[dict]:
        """
        Get list of available forex pairs.

        Returns:
            List of forex pair dictionaries.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            pairs = client.get_forex_pairs_list().as_json()
            logger.info(f"Retrieved {len(pairs)} forex pairs from Twelve Data")
            return pairs
        except Exception as e:
            logger.error(f"Failed to get forex pairs: {e}")
            return []

    async def get_cryptocurrencies(self) -> list[dict]:
        """
        Get list of available cryptocurrencies.

        Returns:
            List of cryptocurrency dictionaries.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            cryptos = client.get_cryptocurrencies_list().as_json()
            logger.info(f"Retrieved {len(cryptos)} cryptocurrencies from Twelve Data")
            return cryptos
        except Exception as e:
            logger.error(f"Failed to get cryptocurrencies: {e}")
            return []

    async def get_etf_list(self) -> list[dict]:
        """
        Get list of available ETFs.

        Returns:
            List of ETF dictionaries.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            etfs = client.get_etf_list().as_json()
            logger.info(f"Retrieved {len(etfs)} ETFs from Twelve Data")
            return etfs
        except Exception as e:
            logger.error(f"Failed to get ETF list: {e}")
            return []

    async def get_indices(self) -> list[dict]:
        """
        Get list of available indices.

        Returns:
            List of index dictionaries.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            indices = client.get_indices_list().as_json()
            logger.info(f"Retrieved {len(indices)} indices from Twelve Data")
            return indices
        except Exception as e:
            logger.error(f"Failed to get indices: {e}")
            return []

    async def get_exchanges(self, asset_type: str = "stock") -> list[dict]:
        """
        Get list of available exchanges.

        Args:
            asset_type: Type of asset ('stock', 'etf', 'index')

        Returns:
            List of exchange dictionaries.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            exchanges = client.get_exchanges_list(type=asset_type).as_json()
            logger.info(f"Retrieved {len(exchanges)} exchanges from Twelve Data")
            return exchanges
        except Exception as e:
            logger.error(f"Failed to get exchanges: {e}")
            return []

    async def get_time_series(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> dict:
        """
        Get time series (OHLCV) data for a symbol with rate limiting.

        Args:
            symbol: The symbol to get data for (e.g., 'AAPL', 'EUR/USD')
            interval: Time interval ('1min', '5min', '15min', '30min', '45min',
                      '1h', '2h', '4h', '1day', '1week', '1month')
            outputsize: Number of data points (max 5000)
            start_date: Start date (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
            end_date: End date (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
            exchange: Specific exchange for the symbol

        Returns:
            Dictionary with 'meta' and 'values' keys containing OHLCV data.
        """
        client = self._get_client()
        if not client:
            return {"meta": {}, "values": [], "error": "Twelve Data client not available"}

        try:
            # Apply rate limiting before making the API call
            wait_time = await self._rate_limiter.acquire()
            self._total_calls += 1
            self._total_wait_time += wait_time

            if wait_time > 0:
                logger.debug(f"Rate limiter: waited {wait_time:.1f}s before calling API for {symbol}")

            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
            }
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if exchange:
                params["exchange"] = exchange

            ts = client.time_series(**params)
            data = ts.as_json()

            logger.info(f"Retrieved {len(data)} data points for {symbol}")
            return {
                "meta": {
                    "symbol": symbol,
                    "interval": interval,
                    "exchange": exchange,
                    "type": "Time Series",
                },
                "values": data,
            }
        except Exception as e:
            logger.error(f"Failed to get time series for {symbol}: {e}")
            return {"meta": {"symbol": symbol}, "values": [], "error": str(e)}

    async def get_quote(self, symbol: str, exchange: Optional[str] = None) -> dict:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: The symbol to get quote for
            exchange: Specific exchange for the symbol

        Returns:
            Dictionary with quote data (price, volume, change, etc.)
        """
        client = self._get_client()
        if not client:
            return {"error": "Twelve Data client not available"}

        try:
            params = {"symbol": symbol}
            if exchange:
                params["exchange"] = exchange

            quote = client.quote(**params).as_json()
            logger.info(f"Retrieved quote for {symbol}")
            return quote
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    async def get_price(self, symbol: str, exchange: Optional[str] = None) -> dict:
        """
        Get current price for a symbol (lightweight endpoint).

        Args:
            symbol: The symbol to get price for
            exchange: Specific exchange for the symbol

        Returns:
            Dictionary with current price.
        """
        client = self._get_client()
        if not client:
            return {"error": "Twelve Data client not available"}

        try:
            params = {"symbol": symbol}
            if exchange:
                params["exchange"] = exchange

            price = client.price(**params).as_json()
            logger.info(f"Retrieved price for {symbol}: {price.get('price', 'N/A')}")
            return price
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    async def get_symbol_search(self, query: str, outputsize: int = 20) -> list[dict]:
        """
        Search for symbols by name or ticker.

        Args:
            query: Search query (e.g., 'Apple', 'AAPL')
            outputsize: Maximum number of results

        Returns:
            List of matching symbols with details.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            results = client.symbol_search(symbol=query, outputsize=outputsize).as_json()
            logger.info(f"Found {len(results)} symbols matching '{query}'")
            return results
        except Exception as e:
            logger.error(f"Failed to search symbols for '{query}': {e}")
            return []

    async def get_technical_indicators(
        self,
        symbol: str,
        interval: str = "1day",
        indicator: str = "sma",
        outputsize: int = 100,
        **kwargs,
    ) -> dict:
        """
        Get technical indicator data for a symbol.

        Args:
            symbol: The symbol to analyze
            interval: Time interval
            indicator: Indicator name (e.g., 'sma', 'ema', 'rsi', 'macd', 'bbands')
            outputsize: Number of data points
            **kwargs: Additional indicator-specific parameters (e.g., time_period=14)

        Returns:
            Dictionary with indicator values.
        """
        client = self._get_client()
        if not client:
            return {"error": "Twelve Data client not available"}

        try:
            # Map indicator names to client methods
            indicator_methods = {
                "sma": client.sma,
                "ema": client.ema,
                "rsi": client.rsi,
                "macd": client.macd,
                "bbands": client.bbands,
                "stoch": client.stoch,
                "adx": client.adx,
                "atr": client.atr,
                "cci": client.cci,
                "obv": client.obv,
            }

            method = indicator_methods.get(indicator.lower())
            if not method:
                return {"error": f"Unknown indicator: {indicator}"}

            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                **kwargs,
            }

            data = method(**params).as_json()
            logger.info(f"Retrieved {indicator.upper()} for {symbol}")
            return {
                "indicator": indicator.upper(),
                "symbol": symbol,
                "interval": interval,
                "values": data,
            }
        except Exception as e:
            logger.error(f"Failed to get {indicator} for {symbol}: {e}")
            return {"indicator": indicator, "symbol": symbol, "error": str(e)}

    async def get_earnings_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Get earnings calendar.

        Args:
            start_date: Start date (format: 'YYYY-MM-DD')
            end_date: End date (format: 'YYYY-MM-DD')

        Returns:
            List of upcoming earnings announcements.
        """
        client = self._get_client()
        if not client:
            return []

        try:
            params = {}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            earnings = client.get_earnings_calendar(**params).as_json()
            logger.info(f"Retrieved {len(earnings)} earnings events")
            return earnings
        except Exception as e:
            logger.error(f"Failed to get earnings calendar: {e}")
            return []

    async def get_api_usage(self) -> dict:
        """
        Get current API usage statistics.

        Returns:
            Dictionary with API usage info (credits used, remaining, etc.)
        """
        if not self._api_key:
            return {"error": "API key not configured"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.twelvedata.com/api_usage",
                    params={"apikey": self._api_key}
                )
                usage = response.json()
                logger.info(f"API Usage - Daily used: {usage.get('current_usage', 'N/A')}, Plan limit: {usage.get('plan_limit', 'N/A')}")
                return usage
        except Exception as e:
            logger.error(f"Failed to get API usage: {e}")
            return {"error": str(e)}

    def get_status(self) -> dict:
        """Get service status information including rate limiter stats."""
        rate_limiter_status = self._rate_limiter.get_status()
        return {
            "service": "Twelve Data",
            "available": self.is_available(),
            "package_installed": TWELVEDATA_AVAILABLE,
            "api_key_configured": bool(self._api_key),
            "client_initialized": self._initialized,
            "rate_limiter": rate_limiter_status,
            "session_stats": {
                "total_calls": self._total_calls,
                "total_wait_time_seconds": round(self._total_wait_time, 1),
            },
        }

    def get_rate_limiter_status(self) -> dict:
        """Get current rate limiter status."""
        return self._rate_limiter.get_status()


# Global service instance
twelvedata_service = TwelveDataService()
