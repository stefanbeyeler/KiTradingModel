"""Twelve Data API Service - Access to real-time and historical market data.

Includes rate limiting to respect API limits (377 credits/min for Grow plan).
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

    def __init__(self, calls_per_minute: int = 377):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum API credits allowed per minute (default: 377 for Grow plan)
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
        # Rate limiter: 377 credits/minute for Grow plan, configurable via settings
        calls_per_minute = getattr(settings, 'twelvedata_rate_limit', 377)
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
                "timezone": "UTC",  # Always use UTC to avoid timezone confusion
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

    # Supported technical indicators
    SUPPORTED_INDICATORS = [
        # Overlap Studies (Moving Averages)
        "sma", "ema", "wma", "dema", "tema", "kama", "mama", "t3", "trima",
        "vwap",
        # Momentum Indicators
        "rsi", "macd", "stoch", "stochrsi", "willr", "cci", "cmo", "roc", "mom",
        "ppo", "apo", "aroon", "aroonosc", "bop", "mfi", "dx", "adx", "adxr",
        "plus_di", "minus_di", "plus_dm", "minus_dm", "crsi",
        # Volatility Indicators
        "bbands", "atr", "natr", "trange", "percent_b",
        # Volume Indicators
        "obv", "ad", "adosc",
        # Trend Indicators
        "supertrend", "ichimoku", "sar",
        # Price Transform
        "avgprice", "medprice", "typprice", "wclprice",
        # Pattern Recognition
        "pivot_points_hl",
        # Statistical Functions
        "linearregslope",
        # Cycle Indicators
        "ht_trendmode",
    ]

    async def get_technical_indicators(
        self,
        symbol: str,
        interval: str = "1day",
        indicator: str = "sma",
        outputsize: int = 100,
        **kwargs,
    ) -> dict:
        """
        Get technical indicator data for a symbol with rate limiting.

        Uses TwelveData REST API directly for indicator data.

        Args:
            symbol: The symbol to analyze
            interval: Time interval
            indicator: Indicator name (see SUPPORTED_INDICATORS for full list)
            outputsize: Number of data points
            **kwargs: Additional indicator-specific parameters (e.g., time_period=14)

        Returns:
            Dictionary with indicator values.
        """
        if not self._api_key:
            return {"error": "Twelve Data API key not configured"}

        indicator_lower = indicator.lower()
        if indicator_lower not in self.SUPPORTED_INDICATORS:
            return {"error": f"Unknown indicator: {indicator}. Supported: {self.SUPPORTED_INDICATORS}"}

        try:
            # Apply rate limiting before making the API call
            wait_time = await self._rate_limiter.acquire()
            self._total_calls += 1
            self._total_wait_time += wait_time

            if wait_time > 0:
                logger.debug(f"Rate limiter: waited {wait_time:.1f}s before calling {indicator} API for {symbol}")

            # Build API request
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self._api_key,
                **kwargs,
            }

            # Call TwelveData REST API directly
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"https://api.twelvedata.com/{indicator_lower}",
                    params=params
                )
                data = response.json()

            # Check for API errors
            if "status" in data and data["status"] == "error":
                error_msg = data.get("message", "Unknown API error")
                logger.error(f"TwelveData API error for {indicator}/{symbol}: {error_msg}")
                return {"indicator": indicator, "symbol": symbol, "error": error_msg}

            # Extract values from response
            values = data.get("values", data)
            if isinstance(values, dict) and "values" not in values:
                # Some endpoints return data directly
                values = [values] if not isinstance(values, list) else values

            logger.info(f"Retrieved {indicator.upper()} for {symbol} ({len(values) if isinstance(values, list) else 1} values)")
            return {
                "indicator": indicator.upper(),
                "symbol": symbol,
                "interval": interval,
                "values": values,
                "meta": data.get("meta", {}),
            }
        except Exception as e:
            logger.error(f"Failed to get {indicator} for {symbol}: {e}")
            return {"indicator": indicator, "symbol": symbol, "error": str(e)}

    # ==================== Specific Indicator Methods ====================

    async def get_rsi(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get RSI (Relative Strength Index) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="rsi",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_macd(
        self,
        symbol: str,
        interval: str = "1day",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        outputsize: int = 100,
    ) -> dict:
        """Get MACD (Moving Average Convergence Divergence) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="macd",
            outputsize=outputsize,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    async def get_bollinger_bands(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 20,
        sd: float = 2.0,
        ma_type: str = "SMA",
        outputsize: int = 100,
    ) -> dict:
        """Get Bollinger Bands indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="bbands",
            outputsize=outputsize,
            time_period=time_period,
            sd=sd,
            ma_type=ma_type,
        )

    async def get_ema(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 20,
        outputsize: int = 100,
    ) -> dict:
        """Get EMA (Exponential Moving Average) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="ema",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_sma(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 20,
        outputsize: int = 100,
    ) -> dict:
        """Get SMA (Simple Moving Average) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="sma",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_stochastic(
        self,
        symbol: str,
        interval: str = "1day",
        fast_k_period: int = 14,
        slow_k_period: int = 3,
        slow_d_period: int = 3,
        outputsize: int = 100,
    ) -> dict:
        """Get Stochastic Oscillator indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="stoch",
            outputsize=outputsize,
            fast_k_period=fast_k_period,
            slow_k_period=slow_k_period,
            slow_d_period=slow_d_period,
        )

    async def get_adx(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get ADX (Average Directional Index) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="adx",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_atr(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get ATR (Average True Range) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="atr",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_cci(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 20,
        outputsize: int = 100,
    ) -> dict:
        """Get CCI (Commodity Channel Index) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="cci",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_ichimoku(
        self,
        symbol: str,
        interval: str = "1day",
        conversion_line_period: int = 9,
        base_line_period: int = 26,
        leading_span_b_period: int = 52,
        lagging_span_period: int = 26,
        outputsize: int = 100,
    ) -> dict:
        """Get Ichimoku Cloud indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="ichimoku",
            outputsize=outputsize,
            conversion_line_period=conversion_line_period,
            base_line_period=base_line_period,
            leading_span_b_period=leading_span_b_period,
            lagging_span_period=lagging_span_period,
        )

    async def get_supertrend(
        self,
        symbol: str,
        interval: str = "1day",
        period: int = 10,
        multiplier: float = 3.0,
        outputsize: int = 100,
    ) -> dict:
        """Get Supertrend indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="supertrend",
            outputsize=outputsize,
            period=period,
            multiplier=multiplier,
        )

    async def get_williams_r(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get Williams %R indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="willr",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_mfi(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 14,
        outputsize: int = 100,
    ) -> dict:
        """Get MFI (Money Flow Index) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="mfi",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_obv(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 100,
    ) -> dict:
        """Get OBV (On-Balance Volume) indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="obv",
            outputsize=outputsize,
        )

    async def get_aroon(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 25,
        outputsize: int = 100,
    ) -> dict:
        """Get Aroon indicator (Aroon Up and Aroon Down)."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="aroon",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_pivot_points(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 1,
        outputsize: int = 100,
    ) -> dict:
        """Get Pivot Points High/Low indicator."""
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="pivot_points_hl",
            outputsize=outputsize,
            time_period=time_period,
        )

    async def get_vwap(
        self,
        symbol: str,
        interval: str = "1h",
        outputsize: int = 100,
    ) -> dict:
        """
        Get VWAP (Volume Weighted Average Price) indicator.

        VWAP is particularly useful for intraday trading as it shows
        the average price weighted by volume. Institutional traders
        often use VWAP as a benchmark.

        Note: VWAP resets daily, so it's most useful for intraday intervals.
        """
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="vwap",
            outputsize=outputsize,
        )

    async def get_connors_rsi(
        self,
        symbol: str,
        interval: str = "1day",
        rsi_period: int = 3,
        streak_rsi_period: int = 2,
        pct_rank_period: int = 100,
        outputsize: int = 100,
    ) -> dict:
        """
        Get Connors RSI indicator.

        Connors RSI combines three components:
        1. Short-term RSI (default: 3-period)
        2. Up/Down streak length RSI (default: 2-period)
        3. Percent rank of price change (default: 100-period)

        Better suited for mean-reversion strategies than standard RSI.
        """
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="crsi",
            outputsize=outputsize,
            rsi_period=rsi_period,
            streak_rsi_period=streak_rsi_period,
            pct_rank_period=pct_rank_period,
        )

    async def get_linear_regression_slope(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 14,
        series_type: str = "close",
        outputsize: int = 100,
    ) -> dict:
        """
        Get Linear Regression Slope indicator.

        Returns the slope of the linear regression line, which quantifies
        trend strength and direction as a numerical value:
        - Positive slope = uptrend
        - Negative slope = downtrend
        - Magnitude indicates trend strength

        Useful as a feature for ML models like NHITS.
        """
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="linearregslope",
            outputsize=outputsize,
            time_period=time_period,
            series_type=series_type,
        )

    async def get_hilbert_trendmode(
        self,
        symbol: str,
        interval: str = "1day",
        series_type: str = "close",
        outputsize: int = 100,
    ) -> dict:
        """
        Get Hilbert Transform - Trend vs Cycle Mode indicator.

        Returns a value indicating whether the market is in:
        - Trend mode (value = 1): Trending market, use trend-following strategies
        - Cycle mode (value = 0): Ranging market, use mean-reversion strategies

        Useful for adaptive strategy selection and as a regime filter.
        """
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="ht_trendmode",
            outputsize=outputsize,
            series_type=series_type,
        )

    async def get_percent_b(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 20,
        sd: float = 2.0,
        ma_type: str = "SMA",
        outputsize: int = 100,
    ) -> dict:
        """
        Get Percent B (%B) indicator.

        Shows where price is relative to Bollinger Bands as a normalized value:
        - %B > 1: Price is above upper band (overbought)
        - %B = 1: Price is at upper band
        - %B = 0.5: Price is at middle band (SMA)
        - %B = 0: Price is at lower band
        - %B < 0: Price is below lower band (oversold)

        More suitable for ML models than raw Bollinger Bands as it's normalized.
        """
        return await self.get_technical_indicators(
            symbol=symbol,
            interval=interval,
            indicator="percent_b",
            outputsize=outputsize,
            time_period=time_period,
            sd=sd,
            ma_type=ma_type,
        )

    # ==================== Batch Indicator Method ====================

    async def get_multiple_indicators(
        self,
        symbol: str,
        indicators: list[str],
        interval: str = "1day",
        outputsize: int = 100,
        **kwargs,
    ) -> dict:
        """
        Get multiple technical indicators for a symbol.

        Note: Each indicator requires a separate API call with rate limiting.

        Args:
            symbol: The symbol to analyze
            indicators: List of indicator names (e.g., ['rsi', 'macd', 'bbands'])
            interval: Time interval
            outputsize: Number of data points
            **kwargs: Additional parameters passed to all indicators

        Returns:
            Dictionary with results for each indicator
        """
        results = {
            "symbol": symbol,
            "interval": interval,
            "indicators": {},
            "errors": [],
        }

        for indicator in indicators:
            try:
                data = await self.get_technical_indicators(
                    symbol=symbol,
                    interval=interval,
                    indicator=indicator,
                    outputsize=outputsize,
                    **kwargs,
                )
                if "error" in data:
                    results["errors"].append({indicator: data["error"]})
                else:
                    results["indicators"][indicator.upper()] = data.get("values", [])
            except Exception as e:
                results["errors"].append({indicator: str(e)})

        logger.info(f"Retrieved {len(results['indicators'])} indicators for {symbol}")
        return results

    async def get_complete_analysis(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 100,
    ) -> dict:
        """
        Get a complete technical analysis with all major indicators.

        Includes: RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR, CCI, OBV

        Note: This makes 8 API calls with rate limiting, may take time.

        Args:
            symbol: The symbol to analyze
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
