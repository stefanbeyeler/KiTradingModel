"""Yahoo Finance service for fetching historical market data.

This service provides a fallback data source when TwelveData is unavailable
or doesn't have sufficient data. Yahoo Finance is free and has extensive
historical data for stocks, indices, forex, and crypto.

Symbol mapping:
- Forex: EURUSD -> EURUSD=X
- Crypto: BTCUSD -> BTC-USD
- Indices: GER40 -> ^GDAXI, US500 -> ^GSPC
- Commodities: XAUUSD -> GC=F (Gold futures)
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Yahoo Finance fallback will be unavailable.")


# Symbol mapping from internal format to Yahoo Finance format
SYMBOL_MAPPING = {
    # Major Forex pairs
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    # Cross pairs
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURCHF": "EURCHF=X",
    "AUDCAD": "AUDCAD=X",
    "AUDCHF": "AUDCHF=X",
    "AUDJPY": "AUDJPY=X",
    "AUDNZD": "AUDNZD=X",
    "CADCHF": "CADCHF=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "EURAUD": "EURAUD=X",
    "EURCAD": "EURCAD=X",
    "EURNZD": "EURNZD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPCAD": "GBPCAD=X",
    "GBPCHF": "GBPCHF=X",
    "GBPNZD": "GBPNZD=X",
    "NZDCAD": "NZDCAD=X",
    "NZDCHF": "NZDCHF=X",
    "NZDJPY": "NZDJPY=X",

    # Major Crypto
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "BNBUSD": "BNB-USD",
    "SOLUSD": "SOL-USD",
    "ADAUSD": "ADA-USD",
    "XRPUSD": "XRP-USD",
    "DOTUSD": "DOT-USD",
    "LTCUSD": "LTC-USD",
    "BCHUSD": "BCH-USD",
    "LNKUSD": "LINK-USD",
    "XLMUSD": "XLM-USD",
    "UNIUSD": "UNI-USD",
    "AVXUSD": "AVAX-USD",
    "MTCUSD": "MATIC-USD",
    "DOGUSD": "DOGE-USD",
    "XTZUSD": "XTZ-USD",
    "KSMUSD": "KSM-USD",

    # Major Indices
    "US500": "^GSPC",      # S&P 500
    "US30": "^DJI",        # Dow Jones
    "NAS100": "^IXIC",     # NASDAQ Composite (closest to NAS100)
    "GER40": "^GDAXI",     # DAX
    "UK100": "^FTSE",      # FTSE 100
    "FRA40": "^FCHI",      # CAC 40
    "JP225": "^N225",      # Nikkei 225
    "AUS200": "^AXJO",     # ASX 200
    "EURO50": "^STOXX50E", # Euro Stoxx 50

    # Commodities (Futures)
    "XAUUSD": "GC=F",      # Gold
    "XAGUSD": "SI=F",      # Silver
    "XTIUSD": "CL=F",      # WTI Crude Oil
    "XAUEUR": "GC=F",      # Gold (USD base, needs conversion)
    "XAUGBP": "GC=F",      # Gold (USD base, needs conversion)
    "XAUCHF": "GC=F",      # Gold (USD base, needs conversion)
    "XAUJPY": "GC=F",      # Gold (USD base, needs conversion)
    "XAUAUD": "GC=F",      # Gold (USD base, needs conversion)
    "XAGEUR": "SI=F",      # Silver (USD base)
    "XAGAUD": "SI=F",      # Silver (USD base)
}

# Interval mapping from internal format to yfinance format
INTERVAL_MAPPING = {
    "M15": "15m",
    "H1": "1h",
    "D1": "1d",
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "1h": "1h",
    "1day": "1d",
    "1week": "1wk",
    "1month": "1mo",
}


class YFinanceService:
    """Service for fetching data from Yahoo Finance."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yfinance")
        self._available = YFINANCE_AVAILABLE
        if self._available:
            logger.info("YFinance Service initialized")
        else:
            logger.warning("YFinance Service not available (yfinance not installed)")

    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        return self._available

    def _map_symbol(self, symbol: str) -> Optional[str]:
        """Map internal symbol to Yahoo Finance symbol."""
        # First check direct mapping
        if symbol in SYMBOL_MAPPING:
            return SYMBOL_MAPPING[symbol]

        # Try common patterns
        upper = symbol.upper()

        # Forex: Add =X suffix
        if len(upper) == 6 and upper.isalpha():
            return f"{upper}=X"

        # Crypto: Try with -USD suffix
        if upper.endswith("USD") and len(upper) > 3:
            crypto = upper[:-3]
            return f"{crypto}-USD"

        # Already in yfinance format
        if "=" in symbol or "-" in symbol or symbol.startswith("^"):
            return symbol

        return None

    def _map_interval(self, interval: str) -> str:
        """Map internal interval to yfinance interval."""
        return INTERVAL_MAPPING.get(interval.upper(), interval)

    def _fetch_data_sync(
        self,
        symbol: str,
        interval: str,
        period: str = None,
        start: datetime = None,
        end: datetime = None,
    ) -> List[Dict]:
        """Synchronous data fetch (runs in thread pool)."""
        if not self._available:
            return []

        yf_symbol = self._map_symbol(symbol)
        if not yf_symbol:
            logger.warning(f"No Yahoo Finance mapping for symbol: {symbol}")
            return []

        yf_interval = self._map_interval(interval)

        try:
            ticker = yf.Ticker(yf_symbol)

            # Determine period/date range
            if start and end:
                df = ticker.history(start=start, end=end, interval=yf_interval)
            elif period:
                df = ticker.history(period=period, interval=yf_interval)
            else:
                # Default: max available for daily, 60 days for hourly, 7 days for 15min
                if yf_interval == "1d":
                    df = ticker.history(period="1y", interval=yf_interval)
                elif yf_interval == "1h":
                    df = ticker.history(period="60d", interval=yf_interval)
                else:
                    df = ticker.history(period="7d", interval=yf_interval)

            if df.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol} ({yf_symbol})")
                return []

            # Convert to list of dicts
            result = []
            for idx, row in df.iterrows():
                # Ensure timezone-aware UTC timestamp
                ts = idx
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')

                result.append({
                    "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": int(ts.timestamp()),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row.get("Volume", 0)),
                })

            # Sort by datetime descending (newest first, like TwelveData)
            result.sort(key=lambda x: x["datetime"], reverse=True)

            logger.info(f"Yahoo Finance returned {len(result)} rows for {symbol} ({yf_symbol})")
            return result

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return []

    async def get_time_series(
        self,
        symbol: str,
        interval: str = "1d",
        outputsize: int = 100,
        start: datetime = None,
        end: datetime = None,
    ) -> Dict:
        """Get historical time series data.

        Args:
            symbol: Trading symbol (internal format)
            interval: Time interval (M15, H1, D1, or yfinance format)
            outputsize: Number of data points (approximate)
            start: Start date
            end: End date

        Returns:
            Dict with 'values' list or 'error' message
        """
        if not self._available:
            return {"error": "yfinance not installed"}

        # Calculate period based on interval and outputsize
        yf_interval = self._map_interval(interval)
        if not start:
            if yf_interval == "1d":
                period = f"{max(outputsize * 2, 365)}d"
            elif yf_interval == "1h":
                period = "60d"  # yfinance limit for hourly
            elif yf_interval == "15m":
                period = "7d"   # yfinance limit for 15min
            else:
                period = "1mo"
        else:
            period = None

        loop = asyncio.get_event_loop()
        try:
            values = await loop.run_in_executor(
                self._executor,
                self._fetch_data_sync,
                symbol,
                interval,
                period,
                start,
                end,
            )

            if not values:
                return {"error": f"No data available for {symbol}"}

            # Limit to requested size
            values = values[:outputsize]

            return {
                "symbol": symbol,
                "interval": interval,
                "values": values,
                "source": "yfinance",
            }

        except Exception as e:
            logger.error(f"Yahoo Finance async error: {e}")
            return {"error": str(e)}

    async def get_daily_data(
        self,
        symbol: str,
        days: int = 100,
    ) -> List[Dict]:
        """Get daily OHLCV data for training.

        Args:
            symbol: Trading symbol
            days: Number of days of data

        Returns:
            List of OHLCV dicts
        """
        result = await self.get_time_series(
            symbol=symbol,
            interval="1d",
            outputsize=days,
        )

        if "error" in result:
            logger.warning(f"Yahoo Finance: {result['error']}")
            return []

        return result.get("values", [])

    def get_supported_symbols(self) -> List[str]:
        """Get list of symbols with known Yahoo Finance mappings."""
        return list(SYMBOL_MAPPING.keys())

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get basic info about a symbol."""
        if not self._available:
            return None

        yf_symbol = self._map_symbol(symbol)
        if not yf_symbol:
            return None

        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "yf_symbol": yf_symbol,
                "name": info.get("longName") or info.get("shortName"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "type": info.get("quoteType"),
            }
        except Exception as e:
            logger.debug(f"Could not get info for {symbol}: {e}")
            return None


# Singleton instance
yfinance_service = YFinanceService()
