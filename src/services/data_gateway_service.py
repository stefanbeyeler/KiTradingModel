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


# Global singleton instance
data_gateway = DataGatewayService()
