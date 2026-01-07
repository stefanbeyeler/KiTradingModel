"""
Data Service Client - HTTP-Client für den Zugriff auf den Data Service.

Der TCN Service greift auf externe Daten AUSSCHLIESSLICH über den Data Service zu.
Dies entspricht der Architektur in CLAUDE.md:
- Kein direkter Zugriff auf EasyInsight, TwelveData oder andere externe APIs
- Data Service (Port 3001) ist das einzige Gateway für externe Daten
"""

import os
from typing import Optional
import httpx
from loguru import logger


class DataServiceClient:
    """
    HTTP-Client für den Zugriff auf den Data Service.

    Alle externen Datenabfragen (Symbole, OHLCV, Indikatoren) werden
    über den Data Service geroutet.
    """

    def __init__(self):
        # Data Service URL - im Docker-Netzwerk erreichbar
        self._base_url = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
        self._api_prefix = "/api/v1"
        self._http_client: Optional[httpx.AsyncClient] = None
        self._timeout = 30.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def get_available_symbols(self) -> list[dict]:
        """
        Get list of available trading symbols from Data Service.

        Returns:
            List of symbol dictionaries with 'symbol', 'category', etc.
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}{self._api_prefix}/easyinsight/symbols"
            )
            response.raise_for_status()

            data = response.json()

            # Handle response format: {total, categories, symbols: [...]}
            if isinstance(data, dict) and "symbols" in data:
                symbols = data["symbols"]
                logger.debug(f"Fetched {len(symbols)} symbols from Data Service")
                return symbols
            elif isinstance(data, list):
                logger.debug(f"Fetched {len(data)} symbols from Data Service")
                return data
            else:
                logger.warning(f"Unexpected symbols response format: {type(data)}")
                return []

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Data Service at {self._base_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching symbols from Data Service: {e}")
            return []

    async def get_symbol_names(self) -> list[str]:
        """
        Get list of symbol names only.

        Returns:
            List of symbol strings (e.g., ['BTCUSD', 'EURUSD', ...])
        """
        symbols = await self.get_available_symbols()
        return sorted([s.get('symbol') for s in symbols if s.get('symbol')])

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        limit: int = 500
    ) -> tuple[list[dict], str]:
        """
        Get historical OHLCV data from Data Service.

        Tries TwelveData first, then falls back to EasyInsight.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            timeframe: Timeframe (H1, D1, etc.)
            limit: Number of data points to fetch

        Returns:
            Tuple of (data list, source name)
        """
        # Map standard timeframe to TwelveData format
        timeframe_map = {
            "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
            "M45": "45min", "H1": "1h", "H2": "2h", "H4": "4h",
            "D1": "1day", "W1": "1week", "MN": "1month"
        }
        td_interval = timeframe_map.get(timeframe.upper(), timeframe.lower())

        # Try TwelveData first
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}{self._api_prefix}/twelvedata/time_series/{symbol}",
                params={"interval": td_interval, "outputsize": limit}
            )

            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, dict) and "values" in data:
                    values = data["values"]
                    if values:
                        logger.debug(f"Fetched {len(values)} candles for {symbol} from TwelveData")
                        return values, "twelvedata"

        except Exception as e:
            logger.debug(f"TwelveData request failed for {symbol}: {e}")

        # Fallback to EasyInsight
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}{self._api_prefix}/easyinsight/ohlcv/{symbol}",
                params={"timeframe": timeframe, "limit": limit}
            )

            if response.status_code == 200:
                data = response.json()
                if data:
                    logger.debug(f"Fetched {len(data)} candles for {symbol} from EasyInsight")
                    return data, "easyinsight"

        except Exception as e:
            logger.debug(f"EasyInsight request failed for {symbol}: {e}")

        logger.warning(f"No data available for {symbol} {timeframe}")
        return [], "none"

    async def health_check(self) -> bool:
        """Check if Data Service is available."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


# Singleton instance
data_service_client = DataServiceClient()
