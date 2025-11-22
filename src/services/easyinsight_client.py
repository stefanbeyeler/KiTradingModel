"""Client for EasyInsight TimescaleDB API."""

import httpx
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..models.trading_data import TimeSeriesData


class EasyInsightClient:
    """Client for fetching time series data from EasyInsight TimescaleDB API."""

    def __init__(self):
        self.base_url = settings.easyinsight_api_url.rstrip("/")
        self.api_key = settings.easyinsight_api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Initialize the HTTP client."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0
        )
        logger.info(f"Connected to EasyInsight API at {self.base_url}")

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Disconnected from EasyInsight API")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_time_series(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> list[TimeSeriesData]:
        """
        Fetch time series data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL")
            start_date: Start date for data range
            end_date: End date for data range
            interval: Data interval (e.g., "1m", "1h", "1d")

        Returns:
            List of TimeSeriesData points
        """
        if not self._client:
            await self.connect()

        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=settings.default_lookback_days)

        params = {
            "symbol": symbol,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "interval": interval
        }

        try:
            response = await self._client.get("/api/timeseries", params=params)
            response.raise_for_status()
            data = response.json()

            time_series = []
            for item in data.get("data", []):
                ts = TimeSeriesData(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    symbol=symbol,
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item.get("volume", 0)),
                    additional_data=item.get("metadata")
                )
                time_series.append(ts)

            logger.info(f"Fetched {len(time_series)} data points for {symbol}")
            return time_series

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching data for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching time series for {symbol}: {e}")
            raise

    async def get_available_symbols(self) -> list[str]:
        """Get list of available trading symbols."""
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get("/api/symbols")
            response.raise_for_status()
            data = response.json()
            return data.get("symbols", [])
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            raise

    async def get_latest_price(self, symbol: str) -> Optional[TimeSeriesData]:
        """Get the latest price data for a symbol."""
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get(f"/api/latest/{symbol}")
            response.raise_for_status()
            item = response.json()

            return TimeSeriesData(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                symbol=symbol,
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item.get("volume", 0)),
                additional_data=item.get("metadata")
            )
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if the EasyInsight API is accessible."""
        if not self._client:
            await self.connect()

        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
