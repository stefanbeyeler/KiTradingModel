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

# Import zentrale Microservices-Konfiguration
from src.config.microservices import microservices_config


class DataServiceClient:
    """
    HTTP-Client für den Zugriff auf den Data Service.

    Alle externen Datenabfragen (Symbole, OHLCV, Indikatoren) werden
    über den Data Service geroutet.
    """

    def __init__(self):
        # Data Service URL - aus zentraler Konfiguration
        self._base_url = os.getenv("DATA_SERVICE_URL", microservices_config.data_service_url)
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
        # Map timeframe to TwelveData format (supports both standard and TwelveData formats)
        timeframe_map = {
            # Standard format (M1, H1, D1)
            "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1h", "H4": "4h",
            "D1": "1day", "W1": "1week", "MN": "1month",
            # TwelveData format (1min, 1h, 1day) - pass through
            "1MIN": "1min", "5MIN": "5min", "15MIN": "15min", "30MIN": "30min",
            "1H": "1h", "4H": "4h",
            "1DAY": "1day", "1WEEK": "1week", "1MONTH": "1month",
            # Lowercase variants (1d, 1h) - common mistake
            "1D": "1day", "1W": "1week", "1M": "1min",
        }
        td_interval = timeframe_map.get(timeframe.upper(), timeframe)

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

    # =========================================================================
    # HMM Service Integration
    # =========================================================================

    async def get_hmm_regime(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback: int = 500,
        include_history: bool = False
    ) -> Optional[dict]:
        """
        Get current market regime from HMM Service.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            timeframe: Timeframe (1h, 4h, 1d)
            lookback: Number of candles for analysis
            include_history: Include regime history

        Returns:
            Regime detection result or None on error
        """
        hmm_url = os.getenv("HMM_SERVICE_URL", microservices_config.hmm_service_url)

        try:
            client = await self._get_client()
            response = await client.get(
                f"{hmm_url}/api/v1/regime/detect/{symbol}",
                params={
                    "timeframe": timeframe,
                    "lookback": lookback,
                    "include_history": include_history
                }
            )

            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Fetched HMM regime for {symbol}: {data.get('regime', 'unknown')}")
                return data
            else:
                logger.warning(f"HMM regime request failed: {response.status_code}")
                return None

        except httpx.ConnectError as e:
            logger.debug(f"Cannot connect to HMM Service: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching HMM regime for {symbol}: {e}")
            return None

    # =========================================================================
    # NHITS Service Integration
    # =========================================================================

    async def get_nhits_forecast(
        self,
        symbol: str,
        timeframe: str = "H1",
        horizon: int = 24
    ) -> Optional[dict]:
        """
        Get price forecast from NHITS Service.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            timeframe: Timeframe (H1, D1)
            horizon: Forecast horizon in candles

        Returns:
            Forecast result or None on error
        """
        nhits_url = os.getenv("NHITS_SERVICE_URL", microservices_config.nhits_service_url)

        try:
            client = await self._get_client()
            response = await client.get(
                f"{nhits_url}/api/v1/forecast/{symbol}",
                params={
                    "timeframe": timeframe,
                    "horizon": horizon
                }
            )

            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Fetched NHITS forecast for {symbol}: horizon={horizon}")
                return data
            else:
                logger.warning(f"NHITS forecast request failed: {response.status_code}")
                return None

        except httpx.ConnectError as e:
            logger.debug(f"Cannot connect to NHITS Service: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching NHITS forecast for {symbol}: {e}")
            return None

    # =========================================================================
    # Candlestick Service Integration
    # =========================================================================

    async def get_candlestick_patterns(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback: int = 100
    ) -> Optional[dict]:
        """
        Get candlestick patterns from Candlestick Service.

        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            timeframe: Timeframe (1h, 4h, 1d)
            lookback: Number of candles to analyze

        Returns:
            Detected candlestick patterns or None on error
        """
        candlestick_url = os.getenv("CANDLESTICK_SERVICE_URL", microservices_config.candlestick_service_url)

        try:
            client = await self._get_client()
            response = await client.get(
                f"{candlestick_url}/api/v1/scan/{symbol}",
                params={
                    "timeframe": timeframe,
                    "lookback": lookback
                }
            )

            if response.status_code == 200:
                data = response.json()
                patterns = data.get("patterns", [])
                logger.debug(f"Fetched {len(patterns)} candlestick patterns for {symbol}")
                return data
            else:
                logger.warning(f"Candlestick patterns request failed: {response.status_code}")
                return None

        except httpx.ConnectError as e:
            logger.debug(f"Cannot connect to Candlestick Service: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching candlestick patterns for {symbol}: {e}")
            return None

    # =========================================================================
    # Combined Context for Enhanced Detection
    # =========================================================================

    async def get_ml_context(
        self,
        symbol: str,
        timeframe: str = "1h"
    ) -> dict:
        """
        Get combined ML context from all services for enhanced pattern detection.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis

        Returns:
            Combined context with regime, forecast, and candlestick data
        """
        import asyncio

        # Map timeframe to NHITS format (1h -> H1, 4h -> H4, 1d -> D1)
        nhits_timeframe_map = {
            "1h": "H1", "4h": "H4", "1d": "D1",
            "h1": "H1", "h4": "H4", "d1": "D1",
            "1H": "H1", "4H": "H4", "1D": "D1",
            "H1": "H1", "H4": "H4", "D1": "D1",
            "15m": "M15", "m15": "M15", "M15": "M15"
        }
        nhits_tf = nhits_timeframe_map.get(timeframe, "H1")

        # Fetch all ML data in parallel
        regime_task = self.get_hmm_regime(symbol, timeframe)
        forecast_task = self.get_nhits_forecast(symbol, nhits_tf)
        candlestick_task = self.get_candlestick_patterns(symbol, timeframe)

        regime, forecast, candlesticks = await asyncio.gather(
            regime_task, forecast_task, candlestick_task,
            return_exceptions=True
        )

        context = {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": None,
            "forecast": None,
            "candlesticks": None
        }

        # Process regime data
        if isinstance(regime, dict) and "error" not in regime:
            context["regime"] = {
                "current": regime.get("current_regime"),
                "confidence": regime.get("regime_probability"),
                "duration": regime.get("regime_duration"),
                "characteristics": regime.get("market_metrics", {})
            }

        # Process forecast data
        if isinstance(forecast, dict) and "error" not in forecast:
            # NHITS returns predicted_prices array
            predictions = forecast.get("predicted_prices", [])
            context["forecast"] = {
                "horizon_hours": forecast.get("horizon_hours"),
                "current_price": forecast.get("current_price"),
                "predicted_price_1h": forecast.get("predicted_price_1h"),
                "predicted_price_4h": forecast.get("predicted_price_4h"),
                "predicted_price_24h": forecast.get("predicted_price_24h"),
                "predicted_change_1h": forecast.get("predicted_change_percent_1h"),
                "predicted_change_24h": forecast.get("predicted_change_percent_24h"),
                "trend_up_probability": forecast.get("trend_up_probability"),
                "model_confidence": forecast.get("model_confidence"),
                "predictions_count": len(predictions)
            }

        # Process candlestick patterns
        if isinstance(candlesticks, dict) and "error" not in candlesticks:
            patterns = candlesticks.get("patterns", [])
            context["candlesticks"] = {
                "count": len(patterns),
                "patterns": patterns[:5]  # Top 5 patterns
            }

        return context


# Singleton instance
data_service_client = DataServiceClient()
