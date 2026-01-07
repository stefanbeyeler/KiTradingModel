"""Data Service Client - HTTP client for accessing external data sources via Data Service.

This client follows the architecture rule that all external data access goes through
the Data Service (Port 3001). The RAG Service uses this client instead of directly
accessing external data sources.

Note: Candlestick Pattern endpoints are routed to the Candlestick Service (Port 3006)
as these have been migrated from the Data Service.
"""

import httpx
from typing import Optional
from loguru import logger

from ..config import settings


class DataServiceClient:
    """HTTP client for Data Service external sources API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        candlestick_url: Optional[str] = None,
        tcn_url: Optional[str] = None,
        hmm_url: Optional[str] = None,
        nhits_url: Optional[str] = None
    ):
        """Initialize the client.

        Args:
            base_url: Data Service URL, defaults to settings.data_service_url
            candlestick_url: Candlestick Service URL, defaults to settings.candlestick_service_url
            tcn_url: TCN Service URL, defaults to trading-tcn:3003
            hmm_url: HMM Service URL, defaults to trading-hmm:3004
            nhits_url: NHITS Service URL, defaults to trading-nhits:3002
        """
        self._base_url = base_url or getattr(settings, 'data_service_url', 'http://localhost:3001')
        self._candlestick_url = candlestick_url or getattr(settings, 'candlestick_service_url', 'http://trading-candlestick:3006')
        self._tcn_url = tcn_url or getattr(settings, 'tcn_service_url', 'http://trading-tcn:3003')
        self._hmm_url = hmm_url or getattr(settings, 'hmm_service_url', 'http://trading-hmm:3004')
        self._nhits_url = nhits_url or getattr(settings, 'nhits_service_url', 'http://trading-nhits:3002')
        self._timeout = 30.0
        logger.info(f"DataServiceClient initialized with base URL: {self._base_url}")
        logger.info(f"DataServiceClient using Candlestick Service: {self._candlestick_url}")
        logger.info(f"DataServiceClient using TCN Service: {self._tcn_url}")
        logger.info(f"DataServiceClient using HMM Service: {self._hmm_url}")
        logger.info(f"DataServiceClient using NHITS Service: {self._nhits_url}")

    async def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to Data Service."""
        url = f"{self._base_url}/api/v1{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Data Service: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"Error calling Data Service: {e}")
            return {"error": str(e)}

    async def _get_candlestick(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to Candlestick Service."""
        url = f"{self._candlestick_url}/api/v1{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Candlestick Service: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"Error calling Candlestick Service: {e}")
            return {"error": str(e)}

    async def _get_tcn(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to TCN Service."""
        url = f"{self._tcn_url}/api/v1{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from TCN Service: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"Error calling TCN Service: {e}")
            return {"error": str(e)}

    async def _get_hmm(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to HMM Service."""
        url = f"{self._hmm_url}/api/v1{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from HMM Service: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"Error calling HMM Service: {e}")
            return {"error": str(e)}

    async def _get_nhits(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to NHITS Service."""
        url = f"{self._nhits_url}/api/v1{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for forecasts
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from NHITS Service: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"Error calling NHITS Service: {e}")
            return {"error": str(e)}

    async def _post(self, endpoint: str, json: Optional[dict] = None) -> dict:
        """Make POST request to Data Service."""
        url = f"{self._base_url}/api/v1{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=json)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Data Service: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"Error calling Data Service: {e}")
            return {"error": str(e)}

    async def get_available_sources(self) -> dict:
        """Get list of all available external data sources."""
        return await self._get("/external-sources")

    async def get_economic_calendar(
        self,
        symbol: Optional[str] = None,
        days_ahead: int = 7,
        days_back: int = 1
    ) -> dict:
        """Get economic calendar events."""
        params = {"days_ahead": days_ahead, "days_back": days_back}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/economic-calendar", params)

    async def get_sentiment(
        self,
        symbol: Optional[str] = None,
        include_fear_greed: bool = True,
        include_social: bool = True,
        include_options: bool = True,
        include_volatility: bool = True
    ) -> dict:
        """Get market sentiment data."""
        params = {
            "include_fear_greed": include_fear_greed,
            "include_social": include_social,
            "include_options": include_options,
            "include_volatility": include_volatility,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/sentiment", params)

    async def get_onchain(
        self,
        symbol: str,
        include_whale_alerts: bool = True,
        include_exchange_flows: bool = True,
        include_mining: bool = True,
        include_defi: bool = True
    ) -> dict:
        """Get on-chain data for a cryptocurrency."""
        params = {
            "include_whale_alerts": include_whale_alerts,
            "include_exchange_flows": include_exchange_flows,
            "include_mining": include_mining,
            "include_defi": include_defi,
        }
        return await self._get(f"/external-sources/onchain/{symbol}", params)

    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 50,
        include_liquidations: bool = True,
        include_cvd: bool = True
    ) -> dict:
        """Get orderbook and liquidity data."""
        params = {
            "depth": depth,
            "include_liquidations": include_liquidations,
            "include_cvd": include_cvd,
        }
        return await self._get(f"/external-sources/orderbook/{symbol}", params)

    async def get_macro(
        self,
        symbol: Optional[str] = None,
        include_dxy: bool = True,
        include_bonds: bool = True,
        include_correlations: bool = True,
        include_sectors: bool = True
    ) -> dict:
        """Get macro economic and correlation data."""
        params = {
            "include_dxy": include_dxy,
            "include_bonds": include_bonds,
            "include_correlations": include_correlations,
            "include_sectors": include_sectors,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/macro", params)

    async def get_historical_patterns(
        self,
        symbol: Optional[str] = None,
        include_seasonality: bool = True,
        include_drawdowns: bool = True,
        include_events: bool = True,
        include_comparable: bool = True
    ) -> dict:
        """Get historical pattern analysis."""
        params = {
            "include_seasonality": include_seasonality,
            "include_drawdowns": include_drawdowns,
            "include_events": include_events,
            "include_comparable": include_comparable,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/historical-patterns", params)

    async def get_technical_levels(
        self,
        symbol: str,
        include_sr: bool = True,
        include_fib: bool = True,
        include_pivots: bool = True,
        include_vwap: bool = True,
        include_ma: bool = True
    ) -> dict:
        """Get technical price levels."""
        params = {
            "include_sr": include_sr,
            "include_fib": include_fib,
            "include_pivots": include_pivots,
            "include_vwap": include_vwap,
            "include_ma": include_ma,
        }
        return await self._get(f"/external-sources/technical-levels/{symbol}", params)

    async def get_regulatory(
        self,
        symbol: Optional[str] = None,
        include_sec: bool = True,
        include_etf: bool = True,
        include_global: bool = True,
        include_enforcement: bool = True
    ) -> dict:
        """Get regulatory updates."""
        params = {
            "include_sec": include_sec,
            "include_etf": include_etf,
            "include_global": include_global,
            "include_enforcement": include_enforcement,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/regulatory", params)

    async def get_easyinsight(
        self,
        symbol: Optional[str] = None,
        include_symbols: bool = True,
        include_stats: bool = True,
        include_mt5_logs: bool = True
    ) -> dict:
        """Get EasyInsight managed data."""
        params = {
            "include_symbols": include_symbols,
            "include_stats": include_stats,
            "include_mt5_logs": include_mt5_logs,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/easyinsight", params)

    async def get_correlations(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "30d",
        include_matrix: bool = True,
        include_regime: bool = True
    ) -> dict:
        """Get asset correlation data."""
        params = {
            "timeframe": timeframe,
            "include_matrix": include_matrix,
            "include_regime": include_regime,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/correlations", params)

    async def get_volatility_regime(
        self,
        symbol: Optional[str] = None,
        include_vix: bool = True,
        include_atr: bool = True,
        include_bollinger: bool = True,
        include_regime: bool = True
    ) -> dict:
        """Get volatility regime data."""
        params = {
            "include_vix": include_vix,
            "include_atr": include_atr,
            "include_bollinger": include_bollinger,
            "include_regime": include_regime,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/volatility-regime", params)

    async def get_institutional_flow(
        self,
        symbol: Optional[str] = None,
        include_cot: bool = True,
        include_etf: bool = True,
        include_whale: bool = True,
        include_13f: bool = False
    ) -> dict:
        """Get institutional flow data (COT, ETF flows, whale tracking)."""
        params = {
            "include_cot": include_cot,
            "include_etf": include_etf,
            "include_whale": include_whale,
            "include_13f": include_13f,
        }
        if symbol:
            params["symbol"] = symbol
        return await self._get("/external-sources/institutional-flow", params)

    async def fetch_all(
        self,
        symbol: Optional[str] = None,
        source_types: Optional[list[str]] = None,
        min_priority: str = "low"
    ) -> dict:
        """Fetch data from all or selected external sources."""
        # Build URL with query parameters
        endpoint = "/external-sources/fetch-all"
        params = []
        if symbol:
            params.append(f"symbol={symbol}")
        if min_priority:
            params.append(f"min_priority={min_priority}")
        if params:
            endpoint += "?" + "&".join(params)

        # Body is the source_types list (or null for all sources)
        return await self._post(endpoint, source_types)

    async def fetch_trading_context(
        self,
        symbol: str,
        include_types: Optional[list[str]] = None
    ) -> dict:
        """Fetch comprehensive trading context for a symbol."""
        # Body is the include_types list (or null for all)
        return await self._post(f"/external-sources/trading-context/{symbol}", include_types)

    async def get_managed_symbols(self, active_only: bool = True) -> list[str]:
        """Get list of managed symbols from Data Service.

        Args:
            active_only: If True, only return active symbols

        Returns:
            List of symbol names (e.g., ["BTCUSD", "ETHUSD", ...])
        """
        try:
            data = await self._get("/managed-symbols")
            if isinstance(data, list):
                if active_only:
                    return [s["symbol"] for s in data if s.get("status") == "active"]
                return [s["symbol"] for s in data]
            return []
        except Exception as e:
            logger.error(f"Error fetching managed symbols: {e}")
            return []

    # -------------------------------------------------------------------------
    # Candlestick Pattern Methods (routed to Candlestick Service on Port 3006)
    # -------------------------------------------------------------------------

    async def get_candlestick_patterns(
        self,
        symbol: str,
        timeframes: Optional[list[str]] = None,
        lookback_candles: int = 20,
        min_confidence: float = 0.5
    ) -> dict:
        """Get detected candlestick patterns for a symbol.

        Note: This endpoint is served by the Candlestick Service (Port 3006).

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            timeframes: List of timeframes to scan (e.g., ["M5", "H1", "D1"])
            lookback_candles: Number of candles to analyze
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            Dict with detected patterns per timeframe
        """
        params = {
            "lookback_candles": lookback_candles,
            "min_confidence": min_confidence,
        }
        if timeframes:
            params["timeframes"] = ",".join(timeframes)
        return await self._get_candlestick(f"/scan/{symbol}", params)

    async def get_candlestick_pattern_types(self) -> dict:
        """Get all available candlestick pattern types and their descriptions.

        Note: This endpoint is served by the Candlestick Service (Port 3006).
        Uses /examples/list which returns pattern metadata.
        """
        return await self._get_candlestick("/examples/list")

    async def get_candlestick_pattern_summary(self, symbol: str) -> dict:
        """Get a simplified candlestick pattern summary for a symbol.

        Note: This endpoint is served by the Candlestick Service (Port 3006).
        """
        return await self._get_candlestick(f"/history/statistics", {"symbol": symbol})

    async def get_candlestick_pattern_history(
        self,
        symbol: Optional[str] = None,
        pattern_type: Optional[str] = None,
        direction: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> dict:
        """Query historical candlestick pattern detections.

        Note: This endpoint is served by the Candlestick Service (Port 3006).

        Args:
            symbol: Filter by symbol
            pattern_type: Filter by pattern type
            direction: Filter by direction (bullish/bearish)
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results

        Returns:
            Dict with pattern history
        """
        params = {
            "min_confidence": min_confidence,
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol
        if pattern_type:
            params["pattern_type"] = pattern_type
        if direction:
            params["direction"] = direction
        return await self._get_candlestick("/history", params)

    # -------------------------------------------------------------------------
    # TCN Pattern Detection Methods (Port 3003)
    # -------------------------------------------------------------------------

    async def get_tcn_patterns(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback: int = 200,
        threshold: float = 0.5
    ) -> dict:
        """Detect chart patterns using TCN deep learning model.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            lookback: Number of candles for analysis
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            Dict with detected patterns
        """
        params = {
            "timeframe": timeframe,
            "lookback": lookback,
            "threshold": threshold,
        }
        return await self._get_tcn(f"/detect/{symbol}", params)

    async def get_tcn_supported_patterns(self) -> dict:
        """Get list of supported TCN pattern types."""
        return await self._get_tcn("/detect/patterns")

    # -------------------------------------------------------------------------
    # HMM Regime Detection Methods (Port 3004)
    # -------------------------------------------------------------------------

    async def get_hmm_regime(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback: int = 500,
        include_history: bool = False
    ) -> dict:
        """Detect current market regime using HMM model.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            timeframe: Timeframe (1h, 4h, 1d)
            lookback: Number of candles for analysis
            include_history: Include historical regime changes

        Returns:
            Dict with regime classification (bull_trend, bear_trend, sideways, high_volatility)
        """
        params = {
            "timeframe": timeframe,
            "lookback": lookback,
            "include_history": include_history,
        }
        return await self._get_hmm(f"/regime/detect/{symbol}", params)

    async def get_hmm_regime_types(self) -> dict:
        """Get list of supported regime types with descriptions."""
        return await self._get_hmm("/regime/regimes")

    # -------------------------------------------------------------------------
    # NHITS Forecast Methods (Port 3002)
    # -------------------------------------------------------------------------

    async def get_nhits_forecast(
        self,
        symbol: str,
        timeframe: str = "H1",
        horizon: int = 24
    ) -> dict:
        """Generate NHITS price forecast for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            timeframe: Timeframe - M15 (2h), H1 (24h), D1 (7d)
            horizon: Forecast horizon (overridden by timeframe config)

        Returns:
            Dict with predicted prices and confidence intervals
        """
        params = {
            "timeframe": timeframe,
            "horizon": horizon,
        }
        return await self._get_nhits(f"/forecast/{symbol}", params)

    async def get_nhits_models(self) -> dict:
        """Get list of available NHITS models."""
        return await self._get_nhits("/forecast/models")

    async def get_nhits_status(self) -> dict:
        """Get NHITS service status."""
        return await self._get_nhits("/forecast/status")


# Singleton instance
_data_service_client: Optional[DataServiceClient] = None


def get_data_service_client() -> DataServiceClient:
    """Get or create the singleton DataServiceClient instance."""
    global _data_service_client
    if _data_service_client is None:
        _data_service_client = DataServiceClient()
    return _data_service_client
