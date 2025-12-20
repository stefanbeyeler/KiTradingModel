"""EasyInsight Data Source - Managed symbols and MT5 logs from EasyInsight API."""

from datetime import datetime
from typing import Optional
import httpx
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class EasyInsightDataSource(DataSourceBase):
    """
    Data source for EasyInsight managed symbols and MT5 logs.

    This source provides:
    - Managed trading symbols with their configurations
    - Symbol statistics and data availability
    - MT5 trading logs and history
    - NHITS model availability status
    """

    source_type = DataSourceType.EASYINSIGHT

    def __init__(self, api_base_url: str = "http://easyinsight-service:3003"):
        super().__init__()
        self._api_base_url = api_base_url
        self._cache_ttl = 300  # 5 minutes

    async def fetch(
        self,
        symbol: Optional[str] = None,
        include_symbols: bool = True,
        include_stats: bool = True,
        include_mt5_logs: bool = True,
        **kwargs
    ) -> list[DataSourceResult]:
        """
        Fetch data from EasyInsight API.

        Args:
            symbol: Optional specific symbol to fetch
            include_symbols: Include managed symbols data
            include_stats: Include statistics
            include_mt5_logs: Include MT5 trading logs

        Returns:
            List of DataSourceResult objects
        """
        cache_key = self._get_cache_key(
            symbol,
            include_symbols=include_symbols,
            include_stats=include_stats,
            include_mt5_logs=include_mt5_logs
        )

        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        results = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch managed symbols
            if include_symbols:
                symbol_results = await self._fetch_managed_symbols(client, symbol)
                results.extend(symbol_results)

            # Fetch statistics
            if include_stats:
                stats_results = await self._fetch_stats(client)
                results.extend(stats_results)

            # Fetch MT5 logs
            if include_mt5_logs:
                mt5_results = await self._fetch_mt5_logs(client, symbol)
                results.extend(mt5_results)

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(
        self,
        symbol: Optional[str] = None,
        **kwargs
    ) -> list[dict]:
        """Fetch data formatted for RAG ingestion."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_managed_symbols(
        self,
        client: httpx.AsyncClient,
        symbol: Optional[str] = None
    ) -> list[DataSourceResult]:
        """Fetch managed symbols from EasyInsight API."""
        results = []

        try:
            if symbol:
                # Fetch specific symbol
                response = await client.get(
                    f"{self._api_base_url}/api/v1/managed-symbols/{symbol}"
                )
                if response.status_code == 200:
                    data = response.json()
                    results.append(self._create_symbol_result(data))
            else:
                # Fetch all symbols
                response = await client.get(
                    f"{self._api_base_url}/api/v1/managed-symbols"
                )
                if response.status_code == 200:
                    symbols = response.json()
                    for sym_data in symbols:
                        results.append(self._create_symbol_result(sym_data))

        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch managed symbols: {e}")
        except Exception as e:
            logger.error(f"Error fetching managed symbols: {e}")

        return results

    def _create_symbol_result(self, sym_data: dict) -> DataSourceResult:
        """Create a DataSourceResult from symbol data."""
        symbol_name = sym_data.get("symbol", "Unknown")
        category = sym_data.get("category", "unknown")
        status = sym_data.get("status", "unknown")
        has_data = sym_data.get("has_timescaledb_data", False)
        has_model = sym_data.get("has_nhits_model", False)
        total_records = sym_data.get("total_records", 0)

        # Build content string
        content_parts = [
            f"Managed Symbol: {symbol_name}",
            f"Category: {category}",
            f"Status: {status}",
            f"TimescaleDB Data: {'Available' if has_data else 'Not available'}",
        ]

        if has_data:
            first_ts = sym_data.get("first_data_timestamp")
            last_ts = sym_data.get("last_data_timestamp")
            content_parts.append(f"Data Range: {first_ts} to {last_ts}")
            content_parts.append(f"Total Records: {total_records:,}")

        content_parts.append(f"NHITS Model: {'Trained' if has_model else 'Not trained'}")

        if sym_data.get("description"):
            content_parts.append(f"Description: {sym_data['description']}")

        if sym_data.get("tags"):
            content_parts.append(f"Tags: {', '.join(sym_data['tags'])}")

        # Determine priority based on data availability
        if has_model and has_data:
            priority = DataPriority.HIGH
        elif has_data:
            priority = DataPriority.MEDIUM
        else:
            priority = DataPriority.LOW

        return DataSourceResult(
            source_type=DataSourceType.EASYINSIGHT,
            content="\n".join(content_parts),
            symbol=symbol_name,
            priority=priority,
            metadata={
                "metric_type": "managed_symbol",
                "category": category,
                "status": status,
                "has_timescaledb_data": has_data,
                "has_nhits_model": has_model,
                "total_records": total_records,
                "is_favorite": sym_data.get("is_favorite", False),
            },
            raw_data=sym_data
        )

    async def _fetch_stats(self, client: httpx.AsyncClient) -> list[DataSourceResult]:
        """Fetch symbol statistics from EasyInsight API."""
        results = []

        try:
            response = await client.get(
                f"{self._api_base_url}/api/v1/managed-symbols/stats"
            )
            if response.status_code == 200:
                stats = response.json()

                content = f"""EasyInsight Symbol Statistics:
Total Symbols: {stats.get('total_symbols', 0)}
Active Symbols: {stats.get('active_symbols', 0)}
Inactive Symbols: {stats.get('inactive_symbols', 0)}
Suspended Symbols: {stats.get('suspended_symbols', 0)}
With TimescaleDB Data: {stats.get('with_timescaledb_data', 0)}
With NHITS Model: {stats.get('with_nhits_model', 0)}
Favorites: {stats.get('favorites_count', 0)}

By Category:"""

                by_category = stats.get("by_category", {})
                for cat, count in by_category.items():
                    content += f"\n  - {cat}: {count}"

                results.append(DataSourceResult(
                    source_type=DataSourceType.EASYINSIGHT,
                    content=content,
                    priority=DataPriority.MEDIUM,
                    metadata={
                        "metric_type": "symbol_stats",
                        "total_symbols": stats.get("total_symbols", 0),
                        "with_data": stats.get("with_timescaledb_data", 0),
                        "with_model": stats.get("with_nhits_model", 0),
                    },
                    raw_data=stats
                ))

        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch stats: {e}")
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")

        return results

    async def _fetch_mt5_logs(
        self,
        client: httpx.AsyncClient,
        symbol: Optional[str] = None
    ) -> list[DataSourceResult]:
        """Fetch MT5 trading logs from EasyInsight API."""
        results = []

        try:
            # Try to fetch MT5 logs endpoint
            params = {}
            if symbol:
                params["symbol"] = symbol

            response = await client.get(
                f"{self._api_base_url}/api/v1/mt5-logs",
                params=params
            )

            if response.status_code == 200:
                logs = response.json()

                if isinstance(logs, list) and logs:
                    for log_entry in logs[:50]:  # Limit to 50 most recent
                        result = self._create_mt5_log_result(log_entry, symbol)
                        if result:
                            results.append(result)
                elif isinstance(logs, dict):
                    # Single log or summary
                    result = self._create_mt5_log_result(logs, symbol)
                    if result:
                        results.append(result)

        except httpx.RequestError as e:
            logger.debug(f"MT5 logs not available: {e}")
        except Exception as e:
            logger.debug(f"Error fetching MT5 logs: {e}")

        return results

    def _create_mt5_log_result(
        self,
        log_data: dict,
        symbol: Optional[str] = None
    ) -> Optional[DataSourceResult]:
        """Create a DataSourceResult from MT5 log data."""
        if not log_data:
            return None

        log_type = log_data.get("type", "trade")
        log_symbol = log_data.get("symbol", symbol)

        content_parts = [f"MT5 Log Entry ({log_type}):"]

        if log_symbol:
            content_parts.append(f"Symbol: {log_symbol}")

        if "action" in log_data:
            content_parts.append(f"Action: {log_data['action']}")

        if "price" in log_data:
            content_parts.append(f"Price: {log_data['price']}")

        if "volume" in log_data:
            content_parts.append(f"Volume: {log_data['volume']}")

        if "profit" in log_data:
            content_parts.append(f"Profit: {log_data['profit']}")

        if "comment" in log_data:
            content_parts.append(f"Comment: {log_data['comment']}")

        if "timestamp" in log_data:
            content_parts.append(f"Timestamp: {log_data['timestamp']}")

        # Determine priority based on log type
        if log_type in ["error", "warning"]:
            priority = DataPriority.HIGH
        elif log_type in ["trade", "order"]:
            priority = DataPriority.MEDIUM
        else:
            priority = DataPriority.LOW

        return DataSourceResult(
            source_type=DataSourceType.EASYINSIGHT,
            content="\n".join(content_parts),
            symbol=log_symbol,
            priority=priority,
            metadata={
                "metric_type": "mt5_log",
                "log_type": log_type,
            },
            raw_data=log_data
        )
