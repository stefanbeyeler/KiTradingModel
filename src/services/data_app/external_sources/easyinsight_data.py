"""EasyInsight Data Source - Managed symbols and MT5 logs from EasyInsight API."""

from datetime import datetime
from typing import Optional
import httpx
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority

# Import settings for EasyInsight API URL
try:
    from src.config import settings
    from src.config.microservices import microservices_config
    EASYINSIGHT_API_URL = settings.easyinsight_api_url
except ImportError:
    from src.config.microservices import microservices_config
    EASYINSIGHT_API_URL = microservices_config.easyinsight_api_url


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

    def __init__(self, api_base_url: str = None):
        super().__init__()
        # Use EasyInsight API URL for logs, Data Service URL for managed symbols
        self._easyinsight_api_url = api_base_url or EASYINSIGHT_API_URL
        self._data_service_url = microservices_config.data_service_url
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
        """Fetch managed symbols from Data Service."""
        results = []

        try:
            if symbol:
                # Fetch specific symbol
                response = await client.get(
                    f"{self._data_service_url}/api/v1/managed-symbols/{symbol}"
                )
                if response.status_code == 200:
                    data = response.json()
                    results.append(self._create_symbol_result(data))
            else:
                # Fetch all symbols
                response = await client.get(
                    f"{self._data_service_url}/api/v1/managed-symbols"
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
        """Fetch symbol statistics from Data Service."""
        results = []

        try:
            response = await client.get(
                f"{self._data_service_url}/api/v1/managed-symbols/stats"
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
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> list[DataSourceResult]:
        """Fetch MT5 trading logs from EasyInsight API."""
        results = []

        try:
            # Fetch from EasyInsight API logs endpoint
            params = {"limit": limit}
            if symbol:
                params["symbol"] = symbol

            response = await client.get(
                f"{self._easyinsight_api_url}/logs",
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                logs = data.get("data", []) if isinstance(data, dict) else data

                if isinstance(logs, list) and logs:
                    # Group logs by symbol and indicator for better RAG context
                    symbol_indicator_groups = {}
                    for log_entry in logs:
                        log_symbol = log_entry.get("symbol", "Unknown")
                        indicator = log_entry.get("indicator", "info")
                        key = f"{log_symbol}:{indicator}"
                        if key not in symbol_indicator_groups:
                            symbol_indicator_groups[key] = []
                        symbol_indicator_groups[key].append(log_entry)

                    # Create a result for each symbol/indicator combination
                    for key, group_logs in symbol_indicator_groups.items():
                        log_symbol, indicator = key.split(":", 1)
                        result = self._create_mt5_log_group_result(
                            group_logs, log_symbol, indicator
                        )
                        if result:
                            results.append(result)

                    logger.info(f"Fetched {len(logs)} MT5 logs, grouped into {len(results)} RAG documents")

        except httpx.RequestError as e:
            logger.warning(f"MT5 logs not available: {e}")
        except Exception as e:
            logger.error(f"Error fetching MT5 logs: {e}")

        return results

    def _create_mt5_log_group_result(
        self,
        logs: list[dict],
        symbol: str,
        indicator: str
    ) -> Optional[DataSourceResult]:
        """Create a DataSourceResult from grouped MT5 log data."""
        if not logs:
            return None

        # Get the most recent log for timestamp
        most_recent = logs[0]
        timestamp = most_recent.get("timestamp") or most_recent.get("log_time", "")

        # Build content from all logs in the group
        content_parts = [
            f"MT5 Trading Signals - {symbol} ({indicator})",
            f"Signal Count: {len(logs)}",
            f"Latest Update: {timestamp}",
            "",
            "Recent Signals:"
        ]

        # Add up to 10 most recent log entries
        for log in logs[:10]:
            content = log.get("content", log.get("message", ""))
            log_time = log.get("timestamp") or log.get("log_time", "")
            if content:
                # Format timestamp for display
                time_str = ""
                if log_time:
                    try:
                        if isinstance(log_time, str):
                            # Extract just the time part
                            time_str = log_time.split("T")[1][:8] if "T" in log_time else log_time
                    except Exception:
                        time_str = str(log_time)
                content_parts.append(f"  [{time_str}] {content}")

        # Determine priority based on indicator type
        high_priority_indicators = ["FXL", "SeparationCheck", "ENTRY", "EXIT", "SIGNAL"]
        medium_priority_indicators = ["ATR", "RSI", "MACD", "BB", "EMA", "SMA"]

        if any(ind in indicator.upper() for ind in high_priority_indicators):
            priority = DataPriority.HIGH
        elif any(ind in indicator.upper() for ind in medium_priority_indicators):
            priority = DataPriority.MEDIUM
        else:
            priority = DataPriority.LOW

        return DataSourceResult(
            source_type=DataSourceType.EASYINSIGHT,
            content="\n".join(content_parts),
            symbol=symbol,
            priority=priority,
            metadata={
                "metric_type": "mt5_log",
                "indicator": indicator,
                "log_count": len(logs),
                "latest_timestamp": timestamp,
            },
            raw_data={"logs": logs[:10], "total_count": len(logs)}
        )

    def _create_mt5_log_result(
        self,
        log_data: dict,
        symbol: Optional[str] = None
    ) -> Optional[DataSourceResult]:
        """Create a DataSourceResult from a single MT5 log entry."""
        if not log_data:
            return None

        log_symbol = log_data.get("symbol", symbol)
        indicator = log_data.get("indicator", log_data.get("type", "info"))
        content = log_data.get("content", log_data.get("message", ""))
        timestamp = log_data.get("timestamp") or log_data.get("log_time", "")

        content_parts = [
            f"MT5 Signal - {log_symbol} ({indicator})",
            f"Content: {content}",
            f"Timestamp: {timestamp}"
        ]

        if log_data.get("source"):
            content_parts.append(f"Source: {log_data['source']}")

        # Determine priority based on indicator
        high_priority_indicators = ["FXL", "SeparationCheck", "ENTRY", "EXIT", "SIGNAL"]
        if any(ind in indicator.upper() for ind in high_priority_indicators):
            priority = DataPriority.HIGH
        else:
            priority = DataPriority.MEDIUM

        return DataSourceResult(
            source_type=DataSourceType.EASYINSIGHT,
            content="\n".join(content_parts),
            symbol=log_symbol,
            priority=priority,
            metadata={
                "metric_type": "mt5_log",
                "indicator": indicator,
            },
            raw_data=log_data
        )
