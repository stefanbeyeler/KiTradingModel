#!/usr/bin/env python3
"""
Initiales Laden von historischen Daten für alle Symbole.

Lädt Daten von TwelveData/EasyInsight und speichert sie in TimescaleDB.

Usage:
    python scripts/initial_data_load.py
    python scripts/initial_data_load.py --symbols BTCUSD,EURUSD --timeframes H1,D1

Requirements:
    - TimescaleDB connection with schema created
    - TwelveData API key (or EasyInsight as fallback)
    - asyncpg installed
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>LOADER</cyan> | <level>{message}</level>",
    level="INFO",
)


# Default symbols and timeframes
DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "XAUUSD", "US500"]
DEFAULT_TIMEFRAMES = ["M15", "H1", "H4", "D1"]

# Days to fetch per timeframe
DAYS_BACK = {
    "M1": 7,       # 1 week for M1 (high volume)
    "M5": 14,      # 2 weeks for M5
    "M15": 30,     # 1 month for M15
    "M30": 60,     # 2 months for M30
    "M45": 60,     # 2 months for M45
    "H1": 180,     # 6 months for H1
    "H2": 180,     # 6 months for H2
    "H4": 365,     # 1 year for H4
    "D1": 730,     # 2 years for D1
    "W1": 1095,    # 3 years for W1
    "MN": 1825,    # 5 years for MN
}


async def load_historical_data(
    symbols: list[str],
    timeframes: list[str],
    rate_limit_delay: float = 0.5,
):
    """
    Lädt historische Daten für alle konfigurierten Symbole.

    Args:
        symbols: List of trading symbols
        timeframes: List of timeframes
        rate_limit_delay: Delay between API calls in seconds
    """
    from src.services.data_gateway_service import data_gateway
    from src.services.timescaledb_service import timescaledb_service
    from src.services.data_repository import data_repository
    from src.config.timeframes import Timeframe, normalize_timeframe_safe, calculate_limit_for_days

    # Check if TimescaleDB is available
    if not timescaledb_service.is_available:
        logger.error("TimescaleDB not available. Check configuration and password.")
        return

    # Initialize
    await data_repository.initialize()
    await timescaledb_service.initialize()

    total_loaded = 0
    total_errors = 0

    logger.info(f"Loading data for {len(symbols)} symbols and {len(timeframes)} timeframes")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")

    for symbol in symbols:
        logger.info(f"Processing symbol: {symbol}")

        for tf_str in timeframes:
            tf = normalize_timeframe_safe(tf_str, Timeframe.H1)
            days = DAYS_BACK.get(tf.value, 30)
            limit = calculate_limit_for_days(tf, days)

            # Cap at 5000 (TwelveData limit)
            limit = min(limit, 5000)

            logger.info(f"  {symbol}/{tf.value}: Loading {days} days ({limit} candles)")

            try:
                # Force refresh to bypass cache and fetch from API
                data, source = await data_gateway.get_historical_data_with_fallback(
                    symbol=symbol,
                    timeframe=tf.value,
                    limit=limit,
                    force_refresh=True,
                )

                if data:
                    logger.info(f"    ✓ Loaded {len(data)} records from {source}")
                    total_loaded += len(data)
                else:
                    logger.warning(f"    ✗ No data returned")
                    total_errors += 1

            except Exception as e:
                logger.error(f"    ✗ Error: {e}")
                total_errors += 1

            # Rate limiting
            await asyncio.sleep(rate_limit_delay)

    logger.info("=" * 60)
    logger.info(f"Initial data load completed!")
    logger.info(f"Total records loaded: {total_loaded}")
    logger.info(f"Total errors: {total_errors}")

    # Close connections
    await data_repository.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load historical market data into TimescaleDB"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help=f"Comma-separated list of symbols (default: {','.join(DEFAULT_SYMBOLS)})",
        default=",".join(DEFAULT_SYMBOLS),
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        help=f"Comma-separated list of timeframes (default: {','.join(DEFAULT_TIMEFRAMES)})",
        default=",".join(DEFAULT_TIMEFRAMES),
    )
    parser.add_argument(
        "--delay",
        type=float,
        help="Delay between API calls in seconds (default: 0.5)",
        default=0.5,
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",")]

    logger.info("=" * 60)
    logger.info("Initial Data Load Tool for TimescaleDB")
    logger.info("=" * 60)

    try:
        await load_historical_data(
            symbols=symbols,
            timeframes=timeframes,
            rate_limit_delay=args.delay,
        )
    except KeyboardInterrupt:
        logger.info("Data load cancelled by user")
    except Exception as e:
        logger.error(f"Data load failed with exception: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
