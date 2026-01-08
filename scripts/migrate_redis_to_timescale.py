#!/usr/bin/env python3
"""
Migration von Redis-Cache-Daten zu TimescaleDB.

Liest alle gecachten OHLCV-Daten aus Redis und
speichert sie persistent in TimescaleDB.

Usage:
    python scripts/migrate_redis_to_timescale.py

Requirements:
    - Redis connection
    - TimescaleDB connection with schema created
    - asyncpg installed
"""

import asyncio
import json
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
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>MIGRATION</cyan> | <level>{message}</level>",
    level="INFO",
)


async def migrate_cached_data():
    """Migriert alle OHLCV-Daten aus Redis nach TimescaleDB."""

    # Import services
    from src.services.timescaledb_service import timescaledb_service
    from src.services.cache_service import cache_service

    # Check if TimescaleDB is available
    if not timescaledb_service.is_available:
        logger.error("TimescaleDB not available. Check configuration and password.")
        return

    # Initialize connections
    await cache_service.connect()
    await timescaledb_service.initialize()

    logger.info("Starting Redis to TimescaleDB migration...")

    # Check Redis connection
    if not cache_service._redis_available:
        logger.error("Redis not available")
        return

    # Pattern to find all OHLCV keys
    pattern = "trading:ohlcv:*"
    migrated_count = 0
    error_count = 0

    try:
        # Scan for all OHLCV keys
        cursor = 0
        all_keys = []
        while True:
            cursor, keys = await cache_service._redis.scan(cursor, match=pattern, count=1000)
            all_keys.extend(keys)
            if cursor == 0:
                break

        logger.info(f"Found {len(all_keys)} OHLCV cache entries")

        for key in all_keys:
            try:
                # Key format: trading:ohlcv:SYMBOL:TIMEFRAME:hash
                parts = key.split(":")
                if len(parts) >= 4:
                    symbol = parts[2]
                    timeframe = parts[3]

                    # Read data from Redis
                    data = await cache_service._redis.get(key)
                    if data:
                        records = json.loads(data)

                        if records and isinstance(records, list):
                            # Upsert to TimescaleDB
                            count = await timescaledb_service.upsert_ohlcv(
                                symbol=symbol,
                                timeframe=timeframe,
                                data=records,
                                source="redis_migration",
                            )

                            if count > 0:
                                migrated_count += count
                                logger.info(f"Migrated {count} records: {symbol}/{timeframe}")
                            else:
                                logger.warning(f"No records migrated for {symbol}/{timeframe}")

            except Exception as e:
                error_count += 1
                logger.error(f"Error migrating {key}: {e}")

        logger.info(f"Migration completed: {migrated_count} records migrated, {error_count} errors")

    except Exception as e:
        logger.error(f"Migration failed: {e}")

    finally:
        await cache_service.disconnect()
        await timescaledb_service.close()


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Redis to TimescaleDB Migration Tool")
    logger.info("=" * 60)

    try:
        await migrate_cached_data()
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
    except Exception as e:
        logger.error(f"Migration failed with exception: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
