"""Database API Routes for TimescaleDB access.

This module provides direct access to TimescaleDB data:
- OHLCV data queries
- Indicator data queries
- Freshness tracking
- Database statistics
- Data synchronization
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from loguru import logger

from ...data_repository import data_repository
from ...timescaledb_service import timescaledb_service
from ....config.timeframes import normalize_timeframe_safe, Timeframe
from ....config.database import SUPPORTED_TIMEFRAMES


router = APIRouter(prefix="/db", tags=["5. Database"])


# ==================== Symbol Categorization ====================

# Known crypto base symbols
CRYPTO_BASES = {
    # Major coins
    'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK', 'LNK', 'BNB', 'SOL',
    'DOGE', 'DOG', 'AVAX', 'MATIC', 'MTC', 'POL', 'UNI', 'ATOM', 'XLM', 'ALGO', 'FIL',
    'KSM',  # Kusama
    'TRX', 'ETC', 'XMR', 'AAVE', 'MKR', 'COMP', 'SNX', 'YFI', 'SUSHI', 'CRV',
    # DeFi & Utility
    'BAL', 'REN', 'KNC', 'ZRX', 'BAT', 'ENJ', 'MANA', 'SAND', 'AXS', 'CHZ',
    'FLOW', 'NEAR', 'FTM', 'ONE', 'VET', 'THETA', 'EGLD', 'ICP', 'HNT', 'XTZ',
    'EOS', 'NEO', 'WAVES', 'DASH', 'ZEC', 'QTUM', 'ICX', 'ZIL', 'ONT', 'IOST',
    'NANO', 'BTT', 'WIN', 'SXP', 'STORJ', 'ANT', 'BAND', 'NMR', 'OGN', 'CELO',
    'ANKR', 'SKL', 'GRT', 'LRC', 'PERP', 'DYDX', 'IMX', 'APE', 'GMT', 'OP',
    # Layer 2 & New chains
    'ARB', 'SUI', 'SEI', 'TIA', 'JUP', 'WIF', 'PEPE', 'SHIB', 'FLOKI', 'BONK',
    'AVX', 'TON', 'RENDER', 'FET', 'INJ', 'STX', 'RUNE', 'ORDI', 'WLD', 'PYTH',
    'JTO', 'BLUR', 'STRK', 'DYM', 'PIXEL', 'PORTAL', 'ALT', 'MANTA', 'PENDLE',
    # Additional common coins
    'HBAR', 'KAS', 'QNT', 'AR', 'TAO', 'MNT', 'BEAM', 'RNDR', 'TWT', 'CFX',
    'GALA', 'ROSE', 'KAVA', 'AKT', 'GMX', 'MASK', 'ACH', 'API3', 'SSV', 'AGIX',
    'OCEAN', 'JASMY', 'MAGIC', 'HIGH', 'LQTY', 'RPL', 'FXS', 'CVX', 'OSMO',
    'KLAY', 'CAKE', 'LUNA', 'LUNC', 'UST', 'USDT', 'USDC', 'DAI', 'TUSD', 'BUSD',
    # Meme coins & others
    'BABYDOGE', 'ELON', 'SAFEMOON', 'WOJAK', 'TURBO', 'LADYS', 'MONG', 'BOB',
    'BRETT', 'POPCAT', 'MOG', 'NEIRO', 'GOAT', 'PNUT', 'ACT', 'MOODENG',
}

# Known forex currency codes
FOREX_CURRENCIES = {
    'EUR', 'GBP', 'AUD', 'NZD', 'USD', 'CAD', 'CHF', 'JPY',
    'SEK', 'NOK', 'DKK', 'PLN', 'HUF', 'CZK', 'TRY', 'ZAR',
    'MXN', 'SGD', 'HKD', 'CNH', 'CNY',
}

# Known indices (exact matches)
INDICES_EXACT = {
    'US30', 'US500', 'US100', 'US2000',
    'NAS100', 'NASDAQ', 'SPX', 'SPX500', 'DJ30', 'DJI',
    'DAX', 'DAX30', 'DAX40', 'GER30', 'GER40', 'DE30', 'DE40',
    'FTSE', 'FTSE100', 'UK100',
    'CAC', 'CAC40', 'FRA40', 'FR40',
    'STOXX', 'STOXX50', 'EU50', 'EUSTX50',
    'NIKKEI', 'NIK225', 'NK225', 'JP225', 'JPN225',
    'HANG', 'HSI', 'HK50', 'HANGSENG',
    'AUS200', 'ASX200', 'AU200',
    'IBEX', 'IBEX35', 'ESP35', 'ES35',
    'SMI', 'SMI20', 'SUI20', 'CH20',
    'MIB', 'FTMIB', 'IT40', 'ITA40',
    'AEX', 'AEX25', 'NL25',
    'CHINA50', 'CHINAH', 'CN50',
    'INDIA50', 'NIFTY', 'SENSEX',
    'RUSSELL', 'RUT', 'RTY',
    'VIX', 'UVXY', 'SVXY',
}

# Known commodities (exact matches)
COMMODITIES_EXACT = {
    'XAU', 'XAUUSD', 'GOLD',
    'XAG', 'XAGUSD', 'SILVER',
    'XPT', 'XPTUSD', 'PLATINUM', 'PLAT',
    'XPD', 'XPDUSD', 'PALLADIUM', 'PALL',
    'XBRUSD', 'BRENT', 'UKOIL',
    'XTIUSD', 'WTI', 'USOIL', 'CRUDEOIL', 'OIL',
    'NGAS', 'NATGAS', 'XNGUSD',
    'COPPER', 'XCU', 'HG',
    'WHEAT', 'CORN', 'SOYBEAN', 'SOYBEANS',
    'COFFEE', 'SUGAR', 'COCOA', 'COTTON',
}

# Common crypto pair suffixes
CRYPTO_SUFFIXES = {
    'USD', 'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD',  # Stablecoins
    'EUR', 'GBP', 'JPY', 'TRY', 'BRL', 'AUD',  # Fiat
    'BTC', 'ETH', 'BNB', 'SOL',  # Crypto bases
}


def categorize_symbol(symbol: str) -> str:
    """Categorize a trading symbol into asset class."""
    import re
    upper = symbol.upper()

    # 1. Check for indices (exact match first)
    if upper in INDICES_EXACT:
        return 'Indizes'
    # Index pattern match (e.g., UK100, JP225, etc.)
    if re.match(r'^[A-Z]{2,6}\d{2,3}$', upper):
        return 'Indizes'

    # 2. Check for commodities (exact or starts with)
    for com in COMMODITIES_EXACT:
        if upper == com or upper.startswith(com):
            return 'Rohstoffe'

    # 3. Check for forex (6-char pairs of known currencies)
    if len(upper) == 6:
        base = upper[:3]
        quote = upper[3:]
        if base in FOREX_CURRENCIES and quote in FOREX_CURRENCIES:
            return 'Forex'

    # 4. Check for crypto (base + suffix)
    for base in CRYPTO_BASES:
        if upper.startswith(base):
            suffix = upper[len(base):]
            if suffix == '' or suffix in CRYPTO_SUFFIXES:
                return 'Krypto'

    # 5. Check for stocks (1-5 uppercase letters, not a forex currency)
    if re.match(r'^[A-Z]{1,5}$', upper) and upper not in FOREX_CURRENCIES:
        return 'Aktien'

    # 6. Everything else
    return 'Andere'


# ==================== Response Models ====================

class OHLCVResponse(BaseModel):
    """OHLCV response model."""
    symbol: str
    timeframe: str
    count: int
    source: str
    data: list


class FreshnessResponse(BaseModel):
    """Freshness status response model."""
    symbol: str
    timeframe: str
    data_type: str
    last_updated: Optional[str]
    last_timestamp: Optional[str]
    record_count: Optional[int]
    source: Optional[str]
    is_stale: bool


class IndicatorsResponse(BaseModel):
    """Indicators response model."""
    symbol: str
    timeframe: str
    category: str
    count: int
    data: list | dict


class DBStatsResponse(BaseModel):
    """Database statistics response model."""
    available: bool
    table_sizes: dict
    row_counts: dict
    pool_size: int
    pool_free: int


# ==================== OHLCV Endpoints ====================

@router.get("/ohlcv/{symbol}", response_model=OHLCVResponse)
async def get_ohlcv_from_db(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe (M1, H1, D1, etc.)"),
    limit: int = Query(default=500, ge=1, le=10000),
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    force_refresh: bool = Query(default=False, description="Cache umgehen"),
):
    """
    OHLCV-Daten aus TimescaleDB abrufen.

    Verwendet den 3-Layer-Cache (Redis → TimescaleDB → API).
    """
    try:
        data, source = await data_repository.get_ohlcv(
            symbol=symbol.upper(),
            timeframe=timeframe,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            force_refresh=force_refresh,
        )

        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)

        return OHLCVResponse(
            symbol=symbol.upper(),
            timeframe=tf.value,
            count=len(data),
            source=source,
            data=data,
        )
    except Exception as e:
        logger.error(f"Failed to get OHLCV for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ohlcv/{symbol}/latest")
async def get_latest_ohlcv(
    symbol: str,
    timeframe: str = Query(default="H1"),
):
    """Get the latest OHLCV timestamp for symbol/timeframe."""
    try:
        latest = await timescaledb_service.get_latest_timestamp(
            symbol=symbol.upper(),
            timeframe=timeframe,
        )

        if not latest:
            raise HTTPException(status_code=404, detail="No data found")

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "latest_timestamp": latest.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest timestamp for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Freshness Endpoints ====================

@router.get("/freshness/{symbol}", response_model=FreshnessResponse)
async def get_data_freshness(
    symbol: str,
    timeframe: str = Query(default="H1"),
    data_type: str = Query(default="ohlcv"),
):
    """Aktualitätsstatus für Symbol/Timeframe abrufen."""
    try:
        freshness = await data_repository.get_freshness(
            symbol=symbol.upper(),
            timeframe=timeframe,
            data_type=data_type,
        )

        is_stale = await data_repository.is_data_stale(
            symbol=symbol.upper(),
            timeframe=timeframe,
            data_type=data_type,
        )

        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)

        return FreshnessResponse(
            symbol=symbol.upper(),
            timeframe=tf.value,
            data_type=data_type,
            last_updated=freshness.get("last_updated") if freshness else None,
            last_timestamp=freshness.get("last_timestamp") if freshness else None,
            record_count=freshness.get("record_count") if freshness else None,
            source=freshness.get("source") if freshness else None,
            is_stale=is_stale,
        )
    except Exception as e:
        logger.error(f"Failed to get freshness for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/freshness")
async def get_all_freshness(
    symbol: Optional[str] = Query(default=None),
):
    """Get freshness status for all timeframes (optionally filtered by symbol)."""
    try:
        results = {}

        for tf in SUPPORTED_TIMEFRAMES:
            if symbol:
                freshness = await data_repository.get_freshness(
                    symbol=symbol.upper(),
                    timeframe=tf,
                )
                if freshness:
                    results[tf] = freshness
            else:
                # For all symbols, just check if timescaledb is available
                results[tf] = {"available": timescaledb_service.is_available}

        return {
            "symbol": symbol.upper() if symbol else "all",
            "timeframes": results,
            "db_available": timescaledb_service.is_available,
        }
    except Exception as e:
        logger.error(f"Failed to get all freshness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Indicator Endpoints ====================

@router.get("/indicators/{symbol}", response_model=IndicatorsResponse)
async def get_indicators_from_db(
    symbol: str,
    timeframe: str = Query(default="H1"),
    category: str = Query(
        default="all",
        description="Kategorie: all, momentum, volatility, trend, ma, volume, levels",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    force_refresh: bool = Query(default=False),
):
    """
    Technische Indikatoren aus TimescaleDB abrufen.

    Kategorien:
    - all: Alle Indikatoren aus allen Tabellen
    - momentum: RSI, MACD, Stochastic, ADX, etc.
    - volatility: Bollinger Bands, ATR, Keltner, etc.
    - trend: Ichimoku, Supertrend, PSAR, Aroon, etc.
    - ma: Moving Averages (SMA, EMA, WMA, etc.)
    - volume: OBV, A/D, Chaikin, etc.
    - levels: Pivot Points, Fibonacci, Camarilla
    """
    try:
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)

        if category == "all":
            data, source = await data_repository.get_all_indicators(
                symbol=symbol.upper(),
                timeframe=tf.value,
                limit=limit,
                force_refresh=force_refresh,
            )
            return IndicatorsResponse(
                symbol=symbol.upper(),
                timeframe=tf.value,
                category="all",
                count=sum(len(v) for v in data.values()) if isinstance(data, dict) else len(data),
                data=data,
            )

        # Get specific category
        method_map = {
            "momentum": timescaledb_service.get_momentum_indicators,
            "volatility": timescaledb_service.get_volatility_indicators,
            "trend": timescaledb_service.get_trend_indicators,
            "ma": timescaledb_service.get_ma_indicators,
        }

        if category not in method_map:
            raise HTTPException(status_code=400, detail=f"Unknown category: {category}")

        data = await method_map[category](symbol.upper(), tf.value, limit)

        return IndicatorsResponse(
            symbol=symbol.upper(),
            timeframe=tf.value,
            category=category,
            count=len(data),
            data=data,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/momentum")
async def get_momentum_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000),
    indicators: Optional[list[str]] = Query(
        default=None,
        description="Spezifische Indikatoren: rsi_14, macd_line, stoch_k, etc.",
    ),
):
    """
    Momentum-Indikatoren abrufen.

    Verfügbare Indikatoren:
    - rsi_14, rsi_7, rsi_21, stoch_rsi, connors_rsi
    - stoch_k, stoch_d
    - macd_line, macd_signal, macd_histogram
    - cci, williams_r, roc, momentum
    - adx, plus_di, minus_di, mfi
    """
    try:
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
        data = await timescaledb_service.get_momentum_indicators(
            symbol.upper(), tf.value, limit, indicators
        )
        return {
            "symbol": symbol.upper(),
            "timeframe": tf.value,
            "category": "momentum",
            "count": len(data),
            "data": data,
        }
    except Exception as e:
        logger.error(f"Failed to get momentum indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/volatility")
async def get_volatility_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Volatilitäts-Indikatoren abrufen.

    Enthält: Bollinger Bands, ATR, Keltner Channel, Donchian Channel
    """
    try:
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
        data = await timescaledb_service.get_volatility_indicators(
            symbol.upper(), tf.value, limit
        )
        return {
            "symbol": symbol.upper(),
            "timeframe": tf.value,
            "category": "volatility",
            "count": len(data),
            "data": data,
        }
    except Exception as e:
        logger.error(f"Failed to get volatility indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/trend")
async def get_trend_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Trend-Indikatoren abrufen.

    Enthält: Ichimoku, Supertrend, Parabolic SAR, Aroon, Linear Regression
    """
    try:
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
        data = await timescaledb_service.get_trend_indicators(
            symbol.upper(), tf.value, limit
        )
        return {
            "symbol": symbol.upper(),
            "timeframe": tf.value,
            "category": "trend",
            "count": len(data),
            "data": data,
        }
    except Exception as e:
        logger.error(f"Failed to get trend indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/ma")
async def get_ma_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Moving Average Indikatoren abrufen.

    Enthält: SMA (20, 50, 200), EMA (12, 26, 50, 200), WMA, DEMA, TEMA, VWAP
    """
    try:
        tf = normalize_timeframe_safe(timeframe, Timeframe.H1)
        data = await timescaledb_service.get_ma_indicators(
            symbol.upper(), tf.value, limit
        )
        return {
            "symbol": symbol.upper(),
            "timeframe": tf.value,
            "category": "moving_averages",
            "count": len(data),
            "data": data,
        }
    except Exception as e:
        logger.error(f"Failed to get MA indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Statistics Endpoints ====================

@router.get("/stats")
async def get_db_statistics():
    """Datenbankstatistiken abrufen."""
    try:
        stats = await data_repository.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get DB statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_db_health():
    """TimescaleDB Health Check."""
    try:
        health = await timescaledb_service.health_check()
        return health
    except Exception as e:
        logger.error(f"Failed to get DB health: {e}")
        return {
            "status": "error",
            "available": False,
            "error": str(e),
        }


# ==================== Sync Endpoints ====================

@router.post("/sync/{symbol}")
async def sync_symbol_data(
    symbol: str,
    timeframes: list[str] = Query(default=["H1", "D1"]),
    limit: int = Query(default=500, ge=1, le=5000),
):
    """
    Daten für Symbol von externen APIs synchronisieren.

    Holt Daten von TwelveData/EasyInsight und speichert sie in TimescaleDB.
    """
    from ...data_gateway_service import data_gateway

    try:
        results = {}
        for tf in timeframes:
            tf_normalized = normalize_timeframe_safe(tf, Timeframe.H1)

            # Force refresh to get fresh data from API
            data, source = await data_gateway.get_historical_data_with_fallback(
                symbol=symbol.upper(),
                limit=limit,
                timeframe=tf_normalized.value,
                force_refresh=True,
            )

            results[tf_normalized.value] = {
                "records": len(data),
                "source": source,
                "status": "synced" if data else "no_data",
            }

        return {
            "symbol": symbol.upper(),
            "sync_results": results,
            "db_available": timescaledb_service.is_available,
        }
    except Exception as e:
        logger.error(f"Failed to sync {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeframes")
async def get_supported_timeframes():
    """Get list of supported timeframes."""
    return {
        "timeframes": SUPPORTED_TIMEFRAMES,
        "count": len(SUPPORTED_TIMEFRAMES),
    }


# ==================== Data Validation Endpoints ====================

@router.get("/coverage")
async def get_data_coverage():
    """
    Datenabdeckung pro Symbol und Timeframe abrufen.

    Zeigt an, wie viele Datenpunkte pro Symbol/Timeframe-Kombination
    in TimescaleDB vorhanden sind.
    """
    from ....config.database import get_ohlcv_table_name

    try:
        if not timescaledb_service.is_available:
            return {
                "status": "unavailable",
                "message": "TimescaleDB is not available",
                "symbols": {},
            }

        if not timescaledb_service._pool:
            await timescaledb_service.initialize()
        pool = timescaledb_service._pool

        # Query each timeframe table separately
        coverage = {}
        async with pool.acquire() as conn:
            for tf in SUPPORTED_TIMEFRAMES:
                table = get_ohlcv_table_name(tf)
                try:
                    query = f"""
                        SELECT symbol, COUNT(*) as count
                        FROM {table}
                        GROUP BY symbol
                        ORDER BY symbol
                    """
                    rows = await conn.fetch(query)
                    for row in rows:
                        symbol = row["symbol"]
                        count = row["count"]
                        if symbol not in coverage:
                            coverage[symbol] = {}
                        coverage[symbol][tf] = count
                except Exception as e:
                    logger.debug(f"Table {table} not found or error: {e}")
                    continue

        # Categorize symbols
        categories = {
            'Krypto': [],
            'Forex': [],
            'Indizes': [],
            'Rohstoffe': [],
            'Aktien': [],
            'Andere': [],
        }
        for symbol in coverage.keys():
            cat = categorize_symbol(symbol)
            categories[cat].append(symbol)

        # Sort symbols within each category
        for cat in categories:
            categories[cat].sort()

        return {
            "status": "ok",
            "symbols_count": len(coverage),
            "symbols": coverage,
            "categories": categories,
        }
    except Exception as e:
        logger.error(f"Failed to get data coverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality-check")
async def run_quality_check():
    """
    Datenqualitätsprüfung durchführen.

    Prüft auf:
    - Lücken in Zeitreihen
    - Duplikate
    - Ungültige OHLC-Werte (High < Low, etc.)
    - Null/NaN-Werte
    """
    from ....config.database import get_ohlcv_table_name

    try:
        if not timescaledb_service.is_available:
            return {
                "status": "unavailable",
                "message": "TimescaleDB is not available",
            }

        if not timescaledb_service._pool:
            await timescaledb_service.initialize()
        pool = timescaledb_service._pool

        total_duplicates = 0
        total_invalid = 0
        total_nulls = 0
        total_gaps = 0

        async with pool.acquire() as conn:
            for tf in SUPPORTED_TIMEFRAMES:
                table = get_ohlcv_table_name(tf)
                try:
                    # Check for duplicates
                    dup_query = f"""
                        SELECT COUNT(*) FROM (
                            SELECT symbol, snapshot_time, COUNT(*)
                            FROM {table}
                            GROUP BY symbol, snapshot_time
                            HAVING COUNT(*) > 1
                        ) as dups
                    """
                    dup_result = await conn.fetchval(dup_query)
                    total_duplicates += dup_result or 0

                    # Check for invalid OHLC
                    invalid_query = f"""
                        SELECT COUNT(*) FROM {table}
                        WHERE high < low
                           OR close > high * 1.001
                           OR close < low * 0.999
                    """
                    invalid_result = await conn.fetchval(invalid_query)
                    total_invalid += invalid_result or 0

                    # Check for null values
                    null_query = f"""
                        SELECT COUNT(*) FROM {table}
                        WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                    """
                    null_result = await conn.fetchval(null_query)
                    total_nulls += null_result or 0

                except Exception as e:
                    logger.debug(f"Quality check for {table} failed: {e}")
                    continue

        return {
            "status": "ok",
            "gaps": total_gaps,
            "duplicates": total_duplicates,
            "invalid_ohlc": total_invalid,
            "null_values": total_nulls,
            "quality_score": "good" if (total_duplicates == 0 and total_invalid == 0 and total_nulls == 0) else "needs_attention",
        }
    except Exception as e:
        logger.error(f"Failed to run quality check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_db_summary():
    """
    Zusammenfassung der Datenbankdaten.

    Zeigt Gesamtstatistiken und Zeitbereich der Daten.
    """
    from ....config.database import get_ohlcv_table_name

    try:
        if not timescaledb_service.is_available:
            return {
                "status": "unavailable",
                "available": False,
            }

        if not timescaledb_service._pool:
            await timescaledb_service.initialize()
        pool = timescaledb_service._pool

        all_symbols = set()
        total_rows = 0
        oldest_entry = None
        newest_entry = None
        timeframe_counts = {}

        async with pool.acquire() as conn:
            for tf in SUPPORTED_TIMEFRAMES:
                table = get_ohlcv_table_name(tf)
                try:
                    # Get stats for this timeframe
                    stats_query = f"""
                        SELECT
                            COUNT(DISTINCT symbol) as symbols_count,
                            COUNT(*) as total_rows,
                            MIN(snapshot_time) as oldest_entry,
                            MAX(snapshot_time) as newest_entry
                        FROM {table}
                    """
                    stats = await conn.fetchrow(stats_query)

                    if stats["total_rows"] > 0:
                        # Get symbols
                        symbols_query = f"SELECT DISTINCT symbol FROM {table}"
                        symbol_rows = await conn.fetch(symbols_query)
                        for row in symbol_rows:
                            all_symbols.add(row["symbol"])

                        total_rows += stats["total_rows"]
                        timeframe_counts[tf] = stats["total_rows"]

                        if stats["oldest_entry"]:
                            if oldest_entry is None or stats["oldest_entry"] < oldest_entry:
                                oldest_entry = stats["oldest_entry"]
                        if stats["newest_entry"]:
                            if newest_entry is None or stats["newest_entry"] > newest_entry:
                                newest_entry = stats["newest_entry"]

                except Exception as e:
                    logger.debug(f"Summary for {table} failed: {e}")
                    continue

        return {
            "status": "ok",
            "available": True,
            "symbols_count": len(all_symbols),
            "total_rows": total_rows,
            "oldest_entry": oldest_entry.isoformat() if oldest_entry else None,
            "newest_entry": newest_entry.isoformat() if newest_entry else None,
            "timeframe_counts": timeframe_counts,
        }
    except Exception as e:
        logger.error(f"Failed to get DB summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
