"""Symbol Management Service - CRUD operations for trading symbols.

WICHTIG: Dieser Service speichert Symbols in TimescaleDB (tradingdataservice).
Er verwendet asyncpg für Datenbankzugriff und den DataGatewayService für
externe Datenabrufe (EasyInsight API).

Siehe: DEVELOPMENT_GUIDELINES.md - Datenzugriff-Architektur
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
import asyncpg

from ..config import settings
from .data_gateway_service import data_gateway
from .twelvedata_service import TwelveDataService
from ..models.symbol_data import (
    ManagedSymbol,
    SymbolCategory,
    SymbolSubcategory,
    SymbolStatus,
    SymbolCreateRequest,
    SymbolUpdateRequest,
    SymbolImportResult,
    SymbolStats,
)


class SymbolService:
    """Service for managing trading symbols in TimescaleDB."""

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._db_url = f"postgresql://postgres:postgres@{settings.timescale_host}:{settings.timescale_port}/tradingdataservice"
        self._cache: dict[str, ManagedSymbol] = {}
        self._cache_loaded = False
        # Fallback to JSON file if DB not available
        self._data_file = Path("data/symbols/symbols.json")

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        """Get or create database connection pool."""
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    self._db_url,
                    min_size=1,
                    max_size=5,
                    command_timeout=30,
                )
                logger.info(f"Connected to TimescaleDB: {settings.timescale_host}")
            except Exception as e:
                logger.warning(f"Failed to connect to TimescaleDB: {e}")
                return None
        return self._pool

    async def _ensure_cache_loaded(self):
        """Ensure symbols are loaded into cache from DB."""
        if self._cache_loaded:
            return

        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("SELECT * FROM symbols")
                    for row in rows:
                        symbol = self._row_to_symbol(row)
                        self._cache[symbol.symbol] = symbol
                    self._cache_loaded = True
                    logger.info(f"Loaded {len(self._cache)} symbols from TimescaleDB")
                    return
            except Exception as e:
                logger.warning(f"Failed to load symbols from DB: {e}")

        # Fallback to JSON
        self._load_from_json()
        self._cache_loaded = True

    def _load_from_json(self):
        """Load symbols from JSON file (fallback)."""
        if self._data_file.exists():
            try:
                with open(self._data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for symbol_data in data:
                        symbol = ManagedSymbol(**symbol_data)
                        self._cache[symbol.symbol] = symbol
                logger.info(f"Loaded {len(self._cache)} symbols from JSON fallback")
            except Exception as e:
                logger.error(f"Failed to load symbols from JSON: {e}")

    def _row_to_symbol(self, row: asyncpg.Record) -> ManagedSymbol:
        """Convert database row to ManagedSymbol."""
        return ManagedSymbol(
            symbol=row["symbol"],
            display_name=row.get("display_name"),
            category=SymbolCategory(row["category"]) if row.get("category") else SymbolCategory.OTHER,
            subcategory=SymbolSubcategory(row["subcategory"]) if row.get("subcategory") else None,
            status=SymbolStatus(row["status"]) if row.get("status") else SymbolStatus.ACTIVE,
            description=row.get("description"),
            base_currency=row.get("base_currency"),
            quote_currency=row.get("quote_currency"),
            pip_value=float(row["pip_value"]) if row.get("pip_value") else None,
            min_lot_size=float(row["min_lot_size"]) if row.get("min_lot_size") else 0.01,
            max_lot_size=float(row["max_lot_size"]) if row.get("max_lot_size") else 100.0,
            has_timescaledb_data=row.get("has_timescaledb_data", False),
            first_data_timestamp=row.get("first_data_at"),
            last_data_timestamp=row.get("last_data_at"),
            total_records=row.get("total_records", 0),
            has_nhits_model=row.get("has_nhits_model", False),
            nhits_model_trained_at=row.get("nhits_model_trained_at"),
            twelvedata_symbol=row.get("twelvedata_symbol"),
            easyinsight_symbol=row.get("easyinsight_symbol"),
            twelvedata_available=row.get("twelvedata_available"),
            easyinsight_available=row.get("easyinsight_available"),
            preferred_data_source=row.get("preferred_data_source"),
            is_favorite=row.get("is_favorite", False),
            notes=row.get("notes"),
            tags=list(row["tags"]) if row.get("tags") else [],
            aliases=list(row["aliases"]) if row.get("aliases") else [],
            created_at=row.get("created_at", datetime.utcnow()),
            updated_at=row.get("updated_at", datetime.utcnow()),
        )

    def _detect_category(self, symbol: str) -> SymbolCategory:
        """Auto-detect symbol category based on naming patterns."""
        symbol_upper = symbol.upper()

        # Forex pairs (6 characters, both parts are currencies)
        forex_currencies = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
        if len(symbol_upper) == 6:
            base = symbol_upper[:3]
            quote = symbol_upper[3:]
            if base in forex_currencies and quote in forex_currencies:
                return SymbolCategory.FOREX

        # Crypto patterns
        crypto_bases = {"BTC", "ETH", "XRP", "LTC", "ADA", "DOT", "SOL", "DOGE", "AVAX", "MATIC"}
        if any(symbol_upper.startswith(c) for c in crypto_bases):
            return SymbolCategory.CRYPTO
        if "USDT" in symbol_upper or "USDC" in symbol_upper:
            return SymbolCategory.CRYPTO

        # Indices
        indices = {"US30", "US500", "US100", "GER40", "UK100", "JPN225", "AUS200", "SPX", "NDX", "DJI"}
        if symbol_upper in indices or any(symbol_upper.startswith(i) for i in indices):
            return SymbolCategory.INDEX

        # Commodities
        commodities = {"GOLD", "SILVER", "XAUUSD", "XAGUSD", "OIL", "WTI", "BRENT", "NATGAS"}
        if symbol_upper in commodities or any(symbol_upper.startswith(c) for c in commodities):
            return SymbolCategory.COMMODITY

        return SymbolCategory.OTHER

    def _parse_forex_pair(self, symbol: str) -> tuple[Optional[str], Optional[str]]:
        """Parse forex pair into base and quote currencies."""
        if len(symbol) == 6:
            return symbol[:3].upper(), symbol[3:].upper()
        return None, None

    def _generate_twelvedata_symbol(self, symbol: str, category: SymbolCategory) -> Optional[str]:
        """Generate the Twelve Data API symbol format."""
        symbol_upper = symbol.upper()

        special_mappings = {
            "AVXUSD": "AVAX/USD",
            "DOGUSD": "DOGE/USD",
            "LNKUSD": "LINK/USD",
            "MTCUSD": "MATIC/USD",
            "AUS200": "XJO",
            "EURO50": "STOXX50E",
            "FRA40": "FCHI",
            "GER40": "GDAXI",
            "JP225": "N225",
            "NAS100": "NDX",
            "UK100": "FTSE",
            "US30": "DJI",
            "US500": "SPX",
            "XTIUSD": "CL/USD",
        }

        if symbol_upper in special_mappings:
            return special_mappings[symbol_upper]

        if len(symbol_upper) == 6:
            base = symbol_upper[:3]
            quote = symbol_upper[3:]
            return f"{base}/{quote}"

        return None

    def _generate_easyinsight_symbol(self, symbol: str, category: SymbolCategory) -> str:
        """Generate the EasyInsight API symbol format."""
        symbol_upper = symbol.upper()
        special_mappings = {"DOGUSD": "DOGEUSD"}
        return special_mappings.get(symbol_upper, symbol_upper)

    async def get_all_symbols(
        self,
        category: Optional[SymbolCategory] = None,
        status: Optional[SymbolStatus] = None,
        favorites_only: bool = False,
        with_data_only: bool = False,
    ) -> list[ManagedSymbol]:
        """Get all managed symbols with optional filtering."""
        await self._ensure_cache_loaded()

        symbols = list(self._cache.values())

        if category:
            symbols = [s for s in symbols if s.category == category]
        if status:
            symbols = [s for s in symbols if s.status == status]
        if favorites_only:
            symbols = [s for s in symbols if s.is_favorite]
        if with_data_only:
            symbols = [s for s in symbols if s.has_timescaledb_data]

        return sorted(symbols, key=lambda s: (not s.is_favorite, s.symbol))

    async def get_symbol(self, symbol: str) -> Optional[ManagedSymbol]:
        """Get a specific managed symbol."""
        await self._ensure_cache_loaded()
        return self._cache.get(symbol.upper())

    async def get_symbol_by_alias(self, alias: str) -> Optional[ManagedSymbol]:
        """Get a symbol by its alias."""
        await self._ensure_cache_loaded()
        alias_upper = alias.upper()
        for symbol in self._cache.values():
            if alias_upper in [a.upper() for a in symbol.aliases]:
                return symbol
        return None

    async def resolve_symbol(self, identifier: str) -> Optional[ManagedSymbol]:
        """Resolve a symbol by its ID or any of its aliases."""
        symbol = await self.get_symbol(identifier)
        if symbol:
            return symbol
        return await self.get_symbol_by_alias(identifier)

    async def get_easyinsight_symbol(self, identifier: str) -> str:
        """Get the EasyInsight API symbol for a given identifier."""
        symbol = await self.resolve_symbol(identifier)
        if symbol and symbol.easyinsight_symbol:
            return symbol.easyinsight_symbol
        return identifier.upper()

    async def get_twelvedata_symbol(self, identifier: str) -> Optional[str]:
        """Get the TwelveData API symbol for a given identifier."""
        symbol = await self.resolve_symbol(identifier)
        if symbol and symbol.twelvedata_symbol:
            return symbol.twelvedata_symbol
        return None

    async def is_twelvedata_available(self, identifier: str) -> bool:
        """
        Check if TwelveData API is available for a symbol.

        Priority:
        1. Explicit flag in managed symbol (twelvedata_available)
        2. Auto-detect via TwelveDataService.UNSUPPORTED_SYMBOLS
        """
        symbol = await self.resolve_symbol(identifier)

        # If explicitly set in managed symbol, use that
        if symbol and symbol.twelvedata_available is not None:
            return symbol.twelvedata_available

        # Auto-detect using TwelveDataService's unsupported list
        symbol_id = identifier.upper()
        return symbol_id not in TwelveDataService.UNSUPPORTED_SYMBOLS

    async def is_easyinsight_available(self, identifier: str) -> bool:
        """
        Check if EasyInsight API is available for a symbol.

        Priority:
        1. Explicit flag in managed symbol (easyinsight_available)
        2. Default to True (EasyInsight supports most symbols)
        """
        symbol = await self.resolve_symbol(identifier)

        # If explicitly set in managed symbol, use that
        if symbol and symbol.easyinsight_available is not None:
            return symbol.easyinsight_available

        # EasyInsight typically supports all symbols from the MT5 broker
        return True

    async def get_preferred_data_source(self, identifier: str) -> str:
        """
        Get the preferred data source for a symbol.

        Returns:
            'twelvedata', 'easyinsight', 'yfinance', or 'auto'
        """
        symbol = await self.resolve_symbol(identifier)

        # If explicitly set, use that
        if symbol and symbol.preferred_data_source:
            return symbol.preferred_data_source

        # Auto-determine based on availability
        td_available = await self.is_twelvedata_available(identifier)
        ei_available = await self.is_easyinsight_available(identifier)

        if td_available:
            return "twelvedata"
        elif ei_available:
            return "easyinsight"
        else:
            return "yfinance"

    async def get_data_source_info(self, identifier: str) -> dict:
        """
        Get comprehensive data source availability info for a symbol.

        Returns dict with:
        - twelvedata_available: bool
        - easyinsight_available: bool
        - preferred_source: str
        - auto_detected: bool (whether values were auto-detected or explicit)
        """
        symbol = await self.resolve_symbol(identifier)
        symbol_id = identifier.upper()

        td_explicit = symbol.twelvedata_available if symbol else None
        ei_explicit = symbol.easyinsight_available if symbol else None
        preferred = symbol.preferred_data_source if symbol else None

        td_available = await self.is_twelvedata_available(identifier)
        ei_available = await self.is_easyinsight_available(identifier)
        pref_source = await self.get_preferred_data_source(identifier)

        return {
            "symbol": symbol_id,
            "twelvedata": {
                "available": td_available,
                "explicit": td_explicit is not None,
                "symbol": symbol.twelvedata_symbol if symbol else self._generate_twelvedata_symbol(symbol_id, SymbolCategory.OTHER),
            },
            "easyinsight": {
                "available": ei_available,
                "explicit": ei_explicit is not None,
                "symbol": symbol.easyinsight_symbol if symbol else symbol_id,
            },
            "preferred_source": pref_source,
            "preferred_explicit": preferred is not None,
        }

    async def set_data_source_availability(
        self,
        symbol_id: str,
        twelvedata_available: Optional[bool] = None,
        easyinsight_available: Optional[bool] = None,
        preferred_data_source: Optional[str] = None,
    ) -> Optional[ManagedSymbol]:
        """
        Set data source availability flags for a symbol.

        Args:
            symbol_id: Symbol identifier
            twelvedata_available: True/False to explicitly set, None to auto-detect
            easyinsight_available: True/False to explicitly set, None to auto-detect
            preferred_data_source: 'twelvedata', 'easyinsight', 'yfinance', or None for auto

        Returns:
            Updated ManagedSymbol or None if symbol not found
        """
        symbol_id = symbol_id.upper()
        await self._ensure_cache_loaded()

        symbol = self._cache.get(symbol_id)
        if not symbol:
            return None

        # Update fields
        symbol.twelvedata_available = twelvedata_available
        symbol.easyinsight_available = easyinsight_available
        symbol.preferred_data_source = preferred_data_source
        symbol.updated_at = datetime.utcnow()

        # Update in database
        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE symbols SET
                            twelvedata_available = $2,
                            easyinsight_available = $3,
                            preferred_data_source = $4,
                            updated_at = $5
                        WHERE symbol = $1
                    """,
                        symbol_id,
                        twelvedata_available,
                        easyinsight_available,
                        preferred_data_source,
                        symbol.updated_at,
                    )
            except Exception as e:
                logger.error(f"Failed to update data source availability for {symbol_id}: {e}")

        self._cache[symbol_id] = symbol
        logger.info(f"Updated data source availability for {symbol_id}: TD={twelvedata_available}, EI={easyinsight_available}, Preferred={preferred_data_source}")
        return symbol

    async def create_symbol(self, request: SymbolCreateRequest) -> ManagedSymbol:
        """Create a new managed symbol in the database."""
        symbol_id = request.symbol.upper()

        await self._ensure_cache_loaded()
        if symbol_id in self._cache:
            raise ValueError(f"Symbol '{symbol_id}' already exists")

        category = request.category
        if category == SymbolCategory.FOREX:
            category = self._detect_category(symbol_id)

        base, quote = self._parse_forex_pair(symbol_id)
        if request.base_currency:
            base = request.base_currency
        if request.quote_currency:
            quote = request.quote_currency

        twelvedata_sym = request.twelvedata_symbol or self._generate_twelvedata_symbol(symbol_id, category)
        easyinsight_sym = request.easyinsight_symbol or self._generate_easyinsight_symbol(symbol_id, category)

        now = datetime.utcnow()
        symbol = ManagedSymbol(
            symbol=symbol_id,
            display_name=request.display_name or symbol_id,
            category=category,
            subcategory=request.subcategory,
            status=SymbolStatus.ACTIVE,
            description=request.description,
            base_currency=base,
            quote_currency=quote,
            pip_value=request.pip_value,
            min_lot_size=request.min_lot_size,
            max_lot_size=request.max_lot_size,
            twelvedata_symbol=twelvedata_sym,
            easyinsight_symbol=easyinsight_sym,
            twelvedata_available=request.twelvedata_available,
            easyinsight_available=request.easyinsight_available,
            preferred_data_source=request.preferred_data_source,
            notes=request.notes,
            tags=request.tags,
            aliases=request.aliases,
            created_at=now,
            updated_at=now,
        )

        # Check data availability
        await self._update_symbol_data_info(symbol)

        # Save to database
        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO symbols (
                            symbol, display_name, category, subcategory, status, description,
                            base_currency, quote_currency, pip_value, min_lot_size, max_lot_size,
                            twelvedata_symbol, easyinsight_symbol, is_active, is_favorite,
                            has_timescaledb_data, first_data_at, last_data_at, total_records,
                            has_nhits_model, nhits_model_trained_at, notes, tags, aliases,
                            created_at, updated_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                            $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26
                        )
                    """,
                        symbol.symbol,
                        symbol.display_name,
                        symbol.category.value,
                        symbol.subcategory.value if symbol.subcategory else None,
                        symbol.status.value,
                        symbol.description,
                        symbol.base_currency,
                        symbol.quote_currency,
                        symbol.pip_value,
                        symbol.min_lot_size,
                        symbol.max_lot_size,
                        symbol.twelvedata_symbol,
                        symbol.easyinsight_symbol,
                        symbol.status == SymbolStatus.ACTIVE,
                        symbol.is_favorite,
                        symbol.has_timescaledb_data,
                        symbol.first_data_timestamp,
                        symbol.last_data_timestamp,
                        symbol.total_records,
                        symbol.has_nhits_model,
                        symbol.nhits_model_trained_at,
                        symbol.notes,
                        symbol.tags,
                        symbol.aliases,
                        symbol.created_at,
                        symbol.updated_at,
                    )
            except Exception as e:
                logger.error(f"Failed to insert symbol into DB: {e}")
                raise

        self._cache[symbol_id] = symbol
        logger.info(f"Created managed symbol: {symbol_id}")
        return symbol

    async def update_symbol(
        self, symbol_id: str, request: SymbolUpdateRequest
    ) -> Optional[ManagedSymbol]:
        """Update an existing managed symbol."""
        symbol_id = symbol_id.upper()
        await self._ensure_cache_loaded()

        symbol = self._cache.get(symbol_id)
        if not symbol:
            return None

        update_data = request.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                setattr(symbol, field, value)

        symbol.updated_at = datetime.utcnow()

        # Update in database
        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE symbols SET
                            display_name = $2, category = $3, subcategory = $4, status = $5,
                            description = $6, base_currency = $7, quote_currency = $8,
                            pip_value = $9, min_lot_size = $10, max_lot_size = $11,
                            twelvedata_symbol = $12, easyinsight_symbol = $13,
                            is_active = $14, is_favorite = $15, notes = $16, tags = $17,
                            aliases = $18, updated_at = $19
                        WHERE symbol = $1
                    """,
                        symbol.symbol,
                        symbol.display_name,
                        symbol.category.value,
                        symbol.subcategory.value if symbol.subcategory else None,
                        symbol.status.value,
                        symbol.description,
                        symbol.base_currency,
                        symbol.quote_currency,
                        symbol.pip_value,
                        symbol.min_lot_size,
                        symbol.max_lot_size,
                        symbol.twelvedata_symbol,
                        symbol.easyinsight_symbol,
                        symbol.status == SymbolStatus.ACTIVE,
                        symbol.is_favorite,
                        symbol.notes,
                        symbol.tags,
                        symbol.aliases,
                        symbol.updated_at,
                    )
            except Exception as e:
                logger.error(f"Failed to update symbol in DB: {e}")

        self._cache[symbol_id] = symbol
        logger.info(f"Updated managed symbol: {symbol_id}")
        return symbol

    async def delete_symbol(self, symbol_id: str) -> bool:
        """Delete a managed symbol."""
        symbol_id = symbol_id.upper()
        await self._ensure_cache_loaded()

        if symbol_id not in self._cache:
            return False

        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("DELETE FROM symbols WHERE symbol = $1", symbol_id)
            except Exception as e:
                logger.error(f"Failed to delete symbol from DB: {e}")

        del self._cache[symbol_id]
        logger.info(f"Deleted managed symbol: {symbol_id}")
        return True

    async def toggle_favorite(self, symbol_id: str) -> Optional[ManagedSymbol]:
        """Toggle favorite status for a symbol."""
        symbol_id = symbol_id.upper()
        await self._ensure_cache_loaded()

        symbol = self._cache.get(symbol_id)
        if not symbol:
            return None

        symbol.is_favorite = not symbol.is_favorite
        symbol.updated_at = datetime.utcnow()

        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE symbols SET is_favorite = $2, updated_at = $3 WHERE symbol = $1",
                        symbol_id, symbol.is_favorite, symbol.updated_at
                    )
            except Exception as e:
                logger.error(f"Failed to update favorite in DB: {e}")

        self._cache[symbol_id] = symbol
        return symbol

    async def _update_symbol_data_info(self, symbol: ManagedSymbol):
        """Update symbol with data information via Data Gateway."""
        try:
            symbol_info = await data_gateway.get_symbol_info(symbol.symbol)

            if symbol_info and symbol_info.get("count", 0) > 0:
                symbol.has_timescaledb_data = True
                earliest = symbol_info.get("earliest")
                latest = symbol_info.get("latest")
                if earliest:
                    symbol.first_data_timestamp = datetime.fromisoformat(
                        earliest.replace("+01:00", "+00:00").replace("+00:00", "")
                    )
                if latest:
                    symbol.last_data_timestamp = datetime.fromisoformat(
                        latest.replace("+01:00", "+00:00").replace("+00:00", "")
                    )
                symbol.total_records = symbol_info.get("count", 0)
            else:
                symbol.has_timescaledb_data = False
                symbol.first_data_timestamp = None
                symbol.last_data_timestamp = None
                symbol.total_records = 0
        except Exception as e:
            logger.warning(f"Failed to fetch data info for {symbol.symbol}: {e}")

    async def _check_nhits_model(self, symbol: ManagedSymbol):
        """Check if NHITS model exists for symbol."""
        model_path = Path(settings.nhits_model_path) / f"{symbol.symbol}_model.pt"
        metadata_path = Path(settings.nhits_model_path) / f"{symbol.symbol}_metadata.json"

        symbol.has_nhits_model = model_path.exists()

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    trained_at = metadata.get("trained_at")
                    if trained_at:
                        symbol.nhits_model_trained_at = datetime.fromisoformat(trained_at)
            except Exception:
                pass

    def _map_api_category_to_enum(self, api_category: str) -> SymbolCategory:
        """Map EasyInsight API category string to SymbolCategory enum."""
        category_map = {
            "Crypto": SymbolCategory.CRYPTO,
            "Forex": SymbolCategory.FOREX,
            "Indices": SymbolCategory.INDEX,
            "Metals": SymbolCategory.COMMODITY,
            "Extra": SymbolCategory.COMMODITY,
        }
        return category_map.get(api_category, SymbolCategory.OTHER)

    async def import_from_easyinsight(self) -> SymbolImportResult:
        """Import all symbols via Data Gateway into the database."""
        result = SymbolImportResult(
            total_found=0, imported=0, updated=0, skipped=0
        )

        try:
            symbols_data = await data_gateway.get_available_symbols()
            result.total_found = len(symbols_data)

            pool = await self._get_pool()

            for symbol_info in symbols_data:
                symbol_id = symbol_info.get("symbol")
                if not symbol_id:
                    continue

                try:
                    await self._ensure_cache_loaded()
                    existing = self._cache.get(symbol_id)

                    # Parse timestamps
                    first_timestamp = None
                    last_timestamp = None
                    earliest = symbol_info.get("earliest")
                    latest = symbol_info.get("latest")
                    if earliest:
                        try:
                            first_timestamp = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                        except ValueError:
                            first_timestamp = datetime.fromisoformat(earliest.split("+")[0])
                    if latest:
                        try:
                            last_timestamp = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                        except ValueError:
                            last_timestamp = datetime.fromisoformat(latest.split("+")[0])

                    total_records = symbol_info.get("count", 0)

                    if existing:
                        # Update existing
                        existing.has_timescaledb_data = total_records > 0
                        existing.first_data_timestamp = first_timestamp
                        existing.last_data_timestamp = last_timestamp
                        existing.total_records = total_records
                        existing.updated_at = datetime.utcnow()

                        api_category = symbol_info.get("category")
                        if api_category:
                            existing.category = self._map_api_category_to_enum(api_category)

                        if not existing.twelvedata_symbol:
                            existing.twelvedata_symbol = self._generate_twelvedata_symbol(
                                symbol_id, existing.category
                            )

                        await self._check_nhits_model(existing)

                        # Update in DB
                        if pool:
                            try:
                                async with pool.acquire() as conn:
                                    await conn.execute("""
                                        UPDATE symbols SET
                                            category = $2, has_timescaledb_data = $3,
                                            first_data_at = $4, last_data_at = $5,
                                            total_records = $6, twelvedata_symbol = $7,
                                            has_nhits_model = $8, nhits_model_trained_at = $9,
                                            updated_at = $10
                                        WHERE symbol = $1
                                    """,
                                        symbol_id,
                                        existing.category.value,
                                        existing.has_timescaledb_data,
                                        existing.first_data_timestamp,
                                        existing.last_data_timestamp,
                                        existing.total_records,
                                        existing.twelvedata_symbol,
                                        existing.has_nhits_model,
                                        existing.nhits_model_trained_at,
                                        existing.updated_at,
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to update {symbol_id} in DB: {e}")

                        self._cache[symbol_id] = existing
                        result.updated += 1
                    else:
                        # Create new
                        api_category = symbol_info.get("category")
                        category = (
                            self._map_api_category_to_enum(api_category)
                            if api_category
                            else self._detect_category(symbol_id)
                        )
                        base, quote = self._parse_forex_pair(symbol_id)
                        twelvedata_sym = self._generate_twelvedata_symbol(symbol_id, category)
                        now = datetime.utcnow()

                        new_symbol = ManagedSymbol(
                            symbol=symbol_id,
                            display_name=symbol_id,
                            category=category,
                            status=SymbolStatus.ACTIVE,
                            base_currency=base,
                            quote_currency=quote,
                            has_timescaledb_data=total_records > 0,
                            first_data_timestamp=first_timestamp,
                            last_data_timestamp=last_timestamp,
                            total_records=total_records,
                            twelvedata_symbol=twelvedata_sym,
                            created_at=now,
                            updated_at=now,
                        )

                        await self._check_nhits_model(new_symbol)

                        # Insert into DB
                        if pool:
                            try:
                                async with pool.acquire() as conn:
                                    await conn.execute("""
                                        INSERT INTO symbols (
                                            symbol, display_name, category, status,
                                            base_currency, quote_currency, twelvedata_symbol,
                                            is_active, is_favorite, has_timescaledb_data,
                                            first_data_at, last_data_at, total_records,
                                            has_nhits_model, nhits_model_trained_at,
                                            tags, aliases, created_at, updated_at
                                        ) VALUES (
                                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                                            $11, $12, $13, $14, $15, $16, $17, $18, $19
                                        )
                                        ON CONFLICT (symbol) DO UPDATE SET
                                            has_timescaledb_data = EXCLUDED.has_timescaledb_data,
                                            first_data_at = EXCLUDED.first_data_at,
                                            last_data_at = EXCLUDED.last_data_at,
                                            total_records = EXCLUDED.total_records,
                                            updated_at = EXCLUDED.updated_at
                                    """,
                                        new_symbol.symbol,
                                        new_symbol.display_name,
                                        new_symbol.category.value,
                                        new_symbol.status.value,
                                        new_symbol.base_currency,
                                        new_symbol.quote_currency,
                                        new_symbol.twelvedata_symbol,
                                        True,  # is_active
                                        False,  # is_favorite
                                        new_symbol.has_timescaledb_data,
                                        new_symbol.first_data_timestamp,
                                        new_symbol.last_data_timestamp,
                                        new_symbol.total_records,
                                        new_symbol.has_nhits_model,
                                        new_symbol.nhits_model_trained_at,
                                        [],  # tags
                                        [],  # aliases
                                        new_symbol.created_at,
                                        new_symbol.updated_at,
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to insert {symbol_id} into DB: {e}")

                        self._cache[symbol_id] = new_symbol
                        result.imported += 1

                    result.symbols.append(symbol_id)

                except Exception as e:
                    error_msg = f"Error processing {symbol_id}: {str(e)}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)
                    result.skipped += 1

            logger.info(
                f"Import complete: {result.imported} imported, "
                f"{result.updated} updated, {result.skipped} skipped"
            )

        except Exception as e:
            error_msg = f"Import via Data Gateway failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    async def import_from_timescaledb(self) -> SymbolImportResult:
        """Import all symbols via Data Gateway (legacy name for compatibility)."""
        return await self.import_from_easyinsight()

    async def migrate_from_json(self) -> dict:
        """Migrate existing symbols from JSON file to database."""
        if not self._data_file.exists():
            return {"migrated": 0, "errors": [], "message": "No JSON file found"}

        migrated = 0
        errors = []

        try:
            with open(self._data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            pool = await self._get_pool()
            if not pool:
                return {"migrated": 0, "errors": ["Database not available"]}

            for symbol_data in data:
                try:
                    symbol = ManagedSymbol(**symbol_data)

                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO symbols (
                                symbol, display_name, category, subcategory, status,
                                description, base_currency, quote_currency, pip_value,
                                min_lot_size, max_lot_size, twelvedata_symbol,
                                easyinsight_symbol, is_active, is_favorite,
                                has_timescaledb_data, first_data_at, last_data_at,
                                total_records, has_nhits_model, nhits_model_trained_at,
                                notes, tags, aliases, created_at, updated_at
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                                $13, $14, $15, $16, $17, $18, $19, $20, $21, $22,
                                $23, $24, $25, $26
                            )
                            ON CONFLICT (symbol) DO UPDATE SET
                                display_name = EXCLUDED.display_name,
                                category = EXCLUDED.category,
                                subcategory = EXCLUDED.subcategory,
                                status = EXCLUDED.status,
                                description = EXCLUDED.description,
                                twelvedata_symbol = EXCLUDED.twelvedata_symbol,
                                easyinsight_symbol = EXCLUDED.easyinsight_symbol,
                                is_favorite = EXCLUDED.is_favorite,
                                has_timescaledb_data = EXCLUDED.has_timescaledb_data,
                                first_data_at = EXCLUDED.first_data_at,
                                last_data_at = EXCLUDED.last_data_at,
                                total_records = EXCLUDED.total_records,
                                has_nhits_model = EXCLUDED.has_nhits_model,
                                notes = EXCLUDED.notes,
                                tags = EXCLUDED.tags,
                                aliases = EXCLUDED.aliases,
                                updated_at = EXCLUDED.updated_at
                        """,
                            symbol.symbol,
                            symbol.display_name,
                            symbol.category.value,
                            symbol.subcategory.value if symbol.subcategory else None,
                            symbol.status.value,
                            symbol.description,
                            symbol.base_currency,
                            symbol.quote_currency,
                            symbol.pip_value,
                            symbol.min_lot_size,
                            symbol.max_lot_size,
                            symbol.twelvedata_symbol,
                            symbol.easyinsight_symbol,
                            symbol.status == SymbolStatus.ACTIVE,
                            symbol.is_favorite,
                            symbol.has_timescaledb_data,
                            symbol.first_data_timestamp,
                            symbol.last_data_timestamp,
                            symbol.total_records,
                            symbol.has_nhits_model,
                            symbol.nhits_model_trained_at,
                            symbol.notes,
                            symbol.tags,
                            symbol.aliases,
                            symbol.created_at,
                            symbol.updated_at,
                        )

                    self._cache[symbol.symbol] = symbol
                    migrated += 1

                except Exception as e:
                    errors.append(f"{symbol_data.get('symbol', 'unknown')}: {str(e)}")

            # Reset cache to force reload from DB
            self._cache_loaded = False

            logger.info(f"Migrated {migrated} symbols from JSON to TimescaleDB")

        except Exception as e:
            errors.append(f"Migration failed: {str(e)}")

        return {
            "migrated": migrated,
            "errors": errors,
            "total": len(data) if 'data' in dir() else 0,
        }

    async def refresh_symbol_data(self, symbol_id: str) -> Optional[ManagedSymbol]:
        """Refresh data info for a specific symbol via Data Gateway."""
        symbol_id = symbol_id.upper()
        await self._ensure_cache_loaded()

        symbol = self._cache.get(symbol_id)
        if not symbol:
            return None

        await self._update_symbol_data_info(symbol)
        await self._check_nhits_model(symbol)
        symbol.updated_at = datetime.utcnow()

        # Update in DB
        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE symbols SET
                            has_timescaledb_data = $2, first_data_at = $3,
                            last_data_at = $4, total_records = $5,
                            has_nhits_model = $6, nhits_model_trained_at = $7,
                            updated_at = $8
                        WHERE symbol = $1
                    """,
                        symbol_id,
                        symbol.has_timescaledb_data,
                        symbol.first_data_timestamp,
                        symbol.last_data_timestamp,
                        symbol.total_records,
                        symbol.has_nhits_model,
                        symbol.nhits_model_trained_at,
                        symbol.updated_at,
                    )
            except Exception as e:
                logger.warning(f"Failed to update symbol data in DB: {e}")

        self._cache[symbol_id] = symbol
        return symbol

    async def get_stats(self) -> SymbolStats:
        """Get statistics about managed symbols."""
        await self._ensure_cache_loaded()
        symbols = list(self._cache.values())

        by_category = {}
        for cat in SymbolCategory:
            count = len([s for s in symbols if s.category == cat])
            if count > 0:
                by_category[cat.value] = count

        by_subcategory = {}
        for subcat in SymbolSubcategory:
            count = len([s for s in symbols if s.subcategory == subcat])
            if count > 0:
                by_subcategory[subcat.value] = count

        return SymbolStats(
            total_symbols=len(symbols),
            active_symbols=len([s for s in symbols if s.status == SymbolStatus.ACTIVE]),
            inactive_symbols=len([s for s in symbols if s.status == SymbolStatus.INACTIVE]),
            suspended_symbols=len([s for s in symbols if s.status == SymbolStatus.SUSPENDED]),
            with_timescaledb_data=len([s for s in symbols if s.has_timescaledb_data]),
            with_nhits_model=len([s for s in symbols if s.has_nhits_model]),
            by_category=by_category,
            by_subcategory=by_subcategory,
            favorites_count=len([s for s in symbols if s.is_favorite]),
        )

    async def search_symbols(
        self,
        query: str,
        limit: int = 20,
    ) -> list[ManagedSymbol]:
        """Search symbols by name, description, tags, or aliases."""
        await self._ensure_cache_loaded()
        query = query.lower()
        results = []

        for symbol in self._cache.values():
            score = 0

            if symbol.symbol.lower() == query:
                score = 100
            elif any(alias.lower() == query for alias in symbol.aliases):
                score = 95
            elif symbol.symbol.lower().startswith(query):
                score = 80
            elif any(alias.lower().startswith(query) for alias in symbol.aliases):
                score = 75
            elif query in symbol.symbol.lower():
                score = 60
            elif any(query in alias.lower() for alias in symbol.aliases):
                score = 55
            elif symbol.display_name and query in symbol.display_name.lower():
                score = 50
            elif any(query in tag.lower() for tag in symbol.tags):
                score = 40
            elif symbol.description and query in symbol.description.lower():
                score = 30

            if score > 0:
                results.append((score, symbol))

        results.sort(key=lambda x: (-x[0], x[1].symbol))
        return [s for _, s in results[:limit]]

    async def migrate_api_symbols(self) -> dict:
        """Migrate all existing symbols to have proper TwelveData and EasyInsight symbols."""
        await self._ensure_cache_loaded()
        updated_count = 0
        skipped_count = 0
        errors = []

        pool = await self._get_pool()

        for symbol in self._cache.values():
            try:
                changed = False

                if not symbol.twelvedata_symbol:
                    symbol.twelvedata_symbol = self._generate_twelvedata_symbol(
                        symbol.symbol, symbol.category
                    )
                    changed = True

                if not symbol.easyinsight_symbol:
                    symbol.easyinsight_symbol = self._generate_easyinsight_symbol(
                        symbol.symbol, symbol.category
                    )
                    changed = True

                if changed:
                    symbol.updated_at = datetime.utcnow()
                    updated_count += 1

                    if pool:
                        try:
                            async with pool.acquire() as conn:
                                await conn.execute("""
                                    UPDATE symbols SET
                                        twelvedata_symbol = $2,
                                        easyinsight_symbol = $3,
                                        updated_at = $4
                                    WHERE symbol = $1
                                """,
                                    symbol.symbol,
                                    symbol.twelvedata_symbol,
                                    symbol.easyinsight_symbol,
                                    symbol.updated_at,
                                )
                        except Exception as e:
                            logger.warning(f"Failed to update {symbol.symbol} in DB: {e}")
                else:
                    skipped_count += 1

            except Exception as e:
                errors.append(f"{symbol.symbol}: {str(e)}")
                logger.error(f"Error migrating symbol {symbol.symbol}: {e}")

        if updated_count > 0:
            logger.info(f"Migrated {updated_count} symbols with API symbols")

        return {
            "updated": updated_count,
            "skipped": skipped_count,
            "errors": errors,
            "total": len(self._cache),
        }

    async def cleanup_redundant_aliases(self) -> dict:
        """Remove aliases that are identical to TwelveData or EasyInsight symbols."""
        await self._ensure_cache_loaded()
        cleaned_count = 0
        removed_aliases = []
        errors = []

        pool = await self._get_pool()

        for symbol in self._cache.values():
            try:
                if not symbol.aliases:
                    continue

                td_symbol = symbol.twelvedata_symbol or ""
                ei_symbol = symbol.easyinsight_symbol or ""

                original_aliases = symbol.aliases.copy()
                symbol.aliases = [
                    alias
                    for alias in symbol.aliases
                    if alias != td_symbol and alias != ei_symbol
                ]

                removed = set(original_aliases) - set(symbol.aliases)
                if removed:
                    symbol.updated_at = datetime.utcnow()
                    cleaned_count += 1
                    removed_aliases.append({
                        "symbol": symbol.symbol,
                        "removed": list(removed),
                    })

                    if pool:
                        try:
                            async with pool.acquire() as conn:
                                await conn.execute("""
                                    UPDATE symbols SET aliases = $2, updated_at = $3
                                    WHERE symbol = $1
                                """,
                                    symbol.symbol,
                                    symbol.aliases,
                                    symbol.updated_at,
                                )
                        except Exception as e:
                            logger.warning(f"Failed to update aliases for {symbol.symbol}: {e}")

            except Exception as e:
                errors.append(f"{symbol.symbol}: {str(e)}")
                logger.error(f"Error cleaning aliases for {symbol.symbol}: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned redundant aliases from {cleaned_count} symbols")

        return {
            "cleaned": cleaned_count,
            "removed_aliases": removed_aliases,
            "errors": errors,
            "total": len(self._cache),
        }


# Global service instance
symbol_service = SymbolService()
