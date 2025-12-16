"""Symbol Management Service - CRUD operations for trading symbols."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx
from loguru import logger

from ..config import settings
from ..models.symbol_data import (
    ManagedSymbol,
    SymbolCategory,
    SymbolStatus,
    SymbolCreateRequest,
    SymbolUpdateRequest,
    SymbolImportResult,
    SymbolStats,
)


class SymbolService:
    """Service for managing trading symbols."""

    def __init__(self):
        self._symbols: dict[str, ManagedSymbol] = {}
        self._data_file = Path("data/symbols.json")
        self._http_client: Optional[httpx.AsyncClient] = None
        self._load_symbols()

    def _load_symbols(self):
        """Load symbols from JSON file."""
        if self._data_file.exists():
            try:
                with open(self._data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for symbol_data in data:
                        symbol = ManagedSymbol(**symbol_data)
                        self._symbols[symbol.symbol] = symbol
                logger.info(f"Loaded {len(self._symbols)} managed symbols")
            except Exception as e:
                logger.error(f"Failed to load symbols: {e}")
                self._symbols = {}

    def _save_symbols(self):
        """Save symbols to JSON file."""
        try:
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            data = [s.model_dump(mode="json") for s in self._symbols.values()]
            with open(self._data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved {len(self._symbols)} symbols to {self._data_file}")
        except Exception as e:
            logger.error(f"Failed to save symbols: {e}")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for EasyInsight API."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=settings.easyinsight_api_url,
                timeout=30.0
            )
        return self._http_client

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

    async def get_all_symbols(
        self,
        category: Optional[SymbolCategory] = None,
        status: Optional[SymbolStatus] = None,
        favorites_only: bool = False,
        with_data_only: bool = False,
    ) -> list[ManagedSymbol]:
        """Get all managed symbols with optional filtering."""
        symbols = list(self._symbols.values())

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
        return self._symbols.get(symbol.upper())

    async def create_symbol(self, request: SymbolCreateRequest) -> ManagedSymbol:
        """Create a new managed symbol."""
        symbol_id = request.symbol.upper()

        if symbol_id in self._symbols:
            raise ValueError(f"Symbol '{symbol_id}' already exists")

        # Auto-detect category if not specified
        category = request.category
        if category == SymbolCategory.FOREX:
            category = self._detect_category(symbol_id)

        # Parse forex pair
        base, quote = self._parse_forex_pair(symbol_id)
        if request.base_currency:
            base = request.base_currency
        if request.quote_currency:
            quote = request.quote_currency

        symbol = ManagedSymbol(
            symbol=symbol_id,
            display_name=request.display_name or symbol_id,
            category=category,
            status=SymbolStatus.ACTIVE,
            description=request.description,
            base_currency=base,
            quote_currency=quote,
            pip_value=request.pip_value,
            min_lot_size=request.min_lot_size,
            max_lot_size=request.max_lot_size,
            notes=request.notes,
            tags=request.tags,
        )

        # Check data availability via EasyInsight API
        await self._update_symbol_data_info(symbol)

        self._symbols[symbol_id] = symbol
        self._save_symbols()

        logger.info(f"Created managed symbol: {symbol_id}")
        return symbol

    async def update_symbol(
        self, symbol_id: str, request: SymbolUpdateRequest
    ) -> Optional[ManagedSymbol]:
        """Update an existing managed symbol."""
        symbol_id = symbol_id.upper()
        symbol = self._symbols.get(symbol_id)

        if not symbol:
            return None

        # Update fields if provided
        update_data = request.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if value is not None:
                setattr(symbol, field, value)

        symbol.updated_at = datetime.utcnow()

        self._symbols[symbol_id] = symbol
        self._save_symbols()

        logger.info(f"Updated managed symbol: {symbol_id}")
        return symbol

    async def delete_symbol(self, symbol_id: str) -> bool:
        """Delete a managed symbol."""
        symbol_id = symbol_id.upper()

        if symbol_id not in self._symbols:
            return False

        del self._symbols[symbol_id]
        self._save_symbols()

        logger.info(f"Deleted managed symbol: {symbol_id}")
        return True

    async def toggle_favorite(self, symbol_id: str) -> Optional[ManagedSymbol]:
        """Toggle favorite status for a symbol."""
        symbol_id = symbol_id.upper()
        symbol = self._symbols.get(symbol_id)

        if not symbol:
            return None

        symbol.is_favorite = not symbol.is_favorite
        symbol.updated_at = datetime.utcnow()

        self._symbols[symbol_id] = symbol
        self._save_symbols()

        return symbol

    async def _update_symbol_data_info(self, symbol: ManagedSymbol):
        """Update symbol with data information from EasyInsight API."""
        try:
            client = await self._get_http_client()
            response = await client.get("/symbols")
            response.raise_for_status()

            symbols_data = response.json()

            # Find the symbol in the API response
            symbol_info = next(
                (s for s in symbols_data if s.get("symbol") == symbol.symbol),
                None
            )

            if symbol_info and symbol_info.get("count", 0) > 0:
                symbol.has_timescaledb_data = True
                # Parse ISO timestamps from API
                earliest = symbol_info.get("earliest")
                latest = symbol_info.get("latest")
                if earliest:
                    symbol.first_data_timestamp = datetime.fromisoformat(earliest.replace("+01:00", "+00:00").replace("+00:00", ""))
                if latest:
                    symbol.last_data_timestamp = datetime.fromisoformat(latest.replace("+01:00", "+00:00").replace("+00:00", ""))
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
            "Extra": SymbolCategory.COMMODITY,  # Oil etc.
        }
        return category_map.get(api_category, SymbolCategory.OTHER)

    async def import_from_easyinsight(self) -> SymbolImportResult:
        """Import all symbols from EasyInsight API."""
        result = SymbolImportResult(
            total_found=0,
            imported=0,
            updated=0,
            skipped=0,
        )

        try:
            client = await self._get_http_client()
            response = await client.get("/symbols")
            response.raise_for_status()

            symbols_data = response.json()
            result.total_found = len(symbols_data)

            for symbol_info in symbols_data:
                symbol_id = symbol_info.get("symbol")
                if not symbol_id:
                    continue

                try:
                    existing = self._symbols.get(symbol_id)

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
                        # Update existing symbol with new data info
                        existing.has_timescaledb_data = total_records > 0
                        existing.first_data_timestamp = first_timestamp
                        existing.last_data_timestamp = last_timestamp
                        existing.total_records = total_records
                        existing.updated_at = datetime.utcnow()

                        # Update category from API if it was previously auto-detected
                        api_category = symbol_info.get("category")
                        if api_category:
                            existing.category = self._map_api_category_to_enum(api_category)

                        # Check NHITS model
                        await self._check_nhits_model(existing)

                        self._symbols[symbol_id] = existing
                        result.updated += 1
                    else:
                        # Create new symbol with category from API
                        api_category = symbol_info.get("category")
                        category = self._map_api_category_to_enum(api_category) if api_category else self._detect_category(symbol_id)
                        base, quote = self._parse_forex_pair(symbol_id)

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
                        )

                        # Check NHITS model
                        await self._check_nhits_model(new_symbol)

                        self._symbols[symbol_id] = new_symbol
                        result.imported += 1

                    result.symbols.append(symbol_id)

                except Exception as e:
                    error_msg = f"Error processing {symbol_id}: {str(e)}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)
                    result.skipped += 1

            self._save_symbols()
            logger.info(
                f"Import complete: {result.imported} imported, "
                f"{result.updated} updated, {result.skipped} skipped"
            )

        except Exception as e:
            error_msg = f"Import from EasyInsight API failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    async def import_from_timescaledb(self) -> SymbolImportResult:
        """Import all symbols from EasyInsight API (legacy name for compatibility)."""
        return await self.import_from_easyinsight()

    async def refresh_symbol_data(self, symbol_id: str) -> Optional[ManagedSymbol]:
        """Refresh data info for a specific symbol from EasyInsight API."""
        symbol_id = symbol_id.upper()
        symbol = self._symbols.get(symbol_id)

        if not symbol:
            return None

        await self._update_symbol_data_info(symbol)
        await self._check_nhits_model(symbol)
        symbol.updated_at = datetime.utcnow()

        self._symbols[symbol_id] = symbol
        self._save_symbols()

        return symbol

    async def get_stats(self) -> SymbolStats:
        """Get statistics about managed symbols."""
        symbols = list(self._symbols.values())

        by_category = {}
        for cat in SymbolCategory:
            count = len([s for s in symbols if s.category == cat])
            if count > 0:
                by_category[cat.value] = count

        return SymbolStats(
            total_symbols=len(symbols),
            active_symbols=len([s for s in symbols if s.status == SymbolStatus.ACTIVE]),
            inactive_symbols=len([s for s in symbols if s.status == SymbolStatus.INACTIVE]),
            suspended_symbols=len([s for s in symbols if s.status == SymbolStatus.SUSPENDED]),
            with_timescaledb_data=len([s for s in symbols if s.has_timescaledb_data]),
            with_nhits_model=len([s for s in symbols if s.has_nhits_model]),
            by_category=by_category,
            favorites_count=len([s for s in symbols if s.is_favorite]),
        )

    async def search_symbols(
        self,
        query: str,
        limit: int = 20,
    ) -> list[ManagedSymbol]:
        """Search symbols by name, description, or tags."""
        query = query.lower()
        results = []

        for symbol in self._symbols.values():
            score = 0

            # Exact match on symbol
            if symbol.symbol.lower() == query:
                score = 100
            # Symbol starts with query
            elif symbol.symbol.lower().startswith(query):
                score = 80
            # Symbol contains query
            elif query in symbol.symbol.lower():
                score = 60
            # Display name match
            elif symbol.display_name and query in symbol.display_name.lower():
                score = 50
            # Description match
            elif symbol.description and query in symbol.description.lower():
                score = 30
            # Tag match
            elif any(query in tag.lower() for tag in symbol.tags):
                score = 40

            if score > 0:
                results.append((score, symbol))

        # Sort by score descending, then by symbol name
        results.sort(key=lambda x: (-x[0], x[1].symbol))

        return [s for _, s in results[:limit]]


# Global service instance
symbol_service = SymbolService()
