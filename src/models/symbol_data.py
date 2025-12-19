"""Pydantic models for symbol management."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class SymbolCategory(str, Enum):
    """Categories for trading symbols."""
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"
    ETF = "etf"
    OTHER = "other"


class SymbolStatus(str, Enum):
    """Status of a trading symbol."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class SymbolSubcategory(str, Enum):
    """Sub-categories for more granular symbol classification."""
    # Forex subcategories
    MAJOR = "major"           # Major currency pairs (EUR/USD, GBP/USD, etc.)
    MINOR = "minor"           # Minor currency pairs (EUR/GBP, etc.)
    EXOTIC = "exotic"         # Exotic currency pairs

    # Crypto subcategories
    LARGE_CAP = "large_cap"   # Top cryptocurrencies by market cap
    MID_CAP = "mid_cap"       # Mid-sized cryptocurrencies
    SMALL_CAP = "small_cap"   # Smaller cryptocurrencies
    DEFI = "defi"             # DeFi tokens
    MEME = "meme"             # Meme coins
    STABLECOIN = "stablecoin" # Stablecoins

    # Stock subcategories
    TECH = "tech"             # Technology stocks
    FINANCE = "finance"       # Financial sector
    HEALTHCARE = "healthcare" # Healthcare sector
    ENERGY = "energy"         # Energy sector
    CONSUMER = "consumer"     # Consumer goods/services
    INDUSTRIAL = "industrial" # Industrial sector

    # Index subcategories
    GLOBAL = "global"         # Global indices
    REGIONAL = "regional"     # Regional indices
    SECTOR = "sector"         # Sector-specific indices

    # Commodity subcategories
    PRECIOUS_METAL = "precious_metal"  # Gold, Silver, Platinum
    BASE_METAL = "base_metal"          # Copper, Aluminum, etc.
    AGRICULTURE = "agriculture"        # Wheat, Corn, Coffee, etc.
    ENERGY_COMMODITY = "energy_commodity"  # Oil, Natural Gas

    # General
    OTHER = "other"           # Uncategorized


class ManagedSymbol(BaseModel):
    """A managed trading symbol with metadata."""
    symbol: str = Field(..., description="Trading symbol identifier (e.g., EURUSD)")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    category: SymbolCategory = Field(default=SymbolCategory.FOREX, description="Symbol category")
    subcategory: Optional[SymbolSubcategory] = Field(None, description="Symbol subcategory for finer classification")
    status: SymbolStatus = Field(default=SymbolStatus.ACTIVE, description="Symbol status")
    description: Optional[str] = Field(None, description="Symbol description")

    # Trading metadata
    base_currency: Optional[str] = Field(None, description="Base currency (e.g., EUR)")
    quote_currency: Optional[str] = Field(None, description="Quote currency (e.g., USD)")
    pip_value: Optional[float] = Field(None, description="Pip value for this symbol")
    min_lot_size: Optional[float] = Field(default=0.01, description="Minimum lot size")
    max_lot_size: Optional[float] = Field(default=100.0, description="Maximum lot size")

    # Data availability from TimescaleDB
    has_timescaledb_data: bool = Field(default=False, description="Whether data exists in TimescaleDB")
    first_data_timestamp: Optional[datetime] = Field(None, description="First data point timestamp")
    last_data_timestamp: Optional[datetime] = Field(None, description="Last data point timestamp")
    total_records: Optional[int] = Field(None, description="Total records in TimescaleDB")

    # Model status
    has_nhits_model: bool = Field(default=False, description="Whether NHITS model exists")
    nhits_model_trained_at: Optional[datetime] = Field(None, description="When NHITS model was last trained")

    # External API symbol mappings
    twelvedata_symbol: Optional[str] = Field(None, description="Symbol format for Twelve Data API (e.g., BTC/USD, EUR/USD)")

    # User preferences
    is_favorite: bool = Field(default=False, description="User marked as favorite")
    notes: Optional[str] = Field(None, description="User notes about this symbol")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    aliases: list[str] = Field(default_factory=list, description="Alternative symbol names (e.g., BTC/USD for BTCUSD)")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SymbolCreateRequest(BaseModel):
    """Request to create a new managed symbol."""
    symbol: str
    display_name: Optional[str] = None
    category: SymbolCategory = SymbolCategory.FOREX
    subcategory: Optional[SymbolSubcategory] = None
    description: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    pip_value: Optional[float] = None
    min_lot_size: Optional[float] = 0.01
    max_lot_size: Optional[float] = 100.0
    twelvedata_symbol: Optional[str] = None
    notes: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class SymbolUpdateRequest(BaseModel):
    """Request to update an existing managed symbol."""
    display_name: Optional[str] = None
    category: Optional[SymbolCategory] = None
    subcategory: Optional[SymbolSubcategory] = None
    status: Optional[SymbolStatus] = None
    description: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    pip_value: Optional[float] = None
    min_lot_size: Optional[float] = None
    max_lot_size: Optional[float] = None
    twelvedata_symbol: Optional[str] = None
    is_favorite: Optional[bool] = None
    notes: Optional[str] = None
    tags: Optional[list[str]] = None
    aliases: Optional[list[str]] = None


class SymbolImportResult(BaseModel):
    """Result of importing symbols from TimescaleDB."""
    total_found: int = Field(..., description="Total symbols found in TimescaleDB")
    imported: int = Field(..., description="Number of symbols imported")
    updated: int = Field(..., description="Number of existing symbols updated")
    skipped: int = Field(..., description="Number of symbols skipped")
    errors: list[str] = Field(default_factory=list, description="Any errors during import")
    symbols: list[str] = Field(default_factory=list, description="List of imported/updated symbols")


class SymbolStats(BaseModel):
    """Statistics about managed symbols."""
    total_symbols: int
    active_symbols: int
    inactive_symbols: int
    suspended_symbols: int
    with_timescaledb_data: int
    with_nhits_model: int
    by_category: dict[str, int]
    by_subcategory: dict[str, int] = Field(default_factory=dict)
    favorites_count: int
# Force reload
