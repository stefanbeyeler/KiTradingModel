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


class ManagedSymbol(BaseModel):
    """A managed trading symbol with metadata."""
    symbol: str = Field(..., description="Trading symbol identifier (e.g., EURUSD)")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    category: SymbolCategory = Field(default=SymbolCategory.FOREX, description="Symbol category")
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

    # User preferences
    is_favorite: bool = Field(default=False, description="User marked as favorite")
    notes: Optional[str] = Field(None, description="User notes about this symbol")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SymbolCreateRequest(BaseModel):
    """Request to create a new managed symbol."""
    symbol: str
    display_name: Optional[str] = None
    category: SymbolCategory = SymbolCategory.FOREX
    description: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    pip_value: Optional[float] = None
    min_lot_size: Optional[float] = 0.01
    max_lot_size: Optional[float] = 100.0
    notes: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class SymbolUpdateRequest(BaseModel):
    """Request to update an existing managed symbol."""
    display_name: Optional[str] = None
    category: Optional[SymbolCategory] = None
    status: Optional[SymbolStatus] = None
    description: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    pip_value: Optional[float] = None
    min_lot_size: Optional[float] = None
    max_lot_size: Optional[float] = None
    is_favorite: Optional[bool] = None
    notes: Optional[str] = None
    tags: Optional[list[str]] = None


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
    favorites_count: int
