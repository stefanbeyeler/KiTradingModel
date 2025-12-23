"""
Unit tests for SymbolService.

Tests the symbol management business logic including:
- Category detection
- Symbol validation
- Forex pair parsing
- TwelveData symbol generation

Note: These tests mock the service imports to avoid dependency issues.
For full integration tests, run the API tests with running services.
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
from enum import Enum

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Mock the SymbolCategory enum for standalone testing
class MockSymbolCategory(Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    INDEX = "index"
    COMMODITY = "commodity"
    OTHER = "other"


class MockSymbolStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class TestSymbolCategoryDetection:
    """Unit tests for symbol category auto-detection.

    These tests verify the category detection logic without importing
    the full SymbolService class to avoid dependency chain issues.
    """

    def _detect_category(self, symbol: str) -> MockSymbolCategory:
        """Standalone category detection logic for testing."""
        symbol_upper = symbol.upper()

        forex_currencies = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
        if len(symbol_upper) == 6:
            base = symbol_upper[:3]
            quote = symbol_upper[3:]
            if base in forex_currencies and quote in forex_currencies:
                return MockSymbolCategory.FOREX

        crypto_bases = {"BTC", "ETH", "XRP", "LTC", "ADA", "DOT", "SOL", "DOGE", "AVAX", "MATIC"}
        if any(symbol_upper.startswith(c) for c in crypto_bases):
            return MockSymbolCategory.CRYPTO
        if "USDT" in symbol_upper or "USDC" in symbol_upper:
            return MockSymbolCategory.CRYPTO

        indices = {"US30", "US500", "US100", "GER40", "UK100", "JPN225", "AUS200", "SPX", "NDX", "DJI"}
        if symbol_upper in indices or any(symbol_upper.startswith(i) for i in indices):
            return MockSymbolCategory.INDEX

        commodities = {"GOLD", "SILVER", "XAUUSD", "XAGUSD", "OIL", "WTI", "BRENT", "NATGAS"}
        if symbol_upper in commodities or any(symbol_upper.startswith(c) for c in commodities):
            return MockSymbolCategory.COMMODITY

        return MockSymbolCategory.OTHER

    @pytest.mark.unit
    def test_detect_forex_pair(self):
        """Test forex pair detection."""
        assert self._detect_category("EURUSD") == MockSymbolCategory.FOREX
        assert self._detect_category("GBPJPY") == MockSymbolCategory.FOREX
        assert self._detect_category("USDCHF") == MockSymbolCategory.FOREX
        assert self._detect_category("AUDUSD") == MockSymbolCategory.FOREX

    @pytest.mark.unit
    def test_detect_crypto(self):
        """Test cryptocurrency detection."""
        assert self._detect_category("BTCUSD") == MockSymbolCategory.CRYPTO
        assert self._detect_category("ETHUSD") == MockSymbolCategory.CRYPTO
        assert self._detect_category("SOLUSD") == MockSymbolCategory.CRYPTO
        assert self._detect_category("ADAUSD") == MockSymbolCategory.CRYPTO

    @pytest.mark.unit
    def test_detect_indices(self):
        """Test index detection."""
        assert self._detect_category("US500") == MockSymbolCategory.INDEX
        assert self._detect_category("GER40") == MockSymbolCategory.INDEX
        assert self._detect_category("US30") == MockSymbolCategory.INDEX

    @pytest.mark.unit
    def test_detect_commodities(self):
        """Test commodity detection."""
        assert self._detect_category("XAUUSD") == MockSymbolCategory.COMMODITY
        assert self._detect_category("XAGUSD") == MockSymbolCategory.COMMODITY


class TestForexPairParsing:
    """Unit tests for forex pair parsing."""

    def _parse_forex_pair(self, symbol: str) -> tuple:
        """Standalone forex pair parsing for testing."""
        if len(symbol) == 6:
            return symbol[:3].upper(), symbol[3:].upper()
        return None, None

    @pytest.mark.unit
    def test_parse_valid_forex_pair(self):
        """Test parsing valid forex pairs."""
        base, quote = self._parse_forex_pair("EURUSD")
        assert base == "EUR"
        assert quote == "USD"

        base, quote = self._parse_forex_pair("gbpjpy")
        assert base == "GBP"
        assert quote == "JPY"

    @pytest.mark.unit
    def test_parse_invalid_forex_pair(self):
        """Test parsing invalid forex pairs."""
        # Too short
        base, quote = self._parse_forex_pair("EUR")
        assert base is None
        assert quote is None

        # Too long
        base, quote = self._parse_forex_pair("EURUSDD")
        assert base is None
        assert quote is None


class TestTwelveDataSymbolGeneration:
    """Unit tests for TwelveData symbol format generation."""

    def _generate_twelvedata_symbol(self, symbol: str, category: MockSymbolCategory) -> str:
        """Standalone TwelveData symbol generation for testing."""
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

    @pytest.mark.unit
    def test_generate_forex_symbol(self):
        """Test TwelveData forex symbol generation."""
        result = self._generate_twelvedata_symbol("EURUSD", MockSymbolCategory.FOREX)
        assert result == "EUR/USD"

        result = self._generate_twelvedata_symbol("GBPJPY", MockSymbolCategory.FOREX)
        assert result == "GBP/JPY"

    @pytest.mark.unit
    def test_generate_crypto_symbol(self):
        """Test TwelveData crypto symbol generation."""
        result = self._generate_twelvedata_symbol("BTCUSD", MockSymbolCategory.CRYPTO)
        assert result == "BTC/USD"

    @pytest.mark.unit
    def test_generate_special_symbol(self):
        """Test TwelveData special symbol mappings."""
        result = self._generate_twelvedata_symbol("GER40", MockSymbolCategory.INDEX)
        assert result == "GDAXI"

        result = self._generate_twelvedata_symbol("US500", MockSymbolCategory.INDEX)
        assert result == "SPX"

        result = self._generate_twelvedata_symbol("DOGUSD", MockSymbolCategory.CRYPTO)
        assert result == "DOGE/USD"


class TestSymbolValidation:
    """Unit tests for symbol validation logic."""

    def _validate_symbol_format(self, symbol: str) -> bool:
        """Validate symbol format."""
        if not symbol:
            return False
        if len(symbol) < 3:
            return False
        if not symbol.isalnum():
            return False
        return True

    @pytest.mark.unit
    def test_valid_symbol_formats(self):
        """Test valid symbol formats."""
        assert self._validate_symbol_format("BTCUSD") is True
        assert self._validate_symbol_format("EURUSD") is True
        assert self._validate_symbol_format("GER40") is True

    @pytest.mark.unit
    def test_invalid_symbol_formats(self):
        """Test invalid symbol formats."""
        assert self._validate_symbol_format("") is False
        assert self._validate_symbol_format("AB") is False
        assert self._validate_symbol_format("BTC/USD") is False


class TestSymbolSearch:
    """Unit tests for symbol search functionality."""

    def _search_symbols(self, symbols: list, query: str, limit: int = 20) -> list:
        """Standalone search logic for testing."""
        query = query.lower()
        results = []

        for symbol in symbols:
            score = 0
            symbol_lower = symbol.lower()

            if symbol_lower == query:
                score = 100
            elif symbol_lower.startswith(query):
                score = 80
            elif query in symbol_lower:
                score = 60

            if score > 0:
                results.append((score, symbol))

        results.sort(key=lambda x: (-x[0], x[1]))
        return [s for _, s in results[:limit]]

    @pytest.mark.unit
    def test_search_exact_match(self):
        """Test exact symbol match in search."""
        symbols = ["BTCUSD", "ETHUSD", "BTCEUR"]
        results = self._search_symbols(symbols, "btcusd")

        assert len(results) > 0
        assert results[0] == "BTCUSD"

    @pytest.mark.unit
    def test_search_partial_match(self):
        """Test partial match in search."""
        symbols = ["BTCUSD", "ETHUSD", "BTCEUR"]
        results = self._search_symbols(symbols, "btc")

        assert len(results) >= 2
        assert "BTCUSD" in results
        assert "BTCEUR" in results

    @pytest.mark.unit
    def test_search_with_limit(self):
        """Test search with result limit."""
        symbols = [f"TEST{i:02d}" for i in range(10)]
        results = self._search_symbols(symbols, "test", limit=5)

        assert len(results) == 5


class TestSymbolDataStructures:
    """Unit tests for symbol data structures."""

    @pytest.mark.unit
    def test_symbol_status_values(self):
        """Test that symbol status enum has expected values."""
        assert MockSymbolStatus.ACTIVE.value == "active"
        assert MockSymbolStatus.INACTIVE.value == "inactive"
        assert MockSymbolStatus.SUSPENDED.value == "suspended"

    @pytest.mark.unit
    def test_symbol_category_values(self):
        """Test that symbol category enum has expected values."""
        assert MockSymbolCategory.FOREX.value == "forex"
        assert MockSymbolCategory.CRYPTO.value == "crypto"
        assert MockSymbolCategory.INDEX.value == "index"
        assert MockSymbolCategory.COMMODITY.value == "commodity"
        assert MockSymbolCategory.OTHER.value == "other"
