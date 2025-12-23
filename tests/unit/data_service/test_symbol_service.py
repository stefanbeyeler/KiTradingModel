"""
Unit tests for SymbolService.

Tests the symbol management business logic including:
- Category detection
- Symbol validation
- Forex pair parsing
- TwelveData symbol generation
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestSymbolCategoryDetection:
    """Unit tests for symbol category auto-detection."""

    @pytest.mark.unit
    def test_detect_forex_pair(self):
        """Test forex pair detection."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import SymbolCategory

        # Mock the _load_symbols to avoid file I/O
        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Forex pairs
            assert service._detect_category("EURUSD") == SymbolCategory.FOREX
            assert service._detect_category("GBPJPY") == SymbolCategory.FOREX
            assert service._detect_category("USDCHF") == SymbolCategory.FOREX
            assert service._detect_category("AUDUSD") == SymbolCategory.FOREX

    @pytest.mark.unit
    def test_detect_crypto(self):
        """Test cryptocurrency detection."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import SymbolCategory

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Crypto pairs
            assert service._detect_category("BTCUSD") == SymbolCategory.CRYPTO
            assert service._detect_category("ETHUSD") == SymbolCategory.CRYPTO
            assert service._detect_category("SOLUSD") == SymbolCategory.CRYPTO
            assert service._detect_category("ADAUSD") == SymbolCategory.CRYPTO

    @pytest.mark.unit
    def test_detect_indices(self):
        """Test index detection."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import SymbolCategory

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Indices
            assert service._detect_category("US500") == SymbolCategory.INDEX
            assert service._detect_category("GER40") == SymbolCategory.INDEX
            assert service._detect_category("US30") == SymbolCategory.INDEX

    @pytest.mark.unit
    def test_detect_commodities(self):
        """Test commodity detection."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import SymbolCategory

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Commodities
            assert service._detect_category("XAUUSD") == SymbolCategory.COMMODITY
            assert service._detect_category("XAGUSD") == SymbolCategory.COMMODITY


class TestForexPairParsing:
    """Unit tests for forex pair parsing."""

    @pytest.mark.unit
    def test_parse_valid_forex_pair(self):
        """Test parsing valid forex pairs."""
        from src.services.symbol_service import SymbolService

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            base, quote = service._parse_forex_pair("EURUSD")
            assert base == "EUR"
            assert quote == "USD"

            base, quote = service._parse_forex_pair("gbpjpy")
            assert base == "GBP"
            assert quote == "JPY"

    @pytest.mark.unit
    def test_parse_invalid_forex_pair(self):
        """Test parsing invalid forex pairs."""
        from src.services.symbol_service import SymbolService

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Too short
            base, quote = service._parse_forex_pair("EUR")
            assert base is None
            assert quote is None

            # Too long
            base, quote = service._parse_forex_pair("EURUSDD")
            assert base is None
            assert quote is None


class TestTwelveDataSymbolGeneration:
    """Unit tests for TwelveData symbol format generation."""

    @pytest.mark.unit
    def test_generate_forex_symbol(self):
        """Test TwelveData forex symbol generation."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import SymbolCategory

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            result = service._generate_twelvedata_symbol("EURUSD", SymbolCategory.FOREX)
            assert result == "EUR/USD"

            result = service._generate_twelvedata_symbol("GBPJPY", SymbolCategory.FOREX)
            assert result == "GBP/JPY"

    @pytest.mark.unit
    def test_generate_crypto_symbol(self):
        """Test TwelveData crypto symbol generation."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import SymbolCategory

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            result = service._generate_twelvedata_symbol("BTCUSD", SymbolCategory.CRYPTO)
            assert result == "BTC/USD"

    @pytest.mark.unit
    def test_generate_special_symbol(self):
        """Test TwelveData special symbol mappings."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import SymbolCategory

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Special mappings
            result = service._generate_twelvedata_symbol("GER40", SymbolCategory.INDEX)
            assert result == "GDAXI"

            result = service._generate_twelvedata_symbol("US500", SymbolCategory.INDEX)
            assert result == "SPX"

            result = service._generate_twelvedata_symbol("DOGUSD", SymbolCategory.CRYPTO)
            assert result == "DOGE/USD"


class TestSymbolServiceCRUD:
    """Unit tests for symbol CRUD operations."""

    @pytest.mark.unit
    async def test_get_symbol_not_found(self):
        """Test getting a non-existent symbol."""
        from src.services.symbol_service import SymbolService

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()
            service._symbols = {}

            result = await service.get_symbol("NONEXISTENT")
            assert result is None

    @pytest.mark.unit
    async def test_get_symbol_case_insensitive(self):
        """Test symbol lookup is case-insensitive."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import ManagedSymbol, SymbolCategory, SymbolStatus

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Add a symbol
            test_symbol = ManagedSymbol(
                symbol="BTCUSD",
                display_name="Bitcoin USD",
                category=SymbolCategory.CRYPTO,
                status=SymbolStatus.ACTIVE
            )
            service._symbols["BTCUSD"] = test_symbol

            # Lookup with different cases
            result = await service.get_symbol("btcusd")
            assert result is not None
            assert result.symbol == "BTCUSD"

            result = await service.get_symbol("BtCuSd")
            assert result is not None

    @pytest.mark.unit
    async def test_toggle_favorite(self):
        """Test toggling favorite status."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import ManagedSymbol, SymbolCategory, SymbolStatus

        with patch.object(SymbolService, '_load_symbols'):
            with patch.object(SymbolService, '_save_symbols'):
                service = SymbolService()

                test_symbol = ManagedSymbol(
                    symbol="BTCUSD",
                    display_name="Bitcoin USD",
                    category=SymbolCategory.CRYPTO,
                    status=SymbolStatus.ACTIVE,
                    is_favorite=False
                )
                service._symbols["BTCUSD"] = test_symbol

                # Toggle on
                result = await service.toggle_favorite("BTCUSD")
                assert result is not None
                assert result.is_favorite is True

                # Toggle off
                result = await service.toggle_favorite("BTCUSD")
                assert result is not None
                assert result.is_favorite is False

    @pytest.mark.unit
    async def test_delete_symbol(self):
        """Test deleting a symbol."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import ManagedSymbol, SymbolCategory, SymbolStatus

        with patch.object(SymbolService, '_load_symbols'):
            with patch.object(SymbolService, '_save_symbols'):
                service = SymbolService()

                test_symbol = ManagedSymbol(
                    symbol="TESTXYZ",
                    display_name="Test Symbol",
                    category=SymbolCategory.OTHER,
                    status=SymbolStatus.ACTIVE
                )
                service._symbols["TESTXYZ"] = test_symbol

                # Delete existing
                result = await service.delete_symbol("TESTXYZ")
                assert result is True
                assert "TESTXYZ" not in service._symbols

                # Delete non-existent
                result = await service.delete_symbol("NONEXISTENT")
                assert result is False


class TestSymbolSearch:
    """Unit tests for symbol search functionality."""

    @pytest.mark.unit
    async def test_search_exact_match(self):
        """Test exact symbol match in search."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import ManagedSymbol, SymbolCategory, SymbolStatus

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            # Add test symbols
            for sym in ["BTCUSD", "ETHUSD", "BTCEUR"]:
                service._symbols[sym] = ManagedSymbol(
                    symbol=sym,
                    display_name=sym,
                    category=SymbolCategory.CRYPTO,
                    status=SymbolStatus.ACTIVE
                )

            results = await service.search_symbols("btcusd")
            assert len(results) > 0
            assert results[0].symbol == "BTCUSD"  # Exact match first

    @pytest.mark.unit
    async def test_search_partial_match(self):
        """Test partial match in search."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import ManagedSymbol, SymbolCategory, SymbolStatus

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            for sym in ["BTCUSD", "ETHUSD", "BTCEUR"]:
                service._symbols[sym] = ManagedSymbol(
                    symbol=sym,
                    display_name=sym,
                    category=SymbolCategory.CRYPTO,
                    status=SymbolStatus.ACTIVE
                )

            results = await service.search_symbols("btc")
            assert len(results) >= 2
            # BTC symbols should be found
            symbols = [r.symbol for r in results]
            assert "BTCUSD" in symbols
            assert "BTCEUR" in symbols

    @pytest.mark.unit
    async def test_search_with_limit(self):
        """Test search with result limit."""
        from src.services.symbol_service import SymbolService
        from src.models.symbol_data import ManagedSymbol, SymbolCategory, SymbolStatus

        with patch.object(SymbolService, '_load_symbols'):
            service = SymbolService()

            for i in range(10):
                sym = f"TEST{i:02d}"
                service._symbols[sym] = ManagedSymbol(
                    symbol=sym,
                    display_name=sym,
                    category=SymbolCategory.OTHER,
                    status=SymbolStatus.ACTIVE
                )

            results = await service.search_symbols("test", limit=5)
            assert len(results) == 5
