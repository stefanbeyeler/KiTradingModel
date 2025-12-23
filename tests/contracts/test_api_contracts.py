"""
API Contract tests for response schema validation.

Tests that API responses conform to expected schemas using Pydantic models.
"""
import pytest
import httpx
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Optional, Any
from datetime import datetime

from conftest import SERVICE_URLS


# ========== Response Schemas ==========

class HealthResponse(BaseModel):
    """Schema for health endpoint response."""
    status: str
    service: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    timestamp: Optional[str] = None
    uptime: Optional[float] = None

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        valid_statuses = ['healthy', 'ok', 'running', 'up']
        if v.lower() not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v


class SymbolResponse(BaseModel):
    """Schema for symbol data."""
    symbol: str
    display_name: Optional[str] = None
    category: Optional[str] = None
    exchange: Optional[str] = None
    enabled: Optional[bool] = None
    status: Optional[str] = None
    intervals: Optional[List[str]] = None


class OHLCVDataPoint(BaseModel):
    """Schema for a single OHLCV data point."""
    timestamp: Optional[str] = None
    time: Optional[str] = None
    datetime: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    @field_validator('open', 'high', 'low', 'close')
    @classmethod
    def validate_positive_price(cls, v):
        if v < 0:
            raise ValueError("Price values must be non-negative")
        return v


class ForecastResponse(BaseModel):
    """Schema for forecast response."""
    symbol: str
    predictions: Optional[List[float]] = None
    forecast: Optional[List[float]] = None
    horizon: Optional[int] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None


class RAGQueryResponse(BaseModel):
    """Schema for RAG query response."""
    results: Optional[List[Any]] = None
    query: Optional[str] = None
    total: Optional[int] = None


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


# ========== Contract Tests ==========

class TestHealthResponseContract:
    """Contract tests for health endpoint responses."""

    @pytest.mark.contract
    async def test_data_service_health_contract(self):
        """Data Service health response must match schema."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{SERVICE_URLS['data']}/health")

                if response.status_code == 200:
                    data = response.json()
                    try:
                        HealthResponse(**data)
                    except ValidationError as e:
                        pytest.fail(f"Data service health response invalid: {e}")
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.contract
    async def test_nhits_service_health_contract(self):
        """NHITS Service health response must match schema."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{SERVICE_URLS['nhits']}/health")

                if response.status_code == 200:
                    data = response.json()
                    try:
                        HealthResponse(**data)
                    except ValidationError as e:
                        pytest.fail(f"NHITS service health response invalid: {e}")
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.contract
    async def test_rag_service_health_contract(self):
        """RAG Service health response must match schema."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{SERVICE_URLS['rag']}/health")

                if response.status_code == 200:
                    data = response.json()
                    try:
                        HealthResponse(**data)
                    except ValidationError as e:
                        pytest.fail(f"RAG service health response invalid: {e}")
            except httpx.ConnectError:
                pytest.skip("RAG service not reachable")


class TestSymbolResponseContract:
    """Contract tests for symbol responses."""

    @pytest.mark.contract
    async def test_symbols_list_contract(self):
        """Symbol list response must match schema."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{SERVICE_URLS['data']}/api/v1/symbols")

                if response.status_code == 200:
                    data = response.json()
                    assert isinstance(data, list)

                    # Validate first 5 symbols
                    for item in data[:5]:
                        try:
                            SymbolResponse(**item)
                        except ValidationError as e:
                            pytest.fail(f"Symbol response invalid: {e}")
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestOHLCVResponseContract:
    """Contract tests for OHLCV data responses."""

    @pytest.mark.contract
    async def test_ohlcv_response_contract(self):
        """OHLCV response must match schema."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/BTCUSD",
                    params={"limit": 10}
                )

                if response.status_code == 200:
                    data = response.json()

                    # Handle both list and dict responses
                    items = data.get("data", data) if isinstance(data, dict) else data

                    if isinstance(items, list):
                        # Validate first 5 data points
                        for item in items[:5]:
                            try:
                                OHLCVDataPoint(**item)
                            except ValidationError as e:
                                pytest.fail(f"OHLCV response invalid: {e}")
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.contract
    async def test_ohlcv_price_values_positive(self):
        """OHLCV prices must be positive values."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/BTCUSD",
                    params={"limit": 10}
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("data", data) if isinstance(data, dict) else data

                    if isinstance(items, list):
                        for item in items:
                            assert item.get("open", 0) >= 0
                            assert item.get("high", 0) >= 0
                            assert item.get("low", 0) >= 0
                            assert item.get("close", 0) >= 0
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")

    @pytest.mark.contract
    async def test_ohlcv_high_low_consistency(self):
        """OHLCV high must be >= low."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/ohlcv/BTCUSD",
                    params={"limit": 10}
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("data", data) if isinstance(data, dict) else data

                    if isinstance(items, list):
                        for item in items:
                            high = item.get("high", 0)
                            low = item.get("low", 0)
                            assert high >= low, f"High ({high}) must be >= Low ({low})"
            except httpx.ConnectError:
                pytest.skip("Data service not reachable")


class TestForecastResponseContract:
    """Contract tests for forecast responses."""

    @pytest.mark.contract
    async def test_forecast_response_contract(self):
        """Forecast response must match schema."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={"symbol": "BTCUSD", "horizon": 24}
                )

                if response.status_code == 200:
                    data = response.json()
                    try:
                        ForecastResponse(**data)
                    except ValidationError as e:
                        pytest.fail(f"Forecast response invalid: {e}")
            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")


class TestErrorResponseContract:
    """Contract tests for error responses."""

    @pytest.mark.contract
    async def test_validation_error_contract(self):
        """Validation error response must be properly formatted."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Send invalid request
                response = await client.post(
                    f"{SERVICE_URLS['nhits']}/api/v1/forecast",
                    json={}  # Missing required fields
                )

                # Should return 422 for validation error
                assert response.status_code == 422

                data = response.json()
                # FastAPI validation error format
                assert "detail" in data

            except httpx.ConnectError:
                pytest.skip("NHITS service not reachable")

    @pytest.mark.contract
    async def test_not_found_error_contract(self):
        """Not found error response must be properly formatted."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{SERVICE_URLS['data']}/api/v1/nonexistent-endpoint"
                )

                assert response.status_code in [404, 405]

            except httpx.ConnectError:
                pytest.skip("Data service not reachable")
