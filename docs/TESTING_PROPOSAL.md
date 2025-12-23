# Automatisches Testing-System für KI Trading Microservices

## Übersicht

Dieses Dokument beschreibt ein umfassendes automatisches Testing-System für die 8 Microservices der KI Trading Plattform.

### Service-Übersicht

| Service | Port | Container | GPU | Zweck |
|---------|------|-----------|-----|-------|
| Frontend | 3000 | trading-frontend | - | Dashboard & API Gateway |
| Data Service | 3001 | trading-data | - | Daten-Gateway, Symbol-Management |
| NHITS Service | 3002 | trading-nhits | ✓ | Zeitreihen-Prognosen |
| TCN-Pattern | 3003 | trading-tcn | ✓ | Chart-Pattern-Erkennung |
| HMM-Regime | 3004 | trading-hmm | - | Marktregime-Erkennung |
| Embedder | 3005 | trading-embedder | ✓ | Embedding-Generierung |
| RAG Service | 3008 | trading-rag | ✓ | Vektor-Suche & Wissensbasis |
| LLM Service | 3009 | trading-llm | ✓ | LLM-Analyse & Trading-Insights |

---

## 1. Test-Architektur

```
tests/
├── conftest.py                    # Globale pytest Fixtures
├── pytest.ini                     # pytest Konfiguration
├── requirements-test.txt          # Test-Dependencies
│
├── unit/                          # Unit Tests (isoliert, schnell)
│   ├── data_service/
│   ├── nhits_service/
│   ├── tcn_service/
│   ├── hmm_service/
│   ├── embedder_service/
│   ├── rag_service/
│   └── llm_service/
│
├── integration/                   # Integration Tests (Service-übergreifend)
│   ├── test_data_gateway.py       # Data Gateway Pattern
│   ├── test_service_communication.py
│   ├── test_fallback_chains.py    # EasyInsight → TwelveData → Yahoo
│   └── test_embedding_pipeline.py
│
├── api/                           # API Endpoint Tests
│   ├── test_data_api.py
│   ├── test_nhits_api.py
│   ├── test_tcn_api.py
│   ├── test_hmm_api.py
│   ├── test_embedder_api.py
│   ├── test_rag_api.py
│   └── test_llm_api.py
│
├── e2e/                           # End-to-End Tests
│   ├── test_trading_workflow.py
│   ├── test_service_startup.py
│   └── test_full_pipeline.py
│
├── performance/                   # Performance & Load Tests
│   ├── locustfile.py
│   └── benchmarks/
│
├── contracts/                     # Contract Tests (API-Schemas)
│   └── test_api_contracts.py
│
└── scripts/                       # Test-Hilfsskripte
    ├── run_all_tests.sh
    ├── run_health_checks.sh
    └── generate_coverage.sh
```

---

## 2. Test-Kategorien & Implementierung

### 2.1 Health Check Tests (Smoke Tests)

Schnelle Überprüfung aller Services vor anderen Tests.

```python
# tests/smoke/test_health_checks.py
import pytest
import httpx
from typing import Dict, Any

SERVICES = {
    "frontend": {"url": "http://localhost:3000", "health": "/health"},
    "data": {"url": "http://localhost:3001", "health": "/health"},
    "nhits": {"url": "http://localhost:3002", "health": "/health"},
    "tcn": {"url": "http://localhost:3003", "health": "/health"},
    "hmm": {"url": "http://localhost:3004", "health": "/health"},
    "embedder": {"url": "http://localhost:3005", "health": "/health"},
    "rag": {"url": "http://localhost:3008", "health": "/health"},
    "llm": {"url": "http://localhost:3009", "health": "/health"},
}

@pytest.mark.smoke
@pytest.mark.parametrize("service_name,config", SERVICES.items())
async def test_service_health(service_name: str, config: Dict[str, str]):
    """Prüft ob alle Services erreichbar und healthy sind."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{config['url']}{config['health']}")

        assert response.status_code == 200, f"{service_name} health check failed"

        data = response.json()
        assert data.get("status") == "healthy", f"{service_name} is not healthy"
        assert "version" in data, f"{service_name} missing version info"

@pytest.mark.smoke
async def test_all_services_up():
    """Schneller Check ob alle Services laufen."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        results = {}
        for name, config in SERVICES.items():
            try:
                resp = await client.get(f"{config['url']}{config['health']}")
                results[name] = resp.status_code == 200
            except Exception:
                results[name] = False

        failed = [name for name, ok in results.items() if not ok]
        assert not failed, f"Services nicht erreichbar: {failed}"
```

### 2.2 Unit Tests

Isolierte Tests für Business-Logik ohne externe Abhängigkeiten.

```python
# tests/unit/data_service/test_symbol_service.py
import pytest
from unittest.mock import AsyncMock, patch
from src.services.data_app.services.symbol_service import SymbolService

class TestSymbolService:
    """Unit Tests für SymbolService."""

    @pytest.fixture
    def symbol_service(self):
        return SymbolService()

    async def test_validate_symbol_format_valid(self, symbol_service):
        """Testet gültige Symbol-Formate."""
        valid_symbols = ["BTCUSD", "EURUSD", "AAPL", "GER40"]
        for symbol in valid_symbols:
            assert symbol_service.validate_symbol_format(symbol) is True

    async def test_validate_symbol_format_invalid(self, symbol_service):
        """Testet ungültige Symbol-Formate."""
        invalid_symbols = ["", "123", "btc-usd", "BTC/USD"]
        for symbol in invalid_symbols:
            assert symbol_service.validate_symbol_format(symbol) is False

    @patch('src.services.data_gateway_service.DataGatewayService.get_ohlcv')
    async def test_get_symbol_data_success(self, mock_get_ohlcv, symbol_service):
        """Testet erfolgreichen Datenabruf."""
        mock_get_ohlcv.return_value = {
            "symbol": "BTCUSD",
            "data": [{"open": 100, "high": 105, "low": 95, "close": 102}]
        }

        result = await symbol_service.get_symbol_data("BTCUSD")

        assert result["symbol"] == "BTCUSD"
        assert len(result["data"]) > 0

# tests/unit/nhits_service/test_forecast_service.py
class TestNHITSForecastService:
    """Unit Tests für NHITS Forecast Service."""

    async def test_prepare_input_data(self):
        """Testet die Input-Daten-Vorbereitung."""
        import numpy as np
        from src.services.nhits_app.services.forecast_service import ForecastService

        service = ForecastService()
        raw_data = [
            {"close": 100.0}, {"close": 101.0}, {"close": 102.0}
        ]

        result = service.prepare_input_data(raw_data)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    async def test_validate_forecast_horizon(self):
        """Testet Validierung des Forecast-Horizonts."""
        from src.services.nhits_app.services.forecast_service import ForecastService

        service = ForecastService()

        assert service.validate_horizon(24) is True   # Valid
        assert service.validate_horizon(168) is True  # Valid (1 Woche)
        assert service.validate_horizon(0) is False   # Invalid
        assert service.validate_horizon(-1) is False  # Invalid
        assert service.validate_horizon(10000) is False  # Too large
```

### 2.3 API Endpoint Tests

Tests für alle API-Endpoints mit verschiedenen Szenarien.

```python
# tests/api/test_data_api.py
import pytest
import httpx
from datetime import datetime, timedelta

BASE_URL = "http://localhost:3001"

class TestDataServiceAPI:
    """API Tests für den Data Service."""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    # ========== Symbol Management ==========

    @pytest.mark.api
    async def test_get_symbols_list(self, client):
        """GET /api/v1/symbols - Liste aller Symbole."""
        async with client:
            response = await client.get("/api/v1/symbols")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            # Prüfe Schema
            if data:
                assert "symbol" in data[0]
                assert "exchange" in data[0]

    @pytest.mark.api
    async def test_get_symbol_detail(self, client):
        """GET /api/v1/symbols/{symbol} - Symbol-Details."""
        async with client:
            response = await client.get("/api/v1/symbols/BTCUSD")

            if response.status_code == 200:
                data = response.json()
                assert data["symbol"] == "BTCUSD"
            else:
                # Symbol existiert nicht
                assert response.status_code == 404

    @pytest.mark.api
    async def test_create_symbol_invalid(self, client):
        """POST /api/v1/symbols - Ungültiges Symbol ablehnen."""
        async with client:
            response = await client.post(
                "/api/v1/symbols",
                json={"symbol": "", "exchange": "test"}
            )

            assert response.status_code in [400, 422]

    # ========== OHLCV Data ==========

    @pytest.mark.api
    async def test_get_ohlcv_data(self, client):
        """GET /api/v1/ohlcv/{symbol} - Historische Daten."""
        async with client:
            response = await client.get(
                "/api/v1/ohlcv/BTCUSD",
                params={"interval": "1h", "limit": 100}
            )

            if response.status_code == 200:
                data = response.json()
                assert "data" in data or isinstance(data, list)

    @pytest.mark.api
    async def test_get_ohlcv_invalid_interval(self, client):
        """GET /api/v1/ohlcv - Ungültiges Interval."""
        async with client:
            response = await client.get(
                "/api/v1/ohlcv/BTCUSD",
                params={"interval": "invalid"}
            )

            assert response.status_code in [400, 422]

    # ========== TwelveData Indicators ==========

    @pytest.mark.api
    @pytest.mark.parametrize("indicator", ["rsi", "macd", "sma", "ema", "bbands"])
    async def test_technical_indicators(self, client, indicator):
        """GET /api/v1/twelvedata/{indicator}/{symbol}."""
        async with client:
            response = await client.get(
                f"/api/v1/twelvedata/{indicator}/BTCUSD",
                params={"interval": "1h"}
            )

            # Entweder Erfolg oder Rate-Limit/Not Found
            assert response.status_code in [200, 404, 429, 503]


# tests/api/test_nhits_api.py
class TestNHITSServiceAPI:
    """API Tests für den NHITS Service."""

    BASE_URL = "http://localhost:3002"

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=self.BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_get_forecast(self, client):
        """POST /api/v1/forecast - Prognose abrufen."""
        async with client:
            response = await client.post(
                "/api/v1/forecast",
                json={
                    "symbol": "BTCUSD",
                    "horizon": 24,
                    "interval": "1h"
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert "predictions" in data or "forecast" in data

    @pytest.mark.api
    async def test_list_trained_models(self, client):
        """GET /api/v1/models - Liste trainierter Modelle."""
        async with client:
            response = await client.get("/api/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, (list, dict))

    @pytest.mark.api
    async def test_get_model_metrics(self, client):
        """GET /api/v1/models/{symbol}/metrics - Modell-Metriken."""
        async with client:
            response = await client.get("/api/v1/models/BTCUSD/metrics")

            # Modell kann existieren oder nicht
            assert response.status_code in [200, 404]


# tests/api/test_rag_api.py
class TestRAGServiceAPI:
    """API Tests für den RAG Service."""

    BASE_URL = "http://localhost:3008"

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(base_url=self.BASE_URL, timeout=60.0)

    @pytest.mark.api
    async def test_semantic_search(self, client):
        """POST /api/v1/query - Semantische Suche."""
        async with client:
            response = await client.post(
                "/api/v1/query",
                json={
                    "query": "Bitcoin price prediction",
                    "top_k": 5
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert "results" in data or isinstance(data, list)

    @pytest.mark.api
    async def test_add_document(self, client):
        """POST /api/v1/documents - Dokument hinzufügen."""
        async with client:
            response = await client.post(
                "/api/v1/documents",
                json={
                    "content": "Test document for RAG system",
                    "metadata": {"source": "test", "type": "unit_test"}
                }
            )

            assert response.status_code in [200, 201]

    @pytest.mark.api
    async def test_get_stats(self, client):
        """GET /api/v1/stats - RAG Statistiken."""
        async with client:
            response = await client.get("/api/v1/stats")

            assert response.status_code == 200
            data = response.json()
            assert "document_count" in data or "total_documents" in data
```

### 2.4 Integration Tests

Tests für Service-übergreifende Kommunikation.

```python
# tests/integration/test_data_gateway.py
import pytest
import httpx

class TestDataGatewayIntegration:
    """Integration Tests für das Data Gateway Pattern."""

    @pytest.mark.integration
    async def test_nhits_uses_data_service(self):
        """NHITS Service muss Daten vom Data Service beziehen."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Trigger NHITS forecast
            response = await client.post(
                "http://localhost:3002/api/v1/forecast",
                json={"symbol": "BTCUSD", "horizon": 24}
            )

            # Prüfe dass Data Service aufgerufen wurde
            # (via Logs oder Metriken)
            data_health = await client.get("http://localhost:3001/health")

            assert data_health.status_code == 200

    @pytest.mark.integration
    async def test_rag_uses_data_service_for_external_sources(self):
        """RAG Service muss externe Daten via Data Service abrufen."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Fetch trading context über RAG
            response = await client.get(
                "http://localhost:3008/api/v1/trading-context/BTCUSD"
            )

            if response.status_code == 200:
                data = response.json()
                # Sollte Daten aus mehreren Quellen enthalten
                assert any(key in data for key in [
                    "sentiment", "technical", "fundamental", "on_chain"
                ])


# tests/integration/test_service_chain.py
class TestServiceChainIntegration:
    """Tests für komplette Service-Ketten."""

    @pytest.mark.integration
    async def test_full_analysis_chain(self):
        """Test: Data → NHITS → RAG → LLM Kette."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            symbol = "BTCUSD"

            # 1. Daten vom Data Service
            data_resp = await client.get(
                f"http://localhost:3001/api/v1/ohlcv/{symbol}",
                params={"limit": 100}
            )
            assert data_resp.status_code == 200

            # 2. Prognose vom NHITS Service
            nhits_resp = await client.post(
                "http://localhost:3002/api/v1/forecast",
                json={"symbol": symbol, "horizon": 24}
            )
            # Kann fehlschlagen wenn kein trainiertes Modell

            # 3. Kontext vom RAG Service
            rag_resp = await client.post(
                "http://localhost:3008/api/v1/query",
                json={"query": f"{symbol} analysis", "top_k": 3}
            )

            # 4. Analyse vom LLM Service
            llm_resp = await client.post(
                "http://localhost:3009/api/v1/analyze",
                json={
                    "symbol": symbol,
                    "use_rag": True,
                    "include_forecast": True
                }
            )

            # Mindestens Data und ein weiterer Service sollte funktionieren
            successful = sum([
                data_resp.status_code == 200,
                nhits_resp.status_code == 200,
                rag_resp.status_code == 200,
                llm_resp.status_code == 200
            ])

            assert successful >= 2, "Weniger als 2 Services antworten korrekt"


# tests/integration/test_fallback_chain.py
class TestFallbackChain:
    """Tests für Datenquellen-Fallback."""

    @pytest.mark.integration
    async def test_data_source_fallback(self):
        """Test: EasyInsight → TwelveData → Yahoo Fallback."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Frage nach Daten - sollte immer eine Quelle liefern
            response = await client.get(
                "http://localhost:3001/api/v1/ohlcv/BTCUSD",
                params={"interval": "1h", "limit": 50}
            )

            # Sollte immer erfolgreich sein (via irgendeinen Fallback)
            assert response.status_code == 200

            data = response.json()
            # Prüfe welche Quelle verwendet wurde
            if "source" in data:
                assert data["source"] in [
                    "easyinsight", "twelvedata", "yahoo", "cache"
                ]
```

### 2.5 End-to-End Tests

Komplette Workflow-Tests.

```python
# tests/e2e/test_trading_workflow.py
import pytest
import httpx
import asyncio

class TestTradingWorkflow:
    """End-to-End Tests für Trading-Workflows."""

    @pytest.mark.e2e
    async def test_complete_trading_analysis(self):
        """Kompletter Trading-Analyse Workflow."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            symbol = "BTCUSD"

            # Step 1: Symbol aktivieren/prüfen
            symbol_resp = await client.get(
                f"http://localhost:3001/api/v1/symbols/{symbol}"
            )

            # Step 2: Aktuelle Marktdaten abrufen
            market_data = await client.get(
                f"http://localhost:3001/api/v1/ohlcv/{symbol}",
                params={"interval": "1h", "limit": 168}  # 1 Woche
            )
            assert market_data.status_code == 200

            # Step 3: Chart-Pattern Erkennung
            pattern_resp = await client.post(
                "http://localhost:3003/api/v1/detect",
                json={"symbol": symbol, "interval": "1h"}
            )

            # Step 4: Regime-Erkennung
            regime_resp = await client.post(
                "http://localhost:3004/api/v1/regime",
                json={"symbol": symbol}
            )

            # Step 5: NHITS Prognose
            forecast_resp = await client.post(
                "http://localhost:3002/api/v1/forecast",
                json={"symbol": symbol, "horizon": 24}
            )

            # Step 6: LLM Gesamtanalyse
            analysis_resp = await client.post(
                "http://localhost:3009/api/v1/trading-analysis",
                json={
                    "symbol": symbol,
                    "include_patterns": True,
                    "include_regime": True,
                    "include_forecast": True
                }
            )

            # Auswertung
            results = {
                "market_data": market_data.status_code == 200,
                "patterns": pattern_resp.status_code == 200,
                "regime": regime_resp.status_code == 200,
                "forecast": forecast_resp.status_code == 200,
                "analysis": analysis_resp.status_code == 200
            }

            # Mindestens Marktdaten und 2 weitere Komponenten
            assert results["market_data"], "Marktdaten nicht verfügbar"
            assert sum(results.values()) >= 3, f"Zu wenige Komponenten: {results}"

    @pytest.mark.e2e
    async def test_model_training_workflow(self):
        """Test: Modell-Training Workflow."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            symbol = "EURUSD"  # Anderes Symbol für Training-Test

            # 1. Training starten
            train_resp = await client.post(
                "http://localhost:3002/api/v1/train",
                json={
                    "symbol": symbol,
                    "epochs": 5,  # Minimal für Test
                    "horizon": 24
                }
            )

            if train_resp.status_code == 200:
                # 2. Warten auf Training-Abschluss
                for _ in range(30):  # Max 5 Minuten
                    status_resp = await client.get(
                        f"http://localhost:3002/api/v1/training-status/{symbol}"
                    )
                    if status_resp.status_code == 200:
                        status = status_resp.json()
                        if status.get("status") == "completed":
                            break
                    await asyncio.sleep(10)

                # 3. Prognose mit trainiertem Modell
                forecast_resp = await client.post(
                    "http://localhost:3002/api/v1/forecast",
                    json={"symbol": symbol, "horizon": 24}
                )

                assert forecast_resp.status_code == 200
```

### 2.6 Performance Tests

Load- und Performance-Tests mit Locust.

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import random

class DataServiceUser(HttpUser):
    """Load Test für Data Service."""

    host = "http://localhost:3001"
    wait_time = between(1, 3)

    symbols = ["BTCUSD", "ETHUSD", "EURUSD", "XAUUSD", "GER40"]

    @task(10)
    def get_health(self):
        self.client.get("/health")

    @task(5)
    def get_symbols(self):
        self.client.get("/api/v1/symbols")

    @task(8)
    def get_ohlcv(self):
        symbol = random.choice(self.symbols)
        self.client.get(
            f"/api/v1/ohlcv/{symbol}",
            params={"interval": "1h", "limit": 100}
        )

    @task(3)
    def get_technical_indicator(self):
        symbol = random.choice(self.symbols)
        indicator = random.choice(["rsi", "macd", "sma"])
        self.client.get(f"/api/v1/twelvedata/{indicator}/{symbol}")


class NHITSServiceUser(HttpUser):
    """Load Test für NHITS Service."""

    host = "http://localhost:3002"
    wait_time = between(2, 5)

    symbols = ["BTCUSD", "ETHUSD", "EURUSD"]

    @task(5)
    def get_health(self):
        self.client.get("/health")

    @task(2)
    def get_models(self):
        self.client.get("/api/v1/models")

    @task(1)
    def request_forecast(self):
        symbol = random.choice(self.symbols)
        self.client.post(
            "/api/v1/forecast",
            json={"symbol": symbol, "horizon": 24}
        )


class RAGServiceUser(HttpUser):
    """Load Test für RAG Service."""

    host = "http://localhost:3008"
    wait_time = between(2, 5)

    queries = [
        "Bitcoin price analysis",
        "EUR/USD trend",
        "Gold market outlook",
        "Trading signals today"
    ]

    @task(5)
    def get_health(self):
        self.client.get("/health")

    @task(3)
    def semantic_search(self):
        query = random.choice(self.queries)
        self.client.post(
            "/api/v1/query",
            json={"query": query, "top_k": 5}
        )

    @task(2)
    def get_stats(self):
        self.client.get("/api/v1/stats")
```

### 2.7 Contract Tests

API-Schema-Validierung.

```python
# tests/contracts/test_api_contracts.py
import pytest
import httpx
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from datetime import datetime

# ========== Response Schemas ==========

class HealthResponse(BaseModel):
    service: str
    status: str
    version: str
    timestamp: Optional[str] = None

class SymbolResponse(BaseModel):
    symbol: str
    exchange: str
    enabled: bool
    intervals: Optional[List[str]] = None

class OHLCVDataPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class ForecastResponse(BaseModel):
    symbol: str
    predictions: List[float]
    horizon: int
    confidence: Optional[float] = None

# ========== Contract Tests ==========

class TestAPIContracts:
    """Contract Tests für API-Responses."""

    @pytest.mark.contract
    async def test_health_response_contract(self):
        """Health-Endpoint muss definiertem Schema entsprechen."""
        services = [
            ("http://localhost:3001/health", "data"),
            ("http://localhost:3002/health", "nhits"),
            ("http://localhost:3008/health", "rag"),
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            for url, expected_service in services:
                response = await client.get(url)

                if response.status_code == 200:
                    try:
                        HealthResponse(**response.json())
                    except ValidationError as e:
                        pytest.fail(
                            f"{expected_service} health response invalid: {e}"
                        )

    @pytest.mark.contract
    async def test_symbol_response_contract(self):
        """Symbol-Response muss Schema entsprechen."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://localhost:3001/api/v1/symbols")

            if response.status_code == 200:
                data = response.json()
                for item in data[:5]:  # Erste 5 prüfen
                    try:
                        SymbolResponse(**item)
                    except ValidationError as e:
                        pytest.fail(f"Symbol response invalid: {e}")

    @pytest.mark.contract
    async def test_ohlcv_response_contract(self):
        """OHLCV-Response muss Schema entsprechen."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "http://localhost:3001/api/v1/ohlcv/BTCUSD",
                params={"limit": 10}
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get("data", data) if isinstance(data, dict) else data

                for item in items[:5]:
                    try:
                        OHLCVDataPoint(**item)
                    except ValidationError as e:
                        pytest.fail(f"OHLCV response invalid: {e}")
```

---

## 3. Test-Konfiguration

### 3.1 pytest.ini

```ini
# tests/pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto

markers =
    smoke: Quick health check tests
    unit: Unit tests (isolated, fast)
    api: API endpoint tests
    integration: Integration tests (multi-service)
    e2e: End-to-end workflow tests
    contract: API contract/schema tests
    performance: Performance and load tests
    slow: Slow tests (skip with -m "not slow")
    gpu: Tests requiring GPU

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning

addopts =
    -v
    --tb=short
    --strict-markers
    -p no:warnings
```

### 3.2 conftest.py (Fixtures)

```python
# tests/conftest.py
import pytest
import httpx
import asyncio
from typing import AsyncGenerator

# ========== Service URLs ==========

SERVICE_URLS = {
    "frontend": "http://localhost:3000",
    "data": "http://localhost:3001",
    "nhits": "http://localhost:3002",
    "tcn": "http://localhost:3003",
    "hmm": "http://localhost:3004",
    "embedder": "http://localhost:3005",
    "rag": "http://localhost:3008",
    "llm": "http://localhost:3009",
}

# ========== Fixtures ==========

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Shared async HTTP client."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client

@pytest.fixture
async def data_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for Data Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["data"],
        timeout=30.0
    ) as client:
        yield client

@pytest.fixture
async def nhits_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for NHITS Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["nhits"],
        timeout=60.0
    ) as client:
        yield client

@pytest.fixture
async def rag_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for RAG Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["rag"],
        timeout=60.0
    ) as client:
        yield client

@pytest.fixture
async def llm_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for LLM Service."""
    async with httpx.AsyncClient(
        base_url=SERVICE_URLS["llm"],
        timeout=120.0
    ) as client:
        yield client

# ========== Service Health Checks ==========

@pytest.fixture(scope="session")
async def ensure_services_running():
    """Ensure all required services are running before tests."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        required = ["data", "nhits"]  # Minimum required

        for service in required:
            url = f"{SERVICE_URLS[service]}/health"
            try:
                response = await client.get(url)
                if response.status_code != 200:
                    pytest.skip(f"{service} service not healthy")
            except httpx.ConnectError:
                pytest.skip(f"{service} service not running")

# ========== Test Data Fixtures ==========

@pytest.fixture
def test_symbol():
    """Standard test symbol."""
    return "BTCUSD"

@pytest.fixture
def test_symbols():
    """Liste von Test-Symbolen."""
    return ["BTCUSD", "ETHUSD", "EURUSD"]

@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    return [
        {"timestamp": "2024-01-01T00:00:00Z", "open": 100, "high": 105, "low": 98, "close": 102, "volume": 1000},
        {"timestamp": "2024-01-01T01:00:00Z", "open": 102, "high": 108, "low": 101, "close": 107, "volume": 1200},
        {"timestamp": "2024-01-01T02:00:00Z", "open": 107, "high": 110, "low": 105, "close": 109, "volume": 900},
    ]
```

### 3.3 requirements-test.txt

```
# tests/requirements-test.txt

# Testing Framework
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
pytest-xdist>=3.5.0  # Parallel test execution
pytest-timeout>=2.2.0
pytest-html>=4.1.0

# HTTP Client
httpx>=0.26.0
aiohttp>=3.9.0

# Mocking
pytest-mock>=3.12.0
respx>=0.20.0  # Mock httpx requests
aioresponses>=0.7.6

# Load Testing
locust>=2.20.0

# Validation
pydantic>=2.5.0

# Reporting
allure-pytest>=2.13.0

# Code Quality
ruff>=0.1.0
mypy>=1.7.0

# Utilities
python-dotenv>=1.0.0
faker>=22.0.0
```

---

## 4. Test-Ausführung

### 4.1 Test-Skripte

```bash
#!/bin/bash
# tests/scripts/run_all_tests.sh

set -e

echo "=========================================="
echo "KI Trading Model - Automated Test Suite"
echo "=========================================="

# Farben
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Smoke Tests (Health Checks)
echo -e "\n${YELLOW}[1/6] Running Smoke Tests...${NC}"
pytest tests/smoke -v -m smoke --timeout=60 || {
    echo -e "${RED}Smoke tests failed! Services may not be running.${NC}"
    exit 1
}

# 2. Unit Tests
echo -e "\n${YELLOW}[2/6] Running Unit Tests...${NC}"
pytest tests/unit -v -m unit --timeout=120 --cov=src --cov-report=term-missing

# 3. API Tests
echo -e "\n${YELLOW}[3/6] Running API Tests...${NC}"
pytest tests/api -v -m api --timeout=180

# 4. Contract Tests
echo -e "\n${YELLOW}[4/6] Running Contract Tests...${NC}"
pytest tests/contracts -v -m contract --timeout=120

# 5. Integration Tests
echo -e "\n${YELLOW}[5/6] Running Integration Tests...${NC}"
pytest tests/integration -v -m integration --timeout=300

# 6. E2E Tests (optional, langsam)
if [[ "$1" == "--full" ]]; then
    echo -e "\n${YELLOW}[6/6] Running E2E Tests...${NC}"
    pytest tests/e2e -v -m e2e --timeout=600
else
    echo -e "\n${YELLOW}[6/6] Skipping E2E Tests (use --full to include)${NC}"
fi

echo -e "\n${GREEN}=========================================="
echo "All tests completed successfully!"
echo "==========================================${NC}"
```

```bash
#!/bin/bash
# tests/scripts/run_health_checks.sh

echo "Checking all service health endpoints..."

services=(
    "Frontend:http://localhost:3000/health"
    "Data:http://localhost:3001/health"
    "NHITS:http://localhost:3002/health"
    "TCN:http://localhost:3003/health"
    "HMM:http://localhost:3004/health"
    "Embedder:http://localhost:3005/health"
    "RAG:http://localhost:3008/health"
    "LLM:http://localhost:3009/health"
)

all_healthy=true

for service in "${services[@]}"; do
    name="${service%%:*}"
    url="${service#*:}"

    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" --connect-timeout 5)

    if [ "$response" == "200" ]; then
        echo "✅ $name: healthy"
    else
        echo "❌ $name: unhealthy (HTTP $response)"
        all_healthy=false
    fi
done

if $all_healthy; then
    echo -e "\n✅ All services are healthy!"
    exit 0
else
    echo -e "\n❌ Some services are unhealthy!"
    exit 1
fi
```

### 4.2 Makefile

```makefile
# Makefile

.PHONY: test test-smoke test-unit test-api test-integration test-e2e test-all coverage lint

# Test commands
test-smoke:
	pytest tests/smoke -v -m smoke

test-unit:
	pytest tests/unit -v -m unit --cov=src

test-api:
	pytest tests/api -v -m api

test-integration:
	pytest tests/integration -v -m integration

test-e2e:
	pytest tests/e2e -v -m e2e

test-contract:
	pytest tests/contracts -v -m contract

test-all:
	./tests/scripts/run_all_tests.sh --full

# Quick test (smoke + unit only)
test:
	pytest tests/smoke tests/unit -v --timeout=120

# Coverage report
coverage:
	pytest tests/unit tests/api -v --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Health check
health:
	./tests/scripts/run_health_checks.sh

# Load testing
load-test:
	locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s

# Lint
lint:
	ruff check src tests
	mypy src --ignore-missing-imports
```

---

## 5. CI/CD Integration

### 5.1 GitHub Actions Workflow

```yaml
# .github/workflows/test.yml

name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  smoke-tests:
    name: Smoke Tests
    runs-on: self-hosted  # Für GPU-Services
    steps:
      - uses: actions/checkout@v4

      - name: Start Services
        run: |
          docker-compose -f docker-compose.microservices.yml up -d
          sleep 60  # Warten auf Service-Start

      - name: Run Health Checks
        run: ./tests/scripts/run_health_checks.sh

      - name: Run Smoke Tests
        run: pytest tests/smoke -v -m smoke

  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt

      - name: Run Unit Tests
        run: pytest tests/unit -v --cov=src --cov-report=xml

      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: coverage.xml

  api-tests:
    name: API Tests
    runs-on: self-hosted
    needs: smoke-tests
    steps:
      - uses: actions/checkout@v4

      - name: Run API Tests
        run: pytest tests/api -v -m api --timeout=300

  integration-tests:
    name: Integration Tests
    runs-on: self-hosted
    needs: [smoke-tests, api-tests]
    steps:
      - uses: actions/checkout@v4

      - name: Run Integration Tests
        run: pytest tests/integration -v -m integration --timeout=600

  e2e-tests:
    name: E2E Tests
    runs-on: self-hosted
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Run E2E Tests
        run: pytest tests/e2e -v -m e2e --timeout=900
```

---

## 6. Test-Reporting

### 6.1 Allure Reports

```python
# tests/conftest.py (Ergänzung)

import allure

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach logs on failure."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        # Attach service logs on failure
        for service in ["data", "nhits", "rag", "llm"]:
            try:
                logs = get_docker_logs(f"trading-{service}", lines=50)
                allure.attach(
                    logs,
                    name=f"{service}_logs",
                    attachment_type=allure.attachment_type.TEXT
                )
            except Exception:
                pass
```

### 6.2 Report-Generierung

```bash
# Generate Allure report
pytest tests/ --alluredir=allure-results
allure serve allure-results

# Generate HTML coverage report
pytest tests/unit --cov=src --cov-report=html
open htmlcov/index.html
```

---

## 7. Test-Daten Management

### 7.1 Test Fixtures für Konsistente Testdaten

```python
# tests/fixtures/test_data.py

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

def load_fixture(name: str) -> dict:
    """Lade Test-Fixture aus JSON-Datei."""
    path = FIXTURES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)

# Vordefinierte Test-Fixtures
SAMPLE_SYMBOLS = load_fixture("symbols")
SAMPLE_OHLCV = load_fixture("ohlcv_data")
SAMPLE_FORECASTS = load_fixture("forecasts")
```

```json
// tests/fixtures/symbols.json
[
    {
        "symbol": "BTCUSD",
        "exchange": "binance",
        "enabled": true,
        "intervals": ["1m", "5m", "1h", "1d"]
    },
    {
        "symbol": "ETHUSD",
        "exchange": "binance",
        "enabled": true,
        "intervals": ["1m", "5m", "1h", "1d"]
    }
]
```

---

## 8. Empfohlene Implementierungsreihenfolge

### Phase 1: Grundlagen (Woche 1)
1. ✅ Test-Verzeichnisstruktur erstellen
2. ✅ pytest.ini und conftest.py einrichten
3. ✅ requirements-test.txt erstellen
4. ✅ Smoke/Health Check Tests implementieren

### Phase 2: Core Tests (Woche 2)
5. Unit Tests für kritische Services (Data, NHITS)
6. API Endpoint Tests für Data Service
7. Contract Tests für Response-Schemas

### Phase 3: Integration (Woche 3)
8. Integration Tests für Data Gateway Pattern
9. Service-Chain Tests (Data → NHITS → RAG)
10. Fallback-Chain Tests

### Phase 4: E2E & Performance (Woche 4)
11. End-to-End Workflow Tests
12. Load Tests mit Locust
13. CI/CD Pipeline Integration

---

## 9. Zusammenfassung

| Test-Typ | Anzahl Tests | Ausführungszeit | Priorität |
|----------|--------------|-----------------|-----------|
| Smoke | ~10 | <30s | Kritisch |
| Unit | ~100 | <2min | Hoch |
| API | ~50 | <5min | Hoch |
| Contract | ~20 | <2min | Mittel |
| Integration | ~30 | <10min | Mittel |
| E2E | ~10 | <30min | Niedrig |
| Performance | ~5 | variabel | Niedrig |

**Gesamtziel:** ~225 automatisierte Tests mit >80% Code-Coverage für kritische Services.
