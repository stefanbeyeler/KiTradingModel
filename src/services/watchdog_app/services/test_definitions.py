"""Test-Definitionen für alle Microservices.

Enthält alle API-Tests, die vom Watchdog ausgeführt werden können.
Kategorisiert nach: smoke, api, contract, integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional

import httpx

# Import zentrale Microservices-Konfiguration
from src.config.microservices import microservices_config


class TestCategory(str, Enum):
    """Test-Kategorien."""
    SMOKE = "smoke"          # Schnelle Health-Checks
    API = "api"              # API-Endpoint Tests
    CONTRACT = "contract"    # Schema-Validierung
    INTEGRATION = "integration"  # Service-übergreifend
    E2E = "e2e"              # End-to-End Workflows


class TestPriority(str, Enum):
    """Test-Prioritäten."""
    CRITICAL = "critical"    # Muss immer laufen
    HIGH = "high"            # Wichtig für Funktionalität
    MEDIUM = "medium"        # Normale Tests
    LOW = "low"              # Nice-to-have


@dataclass
class TestDefinition:
    """Definition eines einzelnen Tests."""
    name: str
    service: str
    category: TestCategory
    priority: TestPriority
    endpoint: str
    method: str = "GET"
    body: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    expected_status: List[int] = field(default_factory=lambda: [200])
    timeout: float = 30.0
    description: str = ""
    # Validierung
    validate_json: bool = True
    required_fields: List[str] = field(default_factory=list)
    schema_validator: Optional[Callable[[Dict], bool]] = None
    # Abhängigkeiten
    depends_on: List[str] = field(default_factory=list)


class ServiceURLs:
    """Service-URLs für Docker-Umgebung (aus zentraler Konfiguration)."""

    FRONTEND = microservices_config.get_service_url("frontend")
    DATA = microservices_config.data_service_url
    NHITS = microservices_config.nhits_service_url
    TCN = microservices_config.tcn_service_url
    TCN_TRAIN = microservices_config.tcn_train_url
    HMM = microservices_config.hmm_service_url
    EMBEDDER = microservices_config.embedder_service_url
    CANDLESTICK = microservices_config.candlestick_service_url
    CANDLESTICK_TRAIN = microservices_config.candlestick_train_url
    RAG = microservices_config.rag_service_url
    LLM = microservices_config.llm_service_url
    WATCHDOG = microservices_config.watchdog_service_url
    EASYINSIGHT = microservices_config.easyinsight_api_url.replace("/api", "")


# ============ Schema Validators ============

def validate_health_response(data: Dict) -> bool:
    """Validiert Health-Response-Schema."""
    if not isinstance(data, dict):
        return False
    status = data.get("status", "").lower()
    return status in ["healthy", "ok", "running", "up"]


def validate_ohlcv_data(data: Dict) -> bool:
    """Validiert OHLCV-Daten."""
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("data", data.get("values", []))
    else:
        return False

    if not items:
        return True  # Leere Liste ist OK

    for item in items[:5]:  # Prüfe erste 5
        if not isinstance(item, dict):
            return False
        high = item.get("high", 0)
        low = item.get("low", 0)
        if high < low:
            return False
    return True


def validate_forecast_response(data: Dict) -> bool:
    """Validiert Forecast-Response."""
    if not isinstance(data, dict):
        return False
    # Muss Symbol oder predictions haben
    return "symbol" in data or "predictions" in data or "forecast" in data


def validate_symbols_list(data: Any) -> bool:
    """Validiert Symbol-Listen."""
    if isinstance(data, list):
        return True
    if isinstance(data, dict):
        return "symbols" in data or "data" in data
    return False


# ============ Test-Definitionen ============

def get_all_tests() -> List[TestDefinition]:
    """Gibt alle Test-Definitionen zurück."""
    tests = []

    # ===== SMOKE TESTS (Health Checks) =====
    smoke_services = [
        ("frontend", ServiceURLs.FRONTEND, "Frontend Dashboard"),
        ("data", ServiceURLs.DATA, "Data Service"),
        ("nhits", ServiceURLs.NHITS, "NHITS Service"),
        ("tcn", ServiceURLs.TCN, "TCN-Pattern Service"),
        ("tcn_train", ServiceURLs.TCN_TRAIN, "TCN-Train Service"),
        ("hmm", ServiceURLs.HMM, "HMM-Regime Service"),
        ("embedder", ServiceURLs.EMBEDDER, "Embedder Service"),
        ("candlestick", ServiceURLs.CANDLESTICK, "Candlestick Service"),
        ("candlestick_train", ServiceURLs.CANDLESTICK_TRAIN, "Candlestick-Train Service"),
        ("rag", ServiceURLs.RAG, "RAG Service"),
        ("llm", ServiceURLs.LLM, "LLM Service"),
        ("watchdog", ServiceURLs.WATCHDOG, "Watchdog Service"),
    ]

    for service_key, base_url, display_name in smoke_services:
        tests.append(TestDefinition(
            name=f"{display_name} Health Check",
            service=service_key,
            category=TestCategory.SMOKE,
            priority=TestPriority.CRITICAL,
            endpoint=f"{base_url}/health",
            expected_status=[200],
            timeout=10.0,
            description=f"Prüft ob {display_name} erreichbar ist",
            schema_validator=validate_health_response
        ))

    # ===== DATA SERVICE API TESTS =====
    data_url = ServiceURLs.DATA

    # Symbol-Management
    tests.extend([
        TestDefinition(
            name="Get Managed Symbols",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.CRITICAL,
            endpoint=f"{data_url}/api/v1/managed-symbols",
            expected_status=[200],
            description="Liste aller verwalteten Symbole",
            schema_validator=validate_symbols_list
        ),
        TestDefinition(
            name="Get Managed Symbols Stats",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/managed-symbols/stats",
            expected_status=[200],
            description="Statistiken der verwalteten Symbole"
        ),
        TestDefinition(
            name="Search Symbols",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/managed-symbols/search",
            params={"q": "BTC"},
            expected_status=[200, 422],
            description="Symbol-Suche"
        ),
        TestDefinition(
            name="Get Available EasyInsight Symbols",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{data_url}/api/v1/managed-symbols/available/easyinsight",
            expected_status=[200, 503],
            description="Verfügbare EasyInsight-Symbole"
        ),
        TestDefinition(
            name="Get YFinance Symbols",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/yfinance/symbols",
            expected_status=[200],
            description="Yahoo Finance Symbol-Liste"
        ),
    ])

    # OHLCV-Daten
    tests.extend([
        TestDefinition(
            name="Get Time Series (BTCUSD)",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.CRITICAL,
            endpoint=f"{data_url}/api/v1/twelvedata/time_series/BTCUSD",
            params={"interval": "1h", "outputsize": 50},
            expected_status=[200, 404, 422, 429, 503],
            timeout=60.0,
            description="Historische OHLCV-Daten von TwelveData"
        ),
        TestDefinition(
            name="Get Time Series (EURUSD)",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{data_url}/api/v1/twelvedata/time_series/EURUSD",
            params={"interval": "1h", "outputsize": 50},
            expected_status=[200, 404, 422, 429, 503],
            timeout=60.0,
            description="Forex-Daten von TwelveData"
        ),
        TestDefinition(
            name="Get YFinance Time Series",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/yfinance/time-series/AAPL",
            params={"interval": "1h", "period": "5d"},
            expected_status=[200, 400, 404, 422, 503],
            timeout=60.0,
            description="Yahoo Finance Zeitreihen"
        ),
        TestDefinition(
            name="Get Training Data",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/training-data/BTCUSD",
            expected_status=[200, 404],
            description="Gecachte Trainingsdaten"
        ),
        TestDefinition(
            name="Get Live Data",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{data_url}/api/v1/managed-symbols/live-data/BTCUSD",
            expected_status=[200, 404, 503],
            description="Live-Marktdaten"
        ),
    ])

    # Technische Indikatoren
    indicators = ["rsi", "macd", "sma", "ema", "bbands", "stoch", "atr", "adx"]
    for indicator in indicators:
        tests.append(TestDefinition(
            name=f"TwelveData {indicator.upper()}",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/twelvedata/{indicator}/EURUSD",
            params={"interval": "1h"},
            expected_status=[200, 404, 429, 503],
            timeout=60.0,
            description=f"{indicator.upper()} Indikator"
        ))

    # Externe Quellen
    tests.extend([
        TestDefinition(
            name="List External Sources",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/external-sources",
            expected_status=[200],
            description="Liste externer Datenquellen"
        ),
        TestDefinition(
            name="Get Economic Calendar",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{data_url}/api/v1/external-sources/economic-calendar",
            expected_status=[200, 503],
            description="Wirtschaftskalender"
        ),
        TestDefinition(
            name="Get Sentiment Data",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{data_url}/api/v1/external-sources/sentiment",
            expected_status=[200, 503],
            description="Marktstimmung"
        ),
        TestDefinition(
            name="Get Technical Levels",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/external-sources/technical-levels/BTCUSD",
            expected_status=[200, 404, 503],
            description="Technische Level (S/R, Fibonacci)"
        ),
        TestDefinition(
            name="Get Macro Data",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{data_url}/api/v1/external-sources/macro",
            expected_status=[200, 503],
            description="Makroökonomische Daten"
        ),
        TestDefinition(
            name="Fetch Trading Context",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{data_url}/api/v1/external-sources/trading-context/BTCUSD",
            method="POST",
            expected_status=[200, 503],
            timeout=60.0,
            description="Aggregierter Trading-Kontext"
        ),
    ])

    # Note: Pattern-Erkennung wurde in Candlestick Service (Port 3006) verschoben
    # Siehe CANDLESTICK SERVICE API TESTS weiter unten

    # Konfiguration
    tests.extend([
        TestDefinition(
            name="Get Timezone Config",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{data_url}/api/v1/config/timezone",
            expected_status=[200],
            description="Aktuelle Zeitzone"
        ),
        TestDefinition(
            name="List Timezones",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{data_url}/api/v1/config/timezones",
            expected_status=[200],
            description="Verfügbare Zeitzonen"
        ),
        TestDefinition(
            name="Get Strategies",
            service="data",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{data_url}/api/v1/strategies",
            expected_status=[200],
            description="Trading-Strategien"
        ),
    ])

    # ===== NHITS SERVICE API TESTS =====
    nhits_url = ServiceURLs.NHITS

    tests.extend([
        TestDefinition(
            name="NHITS Forecast Status",
            service="nhits",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{nhits_url}/api/v1/forecast/status",
            expected_status=[200],
            description="Forecast-Service Status"
        ),
        TestDefinition(
            name="NHITS Models List",
            service="nhits",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{nhits_url}/api/v1/forecast/models",
            expected_status=[200],
            description="Verfügbare NHITS-Modelle"
        ),
        TestDefinition(
            name="Generate Forecast (BTCUSD)",
            service="nhits",
            category=TestCategory.API,
            priority=TestPriority.CRITICAL,
            endpoint=f"{nhits_url}/api/v1/forecast/BTCUSD",
            expected_status=[200, 404, 503],
            timeout=120.0,
            description="BTCUSD Preis-Prognose",
            schema_validator=validate_forecast_response
        ),
        TestDefinition(
            name="Get Forecast Favorites",
            service="nhits",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{nhits_url}/api/v1/forecast/favorites",
            expected_status=[200],
            description="Favorisierte Symbole"
        ),
    ])

    # ===== TCN SERVICE API TESTS =====
    tcn_url = ServiceURLs.TCN

    tests.extend([
        TestDefinition(
            name="Get Pattern Types",
            service="tcn",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{tcn_url}/api/v1/patterns",
            expected_status=[200],
            description="Unterstützte Pattern-Typen"
        ),
        TestDefinition(
            name="Get TCN Models",
            service="tcn",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{tcn_url}/api/v1/models",
            expected_status=[200],
            description="Trainierte TCN-Modelle"
        ),
        TestDefinition(
            name="TCN Pattern Detection",
            service="tcn",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{tcn_url}/api/v1/detect",
            method="POST",
            body={"symbol": "BTCUSD", "timeframe": "1h"},
            expected_status=[200, 404, 422, 503],
            timeout=60.0,
            description="Pattern-Erkennung via TCN"
        ),
        TestDefinition(
            name="Get Pattern History",
            service="tcn",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{tcn_url}/api/v1/history/BTCUSD",
            expected_status=[200, 404],
            description="Pattern-Historie"
        ),
    ])

    # ===== TCN-TRAIN SERVICE API TESTS =====
    tcn_train_url = ServiceURLs.TCN_TRAIN

    tests.extend([
        TestDefinition(
            name="TCN Training Status",
            service="tcn_train",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{tcn_train_url}/api/v1/training/status",
            expected_status=[200],
            description="Training-Status"
        ),
        TestDefinition(
            name="List Training Jobs",
            service="tcn_train",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{tcn_train_url}/api/v1/training/jobs",
            expected_status=[200],
            description="Training-Jobs"
        ),
    ])

    # ===== HMM SERVICE API TESTS =====
    hmm_url = ServiceURLs.HMM

    tests.extend([
        TestDefinition(
            name="Get Market Regime",
            service="hmm",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{hmm_url}/api/v1/regime/BTCUSD",
            expected_status=[200, 404, 503],
            description="Aktuelles Markt-Regime"
        ),
        TestDefinition(
            name="Get Regime Probabilities",
            service="hmm",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{hmm_url}/api/v1/regime/BTCUSD/probabilities",
            expected_status=[200, 404, 503],
            description="Regime-Wahrscheinlichkeiten"
        ),
        TestDefinition(
            name="Score Trading Signal",
            service="hmm",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{hmm_url}/api/v1/score",
            method="POST",
            body={"symbol": "BTCUSD", "signal": "buy"},
            expected_status=[200, 422, 503],
            description="Signal-Bewertung"
        ),
    ])

    # ===== EMBEDDER SERVICE API TESTS =====
    embedder_url = ServiceURLs.EMBEDDER

    tests.extend([
        TestDefinition(
            name="Get Embedding Models",
            service="embedder",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{embedder_url}/api/v1/models",
            expected_status=[200],
            description="Verfügbare Embedding-Modelle"
        ),
        TestDefinition(
            name="Generate Text Embedding",
            service="embedder",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{embedder_url}/api/v1/embed/text",
            method="POST",
            body={"text": "Bitcoin price analysis"},
            expected_status=[200, 422, 503],
            timeout=60.0,
            description="Text-zu-Vektor"
        ),
        TestDefinition(
            name="Calculate Similarity",
            service="embedder",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{embedder_url}/api/v1/similarity",
            method="POST",
            body={"text1": "Bitcoin bullish", "text2": "BTC rising"},
            expected_status=[200, 422, 503],
            timeout=60.0,
            description="Text-Ähnlichkeit"
        ),
    ])

    # ===== CANDLESTICK SERVICE API TESTS =====
    candlestick_url = ServiceURLs.CANDLESTICK

    tests.extend([
        TestDefinition(
            name="Get Candlestick Pattern Types",
            service="candlestick",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{candlestick_url}/api/v1/patterns",
            expected_status=[200],
            description="Unterstützte Candlestick-Pattern-Typen"
        ),
        TestDefinition(
            name="Candlestick Pattern Scan",
            service="candlestick",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{candlestick_url}/api/v1/scan/BTCUSD",
            expected_status=[200, 404, 503],
            timeout=60.0,
            description="Pattern-Scan für BTCUSD"
        ),
        TestDefinition(
            name="Multi-Timeframe Pattern Scan",
            service="candlestick",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{candlestick_url}/api/v1/scan",
            method="POST",
            body={"symbol": "BTCUSD", "timeframes": ["H1", "H4"]},
            expected_status=[200, 404, 422, 503],
            timeout=60.0,
            description="Multi-Timeframe Pattern-Erkennung"
        ),
        TestDefinition(
            name="Get Candlestick Chart Data",
            service="candlestick",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{candlestick_url}/api/v1/chart/BTCUSD",
            expected_status=[200, 404, 503],
            description="Chart-Daten mit Patterns"
        ),
        TestDefinition(
            name="Get Pattern History",
            service="candlestick",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{candlestick_url}/api/v1/history",
            expected_status=[200],
            description="Pattern-Historie"
        ),
        TestDefinition(
            name="Get Pattern History by Symbol",
            service="candlestick",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{candlestick_url}/api/v1/history/BTCUSD",
            expected_status=[200, 404],
            description="Pattern-Historie für BTCUSD"
        ),
        TestDefinition(
            name="Get Pattern Statistics",
            service="candlestick",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{candlestick_url}/api/v1/history/statistics",
            expected_status=[200],
            description="Pattern-Statistiken"
        ),
    ])

    # ===== CANDLESTICK-TRAIN SERVICE API TESTS =====
    candlestick_train_url = ServiceURLs.CANDLESTICK_TRAIN

    tests.extend([
        TestDefinition(
            name="Candlestick Training Status",
            service="candlestick_train",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{candlestick_train_url}/api/v1/train/status",
            expected_status=[200],
            description="Training-Status"
        ),
        TestDefinition(
            name="List Candlestick Training Jobs",
            service="candlestick_train",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{candlestick_train_url}/api/v1/train/jobs",
            expected_status=[200],
            description="Training-Jobs"
        ),
        TestDefinition(
            name="Get Training Progress",
            service="candlestick_train",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{candlestick_train_url}/api/v1/train/progress",
            expected_status=[200],
            description="Training-Fortschritt"
        ),
        TestDefinition(
            name="Get Candlestick Model Info",
            service="candlestick_train",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{candlestick_train_url}/api/v1/model",
            expected_status=[200, 404],
            description="Modell-Informationen"
        ),
    ])

    # ===== RAG SERVICE API TESTS =====
    rag_url = ServiceURLs.RAG

    tests.extend([
        TestDefinition(
            name="Get RAG Index Stats",
            service="rag",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{rag_url}/api/v1/rag/stats",
            expected_status=[200],
            description="RAG-Index Statistiken"
        ),
        TestDefinition(
            name="RAG Semantic Query",
            service="rag",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{rag_url}/api/v1/rag/query",
            method="POST",
            body={"query": "Bitcoin market analysis", "top_k": 5},
            expected_status=[200, 422, 503],
            timeout=60.0,
            description="Semantische Suche"
        ),
        TestDefinition(
            name="List RAG Documents",
            service="rag",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{rag_url}/api/v1/rag/documents",
            expected_status=[200],
            description="Indexierte Dokumente"
        ),
        TestDefinition(
            name="Get RAG Sources",
            service="rag",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{rag_url}/api/v1/rag/sources",
            expected_status=[200],
            description="Verfügbare Quellen"
        ),
    ])

    # ===== LLM SERVICE API TESTS =====
    llm_url = ServiceURLs.LLM

    tests.extend([
        TestDefinition(
            name="LLM Model Status",
            service="llm",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{llm_url}/api/v1/llm/status",
            expected_status=[200],
            description="LLM-Model Status"
        ),
        TestDefinition(
            name="LLM Trading Analysis",
            service="llm",
            category=TestCategory.API,
            priority=TestPriority.CRITICAL,
            endpoint=f"{llm_url}/api/v1/analyze",
            method="POST",
            body={"symbol": "BTCUSD", "context": "Price at 50000, RSI at 65"},
            expected_status=[200, 422, 503],
            timeout=120.0,
            description="Trading-Analyse via LLM"
        ),
        TestDefinition(
            name="LLM Chat",
            service="llm",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{llm_url}/api/v1/chat",
            method="POST",
            body={"message": "What is the current market trend?"},
            expected_status=[200, 422, 503],
            timeout=120.0,
            description="Chat mit LLM"
        ),
        TestDefinition(
            name="Generate Trading Signal",
            service="llm",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{llm_url}/api/v1/signal",
            method="POST",
            body={"symbol": "BTCUSD"},
            expected_status=[200, 422, 503],
            timeout=120.0,
            description="Trading-Signal generieren"
        ),
    ])

    # ===== WATCHDOG SERVICE API TESTS =====
    watchdog_url = ServiceURLs.WATCHDOG

    tests.extend([
        TestDefinition(
            name="Watchdog System Status",
            service="watchdog",
            category=TestCategory.API,
            priority=TestPriority.HIGH,
            endpoint=f"{watchdog_url}/api/v1/status",
            expected_status=[200],
            description="Gesamt-System-Status"
        ),
        TestDefinition(
            name="List Monitored Services",
            service="watchdog",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{watchdog_url}/api/v1/services",
            expected_status=[200],
            description="Überwachte Services"
        ),
        TestDefinition(
            name="Get Alert History",
            service="watchdog",
            category=TestCategory.API,
            priority=TestPriority.MEDIUM,
            endpoint=f"{watchdog_url}/api/v1/alerts/history",
            params={"limit": 10},
            expected_status=[200],
            description="Alert-Historie"
        ),
        TestDefinition(
            name="Get Watchdog Config",
            service="watchdog",
            category=TestCategory.API,
            priority=TestPriority.LOW,
            endpoint=f"{watchdog_url}/api/v1/config",
            expected_status=[200],
            description="Watchdog-Konfiguration"
        ),
    ])

    # ===== CONTRACT TESTS (Schema-Validierung) =====
    tests.extend([
        TestDefinition(
            name="Contract: Health Response Schema",
            service="data",
            category=TestCategory.CONTRACT,
            priority=TestPriority.HIGH,
            endpoint=f"{data_url}/health",
            expected_status=[200],
            description="Health-Response muss Schema entsprechen",
            schema_validator=validate_health_response
        ),
        TestDefinition(
            name="Contract: OHLCV Data Integrity",
            service="data",
            category=TestCategory.CONTRACT,
            priority=TestPriority.CRITICAL,
            endpoint=f"{data_url}/api/v1/twelvedata/time_series/BTCUSD",
            params={"interval": "1h", "outputsize": 20},
            expected_status=[200, 404, 429, 503],
            timeout=60.0,
            description="OHLCV-Daten: High >= Low",
            schema_validator=validate_ohlcv_data
        ),
        TestDefinition(
            name="Contract: Forecast Response Schema",
            service="nhits",
            category=TestCategory.CONTRACT,
            priority=TestPriority.HIGH,
            endpoint=f"{nhits_url}/api/v1/forecast/BTCUSD",
            expected_status=[200, 404, 503],
            timeout=120.0,
            description="Forecast muss Symbol/Predictions enthalten",
            schema_validator=validate_forecast_response
        ),
        TestDefinition(
            name="Contract: Symbols List Schema",
            service="data",
            category=TestCategory.CONTRACT,
            priority=TestPriority.HIGH,
            endpoint=f"{data_url}/api/v1/managed-symbols",
            expected_status=[200],
            description="Symbol-Liste muss Array oder Dict sein",
            schema_validator=validate_symbols_list
        ),
    ])

    # ===== INTEGRATION TESTS =====
    tests.extend([
        TestDefinition(
            name="Integration: Data-to-NHITS Chain",
            service="integration",
            category=TestCategory.INTEGRATION,
            priority=TestPriority.CRITICAL,
            endpoint=f"{nhits_url}/api/v1/forecast/BTCUSD",
            expected_status=[200, 404, 503],
            timeout=180.0,
            description="Data Service -> NHITS Forecast",
            depends_on=["Data Service Health Check", "NHITS Service Health Check"]
        ),
        TestDefinition(
            name="Integration: Pattern Detection Chain",
            service="integration",
            category=TestCategory.INTEGRATION,
            priority=TestPriority.HIGH,
            endpoint=f"{tcn_url}/api/v1/detect",
            method="POST",
            body={"symbol": "BTCUSD", "timeframe": "1h"},
            expected_status=[200, 404, 422, 503],
            timeout=180.0,
            description="Data Service -> TCN Pattern",
            depends_on=["Data Service Health Check", "TCN-Pattern Service Health Check"]
        ),
        TestDefinition(
            name="Integration: RAG Query with Context",
            service="integration",
            category=TestCategory.INTEGRATION,
            priority=TestPriority.HIGH,
            endpoint=f"{rag_url}/api/v1/rag/query",
            method="POST",
            body={"query": "BTCUSD analysis", "top_k": 3, "include_context": True},
            expected_status=[200, 422, 503],
            timeout=120.0,
            description="RAG mit externem Kontext",
            depends_on=["Data Service Health Check", "RAG Service Health Check"]
        ),
        TestDefinition(
            name="Integration: Full Analysis Pipeline",
            service="integration",
            category=TestCategory.INTEGRATION,
            priority=TestPriority.CRITICAL,
            endpoint=f"{llm_url}/api/v1/analyze",
            method="POST",
            body={"symbol": "BTCUSD", "include_forecast": True, "include_patterns": True},
            expected_status=[200, 422, 503],
            timeout=300.0,
            description="Vollständige Analyse-Pipeline",
            depends_on=[
                "Data Service Health Check",
                "NHITS Service Health Check",
                "TCN-Pattern Service Health Check",
                "LLM Service Health Check"
            ]
        ),
    ])

    return tests


def get_tests_by_category(category: TestCategory) -> List[TestDefinition]:
    """Filtert Tests nach Kategorie."""
    return [t for t in get_all_tests() if t.category == category]


def get_tests_by_service(service: str) -> List[TestDefinition]:
    """Filtert Tests nach Service."""
    return [t for t in get_all_tests() if t.service == service]


def get_tests_by_priority(priority: TestPriority) -> List[TestDefinition]:
    """Filtert Tests nach Priorität."""
    return [t for t in get_all_tests() if t.priority == priority]


def get_critical_tests() -> List[TestDefinition]:
    """Gibt nur kritische Tests zurück (für Smoke-Testing)."""
    return [
        t for t in get_all_tests()
        if t.priority == TestPriority.CRITICAL or t.category == TestCategory.SMOKE
    ]
