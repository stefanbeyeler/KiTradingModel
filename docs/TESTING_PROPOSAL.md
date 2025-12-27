# Testing-System für KI Trading Microservices

## Übersicht

Die komplette Testing-Funktionalität ist in den **Watchdog Service** integriert. Dieser testet alle Microservice-Endpoints zentral und ermöglicht kontinuierliches Monitoring.

### Service-Übersicht

| Service | Port | Container | GPU | Zweck |
|---------|------|-----------|-----|-------|
| Frontend | 3000 | trading-frontend | - | Dashboard & API Gateway |
| Data Service | 3001 | trading-data | - | Daten-Gateway, Symbol-Management |
| NHITS Service | 3002 | trading-nhits | CUDA | Zeitreihen-Prognosen |
| TCN-Pattern | 3003 | trading-tcn | CUDA | Chart-Pattern-Erkennung |
| TCN-Train | 3011 | trading-tcn-train | CUDA | TCN-Modell-Training |
| HMM-Regime | 3004 | trading-hmm | - | Marktregime-Erkennung |
| Embedder | 3005 | trading-embedder | CUDA | Embedding-Generierung |
| RAG Service | 3008 | trading-rag | CUDA | Vektor-Suche & Wissensbasis |
| LLM Service | 3009 | trading-llm | CUDA | LLM-Analyse & Trading-Insights |
| **Watchdog** | 3010 | trading-watchdog | - | **Testing & Monitoring** |

---

## 1. Test-Architektur

```
Watchdog Service (Port 3010)
├── test_definitions.py     # ~90 Test-Definitionen
├── test_runner.py          # Test-Ausführung & Management
└── routes.py               # REST API für Tests

Getestete Services:
├── Frontend (3000)         # Health, Proxy-Checks
├── Data Service (3001)     # OHLCV, Symbols, Indicators
├── NHITS Service (3002)    # Forecasts, Models
├── TCN-Pattern (3003)      # Pattern Detection
├── TCN-Train (3011)        # Training Jobs
├── HMM-Regime (3004)       # Regime Detection
├── Embedder (3005)         # Embeddings
├── RAG Service (3008)      # Search, Context
└── LLM Service (3009)      # Analysis, Chat
```

---

## 2. Test-Kategorien

### 2.1 Smoke Tests
Schnelle Health-Checks für alle Services.

**Zweck:** Prüfen, ob Services erreichbar und grundlegend funktionsfähig sind.

```bash
make test-smoke
# oder
curl -X POST http://10.1.19.101:3010/api/v1/tests/run/smoke
```

### 2.2 API Tests
Vollständige Endpoint-Tests mit Request/Response-Validierung.

**Zweck:** Alle API-Endpoints auf korrekte Funktion prüfen.

```bash
curl -X POST http://10.1.19.101:3010/api/v1/tests/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "api"}'
```

### 2.3 Contract Tests
Schema-Validierung für API-Responses.

**Zweck:** Sicherstellen, dass Response-Formate den Spezifikationen entsprechen.

```bash
curl -X POST http://10.1.19.101:3010/api/v1/tests/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "contract"}'
```

### 2.4 Integration Tests
Service-übergreifende Workflows.

**Zweck:** End-to-End-Flows testen (z.B. Data → NHITS → LLM).

```bash
curl -X POST http://10.1.19.101:3010/api/v1/tests/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "integration"}'
```

---

## 3. Test-Prioritäten

| Priorität | Beschreibung | Beispiele |
|-----------|--------------|-----------|
| **critical** | Grundlegende Funktionen | Health-Checks, Data Gateway |
| **high** | Wichtige Features | OHLCV-Daten, Forecasts |
| **medium** | Standard-Funktionen | Indicators, Patterns |
| **low** | Optionale Features | Bulk-Operations |

---

## 4. Test-API-Endpoints

### Basis-URL
```
http://10.1.19.101:3010/api/v1/tests
```

### Endpoints

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/modes` | Verfügbare Test-Modi |
| GET | `/definitions` | Alle Test-Definitionen |
| GET | `/definitions?service=data` | Tests für bestimmten Service |
| POST | `/run` | Tests starten (mit Body) |
| POST | `/run/smoke` | Smoke-Tests starten |
| POST | `/run/critical` | Kritische Tests starten |
| POST | `/run/full` | Alle Tests starten |
| POST | `/run/service/{name}` | Service-spezifische Tests |
| GET | `/status` | Aktueller Test-Status |
| GET | `/stream` | Live-Updates (SSE) |
| GET | `/history` | Test-Historie (50 Runs) |
| GET | `/history/{run_id}` | Einzelner Test-Run |

### Request-Body für /run

```json
{
  "mode": "full",
  "services": ["data", "nhits", "llm"],
  "categories": ["smoke", "api"],
  "priorities": ["critical", "high"]
}
```

---

## 5. Test-Definitionen

### Struktur

```python
@dataclass
class TestDefinition:
    name: str                           # Eindeutiger Test-Name
    service: str                        # Ziel-Service
    category: TestCategory              # smoke, api, contract, integration
    priority: TestPriority              # critical, high, medium, low
    endpoint: str                       # API-Endpoint
    method: str = "GET"                 # HTTP-Methode
    body: Optional[Dict] = None         # Request-Body
    expected_status: List[int] = [200]  # Erwartete Status-Codes
    timeout: float = 30.0               # Timeout in Sekunden
    schema_validator: Callable = None   # Schema-Validierungsfunktion
    depends_on: List[str] = []          # Test-Abhängigkeiten
```

### Beispiel-Definitionen

```python
# Smoke Test
TestDefinition(
    name="data_health",
    service="data",
    category=TestCategory.SMOKE,
    priority=TestPriority.CRITICAL,
    endpoint="/health",
    expected_status=[200],
    schema_validator=validate_health_response
)

# API Test
TestDefinition(
    name="data_ohlcv_btcusd",
    service="data",
    category=TestCategory.API,
    priority=TestPriority.HIGH,
    endpoint="/api/v1/history?symbol=BTCUSD&interval=1h&limit=100",
    schema_validator=validate_ohlcv_data,
    depends_on=["data_health"]
)

# Integration Test
TestDefinition(
    name="integration_data_nhits_forecast",
    service="nhits",
    category=TestCategory.INTEGRATION,
    priority=TestPriority.HIGH,
    endpoint="/api/v1/forecast",
    method="POST",
    body={"symbol": "BTCUSD", "horizon": 24},
    depends_on=["data_ohlcv_btcusd", "nhits_health"]
)
```

---

## 6. Schema-Validatoren

```python
def validate_health_response(data: Dict) -> bool:
    """Validiert Health-Response Format."""
    required = ["status", "service", "timestamp"]
    return all(key in data for key in required)

def validate_ohlcv_data(data: Dict) -> bool:
    """Validiert OHLCV-Daten Format."""
    if "data" not in data:
        return False
    if not isinstance(data["data"], list):
        return False
    if len(data["data"]) > 0:
        candle = data["data"][0]
        required = ["timestamp", "open", "high", "low", "close"]
        return all(key in candle for key in required)
    return True

def validate_forecast_response(data: Dict) -> bool:
    """Validiert Forecast-Response Format."""
    required = ["symbol", "predictions", "confidence"]
    return all(key in data for key in required)
```

---

## 7. Makefile-Integration

```makefile
WATCHDOG_URL ?= http://10.1.19.101:3010

# Smoke Tests
test-smoke:
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/smoke | python -m json.tool

# Kritische Tests
test-critical:
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/critical | python -m json.tool

# Vollständige Tests
test-full:
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/full | python -m json.tool

# Service-spezifische Tests
test-service:
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/service/$(SERVICE) | python -m json.tool

# Test-Status
test-status:
	@curl -s $(WATCHDOG_URL)/api/v1/tests/status | python -m json.tool

# Test-Historie
test-history:
	@curl -s $(WATCHDOG_URL)/api/v1/tests/history | python -m json.tool
```

---

## 8. Live-Monitoring

### Server-Sent Events (SSE)

```bash
curl http://10.1.19.101:3010/api/v1/tests/stream
```

Ausgabe:
```
data: {"type": "test_start", "test": "data_health", "timestamp": "..."}
data: {"type": "test_complete", "test": "data_health", "status": "passed", "duration": 0.15}
data: {"type": "test_start", "test": "nhits_health", "timestamp": "..."}
data: {"type": "run_complete", "passed": 85, "failed": 2, "skipped": 3}
```

---

## 9. Test-Ergebnisse

### Response-Format

```json
{
  "run_id": "test_run_20241227_143022",
  "status": "completed",
  "mode": "full",
  "started_at": "2024-12-27T14:30:22Z",
  "completed_at": "2024-12-27T14:32:15Z",
  "duration": 113.5,
  "summary": {
    "total": 90,
    "passed": 85,
    "failed": 2,
    "skipped": 3
  },
  "results": [
    {
      "test": "data_health",
      "status": "passed",
      "duration": 0.15,
      "response_status": 200
    },
    {
      "test": "nhits_forecast_btcusd",
      "status": "failed",
      "duration": 5.2,
      "error": "Timeout exceeded"
    }
  ]
}
```

---

## 10. Abhängigkeits-Management

### Service-Abhängigkeiten

Tests werden automatisch übersprungen, wenn abhängige Services nicht erreichbar sind:

```
Data Service (3001)
    └── NHITS Service (3002) benötigt Data Service
    └── TCN Service (3003) benötigt Data Service
    └── RAG Service (3008) benötigt Data Service

Embedder Service (3005)
    └── RAG Service (3008) benötigt Embedder

RAG Service (3008)
    └── LLM Service (3009) benötigt RAG
```

### Test-Abhängigkeiten

```python
# nhits_forecast_btcusd wird übersprungen wenn data_health fehlschlägt
TestDefinition(
    name="nhits_forecast_btcusd",
    depends_on=["data_health", "nhits_health"]
)
```

---

## 11. Implementierung

### Dateien

| Datei | Beschreibung |
|-------|--------------|
| `src/services/watchdog_app/services/test_definitions.py` | ~90 Test-Definitionen |
| `src/services/watchdog_app/services/test_runner.py` | TestRunnerService Klasse |
| `src/services/watchdog_app/api/routes.py` | REST API Endpoints |

### Test-Runner Klasse

```python
class TestRunnerService:
    async def start_test_run(
        self,
        mode: RunMode,
        services: Optional[List[str]] = None,
        categories: Optional[List[TestCategory]] = None,
        priorities: Optional[List[TestPriority]] = None
    ) -> str:
        """Startet einen Test-Run und gibt die Run-ID zurück."""

    async def run_single_test(self, test: TestDefinition) -> TestResult:
        """Führt einen einzelnen Test aus."""

    async def check_service_health(self, service: str) -> bool:
        """Prüft ob ein Service erreichbar ist."""

    def get_test_status(self, run_id: str) -> Optional[TestRunStatus]:
        """Gibt den Status eines Test-Runs zurück."""

    def get_history(self, limit: int = 50) -> List[TestRunSummary]:
        """Gibt die Test-Historie zurück."""
```

---

## 12. Best Practices

### Neue Tests hinzufügen

1. Test-Definition in `test_definitions.py` hinzufügen
2. Schema-Validator erstellen falls nötig
3. Abhängigkeiten definieren
4. Kategorie und Priorität festlegen

### Test-Wartung

- **Smoke Tests**: Sollten < 1s pro Test dauern
- **API Tests**: Sollten < 5s pro Test dauern
- **Integration Tests**: Dürfen länger dauern (bis zu 30s)
- **Timeouts**: Immer explizit setzen

### Fehlerbehandlung

- Tests sollten idempotent sein
- Keine Seiteneffekte in Smoke/API Tests
- Integration Tests können Testdaten erstellen/löschen
