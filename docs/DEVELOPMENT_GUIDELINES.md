# Entwicklungsvorgaben - KI Trading Model

Dieses Dokument definiert die verbindlichen Entwicklungsstandards für das KI Trading Model Projekt.

## Inhaltsverzeichnis

1. [Projektübersicht](#projektübersicht)
   - [Architektur](#architektur)
   - [Kernprinzipien](#kernprinzipien)
   - [Datenzugriff-Architektur](#datenzugriff-architektur)
2. [Technologie-Stack](#technologie-stack)
3. [Code-Stil und Formatierung](#code-stil-und-formatierung)
4. [Namenskonventionen](#namenskonventionen)
5. [Import-Organisation](#import-organisation)
6. [Typisierung](#typisierung)
7. [Async-Patterns](#async-patterns)
8. [API-Design](#api-design)
9. [Fehlerbehandlung](#fehlerbehandlung)
10. [Logging](#logging)
11. [Konfiguration](#konfiguration)
12. [Testing](#testing)
13. [Git-Workflow](#git-workflow)
14. [Dokumentation](#dokumentation)
15. [Sicherheit](#sicherheit)

---

## Projektübersicht

### Architektur

Das Projekt verwendet eine **Monolith + Microservices Hybrid-Architektur**:

```
KiTradingModel/
├── src/
│   ├── api/routes.py              # API-Endpunkte (thematisch gruppiert)
│   ├── config/settings.py         # Pydantic Settings mit GPU-Erkennung
│   ├── models/                    # Pydantic Datenmodelle
│   ├── services/                  # Geschäftslogik
│   │   ├── {service}_service.py   # Service-Klassen
│   │   └── rag_data_sources/      # RAG-Datenquellen (Plugin-System)
│   ├── service_registry.py        # Globale Service-Instanzen
│   └── version.py                 # Git-basierte Versionierung
├── docker/services/               # Microservice Dockerfiles
├── docs/                          # Dokumentation
├── static/                        # Frontend Dashboard
├── scripts/                       # Hilfsskripte
└── trading_strategies/            # Strategiedefinitionen
```

### Kernprinzipien

1. **Async-First**: Alle I/O-Operationen sind asynchron
2. **GPU-Aware**: Automatische CUDA-Erkennung und -Nutzung
3. **API-First**: Externe Daten via APIs, keine lokale Datenbank
4. **Modular**: Pluggable RAG-Datenquellen und Services
5. **Data Service als Gateway**: Zentraler Datenzugriff über den Data Service

### Datenzugriff-Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                     Externe Datenquellen                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ EasyInsight │  │ TwelveData  │  │ Weitere APIs            │  │
│  │ TimescaleDB │  │ API         │  │ (Sentiment, OnChain...) │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                │
│         └────────────────┼─────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │     DATA SERVICE      │  ◄── Einziges Gateway    │
│              │      (Port 3001)      │      für externe Daten   │
│              └───────────┬───────────┘                          │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ NHITS       │  │ RAG         │  │ LLM         │              │
│  │ Service     │  │ Service     │  │ Service     │              │
│  │ (Port 3002) │  │ (Port 3003) │  │ (Port 3004) │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

#### Verbindliche Regeln für Datenzugriff

| Regel | Beschreibung |
|-------|--------------|
| **Kein direkter DB-Zugriff** | Der Zugriff auf die EasyInsight TimescaleDB erfolgt **ausschliesslich** über die EasyInsight API. Direkte Datenbankverbindungen (z.B. via `asyncpg`, `psycopg2`) sind **verboten**. |
| **Data Service als Gateway** | Alle Services (NHITS, RAG, LLM) greifen auf externe Daten **ausschliesslich** über den Data Service zu. Direkte API-Aufrufe zu externen Datenquellen aus anderen Services sind nicht erlaubt. |
| **Fallback-Strategie** | Der Data Service implementiert Fallback-Logik (z.B. TwelveData als Backup für EasyInsight). Diese Logik gehört nur in den Data Service. |

#### Beispiele

```python
# ══════════════════════════════════════════════════════════════════
# FALSCH: Direkter Datenbankzugriff aus einem Service
# ══════════════════════════════════════════════════════════════════
import asyncpg

class NHITSTrainingService:
    async def get_training_data(self, symbol: str):
        # VERBOTEN! Kein direkter DB-Zugriff
        conn = await asyncpg.connect("postgresql://...")
        data = await conn.fetch("SELECT * FROM ohlcv WHERE symbol = $1", symbol)
        return data

# ══════════════════════════════════════════════════════════════════
# FALSCH: Direkter API-Aufruf zu EasyInsight aus NHITS-Service
# ══════════════════════════════════════════════════════════════════
class NHITSTrainingService:
    async def get_training_data(self, symbol: str):
        # VERBOTEN! Andere Services dürfen nicht direkt auf externe APIs zugreifen
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{EASYINSIGHT_API}/ohlcv/{symbol}")
            return response.json()

# ══════════════════════════════════════════════════════════════════
# RICHTIG: Zugriff über Data Service
# ══════════════════════════════════════════════════════════════════
class NHITSTrainingService:
    def __init__(self, data_service_url: str = "http://data-service:3001"):
        self.data_service_url = data_service_url

    async def get_training_data(self, symbol: str, timeframe: str = "1h"):
        """Trainingsdaten über den Data Service abrufen."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.data_service_url}/api/v1/ohlcv/{symbol}",
                params={"timeframe": timeframe, "limit": 1000}
            )
            response.raise_for_status()
            return response.json()
```

#### Vorteile dieser Architektur

1. **Zentrale Fehlerbehandlung**: Retry-Logik und Fallbacks an einer Stelle
2. **Einheitliches Caching**: Data Service kann Anfragen cachen
3. **Einfachere Wartung**: API-Änderungen nur im Data Service anpassen
4. **Bessere Überwachung**: Alle Datenzugriffe über einen Service monitorbar
5. **Sicherheit**: Credentials nur im Data Service konfiguriert

---

## Technologie-Stack

### Versionen

| Technologie | Version | Hinweis |
|-------------|---------|---------|
| Python | 3.11+ | Unterstützt bis 3.13 |
| FastAPI | 0.104.0+ | Web-Framework |
| Pydantic | 2.5.0+ | Datenvalidierung |
| PyTorch | 2.0.0+ | ML/DL Framework |
| CUDA | 12.1+ | GPU-Unterstützung |
| FAISS | - | Vektorsuche |
| Ollama | - | LLM-Backend (Llama 3.1) |

### Abhängigkeiten

Neue Abhängigkeiten müssen in `requirements.txt` mit Mindestversion angegeben werden:

```text
package-name>=1.0.0
```

---

## Code-Stil und Formatierung

### Tools

| Tool | Zweck | Konfiguration |
|------|-------|---------------|
| **Black** | Code-Formatierung | Zeilenlänge: 120 |
| **isort** | Import-Sortierung | Black-kompatibel |
| **flake8** | Linting | Optional |

### Formatierung

```bash
# Vor jedem Commit ausführen
black --line-length 120 src/
isort src/
```

### Grundregeln

- **Zeilenlänge**: Maximal 120 Zeichen
- **Einrückung**: 4 Leerzeichen (kein Tab)
- **Leerzeilen**: 2 zwischen Top-Level-Definitionen, 1 zwischen Methoden
- **Trailing Commas**: Bei mehrzeiligen Strukturen verwenden
- **Anführungszeichen**: Doppelte Anführungszeichen (`"`) bevorzugt

---

## Namenskonventionen

### Übersicht

| Element | Konvention | Beispiel |
|---------|------------|----------|
| Klassen | PascalCase | `AnalysisService`, `ForecastResult` |
| Funktionen | snake_case | `fetch_market_data()` |
| Methoden (öffentlich) | snake_case | `analyze()`, `generate_forecast()` |
| Methoden (privat) | _snake_case | `_calculate_indicators()` |
| Konstanten | UPPER_SNAKE_CASE | `DEFAULT_LOOKBACK_DAYS` |
| Variablen | snake_case | `market_data`, `symbol_list` |
| Enums | PascalCase/UPPER | `TradeDirection.LONG` |
| Module | snake_case | `analysis_service.py` |
| Packages | snake_case | `rag_data_sources` |

### Beispiele

```python
# Konstanten
BASE_VERSION = "1.0.0"
DEFAULT_TIMEOUT = 30

# Klassen
class NHITSTrainingService:
    """Service für NHITS-Modell-Training."""

    def __init__(self):
        self._model_cache = {}  # Privat

    def train_model(self, symbol: str) -> TrainingResult:
        """Öffentliche Methode."""
        pass

    def _prepare_data(self, df: pd.DataFrame) -> torch.Tensor:
        """Private Hilfsmethode."""
        pass

# Enums
class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
```

---

## Import-Organisation

### Reihenfolge

Imports werden in folgender Reihenfolge gruppiert (durch Leerzeile getrennt):

1. **Standard Library**
2. **Third-Party Packages**
3. **Lokale Imports**

### Beispiel

```python
# 1. Standard Library
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# 2. Third-Party
import httpx
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

# 3. Lokale Imports
from ..config import settings
from ..models.trading_data import TimeSeriesData
from ..services import AnalysisService
```

### Regeln

- **Absolute Imports** für Third-Party Packages
- **Relative Imports** für lokale Module innerhalb des Projekts
- **Keine Wildcard-Imports** (`from module import *`)
- **Explizite Imports** bevorzugen (`from module import Class`)

---

## Typisierung

### Grundsätze

- **Vollständige Type Hints** auf allen Funktionen und Methoden
- **Pydantic Models** für alle API-Requests und -Responses
- **Optional[T]** für nullable Werte
- **Kein Any** außer in begründeten Ausnahmefällen

### Beispiele

```python
from typing import Dict, List, Optional, Union

def fetch_market_data(
    symbol: str,
    timeframe: str = "1h",
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Marktdaten abrufen."""
    pass

async def analyze_symbol(
    symbol: str,
    include_forecast: bool = True
) -> AnalysisResult:
    """Symbol analysieren."""
    pass
```

### Pydantic Models

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ForecastRequest(BaseModel):
    """Request für Forecast-Generierung."""

    symbol: str = Field(..., description="Trading-Symbol (z.B. BTCUSD)")
    timeframe: str = Field(default="1h", description="Zeitrahmen")
    horizon: int = Field(default=24, ge=1, le=168, description="Vorhersagehorizont")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSD",
                "timeframe": "1h",
                "horizon": 24
            }
        }
```

---

## Async-Patterns

### Grundregeln

1. **Async für I/O-Operationen** (API-Calls, Datei-Operationen)
2. **ThreadPoolExecutor für CPU-intensive Tasks** (Training, Berechnungen)
3. **Context Manager** für Ressourcen-Cleanup

### Beispiele

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Async für I/O
async def fetch_data(symbol: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/data/{symbol}")
        return response.json()

# ThreadPoolExecutor für CPU-bound
async def train_model_async(symbol: str) -> TrainingResult:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            self._train_model_sync,
            symbol
        )
    return result

# Parallele Ausführung
async def fetch_multiple_symbols(symbols: List[str]) -> List[dict]:
    tasks = [fetch_data(symbol) for symbol in symbols]
    return await asyncio.gather(*tasks)
```

### Anti-Patterns (vermeiden)

```python
# FALSCH: Blocking Call in async Funktion
async def bad_example():
    time.sleep(5)  # Blockiert den Event Loop!

# RICHTIG:
async def good_example():
    await asyncio.sleep(5)
```

---

## API-Design

### Endpoint-Konventionen

- **snake_case** für Pfade und Parameter
- **RESTful** Ressourcen-basiert
- **Thematische Gruppierung** mit APIRouter

```python
from fastapi import APIRouter, HTTPException, status

# Router pro Thema
forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])
training_router = APIRouter(prefix="/training", tags=["Training"])

@forecast_router.get("/{symbol}")
async def get_forecast(
    symbol: str,
    timeframe: str = "1h",
    horizon: int = 24
) -> ForecastResponse:
    """
    Forecast für ein Symbol abrufen.

    Args:
        symbol: Trading-Symbol (z.B. BTCUSD)
        timeframe: Zeitrahmen (z.B. 1h, 4h, 1d)
        horizon: Vorhersagehorizont in Perioden

    Returns:
        ForecastResponse mit Vorhersagewerten
    """
    pass
```

### HTTP Status Codes

| Code | Verwendung |
|------|------------|
| 200 | Erfolgreiche Abfrage |
| 201 | Ressource erstellt |
| 400 | Ungültige Anfrage |
| 404 | Ressource nicht gefunden |
| 500 | Interner Serverfehler |
| 503 | Service nicht verfügbar |

### Response-Format

```python
# Erfolg
{
    "success": true,
    "data": { ... },
    "message": "Optional"
}

# Fehler
{
    "success": false,
    "error": "Fehlerbeschreibung",
    "detail": "Optionale Details"
}
```

---

## Fehlerbehandlung

### Grundsätze

1. **Immer loggen** vor dem Raise
2. **HTTPException** mit passendem Status-Code
3. **Aussagekräftige Fehlermeldungen**
4. **Keine sensiblen Daten** in Fehlermeldungen

### Beispiel

```python
from fastapi import HTTPException, status
from loguru import logger

async def get_forecast(symbol: str) -> ForecastResponse:
    try:
        model = self._load_model(symbol)
        if model is None:
            logger.warning(f"Kein Modell gefunden für {symbol}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Kein trainiertes Modell für Symbol {symbol} verfügbar"
            )

        forecast = await self._generate_forecast(model, symbol)
        return ForecastResponse(data=forecast)

    except HTTPException:
        raise  # HTTPExceptions durchreichen
    except Exception as e:
        logger.error(f"Fehler bei Forecast-Generierung für {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interner Fehler bei der Forecast-Generierung"
        )
```

### Retry-Pattern (mit tenacity)

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def fetch_with_retry(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
```

---

## Logging

### Logger: Loguru

```python
from loguru import logger

# Log-Levels
logger.debug("Detaillierte Debug-Information")
logger.info("Allgemeine Information")
logger.warning("Warnung, die beachtet werden sollte")
logger.error("Fehler aufgetreten")
logger.critical("Kritischer Fehler")
```

### Konventionen

```python
# Mit Kontext
logger.info(f"Starte Training für {symbol} mit {len(data)} Datenpunkten")

# Bei Fehlern: Exception-Info mitloggen
try:
    result = process_data(data)
except Exception as e:
    logger.error(f"Verarbeitung fehlgeschlagen für {symbol}: {e}")
    raise

# Strukturierte Logs für wichtige Events
logger.info(
    "Training abgeschlossen",
    extra={
        "symbol": symbol,
        "duration_seconds": duration,
        "loss": final_loss
    }
)
```

### Log-Konfiguration

- **Level**: Via `LOG_LEVEL` Environment-Variable
- **Rotation**: Täglich mit 7-Tage-Retention
- **Format**: Timestamp, Level, Module, Message

---

## Konfiguration

### Pydantic Settings

```python
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field

class Settings(BaseSettings):
    """Anwendungskonfiguration."""

    # API-Verbindungen
    easyinsight_api_url: str = Field(
        default="http://10.1.19.102:3000/api",
        description="EasyInsight API URL"
    )

    # LLM-Konfiguration
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")

    # Training-Parameter
    nhits_batch_size: int = Field(default=32, ge=1)
    nhits_learning_rate: float = Field(default=0.001, gt=0)

    @computed_field
    @property
    def device(self) -> str:
        """GPU automatisch erkennen."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

### Environment-Variablen

- **Produktionskonfiguration**: `.env` (git-ignored)
- **Vorlage**: `.env.example` (git-tracked)
- **Naming**: UPPER_SNAKE_CASE

```bash
# .env.example
EASYINSIGHT_API_URL=http://localhost:3000/api
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
LOG_LEVEL=INFO
```

---

## Testing

### Framework: pytest + pytest-asyncio

```bash
pip install pytest pytest-asyncio pytest-cov httpx
```

### Struktur

```
tests/
├── conftest.py              # Fixtures
├── unit/
│   ├── test_services/
│   │   ├── test_analysis_service.py
│   │   └── test_forecast_service.py
│   └── test_models/
│       └── test_trading_data.py
└── integration/
    └── test_api/
        └── test_endpoints.py
```

### Beispiele

```python
# tests/conftest.py
import pytest
from httpx import AsyncClient
from src.api.routes import app

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

# tests/unit/test_services/test_forecast_service.py
import pytest
from src.services.forecast_service import ForecastService

class TestForecastService:

    @pytest.fixture
    def service(self):
        return ForecastService()

    @pytest.mark.asyncio
    async def test_generate_forecast_success(self, service):
        result = await service.generate_forecast("BTCUSD", horizon=24)
        assert result is not None
        assert len(result.predictions) == 24

    @pytest.mark.asyncio
    async def test_generate_forecast_invalid_symbol(self, service):
        with pytest.raises(ValueError):
            await service.generate_forecast("INVALID", horizon=24)
```

### Test-Ausführung

```bash
# Alle Tests
pytest

# Mit Coverage
pytest --cov=src --cov-report=html

# Nur Unit-Tests
pytest tests/unit/

# Verbose
pytest -v
```

---

## Git-Workflow

### Branch-Strategie

```
main                    # Produktionscode
├── feature/xxx         # Neue Features
├── fix/xxx             # Bugfixes
├── refactor/xxx        # Refactoring
└── docs/xxx            # Dokumentation
```

### Commit-Konvention (Conventional Commits)

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Types

| Type | Beschreibung |
|------|--------------|
| `feat` | Neues Feature |
| `fix` | Bugfix |
| `refactor` | Code-Refactoring |
| `docs` | Dokumentation |
| `test` | Tests |
| `chore` | Wartung, Dependencies |
| `perf` | Performance-Verbesserung |

#### Scopes

| Scope | Beschreibung |
|-------|--------------|
| `nhits` | NHITS-Training/Forecasting |
| `rag` | RAG-System |
| `llm` | LLM-Integration |
| `api` | API-Endpunkte |
| `config` | Konfiguration |
| `docker` | Container/Deployment |

#### Beispiele

```bash
feat(nhits): add multi-timeframe support for training
fix(api): handle missing symbol gracefully in forecast endpoint
refactor(services): extract common HTTP client logic
docs(api): add OpenAPI examples for training endpoints
test(forecast): add unit tests for horizon validation
chore: update dependencies to latest versions
```

### Pre-Commit Checks

```bash
# Vor jedem Commit
black --check src/
isort --check-only src/
pytest tests/unit/
```

---

## Dokumentation

### Code-Dokumentation

#### Docstrings (Google Style)

```python
def train_model(
    self,
    symbol: str,
    timeframe: str = "1h",
    epochs: int = 100
) -> TrainingResult:
    """
    Trainiert ein NHITS-Modell für das angegebene Symbol.

    Args:
        symbol: Trading-Symbol (z.B. BTCUSD)
        timeframe: Zeitrahmen für die Daten (z.B. 1h, 4h, 1d)
        epochs: Anzahl der Trainings-Epochen

    Returns:
        TrainingResult mit Modell-Metriken und Pfad

    Raises:
        ValueError: Wenn Symbol ungültig ist
        TrainingError: Wenn Training fehlschlägt

    Example:
        >>> service = NHITSTrainingService()
        >>> result = service.train_model("BTCUSD", epochs=50)
        >>> print(result.loss)
        0.0023
    """
    pass
```

#### Wann Docstrings schreiben

- **Immer**: Öffentliche Klassen, Methoden, Funktionen
- **Immer**: Komplexe private Methoden
- **Nicht nötig**: Triviale Getter/Setter, offensichtliche Hilfsmethoden

### Inline-Kommentare

```python
# GUT: Erklärt WARUM
# Exponential backoff verhindert Rate-Limiting bei der API
await asyncio.sleep(2 ** attempt)

# SCHLECHT: Erklärt WAS (offensichtlich aus dem Code)
# Inkrementiere den Counter
counter += 1
```

### Architektur-Dokumentation

- Neue Features → `docs/` Ordner aktualisieren
- API-Änderungen → README.md aktualisieren
- Deployment-Änderungen → DEPLOYMENT_*.md aktualisieren

---

## Sicherheit

### Grundregeln

1. **Keine Secrets im Code** - Nur via Environment-Variablen
2. **Keine sensiblen Daten loggen** - Passwörter, API-Keys, etc.
3. **Input-Validierung** - Pydantic für alle externen Inputs
4. **Kein SQL** - Projekt verwendet API-basierte Datenzugriffe

### Environment-Variablen

```python
# RICHTIG
api_key = os.getenv("API_KEY")

# FALSCH
api_key = "sk-1234567890abcdef"
```

### Git-Ignore

Folgende Dateien dürfen NIE committed werden:

```gitignore
.env
*.pem
*.key
credentials.json
secrets/
```

### API-Sicherheit

```python
# Rate-Limiting und Timeout immer setzen
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url)
```

---

## Checkliste für Pull Requests

- [ ] Code ist mit Black formatiert
- [ ] Imports sind mit isort sortiert
- [ ] Alle Funktionen haben Type Hints
- [ ] Docstrings für öffentliche APIs vorhanden
- [ ] Keine sensiblen Daten im Code
- [ ] Tests für neue Funktionalität geschrieben
- [ ] Alle Tests bestehen
- [ ] Commit-Messages folgen Konvention
- [ ] README/Docs bei Bedarf aktualisiert

---

## Referenzen

- [FastAPI Dokumentation](https://fastapi.tiangolo.com/)
- [Pydantic Dokumentation](https://docs.pydantic.dev/)
- [PyTorch Dokumentation](https://pytorch.org/docs/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

*Letzte Aktualisierung: Dezember 2024*
