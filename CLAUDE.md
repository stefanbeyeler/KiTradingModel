# KI Trading Model - Claude Code Anweisungen

Dieses Dokument enthält projektspezifische Anweisungen für Claude Code.

## Kritische Architektur-Regeln

### Datenzugriff-Architektur (VERBINDLICH)

```text
Externe APIs (EasyInsight, TwelveData, Yahoo Finance)
                         │
                         ▼
              ┌─────────────────────┐
              │    DATA SERVICE     │  ◄── Einziges Gateway für externe Daten
              │     (Port 3001)     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    REDIS CACHE      │  ◄── Zentraler Cache für alle Services
              │     (Port 6379)     │      512MB, LRU-Eviction, Persistenz
              └──────────┬──────────┘
                         │
     ┌───────┬───────┬───┴───┬───────┬───────┐
     ▼       ▼       ▼       ▼       ▼       ▼
  NHITS    TCN     HMM   Embedder  RAG     LLM
 :3002   :3003   :3004   :3005   :3008   :3009
                                           │
                                      Watchdog
                                       :3010
```

### Verbindliche Regeln

1. **Kein direkter Datenbankzugriff**
   - KEIN `asyncpg`, `psycopg2` oder andere DB-Treiber
   - EasyInsight TimescaleDB nur über die EasyInsight REST API

2. **Data Service als einziges Gateway**
   - NHITS, RAG und LLM Services greifen auf externe Daten NUR über den Data Service zu
   - Verwende `DataGatewayService` (src/services/data_gateway_service.py) als Singleton

3. **Datenquellen-Hierarchie im Data Service**
   - **TwelveData** (primär): OHLC-Daten für alle Timeframes (M1-MN)
   - **EasyInsight** (1. Fallback): Zusätzliche Indikatoren und TimescaleDB-Daten
   - **Yahoo Finance** (2. Fallback): Kostenlose historische Daten

4. **Caching über Redis**
   - Alle Daten werden im Redis-Cache zwischengespeichert
   - Verwende `CacheService` (`src/services/cache_service.py`) für Cache-Zugriffe
   - TTL-Werte sind kategoriebasiert (siehe Caching-Architektur unten)

### Beispiele

```python
# ❌ FALSCH: Direkter API-Aufruf aus NHITS-Service
class NHITSTrainingService:
    async def get_data(self, symbol: str):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://easyinsight-api/ohlcv/{symbol}")
            return response.json()

# ✅ RICHTIG: Über Data Gateway
from .data_gateway_service import data_gateway

class NHITSTrainingService:
    async def get_data(self, symbol: str):
        return await data_gateway.get_historical_data(symbol, limit=1000)
```

## Caching-Architektur (VERBINDLICH)

### Architektur-Übersicht

```text
                    ┌─────────────────────┐
                    │   Redis Cache       │  ◄── Zentraler Cache (Port 6379)
                    │  (trading-redis)    │      512MB, LRU-Eviction
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌──────────┐     ┌──────────┐     ┌──────────┐
       │  Data    │     │  NHITS   │     │   RAG    │
       │ Service  │     │ Service  │     │ Service  │
       └──────────┘     └──────────┘     └──────────┘
```

### Caching-Strategie

**Data Service als Gateway + Cache:**
- Daten werden beim ersten Abruf von externen APIs geholt
- Nach Abruf werden Daten in Redis gecacht
- Nachfolgende Anfragen werden aus dem Cache bedient
- TTL-basierte automatische Invalidierung

### Cache-Kategorien und TTL

| Kategorie | TTL (Sekunden) | Beschreibung |
|-----------|----------------|--------------|
| `MARKET_DATA` | 60 | Echtzeit-Kurse, schnelle Updates |
| `OHLCV` | 300 | Kerzendaten (5 Minuten) |
| `INDICATORS` | 300 | Technische Indikatoren |
| `SYMBOLS` | 3600 | Symbol-Listen (1 Stunde) |
| `METADATA` | 3600 | Metadaten |
| `SENTIMENT` | 900 | Sentiment-Daten (15 Minuten) |
| `ECONOMIC` | 1800 | Wirtschaftskalender (30 Minuten) |
| `ONCHAIN` | 600 | On-Chain Daten (10 Minuten) |
| `TRAINING` | 21600 | Training-Daten (6 Stunden) |

### Verwendung

```python
from src.services.cache_service import cache_service, CacheCategory

# Daten aus Cache holen
cached_data = await cache_service.get(CacheCategory.OHLCV, symbol, params={"limit": 500})

# Daten in Cache speichern
await cache_service.set(CacheCategory.OHLCV, data, symbol, params={"limit": 500})

# Cache-Kategorie löschen
await cache_service.clear_category(CacheCategory.MARKET_DATA)

# Cache-Statistiken
stats = cache_service.get_stats()
```

### Redis-Konfiguration

```yaml
# docker-compose.microservices.yml
redis:
  image: redis:7-alpine
  container_name: trading-redis
  command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
  ports:
    - "6379:6379"
```

**Wichtige Einstellungen:**
- `maxmemory 512mb`: Maximaler Speicherverbrauch
- `maxmemory-policy allkeys-lru`: Least Recently Used Eviction
- `appendonly yes`: Persistenz aktiviert (optional)

### Fallback-Verhalten

Wenn Redis nicht verfügbar ist:
1. CacheService fällt automatisch auf In-Memory Cache zurück
2. Warnung wird geloggt
3. System bleibt funktionsfähig (aber ohne verteilten Cache)

## Swagger UI / OpenAPI Tags (VERBINDLICH)

### Kurze Tag-Beschreibungen

OpenAPI Tag-Beschreibungen müssen **kurz und einzeilig** sein (max. 60 Zeichen), damit sie in der Swagger UI korrekt angezeigt werden.

```python
# ❌ FALSCH: Lange, mehrzeilige Beschreibungen
openapi_tags = [
    {
        "name": "2. Training",
        "description": """Training-Jobs starten, überwachen und verwalten.

**Features:**
- Job-Management
- Scheduler
- Model-Versioning
"""
    },
]

# ✅ RICHTIG: Kurze, einzeilige Beschreibungen
openapi_tags = [
    {
        "name": "1. System",
        "description": "Health checks und Service-Informationen"
    },
    {
        "name": "2. Training",
        "description": "Training-Jobs starten, überwachen und verwalten"
    },
]
```

### Gründe

- Lange Beschreibungen werden in Swagger UI inline neben dem Tag-Namen angezeigt
- Dies macht die Tags schwer lesbar und die Navigation unübersichtlich
- Kurze Beschreibungen (1 Satz, max. 60 Zeichen) halten die UI sauber

## Code-Stil

- **Python 3.11+**
- **Async-First**: Alle I/O-Operationen asynchron
- **Type Hints**: Vollständig auf allen Funktionen
- **Pydantic**: Für alle API-Requests/Responses
- **Zeilenlänge**: Max 120 Zeichen
- **Imports**: Standard → Third-Party → Lokal (getrennt durch Leerzeilen)

## Microservices Ports

| Service | Port | Swagger UI | GPU |
|---------|------|------------|-----|
| Frontend (Dashboard) | 3000 | - | - |
| Data Service | 3001 | /docs | - |
| NHITS Service | 3002 | /docs | CUDA |
| TCN-Pattern Service | 3003 | /docs | CUDA |
| HMM-Regime Service | 3004 | /docs | - |
| Embedder Service | 3005 | /docs | CUDA |
| Candlestick Service | 3006 | /docs | - |
| **Redis Cache** | **6379** | - | - |
| RAG Service | 3008 | /docs | CUDA |
| LLM Service | 3009 | /docs | CUDA |
| Watchdog Service | 3010 | /docs | - |
| TCN-Train Service | 3013 | /docs | CUDA |
| Candlestick-Train Service | 3016 | /docs | CUDA |

## GPU-Konfiguration (NVIDIA Thor / Jetson)

### Hardware-Umgebung

- **System**: NVIDIA Tegra (JetPack R38)
- **Architektur**: aarch64 (ARM64)
- **GPU**: NVIDIA Thor mit Compute Capability **sm_110**
- **CUDA Version**: 13.0

### PyTorch Kompatibilität (KRITISCH)

**NVIDIA Thor (sm_110) wird von PyTorch noch NICHT unterstützt!**

| PyTorch Version | Unterstützte Compute Capabilities |
|-----------------|-----------------------------------|
| 2.5.x           | sm_50, sm_80, sm_86, sm_89, sm_90, sm_90a |
| 2.6+ (erwartet) | sm_110 Support geplant |

### Vor Container-Umbau IMMER prüfen

Bevor GPU-bezogene Dockerfile-Änderungen vorgenommen werden:

```bash
# 1. Aktuelle PyTorch-Version auf PyPI prüfen
pip index versions torch

# 2. CUDA Support für sm_110 prüfen (auf dem Server)
ssh sbeyeler@10.1.19.101 "docker exec <container> python3 -c \"
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
\""

# 3. PyTorch Release Notes auf sm_110 Support prüfen
# https://pytorch.org/get-started/locally/
# https://github.com/pytorch/pytorch/releases
```

### Aktueller Status (Stand: Dezember 2024)

- **TCN, NHITS, Embedder, RAG, LLM**: Laufen auf **CPU**
- **Grund**: PyTorch 2.5.x unterstützt sm_110 nicht
- **Training**: ~4x langsamer als mit GPU, aber funktional

### Wenn PyTorch 2.6+ verfügbar ist

Sobald sm_110 Support bestätigt ist, Dockerfiles anpassen:

```dockerfile
# Option 1: NVIDIA L4T Base Image (bevorzugt für Jetson)
FROM nvcr.io/nvidia/l4t-pytorch:r38.x.x-pthX.X-py3

# Option 2: CUDA Base Image mit PyTorch
FROM nvidia/cuda:13.0-runtime-ubuntu22.04
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### Betroffene Dockerfiles

- `docker/services/tcn/Dockerfile`
- `docker/services/nhits/Dockerfile`
- `docker/services/embedder/Dockerfile`
- `docker/services/rag/Dockerfile`
- `docker/services/llm/Dockerfile`

## API-Dokumentation (Swagger UI)

Die Swagger-Dokumentationen aller Microservices sind zentral über das Dashboard erreichbar:

**Dashboard URL:** `http://10.1.19.101:3000/`

Alle Services sind via Nginx-Proxy unter folgenden Pfaden erreichbar:

| Service | Swagger UI URL | Proxy-Pfad |
|---------|----------------|------------|
| Data Service | http://10.1.19.101:3000/data/docs | /data/* |
| NHITS Service | http://10.1.19.101:3000/nhits/docs | /nhits/* |
| TCN-Pattern Service | http://10.1.19.101:3000/tcn/docs | /tcn/* |
| HMM-Regime Service | http://10.1.19.101:3000/hmm/docs | /hmm/* |
| Embedder Service | http://10.1.19.101:3000/embedder/docs | /embedder/* |
| RAG Service | http://10.1.19.101:3000/rag/docs | /rag/* |
| LLM Service | http://10.1.19.101:3000/llm/docs | /llm/* |
| Watchdog Service | http://10.1.19.101:3000/watchdog/docs | /watchdog/* |

Die Swagger UI bietet:
- Interaktive API-Dokumentation
- Request/Response-Schemas
- Direkte API-Tests im Browser

## TwelveData Technical Indicators

Der `TwelveDataService` (`src/services/twelvedata_service.py`) bietet Zugriff auf technische Indikatoren:

### Verfügbare Indikatoren

| Kategorie | Indikatoren |
|-----------|-------------|
| **Trend/Moving Averages** | SMA, EMA, WMA, DEMA, TEMA, KAMA, MAMA, T3, TRIMA, VWAP |
| **Momentum** | RSI, MACD, Stoch, StochRSI, Williams %R, CCI, CMO, ROC, MOM, PPO, APO, Aroon, AroonOsc, BOP, MFI, DX, ADX, ADXR, +DI, -DI, **Connors RSI** |
| **Volatilität** | Bollinger Bands, ATR, NATR, TRange, **Percent B** |
| **Volumen** | OBV, A/D, ADOSC |
| **Trend-Filter** | Supertrend, Ichimoku, Parabolic SAR |
| **ML-Features** | **Linear Regression Slope**, **Hilbert Trend Mode** |

### Neue Indikatoren für ML/NHITS

| Indikator | API-Endpoint | Beschreibung |
|-----------|--------------|--------------|
| **VWAP** | `/twelvedata/vwap/{symbol}` | Volume Weighted Average Price - Institutioneller Benchmark |
| **Connors RSI** | `/twelvedata/crsi/{symbol}` | 3-Komponenten RSI für Mean-Reversion |
| **Linear Reg Slope** | `/twelvedata/linearregslope/{symbol}` | Trendstärke als numerischer Wert |
| **Hilbert Trend Mode** | `/twelvedata/ht_trendmode/{symbol}` | Trend (1) vs. Range (0) Klassifikation |
| **Percent B** | `/twelvedata/percent_b/{symbol}` | BBands-Position normalisiert (0-1) |

### Beispiel

```python
from src.services.twelvedata_service import twelvedata_service

# Einzelner Indikator
data = await twelvedata_service.get_connors_rsi("AAPL", interval="1day")

# Mehrere Indikatoren
data = await twelvedata_service.get_multiple_indicators(
    symbol="BTCUSD",
    indicators=["rsi", "crsi", "linearregslope", "ht_trendmode"],
    interval="1h"
)
```

## External Data Sources (Gateway)

Der Data Service stellt 9 externe Datenquellen als Gateway für den RAG Service bereit:

### Verfügbare Quellen

| Quelle | Endpoint | Beschreibung |
|--------|----------|--------------|
| **Economic Calendar** | `/external-sources/economic-calendar` | Fed, ECB, CPI, NFP, GDP Events |
| **Sentiment** | `/external-sources/sentiment` | Fear & Greed, Social Media, VIX |
| **On-Chain** | `/external-sources/onchain/{symbol}` | Whale Alerts, Exchange Flows |
| **Orderbook** | `/external-sources/orderbook/{symbol}` | Bid/Ask Walls, Liquidations |
| **Macro** | `/external-sources/macro` | DXY, Bond Yields, Korrelationen |
| **Historical Patterns** | `/external-sources/historical-patterns` | Saisonalität, Drawdowns |
| **Technical Levels** | `/external-sources/technical-levels/{symbol}` | S/R, Fibonacci, Pivots |
| **Regulatory** | `/external-sources/regulatory` | SEC, ETF News, Regulation |
| **EasyInsight** | `/external-sources/easyinsight` | Managed Symbols, MT5 Logs |

### Aggregierte Endpoints

| Endpoint | Beschreibung |
|----------|--------------|
| `POST /external-sources/fetch-all` | Alle Quellen parallel abrufen |
| `POST /external-sources/trading-context/{symbol}` | Kompletter Trading-Kontext |

### Architektur

```text
RAG Service (Port 3008)
        │
        │ DataFetcherProxy
        ▼
Data Service (Port 3001)
        │
        │ /api/v1/external-sources/*
        ▼
External APIs (Alternative.me, EasyInsight, etc.)
```

Der RAG Service nutzt den `DataFetcherProxy` um Daten vom Data Service Gateway abzurufen.

## Testing (Watchdog-integriert)

Die komplette Testing-Funktionalität ist im **Watchdog Service** zentralisiert.

### Test-Ausführung

```bash
# Via Makefile
make test-smoke          # Schnelle Health-Checks
make test-full           # Alle Tests
make test-service SERVICE=data  # Service-spezifisch

# Via curl
curl -X POST http://10.1.19.101:3010/api/v1/tests/run/smoke
```

### Test-Kategorien

| Kategorie | Beschreibung |
|-----------|--------------|
| **smoke** | Health-Checks (< 1s) |
| **api** | Endpoint-Tests (< 5s) |
| **contract** | Schema-Validierung |
| **integration** | Service-übergreifend |

### Implementierung

- `src/services/watchdog_app/services/test_definitions.py` - ~90 Test-Definitionen
- `src/services/watchdog_app/services/test_runner.py` - Test-Runner
- `src/services/watchdog_app/api/routes.py` - API-Endpoints

Vollständige Dokumentation: `docs/TESTING_PROPOSAL.md`

## Commit-Konvention

```text
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`
Scopes: `nhits`, `tcn`, `hmm`, `embedder`, `rag`, `llm`, `data`, `watchdog`, `frontend`, `api`, `config`, `docker`

## Zeitzonen-Handling (VERBINDLICH)

### Grundregeln

1. **Speicherung immer in UTC (ISO 8601)**
   - Alle Timestamps intern als UTC speichern
   - Format: `2024-01-15T10:30:45Z` oder `2024-01-15T10:30:45+00:00`
   - Niemals naive datetime-Objekte verwenden

2. **Präsentation in lokaler Zeit**
   - Alle Timestamps für die Anzeige in die konfigurierte Zeitzone konvertieren
   - Zeitzone ist konfigurierbar (Standard: `Europe/Zurich`)

3. **Konfiguration (EINZIGE QUELLE)**
   - Zeitzone wird in `src/config/settings.py` als `display_timezone` definiert
   - Umgebungsvariable: `DISPLAY_TIMEZONE`
   - **NIEMALS** `"Europe/Zurich"` oder andere Zeitzonen hardcoden
   - **IMMER** `settings.display_timezone` importieren und verwenden

### Beispiele

```python
from src.utils.timezone_utils import to_utc, to_display_timezone, format_for_display

# ❌ FALSCH: Naive datetime oder inkonsistente Zeitzonen
timestamp = datetime.now()  # Naive, keine Zeitzone
timestamp_str = data.get("snapshot_time")  # Direkt durchreichen

# ✅ RICHTIG: UTC für Speicherung
timestamp = datetime.now(timezone.utc)
utc_timestamp = to_utc(data.get("snapshot_time"))

# ✅ RICHTIG: Lokale Zeit für Anzeige
display_time = to_display_timezone(utc_timestamp)
formatted = format_for_display(utc_timestamp)  # "20.12.2024, 10:18:00 CET"
```

### API-Responses

- Interne Felder (`*_utc`): ISO 8601 UTC-Format
- Anzeige-Felder (`*_display`): Lokalisiertes Format mit Zeitzonenangabe

## Dokumentation

Vollständige Entwicklungsrichtlinien: `DEVELOPMENT_GUIDELINES.md`
