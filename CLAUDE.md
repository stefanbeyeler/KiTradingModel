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
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│    REDIS CACHE      │       │   TIMESCALEDB       │
│     (Port 6379)     │       │   (10.1.19.102)     │
│   Hot Data (TTL)    │       │  Persistent Storage │
└──────────┬──────────┘       └─────────────────────┘
           │
     ┌─────┴─────┬───────┬───────┬───────┬───────┐
     ▼           ▼       ▼       ▼       ▼       ▼
  NHITS        TCN     HMM   Embedder  RAG     LLM
 :3002       :3003   :3004   :3005   :3008   :3009
                                               │
                                          Watchdog
                                           :3010
```

### 3-Layer-Caching-Strategie

```text
Anfrage → 1. Redis Cache → 2. TimescaleDB → 3. Externe APIs
                ↓              ↓                 ↓
            (Hot Data)    (Persistent)     (Fresh Data)
```

### Verbindliche Regeln

1. **Datenbankzugriff nur im Data Service**
   - NUR der Data Service verwendet `asyncpg` für TimescaleDB
   - Andere Services greifen auf Daten NUR über den Data Service HTTP API zu
   - Verwende `DataRepository` für 3-Layer-Caching

2. **Data Service als einziges Gateway**
   - NHITS, RAG und LLM Services greifen auf externe Daten NUR über den Data Service zu
   - Verwende `DataGatewayService` (src/services/data_gateway_service.py) als Singleton

3. **Datenquellen-Hierarchie im Data Service**
   - **Redis Cache** (Layer 1): Schneller Zugriff auf Hot Data
   - **TimescaleDB** (Layer 2): Persistente Speicherung aller historischen Daten
   - **TwelveData** (Layer 3 primär): OHLC-Daten für alle Timeframes (M1-MN)
   - **EasyInsight** (Layer 3 Fallback): Zusätzliche Indikatoren
   - **Yahoo Finance** (Layer 3 Fallback 2): Kostenlose historische Daten

4. **Caching über Redis + TimescaleDB**
   - Hot Data wird im Redis-Cache zwischengespeichert (TTL-basiert)
   - Alle Daten werden in TimescaleDB persistent gespeichert
   - Verwende `DataRepository` (`src/services/data_repository.py`) für 3-Layer-Zugriff
   - Verwende `CacheService` (`src/services/cache_service.py`) für reine Cache-Zugriffe

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

### Architektur-Übersicht (3-Layer)

```text
┌───────────────────────────────────────────────────────┐
│                    DATA SERVICE                        │
│                     (Port 3001)                        │
└───────────────────────────┬───────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  REDIS CACHE    │ │  TIMESCALEDB    │ │ EXTERNE APIs    │
│  (Layer 1)      │ │  (Layer 2)      │ │ (Layer 3)       │
│  Hot Data       │ │  Persistent     │ │ Fresh Data      │
│  trading-redis  │ │  10.1.19.102    │ │ TwelveData, etc │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │
┌───────┴───────┬───────┬───────┬───────┬───────┐
▼               ▼       ▼       ▼       ▼       ▼
NHITS          TCN     HMM   Embedder  RAG     LLM
```

### 3-Layer-Caching-Strategie

**Layer 1: Redis Cache (Hot Data)**
- Schneller Zugriff auf häufig abgerufene Daten
- TTL-basierte automatische Invalidierung
- 512MB LRU-Eviction

**Layer 2: TimescaleDB (Persistent)**
- Persistente Speicherung aller historischen Daten
- Hypertables für optimierte Zeitreihenspeicherung
- Automatische Kompression und Retention

**Layer 3: Externe APIs (Fresh Data)**
- TwelveData, EasyInsight, Yahoo Finance
- Nur bei Cache-Miss und DB-Miss abgerufen
- Daten werden nach Abruf in Layer 1 & 2 gespeichert

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

## Timeframe-Standardisierung (VERBINDLICH)

### Zentrale Konfiguration

Alle Timeframes werden über die zentrale Konfiguration in `src/config/timeframes.py` standardisiert.
Der Data Service normalisiert alle Timeframes beim Laden und stellt den nachgelagerten Services immer konsistente Bezeichnungen zur Verfügung.

### Standard-Format

| Timeframe | Standard | Alternativen (automatisch gemappt) |
|-----------|----------|-----------------------------------|
| 1 Minute | `M1` | `1m`, `1min`, `1minute` |
| 5 Minuten | `M5` | `5m`, `5min`, `5minutes` |
| 15 Minuten | `M15` | `15m`, `15min`, `15minutes` |
| 30 Minuten | `M30` | `30m`, `30min`, `30minutes` |
| 45 Minuten | `M45` | `45m`, `45min`, `45minutes` |
| 1 Stunde | `H1` | `1h`, `1hour`, `60m` |
| 2 Stunden | `H2` | `2h`, `2hour` |
| 4 Stunden | `H4` | `4h`, `4hour` |
| 1 Tag | `D1` | `1d`, `1day`, `daily` |
| 1 Woche | `W1` | `1wk`, `1week`, `weekly` |
| 1 Monat | `MN` | `1mo`, `1month`, `monthly` |

### Verwendung

```python
from src.config.timeframes import (
    Timeframe,
    normalize_timeframe,
    normalize_timeframe_safe,
    to_twelvedata,
    to_yfinance,
    get_candles_per_day,
    calculate_limit_for_days,
)

# Normalisieren eines beliebigen Timeframe-Formats
tf = normalize_timeframe("1h")  # Returns: Timeframe.H1
tf = normalize_timeframe("daily")  # Returns: Timeframe.D1

# Sicher normalisieren mit Fallback
tf = normalize_timeframe_safe("invalid", Timeframe.H1)  # Returns: Timeframe.H1

# Konvertieren zu TwelveData-Format
td_interval = to_twelvedata(Timeframe.H1)  # Returns: "1h"

# Konvertieren zu Yahoo Finance-Format
yf_interval = to_yfinance(Timeframe.D1)  # Returns: "1d"

# Kerzen pro Tag berechnen
candles = get_candles_per_day(Timeframe.H1)  # Returns: 24.0

# Limit für Anzahl Tage berechnen
limit = calculate_limit_for_days(Timeframe.H1, days=30)  # Returns: 720
```

### Verbindliche Regeln

1. **Niemals Timeframes hardcoden**
   - ❌ `interval_map = {"H1": "1h", "D1": "1day"}`
   - ✅ `from src.config.timeframes import to_twelvedata`

2. **Immer normalisieren vor Verarbeitung**
   - ❌ `timeframe.upper()`
   - ✅ `normalize_timeframe(timeframe).value`

3. **Cache-Keys immer mit Standard-Format**
   - ❌ `cache_key = f"{symbol}_{user_input_timeframe}"`
   - ✅ `cache_key = f"{symbol}_{normalize_timeframe(timeframe).value}"`

4. **API-Responses mit Standard-Timeframe**
   - Alle Responses enthalten `"timeframe": "H1"` im Standard-Format
   - Clients können beliebige Formate senden, erhalten aber immer das Standard-Format zurück

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

### Deutsche Schreibweisen (UI-Texte)

In allen HTML-Dateien und UI-Texten sind deutsche Umlaute korrekt zu verwenden:

| Korrekt | Falsch |
|---------|--------|
| `zurücksetzen` | `zurucksetzen` |
| `Übersicht` | `Ubersicht` |
| `für` | `fur` |
| `Größe` | `Grosse` |

**Beispiele:**
- `Filter zurücksetzen` (nicht `Filter zurucksetzen`)
- `Datenquellen-Übersicht` (nicht `Datenquellen-Ubersicht`)

## Candlestick Pattern Charts (VERBINDLICH)

### Chart-Kontext-Regel

Bei der Darstellung von erkannten Candlestick-Patterns muss IMMER ein konsistenter Kontext angezeigt werden:

- **5 Kerzen VOR dem Pattern** (Kontext für Trend-Erkennung)
- **Das Pattern selbst** (1-5 Kerzen je nach Typ)
- **5 Kerzen NACH dem Pattern** (Bestätigung/Entwicklung)

### Implementierung

```javascript
// Pattern-Timestamp zeigt auf die LETZTE Kerze des Patterns
const patternStartIdx = patternIdx - (patternCandleCount - 1);
const candlesBefore = 5;
const candlesAfter = 5;
const startIdx = Math.max(0, patternStartIdx - candlesBefore);
const endIdx = Math.min(allCandles.length, patternIdx + candlesAfter + 1);
```

### Betroffene Funktionen

| Funktion | Datei | Beschreibung |
|----------|-------|--------------|
| `showPatternModal()` | config-candlestick.html | Modal für Pattern-Details |
| `fetchOhlcForRevalidation()` | config-candlestick.html | Daten für Revalidierungs-Charts |
| `loadRevalDetailChart()` | config-candlestick.html | Detail-Chart in Revalidierung |
| `loadRevalidationItemChart()` | config-candlestick.html | Item-Charts in Revalidierungs-Liste |

### Multi-Kerzen-Patterns

Für Patterns mit mehreren Kerzen muss `getPatternCandleCount()` verwendet werden:

| Pattern-Typ | Kerzenanzahl |
|-------------|-------------|
| Hammer, Doji, etc. | 1 |
| Engulfing, Harami, etc. | 2 |
| Morning/Evening Star, Three Inside, etc. | 3 |
| Rising/Falling Three Methods, Tower | 5 |

### Beispiel

Ein **Three Inside Down** Pattern (3 Kerzen) wird angezeigt als:
- 5 Kontext-Kerzen (Trend vor dem Pattern)
- 3 Pattern-Kerzen (hervorgehoben)
- 5 Kontext-Kerzen (Entwicklung nach dem Pattern)
- **Gesamt: 13 Kerzen im Chart**

## Microservices Ports

### Inference Services (High Priority)

| Service | Port | Swagger UI | GPU | Beschreibung |
|---------|------|------------|-----|--------------|
| Frontend (Dashboard) | 3000 | - | - | Nginx Proxy + Dashboard |
| Data Service | 3001 | /docs | - | Zentrales Data Gateway |
| NHITS Service | 3002 | /docs | CUDA | Price Forecasts (Inference) |
| TCN-Pattern Service | 3003 | /docs | CUDA | Chart Patterns (Inference) |
| HMM-Regime Service | 3004 | /docs | - | Regime Detection (Inference) |
| Embedder Service | 3005 | /docs | CUDA | Feature Embeddings |
| Candlestick Service | 3006 | /docs | - | Candlestick Patterns (Inference) |
| **Redis Cache** | **6379** | - | - | Zentraler Cache |
| RAG Service | 3008 | /docs | CUDA | Vector Search |
| LLM Service | 3009 | /docs | CUDA | LLM Analysis |
| Watchdog Service | 3010 | /docs | - | Monitoring + Training Orchestrator |

### Training Services (Low Priority, orchestriert von Watchdog)

| Service | Port | Swagger UI | GPU | Beschreibung |
|---------|------|------------|-----|--------------|
| NHITS-Train Service | 3012 | /docs | CUDA | NHITS Model Training |
| TCN-Train Service | 3013 | /docs | CUDA | TCN Pattern Training |
| HMM-Train Service | 3014 | /docs | - | HMM + LightGBM Training |
| Candlestick-Train Service | 3016 | /docs | - | Candlestick Pattern Training |

### Training-Architektur

```text
┌─────────────────────────────────────────────────────────┐
│                   WATCHDOG SERVICE                       │
│                 Training Orchestrator                    │
│                     (Port 3010)                          │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  NHITS    │   │   TCN     │   │   HMM     │   │Candlestick│
   │  TRAIN    │   │  TRAIN    │   │  TRAIN    │   │  TRAIN    │
   │  :3012    │   │  :3013    │   │  :3014    │   │  :3016    │
   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │               │
         │ Shared Volume │               │               │
         ▼               ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  NHITS    │   │   TCN     │   │   HMM     │   │Candlestick│
   │ INFERENCE │   │ INFERENCE │   │ INFERENCE │   │ INFERENCE │
   │  :3002    │   │  :3003    │   │  :3004    │   │  :3006    │
   └───────────┘   └───────────┘   └───────────┘   └───────────┘
```

**Vorteile der separaten Training-Container:**
- Training blockiert keine Inference-API-Requests
- Unabhängige Skalierung von Training und Inference
- Low-Priority (`nice -n 19`) für Training-Prozesse
- Zentrale Koordination über Watchdog Orchestrator

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

## Training Orchestrator (Watchdog-integriert)

Der **Training Orchestrator** im Watchdog Service koordiniert alle ML-Model-Trainings zentral.

### Training-API

```bash
# Training für einen Service starten
curl -X POST http://10.1.19.101:3010/api/v1/training/queue \
  -H "Content-Type: application/json" \
  -d '{"service": "nhits", "symbols": ["BTCUSD", "EURUSD"], "priority": "normal"}'

# Training für alle Services starten
curl -X POST http://10.1.19.101:3010/api/v1/training/train-all

# Training-Status abfragen
curl http://10.1.19.101:3010/api/v1/training/status

# Alle Training-Services Status
curl http://10.1.19.101:3010/api/v1/training/services
```

### Training-Services

| Service | Modell-Typ | Beschreibung |
|---------|------------|--------------|
| **nhits** | Neural Network | Preisvorhersage (H1, D1) |
| **tcn** | Temporal CNN | Chart-Pattern-Erkennung |
| **hmm** | HMM + LightGBM | Regime-Detection + Signal Scorer |
| **candlestick** | CNN | Candlestick-Pattern-Erkennung |

### Priority-Levels

| Priority | Beschreibung |
|----------|--------------|
| **low** | Background-Training, niedrige CPU-Priorität |
| **normal** | Standard-Priorität (default) |
| **high** | Bevorzugte Ausführung |
| **critical** | Sofortige Ausführung |

### Scheduled Training

```bash
# Tägliches Training einrichten
curl -X POST http://10.1.19.101:3010/api/v1/training/schedules \
  -H "Content-Type: application/json" \
  -d '{"service": "nhits", "schedule": "daily"}'

# Schedules anzeigen
curl http://10.1.19.101:3010/api/v1/training/schedules
```

### Implementierung

- `src/services/watchdog_app/services/training_orchestrator.py` - Orchestrator-Service
- `src/services/watchdog_app/api/training_routes.py` - API-Endpoints

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

## API-Dokumentation aktualisieren (VERBINDLICH)

Nach jeder Änderung an API-Endpunkten **müssen** folgende Dateien aktualisiert werden:

### 1. Service Info-Seite

Jeder Service hat eine Info-Seite unter `docker/services/frontend/html/service-{name}.html`:

| Service | Info-Seite |
|---------|------------|
| Data Service | `service-data.html` |
| NHITS Service | `service-nhits.html` |
| TCN-Pattern Service | `service-tcn.html` |
| HMM-Regime Service | `service-hmm.html` |
| Embedder Service | `service-embedder.html` |
| Candlestick Service | `service-candlestick.html` |
| RAG Service | `service-rag.html` |
| LLM Service | `service-llm.html` |
| Watchdog Service | `service-watchdog.html` |

**Aktualisieren bei:**
- Neuen Endpoints
- Geänderten Endpoint-Pfaden
- Entfernten Endpoints
- Neuen Features/Funktionen

### 2. OpenAPI/Swagger Dokumentation

Die Swagger-Dokumentation wird automatisch aus den FastAPI-Routern generiert. Achte auf:

- **Aussagekräftige Docstrings** in allen Endpoint-Funktionen
- **Response-Models** mit Pydantic für korrekte Schema-Dokumentation
- **Tags** für logische Gruppierung (kurz, max. 60 Zeichen)
- **Query-Parameter-Beschreibungen** mit `Query(..., description="...")`

```python
@router.get("/example/{symbol}", response_model=ExampleResponse, tags=["1. Examples"])
async def get_example(
    symbol: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Anzahl der Ergebnisse")
):
    """
    Kurze Beschreibung des Endpoints.

    Gibt Beispieldaten für das angegebene Symbol zurück.
    """
    ...
```

### 3. Checkliste nach API-Änderungen

```text
[ ] Info-Seite aktualisiert (service-{name}.html)
[ ] Docstrings in Router-Funktionen vorhanden
[ ] Response-Model definiert
[ ] Query-Parameter dokumentiert
[ ] Container neu gestartet (docker restart trading-{name})
[ ] Änderungen committed und gepusht
```

## Dokumentation

Vollständige Entwicklungsrichtlinien: `DEVELOPMENT_GUIDELINES.md`
