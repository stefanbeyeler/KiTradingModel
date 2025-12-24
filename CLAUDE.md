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

3. **Fallback-Logik nur im Data Service**
   - TwelveData als Backup für EasyInsight gehört NUR in den Data Service

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
| RAG Service | 3008 | /docs | CUDA |
| LLM Service | 3009 | /docs | CUDA |
| Watchdog Service | 3010 | /docs | - |

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

3. **Konfiguration**
   - Zeitzone wird in `src/core/config.py` als `DISPLAY_TIMEZONE` definiert
   - Umgebungsvariable: `DISPLAY_TIMEZONE`

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
