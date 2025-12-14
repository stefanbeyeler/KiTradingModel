# EasyInsight API Integration

## Übersicht

Das KI Trading Model nutzt die **EasyInsight API** auf Host `10.1.19.102:3000` als primäre Datenquelle für Zeitreihendaten, technische Indikatoren und Marktinformationen.

**Base URL:** `http://10.1.19.102:3000/api`

---

## 📡 Genutzte Endpoints

### 1. `/symbols` - Symbol-Liste

**Zweck:** Liste aller verfügbaren Trading-Symbole mit Metadaten

**Services:**
- `nhits_training_service.py::get_available_symbols()`
- `analysis_service.py::get_available_symbols()`

**Request:**
```http
GET http://10.1.19.102:3000/api/symbols
```

**Response:**
```json
[
  {
    "symbol": "EURUSD",
    "category": "Forex",
    "count": 13441,
    "earliest": "2025-11-26T23:11:00+01:00",
    "latest": "2025-12-14T07:54:00+01:00"
  },
  ...
]
```

**Verwendung:**
- Symbol-Discovery für NHITS-Training
- Verfügbarkeitsprüfung vor Analyse
- Fallback zu TimescaleDB bei Fehler

---

### 2. `/symbol-data-full/{symbol}` - Historische Zeitreihendaten

**Zweck:** Historische OHLC-Daten + alle technischen Indikatoren

**Services:**
- `nhits_training_service.py::get_training_data()`

**Request:**
```http
GET http://10.1.19.102:3000/api/symbol-data-full/EURUSD?limit=720
```

**Parameter:**
- `limit`: Anzahl der Datenpunkte (z.B. 720 für 30 Tage à 24 Stunden)

**Response:**
```json
{
  "columns": [
    "snapshot_time", "symbol", "h1_open", "h1_high", "h1_low", "h1_close",
    "d1_open", "d1_high", "d1_low", "d1_close",
    "m15_open", "m15_high", "m15_low", "m15_close",
    "rsi", "macd_main", "macd_signal", "adx_main", "adx_plusdi", "adx_minusdi",
    "cci", "sto_main", "sto_signal",
    "bb_upper", "bb_base", "bb_lower", "ma_10",
    "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkoua", "ichimoku_senkoub", "ichimoku_chikou",
    "atr_d1", "range_d1", "atr_pct_d1",
    "bid", "ask", "spread", "spread_pct",
    "strength_4h", "strength_1d", "strength_1w",
    "s1_level_m5", "r1_level_m5",
    "category", "source_file", "import_time"
  ],
  "data": [
    {
      "snapshot_time": "2025-12-14T07:54:00+01:00",
      "symbol": "EURUSD",
      "h1_open": 1.17392,
      "h1_high": 1.17452,
      "h1_low": 1.17379,
      "h1_close": 1.17384,
      "rsi": 40.45381,
      "macd_main": -0.00004,
      "macd_signal": 0.0,
      ...
    }
  ]
}
```

**Verwendung:**
- **NHITS Model-Training** mit 20 Features:
  - OHLC: `h1_open`, `h1_high`, `h1_low`, `h1_close`
  - Trend: `rsi`, `macd_main`, `macd_signal`, `adx_main`, `adx_plusdi`, `adx_minusdi`
  - Oszillatoren: `cci`, `sto_main`, `sto_signal`
  - Volatilität: `bb_upper`, `bb_base`, `bb_lower`, `atr_d1`
  - Moving Averages: `ma_10`
  - Ichimoku: `ichimoku_tenkan`, `ichimoku_kijun`
  - Stärke: `strength_4h`, `strength_1d`, `strength_1w`

- **LLM-Kontext** (gespeichert in `additional_data`):
  - Marktmikrostruktur: `bid`, `ask`, `spread`, `spread_pct`
  - Multi-Timeframe: `d1_OHLC`, `m15_OHLC`
  - Ichimoku Cloud: `ichimoku_senkoua`, `ichimoku_senkoub`, `ichimoku_chikou`
  - Support/Resistance: `s1_level_m5`, `r1_level_m5`
  - Volatilität: `range_d1`, `atr_pct_d1`

**Error Handling:**
- Bei API-Fehler: Fehler wird zurückgegeben (kein Fallback mehr)
- **Wichtig:** Direkter TimescaleDB-Zugriff wurde entfernt (API-Only seit 2025-12-14)

---

### 3. `/symbol-latest-full/{symbol}` - Aktuelle Marktdaten

**Zweck:** Aktuelle Marktdaten + alle Indikatoren für ein Symbol (Real-time)

**Services:**
- `analysis_service.py::fetch_latest_market_data()`

**Request:**
```http
GET http://10.1.19.102:3000/api/symbol-latest-full/EURUSD
```

**Response:**
```json
{
  "snapshot_time": "2025-12-14T07:54:00+01:00",
  "symbol": "EURUSD",
  "category": "Forex",
  "bid": 1.17384,
  "ask": 1.17392,
  "spread": 0.00008,
  "spread_pct": 0.01,
  "h1_open": 1.17392,
  "h1_high": 1.17452,
  "h1_low": 1.17379,
  "h1_close": 1.17384,
  "d1_open": 1.17291,
  "d1_high": 1.17498,
  "d1_low": 1.17195,
  "d1_close": 1.17384,
  "rsi": 40.45381,
  "macd_main": -0.00004,
  "macd_signal": 0.0,
  "adx_main": 18.91983,
  "adx_plusdi": 18.60355,
  "adx_minusdi": 31.90659,
  ...
}
```

**Verwendung:**
- Real-time Trading-Analyse
- LLM-Kontext für Empfehlungen
- Aktuelle Marktbedingungen

---

### 4. `/symbol-data-full` - Multi-Symbol Marktübersicht

**Zweck:** Neueste Daten für ALLE Symbole (Market Scanner)

**Services:**
- `analysis_service.py::fetch_all_latest_market_data()`

**Request:**
```http
GET http://10.1.19.102:3000/api/symbol-data-full?limit=1
```

**Parameter:**
- `limit=1`: Nur neuester Snapshot pro Symbol

**Response:**
```json
{
  "columns": [...],
  "data": [
    {
      "symbol": "EURUSD",
      "snapshot_time": "2025-12-14T07:54:00+01:00",
      ...
    },
    {
      "symbol": "GBPUSD",
      "snapshot_time": "2025-12-14T07:54:00+01:00",
      ...
    },
    ...
  ]
}
```

**Verwendung:**
- Market Overview Dashboard
- Multi-Symbol Screening
- Relative Stärke-Analyse

---

## 🔄 Datenabruf-Strategie

### Primary: EasyInsight API
```python
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(
        f"{settings.easyinsight_api_url}/symbol-data-full/{symbol}",
        params={"limit": days * 24}
    )
```

### Error Handling: API-Only (kein Fallback)
```python
try:
    # EasyInsight API call
    ...
except Exception as e:
    logger.error(f"API failed: {e}")
    return []  # Kein Fallback mehr - API-Only Architektur
```

**Wichtig:** Seit 2025-12-14 wurde der direkte TimescaleDB-Zugriff entfernt.
Alle Daten kommen ausschließlich von der EasyInsight API.

---

## 📊 Datenverarbeitung

### NHITS Training
1. **Datenabruf:** `/symbol-data-full/{symbol}?limit=720` (30 Tage × 24h)
2. **Feature-Extraktion:** 20 Indikatoren aus API-Response
3. **Parsing:** ISO-Zeitstempel, Float-Konvertierung
4. **Training:** Multi-variate NHITS mit allen Features

### Trading-Analyse
1. **Datenabruf:** `/symbol-latest-full/{symbol}`
2. **Strukturierung:** `MarketDataSnapshot` Pydantic-Model
3. **LLM-Integration:** Vollständiger Kontext für Empfehlungen

---

## ⚙️ Konfiguration

**Environment Variable:**
```bash
EASYINSIGHT_API_URL=http://10.1.19.102:3000/api
```

**Settings:**
```python
# src/config/settings.py
easyinsight_api_url: str = Field(
    default="http://localhost:3000/api",
    description="EasyInsight API base URL"
)
```

---

## 🔍 Verfügbare Indikatoren

### OHLC-Daten (3 Timeframes)
- M15: `m15_open`, `m15_high`, `m15_low`, `m15_close`
- H1: `h1_open`, `h1_high`, `h1_low`, `h1_close`
- D1: `d1_open`, `d1_high`, `d1_low`, `d1_close`

### Technische Indikatoren
- **RSI:** `rsi` (Relative Strength Index)
- **MACD:** `macd_main`, `macd_signal`
- **ADX:** `adx_main`, `adx_plusdi`, `adx_minusdi`
- **CCI:** `cci` (Commodity Channel Index)
- **Stochastic:** `sto_main`, `sto_signal`
- **Bollinger Bands:** `bb_upper`, `bb_base`, `bb_lower`
- **ATR:** `atr_d1`, `atr_pct_d1`
- **MA:** `ma_10` (Moving Average)
- **Ichimoku:** `ichimoku_tenkan`, `ichimoku_kijun`, `ichimoku_senkoua`,
  `ichimoku_senkoub`, `ichimoku_chikou`

### Marktdaten
- **Preise:** `bid`, `ask`, `spread`, `spread_pct`
- **Volatilität:** `range_d1`, `atr_d1`, `atr_pct_d1`
- **Support/Resistance:** `s1_level_m5`, `r1_level_m5`
- **Stärke:** `strength_4h`, `strength_1d`, `strength_1w`

### Metadaten
- `snapshot_time` - Zeitstempel der Daten
- `symbol` - Trading-Symbol
- `category` - Kategorie (Forex, Crypto, Index, etc.)
- `source_file` - Quelldatei
- `import_time` - Import-Zeitstempel

---

## 📈 Verwendungsbeispiele

### Symbol-Liste abrufen
```python
from src.services.nhits_training_service import nhits_training_service

symbols = await nhits_training_service.get_available_symbols()
# Returns: ['EURUSD', 'GBPUSD', 'BTCUSD', ...]
```

### Trainingsdaten laden
```python
time_series = await nhits_training_service.get_training_data(
    symbol='EURUSD',
    days=30
)
# Returns: List[TimeSeriesData] mit 720 Datenpunkten und 20 Features
```

### Aktuelle Marktdaten
```python
from src.services.analysis_service import analysis_service

snapshot = await analysis_service.fetch_latest_market_data('EURUSD')
# Returns: MarketDataSnapshot mit allen Indikatoren
```

---

## 🚨 Error Handling

### API-Timeout
```python
async with httpx.AsyncClient(timeout=30.0) as client:
    # 30 Sekunden Timeout für große Datenmengen
```

### Fallback-Logik
```python
try:
    # Primary: EasyInsight API
    data = await fetch_from_api(symbol)
except Exception as e:
    logger.warning(f"API error: {e}")
    # Fallback: TimescaleDB
    data = await fetch_from_database(symbol)
```

### Daten-Validierung
```python
if not row.get('h1_close'):
    logger.warning(f"Missing data for {symbol}")
    continue
```

---

## 📊 Performance

- **API Response Time:** ~200-500ms für `/symbols`
- **Data Fetch:** ~1-3s für `/symbol-data-full/{symbol}?limit=720`
- **Parallel Requests:** Möglich für Multi-Symbol-Training
- **Caching:** Keine (Real-time Daten)

---

## 🔐 Zugriff

- **Netzwerk:** Host-Netzwerk (`--network=host` in Docker)
- **Authentifizierung:** Keine (internes Netzwerk)
- **Rate Limiting:** Keine

---

## 📝 Logs

```python
logger.info(f"Fetched {len(symbols)} symbols from EasyInsight API")
logger.info(f"Fetched {len(time_series)} data points for {symbol} from EasyInsight API")
logger.warning(f"Failed to get data from EasyInsight API: {e}, falling back to TimescaleDB")
```

---

## 🔄 Data Access Strategy

### ⚡ API-Only Architecture (seit 2025-12-14)
- **Primär:** EasyInsight API (`http://10.1.19.102:3000/api`)
- **Kein Fallback:** Direkter TimescaleDB-Zugriff wurde entfernt
- **Begründung:** Single Source of Truth, zentrale Wartung, API-First Design
- **Error Handling:** Bei API-Fehler werden Fehler zurückgegeben (keine automatische Wiederholung)

### 📦 Lokale FAISS (RAG)
- **Directory:** `./data/faiss`
- **Verwendung:** Historische Kontext-Daten für LLM
- **Sync:** Manuell oder über API-Trigger (automatischer TimescaleDB-Sync deaktiviert)

---

## ✅ Vorteile der API-Only Integration

1. **Single Source of Truth:** Zentrale Datenquelle ohne Duplikation
2. **Vollständige Indikatoren:** 40+ technische Indikatoren
3. **Multi-Timeframe:** M15, H1, D1 OHLC-Daten
4. **Real-time:** Aktuelle Marktdaten direkt von der API
5. **Skalierbar:** Einfache Erweiterung um neue Symbole
6. **Wartbar:** API-Änderungen zentral gepflegt, keine DB-Schema-Migration nötig
7. **Sicherheit:** Keine direkten Datenbankzugriffe, reduzierte Angriffsfläche
8. **Performance:** Optimierte API-Endpoints mit Caching

---

## 🎯 Verwendete Services

| Service | Endpoint | Verwendung |
|---------|----------|------------|
| `nhits_training_service.py` | `/symbols`, `/symbol-data-full/{symbol}` | Model-Training |
| `analysis_service.py` | `/symbols`, `/symbol-latest-full/{symbol}`, `/symbol-data-full` | Trading-Analyse, LLM-Kontext |

---

**Autor:** KI Trading Model Team
**Letzte Aktualisierung:** 2025-12-14
**Version:** 1.0.0
