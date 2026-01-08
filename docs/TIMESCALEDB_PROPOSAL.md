# TimescaleDB Integration für Data Service

## Übersicht

Dieses Dokument beschreibt die Erweiterung des Data Service um eine persistente Datenspeicherung in PostgreSQL mit TimescaleDB-Erweiterung.

**Ziel:** Alle im Data Service genutzten Daten werden persistent in TimescaleDB gespeichert, während Redis weiterhin als schneller Cache fungiert.

---

## 1. Architektur-Übersicht

### Aktuelle Architektur (Ist-Zustand)

```
Externe APIs (TwelveData, EasyInsight, Yahoo Finance)
                    │
                    ▼
            ┌───────────────┐
            │  Data Service │
            │   (Port 3001) │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  Redis Cache  │  ← Nur Cache, keine Persistenz
            │  (Port 6379)  │
            └───────────────┘
```

### Neue Architektur (Soll-Zustand)

```
Externe APIs (TwelveData, EasyInsight, Yahoo Finance)
                    │
                    ▼
            ┌───────────────┐
            │  Data Service │
            │   (Port 3001) │
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Redis Cache  │       │  TimescaleDB  │
│  (Port 6379)  │       │ (10.1.19.102) │
│   Hot Data    │       │  Persistenz   │
└───────────────┘       └───────────────┘
```

### Datenfluss

```
1. Request kommt an
        │
        ▼
2. Redis Cache Check
   ├── HIT  → Direkt zurückgeben
   └── MISS → Weiter zu Schritt 3
        │
        ▼
3. TimescaleDB Check
   ├── Daten vorhanden & aktuell → In Redis cachen → Zurückgeben
   └── Daten fehlen/veraltet    → Weiter zu Schritt 4
        │
        ▼
4. Externe API abrufen (TwelveData → EasyInsight → YFinance)
        │
        ▼
5. Daten in TimescaleDB speichern (persistent)
        │
        ▼
6. Daten in Redis cachen (für schnellen Zugriff)
        │
        ▼
7. Response zurückgeben
```

---

## 2. Datenbankschema

### 2.1 Server-Konfiguration

| Parameter | Wert |
|-----------|------|
| **Host** | 10.1.19.102 |
| **Port** | 5432 |
| **Datenbank** | tradingdataservice |
| **Schema** | public |
| **Erweiterung** | TimescaleDB |

### 2.2 OHLCV-Tabellen (Hypertables)

**Separate Tabellen pro Timeframe** für optimale Partitionierung und Abfrageleistung.

```sql
-- Basis-Struktur für alle OHLCV-Tabellen
CREATE TABLE ohlcv_{timeframe} (
    -- Primärschlüssel
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,

    -- OHLCV Daten
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),

    -- Metadaten
    source          VARCHAR(20) NOT NULL,  -- 'twelvedata', 'easyinsight', 'yfinance'
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    -- Primärschlüssel
    PRIMARY KEY (timestamp, symbol)
);

-- In Hypertable konvertieren
SELECT create_hypertable('ohlcv_{timeframe}', 'timestamp',
    chunk_time_interval => INTERVAL '{interval}',
    if_not_exists => TRUE
);

-- Indizes
CREATE INDEX IF NOT EXISTS idx_ohlcv_{timeframe}_symbol
    ON ohlcv_{timeframe} (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_{timeframe}_source
    ON ohlcv_{timeframe} (source);
```

**Tabellen-Übersicht:**

| Tabelle | Chunk-Intervall | Beschreibung |
|---------|-----------------|--------------|
| `ohlcv_m1` | 1 day | 1-Minuten-Kerzen |
| `ohlcv_m5` | 1 day | 5-Minuten-Kerzen |
| `ohlcv_m15` | 7 days | 15-Minuten-Kerzen |
| `ohlcv_m30` | 7 days | 30-Minuten-Kerzen |
| `ohlcv_m45` | 7 days | 45-Minuten-Kerzen |
| `ohlcv_h1` | 7 days | 1-Stunden-Kerzen |
| `ohlcv_h2` | 14 days | 2-Stunden-Kerzen |
| `ohlcv_h4` | 30 days | 4-Stunden-Kerzen |
| `ohlcv_d1` | 365 days | Tageskerzen |
| `ohlcv_w1` | 365 days | Wochenkerzen |
| `ohlcv_mn` | 365 days | Monatskerzen |

### 2.3 Technische Indikatoren

#### 2.3.1 Haupt-Indikatoren-Tabelle (JSONB für Flexibilität)

```sql
CREATE TABLE indicators (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,  -- 'M1', 'H1', 'D1', etc.
    indicator_name  VARCHAR(50) NOT NULL,  -- 'RSI', 'MACD', 'BBANDS', etc.

    -- Indikator-Werte (JSONB für Flexibilität)
    values          JSONB NOT NULL,

    -- Metadaten
    parameters      JSONB,                  -- z.B. {"period": 14, "source": "close"}
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe, indicator_name)
);

SELECT create_hypertable('indicators', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_indicators_lookup
    ON indicators (symbol, timeframe, indicator_name, timestamp DESC);

-- Partial Index für häufig abgefragte Indikatoren
CREATE INDEX idx_indicators_rsi
    ON indicators (symbol, timeframe, timestamp DESC)
    WHERE indicator_name = 'RSI';

CREATE INDEX idx_indicators_macd
    ON indicators (symbol, timeframe, timestamp DESC)
    WHERE indicator_name = 'MACD';
```

#### 2.3.2 Optimierte Tabellen für häufig genutzte Indikatoren

Für Performance-kritische Indikatoren separate Tabellen mit festen Spalten:

```sql
-- Moving Averages (SMA, EMA, WMA, etc.)
CREATE TABLE indicators_ma (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- SMA
    sma_20          DECIMAL(20, 8),
    sma_50          DECIMAL(20, 8),
    sma_200         DECIMAL(20, 8),

    -- EMA
    ema_12          DECIMAL(20, 8),
    ema_26          DECIMAL(20, 8),
    ema_50          DECIMAL(20, 8),
    ema_200         DECIMAL(20, 8),

    -- WMA, DEMA, TEMA
    wma_20          DECIMAL(20, 8),
    dema_20         DECIMAL(20, 8),
    tema_20         DECIMAL(20, 8),

    -- VWAP
    vwap            DECIMAL(20, 8),

    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_ma', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_indicators_ma_lookup
    ON indicators_ma (symbol, timeframe, timestamp DESC);
```

```sql
-- Momentum-Indikatoren
CREATE TABLE indicators_momentum (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- RSI Varianten
    rsi_14          DECIMAL(10, 4),
    rsi_7           DECIMAL(10, 4),
    rsi_21          DECIMAL(10, 4),
    stoch_rsi       DECIMAL(10, 4),
    connors_rsi     DECIMAL(10, 4),

    -- Stochastic
    stoch_k         DECIMAL(10, 4),
    stoch_d         DECIMAL(10, 4),

    -- MACD
    macd_line       DECIMAL(20, 8),
    macd_signal     DECIMAL(20, 8),
    macd_histogram  DECIMAL(20, 8),

    -- Weitere Momentum
    cci             DECIMAL(10, 4),
    williams_r      DECIMAL(10, 4),
    roc             DECIMAL(10, 4),
    momentum        DECIMAL(20, 8),

    -- ADX Familie
    adx             DECIMAL(10, 4),
    plus_di         DECIMAL(10, 4),
    minus_di        DECIMAL(10, 4),

    -- MFI (Money Flow Index)
    mfi             DECIMAL(10, 4),

    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_momentum', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_indicators_momentum_lookup
    ON indicators_momentum (symbol, timeframe, timestamp DESC);
```

```sql
-- Volatilitäts-Indikatoren
CREATE TABLE indicators_volatility (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- Bollinger Bands
    bb_upper        DECIMAL(20, 8),
    bb_middle       DECIMAL(20, 8),
    bb_lower        DECIMAL(20, 8),
    bb_width        DECIMAL(10, 6),
    bb_percent_b    DECIMAL(10, 6),

    -- ATR
    atr_14          DECIMAL(20, 8),
    atr_7           DECIMAL(20, 8),
    natr            DECIMAL(10, 6),        -- Normalized ATR (%)
    true_range      DECIMAL(20, 8),

    -- Keltner Channel
    kc_upper        DECIMAL(20, 8),
    kc_middle       DECIMAL(20, 8),
    kc_lower        DECIMAL(20, 8),

    -- Donchian Channel
    dc_upper        DECIMAL(20, 8),
    dc_middle       DECIMAL(20, 8),
    dc_lower        DECIMAL(20, 8),

    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_volatility', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_indicators_volatility_lookup
    ON indicators_volatility (symbol, timeframe, timestamp DESC);
```

```sql
-- Trend-Indikatoren
CREATE TABLE indicators_trend (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- Ichimoku Cloud
    ichimoku_tenkan     DECIMAL(20, 8),
    ichimoku_kijun      DECIMAL(20, 8),
    ichimoku_senkou_a   DECIMAL(20, 8),
    ichimoku_senkou_b   DECIMAL(20, 8),
    ichimoku_chikou     DECIMAL(20, 8),

    -- Supertrend
    supertrend          DECIMAL(20, 8),
    supertrend_direction INTEGER,          -- 1 = Up, -1 = Down

    -- Parabolic SAR
    psar                DECIMAL(20, 8),
    psar_direction      INTEGER,           -- 1 = Up, -1 = Down

    -- Aroon
    aroon_up            DECIMAL(10, 4),
    aroon_down          DECIMAL(10, 4),
    aroon_oscillator    DECIMAL(10, 4),

    -- Linear Regression
    linreg_slope        DECIMAL(20, 8),
    linreg_intercept    DECIMAL(20, 8),
    linreg_r_squared    DECIMAL(10, 6),

    -- Hilbert Transform
    ht_trendmode        INTEGER,           -- 0 = Range, 1 = Trend

    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_trend', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_indicators_trend_lookup
    ON indicators_trend (symbol, timeframe, timestamp DESC);
```

```sql
-- Volumen-Indikatoren
CREATE TABLE indicators_volume (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- On-Balance Volume
    obv             DECIMAL(30, 8),

    -- Accumulation/Distribution
    ad_line         DECIMAL(30, 8),
    ad_oscillator   DECIMAL(20, 8),

    -- Chaikin
    chaikin_mf      DECIMAL(10, 6),        -- Money Flow

    -- Volume MA
    volume_sma_20   DECIMAL(30, 8),
    volume_ratio    DECIMAL(10, 4),        -- Current / SMA

    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_volume', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_indicators_volume_lookup
    ON indicators_volume (symbol, timeframe, timestamp DESC);
```

#### 2.3.3 Pivot Points & Support/Resistance

```sql
CREATE TABLE indicators_levels (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- Classic Pivot Points
    pivot           DECIMAL(20, 8),
    r1              DECIMAL(20, 8),
    r2              DECIMAL(20, 8),
    r3              DECIMAL(20, 8),
    s1              DECIMAL(20, 8),
    s2              DECIMAL(20, 8),
    s3              DECIMAL(20, 8),

    -- Fibonacci Pivot Points
    fib_r1          DECIMAL(20, 8),
    fib_r2          DECIMAL(20, 8),
    fib_r3          DECIMAL(20, 8),
    fib_s1          DECIMAL(20, 8),
    fib_s2          DECIMAL(20, 8),
    fib_s3          DECIMAL(20, 8),

    -- Camarilla Pivot Points
    cam_r1          DECIMAL(20, 8),
    cam_r2          DECIMAL(20, 8),
    cam_r3          DECIMAL(20, 8),
    cam_r4          DECIMAL(20, 8),
    cam_s1          DECIMAL(20, 8),
    cam_s2          DECIMAL(20, 8),
    cam_s3          DECIMAL(20, 8),
    cam_s4          DECIMAL(20, 8),

    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_levels', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);
```

#### 2.3.4 Indikator-Kategorien Übersicht

| Tabelle | Indikatoren | Beschreibung |
|---------|-------------|--------------|
| `indicators` | Alle (JSONB) | Flexible Speicherung für seltene Indikatoren |
| `indicators_ma` | SMA, EMA, WMA, DEMA, TEMA, VWAP | Moving Averages |
| `indicators_momentum` | RSI, MACD, Stoch, CCI, ADX, MFI | Momentum & Oszillatoren |
| `indicators_volatility` | BBands, ATR, Keltner, Donchian | Volatilität & Channels |
| `indicators_trend` | Ichimoku, Supertrend, PSAR, Aroon | Trend-Erkennung |
| `indicators_volume` | OBV, A/D, Chaikin | Volumen-Analyse |
| `indicators_levels` | Pivot Points, Fibonacci | Support/Resistance |

#### 2.3.5 JSONB `values` Beispiele (für `indicators` Tabelle)

```json
// RSI
{"rsi": 65.42}

// MACD
{"macd": 0.0045, "signal": 0.0032, "histogram": 0.0013}

// Bollinger Bands
{"upper": 1.0850, "middle": 1.0800, "lower": 1.0750, "width": 0.0093, "percent_b": 0.72}

// Ichimoku
{
    "tenkan_sen": 1.0810,
    "kijun_sen": 1.0795,
    "senkou_span_a": 1.0802,
    "senkou_span_b": 1.0780,
    "chikou_span": 1.0825
}

// Connors RSI (3 Komponenten)
{"crsi": 58.3, "rsi": 62.1, "streak_rsi": 55.4, "pct_rank": 57.2}

// Supertrend
{"value": 1.0750, "direction": 1, "is_uptrend": true}

// Linear Regression
{"slope": 0.00125, "intercept": 1.0650, "r_squared": 0.87, "forecast": 1.0812}
```

#### 2.3.6 Retention Policies für Indikatoren

```sql
-- Alte Indikator-Daten automatisch löschen (optional)
-- M1 Indikatoren: 7 Tage
SELECT add_retention_policy('indicators', INTERVAL '30 days',
    if_not_exists => TRUE);

-- Optimierte Tabellen: Längere Retention
SELECT add_retention_policy('indicators_ma', INTERVAL '180 days',
    if_not_exists => TRUE);
SELECT add_retention_policy('indicators_momentum', INTERVAL '180 days',
    if_not_exists => TRUE);
SELECT add_retention_policy('indicators_volatility', INTERVAL '180 days',
    if_not_exists => TRUE);
SELECT add_retention_policy('indicators_trend', INTERVAL '180 days',
    if_not_exists => TRUE);

-- Pivot Points: 1 Jahr (für Backtesting)
SELECT add_retention_policy('indicators_levels', INTERVAL '365 days',
    if_not_exists => TRUE);
```

#### 2.3.7 Kompression für Indikatoren

```sql
-- Kompression nach 7 Tagen für alle Indikator-Tabellen
ALTER TABLE indicators SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe,indicator_name'
);
SELECT add_compression_policy('indicators', INTERVAL '7 days');

ALTER TABLE indicators_ma SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);
SELECT add_compression_policy('indicators_ma', INTERVAL '7 days');

ALTER TABLE indicators_momentum SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);
SELECT add_compression_policy('indicators_momentum', INTERVAL '7 days');

-- Analog für andere Tabellen...
```

### 2.4 Marktdaten (Echtzeit-Snapshots)

```sql
CREATE TABLE market_snapshots (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,

    -- Preisdaten
    bid             DECIMAL(20, 8),
    ask             DECIMAL(20, 8),
    last_price      DECIMAL(20, 8) NOT NULL,

    -- Volumen & Spread
    volume          DECIMAL(30, 8),
    spread          DECIMAL(20, 8),

    -- Tagesstatistik
    day_open        DECIMAL(20, 8),
    day_high        DECIMAL(20, 8),
    day_low         DECIMAL(20, 8),
    prev_close      DECIMAL(20, 8),

    -- Metadaten
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('market_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_market_snapshots_symbol
    ON market_snapshots (symbol, timestamp DESC);
```

### 2.5 Symbol-Metadaten

```sql
CREATE TABLE symbols (
    symbol              VARCHAR(20) PRIMARY KEY,
    display_name        VARCHAR(100),
    category            VARCHAR(20) NOT NULL,  -- 'FOREX', 'CRYPTO', 'STOCK', etc.
    subcategory         VARCHAR(20),           -- 'MAJOR', 'MINOR', 'EXOTIC', etc.

    -- Währungspaar-Info
    base_currency       VARCHAR(10),
    quote_currency      VARCHAR(10),

    -- API-Mappings
    twelvedata_symbol   VARCHAR(20),           -- 'BTC/USD'
    easyinsight_symbol  VARCHAR(20),           -- 'BTCUSD'
    yfinance_symbol     VARCHAR(20),           -- 'BTC-USD'

    -- Trading-Metadaten
    pip_value           DECIMAL(10, 6),
    min_lot_size        DECIMAL(10, 4),
    max_lot_size        DECIMAL(10, 2),
    lot_step            DECIMAL(10, 4),

    -- Status
    is_active           BOOLEAN DEFAULT TRUE,
    is_favorite         BOOLEAN DEFAULT FALSE,

    -- Daten-Verfügbarkeit
    first_data_at       TIMESTAMPTZ,
    last_data_at        TIMESTAMPTZ,

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_symbols_category ON symbols (category, subcategory);
CREATE INDEX idx_symbols_active ON symbols (is_active) WHERE is_active = TRUE;
```

### 2.6 Externe Datenquellen (für RAG)

```sql
-- Economic Calendar
CREATE TABLE economic_events (
    id              SERIAL,
    timestamp       TIMESTAMPTZ NOT NULL,
    event_name      VARCHAR(200) NOT NULL,
    country         VARCHAR(10) NOT NULL,
    currency        VARCHAR(10),
    importance      VARCHAR(10),           -- 'low', 'medium', 'high'

    -- Werte
    actual          VARCHAR(50),
    forecast        VARCHAR(50),
    previous        VARCHAR(50),

    -- Metadaten
    source          VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('economic_events', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Sentiment Data
CREATE TABLE sentiment_data (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20),           -- NULL für marktweite Daten

    -- Sentiment-Werte
    fear_greed_index    INTEGER,           -- 0-100
    social_sentiment    DECIMAL(5, 2),     -- -1 bis 1
    news_sentiment      DECIMAL(5, 2),

    -- Volatilitäts-Indikatoren
    vix                 DECIMAL(10, 4),
    put_call_ratio      DECIMAL(5, 3),

    -- Raw Data
    raw_data            JSONB,

    source              VARCHAR(50),
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, COALESCE(symbol, 'MARKET'))
);

SELECT create_hypertable('sentiment_data', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- On-Chain Data (Crypto)
CREATE TABLE onchain_data (
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,

    -- Whale Activity
    whale_transactions  INTEGER,
    large_tx_volume     DECIMAL(30, 8),

    -- Exchange Flows
    exchange_inflow     DECIMAL(30, 8),
    exchange_outflow    DECIMAL(30, 8),
    exchange_netflow    DECIMAL(30, 8),

    -- Mining (BTC)
    hash_rate           DECIMAL(30, 2),
    difficulty          DECIMAL(30, 2),

    -- DeFi
    tvl                 DECIMAL(30, 2),

    -- Raw Data
    raw_data            JSONB,

    source              VARCHAR(50),
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('onchain_data', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);
```

### 2.7 Data Freshness Tracking

```sql
CREATE TABLE data_freshness (
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    data_type       VARCHAR(30) NOT NULL,  -- 'ohlcv', 'indicator', 'sentiment', etc.

    last_updated    TIMESTAMPTZ NOT NULL,
    last_timestamp  TIMESTAMPTZ NOT NULL,   -- Neuester Datenpunkt
    record_count    BIGINT DEFAULT 0,
    source          VARCHAR(20),

    PRIMARY KEY (symbol, timeframe, data_type)
);

CREATE INDEX idx_freshness_updated ON data_freshness (last_updated);
```

---

## 3. Service-Implementierung

### 3.1 Neue Dateien

```
src/services/
├── timescaledb_service.py      # TimescaleDB Verbindungs-Service
├── data_repository.py          # Repository Pattern für DB-Zugriff
└── data_sync_service.py        # Synchronisation externe APIs ↔ DB

src/models/
└── db_models.py                # SQLAlchemy/Pydantic Models

src/config/
└── database.py                 # DB-Konfiguration
```

### 3.2 TimescaleDB Service

```python
# src/services/timescaledb_service.py

import asyncpg
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from src.config.settings import settings
from src.config.timeframes import Timeframe, normalize_timeframe


class TimescaleDBService:
    """
    Service für TimescaleDB-Verbindung und Abfragen.

    WICHTIG: Dieser Service ist die EINZIGE Komponente,
    die direkte Datenbankverbindungen herstellt.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._dsn = (
            f"postgresql://{settings.timescale_user}:{settings.timescale_password}"
            f"@{settings.timescale_host}:{settings.timescale_port}"
            f"/{settings.timescale_database}"
        )

    async def initialize(self) -> None:
        """Verbindungspool initialisieren."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=5,
            max_size=20,
            command_timeout=60,
            statement_cache_size=100
        )
        await self._ensure_schema()

    async def close(self) -> None:
        """Verbindungspool schließen."""
        if self._pool:
            await self._pool.close()

    @asynccontextmanager
    async def connection(self):
        """Context Manager für Datenbankverbindung."""
        async with self._pool.acquire() as conn:
            yield conn

    # === OHLCV Methoden ===

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        OHLCV-Daten aus TimescaleDB abrufen.

        Args:
            symbol: Trading-Symbol (z.B. 'BTCUSD')
            timeframe: Timeframe ('M1', 'H1', 'D1', etc.)
            limit: Maximale Anzahl Datenpunkte
            start_time: Startzeit (optional)
            end_time: Endzeit (optional)

        Returns:
            Liste von OHLCV-Dictionaries
        """
        tf = normalize_timeframe(timeframe)
        table = f"ohlcv_{tf.value.lower()}"

        query = f"""
            SELECT
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                source
            FROM {table}
            WHERE symbol = $1
        """
        params = [symbol]
        param_idx = 2

        if start_time:
            query += f" AND timestamp >= ${param_idx}"
            params.append(start_time)
            param_idx += 1

        if end_time:
            query += f" AND timestamp <= ${param_idx}"
            params.append(end_time)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self.connection() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "timestamp": row["timestamp"].isoformat(),
                "symbol": row["symbol"],
                "timeframe": tf.value,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if row["volume"] else None,
                "source": row["source"]
            }
            for row in rows
        ]

    async def upsert_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        source: str
    ) -> int:
        """
        OHLCV-Daten einfügen oder aktualisieren (Upsert).

        Returns:
            Anzahl der eingefügten/aktualisierten Zeilen
        """
        if not data:
            return 0

        tf = normalize_timeframe(timeframe)
        table = f"ohlcv_{tf.value.lower()}"

        query = f"""
            INSERT INTO {table}
                (timestamp, symbol, open, high, low, close, volume, source, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (timestamp, symbol)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source,
                updated_at = NOW()
        """

        async with self.connection() as conn:
            async with conn.transaction():
                for row in data:
                    await conn.execute(
                        query,
                        row["timestamp"],
                        symbol,
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row.get("volume"),
                        source
                    )

        # Freshness aktualisieren
        await self._update_freshness(symbol, tf.value, "ohlcv", len(data), source)

        return len(data)

    async def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[datetime]:
        """Neuesten Timestamp für Symbol/Timeframe abrufen."""
        tf = normalize_timeframe(timeframe)
        table = f"ohlcv_{tf.value.lower()}"

        query = f"""
            SELECT MAX(timestamp) as latest
            FROM {table}
            WHERE symbol = $1
        """

        async with self.connection() as conn:
            row = await conn.fetchrow(query, symbol)

        return row["latest"] if row else None

    # === Indikatoren Methoden (JSONB-Tabelle) ===

    async def get_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Technische Indikatoren aus JSONB-Tabelle abrufen."""
        tf = normalize_timeframe(timeframe)

        query = """
            SELECT timestamp, values, parameters, source
            FROM indicators
            WHERE symbol = $1
              AND timeframe = $2
              AND indicator_name = $3
            ORDER BY timestamp DESC
            LIMIT $4
        """

        async with self.connection() as conn:
            rows = await conn.fetch(query, symbol, tf.value, indicator_name, limit)

        return [
            {
                "timestamp": row["timestamp"].isoformat(),
                "symbol": symbol,
                "timeframe": tf.value,
                "indicator": indicator_name,
                **row["values"],
                "parameters": row["parameters"],
                "source": row["source"]
            }
            for row in rows
        ]

    async def upsert_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        data: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        source: str
    ) -> int:
        """Indikatoren in JSONB-Tabelle einfügen oder aktualisieren."""
        if not data:
            return 0

        tf = normalize_timeframe(timeframe)

        query = """
            INSERT INTO indicators
                (timestamp, symbol, timeframe, indicator_name, values, parameters, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (timestamp, symbol, timeframe, indicator_name)
            DO UPDATE SET
                values = EXCLUDED.values,
                parameters = EXCLUDED.parameters,
                source = EXCLUDED.source
        """

        async with self.connection() as conn:
            async with conn.transaction():
                for row in data:
                    timestamp = row.pop("timestamp", row.pop("datetime", None))
                    await conn.execute(
                        query,
                        timestamp,
                        symbol,
                        tf.value,
                        indicator_name,
                        row,  # Remaining fields as JSONB values
                        parameters,
                        source
                    )

        # Freshness aktualisieren
        await self._update_freshness(
            symbol, tf.value, f"indicator_{indicator_name.lower()}", len(data), source
        )

        return len(data)

    # === Optimierte Indikator-Tabellen ===

    async def get_momentum_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        indicators: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Momentum-Indikatoren aus optimierter Tabelle abrufen.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            limit: Max. Anzahl Datenpunkte
            indicators: Optionale Liste spezifischer Indikatoren
                       ['rsi_14', 'macd_line', 'stoch_k', etc.]
        """
        tf = normalize_timeframe(timeframe)

        # Spalten auswählen
        if indicators:
            columns = ", ".join(indicators)
        else:
            columns = """
                rsi_14, rsi_7, rsi_21, stoch_rsi, connors_rsi,
                stoch_k, stoch_d,
                macd_line, macd_signal, macd_histogram,
                cci, williams_r, roc, momentum,
                adx, plus_di, minus_di, mfi
            """

        query = f"""
            SELECT timestamp, {columns}, source
            FROM indicators_momentum
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        async with self.connection() as conn:
            rows = await conn.fetch(query, symbol, tf.value, limit)

        return [dict(row) for row in rows]

    async def upsert_momentum_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        source: str
    ) -> int:
        """Momentum-Indikatoren in optimierte Tabelle speichern."""
        if not data:
            return 0

        tf = normalize_timeframe(timeframe)

        # Dynamisch Spalten aus Daten extrahieren
        sample = data[0]
        columns = [k for k in sample.keys() if k != "timestamp"]
        placeholders = ", ".join([f"${i+3}" for i in range(len(columns))])
        column_names = ", ".join(columns)

        update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns])

        query = f"""
            INSERT INTO indicators_momentum
                (timestamp, symbol, timeframe, {column_names}, source)
            VALUES ($1, $2, '{tf.value}', {placeholders}, ${len(columns)+3})
            ON CONFLICT (timestamp, symbol, timeframe)
            DO UPDATE SET {update_clause}, source = EXCLUDED.source
        """

        async with self.connection() as conn:
            async with conn.transaction():
                for row in data:
                    values = [row["timestamp"], symbol] + [row.get(c) for c in columns] + [source]
                    await conn.execute(query, *values)

        await self._update_freshness(symbol, tf.value, "indicators_momentum", len(data), source)
        return len(data)

    async def get_volatility_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Volatilitäts-Indikatoren aus optimierter Tabelle abrufen."""
        tf = normalize_timeframe(timeframe)

        query = """
            SELECT
                timestamp,
                bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b,
                atr_14, atr_7, natr, true_range,
                kc_upper, kc_middle, kc_lower,
                dc_upper, dc_middle, dc_lower,
                source
            FROM indicators_volatility
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        async with self.connection() as conn:
            rows = await conn.fetch(query, symbol, tf.value, limit)

        return [dict(row) for row in rows]

    async def get_trend_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Trend-Indikatoren aus optimierter Tabelle abrufen."""
        tf = normalize_timeframe(timeframe)

        query = """
            SELECT
                timestamp,
                ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
                ichimoku_senkou_b, ichimoku_chikou,
                supertrend, supertrend_direction,
                psar, psar_direction,
                aroon_up, aroon_down, aroon_oscillator,
                linreg_slope, linreg_intercept, linreg_r_squared,
                ht_trendmode,
                source
            FROM indicators_trend
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        async with self.connection() as conn:
            rows = await conn.fetch(query, symbol, tf.value, limit)

        return [dict(row) for row in rows]

    async def get_ma_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Moving Average Indikatoren aus optimierter Tabelle abrufen."""
        tf = normalize_timeframe(timeframe)

        query = """
            SELECT
                timestamp,
                sma_20, sma_50, sma_200,
                ema_12, ema_26, ema_50, ema_200,
                wma_20, dema_20, tema_20,
                vwap,
                source
            FROM indicators_ma
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        async with self.connection() as conn:
            rows = await conn.fetch(query, symbol, tf.value, limit)

        return [dict(row) for row in rows]

    async def get_all_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Alle Indikatoren aus allen optimierten Tabellen abrufen.

        Returns:
            Dict mit Kategorien als Keys und Indikator-Listen als Values
        """
        tf = normalize_timeframe(timeframe)

        results = {}

        # Parallel alle Tabellen abfragen
        async with self.connection() as conn:
            # MA Indikatoren
            ma_query = """
                SELECT * FROM indicators_ma
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC LIMIT $3
            """
            results["moving_averages"] = [
                dict(r) for r in await conn.fetch(ma_query, symbol, tf.value, limit)
            ]

            # Momentum
            mom_query = """
                SELECT * FROM indicators_momentum
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC LIMIT $3
            """
            results["momentum"] = [
                dict(r) for r in await conn.fetch(mom_query, symbol, tf.value, limit)
            ]

            # Volatility
            vol_query = """
                SELECT * FROM indicators_volatility
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC LIMIT $3
            """
            results["volatility"] = [
                dict(r) for r in await conn.fetch(vol_query, symbol, tf.value, limit)
            ]

            # Trend
            trend_query = """
                SELECT * FROM indicators_trend
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC LIMIT $3
            """
            results["trend"] = [
                dict(r) for r in await conn.fetch(trend_query, symbol, tf.value, limit)
            ]

            # Volume
            vol_ind_query = """
                SELECT * FROM indicators_volume
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC LIMIT $3
            """
            results["volume"] = [
                dict(r) for r in await conn.fetch(vol_ind_query, symbol, tf.value, limit)
            ]

            # Levels
            levels_query = """
                SELECT * FROM indicators_levels
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC LIMIT $3
            """
            results["levels"] = [
                dict(r) for r in await conn.fetch(levels_query, symbol, tf.value, limit)
            ]

        return results

    async def upsert_all_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators_data: Dict[str, Any],
        source: str
    ) -> Dict[str, int]:
        """
        Alle Indikatoren auf einmal speichern.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            indicators_data: Dict mit Indikator-Werten:
                {
                    "timestamp": "2024-01-15T10:00:00Z",
                    "rsi_14": 65.4,
                    "macd_line": 0.0045,
                    "bb_upper": 1.0850,
                    ...
                }
            source: Datenquelle

        Returns:
            Dict mit Anzahl der gespeicherten Zeilen pro Tabelle
        """
        tf = normalize_timeframe(timeframe)
        timestamp = indicators_data.get("timestamp")
        counts = {}

        async with self.connection() as conn:
            async with conn.transaction():
                # MA Indikatoren
                ma_fields = ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
                            "ema_50", "ema_200", "wma_20", "dema_20", "tema_20", "vwap"]
                ma_values = {k: indicators_data.get(k) for k in ma_fields if k in indicators_data}
                if ma_values:
                    await self._upsert_indicator_row(
                        conn, "indicators_ma", symbol, tf.value, timestamp, ma_values, source
                    )
                    counts["ma"] = 1

                # Momentum Indikatoren
                mom_fields = ["rsi_14", "rsi_7", "rsi_21", "stoch_rsi", "connors_rsi",
                             "stoch_k", "stoch_d", "macd_line", "macd_signal", "macd_histogram",
                             "cci", "williams_r", "roc", "momentum", "adx", "plus_di", "minus_di", "mfi"]
                mom_values = {k: indicators_data.get(k) for k in mom_fields if k in indicators_data}
                if mom_values:
                    await self._upsert_indicator_row(
                        conn, "indicators_momentum", symbol, tf.value, timestamp, mom_values, source
                    )
                    counts["momentum"] = 1

                # Volatility Indikatoren
                vol_fields = ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent_b",
                             "atr_14", "atr_7", "natr", "true_range",
                             "kc_upper", "kc_middle", "kc_lower", "dc_upper", "dc_middle", "dc_lower"]
                vol_values = {k: indicators_data.get(k) for k in vol_fields if k in indicators_data}
                if vol_values:
                    await self._upsert_indicator_row(
                        conn, "indicators_volatility", symbol, tf.value, timestamp, vol_values, source
                    )
                    counts["volatility"] = 1

                # Trend Indikatoren
                trend_fields = ["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
                               "ichimoku_senkou_b", "ichimoku_chikou", "supertrend",
                               "supertrend_direction", "psar", "psar_direction",
                               "aroon_up", "aroon_down", "aroon_oscillator",
                               "linreg_slope", "linreg_intercept", "linreg_r_squared", "ht_trendmode"]
                trend_values = {k: indicators_data.get(k) for k in trend_fields if k in indicators_data}
                if trend_values:
                    await self._upsert_indicator_row(
                        conn, "indicators_trend", symbol, tf.value, timestamp, trend_values, source
                    )
                    counts["trend"] = 1

        return counts

    async def _upsert_indicator_row(
        self,
        conn,
        table: str,
        symbol: str,
        timeframe: str,
        timestamp,
        values: Dict[str, Any],
        source: str
    ) -> None:
        """Helper: Einzelne Zeile in Indikator-Tabelle einfügen/aktualisieren."""
        columns = list(values.keys())
        placeholders = ", ".join([f"${i+4}" for i in range(len(columns))])
        column_names = ", ".join(columns)
        update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns])

        query = f"""
            INSERT INTO {table}
                (timestamp, symbol, timeframe, {column_names}, source)
            VALUES ($1, $2, $3, {placeholders}, ${len(columns)+4})
            ON CONFLICT (timestamp, symbol, timeframe)
            DO UPDATE SET {update_clause}, source = EXCLUDED.source
        """

        params = [timestamp, symbol, timeframe] + list(values.values()) + [source]
        await conn.execute(query, *params)

    # === Freshness Tracking ===

    async def _update_freshness(
        self,
        symbol: str,
        timeframe: str,
        data_type: str,
        record_count: int,
        source: str
    ) -> None:
        """Data Freshness Tracking aktualisieren."""
        query = """
            INSERT INTO data_freshness
                (symbol, timeframe, data_type, last_updated, last_timestamp, record_count, source)
            VALUES ($1, $2, $3, NOW(), NOW(), $4, $5)
            ON CONFLICT (symbol, timeframe, data_type)
            DO UPDATE SET
                last_updated = NOW(),
                record_count = data_freshness.record_count + EXCLUDED.record_count,
                source = EXCLUDED.source
        """

        async with self.connection() as conn:
            await conn.execute(query, symbol, timeframe, data_type, record_count, source)

    async def get_freshness(
        self,
        symbol: str,
        timeframe: str,
        data_type: str = "ohlcv"
    ) -> Optional[Dict[str, Any]]:
        """Aktualitätsstatus für Symbol/Timeframe abrufen."""
        query = """
            SELECT last_updated, last_timestamp, record_count, source
            FROM data_freshness
            WHERE symbol = $1 AND timeframe = $2 AND data_type = $3
        """

        async with self.connection() as conn:
            row = await conn.fetchrow(query, symbol, timeframe, data_type)

        if not row:
            return None

        return {
            "last_updated": row["last_updated"].isoformat(),
            "last_timestamp": row["last_timestamp"].isoformat(),
            "record_count": row["record_count"],
            "source": row["source"]
        }

    # === Schema Management ===

    async def _ensure_schema(self) -> None:
        """Datenbankschema erstellen falls nicht vorhanden."""
        # Schema-SQL aus separater Datei laden
        # oder hier inline definieren
        pass


# Singleton-Instanz
timescaledb_service = TimescaleDBService()
```

### 3.3 Data Repository

```python
# src/services/data_repository.py

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

from src.services.timescaledb_service import timescaledb_service
from src.services.cache_service import cache_service, CacheCategory
from src.config.timeframes import Timeframe, normalize_timeframe

logger = logging.getLogger(__name__)


class DataRepository:
    """
    Repository Pattern für Datenzugriff.

    Implementiert die 3-Layer-Caching-Strategie:
    1. Redis Cache (Hot Data)
    2. TimescaleDB (Persistente Speicherung)
    3. Externe APIs (Fallback)
    """

    def __init__(self):
        self._db = timescaledb_service
        self._cache = cache_service

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        OHLCV-Daten abrufen mit 3-Layer-Caching.

        Returns:
            Tuple von (data, source) wobei source 'cache', 'db' oder 'api' ist
        """
        tf = normalize_timeframe(timeframe)
        cache_key_params = {
            "limit": limit,
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None
        }

        # 1. Redis Cache Check (wenn nicht force_refresh)
        if not force_refresh:
            cached = await self._cache.get(
                CacheCategory.OHLCV,
                symbol,
                tf.value,
                params=cache_key_params
            )
            if cached:
                logger.debug(f"Cache HIT: {symbol}/{tf.value}")
                return cached, "cache"

        # 2. TimescaleDB Check
        db_data = await self._db.get_ohlcv(
            symbol=symbol,
            timeframe=tf.value,
            limit=limit,
            start_time=start_time,
            end_time=end_time
        )

        if db_data and self._is_data_fresh(db_data, tf):
            # In Redis cachen
            await self._cache.set(
                CacheCategory.OHLCV,
                db_data,
                symbol,
                tf.value,
                params=cache_key_params
            )
            logger.debug(f"DB HIT: {symbol}/{tf.value} ({len(db_data)} rows)")
            return db_data, "db"

        # 3. Daten von API abrufen (wird vom DataGateway gemacht)
        # Repository gibt None zurück, Gateway holt dann von API
        logger.debug(f"DB MISS: {symbol}/{tf.value} - API fetch required")
        return [], "api_required"

    async def save_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        source: str
    ) -> int:
        """
        OHLCV-Daten speichern (DB + Cache).

        Returns:
            Anzahl gespeicherter Zeilen
        """
        if not data:
            return 0

        tf = normalize_timeframe(timeframe)

        # In TimescaleDB speichern
        count = await self._db.upsert_ohlcv(
            symbol=symbol,
            timeframe=tf.value,
            data=data,
            source=source
        )

        # Cache invalidieren (wird bei nächstem Request neu gefüllt)
        await self._cache.delete_pattern(
            f"*:{CacheCategory.OHLCV.value}:{symbol}:{tf.value}:*"
        )

        logger.info(f"Saved {count} OHLCV rows: {symbol}/{tf.value} from {source}")
        return count

    async def get_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        limit: int = 100,
        force_refresh: bool = False
    ) -> tuple[List[Dict[str, Any]], str]:
        """Indikatoren abrufen mit Caching."""
        tf = normalize_timeframe(timeframe)
        cache_key_params = {"indicator": indicator_name, "limit": limit}

        # 1. Redis Cache
        if not force_refresh:
            cached = await self._cache.get(
                CacheCategory.INDICATORS,
                symbol,
                tf.value,
                params=cache_key_params
            )
            if cached:
                return cached, "cache"

        # 2. TimescaleDB
        db_data = await self._db.get_indicators(
            symbol=symbol,
            timeframe=tf.value,
            indicator_name=indicator_name,
            limit=limit
        )

        if db_data:
            await self._cache.set(
                CacheCategory.INDICATORS,
                db_data,
                symbol,
                tf.value,
                params=cache_key_params
            )
            return db_data, "db"

        return [], "api_required"

    async def save_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        data: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        source: str
    ) -> int:
        """Indikatoren speichern."""
        if not data:
            return 0

        tf = normalize_timeframe(timeframe)

        count = await self._db.upsert_indicators(
            symbol=symbol,
            timeframe=tf.value,
            indicator_name=indicator_name,
            data=data,
            parameters=parameters,
            source=source
        )

        # Cache invalidieren
        await self._cache.delete_pattern(
            f"*:{CacheCategory.INDICATORS.value}:{symbol}:{tf.value}:*"
        )

        return count

    def _is_data_fresh(
        self,
        data: List[Dict[str, Any]],
        timeframe: Timeframe
    ) -> bool:
        """
        Prüfen ob Daten aktuell genug sind.

        Basiert auf Timeframe-spezifischen Freshness-Regeln.
        """
        if not data:
            return False

        # Neuesten Timestamp finden
        latest = max(
            datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            for row in data
        )

        # Freshness-Intervalle pro Timeframe
        freshness_intervals = {
            Timeframe.M1: timedelta(minutes=2),
            Timeframe.M5: timedelta(minutes=10),
            Timeframe.M15: timedelta(minutes=30),
            Timeframe.M30: timedelta(hours=1),
            Timeframe.M45: timedelta(hours=2),
            Timeframe.H1: timedelta(hours=2),
            Timeframe.H2: timedelta(hours=4),
            Timeframe.H4: timedelta(hours=8),
            Timeframe.D1: timedelta(days=1),
            Timeframe.W1: timedelta(days=7),
            Timeframe.MN: timedelta(days=30),
        }

        max_age = freshness_intervals.get(timeframe, timedelta(hours=1))
        return datetime.now(latest.tzinfo) - latest < max_age


# Singleton
data_repository = DataRepository()
```

### 3.4 Angepasster Data Gateway Service

```python
# src/services/data_gateway_service.py (Änderungen)

from src.services.data_repository import data_repository

class DataGatewayService:
    """
    Zentrales Data Gateway - EINZIGER Zugriffspunkt für externe Daten.

    Erweitert um TimescaleDB-Persistenz.
    """

    async def get_historical_data(
        self,
        symbol: str,
        limit: int = 500,
        timeframe: str = "H1",
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Historische OHLCV-Daten abrufen.

        Datenfluss:
        1. Redis Cache → 2. TimescaleDB → 3. Externe API
        """
        tf = normalize_timeframe(timeframe)

        # 1. Repository (Cache + DB)
        data, source = await data_repository.get_ohlcv(
            symbol=symbol,
            timeframe=tf.value,
            limit=limit,
            force_refresh=force_refresh
        )

        if data:
            logger.debug(f"Data from {source}: {symbol}/{tf.value}")
            return data

        # 2. Externe API (Fallback-Kette)
        api_data, api_source = await self._fetch_from_api(symbol, tf, limit)

        if api_data:
            # In DB + Cache speichern
            await data_repository.save_ohlcv(
                symbol=symbol,
                timeframe=tf.value,
                data=api_data,
                source=api_source
            )
            return api_data

        return []

    async def _fetch_from_api(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Von externen APIs abrufen mit Fallback-Kette.

        TwelveData → EasyInsight → YFinance
        """
        # TwelveData (Primär)
        try:
            data = await self._twelvedata.get_time_series(
                symbol=symbol,
                interval=to_twelvedata(timeframe),
                outputsize=limit
            )
            if data and data.get("values"):
                return self._convert_twelvedata(data, symbol, timeframe), "twelvedata"
        except Exception as e:
            logger.warning(f"TwelveData failed: {e}")

        # EasyInsight (1. Fallback)
        try:
            data = await self._easyinsight.get_ohlcv(symbol, timeframe.value, limit)
            if data:
                return data, "easyinsight"
        except Exception as e:
            logger.warning(f"EasyInsight failed: {e}")

        # YFinance (2. Fallback)
        try:
            data = await self._yfinance.get_history(
                symbol=symbol,
                interval=to_yfinance(timeframe),
                limit=limit
            )
            if data:
                return data, "yfinance"
        except Exception as e:
            logger.warning(f"YFinance failed: {e}")

        return [], ""
```

---

## 4. Konfiguration

### 4.1 Settings

```python
# src/config/settings.py (Erweiterung)

class Settings(BaseSettings):
    # ... bestehende Settings ...

    # TimescaleDB
    timescale_host: str = Field(default="10.1.19.102")
    timescale_port: int = Field(default=5432)
    timescale_database: str = Field(default="tradingdataservice")
    timescale_user: str = Field(default="trading")
    timescale_password: str = Field(default="")
    timescale_pool_min: int = Field(default=5)
    timescale_pool_max: int = Field(default=20)

    @property
    def timescale_dsn(self) -> str:
        return (
            f"postgresql://{self.timescale_user}:{self.timescale_password}"
            f"@{self.timescale_host}:{self.timescale_port}"
            f"/{self.timescale_database}"
        )
```

### 4.2 Docker Compose

```yaml
# docker-compose.microservices.yml (Erweiterung)

services:
  data-service:
    # ... bestehende Config ...
    environment:
      - TIMESCALE_HOST=10.1.19.102
      - TIMESCALE_PORT=5432
      - TIMESCALE_DATABASE=tradingdataservice
      - TIMESCALE_USER=trading
      - TIMESCALE_PASSWORD=${TIMESCALE_PASSWORD}
    depends_on:
      - redis
    # Kein TimescaleDB Container hier - externe DB auf 10.1.19.102
```

### 4.3 Umgebungsvariablen

```bash
# .env (Erweiterung)

# TimescaleDB (externes Server)
TIMESCALE_HOST=10.1.19.102
TIMESCALE_PORT=5432
TIMESCALE_DATABASE=tradingdataservice
TIMESCALE_USER=trading
TIMESCALE_PASSWORD=<secure_password>
```

---

## 5. Datenbank-Setup auf 10.1.19.102

### 5.1 Initiales Setup

```bash
# Auf dem Server 10.1.19.102

# 1. PostgreSQL + TimescaleDB installieren (falls nicht vorhanden)
sudo apt-get install postgresql-16 postgresql-contrib-16
sudo apt-get install timescaledb-2-postgresql-16

# 2. TimescaleDB aktivieren
sudo timescaledb-tune

# 3. Datenbank erstellen
sudo -u postgres psql

CREATE DATABASE tradingdataservice;
CREATE USER trading WITH ENCRYPTED PASSWORD '<secure_password>';
GRANT ALL PRIVILEGES ON DATABASE tradingdataservice TO trading;

\c tradingdataservice

CREATE EXTENSION IF NOT EXISTS timescaledb;
GRANT ALL ON SCHEMA public TO trading;
```

### 5.2 Schema-Migration

```sql
-- migrations/001_initial_schema.sql

-- TimescaleDB Extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV Tabellen für alle Timeframes
DO $$
DECLARE
    tf_record RECORD;
    chunk_interval INTERVAL;
BEGIN
    FOR tf_record IN
        SELECT * FROM (VALUES
            ('m1', '1 day'),
            ('m5', '1 day'),
            ('m15', '7 days'),
            ('m30', '7 days'),
            ('m45', '7 days'),
            ('h1', '7 days'),
            ('h2', '14 days'),
            ('h4', '30 days'),
            ('d1', '365 days'),
            ('w1', '365 days'),
            ('mn', '365 days')
        ) AS t(timeframe, chunk_int)
    LOOP
        EXECUTE format('
            CREATE TABLE IF NOT EXISTS ohlcv_%s (
                timestamp       TIMESTAMPTZ NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                open            DECIMAL(20, 8) NOT NULL,
                high            DECIMAL(20, 8) NOT NULL,
                low             DECIMAL(20, 8) NOT NULL,
                close           DECIMAL(20, 8) NOT NULL,
                volume          DECIMAL(30, 8),
                source          VARCHAR(20) NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, symbol)
            )', tf_record.timeframe);

        -- In Hypertable konvertieren
        EXECUTE format(
            'SELECT create_hypertable(''ohlcv_%s'', ''timestamp'',
                chunk_time_interval => INTERVAL ''%s'',
                if_not_exists => TRUE)',
            tf_record.timeframe, tf_record.chunk_int
        );

        -- Indizes erstellen
        EXECUTE format('
            CREATE INDEX IF NOT EXISTS idx_ohlcv_%s_symbol
            ON ohlcv_%s (symbol, timestamp DESC)',
            tf_record.timeframe, tf_record.timeframe
        );
    END LOOP;
END $$;

-- =====================================================
-- INDIKATOREN-TABELLEN
-- =====================================================

-- Flexible JSONB-Tabelle für seltene/custom Indikatoren
CREATE TABLE IF NOT EXISTS indicators (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    indicator_name  VARCHAR(50) NOT NULL,
    values          JSONB NOT NULL,
    parameters      JSONB,
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe, indicator_name)
);

SELECT create_hypertable('indicators', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_lookup
    ON indicators (symbol, timeframe, indicator_name, timestamp DESC);

-- Partial Indexes für häufig abgefragte Indikatoren
CREATE INDEX IF NOT EXISTS idx_indicators_rsi
    ON indicators (symbol, timeframe, timestamp DESC)
    WHERE indicator_name = 'RSI';

CREATE INDEX IF NOT EXISTS idx_indicators_macd
    ON indicators (symbol, timeframe, timestamp DESC)
    WHERE indicator_name = 'MACD';

-- =====================================================
-- OPTIMIERTE INDIKATOR-TABELLEN (Performance-kritisch)
-- =====================================================

-- Moving Averages
CREATE TABLE IF NOT EXISTS indicators_ma (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    sma_20          DECIMAL(20, 8),
    sma_50          DECIMAL(20, 8),
    sma_200         DECIMAL(20, 8),
    ema_12          DECIMAL(20, 8),
    ema_26          DECIMAL(20, 8),
    ema_50          DECIMAL(20, 8),
    ema_200         DECIMAL(20, 8),
    wma_20          DECIMAL(20, 8),
    dema_20         DECIMAL(20, 8),
    tema_20         DECIMAL(20, 8),
    vwap            DECIMAL(20, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_ma', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_ma_lookup
    ON indicators_ma (symbol, timeframe, timestamp DESC);

-- Momentum-Indikatoren
CREATE TABLE IF NOT EXISTS indicators_momentum (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    rsi_14          DECIMAL(10, 4),
    rsi_7           DECIMAL(10, 4),
    rsi_21          DECIMAL(10, 4),
    stoch_rsi       DECIMAL(10, 4),
    connors_rsi     DECIMAL(10, 4),
    stoch_k         DECIMAL(10, 4),
    stoch_d         DECIMAL(10, 4),
    macd_line       DECIMAL(20, 8),
    macd_signal     DECIMAL(20, 8),
    macd_histogram  DECIMAL(20, 8),
    cci             DECIMAL(10, 4),
    williams_r      DECIMAL(10, 4),
    roc             DECIMAL(10, 4),
    momentum        DECIMAL(20, 8),
    adx             DECIMAL(10, 4),
    plus_di         DECIMAL(10, 4),
    minus_di        DECIMAL(10, 4),
    mfi             DECIMAL(10, 4),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_momentum', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_momentum_lookup
    ON indicators_momentum (symbol, timeframe, timestamp DESC);

-- Volatilitäts-Indikatoren
CREATE TABLE IF NOT EXISTS indicators_volatility (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    bb_upper        DECIMAL(20, 8),
    bb_middle       DECIMAL(20, 8),
    bb_lower        DECIMAL(20, 8),
    bb_width        DECIMAL(10, 6),
    bb_percent_b    DECIMAL(10, 6),
    atr_14          DECIMAL(20, 8),
    atr_7           DECIMAL(20, 8),
    natr            DECIMAL(10, 6),
    true_range      DECIMAL(20, 8),
    kc_upper        DECIMAL(20, 8),
    kc_middle       DECIMAL(20, 8),
    kc_lower        DECIMAL(20, 8),
    dc_upper        DECIMAL(20, 8),
    dc_middle       DECIMAL(20, 8),
    dc_lower        DECIMAL(20, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_volatility', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_volatility_lookup
    ON indicators_volatility (symbol, timeframe, timestamp DESC);

-- Trend-Indikatoren
CREATE TABLE IF NOT EXISTS indicators_trend (
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    timeframe           VARCHAR(10) NOT NULL,
    ichimoku_tenkan     DECIMAL(20, 8),
    ichimoku_kijun      DECIMAL(20, 8),
    ichimoku_senkou_a   DECIMAL(20, 8),
    ichimoku_senkou_b   DECIMAL(20, 8),
    ichimoku_chikou     DECIMAL(20, 8),
    supertrend          DECIMAL(20, 8),
    supertrend_direction INTEGER,
    psar                DECIMAL(20, 8),
    psar_direction      INTEGER,
    aroon_up            DECIMAL(10, 4),
    aroon_down          DECIMAL(10, 4),
    aroon_oscillator    DECIMAL(10, 4),
    linreg_slope        DECIMAL(20, 8),
    linreg_intercept    DECIMAL(20, 8),
    linreg_r_squared    DECIMAL(10, 6),
    ht_trendmode        INTEGER,
    source              VARCHAR(20) NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_trend', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_trend_lookup
    ON indicators_trend (symbol, timeframe, timestamp DESC);

-- Volumen-Indikatoren
CREATE TABLE IF NOT EXISTS indicators_volume (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    obv             DECIMAL(30, 8),
    ad_line         DECIMAL(30, 8),
    ad_oscillator   DECIMAL(20, 8),
    chaikin_mf      DECIMAL(10, 6),
    volume_sma_20   DECIMAL(30, 8),
    volume_ratio    DECIMAL(10, 4),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_volume', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_volume_lookup
    ON indicators_volume (symbol, timeframe, timestamp DESC);

-- Pivot Points & Levels
CREATE TABLE IF NOT EXISTS indicators_levels (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    pivot           DECIMAL(20, 8),
    r1              DECIMAL(20, 8),
    r2              DECIMAL(20, 8),
    r3              DECIMAL(20, 8),
    s1              DECIMAL(20, 8),
    s2              DECIMAL(20, 8),
    s3              DECIMAL(20, 8),
    fib_r1          DECIMAL(20, 8),
    fib_r2          DECIMAL(20, 8),
    fib_r3          DECIMAL(20, 8),
    fib_s1          DECIMAL(20, 8),
    fib_s2          DECIMAL(20, 8),
    fib_s3          DECIMAL(20, 8),
    cam_r1          DECIMAL(20, 8),
    cam_r2          DECIMAL(20, 8),
    cam_r3          DECIMAL(20, 8),
    cam_r4          DECIMAL(20, 8),
    cam_s1          DECIMAL(20, 8),
    cam_s2          DECIMAL(20, 8),
    cam_s3          DECIMAL(20, 8),
    cam_s4          DECIMAL(20, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_levels', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_levels_lookup
    ON indicators_levels (symbol, timeframe, timestamp DESC);

-- Data Freshness Tracking
CREATE TABLE IF NOT EXISTS data_freshness (
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    data_type       VARCHAR(30) NOT NULL,
    last_updated    TIMESTAMPTZ NOT NULL,
    last_timestamp  TIMESTAMPTZ NOT NULL,
    record_count    BIGINT DEFAULT 0,
    source          VARCHAR(20),
    PRIMARY KEY (symbol, timeframe, data_type)
);

-- Symbols Metadaten
CREATE TABLE IF NOT EXISTS symbols (
    symbol              VARCHAR(20) PRIMARY KEY,
    display_name        VARCHAR(100),
    category            VARCHAR(20) NOT NULL,
    subcategory         VARCHAR(20),
    base_currency       VARCHAR(10),
    quote_currency      VARCHAR(10),
    twelvedata_symbol   VARCHAR(20),
    easyinsight_symbol  VARCHAR(20),
    yfinance_symbol     VARCHAR(20),
    pip_value           DECIMAL(10, 6),
    min_lot_size        DECIMAL(10, 4),
    max_lot_size        DECIMAL(10, 2),
    lot_step            DECIMAL(10, 4),
    is_active           BOOLEAN DEFAULT TRUE,
    is_favorite         BOOLEAN DEFAULT FALSE,
    first_data_at       TIMESTAMPTZ,
    last_data_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Retention Policies (optional - alte Daten automatisch löschen)
-- M1: 30 Tage
SELECT add_retention_policy('ohlcv_m1', INTERVAL '30 days', if_not_exists => TRUE);
-- M5: 90 Tage
SELECT add_retention_policy('ohlcv_m5', INTERVAL '90 days', if_not_exists => TRUE);
-- Höhere Timeframes: Unbegrenzt (keine Retention Policy)
```

---

## 6. API-Erweiterungen

### 6.1 Neue Endpoints

```python
# src/services/data_app/api/db_routes.py

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime

from src.services.data_repository import data_repository
from src.services.timescaledb_service import timescaledb_service

router = APIRouter(prefix="/db", tags=["5. Database"])


@router.get("/ohlcv/{symbol}")
async def get_ohlcv_from_db(
    symbol: str,
    timeframe: str = Query(default="H1", description="Timeframe (M1, H1, D1, etc.)"),
    limit: int = Query(default=500, ge=1, le=10000),
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    force_refresh: bool = Query(default=False, description="Cache umgehen")
):
    """
    OHLCV-Daten direkt aus TimescaleDB abrufen.

    Verwendet den 3-Layer-Cache (Redis → TimescaleDB → API).
    """
    data, source = await data_repository.get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        force_refresh=force_refresh
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "count": len(data),
        "source": source,
        "data": data
    }


@router.get("/freshness/{symbol}")
async def get_data_freshness(
    symbol: str,
    timeframe: str = Query(default="H1")
):
    """Aktualitätsstatus für Symbol/Timeframe abrufen."""
    freshness = await timescaledb_service.get_freshness(symbol, timeframe)

    if not freshness:
        raise HTTPException(status_code=404, detail="No data found")

    return freshness


@router.get("/stats")
async def get_db_statistics():
    """Datenbankstatistiken abrufen."""
    # Implementierung: Hypertable-Statistiken, Chunk-Info, etc.
    pass


@router.post("/sync/{symbol}")
async def sync_symbol_data(
    symbol: str,
    timeframes: list[str] = Query(default=["H1", "D1"]),
    days_back: int = Query(default=365, ge=1, le=3650)
):
    """
    Historische Daten für Symbol synchronisieren.

    Holt Daten von externen APIs und speichert sie in TimescaleDB.
    """
    # Implementierung: Bulk-Sync von historischen Daten
    pass


# =====================================================
# INDIKATOR-ENDPOINTS
# =====================================================

@router.get("/indicators/{symbol}")
async def get_indicators_from_db(
    symbol: str,
    timeframe: str = Query(default="H1"),
    category: str = Query(
        default="all",
        description="Kategorie: all, momentum, volatility, trend, ma, volume, levels"
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    force_refresh: bool = Query(default=False)
):
    """
    Technische Indikatoren aus TimescaleDB abrufen.

    Kategorien:
    - all: Alle Indikatoren aus allen Tabellen
    - momentum: RSI, MACD, Stochastic, ADX, etc.
    - volatility: Bollinger Bands, ATR, Keltner, etc.
    - trend: Ichimoku, Supertrend, PSAR, Aroon, etc.
    - ma: Moving Averages (SMA, EMA, WMA, etc.)
    - volume: OBV, A/D, Chaikin, etc.
    - levels: Pivot Points, Fibonacci, Camarilla
    """
    if category == "all":
        data = await timescaledb_service.get_all_indicators(symbol, timeframe, limit)
        return {"symbol": symbol, "timeframe": timeframe, "indicators": data}

    method_map = {
        "momentum": timescaledb_service.get_momentum_indicators,
        "volatility": timescaledb_service.get_volatility_indicators,
        "trend": timescaledb_service.get_trend_indicators,
        "ma": timescaledb_service.get_ma_indicators,
        "volume": lambda s, t, l: timescaledb_service.get_all_indicators(s, t, l).get("volume", []),
        "levels": lambda s, t, l: timescaledb_service.get_all_indicators(s, t, l).get("levels", []),
    }

    if category not in method_map:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")

    data = await method_map[category](symbol, timeframe, limit)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "category": category,
        "count": len(data),
        "data": data
    }


@router.get("/indicators/{symbol}/momentum")
async def get_momentum_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000),
    indicators: Optional[List[str]] = Query(
        default=None,
        description="Spezifische Indikatoren: rsi_14, macd_line, stoch_k, etc."
    )
):
    """
    Momentum-Indikatoren abrufen.

    Verfügbare Indikatoren:
    - rsi_14, rsi_7, rsi_21, stoch_rsi, connors_rsi
    - stoch_k, stoch_d
    - macd_line, macd_signal, macd_histogram
    - cci, williams_r, roc, momentum
    - adx, plus_di, minus_di, mfi
    """
    data = await timescaledb_service.get_momentum_indicators(
        symbol, timeframe, limit, indicators
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "category": "momentum",
        "count": len(data),
        "data": data
    }


@router.get("/indicators/{symbol}/volatility")
async def get_volatility_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    Volatilitäts-Indikatoren abrufen.

    Enthält: Bollinger Bands, ATR, Keltner Channel, Donchian Channel
    """
    data = await timescaledb_service.get_volatility_indicators(symbol, timeframe, limit)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "category": "volatility",
        "count": len(data),
        "data": data
    }


@router.get("/indicators/{symbol}/trend")
async def get_trend_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    Trend-Indikatoren abrufen.

    Enthält: Ichimoku, Supertrend, Parabolic SAR, Aroon, Linear Regression
    """
    data = await timescaledb_service.get_trend_indicators(symbol, timeframe, limit)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "category": "trend",
        "count": len(data),
        "data": data
    }


@router.get("/indicators/{symbol}/ma")
async def get_ma_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    Moving Average Indikatoren abrufen.

    Enthält: SMA (20, 50, 200), EMA (12, 26, 50, 200), WMA, DEMA, TEMA, VWAP
    """
    data = await timescaledb_service.get_ma_indicators(symbol, timeframe, limit)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "category": "moving_averages",
        "count": len(data),
        "data": data
    }


@router.post("/indicators/{symbol}/sync")
async def sync_indicators(
    symbol: str,
    timeframe: str = Query(default="H1"),
    categories: List[str] = Query(
        default=["momentum", "volatility", "trend"],
        description="Zu synchronisierende Kategorien"
    ),
    days_back: int = Query(default=30, ge=1, le=365)
):
    """
    Indikatoren für Symbol synchronisieren.

    Berechnet Indikatoren aus OHLCV-Daten und speichert sie in TimescaleDB.
    """
    # Implementierung: Indikator-Berechnung + Speicherung
    results = {}
    for category in categories:
        # 1. OHLCV-Daten laden
        # 2. Indikatoren berechnen
        # 3. In DB speichern
        results[category] = {"status": "pending", "records": 0}

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "sync_results": results
    }


@router.get("/indicators/freshness/{symbol}")
async def get_indicator_freshness(
    symbol: str,
    timeframe: str = Query(default="H1")
):
    """Aktualitätsstatus aller Indikator-Kategorien abrufen."""
    categories = ["indicators_ma", "indicators_momentum", "indicators_volatility",
                  "indicators_trend", "indicators_volume", "indicators_levels"]

    freshness = {}
    for cat in categories:
        data = await timescaledb_service.get_freshness(symbol, timeframe, cat)
        if data:
            freshness[cat.replace("indicators_", "")] = data

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "freshness": freshness
    }
```

---

## 7. Migration bestehender Daten

### 7.1 Migrations-Strategie

```python
# scripts/migrate_redis_to_timescale.py

"""
Migration von Redis-Cache-Daten zu TimescaleDB.

Liest alle gecachten OHLCV-Daten aus Redis und
speichert sie persistent in TimescaleDB.
"""

import asyncio
import redis.asyncio as redis
import json
from src.services.timescaledb_service import timescaledb_service


async def migrate_cached_data():
    """Migriert alle OHLCV-Daten aus Redis nach TimescaleDB."""

    # Redis-Verbindung
    redis_client = redis.Redis(host='localhost', port=6379)

    # Alle OHLCV-Keys finden
    pattern = "trading:ohlcv:*"
    keys = await redis_client.keys(pattern)

    print(f"Found {len(keys)} OHLCV cache entries")

    for key in keys:
        try:
            # Key parsen: trading:ohlcv:BTCUSD:H1:hash
            parts = key.decode().split(":")
            if len(parts) >= 4:
                symbol = parts[2]
                timeframe = parts[3]

                # Daten aus Redis lesen
                data = await redis_client.get(key)
                if data:
                    records = json.loads(data)

                    # In TimescaleDB speichern
                    await timescaledb_service.upsert_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=records,
                        source="redis_migration"
                    )
                    print(f"Migrated {len(records)} records: {symbol}/{timeframe}")

        except Exception as e:
            print(f"Error migrating {key}: {e}")

    await redis_client.close()
    print("Migration completed!")


if __name__ == "__main__":
    asyncio.run(migrate_cached_data())
```

### 7.2 Initiale Datenbefüllung

```python
# scripts/initial_data_load.py

"""
Initiales Laden von historischen Daten für alle Symbole.
"""

import asyncio
from datetime import datetime, timedelta

from src.services.data_gateway_service import data_gateway
from src.services.timescaledb_service import timescaledb_service
from src.config.timeframes import Timeframe


SYMBOLS = ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "XAUUSD", "US500"]
TIMEFRAMES = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
DAYS_BACK = {
    Timeframe.M15: 30,   # 30 Tage für M15
    Timeframe.H1: 180,   # 6 Monate für H1
    Timeframe.H4: 365,   # 1 Jahr für H4
    Timeframe.D1: 730,   # 2 Jahre für D1
}


async def load_historical_data():
    """Lädt historische Daten für alle konfigurierten Symbole."""

    await timescaledb_service.initialize()

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            days = DAYS_BACK.get(tf, 30)
            limit = calculate_limit_for_days(tf, days)

            print(f"Loading {symbol}/{tf.value}: {days} days ({limit} candles)")

            try:
                data = await data_gateway.get_historical_data(
                    symbol=symbol,
                    timeframe=tf.value,
                    limit=limit,
                    force_refresh=True
                )
                print(f"  ✓ Loaded {len(data)} records")

            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Rate Limiting respektieren
            await asyncio.sleep(0.5)

    await timescaledb_service.close()
    print("\nInitial data load completed!")


if __name__ == "__main__":
    asyncio.run(load_historical_data())
```

---

## 8. Monitoring & Wartung

### 8.1 Health Check Erweiterung

```python
# src/services/data_app/api/health.py (Erweiterung)

@router.get("/health")
async def health_check():
    """Erweiterter Health Check inkl. TimescaleDB."""

    # Redis Status
    redis_healthy = await cache_service.health_check()

    # TimescaleDB Status
    try:
        async with timescaledb_service.connection() as conn:
            await conn.execute("SELECT 1")
        db_healthy = True
        db_error = None
    except Exception as e:
        db_healthy = False
        db_error = str(e)

    return {
        "status": "healthy" if (redis_healthy and db_healthy) else "degraded",
        "redis": {
            "status": "healthy" if redis_healthy else "unhealthy",
            **cache_service.get_stats()
        },
        "timescaledb": {
            "status": "healthy" if db_healthy else "unhealthy",
            "host": settings.timescale_host,
            "database": settings.timescale_database,
            "error": db_error
        }
    }
```

### 8.2 Retention & Compression

```sql
-- Automatische Kompression für ältere Daten
ALTER TABLE ohlcv_m1 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Kompression nach 7 Tagen
SELECT add_compression_policy('ohlcv_m1', INTERVAL '7 days');

-- Für H1: Kompression nach 30 Tagen
ALTER TABLE ohlcv_h1 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_h1', INTERVAL '30 days');

-- Für D1: Kompression nach 90 Tagen
ALTER TABLE ohlcv_d1 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_d1', INTERVAL '90 days');
```

---

## 9. Implementierungsplan

### Phase 1: Infrastruktur (1-2 Tage)

1. [ ] TimescaleDB auf 10.1.19.102 installieren und konfigurieren
2. [ ] Datenbank `tradingdataservice` erstellen
3. [ ] Benutzer und Berechtigungen einrichten
4. [ ] Schema-Migration ausführen
5. [ ] Netzwerk-Konnektivität von Data Service zu DB testen

### Phase 2: Service-Integration (2-3 Tage)

1. [ ] `timescaledb_service.py` implementieren
2. [ ] `data_repository.py` implementieren
3. [ ] `data_gateway_service.py` anpassen
4. [ ] Settings und Konfiguration erweitern
5. [ ] Unit Tests schreiben

### Phase 3: API & Migration (1-2 Tage)

1. [ ] Neue API-Endpoints implementieren (`/db/*`)
2. [ ] Redis-zu-TimescaleDB Migrationsskript erstellen
3. [ ] Initiales Daten-Load-Skript erstellen
4. [ ] Health Checks erweitern

### Phase 4: Testing & Deployment (1-2 Tage)

1. [ ] Integration Tests
2. [ ] Performance Tests (Latenz, Durchsatz)
3. [ ] Docker-Konfiguration aktualisieren
4. [ ] Dokumentation aktualisieren
5. [ ] Production Deployment

---

## 10. Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| DB-Verbindungsausfall | Mittel | Hoch | Fallback auf Redis-only Modus |
| Hohe Latenz durch Netzwerk | Niedrig | Mittel | Connection Pooling, prepared statements |
| Speicherplatz-Probleme | Niedrig | Mittel | Retention Policies, Kompression |
| Daten-Inkonsistenz | Niedrig | Hoch | Transaktionen, Upsert statt Insert |
| Migration bricht ab | Mittel | Niedrig | Idempotente Migration, Checkpoints |

---

## 11. Vorteile der neuen Architektur

1. **Persistente Datenspeicherung**: Daten überleben Redis-Neustarts und Cache-Eviction
2. **Historische Analysen**: Zugriff auf beliebig alte Daten ohne API-Limits
3. **Reduzierte API-Kosten**: Weniger externe API-Calls durch lokale Datenhaltung
4. **Bessere Performance**: TimescaleDB optimiert für Zeitreihendaten
5. **Flexible Abfragen**: SQL-Abfragen für komplexe Analysen möglich
6. **Skalierbarkeit**: Hypertables partitionieren automatisch
7. **Kompression**: Ältere Daten werden komprimiert gespeichert

---

## 12. Offene Fragen

1. **Backup-Strategie**: Wie oft soll die DB gesichert werden?
2. **Replikation**: Soll ein Read-Replica eingerichtet werden?
3. **Daten-Retention**: Wie lange sollen M1-Daten aufbewahrt werden?
4. **API-Rate-Limits**: Sollen Bulk-Sync-Operationen gedrosselt werden?
