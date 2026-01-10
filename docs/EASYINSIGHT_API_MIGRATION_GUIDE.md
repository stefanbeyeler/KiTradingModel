# EasyInsight API Migration Guide

## Ziel: TwelveData-kompatible API-Endpoints

Dieses Dokument beschreibt die erforderlichen API-Änderungen, damit EasyInsight als vollwertige Alternative zu TwelveData fungieren kann.

---

## Inhaltsverzeichnis

1. [Aktuelle EasyInsight-Struktur](#1-aktuelle-easyinsight-struktur)
2. [Ziel-API-Struktur (TwelveData-kompatibel)](#2-ziel-api-struktur-twelvedata-kompatibel)
3. [Neue Endpoints im Detail](#3-neue-endpoints-im-detail)
4. [Timeframe-Standardisierung](#4-timeframe-standardisierung)
5. [Indikator-Mapping](#5-indikator-mapping)
6. [Response-Formate](#6-response-formate)
7. [Implementierungs-Checkliste](#7-implementierungs-checkliste)

---

## 1. Aktuelle EasyInsight-Struktur

### 1.1 Vorhandene Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/symbols` | GET | Liste aller Symbole |
| `/symbol-data-full/{symbol}` | GET | OHLCV + alle Indikatoren (H1) |
| `/symbol-latest-full/{symbol}` | GET | Neuester Snapshot |
| `/logs` | GET | MT5 Trading Logs |
| `/health` | GET | Service Health-Check |

### 1.2 Vorhandene Datenfelder

**OHLCV-Daten (H1 Timeframe):**
```
h1_open, h1_high, h1_low, h1_close, volume
```

**OHLCV-Daten (M15 Timeframe):**
```
m15_open, m15_high, m15_low, m15_close
```

**OHLCV-Daten (D1 - Rolling, nicht historisch):**
```
d1_open, d1_high, d1_low, d1_close
```

**Momentum-Indikatoren:**
```
rsi                 - Relative Strength Index (14)
macd_main           - MACD Hauptlinie
macd_signal         - MACD Signallinie
stoch_k             - Stochastic %K (Stochastik Fast)
stoch_d             - Stochastic %D (Stochastik Slow)
cci                 - Commodity Channel Index
```

**Volatilitäts-Indikatoren:**
```
atr                 - Average True Range
atr_pct             - ATR als Prozent vom Preis
bb_upper            - Bollinger Band Upper
bb_middle           - Bollinger Band Middle (SMA 20)
bb_lower            - Bollinger Band Lower
range_d1            - Tagesrange (High - Low)
```

**Trend-Indikatoren:**
```
ma100               - Moving Average 100
adx                 - Average Directional Index
adx_plus_di         - +DI (Positive Directional Indicator)
adx_minus_di        - -DI (Negative Directional Indicator)
```

**Ichimoku-Indikatoren:**
```
ichimoku_tenkan     - Tenkan-Sen (Conversion Line)
ichimoku_kijun      - Kijun-Sen (Base Line)
ichimoku_senkou_a   - Senkou Span A (Leading Span A)
ichimoku_senkou_b   - Senkou Span B (Leading Span B)
ichimoku_chikou     - Chikou Span (Lagging Span)
```

**EasyInsight-Proprietäre Indikatoren:**
```
strength            - Multi-Timeframe Trendstärke (0-100)
strength_4h         - Währungsstärke 4H
strength_1d         - Währungsstärke Daily
strength_1w         - Währungsstärke Weekly
```

**Support/Resistance:**
```
s1_level            - Support Level 1
r1_level            - Resistance Level 1
```

---

## 2. Ziel-API-Struktur (TwelveData-kompatibel)

### 2.1 Neue Kern-Endpoints

| Priorität | Endpoint | Methode | Beschreibung |
|-----------|----------|---------|--------------|
| **HOCH** | `/time_series/{symbol}` | GET | OHLCV für alle Timeframes |
| **HOCH** | `/rsi/{symbol}` | GET | RSI Indikator |
| **HOCH** | `/macd/{symbol}` | GET | MACD Indikator |
| **HOCH** | `/bbands/{symbol}` | GET | Bollinger Bands |
| **HOCH** | `/stoch/{symbol}` | GET | Stochastic Oscillator |
| **HOCH** | `/adx/{symbol}` | GET | ADX mit +DI/-DI |
| **HOCH** | `/atr/{symbol}` | GET | Average True Range |
| **MITTEL** | `/ema/{symbol}` | GET | Exponential MA |
| **MITTEL** | `/sma/{symbol}` | GET | Simple MA |
| **MITTEL** | `/cci/{symbol}` | GET | CCI Indikator |
| **MITTEL** | `/ichimoku/{symbol}` | GET | Ichimoku Cloud |
| **MITTEL** | `/indicators/{symbol}` | GET | Mehrere Indikatoren |
| **NIEDRIG** | `/quote/{symbol}` | GET | Echtzeit-Quote |
| **NIEDRIG** | `/price/{symbol}` | GET | Nur aktueller Preis |

### 2.2 Utility-Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/symbols` | GET | Symbol-Listen (bereits vorhanden) |
| `/status` | GET | API-Status und Rate Limits |

---

## 3. Neue Endpoints im Detail

### 3.1 Time Series (OHLCV) - KRITISCH

**Endpoint:** `GET /time_series/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe (siehe Abschnitt 4) |
| `outputsize` | integer | `100` | Anzahl Kerzen (max. 5000) |
| `start_date` | string | - | Start-Datum (YYYY-MM-DD oder YYYY-MM-DD HH:mm:ss) |
| `end_date` | string | - | End-Datum |
| `order` | string | `desc` | Sortierung: `asc` oder `desc` |

**Request-Beispiel:**
```
GET /time_series/BTCUSD?interval=1h&outputsize=500
GET /time_series/EURUSD?interval=1day&start_date=2024-01-01&end_date=2024-01-31
```

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "currency_base": "BTC",
    "currency_quote": "USD",
    "type": "Digital Currency"
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "open": "45000.50",
      "high": "45100.00",
      "low": "44950.25",
      "close": "45050.00",
      "volume": "1250000"
    },
    {
      "datetime": "2024-01-15 09:00:00",
      "open": "44900.00",
      "high": "45010.00",
      "low": "44850.00",
      "close": "45000.50",
      "volume": "1180000"
    }
  ],
  "status": "ok"
}
```

**Wichtige Hinweise:**
- `datetime` Format: `YYYY-MM-DD HH:mm:ss` (ohne Timezone-Suffix)
- Alle numerischen Werte als **Strings** (TwelveData-Kompatibilität)
- `values` Array ist nach `datetime` sortiert (neueste zuerst bei `order=desc`)

---

### 3.2 RSI Indikator

**Endpoint:** `GET /rsi/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `time_period` | integer | `14` | RSI-Periode |
| `series_type` | string | `close` | Preistyp (open, high, low, close) |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "RSI - Relative Strength Index",
      "time_period": 14,
      "series_type": "close"
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "rsi": "65.50"
    },
    {
      "datetime": "2024-01-15 09:00:00",
      "rsi": "62.30"
    }
  ],
  "status": "ok"
}
```

---

### 3.3 MACD Indikator

**Endpoint:** `GET /macd/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `fast_period` | integer | `12` | Schnelle EMA-Periode |
| `slow_period` | integer | `26` | Langsame EMA-Periode |
| `signal_period` | integer | `9` | Signal-Periode |
| `series_type` | string | `close` | Preistyp |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "MACD - Moving Average Convergence Divergence",
      "fast_period": 12,
      "slow_period": 26,
      "signal_period": 9,
      "series_type": "close"
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "macd": "250.30",
      "macd_signal": "240.50",
      "macd_hist": "9.80"
    }
  ],
  "status": "ok"
}
```

**Mapping von aktuellen EasyInsight-Feldern:**
```
macd_main   → macd
macd_signal → macd_signal
(berechnen) → macd_hist = macd - macd_signal
```

---

### 3.4 Bollinger Bands

**Endpoint:** `GET /bbands/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `time_period` | integer | `20` | SMA-Periode |
| `sd` | float | `2.0` | Standardabweichung |
| `ma_type` | string | `SMA` | MA-Typ (SMA, EMA) |
| `series_type` | string | `close` | Preistyp |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "BBANDS - Bollinger Bands",
      "time_period": 20,
      "sd": 2.0,
      "ma_type": "SMA",
      "series_type": "close"
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "upper_band": "45500.00",
      "middle_band": "45000.00",
      "lower_band": "44500.00"
    }
  ],
  "status": "ok"
}
```

**Mapping von aktuellen EasyInsight-Feldern:**
```
bb_upper  → upper_band
bb_middle → middle_band
bb_lower  → lower_band
```

---

### 3.5 Stochastic Oscillator

**Endpoint:** `GET /stoch/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `fast_k_period` | integer | `14` | %K Periode |
| `slow_k_period` | integer | `3` | %K Glättung |
| `slow_d_period` | integer | `3` | %D Periode |
| `slow_kma_type` | string | `SMA` | MA-Typ für %K |
| `slow_dma_type` | string | `SMA` | MA-Typ für %D |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "STOCH - Stochastic Oscillator",
      "fast_k_period": 14,
      "slow_k_period": 3,
      "slow_d_period": 3
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "slow_k": "72.30",
      "slow_d": "68.90"
    }
  ],
  "status": "ok"
}
```

**Mapping von aktuellen EasyInsight-Feldern:**
```
stoch_k → slow_k
stoch_d → slow_d
```

---

### 3.6 ADX Indikator

**Endpoint:** `GET /adx/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `time_period` | integer | `14` | ADX-Periode |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "ADX - Average Directional Index",
      "time_period": 14
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "adx": "35.20",
      "plus_di": "28.50",
      "minus_di": "12.30"
    }
  ],
  "status": "ok"
}
```

**Mapping von aktuellen EasyInsight-Feldern:**
```
adx          → adx
adx_plus_di  → plus_di
adx_minus_di → minus_di
```

---

### 3.7 ATR Indikator

**Endpoint:** `GET /atr/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `time_period` | integer | `14` | ATR-Periode |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "ATR - Average True Range",
      "time_period": 14
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "atr": "180.50"
    }
  ],
  "status": "ok"
}
```

---

### 3.8 EMA Indikator

**Endpoint:** `GET /ema/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `time_period` | integer | `20` | EMA-Periode |
| `series_type` | string | `close` | Preistyp |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "EMA - Exponential Moving Average",
      "time_period": 20,
      "series_type": "close"
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "ema": "45020.00"
    }
  ],
  "status": "ok"
}
```

---

### 3.9 SMA Indikator

**Endpoint:** `GET /sma/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `time_period` | integer | `20` | SMA-Periode |
| `series_type` | string | `close` | Preistyp |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "SMA - Simple Moving Average",
      "time_period": 20,
      "series_type": "close"
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "sma": "45000.00"
    }
  ],
  "status": "ok"
}
```

**Hinweis:** Der aktuelle `ma100` kann als `time_period=100` bereitgestellt werden.

---

### 3.10 CCI Indikator

**Endpoint:** `GET /cci/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `time_period` | integer | `20` | CCI-Periode |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "CCI - Commodity Channel Index",
      "time_period": 20
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "cci": "125.40"
    }
  ],
  "status": "ok"
}
```

---

### 3.11 Ichimoku Cloud

**Endpoint:** `GET /ichimoku/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `conversion_line_period` | integer | `9` | Tenkan-Sen Periode |
| `base_line_period` | integer | `26` | Kijun-Sen Periode |
| `leading_span_b_period` | integer | `52` | Senkou Span B Periode |
| `lagging_span_period` | integer | `26` | Chikou Span Verschiebung |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "Ichimoku Cloud",
      "conversion_line_period": 9,
      "base_line_period": 26,
      "leading_span_b_period": 52,
      "lagging_span_period": 26
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "tenkan_sen": "42450.00",
      "kijun_sen": "42200.00",
      "senkou_span_a": "42350.00",
      "senkou_span_b": "42100.00",
      "chikou_span": "42520.00"
    }
  ],
  "status": "ok"
}
```

**Mapping von aktuellen EasyInsight-Feldern:**
```
ichimoku_tenkan   → tenkan_sen
ichimoku_kijun    → kijun_sen
ichimoku_senkou_a → senkou_span_a
ichimoku_senkou_b → senkou_span_b
ichimoku_chikou   → chikou_span
```

---

### 3.12 Multiple Indicators (Batch-Request)

**Endpoint:** `GET /indicators/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `indicators` | string | - | Kommaseparierte Liste (z.B. `rsi,macd,bbands`) |
| `outputsize` | integer | `100` | Anzahl Werte |

**Request-Beispiel:**
```
GET /indicators/BTCUSD?interval=1h&indicators=rsi,macd,bbands,stoch&outputsize=100
```

**Response-Format:**
```json
{
  "symbol": "BTCUSD",
  "interval": "1h",
  "indicators": {
    "rsi": {
      "meta": {
        "name": "RSI",
        "time_period": 14
      },
      "values": [
        {"datetime": "2024-01-15 10:00:00", "rsi": "65.50"}
      ]
    },
    "macd": {
      "meta": {
        "name": "MACD",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      },
      "values": [
        {"datetime": "2024-01-15 10:00:00", "macd": "250.30", "macd_signal": "240.50", "macd_hist": "9.80"}
      ]
    },
    "bbands": {
      "meta": {
        "name": "BBANDS",
        "time_period": 20,
        "sd": 2.0
      },
      "values": [
        {"datetime": "2024-01-15 10:00:00", "upper_band": "45500.00", "middle_band": "45000.00", "lower_band": "44500.00"}
      ]
    },
    "stoch": {
      "meta": {
        "name": "STOCH",
        "fast_k_period": 14
      },
      "values": [
        {"datetime": "2024-01-15 10:00:00", "slow_k": "72.30", "slow_d": "68.90"}
      ]
    }
  },
  "errors": [],
  "status": "ok"
}
```

---

### 3.13 Quote (Echtzeit)

**Endpoint:** `GET /quote/{symbol}`

**Response-Format:**
```json
{
  "symbol": "BTCUSD",
  "name": "Bitcoin US Dollar",
  "exchange": "Crypto",
  "datetime": "2024-01-15 10:30:45",
  "timestamp": 1705315845,
  "open": "45000.50",
  "high": "45100.00",
  "low": "44950.25",
  "close": "45050.00",
  "volume": "1250000",
  "previous_close": "44900.00",
  "change": "150.00",
  "percent_change": "0.33",
  "fifty_two_week": {
    "low": "25000.00",
    "high": "48000.00"
  }
}
```

---

### 3.14 Price (Lightweight)

**Endpoint:** `GET /price/{symbol}`

**Response-Format:**
```json
{
  "price": "45050.00"
}
```

---

### 3.15 EasyInsight-Proprietäre Indikatoren (Optional)

Diese Indikatoren sind EasyInsight-spezifisch und können zusätzlich zu den TwelveData-kompatiblen Endpoints angeboten werden.

**Endpoint:** `GET /strength/{symbol}`

**Query-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `interval` | string | `1h` | Timeframe |
| `outputsize` | integer | `100` | Anzahl Werte |

**Response-Format:**
```json
{
  "meta": {
    "symbol": "BTCUSD",
    "interval": "1h",
    "indicator": {
      "name": "EasyInsight Multi-Timeframe Strength",
      "description": "Proprietary trend strength indicator"
    }
  },
  "values": [
    {
      "datetime": "2024-01-15 10:00:00",
      "strength": "75.50",
      "strength_4h": "68.20",
      "strength_1d": "71.80",
      "strength_1w": "62.50"
    }
  ],
  "status": "ok"
}
```

---

## 4. Timeframe-Standardisierung

### 4.1 Unterstützte Interval-Werte

EasyInsight muss folgende `interval`-Werte akzeptieren:

| Interval | Bedeutung | Alias (optional) |
|----------|-----------|------------------|
| `1min` | 1 Minute | `1m` |
| `5min` | 5 Minuten | `5m` |
| `15min` | 15 Minuten | `15m` |
| `30min` | 30 Minuten | `30m` |
| `1h` | 1 Stunde | `60min` |
| `4h` | 4 Stunden | `240min` |
| `1day` | 1 Tag | `daily`, `1d` |
| `1week` | 1 Woche | `weekly`, `1w` |
| `1month` | 1 Monat | `monthly`, `1mo` |

### 4.2 Internes Mapping

```python
INTERVAL_TO_INTERNAL = {
    # TwelveData-Format
    "1min": "M1",
    "5min": "M5",
    "15min": "M15",
    "30min": "M30",
    "1h": "H1",
    "4h": "H4",
    "1day": "D1",
    "1week": "W1",
    "1month": "MN",
    # Aliase
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "60min": "H1",
    "240min": "H4",
    "daily": "D1",
    "1d": "D1",
    "weekly": "W1",
    "1w": "W1",
    "monthly": "MN",
    "1mo": "MN",
}
```

### 4.3 Priorität der Timeframes

| Priorität | Timeframe | Anforderung |
|-----------|-----------|-------------|
| **KRITISCH** | `1h` | Primärer Timeframe, vollständige Daten |
| **KRITISCH** | `1day` | Historische Daily-Kerzen (nicht rolling!) |
| **HOCH** | `15min` | Für kurzfristiges Trading |
| **HOCH** | `4h` | Für Swing-Trading |
| **MITTEL** | `5min` | Für Scalping |
| **MITTEL** | `1week` | Für langfristige Analyse |
| **NIEDRIG** | `1min` | Für High-Frequency-Daten |
| **NIEDRIG** | `30min` | Weniger verwendet |
| **NIEDRIG** | `1month` | Für sehr langfristige Analyse |

---

## 5. Indikator-Mapping

### 5.1 Aktuell → TwelveData-Format

| EasyInsight Feld | TwelveData Feld | Endpoint |
|------------------|-----------------|----------|
| `rsi` | `rsi` | `/rsi` |
| `macd_main` | `macd` | `/macd` |
| `macd_signal` | `macd_signal` | `/macd` |
| *(berechnet)* | `macd_hist` | `/macd` |
| `stoch_k` | `slow_k` | `/stoch` |
| `stoch_d` | `slow_d` | `/stoch` |
| `bb_upper` | `upper_band` | `/bbands` |
| `bb_middle` | `middle_band` | `/bbands` |
| `bb_lower` | `lower_band` | `/bbands` |
| `adx` | `adx` | `/adx` |
| `adx_plus_di` | `plus_di` | `/adx` |
| `adx_minus_di` | `minus_di` | `/adx` |
| `atr` | `atr` | `/atr` |
| `cci` | `cci` | `/cci` |
| `ma100` | `sma` (period=100) | `/sma` |
| `ichimoku_tenkan` | `tenkan_sen` | `/ichimoku` |
| `ichimoku_kijun` | `kijun_sen` | `/ichimoku` |
| `ichimoku_senkou_a` | `senkou_span_a` | `/ichimoku` |
| `ichimoku_senkou_b` | `senkou_span_b` | `/ichimoku` |
| `ichimoku_chikou` | `chikou_span` | `/ichimoku` |

### 5.2 Neue Indikatoren (zu implementieren)

Diese Indikatoren müssen neu berechnet werden:

| Indikator | Formel/Beschreibung | Priorität |
|-----------|---------------------|-----------|
| `ema` | Exponential Moving Average | HOCH |
| `sma` | Simple Moving Average (verschiedene Perioden) | HOCH |
| `wma` | Weighted Moving Average | MITTEL |
| `obv` | On-Balance Volume | MITTEL |
| `willr` | Williams %R | MITTEL |
| `mfi` | Money Flow Index | NIEDRIG |
| `aroon` | Aroon Indicator | NIEDRIG |

---

## 6. Response-Formate

### 6.1 Erfolgreiche Response

```json
{
  "meta": { ... },
  "values": [ ... ],
  "status": "ok"
}
```

### 6.2 Fehler-Response

```json
{
  "code": 400,
  "message": "Invalid symbol: INVALID",
  "status": "error"
}
```

**Standard-Fehlercodes:**

| Code | Bedeutung |
|------|-----------|
| `400` | Bad Request (ungültige Parameter) |
| `401` | Unauthorized (fehlende/ungültige API-Key) |
| `404` | Symbol nicht gefunden |
| `429` | Rate Limit überschritten |
| `500` | Interner Server-Fehler |

### 6.3 Datentypen

**WICHTIG:** Alle numerischen Werte werden als **Strings** zurückgegeben (TwelveData-Kompatibilität):

```json
{
  "open": "45000.50",    // String, nicht Number
  "high": "45100.00",
  "low": "44950.25",
  "close": "45050.00",
  "volume": "1250000",
  "rsi": "65.50"
}
```

### 6.4 Datetime-Format

```
Format: YYYY-MM-DD HH:mm:ss
Beispiel: 2024-01-15 10:00:00
```

**Keine Timezone-Angabe im String** - Zeitzone wird im Meta-Block angegeben falls nötig.

---

## 7. Implementierungs-Checkliste

### Phase 1: Kern-Endpoints (KRITISCH)

- [ ] `GET /time_series/{symbol}` - OHLCV für alle Timeframes
  - [ ] Support für `interval` Parameter
  - [ ] Support für `outputsize` Parameter
  - [ ] Support für `start_date` / `end_date`
  - [ ] Historische Daily-Daten (nicht rolling!)

- [ ] `GET /rsi/{symbol}` - RSI Indikator
  - [ ] Konfigurierbarer `time_period`

- [ ] `GET /macd/{symbol}` - MACD Indikator
  - [ ] `macd_hist` Berechnung hinzufügen

- [ ] `GET /bbands/{symbol}` - Bollinger Bands
  - [ ] Konfigurierbarer `sd` Parameter

- [ ] `GET /stoch/{symbol}` - Stochastic
  - [ ] Feld-Umbenennung zu `slow_k`, `slow_d`

- [ ] `GET /adx/{symbol}` - ADX
  - [ ] Feld-Umbenennung zu `plus_di`, `minus_di`

- [ ] `GET /atr/{symbol}` - ATR

### Phase 2: Erweiterte Indikatoren

- [ ] `GET /ema/{symbol}` - EMA (neu berechnen)
- [ ] `GET /sma/{symbol}` - SMA mit variablen Perioden
- [ ] `GET /cci/{symbol}` - CCI
- [ ] `GET /ichimoku/{symbol}` - Ichimoku Cloud
- [ ] `GET /indicators/{symbol}` - Batch-Request

### Phase 3: Utility & Zusätzliche Indikatoren

- [ ] `GET /quote/{symbol}` - Echtzeit-Quote
- [ ] `GET /price/{symbol}` - Lightweight Preis
- [ ] `GET /status` - API-Status
- [ ] `GET /obv/{symbol}` - On-Balance Volume
- [ ] `GET /willr/{symbol}` - Williams %R
- [ ] `GET /mfi/{symbol}` - Money Flow Index

### Phase 4: Proprietäre Indikatoren (Optional)

- [ ] `GET /strength/{symbol}` - Multi-Timeframe Strength
- [ ] `GET /levels/{symbol}` - Support/Resistance Levels

### Technische Anforderungen

- [ ] Response-Format: Alle numerischen Werte als Strings
- [ ] Datetime-Format: `YYYY-MM-DD HH:mm:ss`
- [ ] Timeframe-Aliase unterstützen
- [ ] Konsistente Fehler-Responses
- [ ] Rate Limiting implementieren
- [ ] API-Dokumentation (Swagger/OpenAPI)

---

## Anhang A: Vollständige Symbol-Liste

Die `/symbols` Endpoint sollte folgende Struktur haben:

```json
{
  "data": [
    {
      "symbol": "BTCUSD",
      "name": "Bitcoin US Dollar",
      "currency": "USD",
      "exchange": "Crypto",
      "type": "Digital Currency",
      "country": "",
      "available_timeframes": ["1min", "5min", "15min", "30min", "1h", "4h", "1day", "1week"]
    },
    {
      "symbol": "EURUSD",
      "name": "Euro US Dollar",
      "currency": "USD",
      "exchange": "Forex",
      "type": "Currency Pair",
      "country": "",
      "available_timeframes": ["1min", "5min", "15min", "30min", "1h", "4h", "1day", "1week"]
    }
  ],
  "count": 2
}
```

---

## Anhang B: Rate Limiting

Empfohlene Rate Limits:

| Plan | Requests/Minute | Requests/Tag |
|------|-----------------|--------------|
| Free | 60 | 1000 |
| Basic | 120 | 5000 |
| Pro | 300 | 50000 |

Response-Header bei Rate Limiting:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705315900
```

---

## Anhang C: Beispiel-Integration (Python)

```python
import httpx

class EasyInsightClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key

    async def get_time_series(
        self,
        symbol: str,
        interval: str = "1h",
        outputsize: int = 100
    ) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/time_series/{symbol}",
                params={
                    "interval": interval,
                    "outputsize": outputsize
                },
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            return response.json()

    async def get_indicators(
        self,
        symbol: str,
        indicators: list[str],
        interval: str = "1h",
        outputsize: int = 100
    ) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/indicators/{symbol}",
                params={
                    "interval": interval,
                    "indicators": ",".join(indicators),
                    "outputsize": outputsize
                }
            )
            return response.json()

# Verwendung
client = EasyInsightClient("http://easyinsight-api.local")
data = await client.get_time_series("BTCUSD", interval="1h", outputsize=500)
indicators = await client.get_indicators("BTCUSD", ["rsi", "macd", "bbands"])
```

---

*Dokument erstellt für KI Trading Model Integration*
*Version: 1.0*
*Datum: Januar 2025*
