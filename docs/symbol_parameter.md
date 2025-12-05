# EasyInsight Symbol-Parameter Dokumentation

#### Basis-Attribute

| Attribut | Datentyp | Nullable | Default | Beschreibung |
|----------|----------|----------|---------|--------------|
| `id` | BIGINT | NOT NULL | AUTO_INCREMENT | Primärschlüssel (Teil des zusammengesetzten Schlüssels) |
| `import_time` | TIMESTAMP WITH TIME ZONE | NOT NULL | now() | Zeitpunkt des Datenimports |
| `source_file` | VARCHAR(255) | NULL | - | Name der Quelldatei |
| `data_timestamp` | TIMESTAMP WITH TIME ZONE | NOT NULL | - | Zeitstempel der Marktdaten |
| `category` | VARCHAR(50) | NULL | - | Datenkategorie (z.B. "Forex", "Crypto") |
| `symbol` | VARCHAR(50) | NULL | - | Handelssymbol (z.B. "EURUSD", "BTCUSD") |

#### Preisdaten

| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `ask` | DOUBLE PRECISION | Ask-Preis (Verkaufskurs) |
| `bid` | DOUBLE PRECISION | Bid-Preis (Kaufkurs) |
| `spread` | DOUBLE PRECISION | Spread (Ask - Bid) |

#### OHLC-Daten (Open, High, Low, Close)

**D1 (Daily):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `d1_open` | DOUBLE PRECISION | Eröffnungskurs (täglich) |
| `d1_high` | DOUBLE PRECISION | Tageshoch |
| `d1_low` | DOUBLE PRECISION | Tagestief |
| `d1_close` | DOUBLE PRECISION | Schlusskurs (täglich) |

**H1 (Hourly):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `h1_open` | DOUBLE PRECISION | Eröffnungskurs (stündlich) |
| `h1_high` | DOUBLE PRECISION | Stundenhoch |
| `h1_low` | DOUBLE PRECISION | Stundentief |
| `h1_close` | DOUBLE PRECISION | Schlusskurs (stündlich) |

**M15 (15 Minuten):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `m15_open` | DOUBLE PRECISION | Eröffnungskurs (15 Min) |
| `m15_high` | DOUBLE PRECISION | 15-Min-Hoch |
| `m15_low` | DOUBLE PRECISION | 15-Min-Tief |
| `m15_close` | DOUBLE PRECISION | Schlusskurs (15 Min) |

#### Technische Indikatoren

**ADX (Average Directional Index):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `adx14_main_line` | DOUBLE PRECISION | ADX Hauptlinie (14 Perioden) |
| `adx14_minusdi_line` | DOUBLE PRECISION | -DI Linie |
| `adx14_plusdi_line` | DOUBLE PRECISION | +DI Linie |

**ATR (Average True Range):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `atr_d1` | DOUBLE PRECISION | ATR auf Tagesbasis |
| `range_d1` | DOUBLE PRECISION | Tagesrange |

**Bollinger Bands (200,200):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `bb200200price_close_base_line` | DOUBLE PRECISION | Mittellinie (SMA 200) |
| `bb200200price_close_lower_band` | DOUBLE PRECISION | Unteres Band |
| `bb200200price_close_upper_band` | DOUBLE PRECISION | Oberes Band |

**CCI (Commodity Channel Index):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `cci14price_typical` | DOUBLE PRECISION | CCI (14 Perioden, Typical Price) |

**Ichimoku Kinko Hyo (9,26,52):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `ichimoku92652_tenkansen_line` | DOUBLE PRECISION | Tenkan-sen (Conversion Line) |
| `ichimoku92652_kijunsen_line` | DOUBLE PRECISION | Kijun-sen (Base Line) |
| `ichimoku92652_senkouspana_line` | DOUBLE PRECISION | Senkou Span A (Leading Span A) |
| `ichimoku92652_senkouspanb_line` | DOUBLE PRECISION | Senkou Span B (Leading Span B) |
| `ichimoku92652_chikouspan_line` | DOUBLE PRECISION | Chikou Span (Lagging Span) |

**MACD (12,26,9):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `macd12269price_close_main_line` | DOUBLE PRECISION | MACD Hauptlinie |
| `macd12269price_close_signal_line` | DOUBLE PRECISION | Signal-Linie |

**Moving Average:**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `ma100mode_smaprice_close` | DOUBLE PRECISION | SMA 100 (Close) |

**RSI (Relative Strength Index):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `rsi14price_close` | DOUBLE PRECISION | RSI (14 Perioden, Close) |

**Stochastik (5,3,3):**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `sto533mode_smasto_lowhigh_main_line` | DOUBLE PRECISION | %K Linie |
| `sto533mode_smasto_lowhigh_signal_line` | DOUBLE PRECISION | %D Linie |

**Pivot Points:**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `r1_level_m5` | DOUBLE PRECISION | R1 Widerstand (M5) |
| `s1_level_m5` | DOUBLE PRECISION | S1 Unterstützung (M5) |

**Symbol Strength:**
| Attribut | Datentyp | Beschreibung |
|----------|----------|--------------|
| `strength_4h` | DOUBLE PRECISION | Stärke (4 Stunden) |
| `strength_1d` | DOUBLE PRECISION | Stärke (1 Tag) |
| `strength_1w` | DOUBLE PRECISION | Stärke (1 Woche) |
