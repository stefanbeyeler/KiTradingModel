# RAG Data Sources - Trading Intelligence

Dieses Modul stellt **9 externe Datenquellen** für den RAG (Retrieval Augmented Generation) Service bereit. Alle Quellen sind modular aufgebaut und können einzeln oder kombiniert abgefragt werden.

## Übersicht

| # | Quelle | Modul | Beschreibung |
|---|--------|-------|--------------|
| 1 | Economic Calendar | `economic_calendar.py` | Wirtschaftskalender & Events |
| 2 | On-Chain Data | `onchain_data.py` | Blockchain-Metriken |
| 3 | Sentiment | `sentiment_data.py` | Marktstimmung |
| 4 | Orderbook | `orderbook_data.py` | Orderbuch & Liquidität |
| 5 | Macro Correlation | `macro_correlation.py` | Makrodaten & Korrelationen |
| 6 | Historical Patterns | `historical_patterns.py` | Historische Muster |
| 7 | Technical Levels | `technical_levels.py` | Technische Preisniveaus |
| 8 | Regulatory | `regulatory_updates.py` | Regulatorische Updates |
| 9 | EasyInsight | `easyinsight_data.py` | Managed Symbols & MT5 Logs |

---

## 1. Economic Calendar (`economic_calendar.py`)

**Zweck:** Wirtschaftskalender-Events die Märkte bewegen können.

### Datentypen

| Event | Priorität | Beschreibung |
|-------|-----------|--------------|
| FOMC Decision | CRITICAL | Federal Reserve Zinsentscheidung |
| ECB Decision | CRITICAL | EZB Zinsentscheidung |
| BOJ Decision | HIGH | Bank of Japan Entscheidung |
| NFP | CRITICAL | Non-Farm Payrolls (US Arbeitsmarkt) |
| CPI | CRITICAL | Consumer Price Index (Inflation) |
| GDP | HIGH | Bruttoinlandsprodukt |
| PMI | MEDIUM | Einkaufsmanagerindex |
| Retail Sales | MEDIUM | Einzelhandelsumsätze |

### API Endpoint
```
GET /api/v1/rag/economic-calendar
  ?symbol=BTCUSD          # Optional: Symbolfilter
  &days_ahead=7           # Tage voraus (1-30)
  &days_back=1            # Tage zurück (0-7)
```

### Beispiel Response
```json
{
  "event": "FOMC Interest Rate Decision",
  "date": "2025-01-29",
  "impact": "HIGH",
  "previous": "4.50%",
  "forecast": "4.50%",
  "analysis": "Fed expected to hold rates steady..."
}
```

---

## 2. On-Chain Data (`onchain_data.py`)

**Zweck:** Blockchain-spezifische Metriken für Kryptowährungen.

### Datentypen

| Metrik | Priorität | Beschreibung |
|--------|-----------|--------------|
| Whale Alerts | HIGH | Große Transaktionen (>1000 BTC) |
| Exchange Inflow | HIGH | Zuflüsse zu Börsen (Verkaufsdruck) |
| Exchange Outflow | HIGH | Abflüsse von Börsen (Akkumulation) |
| Mining Hashrate | MEDIUM | Netzwerk-Sicherheit |
| Mining Difficulty | LOW | Mining-Schwierigkeit |
| DeFi TVL | MEDIUM | Total Value Locked in DeFi |
| MVRV Ratio | HIGH | Market Value / Realized Value |
| SOPR | MEDIUM | Spent Output Profit Ratio |

### API Endpoint
```
GET /api/v1/rag/onchain/{symbol}
  ?include_whale_alerts=true
  &include_exchange_flows=true
  &include_mining=true
  &include_defi=true
```

### Beispiel Response
```json
{
  "metric": "whale_alert",
  "symbol": "BTCUSD",
  "value": 2500,
  "unit": "BTC",
  "direction": "exchange_inflow",
  "interpretation": "Large deposit to exchange - potential selling pressure"
}
```

---

## 3. Sentiment Data (`sentiment_data.py`)

**Zweck:** Marktstimmung aus verschiedenen Quellen.

### Datentypen

| Indikator | Priorität | Beschreibung |
|-----------|-----------|--------------|
| Fear & Greed Index | HIGH | Crypto-Sentiment (0-100) |
| Social Sentiment | MEDIUM | Twitter/Reddit Stimmung |
| Put/Call Ratio | HIGH | Options-Sentiment |
| VIX | HIGH | Volatilitätsindex |
| Funding Rates | HIGH | Perpetual Futures Funding |
| Long/Short Ratio | MEDIUM | Positionsverhältnis |

### API Endpoint
```
GET /api/v1/rag/sentiment
  ?symbol=BTCUSD
  &include_fear_greed=true
  &include_social=true
  &include_options=true
  &include_volatility=true
```

### Fear & Greed Interpretation
| Wert | Bedeutung | Trading Signal |
|------|-----------|----------------|
| 0-25 | Extreme Fear | Potentieller Boden |
| 25-45 | Fear | Vorsicht |
| 45-55 | Neutral | Abwarten |
| 55-75 | Greed | Trend stark |
| 75-100 | Extreme Greed | Potentielles Top |

---

## 4. Orderbook Data (`orderbook_data.py`)

**Zweck:** Orderbuch-Analyse und Liquiditätsdaten.

### Datentypen

| Metrik | Priorität | Beschreibung |
|--------|-----------|--------------|
| Bid Walls | HIGH | Große Kauforders (Support) |
| Ask Walls | HIGH | Große Verkaufsorders (Resistance) |
| Liquidation Levels | CRITICAL | Liquidationspreisniveaus |
| Open Interest | HIGH | Offene Positionen |
| CVD | MEDIUM | Cumulative Volume Delta |
| Order Flow | MEDIUM | Kauf/Verkauf-Imbalance |

### API Endpoint
```
GET /api/v1/rag/orderbook/{symbol}
  ?depth=50               # Orderbuch-Tiefe (10-200)
  &include_liquidations=true
  &include_cvd=true
```

### Beispiel Response
```json
{
  "type": "bid_wall",
  "price": 42000,
  "size": 500,
  "size_usd": 21000000,
  "interpretation": "Strong support at $42,000 with 500 BTC bid wall"
}
```

---

## 5. Macro Correlation (`macro_correlation.py`)

**Zweck:** Makroökonomische Daten und Cross-Asset-Korrelationen.

### Datentypen

| Metrik | Priorität | Beschreibung |
|--------|-----------|--------------|
| DXY | HIGH | US Dollar Index |
| US 2Y Yield | HIGH | 2-Jahres Treasury |
| US 10Y Yield | HIGH | 10-Jahres Treasury |
| Yield Curve | HIGH | 10Y-2Y Spread |
| BTC-SPX Correlation | MEDIUM | Bitcoin/S&P500 Korrelation |
| BTC-Gold Correlation | MEDIUM | Bitcoin/Gold Korrelation |
| Sector Rotation | LOW | Sektor-Performance |
| Global Liquidity | MEDIUM | M2 Geldmenge |

### API Endpoint
```
GET /api/v1/rag/macro
  ?symbol=BTCUSD
  &include_dxy=true
  &include_bonds=true
  &include_correlations=true
  &include_sectors=true
```

### Korrelations-Interpretation
| Korrelation | Bedeutung |
|-------------|-----------|
| > 0.7 | Stark positiv - bewegen sich zusammen |
| 0.3 - 0.7 | Moderat positiv |
| -0.3 - 0.3 | Keine signifikante Korrelation |
| -0.7 - -0.3 | Moderat negativ |
| < -0.7 | Stark negativ - bewegen sich gegenläufig |

---

## 6. Historical Patterns (`historical_patterns.py`)

**Zweck:** Historische Muster und Saisonalität.

### Datentypen

| Pattern | Priorität | Beschreibung |
|---------|-----------|--------------|
| Monthly Seasonality | MEDIUM | Durchschnittliche Monatsperformance |
| Weekly Seasonality | LOW | Wochentags-Performance |
| Halving Cycles | HIGH | Bitcoin Halving Auswirkungen |
| Post-Halving Returns | HIGH | Returns nach Halving |
| Historical Drawdowns | MEDIUM | Größte Korrekturen |
| Comparable Periods | MEDIUM | Ähnliche Marktphasen |

### API Endpoint
```
GET /api/v1/rag/historical-patterns
  ?symbol=BTCUSD
  &include_seasonality=true
  &include_drawdowns=true
  &include_events=true
  &include_comparable=true
```

### Bitcoin Saisonalität (Historisch)
| Monat | Durchschnitt | Interpretation |
|-------|--------------|----------------|
| Januar | +7.2% | Starker Start |
| Februar | +3.1% | Moderat positiv |
| März | -2.4% | Schwächer |
| April | +12.8% | Sehr stark |
| Oktober | +18.5% | "Uptober" |
| November | +15.2% | Historisch stark |
| Dezember | +5.8% | Positiv |

---

## 7. Technical Levels (`technical_levels.py`)

**Zweck:** Technische Preisniveaus und Indikatoren.

### Datentypen

| Level | Priorität | Beschreibung |
|-------|-----------|--------------|
| Support Levels | HIGH | Unterstützungszonen |
| Resistance Levels | HIGH | Widerstandszonen |
| Fibonacci Retracement | MEDIUM | 23.6%, 38.2%, 50%, 61.8%, 78.6% |
| Fibonacci Extension | MEDIUM | 127.2%, 161.8%, 261.8% |
| Daily Pivot | MEDIUM | Tägliche Pivot-Punkte |
| Weekly Pivot | HIGH | Wöchentliche Pivot-Punkte |
| VWAP | HIGH | Volume Weighted Average Price |
| SMA 50/200 | HIGH | Simple Moving Averages |
| EMA 21 | MEDIUM | Exponential Moving Average |
| Volume POC | HIGH | Point of Control |

### API Endpoint
```
GET /api/v1/rag/technical-levels/{symbol}
  ?include_sr=true        # Support/Resistance
  &include_fib=true       # Fibonacci
  &include_pivots=true    # Pivot Points
  &include_vwap=true      # VWAP
  &include_ma=true        # Moving Averages
```

### Beispiel Response
```json
{
  "type": "support",
  "level": "major",
  "price": 42000,
  "touches": 3,
  "interpretation": "Major support at $42,000 - tested 3 times"
}
```

---

## 8. Regulatory Updates (`regulatory_updates.py`)

**Zweck:** Regulatorische Nachrichten und Updates.

### Datentypen

| Kategorie | Priorität | Beschreibung |
|-----------|-----------|--------------|
| SEC Filings | HIGH | SEC Einreichungen |
| ETF Approvals | CRITICAL | ETF Genehmigungen |
| ETF Flows | HIGH | ETF Zu-/Abflüsse |
| MiCA | MEDIUM | EU Regulierung |
| Enforcement | HIGH | Durchsetzungsmaßnahmen |
| Stablecoin Regulation | MEDIUM | Stablecoin Regeln |

### API Endpoint
```
GET /api/v1/rag/regulatory
  ?symbol=BTCUSD
  &include_sec=true
  &include_etf=true
  &include_global=true
  &include_enforcement=true
```

### Beispiel Response
```json
{
  "type": "etf_flow",
  "etf": "IBIT",
  "flow_usd": 500000000,
  "direction": "inflow",
  "interpretation": "Strong institutional demand - $500M inflow to BlackRock ETF"
}
```

---

## 9. EasyInsight (`easyinsight_data.py`)

**Zweck:** Interne Symbol-Verwaltung und MT5 Trading-Logs.

### Datentypen

| Metrik | Priorität | Beschreibung |
|--------|-----------|--------------|
| Managed Symbols | HIGH | Verwaltete Trading-Symbole |
| Symbol Stats | MEDIUM | Statistiken zu Symbolen |
| MT5 Logs | MEDIUM | MetaTrader 5 Trading-Logs |
| Model Status | HIGH | NHITS Modell-Status |
| Data Availability | MEDIUM | TimescaleDB Datenverfügbarkeit |

### API Endpoint
```
GET /api/v1/rag/easyinsight
  ?symbol=BTCUSD              # Optional: Spezifisches Symbol
  &include_symbols=true       # Managed Symbols einbeziehen
  &include_stats=true         # Statistiken einbeziehen
  &include_mt5_logs=true      # MT5 Logs einbeziehen
```

### Beispiel Response
```json
{
  "source_type": "easyinsight",
  "content": "Managed Symbol: BTCUSD\nCategory: crypto\nStatus: active\nTimescaleDB Data: Available\nData Range: 2024-01-01 to 2025-12-15\nTotal Records: 8,760\nNHITS Model: Trained",
  "priority": "high",
  "metadata": {
    "metric_type": "managed_symbol",
    "category": "crypto",
    "status": "active",
    "has_timescaledb_data": true,
    "has_nhits_model": true,
    "total_records": 8760
  }
}
```

### Symbol Kategorien
| Kategorie | Beispiele |
|-----------|-----------|
| CRYPTO | BTCUSD, ETHUSD, SOLUSD |
| FOREX | EURUSD, GBPUSD, USDJPY |
| INDEX | US30, US500, GER40 |
| COMMODITY | XAUUSD, XAGUSD, OIL |
| OTHER | Sonstige |

---

## Architektur

### Klassenstruktur

```
DataSourceBase (Abstract)
├── EconomicCalendarSource
├── OnChainDataSource
├── SentimentDataSource
├── OrderbookDataSource
├── MacroCorrelationSource
├── HistoricalPatternsSource
├── TechnicalLevelsSource
├── RegulatoryUpdatesSource
└── EasyInsightDataSource
```

### DataSourceResult

Jede Quelle liefert `DataSourceResult` Objekte:

```python
@dataclass
class DataSourceResult:
    source_type: DataSourceType    # Enum der Quelle
    content: str                   # Textinhalt für RAG
    priority: DataPriority         # CRITICAL, HIGH, MEDIUM, LOW
    metadata: dict                 # Zusätzliche Daten
    timestamp: datetime            # Zeitstempel
    symbol: Optional[str]          # Trading Symbol
```

### Prioritäten

| Priorität | Bedeutung | Beispiele |
|-----------|-----------|-----------|
| CRITICAL | Sofortige Marktauswirkung | FOMC, Liquidationen, ETF Approval |
| HIGH | Starker Einfluss | CPI, Whale Alerts, Major S/R |
| MEDIUM | Moderater Einfluss | PMI, Social Sentiment |
| LOW | Hintergrund-Kontext | Minor News, Weekly Seasonality |

---

## Nutzung

### Alle Quellen abrufen

```python
from src.services.rag_data_sources import get_data_fetcher_service

fetcher = get_data_fetcher_service()
results = await fetcher.fetch_all(symbol="BTCUSD")
```

### Einzelne Quelle abrufen

```python
results = await fetcher.fetch_sentiment(
    symbol="BTCUSD",
    include_fear_greed=True,
    include_social=True
)
```

### Trading Context (alle relevanten Daten)

```python
context = await fetcher.fetch_trading_context(
    symbol="BTCUSD",
    include_types=["sentiment", "levels", "orderbook"]
)
```

### In RAG-Datenbank laden

```bash
# Via API
curl -X POST "http://localhost:3004/api/v1/rag/ingest-all-sources?symbol=BTCUSD"

# Oder alle Symbole
curl -X POST "http://localhost:3004/api/v1/rag/ingest-all-sources"
```

---

## Caching

Alle Quellen implementieren TTL-basiertes Caching:

| Quelle | Cache TTL | Begründung |
|--------|-----------|------------|
| Economic Calendar | 1 Stunde | Events ändern sich selten |
| On-Chain | 5 Minuten | Blockchain-Updates |
| Sentiment | 15 Minuten | Moderate Änderungsrate |
| Orderbook | 1 Minute | Hochfrequente Änderungen |
| Macro | 1 Stunde | Makrodaten ändern sich langsam |
| Historical | 24 Stunden | Historische Daten stabil |
| Technical | 5 Minuten | Preis-Updates |
| Regulatory | 1 Stunde | News-Zyklus |
| EasyInsight | 5 Minuten | Symbol-Updates |

---

## API Endpoints Übersicht

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/v1/rag/sources` | GET | Liste aller Quellen |
| `/api/v1/rag/fetch` | POST | Daten abrufen (optional speichern) |
| `/api/v1/rag/trading-context` | POST | Kompletter Trading-Kontext |
| `/api/v1/rag/economic-calendar` | GET | Wirtschaftskalender |
| `/api/v1/rag/onchain/{symbol}` | GET | On-Chain Daten |
| `/api/v1/rag/sentiment` | GET | Sentiment-Daten |
| `/api/v1/rag/orderbook/{symbol}` | GET | Orderbuch-Daten |
| `/api/v1/rag/macro` | GET | Makro-Daten |
| `/api/v1/rag/historical-patterns` | GET | Historische Muster |
| `/api/v1/rag/technical-levels/{symbol}` | GET | Technische Levels |
| `/api/v1/rag/regulatory` | GET | Regulatorische Updates |
| `/api/v1/rag/easyinsight` | GET | EasyInsight Daten |
| `/api/v1/rag/ingest-all-sources` | POST | Alle Quellen in RAG laden |

---

## Erweiterung

Um eine neue Datenquelle hinzuzufügen:

1. Erstelle neue Klasse in `rag_data_sources/`:
```python
from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority

class NewDataSource(DataSourceBase):
    def __init__(self):
        super().__init__(
            source_type=DataSourceType.NEW_TYPE,
            cache_ttl=300  # 5 Minuten
        )

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        # Implementierung
        pass
```

2. Füge `DataSourceType` in `base.py` hinzu
3. Registriere in `data_fetcher_service.py`
4. Exportiere in `__init__.py`
5. Füge API Endpoint in `main.py` hinzu
