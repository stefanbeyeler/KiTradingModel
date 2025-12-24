# Swagger API Kategorisierung

## Microservices Ports

Das System verwendet eine Microservices-Architektur mit separaten API-Services:

| Service | Port | Swagger UI | Beschreibung |
|---------|------|------------|--------------|
| Frontend Dashboard | 3000 | - | Nginx API Gateway |
| Data Service | 3001 | `/data/docs` | Symbole, Strategien, Daten |
| NHITS Service | 3002 | `/nhits/docs` | Prognosen, Training |
| TCN-Pattern Service | 3003 | `/tcn/docs` | Chart-Pattern-Erkennung |
| HMM-Regime Service | 3004 | `/hmm/docs` | Marktphasen-Erkennung |
| Embedder Service | 3005 | `/embedder/docs` | Embeddings |
| RAG Service | 3008 | `/rag/docs` | Vector Search |
| LLM Service | 3009 | `/llm/docs` | Trading-Analyse |
| Watchdog Service | 3010 | `/watchdog/docs` | Monitoring |

## Zugriff auf Swagger UI

Alle Swagger UIs sind über das Frontend (Port 3000) erreichbar:

```text
http://10.1.19.101:3000/{service}/docs
```

Beispiele:

- Data Service: <http://10.1.19.101:3000/data/docs>
- NHITS Service: <http://10.1.19.101:3000/nhits/docs>
- LLM Service: <http://10.1.19.101:3000/llm/docs>

---

## Data Service (Port 3001)

### Symbol Management

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/symbols` | Alle verwalteten Symbole |
| GET | `/api/v1/symbols/stats` | Symbol-Statistiken |
| GET | `/api/v1/symbols/search` | Symbole durchsuchen |
| POST | `/api/v1/symbols` | Neues Symbol erstellen |
| GET | `/api/v1/symbols/{id}` | Symbol-Details abrufen |
| PUT | `/api/v1/symbols/{id}` | Symbol aktualisieren |
| DELETE | `/api/v1/symbols/{id}` | Symbol löschen |
| POST | `/api/v1/symbols/{id}/favorite` | Favoriten-Status umschalten |

### Trading Strategies

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/strategies` | Alle Strategien auflisten |
| GET | `/api/v1/strategies/default` | Standard-Strategie abrufen |
| GET | `/api/v1/strategies/{id}` | Spezifische Strategie |
| GET | `/api/v1/strategies/{id}/export` | Strategie als Markdown exportieren |
| POST | `/api/v1/strategies` | Neue Strategie erstellen |
| PUT | `/api/v1/strategies/{id}` | Strategie aktualisieren |
| DELETE | `/api/v1/strategies/{id}` | Strategie löschen |
| POST | `/api/v1/strategies/import` | Strategie aus Markdown importieren |

### External Data Sources

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/external-sources/economic-calendar` | Wirtschaftskalender |
| GET | `/api/v1/external-sources/sentiment` | Marktsentiment |
| GET | `/api/v1/external-sources/onchain/{symbol}` | On-Chain Daten |
| GET | `/api/v1/external-sources/orderbook/{symbol}` | Orderbuch-Daten |
| GET | `/api/v1/external-sources/macro` | Makro-Indikatoren |
| POST | `/api/v1/external-sources/fetch-all` | Alle Quellen abrufen |

---

## NHITS Service (Port 3002)

### Forecast

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/forecast/status` | NHITS Service-Status |
| GET | `/api/v1/forecast/models` | Liste aller trainierten Modelle |
| GET | `/api/v1/forecast/{symbol}` | Preisvorhersage generieren |
| GET | `/api/v1/forecast/{symbol}/model` | Modellinformationen |

### Training

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/forecast/train-all` | Batch-Training starten |
| GET | `/api/v1/forecast/training/status` | Training-Status |
| GET | `/api/v1/forecast/training/progress` | Echtzeit-Fortschritt |
| GET | `/api/v1/forecast/training/symbols` | Verfügbare Symbole |
| POST | `/api/v1/forecast/training/cancel` | Training abbrechen |
| POST | `/api/v1/forecast/{symbol}/train` | Einzelsymbol trainieren |

### Performance & Evaluation

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/forecast/performance` | Performance-Metriken |
| GET | `/api/v1/forecast/evaluated` | Evaluierte Vorhersagen |
| POST | `/api/v1/forecast/evaluate` | Predictions evaluieren |
| GET | `/api/v1/forecast/retraining-needed` | Retraining-Bedarf prüfen |

### System

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/system/metrics` | CPU/GPU/Memory Metriken |
| GET | `/api/v1/system/storage` | Storage-Informationen |

---

## TCN-Pattern Service (Port 3003)

### Pattern Detection

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/patterns/scan` | Pattern-Scan durchführen |
| GET | `/api/v1/patterns/types` | Verfügbare Pattern-Typen |
| GET | `/api/v1/patterns/{symbol}` | Aktive Patterns für Symbol |
| GET | `/api/v1/patterns/history/{symbol}` | Pattern-Historie |

### Training

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/training/start` | Training starten |
| GET | `/api/v1/training/status` | Training-Status |
| GET | `/api/v1/training/models` | Trainierte Modelle |

---

## HMM-Regime Service (Port 3004)

### Regime Detection

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/regime/{symbol}` | Aktuelles Marktregime |
| GET | `/api/v1/regime/{symbol}/history` | Regime-Historie |
| GET | `/api/v1/regime/{symbol}/transitions` | Transition-Matrix |

### Signal Scoring

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/score` | Signal-Score berechnen |
| GET | `/api/v1/score/history/{symbol}` | Score-Historie |

---

## Embedder Service (Port 3005)

### Embeddings

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/embed/text` | Text-Embedding erstellen |
| POST | `/api/v1/embed/finbert` | FinBERT Sentiment-Embedding |
| POST | `/api/v1/embed/timeseries` | TimeSeries-Embedding |
| POST | `/api/v1/embed/batch` | Batch-Embedding |

### Models

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/models` | Verfügbare Modelle |
| GET | `/api/v1/models/status` | Modell-Status |

---

## RAG Service (Port 3008)

### Documents

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/rag/document` | Dokument hinzufügen |
| GET | `/api/v1/rag/documents` | Alle Dokumente |
| DELETE | `/api/v1/rag/document/{id}` | Dokument entfernen |

### Query

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/rag/query` | Semantische Suche |
| GET | `/api/v1/rag/stats` | Index-Statistiken |
| POST | `/api/v1/rag/rebuild` | Index neu aufbauen |

---

## LLM Service (Port 3009)

### Analysis

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/analyze` | Trading-Analyse mit LLM |
| POST | `/api/v1/chat` | Chat-Konversation |
| GET | `/api/v1/recommendation/{symbol}` | Handelsempfehlung |

### Model Management

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/llm/status` | LLM-Status |
| POST | `/api/v1/llm/pull` | Modell herunterladen |

---

## Watchdog Service (Port 3010)

### Status

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/status` | Status aller Services |
| GET | `/api/v1/services` | Überwachte Services |
| POST | `/api/v1/check` | Manueller Health-Check |

### Alerts

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/alerts/history` | Alert-Historie |
| POST | `/api/v1/alerts/test` | Test-Alert senden |

### Config

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/config` | Aktuelle Konfiguration |

---

## Health Checks

Alle Services bieten einen Health-Check Endpoint:

| Service | Health Endpoint |
|---------|-----------------|
| Data | `/data/health` |
| NHITS | `/nhits/health` |
| TCN | `/tcn/health` |
| HMM | `/hmm/health` |
| Embedder | `/embedder/health` |
| RAG | `/rag/health` |
| LLM | `/llm/health` |
| Watchdog | `/watchdog/health` |

Beispiel Response:

```json
{
  "status": "healthy",
  "service": "nhits-service",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

---

## Best Practices

### 1. NHITS Forecasting-Workflow

```bash
# 1. Prüfen ob Modell existiert
curl http://10.1.19.101:3000/nhits/api/v1/forecast/EURUSD/model

# 2. Falls nicht: Modell trainieren
curl -X POST "http://10.1.19.101:3000/nhits/api/v1/forecast/EURUSD/train"

# 3. Vorhersage generieren
curl "http://10.1.19.101:3000/nhits/api/v1/forecast/EURUSD?horizon=24"
```

### 2. LLM Analyse mit RAG

```bash
# 1. Trading-Analyse anfragen
curl -X POST "http://10.1.19.101:3000/llm/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD", "question": "Sollte ich kaufen?"}'
```

### 3. Pattern-Scan

```bash
# Multi-Timeframe Pattern-Scan
curl -X POST "http://10.1.19.101:3000/tcn/api/v1/patterns/scan" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD", "timeframes": ["M15", "H1"], "min_confidence": 0.7}'
```

### 4. Service-Monitoring

```bash
# Status aller Services
curl "http://10.1.19.101:3000/watchdog/api/v1/status"

# Test-Alert senden
curl -X POST "http://10.1.19.101:3000/watchdog/api/v1/alerts/test"
```

---

## OpenAPI Spezifikationen

Jeder Service bietet OpenAPI JSON:

```text
http://10.1.19.101:3000/{service}/openapi.json
```

Diese können für automatische Client-Generierung verwendet werden.

---

## Support

- **Dashboard:** <http://10.1.19.101:3000>
- **Dokumentation:** `/docs/` Verzeichnis
- **Architektur:** [MICROSERVICES_ARCHITECTURE.md](./MICROSERVICES_ARCHITECTURE.md)
