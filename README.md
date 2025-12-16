# KI Trading Model

Lokaler KI-Service für die Generierung von Handelsempfehlungen basierend auf Zeitreihendaten mit neuronaler Preisvorhersage.

## Features

- **Llama 3.1 Integration**: Komplexe Marktanalysen und Reasoning über Ollama
- **NHITS Neural Forecasting**: Neuronale Zeitreihenvorhersage mit Multi-Timeframe Support (M15, H1, D1)
- **RAG System**: Historische Trading-Daten mit FAISS für kontextbezogene Empfehlungen
- **Technische Analyse**: Automatische Berechnung von Indikatoren (RSI, MACD, Bollinger Bands, etc.)
- **EasyInsight API Integration**: REST API Anbindung für Zeitreihendaten
- **Symbol Management**: Verwaltung von Trading-Symbolen mit Kategorien und Favoriten
- **Trading Strategien**: Konfigurierbare Strategien mit Import/Export (Markdown)
- **Automatischer RAG-Sync**: Kontinuierliche Synchronisation von Marktdaten in das RAG-System
- **GPU-Beschleunigung**: CUDA-Support für Embeddings und NHITS Training
- **Event-Based Training**: Automatisches Modell-Retraining basierend auf Marktereignissen

## Voraussetzungen

- Python 3.11+ (Python 3.13 unterstützt)
- Ollama mit Llama 3.1 (8B oder 70B)
- EasyInsight API Zugang
- NVIDIA GPU mit CUDA (optional, für beschleunigte Embeddings und NHITS)

## Installation

```bash
# Repository klonen
cd KITradingModel

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt

# Konfiguration anpassen
cp .env.example .env
# .env Datei bearbeiten
```

### GPU-Setup (optional, empfohlen)

Für GPU-beschleunigte Embeddings und NHITS:

```bash
# GPU Setup Script ausführen (Windows)
setup_gpu.bat

# Oder manuell:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Nach der Installation werden Embeddings und NHITS automatisch auf der GPU ausgeführt.

## Konfiguration

Erstelle eine `.env` Datei basierend auf `.env.example`:

```env
# EasyInsight API
EASYINSIGHT_API_URL=http://10.1.19.102:3000/api

# RAG Sync
RAG_SYNC_ENABLED=true
RAG_SYNC_INTERVAL_SECONDS=3600

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Ollama Performance
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_GPU=-1
OLLAMA_NUM_THREAD=16

# FAISS
FAISS_PERSIST_DIRECTORY=./data/faiss
FAISS_USE_GPU=true

# Embeddings (GPU-beschleunigt)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=64
USE_HALF_PRECISION=true

# NHITS Neural Forecasting
NHITS_ENABLED=true
NHITS_HORIZON=24
NHITS_INPUT_SIZE=168
NHITS_USE_GPU=true
NHITS_AUTO_RETRAIN_DAYS=7
```

## Verwendung

### Service starten

```bash
# Lokal
python run.py

# Oder mit uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 3011
```

### API Endpoints

Alle Endpoints sind unter `/api/v1/` verfügbar.

---

#### System & Monitoring

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/health` | Service-Gesundheitsstatus (inkl. NHITS) |
| GET | `/version` | Versionsinformationen |
| GET | `/system/info` | GPU, PyTorch & Konfigurationsdetails |
| GET | `/system/metrics` | CPU/GPU Auslastung in Echtzeit |

```bash
# Health Check (inkl. NHITS Status)
curl "http://localhost:3011/api/v1/health"

# Version Info
curl "http://localhost:3011/api/v1/version"

# System Metrics (für Dashboards)
curl "http://localhost:3011/api/v1/system/metrics"
```

---

#### Trading Analysis & Empfehlungen

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| POST | `/analyze` | Vollständige Marktanalyse mit LLM |
| GET | `/recommendation/{symbol}` | Schnelle Trading-Empfehlung (regelbasiert) |
| GET | `/recommendation/{symbol}?use_llm=true` | Trading-Empfehlung mit LLM-Analyse |
| GET | `/symbols` | Verfügbare Symbole |
| GET | `/symbol-info/{symbol}` | Detaillierte Symbol-Informationen |

```bash
# Vollständige Analyse (langsam, ~30-60s, mit LLM)
curl -X POST "http://localhost:3011/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "lookback_days": 30,
    "include_technical": true
  }'

# Schnelle Empfehlung (regelbasiert, ~100ms)
curl "http://localhost:3011/api/v1/recommendation/BTC-USD"

# Empfehlung mit LLM und spezifischer Strategie
curl "http://localhost:3011/api/v1/recommendation/BTC-USD?use_llm=true&strategy_id=conservative"
```

---

#### NHITS Forecasting (Predictions)

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/forecast/{symbol}` | Preisvorhersage generieren |
| GET | `/forecast/{symbol}?timeframe=M15` | 15-Minuten Vorhersage (2h Horizont) |
| GET | `/forecast/{symbol}?timeframe=D1` | Tägliche Vorhersage (7 Tage Horizont) |
| GET | `/forecast/status` | NHITS Service Status |
| GET | `/forecast/models` | Liste aller trainierten Modelle |
| GET | `/forecast/{symbol}/model` | Modell-Info für Symbol |

**Timeframe-Konfigurationen:**
- **M15**: 15-Minuten Kerzen, 8-Schritt Vorhersage (2 Stunden)
- **H1** (Standard): Stündliche Kerzen, 24-Schritt Vorhersage (24 Stunden)
- **D1**: Tägliche Kerzen, 7-Schritt Vorhersage (7 Tage)

```bash
# Stündliche Vorhersage (Standard, 24h)
curl "http://localhost:3011/api/v1/forecast/EURUSD"

# 15-Minuten Vorhersage (2h Horizont)
curl "http://localhost:3011/api/v1/forecast/EURUSD?timeframe=M15"

# Tägliche Vorhersage (7 Tage)
curl "http://localhost:3011/api/v1/forecast/BTCUSD?timeframe=D1"

# Alle trainierten Modelle anzeigen
curl "http://localhost:3011/api/v1/forecast/models"
```

---

#### NHITS Training

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| POST | `/forecast/{symbol}/train` | Einzelnes Modell trainieren |
| POST | `/forecast/train-all` | Alle Modelle trainieren (Batch) |
| GET | `/forecast/training/status` | Training-Service Status |
| GET | `/forecast/training/progress` | Echtzeit-Fortschritt |
| GET | `/forecast/training/symbols` | Trainierbare Symbole |
| POST | `/forecast/training/cancel` | Laufendes Training abbrechen |

```bash
# Modell für Symbol trainieren
curl -X POST "http://localhost:3011/api/v1/forecast/EURUSD/train"

# Alle Modelle im Hintergrund trainieren
curl -X POST "http://localhost:3011/api/v1/forecast/train-all?background=true"

# Training-Fortschritt abrufen
curl "http://localhost:3011/api/v1/forecast/training/progress"

# Training abbrechen
curl -X POST "http://localhost:3011/api/v1/forecast/training/cancel"
```

**Model Performance & Improvement:**

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/forecast/performance` | Performance-Metriken aller Modelle |
| GET | `/forecast/evaluated` | Evaluierte Vorhersagen |
| GET | `/forecast/retraining-needed` | Modelle die Retraining benötigen |
| POST | `/forecast/retrain-poor-performers` | Schlecht performende Modelle neu trainieren |

```bash
# Model Performance anzeigen
curl "http://localhost:3011/api/v1/forecast/performance"

# Modelle mit schlechter Performance retrainieren
curl -X POST "http://localhost:3011/api/v1/forecast/retrain-poor-performers"
```

**Event-Based Training:**

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/forecast/training/events/status` | Event-Monitor Status |
| GET | `/forecast/training/events/summary` | Ereignis-Zusammenfassung |
| POST | `/forecast/training/events/start` | Event-Monitor starten |
| POST | `/forecast/training/events/stop` | Event-Monitor stoppen |

---

#### Symbol Management

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/managed-symbols` | Alle verwalteten Symbole |
| GET | `/managed-symbols/stats` | Symbol-Statistiken |
| GET | `/managed-symbols/search` | Symbole suchen |
| POST | `/managed-symbols` | Symbol hinzufügen |
| POST | `/managed-symbols/import` | Symbole aus TimescaleDB importieren |
| GET | `/managed-symbols/{id}` | Symbol abrufen |
| PUT | `/managed-symbols/{id}` | Symbol aktualisieren |
| DELETE | `/managed-symbols/{id}` | Symbol löschen |
| POST | `/managed-symbols/{id}/favorite` | Favorit umschalten |
| POST | `/managed-symbols/{id}/refresh` | Daten aktualisieren |

```bash
# Alle Symbole mit Filter
curl "http://localhost:3011/api/v1/managed-symbols?category=crypto&favorites_only=true"

# Symbole aus TimescaleDB importieren
curl -X POST "http://localhost:3011/api/v1/managed-symbols/import"

# Symbol als Favorit markieren
curl -X POST "http://localhost:3011/api/v1/managed-symbols/BTCUSD/favorite"
```

---

#### Trading Strategien

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/strategies` | Alle Strategien |
| GET | `/strategies/default` | Standard-Strategie |
| GET | `/strategies/{id}` | Strategie abrufen |
| GET | `/strategies/{id}/export` | Strategie als Markdown exportieren |
| POST | `/strategies` | Strategie erstellen |
| POST | `/strategies/import` | Strategie aus Markdown importieren |
| PUT | `/strategies/{id}` | Strategie aktualisieren |
| DELETE | `/strategies/{id}` | Strategie löschen |
| POST | `/strategies/{id}/set-default` | Als Standard setzen |

```bash
# Alle Strategien abrufen
curl "http://localhost:3011/api/v1/strategies"

# Strategie als Markdown exportieren
curl "http://localhost:3011/api/v1/strategies/conservative/export" -o strategy.md

# Strategie importieren
curl -X POST "http://localhost:3011/api/v1/strategies/import" \
  -F "file=@strategy.md"
```

---

#### RAG & Knowledge Base

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/rag/stats` | RAG-Statistiken |
| GET | `/rag/query` | Relevante Dokumente abfragen |
| POST | `/rag/document` | Dokument hinzufügen |
| POST | `/rag/persist` | RAG-Index speichern |
| DELETE | `/rag/documents` | Dokumente löschen |

```bash
# RAG Statistiken
curl "http://localhost:3011/api/v1/rag/stats"

# RAG abfragen
curl "http://localhost:3011/api/v1/rag/query?query=RSI%20oversold&symbol=BTC/USD"
```

---

#### LLM Service

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/llm/status` | LLM-Modell Status |
| POST | `/llm/pull` | LLM-Modell herunterladen |

---

#### TimescaleDB Sync

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/sync/status` | Sync-Service Status |
| POST | `/sync/start` | Sync-Service starten |
| POST | `/sync/stop` | Sync-Service stoppen |
| POST | `/sync/manual` | Manuellen Sync auslösen |

---

#### Query Logs & Analytics

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/query-logs` | Query-Historie |
| GET | `/query-logs/stats` | Query-Statistiken |
| GET | `/query-logs/{id}` | Einzelner Log-Eintrag |
| DELETE | `/query-logs` | Logs löschen |

---

## Architektur

```
KITradingModel/
├── src/
│   ├── api/
│   │   └── routes.py             # FastAPI Routes (thematisch gruppiert)
│   ├── models/
│   │   ├── trading_data.py       # Trading & Strategy Models
│   │   ├── symbol_data.py        # Symbol Management Models
│   │   └── forecast_data.py      # NHITS Forecast Models
│   ├── services/
│   │   ├── analysis_service.py           # Hauptanalyse-Pipeline
│   │   ├── llm_service.py                # Llama Integration
│   │   ├── rag_service.py                # FAISS RAG System
│   │   ├── forecast_service.py           # NHITS Forecasting
│   │   ├── nhits_training_service.py     # NHITS Training
│   │   ├── model_improvement_service.py  # Performance Tracking
│   │   ├── event_based_training_service.py # Event-Based Retraining
│   │   ├── strategy_service.py           # Trading Strategien
│   │   ├── symbol_service.py             # Symbol Management
│   │   ├── query_log_service.py          # Query Analytics
│   │   └── timescaledb_sync_service.py   # EasyInsight Sync
│   ├── config/
│   │   └── settings.py           # Pydantic Settings
│   ├── version.py                # Git-basierte Versionierung
│   └── main.py                   # FastAPI App
├── static/                       # Dashboard UI
├── data/
│   ├── faiss/                    # RAG Datenbank
│   └── models/nhits/             # Trainierte NHITS Modelle
├── logs/                         # Log-Dateien
└── requirements.txt
```

## API Dokumentation

Nach dem Start verfügbar unter:
- Swagger UI: http://localhost:3011/docs
- ReDoc: http://localhost:3011/redoc

## Llama Setup

### Mit Ollama

```bash
# Ollama installieren
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: https://ollama.ai/download/windows

# Modell laden (8B für weniger VRAM)
ollama pull llama3.1:8b

# Oder größere Variante für bessere Qualität
ollama pull llama3.1:70b
```

### Hardware-Anforderungen

- **70B Modell**: ~48GB VRAM (oder CPU mit ~140GB RAM)
- **8B Modell**: ~8GB VRAM
- **NHITS Training**: ~2-4GB VRAM pro Modell

## Entwicklung

```bash
# Tests ausführen
pytest tests/

# Code formatieren
black src/
isort src/
```

## Lizenz

MIT License
