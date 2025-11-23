# KI Trading Model

Lokaler KI-Service für die Generierung von Handelsempfehlungen basierend auf Zeitreihendaten.

## Features

- **Llama 3.1 70B Integration**: Komplexe Marktanalysen und Reasoning über Ollama
- **RAG System**: Historische Trading-Daten mit FAISS für kontextbezogene Empfehlungen
- **Technische Analyse**: Automatische Berechnung von Indikatoren (RSI, MACD, Bollinger Bands, etc.)
- **TimescaleDB Integration**: Direkte Verbindung zur TimescaleDB für Zeitreihendaten
- **Automatischer RAG-Sync**: Kontinuierliche Synchronisation von Marktdaten in das RAG-System

## Voraussetzungen

- Python 3.11+
- Ollama mit Llama 3.1 70B
- TimescaleDB Datenbank

## Installation

```bash
# Repository klonen
cd KITradingModel

# Virtuelle Umgebung erstellen
python -m venv venv
venv\Scripts\activate  # Windows
# oder: source venv/bin/activate  # Linux/Mac

# Dependencies installieren
pip install -r requirements.txt

# Konfiguration anpassen
copy .env.example .env
# .env Datei bearbeiten
```

## Konfiguration

Erstelle eine `.env` Datei basierend auf `.env.example`:

```env
# TimescaleDB
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=easyinsight
TIMESCALEDB_USER=postgres
TIMESCALEDB_PASSWORD=your_password

# RAG Sync
RAG_SYNC_ENABLED=true
RAG_SYNC_INTERVAL_SECONDS=300

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b

# FAISS
FAISS_PERSIST_DIRECTORY=./data/faiss
```

## Verwendung

### Service starten

```bash
# Lokal
python run.py

# Oder mit uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Analyse erstellen

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "lookback_days": 30,
    "include_technical": true
  }'
```

#### Schnelle Empfehlung

```bash
curl "http://localhost:8000/api/v1/recommendation/BTC-USD"
```

#### Health Check

```bash
curl "http://localhost:8000/api/v1/health"
```

#### RAG Dokument hinzufügen

```bash
curl -X POST "http://localhost:8000/api/v1/rag/document" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "BTC hat ein Double Bottom Pattern gebildet...",
    "document_type": "pattern",
    "symbol": "BTC/USD"
  }'
```

#### RAG abfragen

```bash
curl "http://localhost:8000/api/v1/rag/query?query=RSI%20oversold&symbol=BTC/USD"
```

## Architektur

```
KITradingModel/
├── src/
│   ├── api/              # FastAPI Routes
│   ├── models/           # Pydantic Datenmodelle
│   ├── services/
│   │   ├── llm_service.py              # Llama 3.1 Integration
│   │   ├── rag_service.py              # FAISS RAG System
│   │   ├── analysis_service.py         # Hauptanalyse-Pipeline
│   │   └── timescaledb_sync_service.py # TimescaleDB Sync Service
│   ├── config.py         # Konfiguration
│   └── main.py           # FastAPI App
├── static/               # Dashboard UI
├── data/
│   └── faiss/            # RAG Datenbank
├── logs/                 # Log-Dateien
└── requirements.txt
```

## API Dokumentation

Nach dem Start verfügbar unter:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Llama 3.1 70B Setup

### Mit Ollama

```bash
# Ollama installieren (falls nicht vorhanden)
# Windows: https://ollama.ai/download/windows

# Modell laden
ollama pull llama3.1:70b

# Oder kleinere Variante für weniger VRAM
ollama pull llama3.1:8b
```

### Hardware-Anforderungen

- **70B Modell**: ~48GB VRAM (oder CPU mit ~140GB RAM)
- **8B Modell**: ~8GB VRAM

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
