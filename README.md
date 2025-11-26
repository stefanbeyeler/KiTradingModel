# KI Trading Model

Lokaler KI-Service für die Generierung von Handelsempfehlungen basierend auf Zeitreihendaten.

## Features

- **Llama 3.1 70B Integration**: Komplexe Marktanalysen und Reasoning über Ollama
- **RAG System**: Historische Trading-Daten mit FAISS für kontextbezogene Empfehlungen
- **Technische Analyse**: Automatische Berechnung von Indikatoren (RSI, MACD, Bollinger Bands, etc.)
- **TimescaleDB Integration**: Direkte Verbindung zur TimescaleDB für Zeitreihendaten
- **Automatischer RAG-Sync**: Kontinuierliche Synchronisation von Marktdaten in das RAG-System
- **GPU-Beschleunigung**: CUDA-Support für Embeddings (RTX 3070 optimiert)

## Voraussetzungen

- Python 3.11+ (Python 3.13 unterstützt)
- Ollama mit Llama 3.1 70B
- TimescaleDB Datenbank
- NVIDIA GPU mit CUDA 12.4 (optional, für beschleunigte Embeddings)

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

### GPU-Setup (optional, empfohlen)

Für GPU-beschleunigte Embeddings:

```bash
# Windows: GPU Setup Script ausführen
setup_gpu.bat

# Oder manuell:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Nach der Installation werden Embeddings automatisch auf der GPU generiert.

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
RAG_SYNC_INTERVAL_SECONDS=60

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b-instruct-q4_K_M

# Ollama Performance (optimiert für i9-13900K)
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_GPU=-1
OLLAMA_NUM_THREAD=16

# FAISS
FAISS_PERSIST_DIRECTORY=./data/faiss
FAISS_USE_GPU=false

# Embeddings (GPU-beschleunigt)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=64
USE_HALF_PRECISION=true
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

Alle Endpoints sind unter `/api/v1/` verfügbar.

#### Analyse & Empfehlungen

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| POST | `/analyze` | Vollständige Marktanalyse mit LLM |
| GET | `/recommendation/{symbol}` | Schnelle Trading-Empfehlung (regelbasiert) |
| GET | `/recommendation/{symbol}?use_llm=true` | Trading-Empfehlung mit LLM-Analyse |
| GET | `/symbols` | Verfügbare Symbole aus TimescaleDB |

```bash
# Vollständige Analyse (langsam, ~30-60s, mit LLM)
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "lookback_days": 30,
    "include_technical": true
  }'

# Schnelle Empfehlung (regelbasiert, ~100ms, ohne LLM)
curl "http://localhost:8000/api/v1/recommendation/BTC-USD"

# Empfehlung mit LLM (langsamer, ~30-60s, detaillierter)
curl "http://localhost:8000/api/v1/recommendation/BTC-USD?use_llm=true"
```

#### System & Health

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/health` | Service-Gesundheitsstatus |
| GET | `/system/info` | GPU, PyTorch & Konfigurationsdetails |

```bash
# Health Check
curl "http://localhost:8000/api/v1/health"

# System Info (GPU-Status, Konfiguration)
curl "http://localhost:8000/api/v1/system/info"
```

Beispiel-Antwort `/system/info`:
```json
{
  "system": {
    "device": "cuda",
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3070",
    "gpu_memory_gb": 8.0,
    "pytorch_version": "2.6.0+cu124"
  },
  "configuration": {
    "ollama_model": "llama3.1:70b-instruct-q4_K_M",
    "embedding_device": "cuda",
    "use_half_precision": true
  }
}
```

#### LLM Service

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/llm/status` | LLM-Modell Status & Optionen |
| POST | `/llm/pull` | LLM-Modell herunterladen |

```bash
# LLM Status
curl "http://localhost:8000/api/v1/llm/status"
```

#### RAG System

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/rag/stats` | RAG-Statistiken (Dokumentanzahl, Device) |
| GET | `/rag/query` | Relevante Dokumente abfragen |
| POST | `/rag/document` | Dokument hinzufügen |
| POST | `/rag/persist` | RAG-Index auf Disk speichern |
| DELETE | `/rag/documents` | Dokumente löschen |

```bash
# RAG Statistiken
curl "http://localhost:8000/api/v1/rag/stats"

# RAG abfragen
curl "http://localhost:8000/api/v1/rag/query?query=RSI%20oversold&symbol=BTC/USD"

# Dokument hinzufügen
curl -X POST "http://localhost:8000/api/v1/rag/document" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "BTC hat ein Double Bottom Pattern gebildet...",
    "document_type": "pattern",
    "symbol": "BTC/USD"
  }'
```

#### TimescaleDB Sync

| Methode | Endpoint | Beschreibung |
|---------|----------|--------------|
| GET | `/sync/status` | Sync-Service Status |
| POST | `/sync/start` | Sync-Service starten |
| POST | `/sync/stop` | Sync-Service stoppen |
| POST | `/sync/manual` | Manuellen Sync auslösen |

```bash
# Sync Status
curl "http://localhost:8000/api/v1/sync/status"

# Manueller Sync (letzte 7 Tage)
curl -X POST "http://localhost:8000/api/v1/sync/manual?days_back=7"
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

### Optimierte Konfiguration (i9-13900K + RTX 3070)

Das System ist für folgende Hardware optimiert:

| Komponente | Spezifikation | Nutzung |
|------------|---------------|---------|
| CPU | Intel i9-13900K (24 Kerne) | Ollama LLM (16 Threads) |
| RAM | 128 GB | LLM-Modell Speicher |
| GPU | RTX 3070 (8 GB VRAM) | Embeddings (FP16) |

Die Konfiguration nutzt:
- **16 CPU-Threads** für Ollama (Performance-Kerne)
- **GPU-beschleunigte Embeddings** mit Half Precision (FP16)
- **8192 Token Context Window** für längere Analysen

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
