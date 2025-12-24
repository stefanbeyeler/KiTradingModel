# KI Trading Model

Microservices-basiertes KI-System für Trading-Analyse und Preisprognosen auf NVIDIA Jetson AGX Thor.

## Features

- **NHITS Neural Forecasting**: Neuronale Zeitreihenvorhersage mit Multi-Timeframe Support (M5, M15, H1, D1)
- **TCN Pattern Recognition**: Deep Learning basierte Chart-Pattern-Erkennung (16 Pattern-Typen)
- **HMM Regime Detection**: Hidden Markov Model zur Marktphasen-Erkennung mit LightGBM Signal-Scoring
- **Embedder Service**: Zentraler Embedding-Service (Text, FinBERT, TimeSeries, Features)
- **RAG System**: FAISS-basierte Vektorsuche und Knowledge Base
- **LLM Analysis**: Trading-Insights powered by Llama 3.1 (8B/70B) via Ollama
- **Watchdog Monitoring**: Service-Monitoring mit Telegram-Alerts
- **EasyInsight Integration**: REST API Anbindung an TimescaleDB Marktdaten
- **Multi-Source Fallback**: TwelveData und Yahoo Finance als Backup-Datenquellen
- **GPU-Beschleunigung**: CUDA-Support auf Jetson AGX Thor (Shared VRAM Architektur)

## Microservices Architektur

| Service | Port | Beschreibung | GPU |
|---------|------|--------------|-----|
| Frontend Dashboard | 3000 | Nginx API Gateway + Web UI | - |
| Data Service | 3001 | Symbol-Management, Strategien, Daten-Gateway | - |
| NHITS Service | 3002 | Preisprognosen und Modell-Training | CUDA |
| TCN-Pattern Service | 3003 | Chart-Pattern-Erkennung (16 Typen) | CUDA |
| HMM-Regime Service | 3004 | Marktphasen-Erkennung, Signal-Scoring | - |
| Embedder Service | 3005 | Text/FinBERT/TimeSeries Embeddings | CUDA |
| RAG Service | 3008 | Vector Search & Knowledge Base | CUDA |
| LLM Service | 3009 | Trading-Analyse mit Llama 3.1 | CUDA |
| Watchdog Service | 3010 | Service-Monitoring, Telegram-Alerts | - |

```text
                         Frontend (Dashboard)
                              Port 3000
                                  |
              +-------+-------+---+---+-------+-------+
              |       |       |       |       |       |
           Data    NHITS    TCN     HMM   Embedder  Watchdog
          :3001   :3002   :3003   :3004   :3005    :3010
                              |
                      +-------+-------+
                      |               |
                     RAG            LLM
                    :3008          :3009
```

## Quick Start

### Mit Docker (Empfohlen)

```bash
# Alle Services starten
docker-compose -f docker-compose.microservices.yml up -d

# Status prüfen
docker-compose -f docker-compose.microservices.yml ps

# Logs anzeigen
docker-compose -f docker-compose.microservices.yml logs -f
```

### Dashboard öffnen

Nach dem Start: <http://10.1.19.101:3000>

## Voraussetzungen

- Docker mit NVIDIA Container Toolkit
- NVIDIA Jetson AGX Thor (oder kompatible GPU mit CUDA)
- Ollama mit Llama 3.1 (8B oder 70B) auf Host
- EasyInsight API Zugang (TimescaleDB)

## Konfiguration

Erstelle eine `.env` Datei:

```bash
# EasyInsight API
EASYINSIGHT_API_URL=http://10.1.19.102:3000/api

# Ollama (läuft auf Host)
OLLAMA_MODEL=llama3.1:70b

# Service Ports (optional, defaults shown)
DATA_SERVICE_PORT=3001
NHITS_SERVICE_PORT=3002
TCN_SERVICE_PORT=3003
HMM_SERVICE_PORT=3004
EMBEDDER_SERVICE_PORT=3005
RAG_SERVICE_PORT=3008
LLM_SERVICE_PORT=3009
WATCHDOG_SERVICE_PORT=3010
```

## API-Dokumentation

Jeder Service bietet eine interaktive Swagger UI:

| Service | Swagger UI |
|---------|------------|
| Data Service | <http://10.1.19.101:3000/data/docs> |
| NHITS Service | <http://10.1.19.101:3000/nhits/docs> |
| TCN Service | <http://10.1.19.101:3000/tcn/docs> |
| HMM Service | <http://10.1.19.101:3000/hmm/docs> |
| Embedder Service | <http://10.1.19.101:3000/embedder/docs> |
| RAG Service | <http://10.1.19.101:3000/rag/docs> |
| LLM Service | <http://10.1.19.101:3000/llm/docs> |
| Watchdog Service | <http://10.1.19.101:3000/watchdog/docs> |

## API-Beispiele

### NHITS Prognosen

```bash
# Preisprognose abrufen
curl "http://10.1.19.101:3000/nhits/api/v1/forecast/EURUSD?horizon=24"

# Training starten
curl -X POST "http://10.1.19.101:3000/nhits/api/v1/forecast/train-all?background=true"

# Training-Fortschritt
curl "http://10.1.19.101:3000/nhits/api/v1/forecast/training/progress"
```

### TCN Pattern Detection

```bash
# Pattern-Scan
curl -X POST "http://10.1.19.101:3000/tcn/api/v1/patterns/scan" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD", "timeframes": ["H1"], "min_confidence": 0.7}'
```

### HMM Regime Detection

```bash
# Aktuelles Marktregime
curl "http://10.1.19.101:3000/hmm/api/v1/regime/EURUSD"

# Signal-Scoring
curl -X POST "http://10.1.19.101:3000/hmm/api/v1/score" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD", "signal_type": "buy"}'
```

### LLM Trading-Analyse

```bash
# Trading-Analyse mit RAG
curl -X POST "http://10.1.19.101:3000/llm/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD", "question": "Sollte ich kaufen oder verkaufen?"}'
```

### Watchdog Status

```bash
# Service-Status aller Microservices
curl "http://10.1.19.101:3000/watchdog/api/v1/status"

# Test-Alert senden
curl -X POST "http://10.1.19.101:3000/watchdog/api/v1/alerts/test"
```

## Health Checks

```bash
# Alle Services prüfen
for port in 3001 3002 3003 3004 3005 3008 3009 3010; do
  echo "Port $port: $(curl -s http://10.1.19.101:$port/health | jq -r .status)"
done
```

## Datenfluss-Architektur

```text
Externe APIs (EasyInsight, TwelveData, Yahoo Finance)
                         |
                         v
              +---------------------+
              |    DATA SERVICE     |  <-- Einziges Gateway für externe Daten
              |     (Port 3001)     |
              +---------+-----------+
                        |
        +-------+-------+-------+-------+
        |       |       |       |       |
        v       v       v       v       v
      NHITS   TCN     HMM   Embedder  RAG
     :3002   :3003   :3004   :3005   :3008
                                        |
                                        v
                                       LLM
                                      :3009
```

## Resource Requirements

| Service | RAM Limit | GPU |
|---------|-----------|-----|
| Frontend | 256 MB | - |
| Data | 4 GB | - |
| NHITS | 16 GB | CUDA |
| TCN | 8 GB | CUDA |
| HMM | 4 GB | - |
| Embedder | 12 GB | CUDA |
| RAG | 8 GB | CUDA |
| LLM | 32 GB | CUDA |
| Watchdog | 512 MB | - |

## Docker Volumes

| Volume | Beschreibung |
|--------|--------------|
| nhits-models | Trainierte NHITS-Modelle |
| tcn-models | Trainierte TCN-Modelle |
| hmm-models | Trainierte HMM-Modelle |
| embedder-models | Embedding-Modelle |
| rag-faiss | FAISS Vector-Datenbank |
| symbols-data | Symbol-Konfigurationen |
| logs-data | Zentrale Log-Dateien |

## Monitoring

### System-Metriken

```bash
# CPU/GPU/Memory Metriken
curl "http://10.1.19.101:3000/nhits/api/v1/system/metrics"

# Storage-Informationen
curl "http://10.1.19.101:3000/nhits/api/v1/system/storage"
```

### Watchdog Telegram-Alerts

Der Watchdog Service sendet Benachrichtigungen bei:

- Service-Ausfällen (Down-Alerts)
- Service-Recovery (Up-Alerts)

Konfiguration in `.env.watchdog`:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_IDS=123456789,987654321
CHECK_INTERVAL_SECONDS=30
ALERT_COOLDOWN_MINUTES=15
```

## Entwicklung

### Lokale Entwicklung

```bash
# Virtuelle Umgebung
python -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Tests
pytest tests/
```

### Code-Stil

- Python 3.11+
- Async-First (alle I/O asynchron)
- Type Hints auf allen Funktionen
- Pydantic für API-Requests/Responses

## Dokumentation

- [Microservices Architektur](docs/MICROSERVICES_ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_MICROSERVICES.md)
- [Port-Konfiguration](docs/PORT_CONFIGURATION.md)
- [Event-Based Training](docs/EVENT_BASED_TRAINING.md)
- [Development Guidelines](docs/DEVELOPMENT_GUIDELINES.md)
- [Watchdog Proposal](docs/WATCHDOG_PROPOSAL.md)

## Lizenz

MIT License
