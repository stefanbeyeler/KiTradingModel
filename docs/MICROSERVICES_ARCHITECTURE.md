# Microservices Architektur

## Übersicht

Das KI Trading Model verwendet eine Microservices-Architektur mit 9 spezialisierten Services, die auf einem NVIDIA Jetson AGX Thor deployed sind.

## Architektur-Diagramm

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          Client (Browser)                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                         Port 3000
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Frontend Dashboard (Nginx)                          │
│                  - Static File Serving                              │
│                  - API Gateway / Reverse Proxy                      │
│                  - Routing zu allen Backend-Services                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────┬───────┬───────┼───────┬───────┬───────┬───────┐
        │       │       │       │       │       │       │       │
        ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
     Data    NHITS    TCN     HMM   Embedder  RAG     LLM   Watchdog
    :3001   :3002   :3003   :3004   :3005   :3008   :3009   :3010
```

## Services

### 1. Frontend Dashboard (Port 3000)

**Funktion:** API Gateway und Web-Interface

- Nginx-basierter Reverse Proxy
- Statische Web-UI (HTML/CSS/JS)
- Routing zu allen Backend-Services
- Health-Check Aggregation
- Resource-Monitoring Dashboard

**Technologien:** Nginx, HTML5, JavaScript

### 2. Data Service (Port 3001)

**Funktion:** Zentrales Daten-Gateway

- Symbol-Management und Kategorisierung
- Trading-Strategien Verwaltung
- EasyInsight API Integration (TimescaleDB)
- TwelveData und Yahoo Finance Fallback
- Externe Datenquellen Gateway
- Technische Indikatoren Berechnung

**Technologien:** Python 3.11, FastAPI, httpx, Pydantic

**Wichtig:** Einziger Service mit direktem Zugriff auf externe Datenquellen!

### 3. NHITS Service (Port 3002)

**Funktion:** Preisprognosen und Modell-Training

- NHITS (Neural Hierarchical Interpolation Time Series) Modelle
- Multi-Timeframe Support (M5, M15, H1, D1)
- GPU-beschleunigtes Training
- Inkrementelles Modell-Update
- Training-Fortschritts-Tracking

**Technologien:** Python 3.11, PyTorch, NeuralForecast, CUDA

**GPU:** Ja (CUDA)

### 4. TCN-Pattern Service (Port 3003)

**Funktion:** Chart-Pattern-Erkennung

- Temporal Convolutional Network
- 16 Chart-Pattern Typen
- Multi-Timeframe Pattern-Scan
- Confidence-basierte Filterung
- Echtzeit-Pattern-Detection

**Technologien:** Python 3.11, PyTorch, CUDA

**GPU:** Ja (CUDA)

### 5. HMM-Regime Service (Port 3004)

**Funktion:** Marktphasen-Erkennung

- Hidden Markov Model (4 Zustände)
- Regime-Klassifikation (Trending/Ranging/etc.)
- LightGBM Signal-Scoring
- Transition-Wahrscheinlichkeiten

**Technologien:** Python 3.11, hmmlearn, LightGBM, scikit-learn

**GPU:** Nein (CPU-optimiert)

### 6. Embedder Service (Port 3005)

**Funktion:** Zentraler Embedding-Service

- Text Embeddings (Sentence-Transformers)
- FinBERT für Finanz-Sentiment
- TimeSeries Embeddings
- Feature Embeddings für ML
- Batch-Verarbeitung

**Technologien:** Python 3.11, Sentence-Transformers, FinBERT, PyTorch, CUDA

**GPU:** Ja (CUDA)

### 7. RAG Service (Port 3008)

**Funktion:** Vector Search & Knowledge Base

- FAISS-basierte Vektorsuche
- Dokumenten-Management
- Semantische Ähnlichkeitssuche
- Chunk-basierte Indexierung
- Persistenter Vektor-Index

**Technologien:** Python 3.11, FAISS, LangChain, Sentence-Transformers

**GPU:** Ja (FAISS-GPU)

### 8. LLM Service (Port 3009)

**Funktion:** Trading-Analyse mit LLM

- Llama 3.1 70B via Ollama
- RAG-augmentierte Antworten
- Trading-spezifische Prompts
- Marktanalysen und Empfehlungen
- Chat-Interface

**Technologien:** Python 3.11, Ollama, LangChain, httpx

**GPU:** Ja (Ollama nutzt GPU)

### 9. Watchdog Service (Port 3010)

**Funktion:** Service-Monitoring

- Health-Check Monitoring aller Services
- Telegram-Alerts bei Ausfällen
- Recovery-Benachrichtigungen
- Cooldown-Mechanismus
- Status-Dashboard API

**Technologien:** Python 3.11, FastAPI, httpx, Telegram Bot API

**GPU:** Nein

## Datenfluss-Architektur

```text
Externe APIs (EasyInsight, TwelveData, Yahoo Finance)
                         │
                         ▼
              ┌─────────────────────┐
              │    DATA SERVICE     │  ◄── Einziges Gateway für externe Daten
              │     (Port 3001)     │
              └──────────┬──────────┘
                         │
     ┌───────┬───────┬───┴───┬───────┬───────┐
     ▼       ▼       ▼       ▼       ▼       ▼
  NHITS    TCN     HMM   Embedder  RAG     LLM
 :3002   :3003   :3004   :3005   :3008   :3009
```

### Regeln

1. **Kein direkter Datenbankzugriff** - Alle Services nutzen den Data Service
2. **Data Service als Gateway** - Einziger Kontaktpunkt zu externen APIs
3. **Fallback-Logik im Data Service** - TwelveData/Yahoo als Backup

## Nginx Proxy-Konfiguration

| Pfad | Ziel | Port |
|------|------|------|
| `/data/*` | Data Service | 3001 |
| `/nhits/*` | NHITS Service | 3002 |
| `/tcn/*` | TCN Service | 3003 |
| `/hmm/*` | HMM Service | 3004 |
| `/embedder/*` | Embedder Service | 3005 |
| `/rag/*` | RAG Service | 3008 |
| `/llm/*` | LLM Service | 3009 |
| `/watchdog/*` | Watchdog Service | 3010 |
| `/easyinsight/*` | EasyInsight API | 10.1.19.102:3000 |

## Docker Netzwerk

Alle Services kommunizieren über das interne Docker-Netzwerk `trading-net`:

```yaml
networks:
  trading-net:
    driver: bridge
```

## Volumes

| Volume | Service | Beschreibung |
|--------|---------|--------------|
| nhits-models | NHITS | Trainierte Prognose-Modelle |
| tcn-models | TCN | Trainierte Pattern-Modelle |
| hmm-models | HMM | HMM/LightGBM Modelle |
| embedder-models | Embedder | Embedding-Modelle Cache |
| rag-faiss | RAG | FAISS Vector-Datenbank |
| symbols-data | Data | Symbol-Konfigurationen |
| logs-data | Alle | Zentrale Log-Dateien |

## GPU-Ressourcen

Der Jetson AGX Thor verwendet Shared VRAM (Unified Memory):

| Service | GPU-Nutzung | Geschätzter VRAM |
|---------|-------------|------------------|
| NHITS | Training & Inference | 4-8 GB |
| TCN | Inference | 2-4 GB |
| Embedder | Model Loading | 2-4 GB |
| RAG | FAISS Index | 1-2 GB |
| LLM (Ollama) | Llama 3.1 | 20-40 GB |

**Gesamt:** ~30-60 GB bei gleichzeitiger Nutzung

## Health Checks

Alle Services implementieren einen `/health` Endpoint:

```json
{
  "status": "healthy",
  "service": "nhits-service",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "training_in_progress": false
}
```

## Swagger UI

Jeder Service bietet interaktive API-Dokumentation:

| Service | URL |
|---------|-----|
| Data | <http://10.1.19.101:3000/data/docs> |
| NHITS | <http://10.1.19.101:3000/nhits/docs> |
| TCN | <http://10.1.19.101:3000/tcn/docs> |
| HMM | <http://10.1.19.101:3000/hmm/docs> |
| Embedder | <http://10.1.19.101:3000/embedder/docs> |
| RAG | <http://10.1.19.101:3000/rag/docs> |
| LLM | <http://10.1.19.101:3000/llm/docs> |
| Watchdog | <http://10.1.19.101:3000/watchdog/docs> |

## Skalierbarkeit

Die Microservices-Architektur ermöglicht:

- **Horizontale Skalierung**: Einzelne Services können repliziert werden
- **Unabhängige Deployments**: Services können einzeln aktualisiert werden
- **Fehler-Isolation**: Ein Service-Ausfall beeinträchtigt andere Services nicht
- **Resource-Optimierung**: GPU-Services können priorisiert werden

## Monitoring

### Watchdog Service

Der Watchdog überwacht alle Services:

- Check-Intervall: 30 Sekunden
- Alert-Cooldown: 15 Minuten
- Telegram-Integration für Echtzeit-Alerts

### System-Metriken

Der NHITS Service bietet System-Metriken:

```bash
curl http://10.1.19.101:3000/nhits/api/v1/system/metrics
```

Response:

```json
{
  "cpu": {"percent": 25, "cores_logical": 12, "temp_celsius": 45},
  "memory": {"percent": 60, "total_gb": 64, "used_gb": 38},
  "gpu": {"available": true, "memory_percent": 50, "temp_celsius": 55}
}
```

## Weiterführende Dokumentation

- [Port-Konfiguration](./PORT_CONFIGURATION.md)
- [Deployment Guide](./DEPLOYMENT_MICROSERVICES.md)
- [Event-Based Training](./EVENT_BASED_TRAINING.md)
- [Watchdog Proposal](./WATCHDOG_PROPOSAL.md)
- [Architektur-Proposal (historisch)](./MICROSERVICES_ARCHITECTURE_PROPOSAL.md)
