# Microservices-Architektur für KI Trading Model

## Executive Summary

Dieses Dokument analysiert die Umstellung von der aktuellen monolithischen Architektur auf eine Microservices-basierte Architektur mit separaten Containern für Frontend, NHITS und LLM.

**Empfehlung**: ✅ **Schrittweise Migration** - Beginnen mit Frontend-Separation, dann bei Bedarf Backend-Split.

---

## Aktuelle Architektur (Monolith)

### Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                          │
│                  ki-trading-jetson:latest                    │
│                       Port 3011                              │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application (src/main.py)                          │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ NHITS Service  │  │  LLM Service   │  │ RAG Service  │  │
│  │ - Training     │  │ - Ollama       │  │ - FAISS      │  │
│  │ - Forecasting  │  │ - Analysis     │  │ - ChromaDB   │  │
│  │ - GPU (CUDA)   │  │ - GPU (CUDA)   │  │              │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ Symbol Mgmt    │  │  Sync Service  │  │ Query Logs   │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Shared Resources                         │  │
│  │  - TimescaleDB Connection                            │  │
│  │  - Model Files (/app/data/models/)                   │  │
│  │  - RAG Database (/app/data/rag/)                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Vor- und Nachteile

**Vorteile:**
- ✅ Einfaches Deployment (ein Container)
- ✅ Keine Inter-Service Communication nötig
- ✅ Geringer Overhead
- ✅ Einfaches Debugging

**Nachteile:**
- ❌ NHITS-Training blockiert andere Services
- ❌ Keine separate Skalierung möglich
- ❌ GPU-Ressourcen-Konflikte zwischen NHITS und LLM
- ❌ Ein Fehler kann gesamtes System lahmlegen
- ❌ Monolithische Updates (kein Rolling Update)

---

## Vorgeschlagene Microservices-Architektur

### Option A: Minimal Split (2-Tier)

**Empfohlen für**: Sofortiger Quick-Win mit minimalem Aufwand

```
┌───────────────────────────────────────────────────────────────┐
│                     Port 3000 - Frontend                       │
│                   (React/Vue Dashboard)                        │
├───────────────────────────────────────────────────────────────┤
│  - Nginx (Static File Serving)                               │
│  - React/Vue Single Page Application                         │
│  - Swagger UI Integration                                    │
│  - API Gateway / Reverse Proxy                               │
│    → Proxy /api/v1/* → Backend (Port 3011)                  │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────┐
│                    Port 3011 - Backend API                     │
│               (Aktueller Monolith unverändert)                │
├───────────────────────────────────────────────────────────────┤
│  FastAPI mit allen Services:                                 │
│  - NHITS (Training + Forecasting)                            │
│  - LLM (Ollama + Analysis)                                   │
│  - RAG, Symbol Management, Sync, etc.                        │
└───────────────────────────────────────────────────────────────┘
```

**Vorteile:**
- ✅ Schnelle Umsetzung (1-2 Tage)
- ✅ Frontend unabhängig deploybar
- ✅ Keine Breaking Changes am Backend
- ✅ Progressive Enhancement möglich
- ✅ Bessere UX (dediziertes Dashboard)

**Nachteile:**
- ⚠️ Backend bleibt Monolith
- ⚠️ GPU-Konflikte bleiben bestehen

---

### Option B: Service Split (3-Tier)

**Empfohlen für**: Mittelfristige Optimierung bei Performance-Problemen

```
┌───────────────────────────────────────────────────────────────┐
│                     Port 3000 - Frontend                       │
│                   Nginx + React/Vue SPA                        │
│                   API Gateway & Routing                        │
└───────────────────────────────────────────────────────────────┘
         ↓                    ↓                      ↓
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  Port 3001      │  │  Port 3002      │  │  Port 3003       │
│  NHITS Service  │  │  LLM Service    │  │  Data Service    │
├─────────────────┤  ├─────────────────┤  ├──────────────────┤
│ FastAPI         │  │ FastAPI         │  │ FastAPI          │
│ GPU: nvidia     │  │ GPU: nvidia     │  │ GPU: none        │
│                 │  │                 │  │                  │
│ Endpoints:      │  │ Endpoints:      │  │ Endpoints:       │
│ - /forecast/*   │  │ - /analyze      │  │ - /symbols/*     │
│ - /training/*   │  │ - /llm/*        │  │ - /strategies/*  │
│ - Model Files   │  │ - /rag/*        │  │ - /sync/*        │
│                 │  │ - Ollama        │  │ - /query-logs/*  │
│                 │  │                 │  │ - TimescaleDB    │
└─────────────────┘  └─────────────────┘  └──────────────────┘
         ↓                    ↓                      ↓
    ┌────────────────────────────────────────────────────┐
    │         Shared Docker Volumes & Network            │
    │  - models:/app/data/models (NHITS ↔ Frontend)    │
    │  - rag:/app/data/rag (LLM ↔ Data)               │
    │  - timescale: network connection                  │
    └────────────────────────────────────────────────────┘
```

**Routing im Frontend (Nginx):**

```nginx
# Port 3000 - API Gateway
location /api/v1/forecast/ {
    proxy_pass http://nhits-service:3001;
}

location /api/v1/analyze {
    proxy_pass http://llm-service:3002;
}

location /api/v1/llm/ {
    proxy_pass http://llm-service:3002;
}

location /api/v1/rag/ {
    proxy_pass http://llm-service:3002;
}

location /api/v1/symbols/ {
    proxy_pass http://data-service:3003;
}

location /api/v1/strategies/ {
    proxy_pass http://data-service:3003;
}

location /api/v1/sync/ {
    proxy_pass http://data-service:3003;
}
```

**Vorteile:**
- ✅ **GPU-Isolation**: NHITS und LLM haben dedizierte GPU-Ressourcen
- ✅ **Unabhängiges Scaling**: NHITS-Training skaliert getrennt von LLM
- ✅ **Fehler-Isolation**: LLM-Crash betrifft NHITS nicht
- ✅ **Unabhängige Updates**: NHITS neu deployen ohne LLM zu berühren
- ✅ **Resource Limits**: Jeder Service hat eigene CPU/Memory Limits
- ✅ **Monitoring**: Separate Metriken pro Service

**Nachteile:**
- ⚠️ **Komplexität**: 3 FastAPI Apps zu maintainen
- ⚠️ **Memory Overhead**: 3x Python Runtime
- ⚠️ **Network Latency**: Inter-Service HTTP Calls
- ⚠️ **Shared State**: Model Files müssen synchronisiert werden

---

### Option C: Full Microservices (4-Tier+)

**Empfohlen für**: Langfristig bei starkem Wachstum

```
Port 3000: Frontend (Nginx + SPA)
Port 3001: NHITS Service (Training & Forecasting)
Port 3002: LLM Service (Ollama + RAG + Analysis)
Port 3003: Data Service (Symbols, Strategies, Sync)
Port 3004: Event Service (Event-Based Training Monitor)
Port 3005: Metrics Service (Prometheus Exporter)
```

**Nicht empfohlen**, da:
- Over-Engineering für aktuelle Anforderungen
- Zu viel Overhead für Jetson Hardware
- Komplexe Service Discovery nötig

---

## Detaillierte Service-Aufteilung

### Service 1: Frontend (Port 3000)

**Technologie:** Nginx + React/Vue

**Zuständigkeiten:**
- Static File Serving (HTML, CSS, JS)
- API Gateway / Reverse Proxy
- WebSocket Proxy (für Echtzeit-Updates)
- SSL Termination
- Rate Limiting
- CORS Handling

**Dependencies:**
- Keine direkten Backend-Dependencies
- Kommuniziert nur über HTTP mit Backend-Services

**Dockerfile:**

```dockerfile
FROM nginx:alpine

# Copy React build
COPY frontend/build /usr/share/nginx/html

# Copy nginx config with API routing
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Resource Requirements:**
- CPU: 0.5 cores
- Memory: 256 MB
- GPU: None

---

### Service 2: NHITS Service (Port 3001)

**Technologie:** FastAPI + PyTorch + CUDA

**Zuständigkeiten:**
- NHITS Model Training
- Price Forecasting
- Model Performance Evaluation
- Event-Based Training
- Model File Management

**API Endpoints:**
- `GET /forecast/{symbol}` - Generate forecast
- `POST /forecast/train-all` - Batch training
- `GET /forecast/training/progress` - Training progress
- `GET /forecast/performance` - Model metrics
- `POST /forecast/training/events/start` - Event monitor

**Dependencies:**
- TimescaleDB (für historische Preisdaten)
- Shared Volume (Model Files)
- EasyInsight API (für Events)

**GPU Requirements:**
- Braucht CUDA für Training
- Kann CPU für Inference nutzen (langsamer)

**main.py (NHITS Service):**

```python
from fastapi import FastAPI
from .routes import forecast_router, training_router
from .services.nhits_training_service import nhits_training_service
from .services.event_based_training_service import event_based_training_service

app = FastAPI(title="NHITS Service", version="1.0.0")

app.include_router(forecast_router, prefix="/api/v1")
app.include_router(training_router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    # Start scheduled training
    await nhits_training_service.start()

@app.on_event("shutdown")
async def shutdown():
    await nhits_training_service.stop()
```

**Resource Requirements:**
- CPU: 4-8 cores (für Training)
- Memory: 8-16 GB
- GPU: 1x NVIDIA (shared with LLM)
- Disk: 10 GB (Models)

---

### Service 3: LLM Service (Port 3002)

**Technologie:** FastAPI + Ollama + FAISS

**Zuständigkeiten:**
- LLM-basierte Trading-Analyse
- RAG (Retrieval Augmented Generation)
- Wissensbasis-Management
- Strategie-Evaluierung

**API Endpoints:**
- `POST /analyze` - Trading analysis with LLM
- `GET /llm/status` - LLM status
- `POST /rag/document` - Add to knowledge base
- `GET /rag/query` - Query RAG system

**Dependencies:**
- Ollama Service (llama3.1:70b)
- FAISS Vector DB
- ChromaDB (optional)

**GPU Requirements:**
- Braucht CUDA für LLM Inference
- Kann CPU für RAG nutzen

**main.py (LLM Service):**

```python
from fastapi import FastAPI
from .routes import llm_router, rag_router, trading_router
from .services.llm_service import LLMService
from .services.rag_service import RAGService

app = FastAPI(title="LLM Service", version="1.0.0")

llm_service = LLMService()
rag_service = RAGService()

app.include_router(llm_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.include_router(trading_router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    await llm_service.initialize()

@app.on_event("shutdown")
async def shutdown():
    await llm_service.cleanup()
```

**Resource Requirements:**
- CPU: 4 cores
- Memory: 16-32 GB (für llama3.1:70b)
- GPU: 1x NVIDIA (shared with NHITS)
- Disk: 50 GB (LLM Models)

---

### Service 4: Data Service (Port 3003)

**Technologie:** FastAPI + PostgreSQL/TimescaleDB

**Zuständigkeiten:**
- Symbol Management
- Trading Strategies
- TimescaleDB Synchronisation
- Query Logs
- System Monitoring

**API Endpoints:**
- `GET /symbols` - List symbols
- `POST /symbols` - Add symbol
- `GET /strategies` - List strategies
- `POST /sync/start` - Start sync service
- `GET /query-logs` - Query logs

**Dependencies:**
- TimescaleDB (Primary Database)
- EasyInsight API (für Symbol Data)

**GPU Requirements:**
- Keine GPU nötig

**Resource Requirements:**
- CPU: 2 cores
- Memory: 2 GB
- GPU: None
- Disk: 5 GB (Logs, Cache)

---

## Docker Compose Architektur

### Option A: Minimal Split

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - BACKEND_URL=http://backend:3011
    networks:
      - trading-net

  backend:
    build:
      context: .
      dockerfile: docker/jetson/Dockerfile
    image: ki-trading-jetson:latest
    restart: unless-stopped
    runtime: nvidia
    ports:
      - "3011:3011"
    volumes:
      - ./:/app:rw
    environment:
      - FAISS_USE_GPU=1
      - EMBEDDING_DEVICE=cuda
      - NHITS_USE_GPU=1
    networks:
      - trading-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

networks:
  trading-net:
    driver: bridge
```

---

### Option B: Service Split

```yaml
version: '3.8'

services:
  # Frontend - API Gateway & UI
  frontend:
    build: ./frontend
    container_name: trading-frontend
    ports:
      - "3000:80"
    depends_on:
      - nhits-service
      - llm-service
      - data-service
    networks:
      - trading-net
    restart: unless-stopped

  # NHITS Service - Training & Forecasting
  nhits-service:
    build:
      context: .
      dockerfile: docker/services/nhits/Dockerfile
    container_name: trading-nhits
    runtime: nvidia
    ports:
      - "3001:3001"
    volumes:
      - models-data:/app/data/models
      - ./src:/app/src:ro
    environment:
      - SERVICE_NAME=nhits
      - NHITS_USE_GPU=1
      - PORT=3001
      - TIMESCALE_HOST=timescaledb
      - EASYINSIGHT_API_URL=http://10.1.19.102:3000
    networks:
      - trading-net
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  # LLM Service - Analysis & RAG
  llm-service:
    build:
      context: .
      dockerfile: docker/services/llm/Dockerfile
    container_name: trading-llm
    runtime: nvidia
    ports:
      - "3002:3002"
    volumes:
      - rag-data:/app/data/rag
      - ollama-models:/root/.ollama
      - ./src:/app/src:ro
    environment:
      - SERVICE_NAME=llm
      - PORT=3002
      - OLLAMA_MODEL=llama3.1:70b
      - FAISS_USE_GPU=1
    networks:
      - trading-net
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  # Data Service - Symbols, Strategies, Sync
  data-service:
    build:
      context: .
      dockerfile: docker/services/data/Dockerfile
    container_name: trading-data
    ports:
      - "3003:3003"
    volumes:
      - ./src:/app/src:ro
      - logs-data:/app/logs
    environment:
      - SERVICE_NAME=data
      - PORT=3003
      - TIMESCALE_HOST=timescaledb
      - TIMESCALE_PORT=5432
      - EASYINSIGHT_API_URL=http://10.1.19.102:3000
    networks:
      - trading-net
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G

  # Ollama Service (für LLM)
  ollama:
    image: ollama/ollama:latest
    container_name: trading-ollama
    runtime: nvidia
    volumes:
      - ollama-models:/root/.ollama
    networks:
      - trading-net
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  models-data:
    driver: local
  rag-data:
    driver: local
  ollama-models:
    driver: local
  logs-data:
    driver: local

networks:
  trading-net:
    driver: bridge
```

---

## GPU-Ressourcen-Management

### Problem: GPU-Sharing zwischen NHITS und LLM

**Jetson AGX Thor** hat begrenzte GPU-Ressourcen:
- Beide Services (NHITS & LLM) brauchen GPU
- Training kann LLM-Queries blockieren
- Memory-Konflikte möglich

### Lösungen

#### Option 1: Time-Slicing (Aktuell)

```yaml
nhits-service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']  # Gleiche GPU
            capabilities: [gpu]

llm-service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']  # Gleiche GPU
            capabilities: [gpu]
```

**Problem:** Beide konkurrieren um GPU-Memory

#### Option 2: GPU Memory Fraction

```python
# NHITS Service
import torch
torch.cuda.set_per_process_memory_fraction(0.4, device=0)  # 40% GPU

# LLM Service
torch.cuda.set_per_process_memory_fraction(0.6, device=0)  # 60% GPU
```

**Vorteil:** Klare Memory-Grenzen

#### Option 3: Priority-Based Scheduling

```yaml
nhits-service:
  environment:
    - CUDA_VISIBLE_DEVICES=0
    - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    - NHITS_TRAINING_PRIORITY=low  # Nutzt GPU nur wenn verfügbar

llm-service:
  environment:
    - CUDA_VISIBLE_DEVICES=0
    - LLM_PRIORITY=high  # Hat Vorrang bei GPU-Nutzung
```

#### Option 4: Separate GPU Pools (falls Multi-GPU)

```yaml
nhits-service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']  # GPU 0 für NHITS

llm-service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['1']  # GPU 1 für LLM
```

**Nur möglich bei Multi-GPU Systemen**

---

## Inter-Service Communication

### HTTP REST (Empfohlen)

```python
# NHITS Service ruft LLM Service
import httpx

async def get_llm_analysis(symbol: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://llm-service:3002/api/v1/analyze",
            json={"symbol": symbol}
        )
        return response.json()
```

**Vorteile:**
- ✅ Einfach zu implementieren
- ✅ Gut für Debugging
- ✅ Standard HTTP Tools nutzbar

**Nachteile:**
- ⚠️ Langsamer als gRPC
- ⚠️ JSON Serialization Overhead

### gRPC (Alternative)

**Nur sinnvoll bei:**
- Sehr hohem Traffic zwischen Services
- Streng typisierte Contracts nötig
- Performance kritisch

**Nicht empfohlen** für aktuelles Setup

### Message Queue (RabbitMQ/Redis)

**Sinnvoll für:**
- Asynchrone Tasks (z.B. Training-Jobs)
- Event-Driven Architecture

**Beispiel:**

```python
# LLM Service publiziert Event
await redis.publish("market-events", {
    "symbol": "EURUSD",
    "event": "high_volatility"
})

# NHITS Service subscribed
async for message in redis.subscribe("market-events"):
    await trigger_retraining(message["symbol"])
```

---

## Migration Strategy

### Phase 1: Vorbereitung (1 Woche)

**Ziel:** Code modularisieren ohne Deployment zu ändern

1. **Router-Gruppierung** (✅ Bereits gemacht!)
   - `forecast_router` → NHITS Endpoints
   - `llm_router` → LLM Endpoints
   - `system_router` → Data Endpoints

2. **Service-Isolation**
   - Keine direkten Imports zwischen Services
   - Klare Interfaces definieren
   - Shared Code in `common/` Modul

3. **Config-Management**
   - Environment-basierte Config pro Service
   - Secrets Management (Vault/Docker Secrets)

**Deliverables:**
- Saubere Router-Struktur ✅
- Service-Dependencies dokumentiert
- Migration-Plan erstellt

---

### Phase 2: Frontend-Separation (1-2 Wochen)

**Ziel:** Frontend als eigener Container

1. **React/Vue Dashboard erstellen**
   - Trading Dashboard UI
   - Swagger UI Integration
   - API Client generieren (aus OpenAPI)

2. **Nginx als API Gateway**
   - Reverse Proxy zu Backend
   - Static File Serving
   - CORS Handling

3. **Docker Setup**
   - Frontend Dockerfile
   - docker-compose mit Frontend + Backend

**Deliverables:**
- Frontend läuft auf Port 3000
- Backend unverändert auf Port 3011
- Nahtloser Übergang für User

---

### Phase 3: Backend-Split (2-4 Wochen)

**Ziel:** NHITS, LLM, Data als separate Services

**Nur wenn nötig!** (Performance-Probleme, Resource-Konflikte)

1. **Service Dockerfiles erstellen**
   - `docker/services/nhits/Dockerfile`
   - `docker/services/llm/Dockerfile`
   - `docker/services/data/Dockerfile`

2. **Main-Files pro Service**
   - `src/services/nhits/main.py`
   - `src/services/llm/main.py`
   - `src/services/data/main.py`

3. **Shared Libraries**
   - `src/common/models.py`
   - `src/common/database.py`
   - `src/common/config.py`

4. **Inter-Service Communication**
   - HTTP Clients implementieren
   - Health Checks
   - Service Discovery (Docker DNS)

5. **Testing**
   - Integration Tests
   - Performance Tests
   - Rollback Plan

**Deliverables:**
- 3 separate Backend-Services
- Monitoring pro Service
- Dokumentierte API Contracts

---

## Monitoring & Observability

### Metrics (Prometheus)

```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
```

**Metriken pro Service:**
- Request Count
- Response Time
- Error Rate
- GPU Utilization
- Memory Usage
- Active Training Jobs

### Logging (Loki)

```yaml
services:
  loki:
    image: grafana/loki
    ports:
      - "3100:3100"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
      - loki
```

### Health Checks

```yaml
nhits-service:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:3001/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 40s
```

### Distributed Tracing (Optional)

**Jaeger** für Request-Tracing über Services hinweg.

Nur sinnvoll bei komplexen Inter-Service Calls.

---

## Cost-Benefit Analysis

### Option A: Minimal Split (Frontend + Backend)

**Aufwand:** 🟢 Niedrig (1-2 Wochen)

**Kosten:**
- Development: 40-80 Stunden
- Testing: 20 Stunden
- Dokumentation: 10 Stunden

**Benefits:**
- ✅ Bessere UX (dediziertes Frontend)
- ✅ Unabhängige Frontend-Updates
- ✅ Einfacheres A/B Testing
- ✅ Bessere SEO (falls relevant)

**ROI:** 🟢 Hoch - Lohnt sich fast immer

---

### Option B: Service Split (NHITS + LLM + Data)

**Aufwand:** 🟡 Mittel (2-4 Wochen)

**Kosten:**
- Development: 120-200 Stunden
- Testing: 60 Stunden
- Dokumentation: 30 Stunden
- Refactoring: 40 Stunden

**Benefits:**
- ✅ GPU-Isolation
- ✅ Unabhängige Skalierung
- ✅ Fehler-Isolation
- ✅ Besseres Monitoring
- ✅ Unabhängige Updates

**ROI:** 🟡 Mittel - Lohnt sich bei Performance-Problemen

**Trigger für Migration:**
- NHITS-Training blockiert LLM-Queries
- GPU Out-of-Memory Errors
- Häufige Service-Restarts nötig
- Team wächst (separate Entwickler pro Service)

---

### Option C: Full Microservices (4+ Services)

**Aufwand:** 🔴 Hoch (4-8 Wochen)

**Kosten:**
- Development: 400+ Stunden
- Infrastructure: Service Mesh, API Gateway
- Operations: Kubernetes/Docker Swarm
- Monitoring: Prometheus, Grafana, Jaeger

**Benefits:**
- ✅ Maximale Flexibilität
- ✅ Enterprise-ready
- ✅ Best Practices

**ROI:** 🔴 Niedrig - Over-Engineering für aktuelle Größe

**Nur sinnvoll bei:**
- 10+ Entwickler im Team
- Millionen Requests/Tag
- Multi-Region Deployment
- SLA Requirements

---

## Empfehlung

### Sofort (Jetzt)

✅ **Minimal Split** implementieren:
- Frontend auf Port 3000 (Nginx + React/Vue)
- Backend bleibt auf Port 3011 (aktueller Monolith)

**Grund:** Quick Win mit minimalem Risiko

---

### Bei Bedarf (3-6 Monate)

⚠️ **Service Split** evaluieren:
- Nur wenn Performance-Probleme auftreten
- Oder wenn Team wächst
- Oder bei häufigen GPU-Konflikten

**Metrics beobachten:**
- GPU Utilization während NHITS Training
- LLM Response Times während Training
- Error Rates
- User Complaints

---

### Langfristig (1+ Jahr)

❌ **Full Microservices** nur bei:
- Signifikantem Wachstum
- Enterprise Requirements
- Multi-Tenant Setup

**Ansonsten:** Stick with Minimal Split oder Service Split

---

## Technische Risiken

### Risk 1: GPU Memory Konflikte

**Problem:** NHITS Training verbraucht gesamte GPU

**Mitigation:**
- Memory Limits setzen (`torch.cuda.set_per_process_memory_fraction`)
- Training in Batches mit Pausen
- LLM auf CPU fallback
- Monitoring & Alerting

---

### Risk 2: Network Latency

**Problem:** HTTP Calls zwischen Services langsamer als In-Process

**Mitigation:**
- Caching implementieren (Redis)
- Async Communication wo möglich
- Batch Requests
- Keep Services lightweight

---

### Risk 3: Shared Volume Sync

**Problem:** Model Files müssen zwischen Services geteilt werden

**Mitigation:**
- Docker Volumes mit Locking
- S3-kompatible Storage (MinIO)
- Model Registry Pattern
- Versionierung

---

### Risk 4: Operational Complexity

**Problem:** Mehr Container = mehr Maintenance

**Mitigation:**
- Gutes Monitoring (Grafana)
- Health Checks überall
- Automated Deployment (CI/CD)
- Runbooks für Common Issues

---

## Schlussfolgerung

### Die richtige Wahl für Sie

**Empfehlung:** ✅ **Starten Sie mit Option A (Minimal Split)**

**Begründung:**
1. Schnell umsetzbar (1-2 Wochen)
2. Minimales Risiko
3. Sofortige UX-Verbesserung
4. Foundation für spätere Migration

**Später evaluieren:** Option B (Service Split) bei konkreten Performance-Problemen

**Vermeiden:** Option C (Full Microservices) - Over-Engineering

---

### Next Steps

1. **Woche 1-2:** Frontend-Prototyp erstellen (React/Vue)
2. **Woche 3:** Nginx API Gateway konfigurieren
3. **Woche 4:** Integration Testing & Deployment
4. **Woche 5+:** Monitoring & Performance-Analyse

**Decision Point (3 Monate):**
- Performance-Metriken reviewen
- User Feedback sammeln
- Entscheiden: Bleiben bei Minimal Split oder zu Service Split migrieren?

---

## Anhang

### Dockerfile Examples

**Frontend:**

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

**NHITS Service:**

```dockerfile
# docker/services/nhits/Dockerfile
FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /app
COPY requirements-nhits.txt .
RUN pip install -r requirements-nhits.txt

COPY src/services/nhits src/services/nhits
COPY src/common src/common
COPY src/models src/models

ENV PORT=3001
EXPOSE 3001

CMD ["python", "-m", "uvicorn", "src.services.nhits.main:app", "--host", "0.0.0.0", "--port", "3001"]
```

### Nginx Config

```nginx
# frontend/nginx.conf
upstream nhits-backend {
    server nhits-service:3001;
}

upstream llm-backend {
    server llm-service:3002;
}

upstream data-backend {
    server data-service:3003;
}

server {
    listen 80;

    # Frontend
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    # API Routing
    location /api/v1/forecast/ {
        proxy_pass http://nhits-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/v1/analyze {
        proxy_pass http://llm-backend;
        proxy_set_header Host $host;
    }

    location /api/v1/symbols/ {
        proxy_pass http://data-backend;
        proxy_set_header Host $host;
    }
}
```

---

## Zusammenfassung

**TL;DR:**

1. **Jetzt:** Minimal Split (Frontend + Backend) ✅
2. **Später:** Service Split nur bei Bedarf ⚠️
3. **Niemals:** Full Microservices (Over-Engineering) ❌

**Key Takeaway:** Microservices sind **kein Selbstzweck**. Nur umsetzen wenn konkrete Probleme gelöst werden müssen.

Ihre aktuelle Router-Struktur ist bereits gut vorbereitet für eine spätere Migration! 🚀
