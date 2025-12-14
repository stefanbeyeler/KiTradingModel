# Microservices Deployment Guide

## Übersicht

Dieses Deployment-Guide beschreibt, wie das KI Trading Model System als Microservices-Architektur deployed wird.

## Architektur

```
Port 3000: Frontend (Nginx + Dashboard)
    ├─> Port 3001: NHITS Service (Training & Forecasting)
    ├─> Port 3002: LLM Service (Analysis & RAG)
    └─> Port 3003: Data Service (Symbols, Strategies, Sync)
         └─> Ollama (Port 11434)
```

## Services

### Frontend (Port 3000)
- **Funktion**: API Gateway + Web Dashboard
- **Technologie**: Nginx
- **GPU**: Nein
- **Memory**: 256 MB

### NHITS Service (Port 3001)
- **Funktion**: NHITS Training & Price Forecasting
- **Technologie**: FastAPI + PyTorch
- **GPU**: Ja (CUDA)
- **Memory**: 16 GB

### LLM Service (Port 3002)
- **Funktion**: Trading Analysis & RAG
- **Technologie**: FastAPI + Ollama (llama3.1:70b)
- **GPU**: Ja (CUDA)
- **Memory**: 32 GB

### Data Service (Port 3003)
- **Funktion**: Symbols, Strategies, Sync
- **Technologie**: FastAPI + PostgreSQL
- **GPU**: Nein
- **Memory**: 4 GB

## Deployment

### Voraussetzungen

```bash
# Docker & Docker Compose
docker --version  # >= 24.0
docker-compose --version  # >= 2.20

# NVIDIA Docker Runtime
nvidia-docker --version
```

### Environment Variables

Erstellen Sie eine `.env` Datei:

```bash
# TimescaleDB
TIMESCALE_HOST=10.1.19.104
TIMESCALE_PORT=5432
TIMESCALE_DB=trading
TIMESCALE_USER=postgres
TIMESCALE_PASSWORD=your_password_here

# EasyInsight API
EASYINSIGHT_API_URL=http://10.1.19.102:3000

# LLM
OLLAMA_MODEL=llama3.1:70b
OLLAMA_BASE_URL=http://ollama:11434

# NHITS
NHITS_AUTO_RETRAIN_DAYS=7
NHITS_USE_GPU=1

# Sync
TIMESCALE_SYNC_ENABLED=true
TIMESCALE_SYNC_INTERVAL_MINUTES=60
```

### Deployment Starten

```bash
# Build alle Services
docker-compose -f docker-compose.microservices.yml build

# Starte alle Services
docker-compose -f docker-compose.microservices.yml up -d

# Logs verfolgen
docker-compose -f docker-compose.microservices.yml logs -f
```

### Service-Status prüfen

```bash
# Alle Container
docker-compose -f docker-compose.microservices.yml ps

# Health Checks
curl http://localhost:3000/health  # Frontend
curl http://localhost:3001/health  # NHITS
curl http://localhost:3002/health  # LLM
curl http://localhost:3003/health  # Data
```

## Service Management

### Einzelnen Service neu starten

```bash
# NHITS Service
docker-compose -f docker-compose.microservices.yml restart nhits-service

# LLM Service
docker-compose -f docker-compose.microservices.yml restart llm-service

# Data Service
docker-compose -f docker-compose.microservices.yml restart data-service
```

### Logs eines Services

```bash
# NHITS Service
docker-compose -f docker-compose.microservices.yml logs -f nhits-service

# LLM Service
docker-compose -f docker-compose.microservices.yml logs -f llm-service
```

### Service Updates

```bash
# Service neu builden
docker-compose -f docker-compose.microservices.yml build nhits-service

# Service neu starten
docker-compose -f docker-compose.microservices.yml up -d nhits-service
```

## Volumes

### Persistente Daten

```bash
# Volumes anzeigen
docker volume ls | grep trading

# Volume inspizieren
docker volume inspect ki-trading-model_models-data

# Volume Backup
docker run --rm \
  -v ki-trading-model_models-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz /data
```

## Monitoring

### Resource Usage

```bash
# CPU/Memory pro Service
docker stats

# GPU Usage
nvidia-smi

# Disk Usage
docker system df
```

### Health Checks

```bash
# Frontend
wget -qO- http://localhost:3000/health

# NHITS
curl -s http://localhost:3001/health | jq

# LLM
curl -s http://localhost:3002/health | jq

# Data
curl -s http://localhost:3003/health | jq
```

## Troubleshooting

### Problem: Service startet nicht

```bash
# Logs prüfen
docker-compose -f docker-compose.microservices.yml logs nhits-service

# Container inspizieren
docker inspect trading-nhits

# In Container einsteigen
docker exec -it trading-nhits /bin/bash
```

### Problem: GPU nicht verfügbar

```bash
# NVIDIA Runtime prüfen
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Container GPU Zugriff prüfen
docker exec trading-nhits nvidia-smi
```

### Problem: Service nicht erreichbar

```bash
# Port Binding prüfen
netstat -tulpn | grep 3001

# Network prüfen
docker network inspect ki-trading-model_trading-net

# Service Connectivity
docker exec trading-frontend ping nhits-service
```

## Migration vom Monolith

### 1. Backup erstellen

```bash
# Aktuellen Container stoppen
docker-compose -f docker/jetson/docker-compose.yml down

# Volumes sichern
docker run --rm \
  -v $(pwd):/backup \
  alpine tar czf /backup/full-backup.tar.gz /app/data
```

### 2. Microservices starten

```bash
# Environment kopieren
cp .env .env.microservices

# Services starten
docker-compose -f docker-compose.microservices.yml up -d
```

### 3. Daten migrieren

```bash
# Models kopieren
docker cp /app/data/models/. trading-nhits:/app/data/models/

# RAG Daten kopieren
docker cp /app/data/rag/. trading-llm:/app/data/rag/
```

### 4. Testen

```bash
# Training starten
curl -X POST "http://localhost:3001/api/v1/forecast/train-all?force=true"

# Forecast abrufen
curl "http://localhost:3001/api/v1/forecast/EURUSD"

# LLM Analysis
curl -X POST "http://localhost:3002/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD"}'
```

## Performance Tuning

### GPU Memory Management

**NHITS Service:**
```python
# In nhits_app/main.py
import torch
torch.cuda.set_per_process_memory_fraction(0.4, device=0)
```

**LLM Service:**
```python
# In llm_app/main.py
import torch
torch.cuda.set_per_process_memory_fraction(0.6, device=0)
```

### Nginx Tuning

```nginx
# In nginx.conf
worker_processes auto;
worker_connections 2048;

# Timeouts
proxy_read_timeout 600s;
proxy_connect_timeout 600s;
```

### Docker Resource Limits

```yaml
# In docker-compose.microservices.yml
services:
  nhits-service:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
```

## Rollback

### Zu Monolith zurückkehren

```bash
# Microservices stoppen
docker-compose -f docker-compose.microservices.yml down

# Monolith starten
docker-compose -f docker/jetson/docker-compose.yml up -d

# Volumes wiederherstellen (falls nötig)
tar xzf full-backup.tar.gz -C /app/data
```

## Security

### Netzwerk-Isolation

```yaml
# Nur Frontend exponieren
services:
  frontend:
    ports:
      - "3000:80"

  nhits-service:
    # Keine externe Ports
    expose:
      - "3001"
```

### Secrets Management

```bash
# Docker Secrets verwenden
echo "your_password" | docker secret create timescale_password -

# In docker-compose.yml
services:
  data-service:
    secrets:
      - timescale_password
    environment:
      - TIMESCALE_PASSWORD_FILE=/run/secrets/timescale_password
```

## Best Practices

### 1. Health Checks aktivieren

Alle Services haben Health Checks - verwenden Sie sie!

```bash
# Auto-restart bei Failure
restart: unless-stopped

healthcheck:
  interval: 30s
  timeout: 10s
  retries: 3
```

### 2. Logs zentralisieren

```bash
# Loki + Grafana Stack
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Backups automatisieren

```bash
# Cronjob für tägliche Backups
0 2 * * * /path/to/backup-script.sh
```

### 4. Updates testen

```bash
# Staging Environment
docker-compose -f docker-compose.staging.yml up -d

# Nach Test: Production Update
docker-compose -f docker-compose.microservices.yml up -d
```

## Support

Bei Problemen:
1. Logs prüfen: `docker-compose logs -f <service>`
2. Health Check: `curl http://localhost:<port>/health`
3. Documentation: `/docs/MICROSERVICES_ARCHITECTURE.md`
4. Issues: GitHub Issues

## Zusammenfassung

**Deployment:**
```bash
docker-compose -f docker-compose.microservices.yml up -d
```

**Monitoring:**
```bash
docker-compose ps
curl http://localhost:3000/health
```

**Updates:**
```bash
docker-compose -f docker-compose.microservices.yml build <service>
docker-compose -f docker-compose.microservices.yml up -d <service>
```

Die Microservices-Architektur ist nun produktionsbereit! 🚀
