# Microservices Deployment Guide

## Übersicht

Dieses Deployment-Guide beschreibt das Deployment des KI Trading Model Systems mit 9 Microservices auf NVIDIA Jetson AGX Thor.

## Architektur

```text
Port 3000: Frontend (Nginx + Dashboard)
    ├─> Port 3001: Data Service (Symbols, Strategies, External Data)
    ├─> Port 3002: NHITS Service (Training & Forecasting)
    ├─> Port 3003: TCN-Pattern Service (Chart-Pattern-Erkennung)
    ├─> Port 3004: HMM-Regime Service (Marktphasen, Signal-Scoring)
    ├─> Port 3005: Embedder Service (Text/FinBERT/TimeSeries)
    ├─> Port 3008: RAG Service (Vector Search, Knowledge Base)
    ├─> Port 3009: LLM Service (Trading-Analyse mit Llama 3.1)
    └─> Port 3010: Watchdog Service (Monitoring, Telegram-Alerts)
```

## Services

| Service | Port | GPU | Memory | Funktion |
|---------|------|-----|--------|----------|
| Frontend | 3000 | - | 256 MB | API Gateway + Dashboard |
| Data Service | 3001 | - | 4 GB | Symbole, Strategien, Daten-Gateway |
| NHITS Service | 3002 | CUDA | 16 GB | Preisprognosen, Training |
| TCN-Pattern | 3003 | CUDA | 8 GB | Chart-Pattern-Erkennung |
| HMM-Regime | 3004 | - | 4 GB | Marktphasen, Signal-Scoring |
| Embedder | 3005 | CUDA | 12 GB | Embeddings |
| RAG Service | 3008 | CUDA | 8 GB | Vector Search |
| LLM Service | 3009 | CUDA | 32 GB | Trading-Analyse |
| Watchdog | 3010 | - | 512 MB | Monitoring, Alerts |

## Voraussetzungen

```bash
# Docker & Docker Compose
docker --version        # >= 24.0
docker compose version  # >= 2.20

# NVIDIA Container Toolkit
nvidia-ctk --version

# GPU verfügbar
nvidia-smi
```

## Environment Variables

Erstellen Sie eine `.env` Datei:

```bash
# EasyInsight API
EASYINSIGHT_API_URL=http://10.1.19.102:3000/api

# LLM (Ollama auf Host)
OLLAMA_MODEL=llama3.1:70b
OLLAMA_BASE_URL=http://host.docker.internal:11434

# NHITS
NHITS_AUTO_RETRAIN_DAYS=7
NHITS_USE_GPU=1

# Watchdog Telegram (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_IDS=123456789
```

## Deployment

### Alle Services starten

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

# Health Checks über Proxy
curl http://10.1.19.101:3000/data/health
curl http://10.1.19.101:3000/nhits/health
curl http://10.1.19.101:3000/tcn/health
curl http://10.1.19.101:3000/hmm/health
curl http://10.1.19.101:3000/embedder/health
curl http://10.1.19.101:3000/rag/health
curl http://10.1.19.101:3000/llm/health
curl http://10.1.19.101:3000/watchdog/health
```

### Watchdog Status

```bash
# Gesamtstatus aller Services
curl http://10.1.19.101:3000/watchdog/api/v1/status

# Test-Alert senden
curl -X POST http://10.1.19.101:3000/watchdog/api/v1/alerts/test
```

## Service Management

### Einzelnen Service neu starten

```bash
docker-compose -f docker-compose.microservices.yml restart nhits-service
docker-compose -f docker-compose.microservices.yml restart llm-service
docker-compose -f docker-compose.microservices.yml restart watchdog-service
```

### Logs eines Services

```bash
docker-compose -f docker-compose.microservices.yml logs -f nhits-service
docker-compose -f docker-compose.microservices.yml logs -f watchdog-service
```

### Service Updates

```bash
# Service neu builden
docker-compose -f docker-compose.microservices.yml build nhits-service

# Service neu starten (Zero-Downtime)
docker-compose -f docker-compose.microservices.yml up -d nhits-service
```

## Volumes

### Persistente Daten

| Volume | Service | Beschreibung |
|--------|---------|--------------|
| nhits-models | NHITS | Trainierte Prognose-Modelle |
| tcn-models | TCN | Trainierte Pattern-Modelle |
| hmm-models | HMM | HMM/LightGBM Modelle |
| embedder-models | Embedder | Embedding-Modelle Cache |
| rag-faiss | RAG | FAISS Vector-Datenbank |
| symbols-data | Data | Symbol-Konfigurationen |
| logs-data | Alle | Zentrale Log-Dateien |

### Volume Backup

```bash
# Volumes anzeigen
docker volume ls | grep trading

# Backup erstellen
docker run --rm \
  -v ki-trading-model_nhits-models:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/nhits-models-backup.tar.gz /data

# Backup wiederherstellen
docker run --rm \
  -v ki-trading-model_nhits-models:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/nhits-models-backup.tar.gz -C /
```

## Monitoring

### Resource Usage

```bash
# CPU/Memory pro Container
docker stats

# GPU Usage
nvidia-smi

# GPU Usage im Container
docker exec trading-nhits nvidia-smi

# Disk Usage
docker system df
```

### Watchdog Monitoring

Der Watchdog Service überwacht alle anderen Services automatisch:

```bash
# Status-API
curl http://10.1.19.101:3000/watchdog/api/v1/status | jq

# Alert-Historie
curl http://10.1.19.101:3000/watchdog/api/v1/alerts/history | jq
```

### System-Metriken

```bash
# NHITS System-Metriken
curl http://10.1.19.101:3000/nhits/api/v1/system/metrics | jq
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
ss -tlnp | grep 3001

# Network prüfen
docker network inspect trading-net

# Service Connectivity
docker exec trading-frontend ping trading-nhits
```

### Problem: Watchdog sendet keine Alerts

```bash
# Telegram-Konfiguration prüfen
docker exec trading-watchdog env | grep TELEGRAM

# Manueller Test-Alert
curl -X POST http://10.1.19.101:3000/watchdog/api/v1/alerts/test
```

## Performance Tuning

### GPU Memory Management (Shared VRAM)

```python
# In jedem GPU-Service
import torch
torch.cuda.set_per_process_memory_fraction(0.2, device=0)  # 20% pro Service
```

### Nginx Tuning

```nginx
# In nginx.conf
worker_processes auto;
worker_connections 2048;

# Timeouts für LLM (längere Antworten)
proxy_read_timeout 600s;
proxy_connect_timeout 60s;
```

### Docker Resource Limits

```yaml
services:
  nhits-service:
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
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
    # Keine externe Ports - nur internes Netzwerk
    expose:
      - "3002"
    networks:
      - trading-net
```

### Secrets Management

```bash
# Docker Secrets verwenden
echo "your_telegram_token" | docker secret create telegram_token -

# In docker-compose.yml
services:
  watchdog-service:
    secrets:
      - telegram_token
    environment:
      - TELEGRAM_BOT_TOKEN_FILE=/run/secrets/telegram_token
```

## Best Practices

### 1. Health Checks aktivieren

```yaml
services:
  nhits-service:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

### 2. Watchdog Telegram konfigurieren

```bash
# 1. Bot bei @BotFather erstellen
# 2. Chat-ID ermitteln
curl https://api.telegram.org/bot<TOKEN>/getUpdates

# 3. In .env.watchdog
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_IDS=123456789,987654321
CHECK_INTERVAL_SECONDS=30
ALERT_COOLDOWN_MINUTES=15
```

### 3. Backups automatisieren

```bash
# Cronjob für tägliche Backups
0 2 * * * /home/user/scripts/backup-volumes.sh
```

### 4. Log-Rotation

```yaml
services:
  nhits-service:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

## Swagger UI Zugriff

Alle API-Dokumentationen über das Dashboard:

| Service | Swagger UI |
|---------|------------|
| Data | <http://10.1.19.101:3000/data/docs> |
| NHITS | <http://10.1.19.101:3000/nhits/docs> |
| TCN | <http://10.1.19.101:3000/tcn/docs> |
| HMM | <http://10.1.19.101:3000/hmm/docs> |
| Embedder | <http://10.1.19.101:3000/embedder/docs> |
| RAG | <http://10.1.19.101:3000/rag/docs> |
| LLM | <http://10.1.19.101:3000/llm/docs> |
| Watchdog | <http://10.1.19.101:3000/watchdog/docs> |

## Quick Reference

### Deployment

```bash
docker-compose -f docker-compose.microservices.yml up -d
```

### Status

```bash
docker-compose -f docker-compose.microservices.yml ps
curl http://10.1.19.101:3000/watchdog/api/v1/status
```

### Logs

```bash
docker-compose -f docker-compose.microservices.yml logs -f
```

### Restart

```bash
docker-compose -f docker-compose.microservices.yml restart <service>
```

### Stop

```bash
docker-compose -f docker-compose.microservices.yml down
```

## Weiterführende Dokumentation

- [Microservices Architektur](./MICROSERVICES_ARCHITECTURE.md)
- [Port-Konfiguration](./PORT_CONFIGURATION.md)
- [Swagger API Kategorien](./SWAGGER_API_CATEGORIES.md)
- [Watchdog Proposal](./WATCHDOG_PROPOSAL.md)
