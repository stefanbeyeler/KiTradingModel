# KI Trading Model - Microservices Architecture

## Quick Start

```bash
# Build and start all services
docker-compose -f docker-compose.microservices.yml up -d

# Check status
docker-compose -f docker-compose.microservices.yml ps

# View logs
docker-compose -f docker-compose.microservices.yml logs -f
```

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     Frontend Dashboard (Port 3000)                   │
│                     Nginx API Gateway + Web UI                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────┬───────┬───────┼───────┬───────┬───────┬───────┐
        │       │       │       │       │       │       │       │
        ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
     Data    NHITS    TCN     HMM   Embedder  RAG     LLM   Watchdog
    :3001   :3002   :3003   :3004   :3005   :3008   :3009   :3010
```

## Services

### Frontend (Port 3000)

- **URL**: <http://10.1.19.101:3000>
- **Function**: API Gateway, Web Dashboard
- **Tech**: Nginx
- **GPU**: No

### Data Service (Port 3001)

- **URL**: <http://10.1.19.101:3001>
- **Docs**: <http://10.1.19.101:3000/data/docs>
- **Function**: Symbol-Management, Strategien, Daten-Gateway
- **Tech**: FastAPI
- **GPU**: No

### NHITS Service (Port 3002)

- **URL**: <http://10.1.19.101:3002>
- **Docs**: <http://10.1.19.101:3000/nhits/docs>
- **Function**: NHITS Training & Price Forecasting
- **Tech**: FastAPI, PyTorch, NeuralForecast
- **GPU**: Yes (CUDA)

### TCN-Pattern Service (Port 3003)

- **URL**: <http://10.1.19.101:3003>
- **Docs**: <http://10.1.19.101:3000/tcn/docs>
- **Function**: Chart-Pattern-Erkennung (16 Pattern-Typen)
- **Tech**: FastAPI, PyTorch, Temporal Convolutional Networks
- **GPU**: Yes (CUDA)

### HMM-Regime Service (Port 3004)

- **URL**: <http://10.1.19.101:3004>
- **Docs**: <http://10.1.19.101:3000/hmm/docs>
- **Function**: Marktphasen-Erkennung, Signal-Scoring mit LightGBM
- **Tech**: FastAPI, hmmlearn, LightGBM
- **GPU**: No

### Embedder Service (Port 3005)

- **URL**: <http://10.1.19.101:3005>
- **Docs**: <http://10.1.19.101:3000/embedder/docs>
- **Function**: Zentraler Embedding-Service (Text, FinBERT, TimeSeries, Features)
- **Tech**: FastAPI, Sentence-Transformers, FinBERT
- **GPU**: Yes (CUDA)

### RAG Service (Port 3008)

- **URL**: <http://10.1.19.101:3008>
- **Docs**: <http://10.1.19.101:3000/rag/docs>
- **Function**: Vector Search & Knowledge Base
- **Tech**: FastAPI, FAISS
- **GPU**: Yes (CUDA)

### LLM Service (Port 3009)

- **URL**: <http://10.1.19.101:3009>
- **Docs**: <http://10.1.19.101:3000/llm/docs>
- **Function**: Trading Analysis mit RAG
- **Tech**: FastAPI, Ollama (Llama 3.1 70B)
- **GPU**: Yes (CUDA)

### Watchdog Service (Port 3010)

- **URL**: <http://10.1.19.101:3010>
- **Docs**: <http://10.1.19.101:3000/watchdog/docs>
- **Function**: Service-Monitoring, Telegram-Alerts
- **Tech**: FastAPI, httpx
- **GPU**: No

## Environment Setup

Create `.env` file:

```bash
# EasyInsight API
EASYINSIGHT_API_URL=http://10.1.19.102:3000/api

# LLM
OLLAMA_MODEL=llama3.1:70b

# Service Ports (optional)
DATA_SERVICE_PORT=3001
NHITS_SERVICE_PORT=3002
TCN_SERVICE_PORT=3003
HMM_SERVICE_PORT=3004
EMBEDDER_SERVICE_PORT=3005
RAG_SERVICE_PORT=3008
LLM_SERVICE_PORT=3009
WATCHDOG_SERVICE_PORT=3010
```

## Common Commands

### Start Services

```bash
docker-compose -f docker-compose.microservices.yml up -d
```

### Stop Services

```bash
docker-compose -f docker-compose.microservices.yml down
```

### View Logs

```bash
# All services
docker-compose -f docker-compose.microservices.yml logs -f

# Specific service
docker-compose -f docker-compose.microservices.yml logs -f nhits-service
```

### Restart Service

```bash
docker-compose -f docker-compose.microservices.yml restart nhits-service
```

### Health Checks

```bash
# Via Proxy
curl http://10.1.19.101:3000/data/health
curl http://10.1.19.101:3000/nhits/health
curl http://10.1.19.101:3000/tcn/health
curl http://10.1.19.101:3000/hmm/health
curl http://10.1.19.101:3000/embedder/health
curl http://10.1.19.101:3000/rag/health
curl http://10.1.19.101:3000/llm/health
curl http://10.1.19.101:3000/watchdog/health
```

## API Examples

### NHITS - Start Training

```bash
curl -X POST "http://10.1.19.101:3000/nhits/api/v1/forecast/train-all?background=true"
```

### NHITS - Get Forecast

```bash
curl "http://10.1.19.101:3000/nhits/api/v1/forecast/EURUSD?horizon=24"
```

### NHITS - Training Progress

```bash
curl "http://10.1.19.101:3000/nhits/api/v1/forecast/training/progress"
```

### TCN - Pattern Scan

```bash
curl -X POST "http://10.1.19.101:3000/tcn/api/v1/patterns/scan" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD", "timeframes": ["H1"], "min_confidence": 0.7}'
```

### HMM - Regime Detection

```bash
curl "http://10.1.19.101:3000/hmm/api/v1/regime/EURUSD"
```

### LLM - Trading Analysis

```bash
curl -X POST "http://10.1.19.101:3000/llm/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD", "question": "Should I buy or sell?"}'
```

### Watchdog - Service Status

```bash
curl "http://10.1.19.101:3000/watchdog/api/v1/status"
```

### Data - List Symbols

```bash
curl "http://10.1.19.101:3000/data/api/v1/symbols"
```

## Monitoring

### Service Status

```bash
docker-compose -f docker-compose.microservices.yml ps
```

### Resource Usage

```bash
docker stats
```

### GPU Usage

```bash
# On host
nvidia-smi

# Inside container
docker exec trading-nhits nvidia-smi
```

### System Metrics

```bash
curl "http://10.1.19.101:3000/nhits/api/v1/system/metrics"
```

## Volumes

Persistent data is stored in Docker volumes:

| Volume | Description |
|--------|-------------|
| nhits-models | Trained NHITS models |
| tcn-models | Trained TCN models |
| hmm-models | Trained HMM models |
| embedder-models | Embedding models |
| rag-faiss | FAISS vector database |
| symbols-data | Symbol configurations |
| logs-data | Centralized log files |

### Backup Volumes

```bash
docker run --rm \
  -v ki-trading-model_nhits-models:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/nhits-models-backup.tar.gz /data
```

## Troubleshooting

### Service won't start

```bash
# Check logs
docker-compose -f docker-compose.microservices.yml logs nhits-service

# Check container
docker inspect trading-nhits
```

### GPU not available

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check service GPU
docker exec trading-nhits nvidia-smi
```

### Port already in use

```bash
# Find process using port
netstat -tulpn | grep 3001

# Or on macOS
lsof -i :3001
```

## Performance

### Resource Requirements

| Service | CPU | Memory | GPU | Disk |
|---------|-----|--------|-----|------|
| Frontend | 0.5 cores | 256 MB | No | 100 MB |
| Data | 2 cores | 4 GB | No | 5 GB |
| NHITS | 4-8 cores | 16 GB | Yes | 10 GB |
| TCN | 2-4 cores | 8 GB | Yes | 5 GB |
| HMM | 2 cores | 4 GB | No | 2 GB |
| Embedder | 2-4 cores | 12 GB | Yes | 10 GB |
| RAG | 2 cores | 8 GB | Yes | 20 GB |
| LLM | 4 cores | 32 GB | Yes | 50 GB |
| Watchdog | 0.5 cores | 512 MB | No | 100 MB |

### GPU Sharing

All GPU-enabled services share the Jetson AGX Thor GPU (Shared VRAM architecture).

## Security

### Network Isolation

Only frontend is exposed externally. Backend services communicate via internal Docker network `trading-net`.

### Secrets

Use environment variables for sensitive data:

```bash
# .env file
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_IDS=123456789
```

## Documentation

- [Microservices Architecture](./MICROSERVICES_ARCHITECTURE.md)
- [Deployment Guide](./DEPLOYMENT_MICROSERVICES.md)
- [Port Configuration](./PORT_CONFIGURATION.md)
- [Event-Based Training](./EVENT_BASED_TRAINING.md)
- [Watchdog Proposal](./WATCHDOG_PROPOSAL.md)

## License

MIT License

---

**Built with:**

- FastAPI
- PyTorch (NHITS, TCN)
- Ollama (Llama 3.1 70B)
- FAISS
- LightGBM
- Docker
- Nginx
- NVIDIA Jetson AGX Thor
