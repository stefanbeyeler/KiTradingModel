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

```
┌─────────────────────────────────────────────────────────────┐
│                  Port 3000 - Frontend                        │
│              Nginx API Gateway + Dashboard                   │
└──────────────┬────────────────┬──────────────┬──────────────┘
               │                │              │
       ┌───────▼─────┐  ┌───────▼──────┐  ┌───▼──────────┐
       │  Port 3001  │  │  Port 3002   │  │  Port 3003   │
       │   NHITS     │  │     LLM      │  │     Data     │
       │  Service    │  │   Service    │  │   Service    │
       └─────────────┘  └──────────────┘  └──────────────┘
        Training &       Analysis &        Symbols &
        Forecasting      RAG              Sync
```

## Services

### Frontend (Port 3000)
- **URL**: http://10.1.19.101:3000
- **Function**: API Gateway, Web Dashboard
- **Tech**: Nginx
- **GPU**: No

### NHITS Service (Port 3001)
- **URL**: http://10.1.19.101:3001
- **Docs**: http://10.1.19.101:3001/docs
- **Function**: NHITS Training & Price Forecasting
- **Tech**: FastAPI, PyTorch
- **GPU**: Yes (CUDA)

### LLM Service (Port 3002)
- **URL**: http://10.1.19.101:3002
- **Docs**: http://10.1.19.101:3002/docs
- **Function**: Trading Analysis & RAG
- **Tech**: FastAPI, Ollama (llama3.1:70b)
- **GPU**: Yes (CUDA)

### Data Service (Port 3003)
- **URL**: http://10.1.19.101:3003
- **Docs**: http://10.1.19.101:3003/docs
- **Function**: Symbol Management, Strategies, Sync
- **Tech**: FastAPI, TimescaleDB
- **GPU**: No

## Environment Setup

Create `.env` file:

```bash
# TimescaleDB
TIMESCALE_HOST=10.1.19.104
TIMESCALE_PORT=5432
TIMESCALE_DB=trading
TIMESCALE_USER=postgres
TIMESCALE_PASSWORD=your_password

# EasyInsight API
EASYINSIGHT_API_URL=http://10.1.19.102:3000

# LLM
OLLAMA_MODEL=llama3.1:70b

# NHITS
NHITS_AUTO_RETRAIN_DAYS=7
NHITS_USE_GPU=1
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
curl http://localhost:3000/health  # Frontend
curl http://localhost:3001/health  # NHITS
curl http://localhost:3002/health  # LLM
curl http://localhost:3003/health  # Data
```

## API Examples

### NHITS - Start Training
```bash
curl -X POST "http://localhost:3001/api/v1/forecast/train-all?force=true&background=true"
```

### NHITS - Get Forecast
```bash
curl "http://localhost:3001/api/v1/forecast/EURUSD?horizon=24"
```

### NHITS - Training Progress
```bash
curl "http://localhost:3001/api/v1/forecast/training/progress"
```

### LLM - Trading Analysis
```bash
curl -X POST "http://localhost:3002/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD", "question": "Should I buy or sell?"}'
```

### Data - List Symbols
```bash
curl "http://localhost:3003/api/v1/symbols"
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
nvidia-smi
```

## Volumes

Persistent data is stored in Docker volumes:

- `models-data`: NHITS trained models
- `rag-data`: LLM knowledge base
- `ollama-models`: LLM model files

### Backup Volumes
```bash
docker run --rm \
  -v ki-trading-model_models-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz /data
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

# Kill process
kill -9 <PID>
```

## Documentation

- [Microservices Architecture](docs/MICROSERVICES_ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_MICROSERVICES.md)
- [Event-Based Training](docs/EVENT_BASED_TRAINING.md)
- [Port Configuration](docs/PORT_CONFIGURATION.md)
- [API Categories](docs/SWAGGER_API_CATEGORIES.md)

## Migration from Monolith

### Backup
```bash
docker-compose -f docker/jetson/docker-compose.yml down
cp -r data/ data-backup/
```

### Deploy Microservices
```bash
docker-compose -f docker-compose.microservices.yml up -d
```

### Migrate Data
```bash
docker cp data/models/. trading-nhits:/app/data/models/
docker cp data/rag/. trading-llm:/app/data/rag/
```

## Performance

### Resource Requirements

| Service | CPU | Memory | GPU | Disk |
|---------|-----|--------|-----|------|
| Frontend | 0.5 cores | 256 MB | No | 100 MB |
| NHITS | 4-8 cores | 16 GB | Yes | 10 GB |
| LLM | 4 cores | 32 GB | Yes | 50 GB |
| Data | 2 cores | 4 GB | No | 5 GB |

### GPU Sharing

NHITS and LLM share GPU resources:
- **NHITS**: 40% GPU memory (training)
- **LLM**: 60% GPU memory (inference)

Adjust in service code:
```python
torch.cuda.set_per_process_memory_fraction(0.4, device=0)  # NHITS
torch.cuda.set_per_process_memory_fraction(0.6, device=0)  # LLM
```

## Security

### Network Isolation
Only frontend is exposed externally. Backend services communicate via internal Docker network.

### Secrets
Use Docker secrets or environment variables for sensitive data.

```bash
echo "password" | docker secret create db_password -
```

## Support

- **Issues**: GitHub Issues
- **Docs**: `/docs/` directory
- **Logs**: `docker-compose logs -f`

## License

Proprietary - KI Trading Model

---

**Built with:**
- FastAPI
- PyTorch (NHITS)
- Ollama (Llama 3.1 70B)
- TimescaleDB
- Docker
- Nginx
