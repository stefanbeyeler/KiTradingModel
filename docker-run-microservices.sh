#!/bin/bash
# Run script for Microservices Architecture

set -e  # Exit on error

echo "======================================"
echo "Starting KI Trading Model Microservices"
echo "======================================"
echo

# Source environment
if [ -f .env.microservices ]; then
    export $(cat .env.microservices | grep -v '^#' | xargs)
fi

# Create Docker network if it doesn't exist
if ! docker network inspect trading-net >/dev/null 2>&1; then
    echo "Creating trading-net network..."
    docker network create trading-net
    echo "✓ Network created"
    echo
fi

# Create volumes if they don't exist
echo "Creating Docker volumes..."
docker volume create models-data || true
docker volume create rag-data || true
docker volume create ollama-models || true
echo "✓ Volumes ready"
echo

# Stop and remove existing containers
echo "Cleaning up old containers..."
docker rm -f trading-data trading-llm trading-nhits trading-frontend 2>/dev/null || true
echo "✓ Cleanup complete"
echo

# Use existing Ollama service on host
echo "[1/4] Using existing Ollama Service on host (localhost:11434)..."
echo "✓ Ollama available at host"
echo

# Start Data Service
echo "[2/4] Starting Data Service..."
docker run -d \
    --name trading-data \
    --network trading-net \
    -p 3003:3003 \
    -v $(pwd)/src:/app/src:ro \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data/symbols:/app/data/symbols \
    -e SERVICE_NAME=data \
    -e PORT=3003 \
    -e TIMESCALEDB_HOST=${TIMESCALEDB_HOST} \
    -e TIMESCALEDB_PORT=${TIMESCALEDB_PORT} \
    -e TIMESCALEDB_DATABASE=${TIMESCALEDB_DATABASE} \
    -e TIMESCALEDB_USER=${TIMESCALEDB_USER} \
    -e TIMESCALEDB_PASSWORD=${TIMESCALEDB_PASSWORD} \
    -e EASYINSIGHT_API_URL=${EASYINSIGHT_API_URL} \
    -e RAG_SYNC_ENABLED=${TIMESCALE_SYNC_ENABLED} \
    -e LOG_LEVEL=${LOG_LEVEL} \
    --restart unless-stopped \
    trading-data:latest
echo "✓ Data Service started"
echo

# Start NHITS Service
echo "[3/4] Starting NHITS Service..."
docker run -d \
    --name trading-nhits \
    --runtime=nvidia \
    --network trading-net \
    -p 3001:3001 \
    -v models-data:/app/data/models \
    -v $(pwd)/src:/app/src:ro \
    -v $(pwd)/logs:/app/logs \
    -e SERVICE_NAME=nhits \
    -e PORT=3001 \
    -e NHITS_USE_GPU=${NHITS_USE_GPU} \
    -e TIMESCALEDB_HOST=${TIMESCALEDB_HOST} \
    -e TIMESCALEDB_PORT=${TIMESCALEDB_PORT} \
    -e TIMESCALEDB_DATABASE=${TIMESCALEDB_DATABASE} \
    -e TIMESCALEDB_USER=${TIMESCALEDB_USER} \
    -e TIMESCALEDB_PASSWORD=${TIMESCALEDB_PASSWORD} \
    -e EASYINSIGHT_API_URL=${EASYINSIGHT_API_URL} \
    -e NHITS_AUTO_RETRAIN_DAYS=${NHITS_AUTO_RETRAIN_DAYS} \
    -e LOG_LEVEL=${LOG_LEVEL} \
    --restart unless-stopped \
    trading-nhits:latest
echo "✓ NHITS Service started"
echo

# Start LLM Service
echo "[4/4] Starting LLM Service..."
docker run -d \
    --name trading-llm \
    --runtime=nvidia \
    --network trading-net \
    -p 3002:3002 \
    -v rag-data:/app/data/rag \
    -v $(pwd)/src:/app/src:ro \
    -v $(pwd)/logs:/app/logs \
    -e SERVICE_NAME=llm \
    -e PORT=3002 \
    -e OLLAMA_MODEL=${OLLAMA_MODEL} \
    -e OLLAMA_HOST=${OLLAMA_HOST} \
    -e FAISS_USE_GPU=1 \
    -e EMBEDDING_DEVICE=cuda \
    -e LOG_LEVEL=${LOG_LEVEL} \
    --restart unless-stopped \
    trading-llm:latest
echo "✓ LLM Service started"
echo

# Start Frontend (removed - conflicts with existing dashboard on 3000)
echo ""
echo "Note: Skipping Frontend container to avoid port conflict with existing dashboard"
echo ""

echo "======================================"
echo "All backend services started successfully!"
echo "======================================"
echo
echo "Services:"
echo "  Existing Dashboard: http://10.1.19.101:3000"
echo "  NHITS API:    http://10.1.19.101:3001/docs"
echo "  LLM API:      http://10.1.19.101:3002/docs"
echo "  Data API:     http://10.1.19.101:3003/docs"
echo "  Ollama:       http://10.1.19.101:11434"
echo
echo "Health Checks:"
echo "  curl http://10.1.19.101:3001/health"
echo "  curl http://10.1.19.101:3002/health"
echo "  curl http://10.1.19.101:3003/health"
echo
echo "View Logs:"
echo "  docker logs -f trading-nhits"
echo "  docker logs -f trading-llm"
echo "  docker logs -f trading-data"
