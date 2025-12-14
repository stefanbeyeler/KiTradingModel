#!/bin/bash
# Build script for Microservices Architecture

set -e  # Exit on error

echo "======================================"
echo "Building KI Trading Model Microservices"
echo "======================================"
echo

# Source environment
if [ -f .env.microservices ]; then
    export $(cat .env.microservices | grep -v '^#' | xargs)
fi

# Build Frontend
echo "[1/4] Building Frontend Service..."
docker build \
    -f docker/services/frontend/Dockerfile \
    -t trading-frontend:latest \
    .
echo "✓ Frontend built successfully"
echo

# Build NHITS Service
echo "[2/4] Building NHITS Service..."
docker build \
    -f docker/services/nhits/Dockerfile \
    -t trading-nhits:latest \
    .
echo "✓ NHITS Service built successfully"
echo

# Build LLM Service
echo "[3/4] Building LLM Service..."
docker build \
    -f docker/services/llm/Dockerfile \
    -t trading-llm:latest \
    .
echo "✓ LLM Service built successfully"
echo

# Build Data Service
echo "[4/4] Building Data Service..."
docker build \
    -f docker/services/data/Dockerfile \
    -t trading-data:latest \
    .
echo "✓ Data Service built successfully"
echo

echo "======================================"
echo "All services built successfully!"
echo "======================================"
echo
echo "Next steps:"
echo "1. Review docker-run-microservices.sh"
echo "2. Run: bash docker-run-microservices.sh"
