#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "=========================================="
echo "Starting KI Trading Model - Full Stack"
echo "=========================================="
echo ""

# Start Backend
echo "[1/2] Starting Backend (KI Trading Model API)..."
bash scripts/jetson_start_simple.sh &
BACKEND_PID=$!

# Wait for backend to build and start
echo "Waiting for backend to initialize..."
sleep 5

# Start Frontend
echo ""
echo "[2/2] Starting Frontend (Dashboard)..."
bash scripts/dashboard_start.sh

echo ""
echo "=========================================="
echo "✅ Full Stack Started Successfully!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - Backend API:  http://localhost:3011/api/v1/"
echo "  - API Docs:     http://localhost:3011/docs"
echo "  - Dashboard:    http://localhost:3001"
echo ""
echo "Check status:"
echo "  docker ps"
echo ""
echo "View logs:"
echo "  docker logs -f ki-trading        # Backend"
echo "  docker logs -f ki-trading-dashboard  # Frontend"
echo ""
echo "Stop all:"
echo "  docker stop ki-trading ki-trading-dashboard"
echo ""
