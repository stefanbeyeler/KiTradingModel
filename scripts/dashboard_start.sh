#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "Starting KI Trading Dashboard..."

# Stop and remove existing container if it exists
echo "Cleaning up existing dashboard container..."
docker stop ki-trading-dashboard 2>/dev/null || true
docker rm ki-trading-dashboard 2>/dev/null || true

# Build the dashboard image
echo "Building dashboard Docker image..."
docker build \
  -t ki-trading-dashboard:latest \
  -f dashboard/Dockerfile \
  .

# Start the dashboard container
echo "Starting dashboard container..."
docker run -d \
  --name ki-trading-dashboard \
  --restart unless-stopped \
  -p 3001:80 \
  --add-host=host.docker.internal:host-gateway \
  ki-trading-dashboard:latest

echo ""
echo "✅ Dashboard started successfully!"
echo ""
echo "Access the dashboard at: http://localhost:3001"
echo "View logs with: docker logs -f ki-trading-dashboard"
echo "Stop with: docker stop ki-trading-dashboard"
echo ""
echo "Dashboard is proxying API requests to backend at localhost:3011"
echo ""
