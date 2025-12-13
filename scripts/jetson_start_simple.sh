#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "Using local Ubuntu-based Dockerfile for Jetson"

# Stop and remove existing container if it exists
echo "Cleaning up existing container..."
docker stop ki-trading 2>/dev/null || true
docker rm ki-trading 2>/dev/null || true

# Build the image
echo "Building Docker image (this may take 5-10 minutes)..."
docker build \
  -t ki-trading-jetson:latest \
  -f docker/jetson/Dockerfile.local \
  .

# Start the container
echo "Starting container..."
docker run -d \
  --name ki-trading \
  --runtime=nvidia \
  --restart unless-stopped \
  -p 3011:3011 \
  -v "$(pwd)":/app:rw \
  --env-file .env \
  ki-trading-jetson:latest

echo ""
echo "✅ Container started successfully!"
echo ""
echo "View logs with: docker logs -f ki-trading"
echo "Stop with: docker stop ki-trading"
echo ""
echo "Waiting for service to start (logs follow)..."
sleep 2
docker logs -f ki-trading
