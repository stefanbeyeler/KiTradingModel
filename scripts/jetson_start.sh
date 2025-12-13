#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR/docker/jetson"

BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/l4t-pytorch:latest}

echo "Using BASE_IMAGE=$BASE_IMAGE"

echo "Building and starting container via docker compose..."
docker compose -f docker-compose.yml build --pull --no-cache --build-arg BASE_IMAGE="$BASE_IMAGE"
docker compose -f docker-compose.yml up -d

echo "Waiting for service to become healthy (logs follow)..."
sleep 2
docker compose -f docker-compose.yml logs --tail=50 --follow
