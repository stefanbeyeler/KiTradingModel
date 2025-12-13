#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR/docker/jetson"

echo "Stopping and removing containers..."
docker compose -f docker-compose.yml down
