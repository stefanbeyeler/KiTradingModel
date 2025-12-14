#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "KI Trading Model - Microservices Status"
echo "=========================================="
echo ""

# Check Docker service
echo "Docker Service:"
if systemctl is-active --quiet docker; then
    echo "  [OK] Docker is running"
else
    echo "  [FAIL] Docker is NOT running"
    echo "  -> Start with: sudo systemctl start docker"
fi
echo ""

# Check Ollama service
echo "Ollama Service:"
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    VERSION=$(curl -s http://localhost:11434/api/version | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo "  [OK] Ollama is running (version: $VERSION)"
else
    echo "  [FAIL] Ollama is NOT running"
    echo "  -> Check with: systemctl status ollama"
fi
echo ""

# Check EasyInsight API connectivity
echo "EasyInsight API (10.1.19.102:3000):"
if curl -s http://10.1.19.102:3000/api/symbols > /dev/null 2>&1; then
    SYMBOL_COUNT=$(curl -s http://10.1.19.102:3000/api/symbols | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
    echo "  [OK] EasyInsight API is reachable ($SYMBOL_COUNT symbols)"
else
    echo "  [FAIL] EasyInsight API is NOT reachable"
fi
echo ""

# Check Microservice containers
echo "Microservice Containers:"
echo ""

check_service() {
    local NAME=$1
    local PORT=$2
    local DISPLAY_NAME=$3

    STATUS=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo "not found")
    HEALTH=$(docker inspect -f '{{.State.Health.Status}}' "$NAME" 2>/dev/null || echo "none")

    echo "  $DISPLAY_NAME ($NAME):"
    if [ "$STATUS" = "running" ]; then
        if [ "$HEALTH" = "healthy" ]; then
            echo "    [OK] Running & Healthy"
        elif [ "$HEALTH" = "unhealthy" ]; then
            echo "    [WARN] Running but Unhealthy"
        else
            echo "    [OK] Running"
        fi
        echo "    -> http://localhost:$PORT"
    elif [ "$STATUS" = "restarting" ]; then
        echo "    [WARN] Restarting"
    elif [ "$STATUS" = "exited" ]; then
        echo "    [FAIL] Stopped"
    else
        echo "    [FAIL] Not found"
    fi
    echo ""
}

check_service "trading-nhits" "3001" "NHITS Service (Training & Forecast)"
check_service "trading-llm" "3002" "LLM Service (Analysis & RAG)"
check_service "trading-data" "3003" "Data Service (Symbols & Sync)"
check_service "trading-frontend" "3000" "Frontend (Dashboard)"

# Health checks
echo "=========================================="
echo "Service Health Checks:"
echo "=========================================="
echo ""

for PORT in 3001 3002 3003; do
    HEALTH=$(curl -s "http://localhost:$PORT/health" 2>/dev/null || echo "")
    if [ -n "$HEALTH" ]; then
        SERVICE=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('service','unknown'))" 2>/dev/null || echo "port-$PORT")
        STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
        echo "  Port $PORT ($SERVICE): $STATUS"
    else
        echo "  Port $PORT: not responding"
    fi
done
echo ""

# Summary
echo "=========================================="
echo "Quick Commands:"
echo "=========================================="
echo "Start all:     docker compose -f docker-compose.microservices.yml up -d"
echo "Stop all:      docker compose -f docker-compose.microservices.yml down"
echo "View logs:     docker logs -f trading-nhits"
echo "               docker logs -f trading-llm"
echo "               docker logs -f trading-data"
echo ""
echo "API Docs:"
echo "  NHITS:       http://localhost:3001/docs"
echo "  LLM:         http://localhost:3002/docs"
echo "  Data:        http://localhost:3003/docs"
echo "  Dashboard:   http://localhost:3000"
echo "=========================================="
