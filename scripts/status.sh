#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "KI Trading Model - System Status"
echo "=========================================="
echo ""

# Check Docker service
echo "🐳 Docker Service:"
if systemctl is-active --quiet docker; then
    echo "  ✅ Docker is running"
else
    echo "  ❌ Docker is NOT running"
    echo "  → Start with: sudo systemctl start docker"
fi
echo ""

# Check Ollama service
echo "🦙 Ollama Service:"
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    VERSION=$(curl -s http://localhost:11434/api/version | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo "  ✅ Ollama is running (version: $VERSION)"
else
    echo "  ❌ Ollama is NOT running"
    echo "  → Check with: systemctl status ollama"
fi
echo ""

# Check TimescaleDB connectivity
echo "🗄️  TimescaleDB (10.1.19.100:5432):"
if timeout 2 bash -c 'cat < /dev/null > /dev/tcp/10.1.19.100/5432' 2>/dev/null; then
    echo "  ✅ TimescaleDB is reachable"
else
    echo "  ❌ TimescaleDB is NOT reachable"
    echo "  → Check network connection and database server"
fi
echo ""

# Check containers
echo "📦 Docker Containers:"
BACKEND_STATUS=$(docker inspect -f '{{.State.Status}}' ki-trading 2>/dev/null || echo "not found")
DASHBOARD_STATUS=$(docker inspect -f '{{.State.Status}}' ki-trading-dashboard 2>/dev/null || echo "not found")

echo "  Backend (ki-trading):"
if [ "$BACKEND_STATUS" = "running" ]; then
    UPTIME=$(docker inspect -f '{{.State.StartedAt}}' ki-trading 2>/dev/null)
    echo "    ✅ Running (started: $UPTIME)"
    echo "    → API: http://localhost:3011/api/v1/"
    echo "    → Docs: http://localhost:3011/docs"
elif [ "$BACKEND_STATUS" = "restarting" ]; then
    echo "    ⚠️  Restarting (check logs)"
    echo "    → Logs: docker logs ki-trading"
elif [ "$BACKEND_STATUS" = "exited" ]; then
    echo "    ❌ Stopped"
    echo "    → Start: docker start ki-trading"
else
    echo "    ❌ Not found"
    echo "    → Create: bash scripts/jetson_start_simple.sh"
fi
echo ""

echo "  Dashboard (ki-trading-dashboard):"
if [ "$DASHBOARD_STATUS" = "running" ]; then
    UPTIME=$(docker inspect -f '{{.State.StartedAt}}' ki-trading-dashboard 2>/dev/null)
    echo "    ✅ Running (started: $UPTIME)"
    echo "    → Dashboard: http://localhost:3001"
elif [ "$DASHBOARD_STATUS" = "restarting" ]; then
    echo "    ⚠️  Restarting (check logs)"
    echo "    → Logs: docker logs ki-trading-dashboard"
elif [ "$DASHBOARD_STATUS" = "exited" ]; then
    echo "    ❌ Stopped"
    echo "    → Start: docker start ki-trading-dashboard"
else
    echo "    ❌ Not found"
    echo "    → Create: bash scripts/dashboard_start.sh"
fi
echo ""

# Quick health check if backend is running
if [ "$BACKEND_STATUS" = "running" ]; then
    echo "🏥 Backend Health Check:"
    HEALTH=$(curl -s http://localhost:3011/api/v1/health 2>/dev/null || echo "failed")
    if [ "$HEALTH" != "failed" ]; then
        echo "  ✅ Backend is healthy"
        echo "  Response: $HEALTH"
    else
        echo "  ❌ Backend health check failed"
        echo "  → Check logs: docker logs ki-trading"
    fi
    echo ""
fi

# Summary
echo "=========================================="
echo "Quick Commands:"
echo "=========================================="
echo "View logs:     docker logs -f ki-trading"
echo "               docker logs -f ki-trading-dashboard"
echo ""
echo "Restart:       docker restart ki-trading"
echo "               docker restart ki-trading-dashboard"
echo ""
echo "Stop all:      docker stop ki-trading ki-trading-dashboard"
echo "Start all:     docker start ki-trading ki-trading-dashboard"
echo ""
echo "Full restart:  bash scripts/start_all.sh"
echo "=========================================="
