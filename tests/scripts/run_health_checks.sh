#!/bin/bash
# Health check script for all KI Trading services

echo "Checking all service health endpoints..."
echo ""

# Get host from environment or use localhost
HOST="${TEST_SERVICE_HOST:-localhost}"

# Define services
declare -A services=(
    ["Frontend"]="http://${HOST}:3000/health"
    ["Data"]="http://${HOST}:3001/health"
    ["NHITS"]="http://${HOST}:3002/health"
    ["TCN"]="http://${HOST}:3003/health"
    ["HMM"]="http://${HOST}:3004/health"
    ["Embedder"]="http://${HOST}:3005/health"
    ["RAG"]="http://${HOST}:3008/health"
    ["LLM"]="http://${HOST}:3009/health"
)

all_healthy=true

for name in "${!services[@]}"; do
    url="${services[$name]}"
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" --connect-timeout 5 2>/dev/null)

    if [ "$response" == "200" ]; then
        echo "✅ $name: healthy"
    else
        echo "❌ $name: unhealthy (HTTP $response)"
        all_healthy=false
    fi
done

echo ""
if $all_healthy; then
    echo "✅ All services are healthy!"
    exit 0
else
    echo "❌ Some services are unhealthy!"
    exit 1
fi
