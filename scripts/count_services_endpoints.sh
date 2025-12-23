#!/bin/bash
# =============================================================================
# Count Services and API Endpoints for Dashboard Header Display
# Output: "Services: X/8" and "Endpoints: Y"
# =============================================================================

HOST="${TEST_HOST:-localhost}"

# Service ports
PORTS=(3000 3001 3002 3003 3004 3005 3008 3009)
TOTAL_SERVICES=8

# Count healthy services
HEALTHY=0
for port in "${PORTS[@]}"; do
    status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 --max-time 3 "http://${HOST}:${port}/health" 2>/dev/null)
    if [ "$status" = "200" ]; then
        ((HEALTHY++))
    fi
done

# Fixed endpoint count based on OpenAPI specs (pre-calculated)
# Data: 45, NHITS: 12, TCN: 8, HMM: 6, Embedder: 5, RAG: 15, LLM: 8 = ~99 endpoints
ENDPOINTS=99

echo "Services: ${HEALTHY}/${TOTAL_SERVICES}"
echo "Endpoints: ${ENDPOINTS}"
