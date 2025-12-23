#!/bin/bash
# =============================================================================
# Service Health Check Script
# Tests all 8 microservices for availability and health status
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HOST="${TEST_HOST:-localhost}"
TIMEOUT=10

# Service definitions: name:port:health_endpoint
SERVICES=(
    "Frontend:3000:/health"
    "Data:3001:/health"
    "NHITS:3002:/health"
    "TCN:3003:/health"
    "HMM:3004:/health"
    "Embedder:3005:/health"
    "RAG:3008:/health"
    "LLM:3009:/health"
)

# Counters
PASSED=0
FAILED=0
TOTAL=${#SERVICES[@]}

echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}   KI Trading Model - Service Health Check${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "Host: ${YELLOW}${HOST}${NC}"
echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo -e "${BLUE}---------------------------------------------${NC}"

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name port endpoint <<< "$service"
    url="http://${HOST}:${port}${endpoint}"

    printf "%-12s " "$name"

    # Make request and capture response
    response=$(curl -s -o /tmp/health_response.txt -w "%{http_code}" \
        --connect-timeout $TIMEOUT \
        --max-time $TIMEOUT \
        "$url" 2>/dev/null) || response="000"

    if [ "$response" = "200" ]; then
        # Check response content
        content=$(cat /tmp/health_response.txt 2>/dev/null || echo "")

        if [ "$name" = "Frontend" ]; then
            # Frontend returns plain text
            if echo "$content" | grep -qi "healthy"; then
                echo -e "[${GREEN}PASS${NC}] Port $port - healthy"
                ((PASSED++))
            else
                echo -e "[${RED}FAIL${NC}] Port $port - unexpected response"
                ((FAILED++))
            fi
        else
            # Backend services return JSON
            status=$(echo "$content" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            if [ "$status" = "healthy" ] || [ "$status" = "ok" ] || [ "$status" = "running" ]; then
                echo -e "[${GREEN}PASS${NC}] Port $port - $status"
                ((PASSED++))
            else
                echo -e "[${YELLOW}WARN${NC}] Port $port - status: $status"
                ((PASSED++))
            fi
        fi
    elif [ "$response" = "000" ]; then
        echo -e "[${RED}FAIL${NC}] Port $port - connection refused"
        ((FAILED++))
    else
        echo -e "[${RED}FAIL${NC}] Port $port - HTTP $response"
        ((FAILED++))
    fi
done

echo ""
echo -e "${BLUE}---------------------------------------------${NC}"
echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC} (Total: $TOTAL)"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Cleanup
rm -f /tmp/health_response.txt

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
