#!/bin/bash
# Run all tests for KI Trading Model

set -e

echo "=========================================="
echo "KI Trading Model - Automated Test Suite"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
FAILED=0

# Change to project root
cd "$(dirname "$0")/../.."

# 1. Smoke Tests (Health Checks)
echo -e "\n${YELLOW}[1/6] Running Smoke Tests...${NC}"
if pytest tests/smoke -v -m smoke --timeout=60; then
    echo -e "${GREEN}Smoke tests passed${NC}"
else
    echo -e "${RED}Smoke tests failed! Services may not be running.${NC}"
    FAILED=1
    # Don't exit - continue with other tests that might work
fi

# 2. Unit Tests
echo -e "\n${YELLOW}[2/6] Running Unit Tests...${NC}"
if pytest tests/unit -v -m unit --timeout=120 --cov=src --cov-report=term-missing; then
    echo -e "${GREEN}Unit tests passed${NC}"
else
    echo -e "${RED}Unit tests failed!${NC}"
    FAILED=1
fi

# 3. API Tests
echo -e "\n${YELLOW}[3/6] Running API Tests...${NC}"
if pytest tests/api -v -m api --timeout=180; then
    echo -e "${GREEN}API tests passed${NC}"
else
    echo -e "${RED}API tests failed!${NC}"
    FAILED=1
fi

# 4. Contract Tests
echo -e "\n${YELLOW}[4/6] Running Contract Tests...${NC}"
if pytest tests/contracts -v -m contract --timeout=120; then
    echo -e "${GREEN}Contract tests passed${NC}"
else
    echo -e "${RED}Contract tests failed!${NC}"
    FAILED=1
fi

# 5. Integration Tests
echo -e "\n${YELLOW}[5/6] Running Integration Tests...${NC}"
if pytest tests/integration -v -m integration --timeout=300; then
    echo -e "${GREEN}Integration tests passed${NC}"
else
    echo -e "${RED}Integration tests failed!${NC}"
    FAILED=1
fi

# 6. E2E Tests (optional, slow)
if [[ "$1" == "--full" ]]; then
    echo -e "\n${YELLOW}[6/6] Running E2E Tests...${NC}"
    if pytest tests/e2e -v -m e2e --timeout=600; then
        echo -e "${GREEN}E2E tests passed${NC}"
    else
        echo -e "${RED}E2E tests failed!${NC}"
        FAILED=1
    fi
else
    echo -e "\n${YELLOW}[6/6] Skipping E2E Tests (use --full to include)${NC}"
fi

# Summary
echo -e "\n=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests completed successfully!${NC}"
    echo "=========================================="
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    echo "=========================================="
    exit 1
fi
