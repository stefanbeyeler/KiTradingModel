#!/bin/bash
# Generate coverage report for KI Trading Model

set -e

cd "$(dirname "$0")/../.."

echo "Generating coverage report..."
echo ""

# Run tests with coverage
pytest tests/unit tests/api \
    -v \
    --cov=src \
    --cov-report=html \
    --cov-report=xml \
    --cov-report=term-missing \
    --timeout=300

echo ""
echo "Coverage reports generated:"
echo "  - HTML: htmlcov/index.html"
echo "  - XML:  coverage.xml"
echo ""

# Try to open HTML report (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    open htmlcov/index.html 2>/dev/null || true
fi
