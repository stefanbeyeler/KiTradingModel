# Makefile for KI Trading Model
# Test and development commands

.PHONY: test test-smoke test-unit test-api test-integration test-e2e test-contract test-all coverage health lint load-test install-test-deps

# Default target
help:
	@echo "Available commands:"
	@echo "  make test              - Run quick tests (smoke + unit)"
	@echo "  make test-smoke        - Run smoke/health check tests"
	@echo "  make test-unit         - Run unit tests with coverage"
	@echo "  make test-api          - Run API endpoint tests"
	@echo "  make test-integration  - Run integration tests"
	@echo "  make test-e2e          - Run end-to-end tests"
	@echo "  make test-contract     - Run contract/schema tests"
	@echo "  make test-all          - Run all tests (full suite)"
	@echo "  make coverage          - Generate coverage report"
	@echo "  make health            - Check all service health endpoints"
	@echo "  make load-test         - Run load tests with Locust"
	@echo "  make lint              - Run linting checks"
	@echo "  make install-test-deps - Install test dependencies"

# Install test dependencies
install-test-deps:
	pip install -r tests/requirements-test.txt

# Quick test (smoke + unit only)
test:
	pytest tests/smoke tests/unit -v --timeout=120

# Test commands
test-smoke:
	pytest tests/smoke -v -m smoke --timeout=60

test-unit:
	pytest tests/unit -v -m unit --cov=src --cov-report=term-missing --timeout=120

test-api:
	pytest tests/api -v -m api --timeout=180

test-integration:
	pytest tests/integration -v -m integration --timeout=300

test-e2e:
	pytest tests/e2e -v -m e2e --timeout=600

test-contract:
	pytest tests/contracts -v -m contract --timeout=120

# Run all tests
test-all:
	./tests/scripts/run_all_tests.sh --full

# Coverage report
coverage:
	pytest tests/unit tests/api -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# Health check
health:
	./tests/scripts/run_health_checks.sh

# Load testing with Locust
load-test:
	locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s

load-test-ui:
	locust -f tests/performance/locustfile.py
	@echo "Open http://localhost:8089 in your browser"

# Lint
lint:
	ruff check src tests
	@echo "Linting complete"

# Type checking
typecheck:
	mypy src --ignore-missing-imports

# Clean up
clean:
	rm -rf htmlcov .coverage coverage.xml .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Docker-related test targets
test-in-docker:
	docker-compose -f docker-compose.microservices.yml up -d
	sleep 60  # Wait for services
	make test-all
	docker-compose -f docker-compose.microservices.yml down
