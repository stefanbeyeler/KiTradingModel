# Makefile for KI Trading Model
# Test commands via Watchdog Service

.PHONY: test test-smoke test-critical test-api test-integration test-contract test-full test-service health lint clean

# Configuration
WATCHDOG_URL ?= http://10.1.19.101:3010

# Default target
help:
	@echo "Test-Befehle via Watchdog Service"
	@echo "=================================="
	@echo ""
	@echo "Test-Modi:"
	@echo "  make test              - Smoke-Tests (schnelle Health-Checks)"
	@echo "  make test-smoke        - Smoke-Tests starten"
	@echo "  make test-critical     - Nur kritische Tests"
	@echo "  make test-api          - Alle API-Endpoint Tests"
	@echo "  make test-contract     - Schema-Validierung Tests"
	@echo "  make test-integration  - Service-uebergreifende Tests"
	@echo "  make test-full         - Alle Tests (volle Suite)"
	@echo ""
	@echo "Service-spezifisch:"
	@echo "  make test-service SERVICE=data    - Tests fuer Data Service"
	@echo "  make test-service SERVICE=nhits   - Tests fuer NHITS Service"
	@echo ""
	@echo "Status und Info:"
	@echo "  make test-status       - Aktueller Test-Status"
	@echo "  make test-history      - Test-Historie anzeigen"
	@echo "  make test-modes        - Verfuegbare Test-Modi"
	@echo "  make test-summary      - Test-Zusammenfassung"
	@echo "  make health            - Service Health-Check"
	@echo ""
	@echo "Sonstiges:"
	@echo "  make lint              - Code-Linting"
	@echo "  make clean             - Aufraemen"
	@echo ""
	@echo "Konfiguration:"
	@echo "  WATCHDOG_URL=$(WATCHDOG_URL)"

# Quick test (smoke)
test: test-smoke

# Smoke-Tests (Health-Checks)
test-smoke:
	@echo "Starte Smoke-Tests..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/smoke | python -m json.tool
	@echo ""
	@echo "Status abrufen mit: make test-status"

# Critical Tests
test-critical:
	@echo "Starte kritische Tests..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/critical | python -m json.tool
	@echo ""
	@echo "Status abrufen mit: make test-status"

# API Tests
test-api:
	@echo "Starte API-Tests..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run \
		-H "Content-Type: application/json" \
		-d '{"mode": "api"}' | python -m json.tool
	@echo ""
	@echo "Status abrufen mit: make test-status"

# Contract Tests
test-contract:
	@echo "Starte Contract-Tests..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run \
		-H "Content-Type: application/json" \
		-d '{"mode": "contract"}' | python -m json.tool
	@echo ""
	@echo "Status abrufen mit: make test-status"

# Integration Tests
test-integration:
	@echo "Starte Integration-Tests..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run \
		-H "Content-Type: application/json" \
		-d '{"mode": "integration"}' | python -m json.tool
	@echo ""
	@echo "Status abrufen mit: make test-status"

# Full Test Suite
test-full:
	@echo "Starte vollstaendige Test-Suite..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/full | python -m json.tool
	@echo ""
	@echo "Status abrufen mit: make test-status"

# Service-spezifische Tests
test-service:
ifndef SERVICE
	@echo "Fehler: SERVICE nicht angegeben"
	@echo "Verwendung: make test-service SERVICE=data"
	@exit 1
endif
	@echo "Starte Tests fuer Service: $(SERVICE)..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/run/service/$(SERVICE) | python -m json.tool
	@echo ""
	@echo "Status abrufen mit: make test-status"

# Test Status
test-status:
	@curl -s $(WATCHDOG_URL)/api/v1/tests/status | python -m json.tool

# Test History
test-history:
	@curl -s $(WATCHDOG_URL)/api/v1/tests/history?limit=5 | python -m json.tool

# Available Test Modes
test-modes:
	@curl -s $(WATCHDOG_URL)/api/v1/tests/modes | python -m json.tool

# Test Summary
test-summary:
	@curl -s $(WATCHDOG_URL)/api/v1/tests/summary | python -m json.tool

# Test Definitions
test-definitions:
	@curl -s $(WATCHDOG_URL)/api/v1/tests/definitions | python -m json.tool

# Abort running test
test-abort:
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/tests/abort | python -m json.tool

# Health check via Watchdog
health:
	@echo "System Status:"
	@curl -s $(WATCHDOG_URL)/api/v1/status | python -m json.tool

# Manual health check trigger
health-check:
	@echo "Triggere Health-Check..."
	@curl -s -X POST $(WATCHDOG_URL)/api/v1/check | python -m json.tool

# List monitored services
services:
	@curl -s $(WATCHDOG_URL)/api/v1/services | python -m json.tool

# Alert History
alerts:
	@curl -s $(WATCHDOG_URL)/api/v1/alerts/history?limit=20 | python -m json.tool

# Watchdog Config
config:
	@curl -s $(WATCHDOG_URL)/api/v1/config | python -m json.tool

# Lint (requires ruff)
lint:
	@echo "Linting src/..."
	@ruff check src --ignore E501
	@echo "Linting complete"

# Type checking (requires mypy)
typecheck:
	mypy src --ignore-missing-imports

# Clean up
clean:
	rm -rf htmlcov .coverage coverage.xml .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Docker rebuild watchdog
rebuild-watchdog:
	docker-compose -f docker-compose.watchdog.yml build --no-cache
	docker-compose -f docker-compose.watchdog.yml up -d

# Deploy watchdog to server
deploy-watchdog:
	rsync -avz --exclude='.git' --exclude='__pycache__' \
		src/services/watchdog_app/ \
		sbeyeler@10.1.19.101:~/KiTradingModel/src/services/watchdog_app/
	ssh sbeyeler@10.1.19.101 "cd ~/KiTradingModel && docker-compose -f docker-compose.watchdog.yml up -d --build"
