# Testing im KI Trading Model

Die komplette Testing-Funktionalität wurde in den **Watchdog Service** verschoben.

## Test-Ausführung via Watchdog API

### Swagger UI
```
http://10.1.19.101:3000/watchdog/docs
```

### Verfügbare Test-Modi

| Modus | Beschreibung | Endpoint |
|-------|--------------|----------|
| **smoke** | Schnelle Health-Checks | `POST /api/v1/tests/run/smoke` |
| **critical** | Nur kritische Tests | `POST /api/v1/tests/run/critical` |
| **api** | Alle API-Endpoint Tests | `POST /api/v1/tests/run {"mode": "api"}` |
| **contract** | Schema-Validierung | `POST /api/v1/tests/run {"mode": "contract"}` |
| **integration** | Service-übergreifend | `POST /api/v1/tests/run {"mode": "integration"}` |
| **full** | Alle Tests | `POST /api/v1/tests/run/full` |

### Beispiele

```bash
# Smoke-Tests starten
curl -X POST http://10.1.19.101:3010/api/v1/tests/run/smoke

# Vollständige Tests starten
curl -X POST http://10.1.19.101:3010/api/v1/tests/run/full

# Tests für bestimmten Service
curl -X POST http://10.1.19.101:3010/api/v1/tests/run/service/data

# Status abfragen
curl http://10.1.19.101:3010/api/v1/tests/status

# Test-Historie
curl http://10.1.19.101:3010/api/v1/tests/history

# Verfügbare Test-Definitionen
curl http://10.1.19.101:3010/api/v1/tests/definitions
```

### Filter-Optionen

```bash
# Tests mit Filtern
curl -X POST http://10.1.19.101:3010/api/v1/tests/run \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "full",
    "services": ["data", "nhits"],
    "priorities": ["critical", "high"]
  }'
```

## Live-Updates via SSE

```bash
# Server-Sent Events Stream
curl http://10.1.19.101:3010/api/v1/tests/stream
```

## Test-Kategorien

- **smoke**: Health-Checks für alle Services
- **api**: Vollständige API-Endpoint Tests
- **contract**: Response-Schema Validierung
- **integration**: Service-übergreifende Workflows

## Implementierung

Die Test-Logik befindet sich in:
- `src/services/watchdog_app/services/test_definitions.py` - Test-Definitionen
- `src/services/watchdog_app/services/test_runner.py` - Test-Ausführung
- `src/services/watchdog_app/api/routes.py` - API-Endpoints
