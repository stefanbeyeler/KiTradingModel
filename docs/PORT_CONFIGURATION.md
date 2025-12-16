# Port-Konfiguration

## Übersicht

Das KI Trading Model System läuft auf zwei verschiedenen Ports mit unterschiedlichen Funktionen.

## Port 3011 - Backend API

### Beschreibung

Der FastAPI-Backend-Server mit allen REST API-Endpunkten.

### URLs

- **API Base**: `http://10.1.19.101:3011/api/v1`
- **Swagger UI**: `http://10.1.19.101:3011/docs`
- **ReDoc**: `http://10.1.19.101:3011/redoc`
- **OpenAPI JSON**: `http://10.1.19.101:3011/openapi.json`

### Verwendung

- Direkte API-Aufrufe via `curl`, `Postman`, `httpx`
- Python-Scripts und Automatisierung
- API-Integration in andere Systeme

### Beispiel

```bash
# Training-Status abfragen
curl "http://10.1.19.101:3011/api/v1/forecast/training/status"

# Training starten
curl -X POST "http://10.1.19.101:3011/api/v1/forecast/train-all?force=true&background=true"

# Training-Fortschritt überwachen
curl "http://10.1.19.101:3011/api/v1/forecast/training/progress"
```

## Microservices Ports

| Service | Port | Beschreibung |
|---------|------|--------------|
| Frontend Dashboard | 3000 | Web-basiertes Dashboard mit API Gateway |
| Data Service | 3001 | Symbol management, strategies, sync |
| NHITS Service | 3002 | Training & Forecasting |
| RAG Service | 3003 | Vector Search & Knowledge Base |
| LLM Service | 3004 | Analysis & Recommendations |

### URLs

- **Dashboard**: `http://10.1.19.101:3000`
- **Data API**: `http://10.1.19.101:3001/docs`
- **NHITS API**: `http://10.1.19.101:3002/docs`
- **RAG API**: `http://10.1.19.101:3003/docs`
- **LLM API**: `http://10.1.19.101:3004/docs`

### Funktionsweise

Das Frontend auf Port 3000 leitet alle API-Anfragen automatisch an die jeweiligen Backend-Services weiter. Dies erfolgt transparent über einen Nginx Reverse Proxy.

```
Browser (Port 3000) → Frontend Dashboard → Backend Services (3001-3004)
```

## Empfohlene Verwendung

### Für interaktive Nutzung

Verwenden Sie **Port 3000** (Frontend Dashboard):

- ✅ Swagger UI im Browser öffnen
- ✅ "Try it out" für schnelle Tests
- ✅ API-Dokumentation durchsuchen
- ✅ Visualisierung von Ergebnissen

**URL**: `http://10.1.19.101:3000`

### Für Automatisierung

Verwenden Sie **Port 3011** (Backend API):

- ✅ Scripts und Cronjobs
- ✅ API-Integration
- ✅ Monitoring-Tools
- ✅ CI/CD Pipelines

**URL**: `http://10.1.19.101:3011/api/v1`

## Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────┐
│                     Client (Browser/curl)                    │
└────────────────┬──────────────────────┬─────────────────────┘
                 │                      │
        Port 3000│                      │Ports 3001-3004
                 │                      │
                 ▼                      ▼
┌────────────────────────┐  ┌──────────────────────────────┐
│  Frontend Dashboard    │  │      Microservices           │
│  - Swagger UI          │  │      - Data (3001)           │
│  - Web Interface       │  │      - NHITS (3002)          │
│  - Nginx Reverse Proxy │──┤      - RAG (3003)            │
└────────────────────────┘  │      - LLM (3004)            │
                            └──────────────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │   EasyInsight    │
                            │   Ollama LLM     │
                            └──────────────────┘
```

## Wichtige Hinweise

### CORS und Sicherheit

- Port 3011 ist direkt zugänglich und akzeptiert API-Requests
- Port 3001 bietet eine zusätzliche Abstraktionsebene mit möglicher Authentifizierung
- Beide Ports sollten nur im internen Netzwerk exponiert sein

### Performance

- Direkte Calls an Port 3011 sind minimal schneller (kein Proxy-Overhead)
- Für produktive Systeme empfiehlt sich Port 3001 (Gateway-Funktionen)

### Monitoring

```bash
# Backend-Health-Check (Port 3011)
curl "http://10.1.19.101:3011/api/v1/health"

# Über Frontend (Port 3000)
curl "http://10.1.19.101:3000/api/v1/health"

# Microservices Health-Checks
curl "http://10.1.19.101:3001/health"  # Data
curl "http://10.1.19.101:3002/health"  # NHITS
curl "http://10.1.19.101:3003/health"  # RAG
curl "http://10.1.19.101:3004/health"  # LLM
```

Beide sollten die gleichen Ergebnisse liefern.

## Troubleshooting

### Problem: "Connection refused" auf Port 3011

**Lösung**: Backend-Container prüfen

```bash
docker ps | grep trading
docker logs <container-id>
```

### Problem: "Connection refused" auf Port 3000

**Lösung**: Frontend-Container prüfen

```bash
docker ps | grep frontend
docker logs trading-frontend
```

### Problem: API-Calls funktionieren nicht über Port 3000

**Prüfen**: Proxy-Konfiguration im Frontend

```bash
# Nginx/Proxy-Config prüfen
docker exec trading-frontend cat /etc/nginx/nginx.conf
```

## Weiterführende Dokumentation

- [Swagger API Kategorien](./SWAGGER_API_CATEGORIES.md)
- [Event-Based Training](./EVENT_BASED_TRAINING.md)
- [NHITS Training Progress API](./NHITS_TRAINING_PROGRESS_API.md)
