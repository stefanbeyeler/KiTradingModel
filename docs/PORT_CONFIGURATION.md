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

## Port 3001 - Frontend Dashboard

### Beschreibung

Web-basiertes Dashboard mit integrierter Swagger-Dokumentation.

### URLs

- **Dashboard**: `http://10.1.19.101:3001`
- **Swagger UI**: `http://10.1.19.101:3001/docs`

### Verwendung

- Interaktive Nutzung über Browser
- Swagger UI mit "Try it out" Funktionalität
- Visualisierung und Monitoring

### Funktionsweise

Das Frontend auf Port 3001 leitet alle API-Anfragen automatisch an das Backend auf Port 3011 weiter. Dies erfolgt transparent über einen Reverse Proxy oder API Gateway.

```
Browser (Port 3001) → Frontend Dashboard → Backend API (Port 3011)
```

## Empfohlene Verwendung

### Für interaktive Nutzung

Verwenden Sie **Port 3001** (Frontend Dashboard):

- ✅ Swagger UI im Browser öffnen
- ✅ "Try it out" für schnelle Tests
- ✅ API-Dokumentation durchsuchen
- ✅ Visualisierung von Ergebnissen

**URL**: `http://10.1.19.101:3001/docs`

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
                 │                      │
        Port 3001│                      │Port 3011
                 │                      │
                 ▼                      ▼
┌────────────────────────┐  ┌──────────────────────────────┐
│  Frontend Dashboard    │  │      Backend API             │
│  - Swagger UI          │  │      - FastAPI Server        │
│  - Web Interface       │  │      - REST Endpoints        │
│  - Reverse Proxy       │──┤      - Business Logic        │
└────────────────────────┘  └──────────────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │   TimescaleDB    │
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

# Über Frontend (Port 3001)
curl "http://10.1.19.101:3001/api/v1/health"
```

Beide sollten die gleichen Ergebnisse liefern.

## Troubleshooting

### Problem: "Connection refused" auf Port 3011

**Lösung**: Backend-Container prüfen

```bash
docker ps | grep trading
docker logs <container-id>
```

### Problem: "Connection refused" auf Port 3001

**Lösung**: Frontend-Container prüfen

```bash
docker ps | grep dashboard
docker logs <container-id>
```

### Problem: API-Calls funktionieren nicht über Port 3001

**Prüfen**: Proxy-Konfiguration im Frontend

```bash
# Nginx/Proxy-Config prüfen
docker exec <frontend-container> cat /etc/nginx/nginx.conf
```

## Weiterführende Dokumentation

- [Swagger API Kategorien](./SWAGGER_API_CATEGORIES.md)
- [Event-Based Training](./EVENT_BASED_TRAINING.md)
- [NHITS Training Progress API](./NHITS_TRAINING_PROGRESS_API.md)
