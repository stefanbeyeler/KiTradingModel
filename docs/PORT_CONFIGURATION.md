# Port-Konfiguration

## Übersicht

Das KI Trading Model System verwendet eine Microservices-Architektur mit 9 Services auf verschiedenen Ports.

## Microservices Ports

| Service | Port | Beschreibung | GPU |
|---------|------|--------------|-----|
| Frontend Dashboard | 3000 | Nginx API Gateway + Web UI | - |
| Data Service | 3001 | Symbol-Management, Strategien, Daten-Gateway | - |
| NHITS Service | 3002 | Preisprognosen und Modell-Training | CUDA |
| TCN-Pattern Service | 3003 | Chart-Pattern-Erkennung (16 Typen) | CUDA |
| HMM-Regime Service | 3004 | Marktphasen-Erkennung, Signal-Scoring | - |
| Embedder Service | 3005 | Text/FinBERT/TimeSeries Embeddings | CUDA |
| RAG Service | 3008 | Vector Search & Knowledge Base | CUDA |
| LLM Service | 3009 | Trading-Analyse mit Llama 3.1 | CUDA |
| Watchdog Service | 3010 | Service-Monitoring, Telegram-Alerts | - |

## Architektur-Diagramm

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     Client (Browser)                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                     Port 3000
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Frontend Dashboard (Nginx)                          │
│                  - Static File Serving                              │
│                  - API Gateway / Reverse Proxy                      │
│                  - Routing zu allen Backend-Services                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────┬───────┬───┴───┬───────┬───────┬───────┬───────┐
        │       │       │       │       │       │       │       │
        ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
     Data    NHITS    TCN     HMM   Embedder  RAG     LLM   Watchdog
    :3001   :3002   :3003   :3004   :3005   :3008   :3009   :3010
```

## Nginx Proxy-Pfade

Das Frontend auf Port 3000 leitet alle API-Anfragen automatisch an die jeweiligen Backend-Services weiter:

| Proxy-Pfad | Ziel-Service | Port |
|------------|--------------|------|
| `/data/*` | Data Service | 3001 |
| `/nhits/*` | NHITS Service | 3002 |
| `/tcn/*` | TCN-Pattern Service | 3003 |
| `/hmm/*` | HMM-Regime Service | 3004 |
| `/embedder/*` | Embedder Service | 3005 |
| `/rag/*` | RAG Service | 3008 |
| `/llm/*` | LLM Service | 3009 |
| `/watchdog/*` | Watchdog Service | 3010 |
| `/easyinsight/*` | EasyInsight API | 10.1.19.102:3000 |

## Swagger UI URLs

Alle API-Dokumentationen sind über das Dashboard erreichbar:

| Service | Swagger UI |
|---------|------------|
| Data Service | <http://10.1.19.101:3000/data/docs> |
| NHITS Service | <http://10.1.19.101:3000/nhits/docs> |
| TCN Service | <http://10.1.19.101:3000/tcn/docs> |
| HMM Service | <http://10.1.19.101:3000/hmm/docs> |
| Embedder Service | <http://10.1.19.101:3000/embedder/docs> |
| RAG Service | <http://10.1.19.101:3000/rag/docs> |
| LLM Service | <http://10.1.19.101:3000/llm/docs> |
| Watchdog Service | <http://10.1.19.101:3000/watchdog/docs> |

## Empfohlene Verwendung

### Für interaktive Nutzung

Verwenden Sie **Port 3000** (Frontend Dashboard):

- Swagger UI im Browser öffnen
- "Try it out" für schnelle Tests
- API-Dokumentation durchsuchen
- Visualisierung von Ergebnissen
- Service-Status im Dashboard

**URL**: <http://10.1.19.101:3000>

### Für Automatisierung

Verwenden Sie die direkten Service-Ports oder den Proxy über Port 3000:

```bash
# Über Proxy (empfohlen)
curl "http://10.1.19.101:3000/nhits/api/v1/forecast/EURUSD"

# Direkt zum Service
curl "http://10.1.19.101:3002/api/v1/forecast/EURUSD"
```

## Health Checks

```bash
# Alle Services über Proxy
curl "http://10.1.19.101:3000/data/health"
curl "http://10.1.19.101:3000/nhits/health"
curl "http://10.1.19.101:3000/tcn/health"
curl "http://10.1.19.101:3000/hmm/health"
curl "http://10.1.19.101:3000/embedder/health"
curl "http://10.1.19.101:3000/rag/health"
curl "http://10.1.19.101:3000/llm/health"
curl "http://10.1.19.101:3000/watchdog/health"

# Oder direkt zu den Services
for port in 3001 3002 3003 3004 3005 3008 3009 3010; do
  echo "Port $port: $(curl -s http://10.1.19.101:$port/health | jq -r .status)"
done
```

## Watchdog Service Status

Der Watchdog Service auf Port 3010 überwacht alle anderen Services:

```bash
# Gesamtstatus aller Services
curl "http://10.1.19.101:3000/watchdog/api/v1/status"

# Alert-Historie
curl "http://10.1.19.101:3000/watchdog/api/v1/alerts/history"

# Test-Alert senden
curl -X POST "http://10.1.19.101:3000/watchdog/api/v1/alerts/test"
```

## Troubleshooting

### Problem: "Connection refused" auf einem Port

**Lösung**: Container prüfen

```bash
docker ps | grep trading
docker logs trading-<service>
```

### Problem: API-Calls funktionieren nicht über Port 3000

**Prüfen**: Nginx-Konfiguration

```bash
docker exec trading-frontend cat /etc/nginx/nginx.conf
```

### Problem: Service startet nicht

**Prüfen**: Container-Logs und GPU-Zugriff

```bash
docker logs trading-nhits
docker exec trading-nhits nvidia-smi
```

## Weiterführende Dokumentation

- [Microservices Architektur](./MICROSERVICES_ARCHITECTURE.md)
- [Deployment Guide](./DEPLOYMENT_MICROSERVICES.md)
- [Swagger API Kategorien](./SWAGGER_API_CATEGORIES.md)
- [Watchdog Proposal](./WATCHDOG_PROPOSAL.md)
