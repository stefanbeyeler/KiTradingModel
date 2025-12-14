# Microservices Data Flow & Architecture

**Version**: 2.0.0
**Datum**: 2025-12-14

## Gesamtarchitektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────────┐
│                         External Systems                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────┐         ┌──────────────────────┐          │
│  │  TimescaleDB         │         │  Ollama Service      │          │
│  │  10.1.19.102:5432    │         │  10.1.19.101:11434   │          │
│  │                      │         │                      │          │
│  │  - OHLCV Data        │         │  - Llama 3.1 70B     │          │
│  │  - Symbols           │         │  - LLM Inference     │          │
│  └──────────────────────┘         └──────────────────────┘          │
│           │                                    │                     │
└───────────┼────────────────────────────────────┼─────────────────────┘
            │                                    │
            │                                    │
┌───────────▼────────────────────────────────────▼─────────────────────┐
│                    KI Trading Model - Microservices                   │
│                         Docker Network: trading-net                   │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │              Frontend Dashboard (Port 3000)                  │     │
│  │              Nginx API Gateway + Static HTML                 │     │
│  └────┬──────────────────┬──────────────────┬───────────────────┘     │
│       │                  │                  │                         │
│  ┌────▼─────────┐  ┌────▼─────────┐  ┌────▼─────────┐               │
│  │ NHITS        │  │ LLM          │  │ Data         │               │
│  │ Service      │  │ Service      │  │ Service      │               │
│  │ Port 3001    │  │ Port 3002    │  │ Port 3003    │               │
│  │              │  │              │  │              │               │
│  │ GPU: Yes     │  │ GPU: Yes     │  │ GPU: No      │               │
│  │ 17.1 GB      │  │ 17.1 GB      │  │ 1.39 GB      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Data Service - Zentrale Datenverwaltung

### Aufgaben & Verantwortlichkeiten

Der **Data Service** (Port 3003) ist der zentrale Hub für alle Datenoperationen:

1. **Symbol Management**
   - Verwaltung aller Trading-Symbole (BTCUSD, ETHUSD, etc.)
   - Kategorisierung (Crypto, Forex, Stocks)
   - Status-Tracking (active, inactive)
   - Metadata (base_currency, quote_currency, lot sizes)

2. **TimescaleDB Integration**
   - Direkte Verbindung zur TimescaleDB (10.1.19.102:5432)
   - Abruf von historischen OHLCV-Daten
   - Synchronisation von Symbol-Metadaten
   - Query-Performance-Monitoring

3. **EasyInsight API Integration**
   - Abruf von Symbol-Listen
   - Backup-Datenquelle für OHLCV-Daten
   - Real-time Updates (wenn verfügbar)

4. **RAG Sync Service**
   - Automatische Synchronisation von Trading-Daten → RAG Knowledge Base
   - Batch-Processing von OHLCV-Records
   - Interval-basierte Updates (300 Sekunden)

### Datenfluss: TimescaleDB → Data Service → Microservices

```
┌────────────────────────────────────────────────────────────────────┐
│                        DATEN-PIPELINE                               │
└────────────────────────────────────────────────────────────────────┘

1. SYMBOL DISCOVERY
   ┌─────────────────┐
   │ EasyInsight API │ ──GET /api/symbols──> Verfügbare Symbole
   └─────────────────┘                       (BTCUSD, ETHUSD, etc.)
           │
           ▼
   ┌─────────────────┐
   │  Data Service   │ ──POST /api/v1/managed-symbols──> Symbol speichern
   │  Symbol Manager │
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │ Local Database  │  symbols.json
   │ (File-based)    │  - symbol: "BTCUSD"
   └─────────────────┘  - exchange: "kraken"
                        - interval: "1h"
                        - enabled: true

2. HISTORISCHE DATEN ABRUFEN
   ┌─────────────────┐
   │  NHITS Service  │ ──GET /api/v1/data/BTCUSD──>
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  Data Service   │ ──Query TimescaleDB──>
   │  Data Provider  │
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  TimescaleDB    │  SELECT * FROM ohlcv_1h
   │  10.1.19.102    │  WHERE symbol = 'BTCUSD'
   └─────────────────┘  ORDER BY time DESC
           │                LIMIT 1000
           ▼
   ┌─────────────────────────────────────┐
   │  Pandas DataFrame                   │
   │  ┌─────┬───────┬───────┬───────┬───┤
   │  │time │ open  │ high  │ low   │clo│
   │  ├─────┼───────┼───────┼───────┼───┤
   │  │ t1  │ 42000 │ 42500 │ 41800 │422│
   │  │ t2  │ 42200 │ 42800 │ 42100 │426│
   │  └─────┴───────┴───────┴───────┴───┘
   └─────────────────────────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  NHITS Service  │ ──Training mit historischen Daten──>
   │  Model Training │
   └─────────────────┘

3. RAG SYNC PIPELINE
   ┌─────────────────┐
   │  Data Service   │ ──Every 5 minutes──>
   │  RAG Sync       │
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  TimescaleDB    │  Query neue Records
   │                 │  seit letztem Sync
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  RAG Service    │  ──Embedding generieren──>
   │  (in Data Srv)  │
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  FAISS Index    │  Vector Store
   │  (Persistent)   │  /app/data/rag/
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  LLM Service    │ ──Nutzt RAG für Context──>
   │  Analysis       │
   └─────────────────┘
```

## Service-zu-Service Kommunikation

### 1. NHITS Service → Data Service

**Use Case**: NHITS-Training für BTCUSD

```python
# NHITS Service ruft Daten ab
async def fetch_training_data(symbol: str):
    # Interner API Call über Docker Network
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://trading-data:3003/api/v1/data/{symbol}",
            params={
                "lookback_days": 90,
                "interval": "1h"
            }
        )
        data = response.json()

    # Data Service query TimescaleDB
    # SELECT time, open, high, low, close, volume
    # FROM ohlcv_1h
    # WHERE symbol = 'BTCUSD'
    # AND time >= NOW() - INTERVAL '90 days'
    # ORDER BY time ASC

    return pd.DataFrame(data["candles"])
```

**Response Format**:
```json
{
  "symbol": "BTCUSD",
  "interval": "1h",
  "count": 2160,
  "candles": [
    {
      "time": "2025-11-15T00:00:00Z",
      "open": 42000.50,
      "high": 42500.00,
      "low": 41800.25,
      "close": 42200.75,
      "volume": 1250000.0
    }
  ]
}
```

### 2. LLM Service → Data Service

**Use Case**: Context-Aware Trading Analysis

```python
# LLM Service holt Symbol-Informationen
async def get_symbol_context(symbol: str):
    # Über Docker Network
    response = await client.get(
        f"http://trading-data:3003/api/v1/managed-symbols/{symbol}"
    )

    # Data Service liefert Metadata + Latest Data
    return {
        "symbol_info": response.json(),
        "latest_price": get_latest_from_timescale(symbol),
        "24h_change": calculate_change(symbol)
    }
```

### 3. Frontend → Alle Services

**API Gateway Routing (Nginx)**:

```nginx
# Frontend proxied Requests an Backend Services
location /api/v1/forecast/ {
    proxy_pass http://trading-nhits:3001/api/v1/forecast/;
}

location /api/v1/analyze {
    proxy_pass http://trading-llm:3002/api/v1/analyze;
}

location /api/v1/symbols {
    proxy_pass http://trading-data:3003/api/v1/symbols;
}
```

## Datenbank-Zugriff: TimescaleDB

### Direkte Verbindung (Data Service)

```python
# src/services/timescaledb_sync_service.py

class TimescaleDBSyncService:
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=settings.timescaledb_host,      # 10.1.19.102
            port=settings.timescaledb_port,      # 5432
            database=settings.timescaledb_database,  # easyinsight
            user=settings.timescaledb_user,
            password=settings.timescaledb_password
        )

    async def fetch_ohlcv_data(self, symbol: str, limit: int = 1000):
        query = """
            SELECT time, open, high, low, close, volume
            FROM ohlcv_1h
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT $2
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, limit)
            return [dict(row) for row in rows]
```

### Indirekte Verbindung (NHITS & LLM Services)

NHITS und LLM Services greifen **nicht direkt** auf TimescaleDB zu!

- **Vorteil**: Single Point of Access → Data Service
- **Entkopplung**: Datenbank-Schema-Änderungen betreffen nur Data Service
- **Caching**: Data Service kann Caching implementieren
- **Security**: Nur Data Service benötigt DB-Credentials

## RAG Sync - Automatische Knowledge Base Updates

### Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG SYNC WORKFLOW                         │
└─────────────────────────────────────────────────────────────┘

Timer (5 min)
     │
     ▼
┌────────────────┐
│ Sync Triggered │
└────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. Query TimescaleDB für neue Records   │
│    WHERE time > last_sync_time          │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 2. Batch Processing (100 records/batch) │
│    - Format zu Text                     │
│    - "BTCUSD at 2025-12-14 10:00"      │
│      "Open: 42000, Close: 42200"       │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 3. Generate Embeddings                  │
│    - sentence-transformers/MiniLM       │
│    - 384-dimensional vectors            │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 4. Update FAISS Index                   │
│    - Add to vector database             │
│    - Persist to /app/data/rag/          │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 5. Update last_sync_time                │
│    - Track progress                     │
└─────────────────────────────────────────┘
```

### Implementierung

```python
# Data Service - RAG Sync
class TimescaleDBSyncService:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.last_sync_time = None

    async def _perform_sync(self):
        # 1. Hole neue Daten seit letztem Sync
        new_records = await self.fetch_new_records(
            since=self.last_sync_time,
            limit=100  # Batch size
        )

        if not new_records:
            logger.warning("No new records to sync")
            return

        # 2. Formatiere für RAG
        documents = []
        for record in new_records:
            text = f"""
            Symbol: {record['symbol']}
            Time: {record['time']}
            Open: {record['open']}, High: {record['high']}
            Low: {record['low']}, Close: {record['close']}
            Volume: {record['volume']}
            """
            documents.append({
                "text": text,
                "metadata": {
                    "symbol": record['symbol'],
                    "time": record['time']
                }
            })

        # 3. Update RAG Knowledge Base
        await self.rag_service.add_documents(documents)

        # 4. Track letzten Sync
        self.last_sync_time = new_records[-1]['time']
        logger.info(f"Synced {len(documents)} records to RAG")
```

## API-Endpunkte: Data Service

### Symbol Management

```
GET    /api/v1/managed-symbols          # Liste aller Symbole
POST   /api/v1/managed-symbols          # Neues Symbol hinzufügen
GET    /api/v1/managed-symbols/{symbol} # Symbol-Details
PUT    /api/v1/managed-symbols/{symbol} # Symbol aktualisieren
DELETE /api/v1/managed-symbols/{symbol} # Symbol löschen
```

### Daten-Zugriff

```
GET /api/v1/data/{symbol}
    ?lookback_days=90
    &interval=1h

GET /api/v1/data/{symbol}/latest

GET /api/v1/data/batch
    ?symbols=BTCUSD,ETHUSD
    &lookback_days=30
```

### Sync Service Management

```
GET  /api/v1/sync/status     # Status des Sync Services
POST /api/v1/sync/start      # Sync Service starten
POST /api/v1/sync/stop       # Sync Service stoppen
POST /api/v1/sync/trigger    # Manueller Sync
```

### System Monitoring

```
GET /api/v1/system/stats     # System-Statistiken
GET /api/v1/health           # Health Check
GET /api/v1/query-logs       # Query Performance Logs
```

## Service-Konfiguration

### Environment Variables

```bash
# Data Service (.env.microservices)
TIMESCALEDB_HOST=10.1.19.102
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=easyinsight
TIMESCALEDB_USER=postgres
TIMESCALEDB_PASSWORD=postgres

EASYINSIGHT_API_URL=http://10.1.19.102:3000

RAG_SYNC_ENABLED=true
RAG_SYNC_INTERVAL_SECONDS=300
RAG_SYNC_BATCH_SIZE=100

LOG_LEVEL=INFO
```

### Docker Volumes

```yaml
volumes:
  - ./src:/app/src:ro              # Source Code (read-only)
  - ./logs:/app/logs               # Log Files
  - ./data/symbols:/app/data/symbols  # Symbol Definitions
```

## Performance & Monitoring

### Query Performance

```python
# Data Service trackt alle Queries
@app.middleware("http")
async def log_queries(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(f"Query: {request.url.path} - {duration:.2f}s")

    # Speichere in Query Log
    await save_query_log({
        "endpoint": request.url.path,
        "duration_ms": duration * 1000,
        "status": response.status_code
    })

    return response
```

### Health Checks

```bash
# Data Service Health
curl http://10.1.19.101:3003/health

{
  "service": "data",
  "status": "healthy",
  "version": "1.0.0.0+unknown",
  "timescaledb_host": "10.1.19.102",
  "sync_service_status": "running"
}
```

## Zusammenfassung

### Datenfluss-Hierarchie

1. **TimescaleDB** (Source of Truth)
   - Historische OHLCV-Daten
   - ~13,000+ Records pro Symbol
   - Hypertables für Performance

2. **Data Service** (Data Hub)
   - Zentraler Zugriffspunkt
   - Symbol Management
   - RAG Sync Coordinator
   - Query Monitoring

3. **NHITS Service** (Consumer)
   - Holt Training-Daten via Data Service
   - Kein direkter DB-Zugriff
   - GPU-basiertes Training

4. **LLM Service** (Consumer)
   - Nutzt RAG Knowledge Base
   - Context-aware Analysis
   - Ollama Integration

5. **Frontend** (User Interface)
   - Nginx API Gateway
   - Routes zu allen Services
   - Dashboard & Swagger UI

### Key Design Principles

✅ **Separation of Concerns**: Jeder Service hat eine klare Verantwortung
✅ **Single Source of Truth**: TimescaleDB → Data Service → Consumers
✅ **Loose Coupling**: Services kommunizieren nur via REST APIs
✅ **Scalability**: Jeder Service kann unabhängig skaliert werden
✅ **Resilience**: Service-Ausfall betrifft nur abhängige Services
✅ **Observability**: Logging, Health Checks, Query Monitoring

---

**Erstellt**: 2025-12-14
**System**: KI Trading Model v2.0.0
**Architektur**: Microservices mit Docker
