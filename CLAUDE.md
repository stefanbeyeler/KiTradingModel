# KI Trading Model - Claude Code Anweisungen

Dieses Dokument enthält projektspezifische Anweisungen für Claude Code.

## Kritische Architektur-Regeln

### Datenzugriff-Architektur (VERBINDLICH)

```
Externe APIs (EasyInsight, TwelveData, etc.)
                    │
                    ▼
         ┌─────────────────────┐
         │    DATA SERVICE     │  ◄── Einziges Gateway für externe Daten
         │     (Port 3001)     │
         └──────────┬──────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
 NHITS Service  RAG Service   LLM Service
 (Port 3002)    (Port 3003)   (Port 3004)
```

### Verbindliche Regeln

1. **Kein direkter Datenbankzugriff**
   - KEIN `asyncpg`, `psycopg2` oder andere DB-Treiber
   - EasyInsight TimescaleDB nur über die EasyInsight REST API

2. **Data Service als einziges Gateway**
   - NHITS, RAG und LLM Services greifen auf externe Daten NUR über den Data Service zu
   - Verwende `DataGatewayService` (src/services/data_gateway_service.py) als Singleton

3. **Fallback-Logik nur im Data Service**
   - TwelveData als Backup für EasyInsight gehört NUR in den Data Service

### Beispiele

```python
# ❌ FALSCH: Direkter API-Aufruf aus NHITS-Service
class NHITSTrainingService:
    async def get_data(self, symbol: str):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://easyinsight-api/ohlcv/{symbol}")
            return response.json()

# ✅ RICHTIG: Über Data Gateway
from .data_gateway_service import data_gateway

class NHITSTrainingService:
    async def get_data(self, symbol: str):
        return await data_gateway.get_historical_data(symbol, limit=1000)
```

## Code-Stil

- **Python 3.11+**
- **Async-First**: Alle I/O-Operationen asynchron
- **Type Hints**: Vollständig auf allen Funktionen
- **Pydantic**: Für alle API-Requests/Responses
- **Zeilenlänge**: Max 120 Zeichen
- **Imports**: Standard → Third-Party → Lokal (getrennt durch Leerzeilen)

## Microservices Ports

| Service | Port |
|---------|------|
| Frontend | 3000 |
| Data Service | 3001 |
| NHITS Service | 3002 |
| RAG Service | 3003 |
| LLM Service | 3004 |

## Commit-Konvention

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`
Scopes: `nhits`, `rag`, `llm`, `api`, `config`, `docker`, `data`

## Zeitzonen-Handling (VERBINDLICH)

### Grundregeln

1. **Speicherung immer in UTC (ISO 8601)**
   - Alle Timestamps intern als UTC speichern
   - Format: `2024-01-15T10:30:45Z` oder `2024-01-15T10:30:45+00:00`
   - Niemals naive datetime-Objekte verwenden

2. **Präsentation in lokaler Zeit**
   - Alle Timestamps für die Anzeige in die konfigurierte Zeitzone konvertieren
   - Zeitzone ist konfigurierbar (Standard: `Europe/Zurich`)

3. **Konfiguration**
   - Zeitzone wird in `src/core/config.py` als `DISPLAY_TIMEZONE` definiert
   - Umgebungsvariable: `DISPLAY_TIMEZONE`

### Beispiele

```python
from src.utils.timezone_utils import to_utc, to_display_timezone, format_for_display

# ❌ FALSCH: Naive datetime oder inkonsistente Zeitzonen
timestamp = datetime.now()  # Naive, keine Zeitzone
timestamp_str = data.get("snapshot_time")  # Direkt durchreichen

# ✅ RICHTIG: UTC für Speicherung
timestamp = datetime.now(timezone.utc)
utc_timestamp = to_utc(data.get("snapshot_time"))

# ✅ RICHTIG: Lokale Zeit für Anzeige
display_time = to_display_timezone(utc_timestamp)
formatted = format_for_display(utc_timestamp)  # "20.12.2024, 10:18:00 CET"
```

### API-Responses

- Interne Felder (`*_utc`): ISO 8601 UTC-Format
- Anzeige-Felder (`*_display`): Lokalisiertes Format mit Zeitzonenangabe

## Dokumentation

Vollständige Entwicklungsrichtlinien: `DEVELOPMENT_GUIDELINES.md`
