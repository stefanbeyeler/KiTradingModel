# NHITS Training Progress API

## Übersicht

Der neue API-Endpunkt `/api/v1/forecast/training/progress` ermöglicht die Echtzeit-Überwachung von laufenden NHITS-Trainings.

## Endpunkt

### GET `/api/v1/forecast/training/progress`

Ruft den aktuellen Fortschritt einer laufenden NHITS-Trainingssitzung ab.

#### Response Model: `TrainingProgressResponse`

**Wenn Training läuft:**

```json
{
  "training_in_progress": true,
  "current_symbol": "EURUSD",
  "total_symbols": 20,
  "completed_symbols": 8,
  "remaining_symbols": 12,
  "progress_percent": 40,
  "results": {
    "successful": 5,
    "failed": 1,
    "skipped": 2
  },
  "timing": {
    "elapsed_seconds": 240,
    "eta_seconds": 360,
    "elapsed_formatted": "4m 0s",
    "eta_formatted": "6m 0s"
  },
  "cancelling": false,
  "started_at": "2025-12-14T10:30:00"
}
```

**Wenn kein Training läuft:**

```json
{
  "training_in_progress": false,
  "current_symbol": null,
  "total_symbols": 0,
  "completed_symbols": 0,
  "remaining_symbols": 0,
  "progress_percent": 0,
  "results": {
    "successful": 0,
    "failed": 0,
    "skipped": 0
  },
  "timing": null,
  "cancelling": false,
  "started_at": null,
  "message": "No training currently running",
  "last_training_run": "2025-12-14T09:15:00"
}
```

## Datenmodelle

### TrainingProgressResponse

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `training_in_progress` | boolean | Gibt an, ob aktuell ein Training läuft |
| `current_symbol` | string \| null | Das Symbol, das gerade trainiert wird |
| `total_symbols` | integer | Gesamtanzahl der zu trainierenden Symbole |
| `completed_symbols` | integer | Anzahl der abgeschlossenen Symbole |
| `remaining_symbols` | integer | Anzahl der verbleibenden Symbole |
| `progress_percent` | integer (0-100) | Fortschritt in Prozent |
| `results` | TrainingProgressResults | Aufschlüsselung der Trainingsergebnisse |
| `timing` | TrainingProgressTiming \| null | Zeitinformationen (nur wenn Training aktiv) |
| `cancelling` | boolean | Gibt an, ob ein Abbruch angefordert wurde |
| `started_at` | string \| null | ISO-Zeitstempel des Trainingsbeginns |
| `message` | string \| null | Statusmeldung (wenn kein Training läuft) |
| `last_training_run` | string \| null | ISO-Zeitstempel des letzten Trainings |

### TrainingProgressResults

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `successful` | integer | Anzahl erfolgreich trainierter Modelle |
| `failed` | integer | Anzahl fehlgeschlagener Trainingsversuche |
| `skipped` | integer | Anzahl übersprungener Modelle (bereits aktuell) |

### TrainingProgressTiming

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `elapsed_seconds` | integer | Verstrichene Zeit seit Trainingsbeginn (Sekunden) |
| `eta_seconds` | integer \| null | Geschätzte verbleibende Zeit (Sekunden) |
| `elapsed_formatted` | string | Verstrichene Zeit in lesbarem Format (z.B. "4m 30s") |
| `eta_formatted` | string \| null | ETA in lesbarem Format (z.B. "6m 15s") |

## Verwendungsbeispiele

### cURL

```bash
# Fortschritt abfragen
curl http://localhost:3011/api/v1/forecast/training/progress
```

### Python (httpx)

```python
import httpx
import asyncio

async def monitor_training():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:3011/api/v1/forecast/training/progress"
        )
        progress = response.json()

        if progress["training_in_progress"]:
            print(f"Training {progress['current_symbol']}...")
            print(f"Progress: {progress['progress_percent']}%")
            print(f"Completed: {progress['completed_symbols']}/{progress['total_symbols']}")
            print(f"ETA: {progress['timing']['eta_formatted']}")
        else:
            print("No training in progress")

asyncio.run(monitor_training())
```

### JavaScript (fetch)

```javascript
async function checkTrainingProgress() {
  const response = await fetch(
    'http://localhost:3011/api/v1/forecast/training/progress'
  );
  const progress = await response.json();

  if (progress.training_in_progress) {
    console.log(`Training ${progress.current_symbol}...`);
    console.log(`Progress: ${progress.progress_percent}%`);
    console.log(`ETA: ${progress.timing.eta_formatted}`);
  } else {
    console.log('No training in progress');
  }
}
```

## Monitoring-Workflow

### 1. Training starten

```bash
curl -X POST "http://localhost:3011/api/v1/forecast/train-all?background=true&force=false"
```

Response:
```json
{
  "status": "started",
  "message": "Training started in background",
  "symbols": "all available"
}
```

### 2. Fortschritt überwachen (Polling)

```python
import asyncio
import httpx

async def monitor_until_complete():
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(
                "http://localhost:3011/api/v1/forecast/training/progress"
            )
            progress = response.json()

            if not progress["training_in_progress"]:
                print("✓ Training completed!")
                break

            # Display progress
            pct = progress["progress_percent"]
            current = progress["current_symbol"]
            completed = progress["completed_symbols"]
            total = progress["total_symbols"]
            eta = progress["timing"]["eta_formatted"]

            print(f"[{pct}%] {current} ({completed}/{total}) - ETA: {eta}")

            # Poll every 2 seconds
            await asyncio.sleep(2)

asyncio.run(monitor_until_complete())
```

### 3. Training abbrechen (falls nötig)

```bash
curl -X POST http://localhost:3011/api/v1/forecast/training/cancel
```

Response:
```json
{
  "success": true,
  "message": "Training cancellation requested. Training will stop after current symbol."
}
```

## Test-Skript

Ein vollständiges Monitoring-Skript ist verfügbar unter:
```
/home/sbeyeler/KiTradingModel/test_training_progress.py
```

Verwendung:
```bash
python test_training_progress.py
```

Das Skript zeigt:
- Echtzeit-Fortschrittsbalken
- Detaillierte Statistiken
- Automatische Aktualisierung alle 2 Sekunden
- Formatierte Zeitangaben

## Swagger UI

Die vollständige API-Dokumentation ist verfügbar unter:
- **Swagger UI:** http://localhost:3011/docs
- **ReDoc:** http://localhost:3011/redoc
- **OpenAPI JSON:** http://localhost:3011/openapi.json

Im Swagger UI finden Sie:
- Interaktive API-Tests
- Vollständige Schema-Definitionen
- Request/Response-Beispiele
- Detaillierte Feldsbeschreibungen

## Verwandte Endpunkte

| Endpunkt | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/v1/forecast/training/status` | GET | Allgemeiner Status des Training-Service |
| `/api/v1/forecast/training/progress` | GET | **Echtzeit-Fortschritt (NEU)** |
| `/api/v1/forecast/train-all` | POST | Batch-Training starten |
| `/api/v1/forecast/training/cancel` | POST | Laufendes Training abbrechen |
| `/api/v1/forecast/training/symbols` | GET | Verfügbare Symbole für Training |
| `/api/v1/forecast/{symbol}/train` | POST | Einzelnes Symbol trainieren |

## Best Practices

### Polling-Intervall
- **Empfohlen:** 2-5 Sekunden
- **Minimum:** 1 Sekunde (um Server nicht zu überlasten)
- **Maximum:** 30 Sekunden (für zeitnahe Updates)

### Error Handling
```python
import httpx

async def safe_progress_check():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "http://localhost:3011/api/v1/forecast/training/progress"
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        print("Request timed out")
        return None
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        return None
```

### UI Integration
Für Frontend-Integration empfiehlt sich:
1. WebSocket für Echtzeit-Updates (zukünftige Erweiterung)
2. Polling mit exponential backoff bei Fehlern
3. Visuelle Fortschrittsanzeige mit ETA
4. Benachrichtigung bei Abschluss

## Zusammenfassung

Der neue `/forecast/training/progress` Endpunkt bietet:
- ✅ Echtzeit-Fortschrittsverfolgung
- ✅ Detaillierte Statistiken (successful/failed/skipped)
- ✅ Zeitschätzungen (elapsed/ETA)
- ✅ Vollständig typisierte Pydantic-Modelle
- ✅ Automatische Swagger-Dokumentation
- ✅ Unterstützung für Training-Abbruch
- ✅ Produktionsreife API
