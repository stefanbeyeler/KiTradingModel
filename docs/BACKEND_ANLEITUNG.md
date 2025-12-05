# Backend Benutzeranleitung

## KI Trading Model - Backend Service

Diese Anleitung beschreibt die Installation, Konfiguration und Nutzung des Backend-Services für das KI Trading Model.

---

## Inhaltsverzeichnis

1. [Systemvoraussetzungen](#1-systemvoraussetzungen)
2. [Installation](#2-installation)
3. [Konfiguration](#3-konfiguration)
4. [Starten des Services](#4-starten-des-services)
5. [API-Endpunkte](#5-api-endpunkte)
6. [Services im Detail](#6-services-im-detail)
7. [Fehlerbehebung](#7-fehlerbehebung)

---

## 1. Systemvoraussetzungen

### Software

| Komponente | Version | Erforderlich |
|------------|---------|--------------|
| Python | 3.10+ | Ja |
| Ollama | Latest | Ja |
| TimescaleDB/PostgreSQL | 14+ | Ja |
| Git | Latest | Optional |

### Hardware-Empfehlungen

- **RAM**: Mindestens 16 GB (32 GB empfohlen für Llama 3.1 70B)
- **GPU**: NVIDIA GPU mit 24+ GB VRAM für optimale LLM-Performance
- **Speicher**: 50 GB freier Festplattenspeicher

### Externe Dienste

- **Ollama**: Lokaler LLM-Server (muss separat installiert werden)
- **TimescaleDB**: Zeitreihendatenbank mit Marktdaten

---

## 2. Installation

### 2.1 Repository klonen

```bash
git clone <repository-url>
cd KITradingModel
```

### 2.2 Virtuelle Umgebung erstellen

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2.3 Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 2.4 Ollama einrichten

1. Ollama von https://ollama.ai herunterladen und installieren
2. LLM-Modell herunterladen:

```bash
# Empfohlen für beste Ergebnisse:
ollama pull llama3.1:70b

# Alternative für weniger Hardware-Ressourcen:
ollama pull llama3.1:8b
```

---

## 3. Konfiguration

### 3.1 Umgebungsvariablen

Kopieren Sie die Beispieldatei und passen Sie die Werte an:

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS
```

### 3.2 Konfigurationsparameter

Bearbeiten Sie die `.env`-Datei:

```ini
# ===== TimescaleDB Verbindung =====
TIMESCALEDB_HOST=localhost          # Datenbank-Host
TIMESCALEDB_PORT=5432               # Datenbank-Port
TIMESCALEDB_DATABASE=easyinsight    # Datenbankname
TIMESCALEDB_USER=postgres           # Benutzername
TIMESCALEDB_PASSWORD=ihr_passwort   # Passwort (ändern!)

# ===== RAG Synchronisation =====
RAG_SYNC_ENABLED=true               # Automatische Sync aktivieren
RAG_SYNC_INTERVAL_SECONDS=300       # Sync-Intervall (5 Minuten)
RAG_SYNC_BATCH_SIZE=100             # Dokumente pro Batch

# ===== Ollama/LLM Einstellungen =====
OLLAMA_HOST=http://localhost:11434  # Ollama-Server URL
OLLAMA_MODEL=llama3.1:70b           # LLM-Modell

# ===== FAISS Vektordatenbank =====
FAISS_PERSIST_DIRECTORY=./data/faiss  # Speicherort für Index

# ===== Embedding-Modell =====
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ===== Service-Einstellungen =====
SERVICE_HOST=0.0.0.0                # Host-Adresse
SERVICE_PORT=8000                   # Port
LOG_LEVEL=INFO                      # Log-Level (DEBUG, INFO, WARNING, ERROR)

# ===== Trading-Analyse =====
DEFAULT_LOOKBACK_DAYS=30            # Standard-Analysezeitraum
MAX_CONTEXT_DOCUMENTS=10            # Max. RAG-Dokumente pro Anfrage
```

---

## 4. Starten des Services

### 4.1 Standard-Start

```bash
python run.py
```

### 4.2 Entwicklungsmodus (mit Auto-Reload)

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4.3 Produktionsmodus

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4.4 Verfügbare URLs nach dem Start

| URL | Beschreibung |
|-----|--------------|
| http://localhost:3011 | API Root |
| http://localhost:3011/dashboard | Web-Dashboard |
| http://localhost:3011/docs | Swagger API-Dokumentation |
| http://localhost:3011/redoc | ReDoc API-Dokumentation |

---

## 5. API-Endpunkte

### 5.1 Health & Status

#### Gesundheitsstatus prüfen
```http
GET /api/v1/health
```

**Antwort:**
```json
{
  "status": "healthy",
  "services": {
    "llm": true,
    "rag": true,
    "timescaledb": true
  }
}
```

#### LLM-Status prüfen
```http
GET /api/v1/llm/status
```

#### Sync-Status prüfen
```http
GET /api/v1/sync/status
```

---

### 5.2 Trading-Analyse

#### Vollständige Marktanalyse
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "symbol": "EURUSD",
  "lookback_days": 30,
  "include_indicators": true
}
```

**Antwort enthält:**
- Technische Indikatoren (RSI, MACD, Bollinger Bands, etc.)
- Trading-Signale
- KI-Empfehlung (BUY/SELL/HOLD)
- Konfidenz-Level
- Entry-Preis, Stop-Loss, Take-Profit
- Risikofaktoren

#### Schnellempfehlung
```http
GET /api/v1/recommendation/{symbol}?lookback_days=30
```

**Beispiel:**
```http
GET /api/v1/recommendation/EURUSD?lookback_days=14
```

#### Verfügbare Symbole abrufen
```http
GET /api/v1/symbols
```

---

### 5.3 RAG-Verwaltung

#### Dokument hinzufügen
```http
POST /api/v1/rag/document
Content-Type: application/json

{
  "content": "Historische Analyse für EURUSD...",
  "metadata": {
    "symbol": "EURUSD",
    "date": "2024-01-15",
    "type": "analysis"
  }
}
```

#### RAG abfragen
```http
GET /api/v1/rag/query?query=EURUSD Trend&symbol=EURUSD&max_results=5
```

#### RAG-Statistiken
```http
GET /api/v1/rag/stats
```

#### Dokumente löschen
```http
DELETE /api/v1/rag/documents?symbol=EURUSD
```

#### Index persistieren
```http
POST /api/v1/rag/persist
```

---

### 5.4 Sync-Steuerung

#### Automatische Synchronisation starten
```http
POST /api/v1/sync/start
```

#### Synchronisation stoppen
```http
POST /api/v1/sync/stop
```

#### Manuelle Synchronisation
```http
POST /api/v1/sync/manual?days_back=7
```

---

## 6. Services im Detail

### 6.1 LLM-Service

Der LLM-Service kommuniziert mit Ollama für die KI-gestützte Analyse.

**Funktionen:**
- Modellverfügbarkeit prüfen
- Trading-Analysen generieren
- JSON-formatierte Empfehlungen erstellen

**Konfiguration:**
```ini
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b
```

**Unterstützte Modelle:**
- `llama3.1:70b` - Beste Qualität (empfohlen)
- `llama3.1:8b` - Schneller, weniger Ressourcen

---

### 6.2 RAG-Service (FAISS)

Der RAG-Service speichert und durchsucht historische Analysen.

**Funktionen:**
- Vektorbasierte Ähnlichkeitssuche
- Persistente Speicherung auf Festplatte
- Automatische Embedding-Generierung

**Speicherort:**
```
data/faiss/
├── index.faiss      # FAISS-Index
└── metadata.json    # Dokumenten-Metadaten
```

---

### 6.3 Analysis-Service

Der Haupt-Analysedienst orchestriert die gesamte Pipeline.

**Berechnete technische Indikatoren:**

| Kategorie | Indikatoren |
|-----------|-------------|
| Gleitende Durchschnitte | SMA 20, SMA 50, SMA 200, EMA 12, EMA 26 |
| Momentum | RSI, MACD, MACD Signal, MACD Histogramm |
| Volatilität | Bollinger Bands (Upper, Middle, Lower), ATR |
| Volumen | OBV (On-Balance Volume) |

**Trend-Klassifizierung:**
- `strong_uptrend` - Starker Aufwärtstrend
- `uptrend` - Aufwärtstrend
- `sideways` - Seitwärtsbewegung
- `downtrend` - Abwärtstrend
- `strong_downtrend` - Starker Abwärtstrend

---

### 6.4 TimescaleDB-Sync-Service

Synchronisiert Marktdaten automatisch mit dem RAG-System.

**Features:**
- Hintergrund-Task mit konfigurierbarem Intervall
- Batch-Verarbeitung für Effizienz
- Manueller Sync für spezifische Zeiträume

**Status-Informationen:**
```json
{
  "running": true,
  "last_sync": "2024-01-15T10:30:00Z",
  "documents_synced": 1250,
  "interval_seconds": 300
}
```

---

## 7. Fehlerbehebung

### 7.1 Häufige Probleme

#### Ollama nicht erreichbar

**Symptom:** `LLM service unavailable`

**Lösung:**
1. Ollama-Status prüfen:
   ```bash
   ollama list
   ```
2. Ollama-Server starten:
   ```bash
   ollama serve
   ```
3. Host-Einstellung prüfen in `.env`:
   ```ini
   OLLAMA_HOST=http://localhost:11434
   ```

#### TimescaleDB-Verbindungsfehler

**Symptom:** `Connection refused` oder `Authentication failed`

**Lösung:**
1. PostgreSQL/TimescaleDB läuft:
   ```bash
   pg_isready -h localhost -p 5432
   ```
2. Zugangsdaten in `.env` prüfen
3. Firewall-Einstellungen prüfen

#### RAG-Index beschädigt

**Symptom:** Fehler beim Laden des FAISS-Index

**Lösung:**
1. Index löschen:
   ```bash
   rm -rf data/faiss/*
   ```
2. Service neu starten
3. Manuelle Synchronisation durchführen:
   ```http
   POST /api/v1/sync/manual?days_back=30
   ```

#### Speicherprobleme

**Symptom:** Out of Memory beim LLM

**Lösung:**
1. Kleineres Modell verwenden:
   ```ini
   OLLAMA_MODEL=llama3.1:8b
   ```
2. Batch-Größe reduzieren:
   ```ini
   RAG_SYNC_BATCH_SIZE=50
   ```

---

### 7.2 Log-Dateien

Logs werden unter `logs/` gespeichert:

```
logs/
└── ki_trading_2024-01-15_10-30-00.log
```

**Log-Level ändern:**
```ini
LOG_LEVEL=DEBUG  # Für detaillierte Diagnose
```

---

### 7.3 API-Fehlercodes

| Code | Bedeutung | Lösung |
|------|-----------|--------|
| 400 | Ungültige Anfrage | Parameter prüfen |
| 404 | Symbol nicht gefunden | Symbol existiert nicht in DB |
| 500 | Interner Serverfehler | Logs prüfen |
| 503 | Service nicht verfügbar | LLM/DB-Verbindung prüfen |

---

## Support

Bei Problemen:
1. Log-Dateien prüfen (`logs/`)
2. Health-Endpunkt aufrufen (`/api/v1/health`)
3. API-Dokumentation konsultieren (`/docs`)

---

*Letzte Aktualisierung: November 2024*
