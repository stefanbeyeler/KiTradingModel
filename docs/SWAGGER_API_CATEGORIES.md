# Swagger API Kategorisierung

## Port-Konfiguration

### Backend API (Port 3011)

- **URL**: `http://10.1.19.101:3011`
- **Verwendung**: Direkte API-Aufrufe (curl, Postman, Scripts)
- **Swagger UI**: `http://10.1.19.101:3011/docs`
- **ReDoc**: `http://10.1.19.101:3011/redoc`
- **OpenAPI JSON**: `http://10.1.19.101:3011/openapi.json`

### Frontend Dashboard (Port 3001)

- **URL**: `http://10.1.19.101:3001`
- **Verwendung**: Web-UI mit integrierter Swagger-Dokumentation
- **Swagger UI**: `http://10.1.19.101:3001/docs` (zeigt Backend-API)
- **Funktionsweise**: API-Calls werden automatisch an Backend (Port 3011) weitergeleitet

**Empfehlung**: Verwenden Sie **Port 3001** für interaktive Nutzung über Browser (Swagger UI) und **Port 3011** für Scripting/Automatisierung.

---

## Übersicht

Die API-Dokumentation ist in 9 thematische Kategorien unterteilt, um die Navigation und Nutzung zu erleichtern.

## API-Kategorien

### 🔮 NHITS Forecast (4 Endpunkte)
**Vorhersagen generieren und Modellinformationen abrufen**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/forecast/status` | NHITS Service-Status und Konfiguration |
| GET | `/api/v1/forecast/models` | Liste aller trainierten Modelle |
| GET | `/api/v1/forecast/{symbol}` | Preisvorhersage für ein Symbol generieren |
| GET | `/api/v1/forecast/{symbol}/model` | Modellinformationen für ein Symbol |

**Verwendung:**
- Preisprognosen für Trading-Entscheidungen
- Modellstatus und Metadaten prüfen
- Confidence-Intervalle und Trends analysieren

---

### 🎓 NHITS Training (11 Endpunkte)
**Modelltraining, Performance-Überwachung und Evaluierung**

#### Batch Training
| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/forecast/train-all` | Batch-Training für alle/ausgewählte Symbole |
| GET | `/api/v1/forecast/training/status` | Training-Service Status |
| **GET** | **`/api/v1/forecast/training/progress`** | **Echtzeit-Fortschrittsüberwachung** ⭐ NEU |
| GET | `/api/v1/forecast/training/symbols` | Verfügbare Symbole für Training |
| POST | `/api/v1/forecast/training/cancel` | Laufendes Training abbrechen |

#### Einzelmodell Training
| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/forecast/{symbol}/train` | Einzelnes Symbol trainieren |

#### Performance & Evaluation
| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/forecast/performance` | Performance-Metriken aller Modelle |
| GET | `/api/v1/forecast/evaluated` | Evaluierte Vorhersagen mit Ergebnissen |
| POST | `/api/v1/forecast/evaluate` | Pending Predictions evaluieren |
| GET | `/api/v1/forecast/retraining-needed` | Modelle, die Retraining benötigen |
| POST | `/api/v1/forecast/retrain-poor-performers` | Schwache Modelle automatisch neu trainieren |

**Verwendung:**
- Modelle trainieren und aktualisieren
- Trainingsprozesse überwachen
- Model-Performance analysieren
- Automatische Verbesserungen durchführen

---

### 📊 Trading Analysis (4 Endpunkte)
**Trading-Empfehlungen und Marktanalysen**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/analyze` | Vollständige Trading-Analyse mit LLM |
| GET | `/api/v1/recommendation/{symbol}` | Schnelle Trading-Empfehlung |
| GET | `/api/v1/symbols` | Verfügbare Trading-Symbole |
| GET | `/api/v1/symbol-info/{symbol}` | Detaillierte Symbol-Informationen |

---

### 📈 Symbol Management (10 Endpunkte)
**Verwaltung von Trading-Symbolen und Daten**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/managed-symbols` | Alle verwalteten Symbole |
| GET | `/api/v1/managed-symbols/stats` | Symbol-Statistiken |
| GET | `/api/v1/managed-symbols/search` | Symbole durchsuchen |
| POST | `/api/v1/managed-symbols/import` | Symbole aus TimescaleDB importieren |
| POST | `/api/v1/managed-symbols` | Neues Symbol erstellen |
| GET | `/api/v1/managed-symbols/{id}` | Symbol-Details abrufen |
| PUT | `/api/v1/managed-symbols/{id}` | Symbol aktualisieren |
| DELETE | `/api/v1/managed-symbols/{id}` | Symbol löschen |
| POST | `/api/v1/managed-symbols/{id}/favorite` | Favoriten-Status umschalten |
| POST | `/api/v1/managed-symbols/{id}/refresh` | Symbol-Daten aktualisieren |

---

### 🎯 Trading Strategies (9 Endpunkte)
**Verwaltung von Trading-Strategien**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/strategies` | Alle Strategien auflisten |
| GET | `/api/v1/strategies/default` | Standard-Strategie abrufen |
| GET | `/api/v1/strategies/{id}` | Spezifische Strategie |
| GET | `/api/v1/strategies/{id}/export` | Strategie als Markdown exportieren |
| POST | `/api/v1/strategies` | Neue Strategie erstellen |
| PUT | `/api/v1/strategies/{id}` | Strategie aktualisieren |
| DELETE | `/api/v1/strategies/{id}` | Strategie löschen |
| POST | `/api/v1/strategies/{id}/set-default` | Als Standard setzen |
| POST | `/api/v1/strategies/import` | Strategie aus Markdown importieren |

---

### 🧠 RAG & Knowledge Base (5 Endpunkte)
**Retrieval-Augmented Generation und Wissensbasis**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| POST | `/api/v1/rag/document` | Dokument zur RAG-Basis hinzufügen |
| GET | `/api/v1/rag/query` | RAG-System abfragen |
| GET | `/api/v1/rag/stats` | RAG-Statistiken |
| DELETE | `/api/v1/rag/documents` | Dokumente löschen |
| POST | `/api/v1/rag/persist` | RAG-Datenbank persistieren |

---

### 🖥️ System & Monitoring (7 Endpunkte)
**Systemstatus, Health-Checks und Synchronisation**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/version` | Versions-Informationen |
| GET | `/api/v1/health` | Health-Check aller Services |
| GET | `/api/v1/system/info` | System- und GPU-Informationen |
| GET | `/api/v1/sync/status` | TimescaleDB Sync-Status |
| POST | `/api/v1/sync/start` | Sync-Service starten |
| POST | `/api/v1/sync/stop` | Sync-Service stoppen |
| POST | `/api/v1/sync/manual` | Manuelle Synchronisation |

---

### 🤖 LLM Service (2 Endpunkte)
**Large Language Model Management**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/llm/status` | LLM-Status und Konfiguration |
| POST | `/api/v1/llm/pull` | LLM-Modell herunterladen |

---

### 📝 Query Logs & Analytics (4 Endpunkte)
**Query-Logging und Statistiken**

| Methode | Endpunkt | Beschreibung |
|---------|----------|--------------|
| GET | `/api/v1/query-logs` | Query-Log Historie |
| GET | `/api/v1/query-logs/stats` | Query-Log Statistiken |
| GET | `/api/v1/query-logs/{id}` | Spezifischer Log-Eintrag |
| DELETE | `/api/v1/query-logs` | Alle Logs löschen |

---

## Statistik

```
📊 API Endpoint Distribution:
════════════════════════════════════════════════════════════════════════════════
🎓 NHITS Training                          11 █████
🎯 Trading Strategies                       9 ████
📈 Symbol Management                       10 █████
📊 Trading Analysis                         4 ██
📝 Query Logs & Analytics                   4 ██
🔮 NHITS Forecast                           4 ██
🖥️ System & Monitoring                     7 ███
🤖 LLM Service                              2 █
🧠 RAG & Knowledge Base                     5 ██
════════════════════════════════════════════════════════════════════════════════
Total Endpoints:                           56
```

## Zugriff auf die Dokumentation

### Swagger UI (Interaktiv)
```
http://localhost:3011/docs
```
- ✅ Interaktive API-Tests
- ✅ "Try it out" Funktionalität
- ✅ Schemas und Beispiele
- ✅ Kategorisierte Darstellung

### ReDoc (Lesbar)
```
http://localhost:3011/redoc
```
- ✅ Übersichtliche Dokumentation
- ✅ Durchsuchbar
- ✅ Export-Funktionen

### OpenAPI JSON
```
http://localhost:3011/openapi.json
```
- ✅ Maschinell lesbar
- ✅ Für Code-Generierung
- ✅ API-Client Integration

## Änderungshistorie

### Version 2.0 (2025-12-14)
**Aufteilung NHITS in Forecast und Training**

- ✅ Neue Kategorie: 🔮 NHITS Forecast (4 Endpunkte)
  - Fokus auf Vorhersagen und Model-Info

- ✅ Neue Kategorie: 🎓 NHITS Training (11 Endpunkte)
  - Fokus auf Training, Performance, Evaluation
  - **NEU:** `/forecast/training/progress` - Echtzeit-Fortschrittsüberwachung

- ✅ Verbesserte Organisation
  - Klarere Trennung von Concerns
  - Bessere Auffindbarkeit
  - Logische Gruppierung

### Migrationsleitfaden

Alle Endpunkte funktionieren wie zuvor, nur die Swagger-Kategorisierung hat sich geändert:

**Vorher:**
- 🔮 NHITS Forecasting (15 Endpunkte)

**Nachher:**
- 🔮 NHITS Forecast (4 Endpunkte) - Vorhersagen
- 🎓 NHITS Training (11 Endpunkte) - Training & Performance

Keine Breaking Changes - alle URLs bleiben identisch!

## Best Practices

### 1. Forecasting-Workflow
```bash
# 1. Prüfen ob Modell existiert
GET /api/v1/forecast/{symbol}/model

# 2. Falls nicht: Modell trainieren
POST /api/v1/forecast/{symbol}/train

# 3. Vorhersage generieren
GET /api/v1/forecast/{symbol}?horizon=24
```

### 2. Training-Workflow
```bash
# 1. Training starten
POST /api/v1/forecast/train-all?background=true

# 2. Fortschritt überwachen
while true; do
  GET /api/v1/forecast/training/progress
  sleep 2
done

# 3. Performance prüfen
GET /api/v1/forecast/performance

# 4. Schwache Modelle identifizieren
GET /api/v1/forecast/retraining-needed

# 5. Automatisch neu trainieren
POST /api/v1/forecast/retrain-poor-performers
```

### 3. Evaluation-Workflow
```bash
# 1. Pending Predictions evaluieren
POST /api/v1/forecast/evaluate

# 2. Ergebnisse ansehen
GET /api/v1/forecast/evaluated?limit=50

# 3. Performance analysieren
GET /api/v1/forecast/performance
```

## Support

Für Fragen zur API-Nutzung:
- **Swagger UI:** http://localhost:3011/docs
- **Dokumentation:** `/docs/NHITS_TRAINING_PROGRESS_API.md`
- **Beispiele:** `/test_training_progress.py`
