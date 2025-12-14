# Event-Based NHITS Training

## Übersicht

Das Event-Based Training System überwacht Trading-Events von der EasyInsight Logs API (`http://10.1.19.102:3000/api/logs`) und triggert automatisch NHITS Model-Retraining, wenn signifikante Marktbewegungen erkannt werden.

## Konzept

### Warum Event-Based Training?

Traditionelles NHITS-Training verwendet:
- **Zeitbasiertes Retraining**: Modelle werden alle X Tage neu trainiert
- **Problem**: Verpasst wichtige Marktbewegungen zwischen den Trainings

Event-Based Training löst dies durch:
- **Event-getriebenes Retraining**: Modelle werden bei signifikanten Events neu trainiert
- **Adaptive Response**: Schnellere Anpassung an Marktveränderungen
- **Ressourcen-Effizienz**: Nur Retraining wenn nötig

### Monitored Events

Die folgenden Event-Typen werden überwacht:

#### 1. **ATR (Average True Range)** - Volatilitätsmessung
```json
{
  "indicator": "ATR",
  "content": "EURUSD ATR: 135%",
  "symbol": "EURUSD"
}
```
- **Trigger**: Wenn ATR > 100% (hohe Volatilität)
- **Bedeutung**: Starke Preisbewegungen, Modell könnte veraltet sein

#### 2. **FXL (Support/Resistance Levels)**
```json
{
  "indicator": "FXL",
  "content": "FXL for EURUSD  Support at 1.0850 | Resistance at 1.0920|",
  "symbol": "EURUSD"
}
```
- **Trigger**: Mehrere FXL-Events innerhalb kurzer Zeit
- **Bedeutung**: Wichtige Preisniveaus durchbrochen, Trend-Änderung

## API Endpunkte

### GET `/api/v1/forecast/training/events/status`

Ruft den Status des Event-Monitors ab.

**Response:**
```json
{
  "running": false,
  "check_interval_minutes": 15,
  "event_threshold": 10,
  "monitored_indicators": ["ATR", "FXL"]
}
```

**Parameter:**
- `running`: Ob der Monitor aktiv ist
- `check_interval_minutes`: Wie oft Events geprüft werden
- `event_threshold`: Anzahl Events für Retraining-Trigger
- `monitored_indicators`: Welche Indikatoren überwacht werden

---

### GET `/api/v1/forecast/training/events/summary`

Analysiert kürzliche Trading-Events.

**Query Parameters:**
- `symbol` (optional): Symbol zum Filtern (z.B. "EURUSD")
- `hours` (default: 24): Zeitfenster in Stunden

**Beispiel:**
```bash
curl "http://10.1.19.101:3011/api/v1/forecast/training/events/summary?symbol=EURUSD&hours=1"
```

**Response:**
```json
{
  "total_events": 119,
  "timeframe_hours": 1,
  "symbol_filter": "EURUSD",
  "indicators": {
    "ATR": 119
  },
  "top_symbols": {
    "EURUSD": 119
  },
  "events": [
    {
      "content": "EURUSD ATR: 55%",
      "indicator": "ATR",
      "symbol": "EURUSD",
      "timestamp": "2025-12-14T08:19:11.605000+01:00"
    }
  ]
}
```

---

### POST `/api/v1/forecast/training/events/start`

Startet den Event-Monitor.

**Beispiel:**
```bash
curl -X POST "http://10.1.19.101:3011/api/v1/forecast/training/events/start"
```

**Response:**
```json
{
  "success": true,
  "message": "Event-based training monitor started",
  "check_interval_minutes": 15
}
```

**Was passiert:**
1. Monitor prüft alle 15 Minuten auf neue Events
2. Analysiert Event-Muster (Volatilität, Preis-Breaks, etc.)
3. Triggert automatisch Retraining bei signifikanten Events
4. Verwendet `force=true` für sofortiges Retraining

---

### POST `/api/v1/forecast/training/events/stop`

Stoppt den Event-Monitor.

**Beispiel:**
```bash
curl -X POST "http://10.1.19.101:3011/api/v1/forecast/training/events/stop"
```

**Response:**
```json
{
  "success": true,
  "message": "Event-based training monitor stopped"
}
```

## Workflow

### 1. Event-Monitor starten

```bash
# Monitor starten
curl -X POST "http://10.1.19.101:3011/api/v1/forecast/training/events/start"

# Status prüfen
curl "http://10.1.19.101:3011/api/v1/forecast/training/events/status"
```

### 2. Events analysieren

```bash
# Alle Events der letzten 24 Stunden
curl "http://10.1.19.101:3011/api/v1/forecast/training/events/summary?hours=24"

# Nur EURUSD Events der letzten Stunde
curl "http://10.1.19.101:3011/api/v1/forecast/training/events/summary?symbol=EURUSD&hours=1"
```

### 3. Automatisches Retraining

Der Monitor arbeitet automatisch:

```
Alle 15 Minuten:
  ├─ Hole Events der letzten 30 Minuten
  ├─ Analysiere Event-Muster pro Symbol
  ├─ Wenn signifikant:
  │   ├─ ATR > 100% (hohe Volatilität)
  │   ├─ Mehrere FXL-Breaks
  │   └─ Triggere Retraining mit force=true
  └─ Warte 15 Minuten
```

## Retraining-Trigger

### Hohe Volatilität (ATR)

```python
if avg_atr > 100%:
    trigger_retraining(symbol)
```

**Beispiel:**
```
Events:
  - EURUSD ATR: 135% (08:10)
  - EURUSD ATR: 142% (08:15)
  - EURUSD ATR: 128% (08:20)

Durchschnitt: 135% > 100%
→ Retraining triggered für EURUSD
```

### Support/Resistance Breaks (FXL)

```python
if count(fxl_events) >= 3:
    trigger_retraining(symbol)
```

**Beispiel:**
```
Events:
  - FXL: Support at 1.0850 (08:10)
  - FXL: Resistance at 1.0920 (08:12)
  - FXL: Support at 1.0840 (08:15)

Anzahl: 3 >= 3
→ Retraining triggered für EURUSD
```

## Konfiguration

Die Event-Thresholds können im Code angepasst werden:

```python
# In event_based_training_service.py
self._event_threshold = 10           # Min. Events für Trigger
self._check_interval_minutes = 15    # Check-Frequenz
self._monitored_indicators = ["ATR", "FXL"]  # Zu überwachende Indikatoren
```

## Integration mit bestehendem Training

Event-Based Training **ergänzt** das bestehende Training:

### Zeitbasiertes Training (Standard)
```bash
# Trainiert alle Modelle alle 7 Tage
NHITS_AUTO_RETRAIN_DAYS=7
```

### Event-Based Training (NEU)
```bash
# Trainiert bei signifikanten Events
POST /forecast/training/events/start
```

### Manuelles Training
```bash
# Jederzeit manuell triggerbar
POST /forecast/train-all?force=true
```

Alle drei Methoden arbeiten **parallel** und **ergänzend**.

## Monitoring

### Training-Status prüfen

```bash
# Laufendes Training
curl "http://10.1.19.101:3011/api/v1/forecast/training/progress"

# Event-Monitor Status
curl "http://10.1.19.101:3011/api/v1/forecast/training/events/status"
```

### Logs prüfen

Der Event-Monitor loggt alle Aktivitäten:

```
INFO: Event-Based Training Service started - check interval: 15min
INFO: Detected significant events for 2 symbols: EURUSD, GBPUSD
INFO: High volatility detected: avg ATR = 135%
INFO: Starting batch training for 2 symbols
```

## Beispiel-Workflow

### Szenario: Marktvolatilität

```bash
# 1. Event-Monitor starten
curl -X POST "http://10.1.19.101:3011/api/v1/forecast/training/events/start"

# 2. Warten auf Events (automatisch alle 15min)
# ... 15 Minuten später ...

# 3. Events werden erkannt
# Logs zeigen:
# INFO: Detected significant events for EURUSD
# INFO: High volatility detected: avg ATR = 135%

# 4. Automatisches Retraining wird gestartet
# INFO: Starting batch training for 1 symbols

# 5. Fortschritt überwachen
curl "http://10.1.19.101:3011/api/v1/forecast/training/progress"

# 6. Nach Training: Neues Modell ist verfügbar
curl "http://10.1.19.101:3011/api/v1/forecast/EURUSD"
# Verwendet das frisch trainierte Modell
```

## Best Practices

### 1. Monitor kontinuierlich laufen lassen

```bash
# In Produktionsumgebung beim Startup
POST /forecast/training/events/start
```

### 2. Event-Summary regelmäßig prüfen

```bash
# Täglich Events analysieren
curl "http://10.1.19.101:3011/api/v1/forecast/training/events/summary?hours=24"
```

### 3. Kombination mit zeitbasiertem Training

```env
# .env
NHITS_AUTO_RETRAIN_DAYS=7  # Basis-Retraining
```

Plus Event-Monitor für zusätzliche Triggers.

### 4. Threshold anpassen

Je nach Marktbedingungen:
- **Ruhiger Markt**: Höhere Thresholds (weniger Retraining)
- **Volatiler Markt**: Niedrigere Thresholds (mehr Retraining)

## Vorteile

✅ **Reaktionsschnell**: Modelle passen sich an Marktänderungen an
✅ **Ressourcen-effizient**: Nur Retraining bei Bedarf
✅ **Automatisch**: Keine manuelle Intervention nötig
✅ **Ergänzend**: Arbeitet mit bestehendem Training zusammen
✅ **Transparent**: Event-Summary zeigt Trigger-Gründe

## Limitierungen

⚠️ **Kein Ersatz für Preisdaten**: Events triggern nur Training, OHLC-Daten kommen von `/api/symbol-data-full`
⚠️ **Check-Intervall**: Events werden nur alle 15 Minuten geprüft
⚠️ **Event-Verfügbarkeit**: Abhängig von EasyInsight Logs API

## Swagger-Dokumentation

Alle Endpunkte sind dokumentiert unter:
- **Swagger UI**: http://localhost:3011/docs
- **Kategorie**: 🎓 NHITS Training
- **Section**: Event-Based Training

## Zusammenfassung

Das Event-Based Training System macht NHITS-Modelle **adaptiver** und **reaktionsschneller** auf Marktveränderungen, indem es Trading-Events von der EasyInsight Logs API nutzt, um automatisch Retraining zu triggern.

**Setup in 3 Schritten:**
1. `POST /forecast/training/events/start` - Monitor starten
2. Events werden automatisch überwacht
3. Bei signifikanten Events: Automatisches Retraining

Das System arbeitet **parallel** zum zeitbasierten Training und ergänzt es optimal! 🚀
