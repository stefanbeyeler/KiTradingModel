# Frontend Benutzeranleitung

## KI Trading Model - Dashboard

Diese Anleitung beschreibt die Nutzung des Web-Dashboards für das KI Trading Model.

---

## Inhaltsverzeichnis

1. [Zugriff auf das Dashboard](#1-zugriff-auf-das-dashboard)
2. [Übersicht der Benutzeroberfläche](#2-übersicht-der-benutzeroberfläche)
3. [Dashboard-Bereiche](#3-dashboard-bereiche)
4. [KI Trading-Analyse](#4-ki-trading-analyse)
5. [Schnellaktionen](#5-schnellaktionen)
6. [Docker-Deployment](#6-docker-deployment)
7. [Tipps und Best Practices](#7-tipps-und-best-practices)

---

## 1. Zugriff auf das Dashboard

### 1.1 Voraussetzungen

- Backend-Service muss laufen (siehe Backend-Anleitung)
- Moderner Webbrowser (Chrome, Firefox, Edge, Safari)

### 1.2 URLs

| Deployment | URL |
|------------|-----|
| Lokal (direkt) | http://localhost:3011/dashboard |
| Docker | http://localhost:3001 |

### 1.3 Erster Start

1. Backend-Service starten:
   ```bash
   python run.py
   ```
2. Browser öffnen und Dashboard-URL aufrufen
3. Warten bis alle Status-Anzeigen grün sind

---

## 2. Übersicht der Benutzeroberfläche

### 2.1 Layout

Das Dashboard ist in folgende Bereiche unterteilt:

```
┌─────────────────────────────────────────────────────────────┐
│  Header (Titel + Verbindungsstatus)                         │
├─────────────────────────────────────────────────────────────┤
│  Übersichtskarten (RAG Docs, Sync, Intervall, Modell)       │
├─────────────────────────────────────────────────────────────┤
│  TimescaleDB Sync │ Service Health │ Schnellaktionen        │
├─────────────────────────────────────────────────────────────┤
│  KI Trading Analyse (Tabs: Vollanalyse, Schnell, RAG)       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Status-Anzeigen

| Farbe | Bedeutung |
|-------|-----------|
| 🟢 Grün | Service aktiv und verbunden |
| 🟡 Orange | Service teilweise verfügbar |
| 🔴 Rot | Service nicht erreichbar |

### 2.3 Automatische Aktualisierung

Das Dashboard aktualisiert sich automatisch alle **10 Sekunden**.

---

## 3. Dashboard-Bereiche

### 3.1 Header

**Elemente:**
- **Titel**: "KI Trading Dashboard"
- **Verbindungsstatus**: Zeigt Echtzeit-Verbindung zum Backend

**Status-Bedeutungen:**
- `Verbunden` - Backend erreichbar
- `Verbindung getrennt` - Backend nicht erreichbar

---

### 3.2 Übersichtskarten

Vier Informationskarten zeigen den aktuellen Systemzustand:

#### RAG Dokumente
- Anzahl der gespeicherten Dokumente im Vektorspeicher
- Mehr Dokumente = besserer historischer Kontext

#### Sync Status
- `Aktiv` - Automatische Synchronisation läuft
- `Gestoppt` - Keine automatische Synchronisation

#### Sync Intervall
- Zeigt das konfigurierte Synchronisationsintervall
- Standard: 300 Sekunden (5 Minuten)

#### LLM Modell
- Name des aktiven Ollama-Modells
- z.B. "llama3.1:70b" oder "llama3.1:8b"

---

### 3.3 TimescaleDB Sync-Steuerung

**Verbindungsinformationen:**
- Host und Datenbank
- Verbindungsstatus

**Steuerungsbuttons:**

| Button | Funktion |
|--------|----------|
| **Sync Starten** | Startet die automatische Hintergrund-Synchronisation |
| **Sync Stoppen** | Stoppt die automatische Synchronisation |
| **Manueller Sync** | Führt sofortige Synchronisation der letzten 7 Tage durch |

**Wann welche Funktion nutzen:**
- **Sync Starten**: Nach Systemstart oder wenn neue Daten benötigt werden
- **Sync Stoppen**: Bei Wartungsarbeiten oder zur Ressourcenschonung
- **Manueller Sync**: Für sofortige Aktualisierung ohne zu warten

---

### 3.4 Service Health

Zeigt den Status der drei Kernkomponenten:

#### Ollama LLM
- Prüft Erreichbarkeit des LLM-Servers
- Grün = Modell geladen und bereit

#### RAG System
- Status der FAISS-Vektordatenbank
- Grün = Index geladen und funktional

#### TimescaleDB Sync
- Status des Synchronisationsdienstes
- Grün = Verbindung aktiv

---

## 4. KI Trading-Analyse

Der Hauptbereich für Trading-Analysen mit drei Tabs:

### 4.1 Tab: Vollanalyse

**Beschreibung:** Umfassende Marktanalyse mit allen technischen Indikatoren und KI-Empfehlung.

**Eingabefelder:**

| Feld | Beschreibung | Beispiel |
|------|--------------|----------|
| Symbol | Trading-Symbol | EURUSD, GBPUSD, BTCUSD |
| Lookback Tage | Analysezeitraum in Tagen | 30 |
| Technische Indikatoren | Checkbox für Details | ✓ aktiviert |

**So führen Sie eine Analyse durch:**

1. Symbol eingeben (Autovervollständigung verfügbar)
2. Lookback-Zeitraum wählen (1-365 Tage)
3. "Technische Indikatoren einbeziehen" aktivieren für detaillierte Daten
4. "Analyse starten" klicken
5. Auf Ergebnis warten (kann 10-60 Sekunden dauern)

**Ergebnis enthält:**

```
┌─────────────────────────────────────────────┐
│  Empfehlung: BUY / SELL / HOLD              │
│  Konfidenz: 85%  ████████░░                 │
├─────────────────────────────────────────────┤
│  Entry-Preis:    1.0850                     │
│  Stop-Loss:      1.0780                     │
│  Take-Profit:    1.0950                     │
├─────────────────────────────────────────────┤
│  Technische Indikatoren:                    │
│  • RSI: 58.3                                │
│  • MACD: 0.0012                             │
│  • Bollinger: Upper 1.0920, Lower 1.0750    │
├─────────────────────────────────────────────┤
│  Risikofaktoren:                            │
│  • Hohe Volatilität erwartet                │
│  • Widerstand bei 1.0900                    │
└─────────────────────────────────────────────┘
```

---

### 4.2 Tab: Schnellempfehlung

**Beschreibung:** Schnelle BUY/SELL/HOLD-Empfehlung ohne detaillierte Analyse.

**Eingabefelder:**

| Feld | Beschreibung | Beispiel |
|------|--------------|----------|
| Symbol | Trading-Symbol | EURUSD |
| Lookback Tage | Analysezeitraum | 14 |

**Vorteile:**
- Schnellere Antwortzeit
- Übersichtliches Ergebnis
- Ideal für schnelle Entscheidungen

**Ergebnis:**
```
Symbol: EURUSD
Empfehlung: BUY
Konfidenz: 78%
```

---

### 4.3 Tab: RAG Abfrage

**Beschreibung:** Durchsucht historische Analysen und Muster.

**Eingabefelder:**

| Feld | Beschreibung | Beispiel |
|------|--------------|----------|
| Suchanfrage | Freitextsuche | "EURUSD Aufwärtstrend" |
| Symbol Filter | Optional | EURUSD |
| Max. Ergebnisse | 1-20 | 5 |

**Anwendungsfälle:**
- Ähnliche historische Situationen finden
- Vergangene Empfehlungen prüfen
- Muster-Recherche

**Beispiel-Suchanfragen:**
- "starker Aufwärtstrend RSI überkauft"
- "Seitwärtsbewegung vor Ausbruch"
- "MACD Kreuzung bullish"

**Ergebnis:**
```
Gefundene Dokumente: 5

1. [2024-01-10] EURUSD - Analyse
   Relevanz: 92%
   "Aufwärtstrend mit RSI bei 68..."

2. [2024-01-05] EURUSD - Muster
   Relevanz: 85%
   "Bullisches Muster erkannt..."
```

---

## 5. Schnellaktionen

Vier Buttons für häufig benötigte Funktionen:

### RAG Persistieren
- **Funktion:** Speichert den FAISS-Index auf Festplatte
- **Wann nutzen:**
  - Vor dem Herunterfahren des Systems
  - Nach vielen neuen Analysen
  - Als regelmäßige Sicherung

### LLM Status
- **Funktion:** Prüft ob das LLM-Modell verfügbar ist
- **Wann nutzen:**
  - Bei Verbindungsproblemen
  - Nach Ollama-Neustart
  - Zur Diagnose

### API Docs
- **Funktion:** Öffnet Swagger-Dokumentation
- **Wann nutzen:**
  - Für API-Integration
  - Zum Testen von Endpunkten
  - Für Entwickler

### Aktualisieren
- **Funktion:** Lädt alle Dashboard-Daten neu
- **Wann nutzen:**
  - Bei veralteter Anzeige
  - Nach manuellen Änderungen
  - Zur sofortigen Statusprüfung

---

## 6. Docker-Deployment

### 6.1 Vorteile des Docker-Deployments

- Isolierte Umgebung für das Frontend
- Einfache Skalierung
- Konsistentes Deployment

### 6.2 Starten mit Docker

```bash
# Im Projektverzeichnis
docker-compose up -d

# Status prüfen
docker-compose ps

# Logs anzeigen
docker-compose logs -f dashboard
```

### 6.3 Zugriff

Nach dem Start ist das Dashboard unter **http://localhost:3001** erreichbar.

### 6.4 Architektur

```
┌─────────────────┐     ┌─────────────────┐
│   Browser       │────▶│  Nginx (Docker) │
│   :3001         │     │  :80            │
└─────────────────┘     └────────┬────────┘
                                 │
                                 │ Proxy /api/*
                                 ▼
                        ┌─────────────────┐
                        │  Backend (Host) │
                        │  :8000          │
                        └─────────────────┘
```

### 6.5 Stoppen

```bash
docker-compose down
```

---

## 7. Tipps und Best Practices

### 7.1 Optimale Nutzung

#### Für Day-Trading
- Lookback: 7-14 Tage
- Häufige Schnellempfehlungen nutzen
- Auf hohe Konfidenz-Werte achten (>75%)

#### Für Swing-Trading
- Lookback: 30-60 Tage
- Vollanalyse bevorzugen
- Technische Indikatoren aktivieren

#### Für Langzeit-Analyse
- Lookback: 90-365 Tage
- RAG-Abfragen für historische Muster
- Mehrere Symbole vergleichen

### 7.2 Interpretation der Ergebnisse

**Konfidenz-Level:**

| Level | Prozent | Bedeutung |
|-------|---------|-----------|
| Sehr hoch | >85% | Starkes Signal, hohe Zuverlässigkeit |
| Hoch | 70-85% | Gutes Signal, beachtenswert |
| Mittel | 50-70% | Unsicheres Signal, vorsichtig sein |
| Niedrig | <50% | Schwaches Signal, weitere Analyse nötig |

**Signaltypen:**

| Signal | Beschreibung |
|--------|--------------|
| STRONG_BUY | Starkes Kaufsignal |
| BUY | Kaufsignal |
| HOLD | Halten, keine Aktion |
| SELL | Verkaufssignal |
| STRONG_SELL | Starkes Verkaufssignal |

### 7.3 Fehlerbehebung

#### Dashboard lädt nicht
1. Backend-Status prüfen: http://localhost:3011/api/v1/health
2. Browser-Konsole auf Fehler prüfen (F12)
3. Cache leeren und neu laden (Strg+F5)

#### Analyse dauert zu lange
- LLM-Status prüfen (Schnellaktion "LLM Status")
- Kleineren Lookback-Zeitraum wählen
- Technische Indikatoren deaktivieren

#### Keine Symbole verfügbar
- TimescaleDB-Verbindung prüfen
- Sync starten oder manuellen Sync durchführen
- Backend-Logs prüfen

#### Status-Anzeigen rot
1. Health-Check durchführen
2. Entsprechenden Service neu starten:
   - Ollama: `ollama serve`
   - Backend: `python run.py`
3. Konfiguration in `.env` prüfen

### 7.4 Tastenkürzel

| Kürzel | Funktion |
|--------|----------|
| F5 | Seite neu laden |
| Tab | Zwischen Feldern wechseln |
| Enter | Formular absenden |

### 7.5 Browser-Kompatibilität

| Browser | Status |
|---------|--------|
| Chrome 90+ | ✅ Vollständig unterstützt |
| Firefox 88+ | ✅ Vollständig unterstützt |
| Edge 90+ | ✅ Vollständig unterstützt |
| Safari 14+ | ✅ Vollständig unterstützt |
| Internet Explorer | ❌ Nicht unterstützt |

---

## Wichtige Hinweise

⚠️ **Haftungsausschluss:** Die KI-Empfehlungen dienen nur zu Informationszwecken und stellen keine Anlageberatung dar. Handeln Sie immer nach eigener Analyse und Risikobewertung.

💡 **Tipp:** Nutzen Sie die RAG-Abfrage, um historische Situationen zu finden, die der aktuellen Marktlage ähneln.

🔄 **Regelmäßig:** Führen Sie "RAG Persistieren" durch, um Ihre Analysedaten zu sichern.

---

*Letzte Aktualisierung: November 2024*
