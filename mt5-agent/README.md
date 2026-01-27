# MT5 Trade Agent

Erfasst Trades aus MetaTrader 5 und sendet sie an den KI Trading Model Data Service.

## Voraussetzungen

- **Windows** (MT5 läuft nur unter Windows)
- **MetaTrader 5** installiert und mit einem Konto verbunden
- **Python 3.8+** installiert
- **Netzwerkzugriff** auf den Data Service (Port 3001)

## Installation

1. **Python-Abhängigkeiten installieren:**

```powershell
pip install -r requirements.txt
```

2. **Terminal registrieren:**

```powershell
# MT5 muss gestartet sein und mit einem Konto verbunden
python mt5_agent.py --register
```

Dies gibt Terminal-ID und API-Key aus. Diese Werte in `.env` speichern.

3. **Konfiguration erstellen:**

```powershell
copy .env.example .env
# .env Datei mit den Werten aus Schritt 2 editieren
```

## Konfiguration

Erstellen Sie eine `.env` Datei mit folgenden Einstellungen:

```env
# Data Service URL (anpassen an Ihre Netzwerkkonfiguration)
DATA_SERVICE_URL=http://10.1.19.101:3001

# Terminal-Registrierung (aus --register Schritt)
TERMINAL_ID=<terminal-uuid>
TERMINAL_API_KEY=<api-key>

# Polling-Intervall in Sekunden (Standard: 5)
POLL_INTERVAL=5

# Heartbeat-Intervall in Sekunden (Standard: 60)
HEARTBEAT_INTERVAL=60

# Log-Level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Optional: Log-Datei
LOG_FILE=mt5_agent.log
```

## Verwendung

### Manueller Start

```powershell
python mt5_agent.py
```

### Als Windows-Dienst (optional)

1. **NSSM installieren** (Non-Sucking Service Manager):
   - Download: https://nssm.cc/download
   - Entpacken und `nssm.exe` in einen Ordner im PATH legen

2. **Dienst installieren:**

```powershell
nssm install MT5TradeAgent "C:\Python311\python.exe" "C:\KiTradingModel\mt5-agent\mt5_agent.py"
nssm set MT5TradeAgent AppDirectory "C:\KiTradingModel\mt5-agent"
nssm set MT5TradeAgent DisplayName "MT5 Trade Agent"
nssm set MT5TradeAgent Description "Records MT5 trades for KI Trading Model"
```

3. **Dienst starten:**

```powershell
net start MT5TradeAgent
```

### Als Startup-Script

Erstellen Sie eine Verknüpfung im Autostart-Ordner:

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
```

## Mehrere Terminals

Für mehrere MT5 Terminals auf derselben Maschine:

1. Jedes Terminal separat registrieren:
   ```powershell
   # Terminal 1 starten, dann:
   python mt5_agent.py --register

   # Terminal 2 starten, dann:
   python mt5_agent.py --register
   ```

2. Separate Konfigurationsdateien erstellen:
   - `terminal1.env`
   - `terminal2.env`

3. Separate Instanzen starten:
   ```powershell
   python mt5_agent.py -c terminal1.env
   python mt5_agent.py -c terminal2.env
   ```

## Funktionsweise

Der Agent:

1. **Verbindet** sich mit MT5 beim Start
2. **Lädt** alle offenen Positionen
3. **Überwacht** in einem Polling-Zyklus:
   - Neue Positionen → Trade erstellen
   - SL/TP Änderungen → Trade aktualisieren
   - Geschlossene Trades → Trade schliessen
4. **Sendet** regelmässig Heartbeats
5. **Meldet** alle Änderungen an den Data Service

## Troubleshooting

### "MT5 initialize failed"

- Stellen Sie sicher, dass MT5 gestartet ist
- Prüfen Sie, ob MT5 mit einem Konto verbunden ist
- Aktivieren Sie "Algo Trading" in MT5

### "HTTP error"

- Prüfen Sie die Netzwerkverbindung zum Data Service
- Überprüfen Sie die `DATA_SERVICE_URL` in der Konfiguration
- Testen Sie mit: `curl http://<ip>:3001/health`

### "Invalid API key"

- Prüfen Sie, ob Terminal-ID und API-Key korrekt sind
- Registrieren Sie das Terminal erneut mit `--register`

## API-Referenz

Der Agent kommuniziert mit folgenden Data Service Endpoints:

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/v1/mt5/terminals` | POST | Terminal registrieren |
| `/api/v1/mt5/terminals/{id}/heartbeat` | POST | Heartbeat senden |
| `/api/v1/mt5/trades` | POST | Neuen Trade melden |
| `/api/v1/mt5/trades/{id}` | PUT | Trade aktualisieren |
| `/api/v1/mt5/trades` | GET | Trade-Liste abrufen |
