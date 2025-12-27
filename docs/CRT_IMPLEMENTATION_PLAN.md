# Candle Range Theory (CRT) - Implementierungsplan

## Übersicht

Dieses Dokument beschreibt die Implementierung der Candle Range Theory (CRT) als neues Modul im KiTradingModel.

**Geschätzter Aufwand:** ~16 Stunden
**Erwarteter Edge:** +0.3-0.8% pro Trade (bei korrekter Anwendung)

---

## 1. Architektur

### 1.1 Systemübersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                     CRT DETECTION MODULE                         │
│                  (Neuer Service oder TCN-Erweiterung)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Session    │    │    Range     │    │   Purge      │
│  Manager    │    │   Tracker    │    │  Detector    │
│             │    │              │    │              │
│ • EST Times │    │ • H4 High/Low│    │ • Sweep H/L  │
│ • H4 Bounds │    │ • State Mgmt │    │ • Re-Entry   │
│ • Key Hours │    │ • Validity   │    │ • Confirm    │
└──────┬──────┘    └──────┬───────┘    └──────┬───────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    DATA SERVICE       │
              │      (Port 3001)      │
              │                       │
              │ • OHLCV (H4, H1, M5)  │
              │ • Technical Levels    │
              │ • Volume Data         │
              └───────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │  HMM    │   │  NHITS  │   │   TCN   │
      │ Regime  │   │Forecast │   │ Pattern │
      │ Filter  │   │ Confirm │   │ Confirm │
      └─────────┘   └─────────┘   └─────────┘
```

### 1.2 Dateistruktur

```
src/
├── services/
│   └── crt_app/                    # Neuer CRT Service
│       ├── __init__.py
│       ├── main.py                 # FastAPI App
│       ├── api/
│       │   └── routes.py           # API Endpoints
│       ├── services/
│       │   ├── session_manager.py  # Session-Zeit Logik
│       │   ├── range_tracker.py    # CRT Range State
│       │   ├── purge_detector.py   # Purge Detection
│       │   └── crt_signal.py       # Signal-Generierung
│       └── models/
│           └── schemas.py          # Pydantic Models
├── utils/
│   └── session_utils.py            # EST Session Utilities
└── config/
    └── crt_config.py               # CRT Konfiguration
```

---

## 2. Komponenten-Details

### 2.1 Session Manager (`session_manager.py`)

**Zweck:** Erkennung der CRT-relevanten Session-Zeiten

```python
# Kernfunktionalität
class SessionManager:
    """
    Verwaltet CRT-relevante Session-Zeiten.

    Key Times (EST):
    - 01:00 AM EST = London Pre-Open
    - 05:00 AM EST = London Open
    - 09:00 AM EST = NY Open

    H4 Candle Boundaries (UTC):
    - 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    """

    CRT_KEY_HOURS_EST = [1, 5, 9]  # 1 AM, 5 AM, 9 AM EST

    def get_current_h4_candle_start(self, timestamp: datetime) -> datetime:
        """Gibt den Start der aktuellen H4-Kerze zurück."""

    def get_next_key_session(self, timestamp: datetime) -> dict:
        """Gibt die nächste Key-Session zurück."""

    def is_key_session_active(self, timestamp: datetime) -> bool:
        """Prüft ob eine Key-Session aktiv ist."""

    def get_session_type(self, timestamp: datetime) -> str:
        """Gibt den Session-Typ zurück (london_pre, london, ny)."""
```

**EST zu UTC Konvertierung:**
```python
# CRT Key Hours in UTC (Winter/Sommer)
EST_TO_UTC_WINTER = 5  # EST = UTC-5 (November-März)
EST_TO_UTC_SUMMER = 4  # EDT = UTC-4 (März-November)

# 1 AM EST = 06:00 UTC (Winter) / 05:00 UTC (Sommer)
# 5 AM EST = 10:00 UTC (Winter) / 09:00 UTC (Sommer)
# 9 AM EST = 14:00 UTC (Winter) / 13:00 UTC (Sommer)
```

### 2.2 Range Tracker (`range_tracker.py`)

**Zweck:** Tracking des CRT Range State

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

class CRTState(Enum):
    """CRT State Machine"""
    WAITING = "waiting"           # Warte auf Key-Session Candle
    RANGE_DEFINED = "range_defined"  # Range definiert, warte auf Purge
    PURGE_ABOVE = "purge_above"   # Purge über CRT High erkannt
    PURGE_BELOW = "purge_below"   # Purge unter CRT Low erkannt
    SIGNAL_LONG = "signal_long"   # Long-Signal (Re-Entry nach Purge Below)
    SIGNAL_SHORT = "signal_short" # Short-Signal (Re-Entry nach Purge Above)
    INVALIDATED = "invalidated"   # Range invalidiert (displaced ohne Re-Entry)

@dataclass
class CRTRange:
    """CRT Range Definition"""
    symbol: str
    session_type: str            # london_pre, london, ny
    candle_start: datetime       # H4 Candle Start Zeit
    crt_high: float              # Range High
    crt_low: float               # Range Low
    crt_open: float              # Candle Open
    crt_close: float             # Candle Close
    volume: float                # Candle Volume
    state: CRTState              # Aktueller State
    created_at: datetime
    purge_price: Optional[float] = None  # Preis bei Purge
    purge_time: Optional[datetime] = None
    signal_price: Optional[float] = None  # Preis bei Signal
    signal_time: Optional[datetime] = None

class RangeTracker:
    """
    Verwaltet aktive CRT Ranges pro Symbol.

    Regeln:
    - Nur EINE aktive Range pro Symbol
    - Range invalidiert nach 24h
    - Range invalidiert bei vollständigem Displacement
    """

    def __init__(self):
        self._active_ranges: dict[str, CRTRange] = {}

    def create_range(self, symbol: str, h4_candle: dict, session_type: str) -> CRTRange:
        """Erstellt eine neue CRT Range."""

    def update_state(self, symbol: str, current_price: float, current_time: datetime) -> CRTRange:
        """Aktualisiert den State basierend auf Preis."""

    def check_purge(self, symbol: str, price: float) -> Optional[str]:
        """Prüft ob ein Purge stattgefunden hat."""

    def check_reentry(self, symbol: str, close_price: float) -> bool:
        """Prüft ob Re-Entry in Range erfolgt ist."""

    def invalidate_range(self, symbol: str, reason: str) -> None:
        """Invalidiert die aktive Range."""

    def get_active_range(self, symbol: str) -> Optional[CRTRange]:
        """Gibt die aktive Range zurück."""
```

### 2.3 Purge Detector (`purge_detector.py`)

**Zweck:** Erkennung von Purge (Liquidity Sweep) und Re-Entry

```python
@dataclass
class PurgeEvent:
    """Purge Event Details"""
    direction: str              # "above" oder "below"
    purge_price: float          # Höchster/Tiefster Preis beim Purge
    purge_wick: float           # Wick-Länge über/unter Range
    purge_time: datetime
    candles_since_range: int    # Kerzen seit Range-Definition
    volume_ratio: float         # Volumen vs. Durchschnitt

@dataclass
class ReEntryEvent:
    """Re-Entry Event Details"""
    direction: str              # "long" oder "short"
    entry_price: float          # Empfohlener Entry
    stop_loss: float            # Stop Loss (über/unter Purge)
    take_profit_1: float        # TP1: Gegenüberliegende Range-Seite
    take_profit_2: float        # TP2: Nächstes HTF Level
    risk_reward: float          # Risk/Reward Ratio
    confidence: float           # Signal-Konfidenz (0-1)

class PurgeDetector:
    """
    Erkennt Purge und Re-Entry Events.

    Purge-Kriterien:
    - Preis durchbricht CRT High/Low
    - Mindestens 0.1% über/unter Range
    - Innerhalb von 12 Stunden nach Range-Definition

    Re-Entry-Kriterien:
    - Candle CLOSE zurück innerhalb der Range
    - Nicht nur Wick, sondern echter Close
    """

    PURGE_MIN_PERCENT = 0.001   # 0.1% Mindest-Purge
    PURGE_MAX_HOURS = 12        # Max. Stunden nach Range

    def detect_purge(
        self,
        crt_range: CRTRange,
        ltf_candles: list[dict]  # M5/M15 Candles
    ) -> Optional[PurgeEvent]:
        """Erkennt Purge über H4 Range."""

    def detect_reentry(
        self,
        crt_range: CRTRange,
        purge: PurgeEvent,
        ltf_candles: list[dict]
    ) -> Optional[ReEntryEvent]:
        """Erkennt Re-Entry nach Purge."""

    def calculate_targets(
        self,
        crt_range: CRTRange,
        purge: PurgeEvent,
        technical_levels: dict
    ) -> dict:
        """Berechnet Entry, SL, TP basierend auf Range und Levels."""
```

### 2.4 CRT Signal Generator (`crt_signal.py`)

**Zweck:** Generierung von Trading-Signalen mit Service-Integration

```python
@dataclass
class CRTSignal:
    """Vollständiges CRT Trading Signal"""
    symbol: str
    direction: str              # "long" oder "short"
    signal_time: datetime

    # CRT Core
    crt_range: CRTRange
    purge_event: PurgeEvent
    reentry_event: ReEntryEvent

    # Trade Parameters
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size_percent: float  # Empfohlene Position (basierend auf Risk)

    # Confidence Factors
    crt_confidence: float       # CRT Pattern Qualität (0-1)
    regime_alignment: float     # HMM Regime Alignment (0-1)
    forecast_alignment: float   # NHITS Forecast Alignment (0-1)
    pattern_alignment: float    # TCN Pattern Alignment (0-1)
    total_confidence: float     # Gewichteter Durchschnitt

    # Risk Metrics
    risk_reward_ratio: float
    max_risk_percent: float

    # Metadata
    session_type: str
    invalidation_price: float   # Bei diesem Preis: Signal ungültig

class CRTSignalGenerator:
    """
    Generiert CRT Signale mit Multi-Service Integration.

    Integration:
    - HMM Service: Regime Filter (Distribution/Accumulation)
    - NHITS Service: Directional Bias
    - TCN Service: Pattern Confluence
    """

    # Service URLs
    HMM_SERVICE_URL = "http://localhost:3004"
    NHITS_SERVICE_URL = "http://localhost:3002"
    TCN_SERVICE_URL = "http://localhost:3003"

    # Confidence Weights
    WEIGHTS = {
        "crt": 0.40,       # CRT Pattern Qualität
        "regime": 0.25,    # HMM Regime
        "forecast": 0.20,  # NHITS Forecast
        "pattern": 0.15,   # TCN Pattern
    }

    async def generate_signal(
        self,
        symbol: str,
        crt_range: CRTRange,
        purge: PurgeEvent,
        reentry: ReEntryEvent
    ) -> CRTSignal:
        """Generiert vollständiges CRT Signal mit Service-Integration."""

    async def _get_regime_alignment(self, symbol: str, direction: str) -> float:
        """Prüft HMM Regime Alignment."""
        # Distribution Phase + Short = 1.0
        # Accumulation Phase + Long = 1.0
        # Neutral = 0.5
        # Gegenteil = 0.0

    async def _get_forecast_alignment(self, symbol: str, direction: str) -> float:
        """Prüft NHITS Forecast Alignment."""
        # Forecast Richtung stimmt mit Signal = 1.0
        # Neutral Forecast = 0.5
        # Gegenteilige Richtung = 0.0

    async def _get_pattern_alignment(self, symbol: str, direction: str) -> float:
        """Prüft TCN Pattern Alignment."""
        # Bearish Pattern + Short = 1.0
        # Bullish Pattern + Long = 1.0
        # Kein Pattern = 0.5

    def _calculate_position_size(
        self,
        entry: float,
        stop_loss: float,
        confidence: float,
        max_risk_percent: float = 2.0
    ) -> float:
        """Berechnet empfohlene Positionsgröße."""
```

---

## 3. API Endpoints

### 3.1 CRT Service API (`routes.py`)

```python
# Port 3007 (oder als TCN-Erweiterung auf 3003)

@router.get("/api/v1/crt/status/{symbol}")
async def get_crt_status(symbol: str) -> dict:
    """
    Gibt den aktuellen CRT Status für ein Symbol zurück.

    Returns:
        - active_range: Aktive CRT Range (falls vorhanden)
        - state: Aktueller State (waiting, range_defined, signal, etc.)
        - next_session: Nächste Key-Session Zeit
    """

@router.get("/api/v1/crt/signal/{symbol}")
async def get_crt_signal(symbol: str) -> Optional[CRTSignal]:
    """
    Gibt ein aktives CRT Signal zurück (falls vorhanden).

    Returns:
        CRTSignal mit Entry, SL, TP und Confidence
    """

@router.get("/api/v1/crt/scan")
async def scan_all_symbols(
    symbols: list[str] = Query(default=[]),
    min_confidence: float = Query(default=0.6)
) -> list[CRTSignal]:
    """
    Scannt alle Symbole nach CRT Signalen.

    Returns:
        Liste von CRT Signalen über Confidence-Schwelle
    """

@router.post("/api/v1/crt/analyze")
async def analyze_symbol(
    symbol: str,
    include_service_integration: bool = True
) -> dict:
    """
    Vollständige CRT Analyse für ein Symbol.

    Returns:
        - crt_status: Range und State
        - session_info: Aktuelle Session
        - service_alignment: HMM, NHITS, TCN Alignment
        - signal: CRT Signal (falls aktiv)
    """

@router.get("/api/v1/crt/history/{symbol}")
async def get_signal_history(
    symbol: str,
    days: int = Query(default=30)
) -> list[dict]:
    """
    Historische CRT Signale für Backtesting.
    """
```

---

## 4. Integration mit bestehenden Services

### 4.1 HMM Regime Filter

```python
# Aufruf an HMM Service (Port 3004)
async def get_regime_for_crt(symbol: str) -> dict:
    """
    GET http://localhost:3004/api/v1/regime/{symbol}

    Mapping für CRT:
    - "accumulation" → Bevorzuge Long Signals
    - "distribution" → Bevorzuge Short Signals
    - "trending_up" → Long OK, Short mit Vorsicht
    - "trending_down" → Short OK, Long mit Vorsicht
    - "ranging" → Beide Richtungen OK
    """
```

### 4.2 NHITS Forecast Confirmation

```python
# Aufruf an NHITS Service (Port 3002)
async def get_forecast_for_crt(symbol: str) -> dict:
    """
    GET http://localhost:3002/api/v1/forecast/{symbol}

    Mapping für CRT:
    - forecast_change > +1% → Long bevorzugt
    - forecast_change < -1% → Short bevorzugt
    - |forecast_change| < 1% → Neutral
    """
```

### 4.3 TCN Pattern Confluence

```python
# Aufruf an TCN Service (Port 3003)
async def get_patterns_for_crt(symbol: str) -> dict:
    """
    GET http://localhost:3003/api/v1/detect/{symbol}?timeframe=H4

    Mapping für CRT:
    - double_top, head_and_shoulders, rising_wedge → Short bevorzugt
    - double_bottom, inv_head_shoulders, falling_wedge → Long bevorzugt
    - Kein Pattern → Neutral
    """
```

---

## 5. Konfiguration

### 5.1 CRT Config (`crt_config.py`)

```python
from pydantic import BaseSettings

class CRTConfig(BaseSettings):
    """CRT Service Konfiguration"""

    # Session Settings
    key_hours_est: list[int] = [1, 5, 9]  # Key Session Hours in EST
    htf_timeframe: str = "H4"              # Higher Timeframe für Range
    ltf_timeframe: str = "M5"              # Lower Timeframe für Entry

    # Range Settings
    range_validity_hours: int = 24         # Max. Gültigkeit einer Range
    min_range_percent: float = 0.3         # Mindest-Range-Größe in %
    max_range_percent: float = 3.0         # Max. Range-Größe in %

    # Purge Settings
    purge_min_percent: float = 0.1         # Mindest-Purge über/unter Range
    purge_max_hours: int = 12              # Max. Zeit für Purge nach Range

    # Signal Settings
    min_confidence: float = 0.6            # Mindest-Confidence für Signal
    min_risk_reward: float = 1.5           # Mindest R:R
    max_risk_percent: float = 2.0          # Max. Risiko pro Trade

    # Service Integration
    use_hmm_filter: bool = True
    use_nhits_confirmation: bool = True
    use_tcn_confluence: bool = True

    # Service URLs
    data_service_url: str = "http://localhost:3001"
    hmm_service_url: str = "http://localhost:3004"
    nhits_service_url: str = "http://localhost:3002"
    tcn_service_url: str = "http://localhost:3003"

    class Config:
        env_prefix = "CRT_"
```

---

## 6. Implementierungsreihenfolge

### Phase 1: Core Utilities (4h)

1. **Session Utilities** (`src/utils/session_utils.py`)
   - EST/EDT Timezone Handling
   - H4 Candle Boundary Berechnung
   - Key Session Detection

2. **CRT Config** (`src/config/crt_config.py`)
   - Konfigurationsklasse
   - Environment Variables

### Phase 2: Core Logic (6h)

3. **Session Manager** (`session_manager.py`)
   - Key Session Detection
   - H4 Candle Tracking

4. **Range Tracker** (`range_tracker.py`)
   - CRT Range State Machine
   - Range Validation

5. **Purge Detector** (`purge_detector.py`)
   - Purge Detection Logic
   - Re-Entry Detection

### Phase 3: Signal Generation (4h)

6. **CRT Signal Generator** (`crt_signal.py`)
   - Signal Generation
   - Service Integration (HMM, NHITS, TCN)

7. **API Routes** (`routes.py`)
   - REST Endpoints
   - Swagger Documentation

### Phase 4: Integration & Testing (2h)

8. **Docker Integration**
   - Dockerfile
   - docker-compose.microservices.yml Update

9. **Tests**
   - Unit Tests für Core Logic
   - Integration Tests mit Services

---

## 7. Beispiel-Workflow

```
1. 05:00 UTC (01:00 EST) - London Pre-Open Session
   ├─ SessionManager: "Key Session 01:00 EST aktiv"
   ├─ H4 Candle schließt um 04:00 UTC
   └─ RangeTracker: Erstelle CRT Range (High: 95000, Low: 94500)

2. 06:30 UTC - Markt bewegt sich
   ├─ Preis steigt auf 95150 (Purge Above!)
   ├─ PurgeDetector: "Purge Above erkannt (+0.16%)"
   └─ RangeTracker: State → PURGE_ABOVE

3. 07:15 UTC - Re-Entry
   ├─ M5 Candle schließt bei 94800 (zurück in Range!)
   ├─ PurgeDetector: "Re-Entry erkannt"
   └─ CRTSignalGenerator: Generiere Short Signal

4. Signal Output:
   {
     "direction": "short",
     "entry": 94800,
     "stop_loss": 95200,  // Über Purge High
     "take_profit_1": 94500,  // Range Low
     "take_profit_2": 94000,  // Nächstes HTF Level
     "risk_reward": 2.5,
     "confidence": 0.78,
     "regime_alignment": 0.85,  // HMM: Distribution
     "forecast_alignment": 0.70,  // NHITS: -1.2% predicted
     "pattern_alignment": 0.65   // TCN: Double Top
   }
```

---

## 8. Risiken & Mitigationen

| Risiko | Mitigation |
|--------|------------|
| False Breakouts | Warte auf Candle CLOSE für Re-Entry |
| Range zu klein | Min. Range Filter (0.3%) |
| Range zu groß | Max. Range Filter (3.0%) |
| Alte Ranges | 24h Invalidierung |
| Whipsaws | Multi-Service Confirmation |
| Session-Zeit Drift | DST-aware Timezone Handling |

---

## 9. Metriken & Monitoring

```python
# Zu trackende Metriken
crt_metrics = {
    "ranges_created": Counter,      # Erstellte Ranges
    "purges_detected": Counter,     # Erkannte Purges
    "signals_generated": Counter,   # Generierte Signale
    "signals_by_direction": Counter,  # Long vs Short
    "avg_confidence": Gauge,        # Durchschnittliche Confidence
    "signal_latency_ms": Histogram, # Signal-Generierung Latenz
}
```

---

## 10. Nächste Schritte

Nach erfolgreicher Implementation:

1. **Backtesting** - Historische Performance validieren
2. **Paper Trading** - Live-Validierung ohne echtes Kapital
3. **Alert System** - Push-Notifications für Signale
4. **Dashboard Integration** - CRT Status im Frontend

