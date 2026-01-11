"""
Pydantic Schemas für Trading Workplace Service.

Definiert Request/Response-Modelle für alle API-Endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class SignalDirection(str, Enum):
    """Trading-Richtung."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class ConfidenceLevel(str, Enum):
    """Konfidenz-Stufen für Trading-Setups."""
    HIGH = "high"          # >= 75
    MODERATE = "moderate"  # >= 60
    LOW = "low"            # >= 50
    WEAK = "weak"          # < 50


class MarketRegime(str, Enum):
    """Markt-Regime Kategorien."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


class SignalAlignment(str, Enum):
    """Übereinstimmung der Signale."""
    STRONG = "strong"        # Alle Signale in gleiche Richtung
    MODERATE = "moderate"    # Mehrheit in gleiche Richtung
    MIXED = "mixed"          # Gemischte Signale
    CONFLICTING = "conflicting"  # Widersprüchliche Signale


class ScanStatus(str, Enum):
    """Scanner-Status."""
    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"


# =============================================================================
# Signal-Quellen Schemas
# =============================================================================

class NHITSSignal(BaseModel):
    """NHITS Preis-Prognose Signal."""
    available: bool = Field(default=False, description="Signal verfügbar")
    trend_probability: float = Field(default=0.5, ge=0.0, le=1.0, description="Trend-Wahrscheinlichkeit")
    direction: SignalDirection = Field(default=SignalDirection.NEUTRAL, description="Prognostizierte Richtung")
    forecast_change_1h: Optional[float] = Field(None, description="1h Preisänderung in %")
    forecast_change_24h: Optional[float] = Field(None, description="24h Preisänderung in %")
    model_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Modell-Konfidenz")


class HMMSignal(BaseModel):
    """HMM Regime-Detection Signal."""
    available: bool = Field(default=False, description="Signal verfügbar")
    regime: MarketRegime = Field(default=MarketRegime.SIDEWAYS, description="Aktuelles Regime")
    regime_probability: float = Field(default=0.5, ge=0.0, le=1.0, description="Regime-Wahrscheinlichkeit")
    signal_score: float = Field(default=50.0, ge=0.0, le=100.0, description="HMM Signal-Score")
    regime_duration: Optional[int] = Field(None, description="Dauer im aktuellen Regime (Bars)")
    alignment: str = Field(default="neutral", description="aligned, neutral, contrary")


class TCNSignal(BaseModel):
    """TCN Chart-Pattern Signal."""
    available: bool = Field(default=False, description="Signal verfügbar")
    patterns: list[str] = Field(default_factory=list, description="Erkannte Patterns")
    pattern_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Pattern-Konfidenz")
    direction: SignalDirection = Field(default=SignalDirection.NEUTRAL, description="Pattern-Richtung")
    price_target: Optional[float] = Field(None, description="Kursziel")


class CandlestickSignal(BaseModel):
    """Candlestick Pattern Signal."""
    available: bool = Field(default=False, description="Signal verfügbar")
    patterns: list[str] = Field(default_factory=list, description="Erkannte Candlestick-Patterns")
    pattern_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="Pattern-Stärke")
    direction: SignalDirection = Field(default=SignalDirection.NEUTRAL, description="Pattern-Richtung")


class TechnicalSignal(BaseModel):
    """Technische Indikatoren Signal."""
    available: bool = Field(default=False, description="Signal verfügbar")
    rsi: Optional[float] = Field(None, ge=0.0, le=100.0, description="RSI (14)")
    rsi_signal: str = Field(default="neutral", description="oversold, neutral, overbought")
    macd_signal: str = Field(default="neutral", description="bullish, neutral, bearish")
    trend_alignment: float = Field(default=0.5, ge=0.0, le=1.0, description="Trend-Übereinstimmung")
    bb_position: Optional[float] = Field(None, description="Position in Bollinger Bands (0-1)")


# =============================================================================
# Trading Setup Schemas
# =============================================================================

class TradingSetup(BaseModel):
    """Aggregiertes Trading-Setup mit Multi-Signal-Scoring."""
    symbol: str = Field(..., description="Trading-Symbol")
    timeframe: str = Field(default="H1", description="Timeframe")
    timestamp: datetime = Field(..., description="Analyse-Zeitstempel (UTC)")

    # Scoring
    direction: SignalDirection = Field(..., description="Empfohlene Richtung")
    composite_score: float = Field(..., ge=0.0, le=100.0, description="Gewichteter Composite-Score")
    confidence_level: ConfidenceLevel = Field(..., description="Konfidenz-Stufe")

    # Einzelne Signale
    nhits_signal: NHITSSignal = Field(default_factory=NHITSSignal, description="NHITS Signal")
    hmm_signal: HMMSignal = Field(default_factory=HMMSignal, description="HMM Signal")
    tcn_signal: TCNSignal = Field(default_factory=TCNSignal, description="TCN Signal")
    candlestick_signal: CandlestickSignal = Field(default_factory=CandlestickSignal, description="Candlestick Signal")
    technical_signal: TechnicalSignal = Field(default_factory=TechnicalSignal, description="Technical Signal")

    # Aggregation
    signal_alignment: SignalAlignment = Field(default=SignalAlignment.MIXED, description="Signal-Übereinstimmung")
    key_drivers: list[str] = Field(default_factory=list, description="Wichtigste Treiber")
    signals_available: int = Field(default=0, ge=0, le=5, description="Anzahl verfügbarer Signale")

    # Preise
    current_price: Optional[float] = Field(None, description="Aktueller Preis")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSD",
                "timeframe": "H1",
                "timestamp": "2024-01-15T10:30:00Z",
                "direction": "long",
                "composite_score": 72.5,
                "confidence_level": "moderate",
                "signal_alignment": "moderate",
                "key_drivers": ["NHITS: bullish (78%)", "HMM: bull_trend", "TCN: ascending_triangle"],
                "signals_available": 4
            }
        }


class SetupListResponse(BaseModel):
    """Response für Top Trading-Setups (Schnellbeurteilung)."""
    timestamp: datetime = Field(..., description="Response-Zeitstempel (UTC)")
    setups: list[TradingSetup] = Field(default_factory=list, description="Trading-Setups sortiert nach Score")
    total_scanned: int = Field(default=0, description="Anzahl gescannter Symbole")
    high_confidence_count: int = Field(default=0, description="Anzahl High-Confidence Setups")
    scan_duration_ms: Optional[float] = Field(None, description="Scan-Dauer in Millisekunden")


# =============================================================================
# Deep Analysis Schemas
# =============================================================================

class DeepAnalysisRequest(BaseModel):
    """Request für vertiefte Analyse."""
    timeframe: str = Field(default="H1", description="Timeframe für Analyse")
    include_rag: bool = Field(default=True, description="RAG-Kontext einbeziehen")
    include_llm: bool = Field(default=True, description="LLM-Analyse generieren")

    class Config:
        json_schema_extra = {
            "example": {
                "timeframe": "H1",
                "include_rag": True,
                "include_llm": True
            }
        }


class EntryExitLevels(BaseModel):
    """Entry- und Exit-Level Empfehlungen."""
    entry_price: Optional[float] = Field(None, description="Empfohlener Entry-Preis")
    stop_loss: Optional[float] = Field(None, description="Stop-Loss Level")
    take_profit_1: Optional[float] = Field(None, description="Take-Profit 1")
    take_profit_2: Optional[float] = Field(None, description="Take-Profit 2")
    take_profit_3: Optional[float] = Field(None, description="Take-Profit 3")
    risk_reward_ratio: Optional[float] = Field(None, description="Risk/Reward Ratio")


class DeepAnalysisResponse(BaseModel):
    """Response für vertiefte Analyse mit RAG + LLM."""
    symbol: str = Field(..., description="Trading-Symbol")
    timeframe: str = Field(..., description="Timeframe")
    timestamp: datetime = Field(..., description="Analyse-Zeitstempel (UTC)")

    # Basis-Setup
    setup: TradingSetup = Field(..., description="Trading-Setup")

    # RAG-Kontext
    similar_patterns: Optional[list[dict]] = Field(None, description="Ähnliche historische Patterns")
    historical_context: Optional[str] = Field(None, description="Historischer Kontext")

    # LLM-Analyse
    analysis_summary: Optional[str] = Field(None, description="LLM Analyse-Zusammenfassung")
    entry_exit_levels: Optional[EntryExitLevels] = Field(None, description="Entry/Exit Empfehlungen")
    risk_assessment: Optional[str] = Field(None, description="Risiko-Bewertung")
    rationale: Optional[str] = Field(None, description="Begründung der Empfehlung")

    # Meta
    rag_sources_used: int = Field(default=0, description="Anzahl RAG-Quellen")
    llm_model: Optional[str] = Field(None, description="Verwendetes LLM-Modell")
    analysis_duration_ms: Optional[float] = Field(None, description="Analyse-Dauer in ms")


# =============================================================================
# Watchlist Schemas
# =============================================================================

class WatchlistItem(BaseModel):
    """Einzelnes Watchlist-Item."""
    symbol: str = Field(..., description="Trading-Symbol")
    is_favorite: bool = Field(default=False, description="Als Favorit markiert")
    alert_threshold: float = Field(default=70.0, ge=0.0, le=100.0, description="Alert-Schwelle")
    timeframe: str = Field(default="H1", description="Bevorzugter Timeframe")
    last_score: Optional[float] = Field(None, description="Letzter Composite-Score")
    last_direction: Optional[SignalDirection] = Field(None, description="Letzte Richtung")
    last_scan: Optional[datetime] = Field(None, description="Letzter Scan-Zeitpunkt")
    alerts_triggered: int = Field(default=0, description="Anzahl ausgelöster Alerts")
    added_at: Optional[datetime] = Field(None, description="Hinzugefügt am")
    notes: Optional[str] = Field(None, max_length=500, description="Benutzer-Notizen")


class WatchlistAddRequest(BaseModel):
    """Request zum Hinzufügen eines Symbols zur Watchlist."""
    symbol: str = Field(..., description="Trading-Symbol")
    is_favorite: bool = Field(default=False, description="Als Favorit markieren")
    alert_threshold: float = Field(default=70.0, ge=0.0, le=100.0, description="Alert-Schwelle")
    timeframe: str = Field(default="H1", description="Bevorzugter Timeframe")
    notes: Optional[str] = Field(None, max_length=500, description="Notizen")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSD",
                "is_favorite": True,
                "alert_threshold": 75.0,
                "timeframe": "H1",
                "notes": "Wichtiger Support bei 40000"
            }
        }


class WatchlistUpdateRequest(BaseModel):
    """Request zum Aktualisieren eines Watchlist-Items."""
    is_favorite: Optional[bool] = Field(None, description="Favorit-Status")
    alert_threshold: Optional[float] = Field(None, ge=0.0, le=100.0, description="Alert-Schwelle")
    timeframe: Optional[str] = Field(None, description="Bevorzugter Timeframe")
    notes: Optional[str] = Field(None, max_length=500, description="Notizen")


class WatchlistResponse(BaseModel):
    """Response für Watchlist-Abfrage."""
    items: list[WatchlistItem] = Field(default_factory=list, description="Watchlist-Items")
    total: int = Field(default=0, description="Gesamtanzahl")
    favorites_count: int = Field(default=0, description="Anzahl Favoriten")
    last_scan: Optional[datetime] = Field(None, description="Letzter globaler Scan")


# =============================================================================
# Scanner Schemas
# =============================================================================

class ScanStatusResponse(BaseModel):
    """Response für Scanner-Status."""
    status: ScanStatus = Field(..., description="Scanner-Status")
    is_running: bool = Field(default=False, description="Scanner läuft")
    scan_interval_seconds: int = Field(default=60, description="Scan-Intervall in Sekunden")
    last_scan_time: Optional[datetime] = Field(None, description="Letzter Scan")
    next_scan_time: Optional[datetime] = Field(None, description="Nächster geplanter Scan")
    symbols_in_queue: int = Field(default=0, description="Symbole in Warteschlange")
    symbols_scanned_total: int = Field(default=0, description="Gesamt gescannte Symbole")
    current_symbol: Optional[str] = Field(None, description="Aktuell gescanntes Symbol")
    errors_count: int = Field(default=0, description="Fehleranzahl seit Start")
    alerts_triggered: int = Field(default=0, description="Ausgelöste Alerts seit Start")


class ScanTriggerResponse(BaseModel):
    """Response für manuellen Scan-Trigger."""
    success: bool = Field(..., description="Erfolgreich gestartet")
    message: str = Field(..., description="Status-Nachricht")
    symbols_to_scan: int = Field(default=0, description="Zu scannende Symbole")


# =============================================================================
# Health & Info Schemas
# =============================================================================

class ServiceHealthResponse(BaseModel):
    """Health-Check Response."""
    service: str = Field(default="workplace", description="Service-Name")
    status: str = Field(..., description="Service-Status")
    version: str = Field(..., description="Service-Version")
    timestamp: datetime = Field(..., description="Zeitstempel")
    scanner_running: bool = Field(default=False, description="Scanner läuft")
    watchlist_size: int = Field(default=0, description="Watchlist-Grösse")
    last_scan: Optional[datetime] = Field(None, description="Letzter Scan")
    services_reachable: dict[str, bool] = Field(
        default_factory=dict,
        description="Erreichbarkeit der ML-Services"
    )


# =============================================================================
# TradingView Config Schemas
# =============================================================================

class TradingViewIndicators(BaseModel):
    """TradingView Standard-Indikatoren Einstellungen."""
    sma: bool = Field(default=True, description="SMA anzeigen")
    ema: bool = Field(default=False, description="EMA anzeigen")
    rsi: bool = Field(default=True, description="RSI anzeigen")
    macd: bool = Field(default=False, description="MACD anzeigen")
    bollinger: bool = Field(default=False, description="Bollinger Bands anzeigen")
    volume: bool = Field(default=True, description="Volume anzeigen")


class TradingViewExchanges(BaseModel):
    """TradingView Exchange-Mapping Einstellungen."""
    crypto: str = Field(default="BITSTAMP", description="Crypto-Börse")
    forex: str = Field(default="FX", description="Forex-Provider")
    stocks: str = Field(default="NASDAQ", description="US-Stock Exchange")


class TradingViewConfig(BaseModel):
    """TradingView Konfiguration."""
    username: Optional[str] = Field(None, description="TradingView Benutzername")
    session_id: Optional[str] = Field(None, description="TradingView Session ID")
    default_interval: str = Field(default="60", description="Standard-Intervall")
    default_style: str = Field(default="1", description="Standard-Chart-Stil")
    theme: str = Field(default="dark", description="Theme (dark/light)")
    indicators: TradingViewIndicators = Field(
        default_factory=TradingViewIndicators,
        description="Standard-Indikatoren"
    )
    exchanges: TradingViewExchanges = Field(
        default_factory=TradingViewExchanges,
        description="Exchange-Mapping"
    )


class TradingViewTestRequest(BaseModel):
    """Request zum Testen der TradingView-Verbindung."""
    username: Optional[str] = Field(None, description="TradingView Benutzername")
    session_id: Optional[str] = Field(None, description="TradingView Session ID")


class TradingViewTestResponse(BaseModel):
    """Response für TradingView-Verbindungstest."""
    valid: bool = Field(..., description="Verbindung gültig")
    error: Optional[str] = Field(None, description="Fehlermeldung bei ungültiger Verbindung")
