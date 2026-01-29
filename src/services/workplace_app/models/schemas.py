"""
Pydantic Schemas für Trading Workplace Service.

Definiert Request/Response-Modelle für alle API-Endpoints.
"""

from __future__ import annotations

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
    atr: Optional[float] = Field(None, description="ATR (14) für Level-Berechnung")
    current_price: Optional[float] = Field(None, description="Aktueller Preis")


class CNNLSTMSignal(BaseModel):
    """CNN-LSTM Multi-Task Signal."""
    available: bool = Field(default=False, description="Signal verfügbar")
    # Price Prediction
    price_direction: SignalDirection = Field(
        default=SignalDirection.NEUTRAL, description="Preis-Richtung"
    )
    price_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Preis-Konfidenz"
    )
    forecast_change_1h: Optional[float] = Field(None, description="1h Preisänderung in %")
    forecast_change_1d: Optional[float] = Field(None, description="1d Preisänderung in %")
    # Pattern Detection
    patterns: list[str] = Field(default_factory=list, description="Erkannte Patterns")
    pattern_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Stärkstes Pattern-Konfidenz"
    )
    # Regime Detection
    regime: MarketRegime = Field(default=MarketRegime.SIDEWAYS, description="Erkanntes Regime")
    regime_probability: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Regime-Wahrscheinlichkeit"
    )
    # Model Info
    model_version: Optional[str] = Field(None, description="CNN-LSTM Modell-Version")


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
    cnn_lstm_signal: "CNNLSTMSignal" = Field(default_factory=lambda: CNNLSTMSignal(), description="CNN-LSTM Signal")

    # Aggregation
    signal_alignment: SignalAlignment = Field(default=SignalAlignment.MIXED, description="Signal-Übereinstimmung")
    key_drivers: list[str] = Field(default_factory=list, description="Wichtigste Treiber")
    signals_available: int = Field(default=0, ge=0, le=6, description="Anzahl verfügbarer Signale")

    # Preise
    current_price: Optional[float] = Field(None, description="Aktueller Preis")

    # Entry/Exit Levels
    entry_exit_levels: Optional["EntryExitLevels"] = Field(
        None, description="Berechnete Entry/Exit-Levels (Entry, SL, TP1-3)"
    )

    # Multi-Timeframe Scores
    timeframe_scores: Optional[dict[str, float]] = Field(None, description="Scores pro Timeframe")

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
    timeframe: str = Field(default="H1", description="Bevorzugter Timeframe (deprecated)")
    timeframes: list[str] = Field(
        default_factory=lambda: ["M5", "M15", "H1", "H4", "D1"],
        description="Zu scannende Timeframes"
    )
    last_score: Optional[float] = Field(None, description="Bester Composite-Score über alle Timeframes")
    best_timeframe: Optional[str] = Field(None, description="Timeframe mit bestem Score")
    last_direction: Optional[SignalDirection] = Field(None, description="Letzte Richtung")
    last_scan: Optional[datetime] = Field(None, description="Letzter Scan-Zeitpunkt")
    alerts_triggered: int = Field(default=0, description="Anzahl ausgelöster Alerts")
    added_at: Optional[datetime] = Field(None, description="Hinzugefügt am")
    notes: Optional[str] = Field(None, max_length=500, description="Benutzer-Notizen")
    # Scores pro Timeframe für detaillierte Ansicht
    timeframe_scores: Optional[dict[str, float]] = Field(None, description="Scores pro Timeframe")


class WatchlistAddRequest(BaseModel):
    """Request zum Hinzufügen eines Symbols zur Watchlist."""
    symbol: str = Field(..., description="Trading-Symbol")
    is_favorite: bool = Field(default=False, description="Als Favorit markieren")
    alert_threshold: float = Field(default=70.0, ge=0.0, le=100.0, description="Alert-Schwelle")
    timeframe: str = Field(default="H1", description="Bevorzugter Timeframe (deprecated)")
    timeframes: list[str] = Field(
        default_factory=lambda: ["M5", "M15", "H1", "H4", "D1"],
        description="Zu scannende Timeframes"
    )
    notes: Optional[str] = Field(None, max_length=500, description="Notizen")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSD",
                "is_favorite": True,
                "alert_threshold": 75.0,
                "timeframes": ["M5", "H1", "H4"],
                "notes": "Wichtiger Support bei 40000"
            }
        }


class WatchlistUpdateRequest(BaseModel):
    """Request zum Aktualisieren eines Watchlist-Items."""
    is_favorite: Optional[bool] = Field(None, description="Favorit-Status")
    alert_threshold: Optional[float] = Field(None, ge=0.0, le=100.0, description="Alert-Schwelle")
    timeframe: Optional[str] = Field(None, description="Bevorzugter Timeframe (deprecated)")
    timeframes: Optional[list[str]] = Field(None, description="Zu scannende Timeframes")
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
    setups_recorded: int = Field(default=0, description="Aufgezeichnete Setups für Evaluation")


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


# =============================================================================
# MT5 Connector Schemas
# =============================================================================


class MT5TerminalStatus(str, Enum):
    """Terminal-Verbindungsstatus."""
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class MT5TradeStatus(str, Enum):
    """Trade-Status."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class MT5TradeType(str, Enum):
    """Trade-Typ."""
    BUY = "buy"
    SELL = "sell"


class MT5LinkType(str, Enum):
    """Link-Typ."""
    AUTO = "auto"
    MANUAL = "manual"


class MT5OutcomeType(str, Enum):
    """Outcome-Typ für Evaluation."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


class MT5Terminal(BaseModel):
    """MT5 Terminal Darstellung."""
    terminal_id: str = Field(..., description="Eindeutige Terminal-ID")
    name: str = Field(..., description="Terminal-Anzeigename")
    account_number: int = Field(..., description="MT5 Kontonummer")
    broker_name: Optional[str] = Field(None, description="Broker-Name")
    server: Optional[str] = Field(None, description="MT5 Server")
    account_type: str = Field(default="real", description="Kontotyp (real/demo/contest)")
    currency: str = Field(default="USD", description="Kontowährung")
    leverage: Optional[int] = Field(None, description="Hebel")
    is_active: bool = Field(default=True, description="Terminal aktiv")
    last_heartbeat: Optional[datetime] = Field(None, description="Letzter Heartbeat")
    status: MT5TerminalStatus = Field(default=MT5TerminalStatus.UNKNOWN, description="Verbindungsstatus")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_data_service(cls, data: dict) -> "MT5Terminal":
        """Erstellt ein MT5Terminal aus Data Service Response."""
        # Berechne Status basierend auf Heartbeat
        status = MT5TerminalStatus.UNKNOWN
        if data.get("last_heartbeat"):
            from datetime import timezone, timedelta
            last_hb = data["last_heartbeat"]
            if isinstance(last_hb, str):
                last_hb = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - last_hb < timedelta(minutes=5):
                status = MT5TerminalStatus.ONLINE
            else:
                status = MT5TerminalStatus.OFFLINE

        return cls(
            terminal_id=data["terminal_id"],
            name=data["name"],
            account_number=data["account_number"],
            broker_name=data.get("broker_name"),
            server=data.get("server"),
            account_type=data.get("account_type", "real"),
            currency=data.get("currency", "USD"),
            leverage=data.get("leverage"),
            is_active=data.get("is_active", True),
            last_heartbeat=data.get("last_heartbeat"),
            status=status,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class MT5Trade(BaseModel):
    """MT5 Trade Darstellung."""
    trade_id: str = Field(..., description="Eindeutige Trade-ID")
    terminal_id: str = Field(..., description="Terminal-ID")
    ticket: int = Field(..., description="MT5 Order-Ticket")
    position_id: Optional[int] = Field(None, description="MT5 Position-ID")
    symbol: str = Field(..., description="Trading-Symbol")
    trade_type: MT5TradeType = Field(..., description="Trade-Typ (buy/sell)")
    entry_time: datetime = Field(..., description="Entry-Zeitpunkt")
    entry_price: float = Field(..., description="Entry-Preis")
    volume: float = Field(..., description="Lot-Grösse")
    exit_time: Optional[datetime] = Field(None, description="Exit-Zeitpunkt")
    exit_price: Optional[float] = Field(None, description="Exit-Preis")
    stop_loss: Optional[float] = Field(None, description="Stop-Loss")
    take_profit: Optional[float] = Field(None, description="Take-Profit")
    profit: Optional[float] = Field(None, description="Gewinn/Verlust")
    profit_pips: Optional[float] = Field(None, description="Gewinn/Verlust in Pips")
    commission: Optional[float] = Field(None, description="Kommission")
    swap: Optional[float] = Field(None, description="Swap")
    status: MT5TradeStatus = Field(default=MT5TradeStatus.OPEN, description="Trade-Status")
    close_reason: Optional[str] = Field(None, description="Schluss-Grund")
    magic_number: Optional[int] = Field(None, description="EA Magic Number")
    comment: Optional[str] = Field(None, description="Trade-Kommentar")
    timeframe: Optional[str] = Field(None, description="Zeitrahmen")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Verknüpftes Setup (optional)
    linked_setup: Optional["MT5TradeSetupLink"] = None


class MT5TradeSetupLink(BaseModel):
    """Verknüpfung zwischen Trade und Trading-Setup."""
    link_id: str = Field(..., description="Eindeutige Link-ID")
    trade_id: str = Field(..., description="Trade-ID")
    setup_symbol: str = Field(..., description="Symbol des Setups")
    setup_timeframe: str = Field(..., description="Timeframe des Setups")
    setup_timestamp: datetime = Field(..., description="Zeitstempel des Setups")
    setup_direction: SignalDirection = Field(..., description="Setup-Richtung")
    setup_score: float = Field(..., ge=0.0, le=100.0, description="Setup-Score")
    setup_confidence: Optional[ConfidenceLevel] = Field(None, description="Setup-Konfidenz")

    # Signal-Komponenten zum Zeitpunkt der Verknüpfung
    nhits_direction: Optional[SignalDirection] = None
    nhits_probability: Optional[float] = None
    hmm_regime: Optional[MarketRegime] = None
    hmm_score: Optional[float] = None
    tcn_patterns: Optional[list[str]] = None
    tcn_confidence: Optional[float] = None
    candlestick_patterns: Optional[list[str]] = None
    candlestick_strength: Optional[float] = None

    # Link-Metadaten
    link_type: MT5LinkType = Field(default=MT5LinkType.AUTO, description="Verknüpfungstyp")
    link_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Link-Konfidenz")
    notes: Optional[str] = None

    # Evaluation (nach Trade-Schluss)
    followed_recommendation: Optional[bool] = Field(
        None, description="Folgte der Trade der Empfehlung?"
    )
    outcome_vs_prediction: Optional[MT5OutcomeType] = Field(
        None, description="Ergebnis vs. Vorhersage"
    )
    created_at: Optional[datetime] = None


class MT5TradeWithSetup(MT5Trade):
    """Trade mit verknüpftem Setup für Detail-Ansicht."""
    setup: Optional[MT5TradeSetupLink] = None
    terminal_name: Optional[str] = None


class MT5PerformanceMetrics(BaseModel):
    """Performance-Metriken für MT5 Trades."""
    # Handels-Statistiken
    total_trades: int = Field(default=0, description="Gesamtanzahl Trades")
    open_trades: int = Field(default=0, description="Offene Trades")
    closed_trades: int = Field(default=0, description="Geschlossene Trades")
    winning_trades: int = Field(default=0, description="Gewinnende Trades")
    losing_trades: int = Field(default=0, description="Verlierende Trades")
    win_rate: float = Field(default=0.0, description="Gewinnrate in %")

    # Profit-Metriken
    total_profit: float = Field(default=0.0, description="Gesamtgewinn")
    total_loss: float = Field(default=0.0, description="Gesamtverlust")
    net_profit: float = Field(default=0.0, description="Netto-Gewinn")
    average_win: Optional[float] = Field(None, description="Durchschnittlicher Gewinn")
    average_loss: Optional[float] = Field(None, description="Durchschnittlicher Verlust")
    profit_factor: float = Field(default=0.0, description="Profit Factor")
    max_drawdown: Optional[float] = Field(None, description="Maximaler Drawdown")

    # Kosten
    total_commission: float = Field(default=0.0, description="Gesamte Kommissionen")
    total_swap: float = Field(default=0.0, description="Gesamte Swaps")

    # Setup-verknüpfte Metriken
    trades_with_setup: int = Field(default=0, description="Trades mit Setup-Verknüpfung")
    trades_following_setup: int = Field(default=0, description="Trades die Setup folgten")
    setup_follow_rate: float = Field(default=0.0, description="Setup-Folgerate in %")
    profit_following_setup: float = Field(default=0.0, description="Gewinn bei Setup-Befolgung")
    profit_against_setup: float = Field(default=0.0, description="Gewinn gegen Setup")
    setup_prediction_accuracy: float = Field(default=0.0, description="Setup-Vorhersage-Genauigkeit in %")

    # Aufschlüsselungen
    trades_by_symbol: dict[str, int] = Field(default_factory=dict, description="Trades pro Symbol")
    trades_by_direction: dict[str, int] = Field(default_factory=dict, description="Trades pro Richtung")
    profit_by_symbol: dict[str, float] = Field(default_factory=dict, description="Profit pro Symbol")


class MT5OverviewResponse(BaseModel):
    """Response für MT5 Dashboard-Übersicht."""
    terminals: list[MT5Terminal] = Field(default_factory=list, description="Registrierte Terminals")
    terminals_online: int = Field(default=0, description="Anzahl Online-Terminals")
    terminals_total: int = Field(default=0, description="Gesamtanzahl Terminals")
    recent_trades: list[MT5Trade] = Field(default_factory=list, description="Letzte Trades")
    open_trades: int = Field(default=0, description="Offene Trades")
    metrics: MT5PerformanceMetrics = Field(
        default_factory=MT5PerformanceMetrics, description="Performance-Metriken"
    )
    last_updated: datetime = Field(default_factory=lambda: datetime.now(), description="Letzte Aktualisierung")


class MT5TradeListResponse(BaseModel):
    """Response für Trade-Liste."""
    trades: list[MT5TradeWithSetup] = Field(default_factory=list, description="Trade-Liste")
    total: int = Field(default=0, description="Gesamtanzahl")
    has_more: bool = Field(default=False, description="Weitere Trades verfügbar")


class MT5TerminalListResponse(BaseModel):
    """Response für Terminal-Liste."""
    terminals: list[MT5Terminal] = Field(default_factory=list, description="Terminal-Liste")
    total: int = Field(default=0, description="Gesamtanzahl")


class MT5LinkRequest(BaseModel):
    """Request zum manuellen Verknüpfen eines Trades mit einem Setup."""
    setup_timestamp: datetime = Field(..., description="Zeitstempel des Setups")
    setup_timeframe: str = Field(default="H1", description="Timeframe des Setups")
    notes: Optional[str] = Field(None, max_length=500, description="Notizen zur Verknüpfung")
