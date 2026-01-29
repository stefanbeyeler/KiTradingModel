"""
Trading Workplace Service Konfiguration.

Zentrale Einstellungen für Signal-Aggregation, Scoring und Scanning.
"""

import os
from typing import Optional

from pydantic_settings import BaseSettings

# Import zentrale Microservices-Konfiguration
from src.config.microservices import microservices_config


class WorkplaceSettings(BaseSettings):
    """Trading Workplace Service Konfiguration."""

    # Service
    workplace_port: int = 3020
    log_level: str = "INFO"

    # Scan-Konfiguration
    scan_interval_seconds: int = 60
    auto_scan_enabled: bool = True

    # Signal-Quellen URLs (aus zentraler Konfiguration, überschreibbar via Env)
    data_service_url: str = microservices_config.data_service_url
    nhits_service_url: str = microservices_config.nhits_service_url
    tcn_service_url: str = microservices_config.tcn_service_url
    hmm_service_url: str = microservices_config.hmm_service_url
    candlestick_service_url: str = microservices_config.candlestick_service_url
    cnn_lstm_service_url: str = microservices_config.cnn_lstm_service_url
    rag_service_url: str = microservices_config.rag_service_url
    llm_service_url: str = microservices_config.llm_service_url

    # HTTP-Client Timeouts
    http_timeout_seconds: float = 30.0
    deep_analysis_timeout_seconds: float = 120.0

    # Scoring-Gewichte (müssen 1.0 ergeben)
    nhits_weight: float = 0.25
    hmm_weight: float = 0.20
    cnn_lstm_weight: float = 0.20
    tcn_weight: float = 0.15
    candlestick_weight: float = 0.10
    technical_weight: float = 0.10

    # Scoring-Boni und Penalties
    alignment_bonus: float = 15.0  # Bonus wenn Signale übereinstimmen
    alignment_penalty: float = -10.0  # Penalty bei Widerspruch

    # Coverage-Penalty (fehlende Signale reduzieren Score)
    min_signals_for_full_score: int = 4  # Mindestens 4 Signale für vollen Score
    coverage_penalty_factor: float = 0.15  # 15% Reduktion pro fehlendem Signal
    min_coverage_factor: float = 0.5  # Mindestens 50% des Scores behalten

    # Alignment-Anforderungen
    min_signals_for_strong_alignment: int = 3  # Mindestens 3 Signale für STRONG
    min_signals_for_moderate_alignment: int = 2  # Mindestens 2 für MODERATE

    # Schwellwerte
    high_confidence_threshold: float = 75.0
    moderate_confidence_threshold: float = 60.0
    min_confidence_threshold: float = 50.0

    # Watchlist
    watchlist_file: str = "/app/data/watchlist.json"
    default_alert_threshold: float = 70.0
    max_watchlist_size: int = 100

    # Trading Strategies
    strategies_file: str = "/app/data/strategies.json"

    # Default Watchlist-Symbole
    default_symbols: list[str] = [
        "BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "USDJPY",
        "XAUUSD", "US500", "US100", "DE40", "AAPL"
    ]

    model_config = {
        "env_file": ".env",
        "env_prefix": "WORKPLACE_",
        "extra": "ignore"
    }

    def validate_weights(self) -> bool:
        """Validiert dass alle Gewichte 1.0 ergeben."""
        total = (
            self.nhits_weight +
            self.hmm_weight +
            self.cnn_lstm_weight +
            self.tcn_weight +
            self.candlestick_weight +
            self.technical_weight
        )
        return abs(total - 1.0) < 0.001


# Singleton Settings-Instanz
settings = WorkplaceSettings()
