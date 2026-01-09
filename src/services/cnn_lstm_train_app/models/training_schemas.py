"""
Pydantic Schemas für CNN-LSTM Training Service.

Definiert Request/Response-Modelle für Training-Endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TrainingStatus(str, Enum):
    """Training-Status Enum."""
    IDLE = "idle"
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingPriority(str, Enum):
    """Training-Prioritaet."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Training Request/Response
# =============================================================================

class TrainingRequest(BaseModel):
    """Request zum Starten eines Trainings."""
    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Liste der zu trainierenden Symbole"
    )
    timeframes: list[str] = Field(
        default=["H1", "D1"],
        description="Timeframes fuer Training"
    )
    epochs: int = Field(default=100, ge=1, le=1000, description="Anzahl Epochen")
    batch_size: int = Field(default=64, ge=8, le=512, description="Batch-Groesse")
    learning_rate: float = Field(default=1e-4, gt=0, le=0.1, description="Lernrate")
    priority: TrainingPriority = Field(
        default=TrainingPriority.NORMAL,
        description="Training-Prioritaet"
    )

    # Task-Gewichtungen
    price_weight: float = Field(default=0.4, ge=0, le=1, description="Gewichtung Preis-Task")
    pattern_weight: float = Field(default=0.35, ge=0, le=1, description="Gewichtung Pattern-Task")
    regime_weight: float = Field(default=0.25, ge=0, le=1, description="Gewichtung Regime-Task")

    # Optionale Einstellungen
    early_stopping_patience: int = Field(default=10, ge=1, description="Early Stopping Patience")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Validation Split")
    save_best_only: bool = Field(default=True, description="Nur bestes Modell speichern")

    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["BTCUSD", "EURUSD", "AAPL"],
                "timeframes": ["H1", "D1"],
                "epochs": 100,
                "batch_size": 64,
                "learning_rate": 0.0001,
                "priority": "normal",
                "price_weight": 0.4,
                "pattern_weight": 0.35,
                "regime_weight": 0.25
            }
        }


class TrainingProgress(BaseModel):
    """Fortschritt des aktuellen Trainings."""
    current_epoch: int = Field(..., description="Aktuelle Epoche")
    total_epochs: int = Field(..., description="Gesamtzahl Epochen")
    current_symbol: Optional[str] = Field(None, description="Aktuelles Symbol")
    symbols_completed: int = Field(default=0, description="Abgeschlossene Symbole")
    total_symbols: int = Field(..., description="Gesamtzahl Symbole")
    current_loss: Optional[float] = Field(None, description="Aktueller Loss")
    best_loss: Optional[float] = Field(None, description="Bester Loss bisher")
    epoch_time_seconds: Optional[float] = Field(None, description="Zeit pro Epoche")
    estimated_remaining_seconds: Optional[float] = Field(None, description="Geschaetzte Restzeit")


class TrainingMetrics(BaseModel):
    """Training-Metriken."""
    total_loss: float = Field(..., description="Gesamt-Loss")
    price_loss: float = Field(..., description="Preis-Task Loss")
    pattern_loss: float = Field(..., description="Pattern-Task Loss")
    regime_loss: float = Field(..., description="Regime-Task Loss")

    # Validation Metrics
    val_total_loss: Optional[float] = Field(None, description="Validation Gesamt-Loss")
    val_price_loss: Optional[float] = Field(None, description="Validation Preis-Loss")
    val_pattern_loss: Optional[float] = Field(None, description="Validation Pattern-Loss")
    val_regime_loss: Optional[float] = Field(None, description="Validation Regime-Loss")

    # Accuracy Metrics
    price_direction_accuracy: Optional[float] = Field(None, description="Richtungs-Genauigkeit")
    pattern_f1_score: Optional[float] = Field(None, description="Pattern F1-Score")
    regime_accuracy: Optional[float] = Field(None, description="Regime-Genauigkeit")


class TrainingStatusResponse(BaseModel):
    """Response fuer Training-Status."""
    job_id: Optional[str] = Field(None, description="Job-ID")
    status: TrainingStatus = Field(..., description="Aktueller Status")
    started_at: Optional[datetime] = Field(None, description="Startzeitpunkt")
    completed_at: Optional[datetime] = Field(None, description="Endzeitpunkt")
    progress: Optional[TrainingProgress] = Field(None, description="Fortschritt")
    metrics: Optional[TrainingMetrics] = Field(None, description="Aktuelle Metriken")
    error_message: Optional[str] = Field(None, description="Fehlermeldung bei Fehler")
    model_path: Optional[str] = Field(None, description="Pfad zum trainierten Modell")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "train_20240115_103000",
                "status": "training",
                "started_at": "2024-01-15T10:30:00Z",
                "progress": {
                    "current_epoch": 25,
                    "total_epochs": 100,
                    "current_symbol": "BTCUSD",
                    "symbols_completed": 1,
                    "total_symbols": 3,
                    "current_loss": 0.0234,
                    "best_loss": 0.0198,
                    "epoch_time_seconds": 12.5,
                    "estimated_remaining_seconds": 937.5
                },
                "metrics": {
                    "total_loss": 0.0234,
                    "price_loss": 0.0156,
                    "pattern_loss": 0.0312,
                    "regime_loss": 0.0234
                }
            }
        }


# =============================================================================
# Training History
# =============================================================================

class TrainingHistoryItem(BaseModel):
    """Einzelner Eintrag in der Training-Historie."""
    job_id: str = Field(..., description="Job-ID")
    status: TrainingStatus = Field(..., description="Endstatus")
    started_at: datetime = Field(..., description="Startzeitpunkt")
    completed_at: Optional[datetime] = Field(None, description="Endzeitpunkt")
    duration_seconds: Optional[float] = Field(None, description="Dauer in Sekunden")
    symbols: list[str] = Field(..., description="Trainierte Symbole")
    timeframes: list[str] = Field(..., description="Trainierte Timeframes")
    epochs_completed: int = Field(..., description="Abgeschlossene Epochen")
    final_loss: Optional[float] = Field(None, description="Finaler Loss")
    model_path: Optional[str] = Field(None, description="Modell-Pfad")
    error_message: Optional[str] = Field(None, description="Fehlermeldung")


class TrainingHistoryResponse(BaseModel):
    """Response fuer Training-Historie."""
    history: list[TrainingHistoryItem]
    total_jobs: int
    successful_jobs: int
    failed_jobs: int


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingConfig(BaseModel):
    """Vollstaendige Training-Konfiguration."""
    # Model Architecture
    input_features: int = Field(default=25, description="Anzahl Input-Features")
    sequence_length: int = Field(default=168, description="Sequenzlaenge")
    cnn_channels: list[int] = Field(
        default=[64, 128, 256],
        description="CNN-Kanaele pro Layer"
    )
    lstm_hidden: int = Field(default=128, description="LSTM Hidden Size")
    lstm_layers: int = Field(default=2, description="Anzahl LSTM-Layer")
    dropout: float = Field(default=0.3, ge=0, le=0.5, description="Dropout-Rate")

    # Training
    epochs: int = Field(default=100, description="Anzahl Epochen")
    batch_size: int = Field(default=64, description="Batch-Groesse")
    learning_rate: float = Field(default=1e-4, description="Lernrate")
    weight_decay: float = Field(default=1e-5, description="Weight Decay")

    # Multi-Task Weights
    price_weight: float = Field(default=0.4, description="Gewichtung Preis-Task")
    pattern_weight: float = Field(default=0.35, description="Gewichtung Pattern-Task")
    regime_weight: float = Field(default=0.25, description="Gewichtung Regime-Task")

    # Scheduler
    scheduler_type: str = Field(default="cosine", description="LR Scheduler Typ")
    warmup_epochs: int = Field(default=5, description="Warmup Epochen")

    # Early Stopping
    early_stopping_patience: int = Field(default=10, description="Early Stopping Patience")
    early_stopping_min_delta: float = Field(default=1e-4, description="Min Delta")

    # Data
    validation_split: float = Field(default=0.2, description="Validation Split")
    shuffle: bool = Field(default=True, description="Daten shufflen")


# =============================================================================
# Model Management
# =============================================================================

class ModelCleanupResponse(BaseModel):
    """Response fuer Model-Cleanup."""
    deleted_models: int = Field(..., description="Anzahl geloeschter Modelle")
    freed_space_mb: float = Field(..., description="Freigegebener Speicher in MB")
    remaining_models: int = Field(..., description="Verbleibende Modelle")
    kept_models: list[str] = Field(..., description="Behaltene Modell-IDs")


class SchedulerConfig(BaseModel):
    """Konfiguration fuer automatisches Training."""
    enabled: bool = Field(default=False, description="Scheduler aktiviert")
    schedule: str = Field(default="daily", description="Schedule: hourly, daily, weekly")
    next_run: Optional[datetime] = Field(None, description="Naechste Ausfuehrung")
    symbols: list[str] = Field(default_factory=list, description="Symbole fuer Auto-Training")
    timeframes: list[str] = Field(default=["H1", "D1"], description="Timeframes")


class SchedulerStatusResponse(BaseModel):
    """Response fuer Scheduler-Status."""
    config: SchedulerConfig
    last_run: Optional[datetime] = Field(None, description="Letzte Ausfuehrung")
    last_run_status: Optional[TrainingStatus] = Field(None, description="Status der letzten Ausfuehrung")
