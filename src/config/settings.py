"""Configuration management for the KI Trading Model service."""

from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
from typing import Optional

# Optional torch import for services that don't need GPU
# (e.g., HMM uses hmmlearn/lightgbm, not PyTorch)
try:
    import torch
    _torch_available = True
except ImportError:
    torch = None  # type: ignore
    _torch_available = False


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # EasyInsight API Configuration
    # All data access goes through this REST API - no direct database connections
    easyinsight_api_url: str = Field(
        default="http://10.1.19.102:3000/api",
        description="EasyInsight API base URL"
    )

    # RAG Sync Configuration
    rag_sync_enabled: bool = Field(
        default=False,
        description="Enable automatic RAG sync with EasyInsight API"
    )
    rag_sync_interval_seconds: int = Field(
        default=3600,
        description="Interval between RAG syncs in seconds"
    )
    rag_sync_batch_size: int = Field(
        default=100,
        description="Number of records to sync per symbol per batch"
    )

    # External Sources Auto-Fetch Configuration
    external_sources_auto_fetch_enabled: bool = Field(
        default=True,
        description="Enable automatic fetching of external data sources for RAG"
    )
    external_sources_fetch_interval_minutes: int = Field(
        default=5,
        description="Interval between external sources fetches in minutes"
    )
    external_sources_symbols: str = Field(
        default="",
        description="Comma-separated list of symbols to fetch. Empty = load from Data Service"
    )
    external_sources_use_managed_symbols: bool = Field(
        default=True,
        description="If True, load symbols dynamically from Data Service managed-symbols endpoint"
    )
    external_sources_min_priority: str = Field(
        default="medium",
        description="Minimum priority level for external sources (critical, high, medium, low)"
    )

    # Ollama Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model to use for analysis"
    )
    ollama_num_ctx: int = Field(
        default=8192,
        description="Context window size for Ollama"
    )
    ollama_num_gpu: int = Field(
        default=-1,
        description="Number of GPU layers (-1 = auto, 0 = CPU only)"
    )
    ollama_num_thread: int = Field(
        default=16,
        description="Number of CPU threads for Ollama"
    )

    # FAISS RAG Configuration
    faiss_persist_directory: str = Field(
        default="data/faiss",
        description="Directory for FAISS index persistence"
    )
    rag_collection_name: str = Field(
        default="trading_history",
        description="Collection name for trading data"
    )
    faiss_use_gpu: bool = Field(
        default=True,
        description="Use GPU for FAISS if available"
    )

    # Embedding Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model for generating embeddings"
    )
    embedding_device: str = Field(
        default="auto",
        description="Device for embeddings: 'auto', 'cuda', 'cpu'"
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Batch size for embedding generation"
    )

    # Service Configuration
    service_host: str = Field(default="0.0.0.0")
    service_port: int = Field(default=3011)
    log_level: str = Field(default="INFO")

    # Microservice URLs (for inter-service communication)
    data_service_url: str = Field(
        default="http://localhost:3001",
        description="URL of the Data Service for external data access"
    )

    # ML Inference Service URLs
    nhits_service_url: str = Field(
        default="http://trading-nhits:3002",
        description="URL of the NHITS Price Forecast Service"
    )
    tcn_service_url: str = Field(
        default="http://trading-tcn:3003",
        description="URL of the TCN Chart Pattern Detection Service"
    )
    hmm_service_url: str = Field(
        default="http://trading-hmm:3004",
        description="URL of the HMM Regime Detection Service"
    )
    embedder_service_url: str = Field(
        default="http://trading-embedder:3005",
        description="URL of the Embedder Feature Embedding Service"
    )
    candlestick_service_url: str = Field(
        default="http://trading-candlestick:3006",
        description="URL of the Candlestick Pattern Detection Service"
    )
    cnn_lstm_service_url: str = Field(
        default="http://trading-cnn-lstm:3007",
        description="URL of the CNN-LSTM Multi-Task Prediction Service"
    )

    # Trading Analysis Configuration
    default_lookback_days: int = Field(
        default=30,
        description="Default number of days to look back for analysis"
    )
    max_context_documents: int = Field(
        default=10,
        description="Maximum number of documents to include in RAG context"
    )

    # Twelve Data API Configuration (Grow Plan: 377 credits/min, no daily limits)
    twelvedata_api_key: str = Field(
        default="",
        description="API key for Twelve Data market data service"
    )
    twelvedata_rate_limit: int = Field(
        default=377,
        description="Max API credits per minute for Twelve Data (377 for Grow plan)"
    )

    # Performance Configuration
    use_half_precision: bool = Field(
        default=True,
        description="Use FP16 for embeddings on GPU (saves VRAM)"
    )

    # Timezone Configuration
    display_timezone: str = Field(
        default="Europe/Zurich",
        description="Timezone for displaying timestamps (e.g., 'Europe/Zurich', 'UTC', 'America/New_York')"
    )
    date_format: str = Field(
        default="%d.%m.%Y",
        description="Date format for display (strftime format)"
    )
    time_format: str = Field(
        default="%H:%M:%S",
        description="Time format for display (strftime format)"
    )
    datetime_format: str = Field(
        default="%d.%m.%Y, %H:%M:%S",
        description="Combined datetime format for display (strftime format)"
    )

    # TimescaleDB Configuration
    timescale_host: str = Field(
        default="10.1.19.102",
        description="TimescaleDB server host"
    )
    timescale_port: int = Field(
        default=5432,
        description="TimescaleDB server port"
    )
    timescale_database: str = Field(
        default="tradingdataservice",
        description="TimescaleDB database name"
    )
    timescale_user: str = Field(
        default="trading",
        description="TimescaleDB username"
    )
    timescale_password: str = Field(
        default="",
        description="TimescaleDB password"
    )
    timescale_pool_min: int = Field(
        default=10,
        description="Minimum connections in TimescaleDB pool"
    )
    timescale_pool_max: int = Field(
        default=50,
        description="Maximum connections in TimescaleDB pool"
    )
    timescale_enabled: bool = Field(
        default=True,
        description="Enable TimescaleDB persistence layer"
    )

    @computed_field
    @property
    def timescale_dsn(self) -> str:
        """Build TimescaleDB connection string."""
        return (
            f"postgresql://{self.timescale_user}:{self.timescale_password}"
            f"@{self.timescale_host}:{self.timescale_port}"
            f"/{self.timescale_database}"
        )

    # NHITS Neural Forecasting Configuration
    nhits_enabled: bool = Field(
        default=True,
        description="Enable NHITS neural forecasting"
    )
    nhits_horizon: int = Field(
        default=24,
        description="Forecast horizon in hours"
    )
    nhits_input_size: int = Field(
        default=168,
        description="Input window size (168 = 7 days of hourly data)"
    )
    nhits_hidden_size: int = Field(
        default=128,
        description="Hidden layer size for NHITS network (128 optimal for trading data)"
    )
    nhits_n_pool_kernel_size: list = Field(
        default=[4, 2, 1],
        description="Hierarchical pooling kernel sizes (multi-scale: coarse, medium, fine)"
    )
    nhits_batch_size: int = Field(
        default=48,
        description="Training batch size (48 optimized for NVIDIA Thor GPU memory)"
    )
    nhits_max_steps: int = Field(
        default=500,
        description="Maximum training steps (reduced from 1000, early stopping typically triggers at 100-200)"
    )
    nhits_learning_rate: float = Field(
        default=3e-3,
        description="Learning rate for training (0.003 for faster convergence)"
    )
    nhits_use_gpu: bool = Field(
        default=True,
        description="Use GPU for NHITS if available"
    )
    nhits_model_path: str = Field(
        default="data/models/nhits",
        description="Directory for NHITS model persistence"
    )
    nhits_auto_retrain_days: int = Field(
        default=7,
        description="Retrain model if older than this many days"
    )
    nhits_train_on_startup: bool = Field(
        default=False,
        description="Train models for all symbols on startup"
    )
    nhits_scheduled_training_enabled: bool = Field(
        default=True,
        description="Enable scheduled periodic training"
    )
    nhits_scheduled_training_interval_hours: int = Field(
        default=24,
        description="Interval for scheduled training in hours"
    )
    nhits_training_symbols: list = Field(
        default=[],
        description="Specific symbols to train (empty = all available)"
    )

    @computed_field
    @property
    def device(self) -> str:
        """Automatically detect best available device."""
        if self.embedding_device != "auto":
            return self.embedding_device
        if _torch_available and torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @computed_field
    @property
    def gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        if not _torch_available or torch is None:
            return False
        return torch.cuda.is_available()

    @computed_field
    @property
    def gpu_name(self) -> str:
        """Get GPU name if available."""
        if not _torch_available or torch is None:
            return "None (torch not installed)"
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "None"

    @computed_field
    @property
    def gpu_memory_gb(self) -> float:
        """Get GPU memory in GB."""
        if not _torch_available or torch is None:
            return 0.0
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
