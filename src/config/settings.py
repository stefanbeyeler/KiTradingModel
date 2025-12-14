"""Configuration management for the KI Trading Model service."""

import torch
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # TimescaleDB Direct Connection REMOVED - Using EasyInsight API only
    # All data access now goes through http://10.1.19.102:3000/api endpoints

    # EasyInsight API Configuration
    easyinsight_api_url: str = Field(
        default="http://10.1.19.102:3000/api",
        description="EasyInsight API base URL"
    )

    # RAG Sync Configuration REMOVED - TimescaleDB sync disabled

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

    # Trading Analysis Configuration
    default_lookback_days: int = Field(
        default=30,
        description="Default number of days to look back for analysis"
    )
    max_context_documents: int = Field(
        default=10,
        description="Maximum number of documents to include in RAG context"
    )

    # Performance Configuration
    use_half_precision: bool = Field(
        default=True,
        description="Use FP16 for embeddings on GPU (saves VRAM)"
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
        default=256,
        description="Hidden layer size for NHITS network"
    )
    nhits_n_pool_kernel_size: list = Field(
        default=[2, 2, 1],
        description="Hierarchical pooling kernel sizes"
    )
    nhits_batch_size: int = Field(
        default=32,
        description="Training batch size"
    )
    nhits_max_steps: int = Field(
        default=500,
        description="Maximum training steps"
    )
    nhits_learning_rate: float = Field(
        default=1e-3,
        description="Learning rate for training"
    )
    nhits_use_gpu: bool = Field(
        default=True,
        description="Use GPU for NHITS if available"
    )
    nhits_model_path: str = Field(
        default="models/nhits",
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
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @computed_field
    @property
    def gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        return torch.cuda.is_available()

    @computed_field
    @property
    def gpu_name(self) -> str:
        """Get GPU name if available."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "None"

    @computed_field
    @property
    def gpu_memory_gb(self) -> float:
        """Get GPU memory in GB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
