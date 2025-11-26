"""Configuration management for the KI Trading Model service."""

import torch
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # TimescaleDB Direct Connection
    timescaledb_host: str = Field(
        default="localhost",
        description="TimescaleDB host"
    )
    timescaledb_port: int = Field(
        default=5432,
        description="TimescaleDB port"
    )
    timescaledb_database: str = Field(
        default="easyinsight",
        description="TimescaleDB database name"
    )
    timescaledb_user: str = Field(
        default="postgres",
        description="TimescaleDB username"
    )
    timescaledb_password: str = Field(
        default="",
        description="TimescaleDB password"
    )

    # RAG Sync Configuration
    rag_sync_enabled: bool = Field(
        default=True,
        description="Enable automatic RAG sync from TimescaleDB"
    )
    rag_sync_interval_seconds: int = Field(
        default=300,
        description="Interval between RAG sync runs (in seconds)"
    )
    rag_sync_batch_size: int = Field(
        default=100,
        description="Number of records to process per sync batch"
    )

    # Ollama Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )
    ollama_model: str = Field(
        default="llama3.1:70b-instruct-q4_K_M",
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
    service_port: int = Field(default=8000)
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
