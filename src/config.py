"""Configuration management for the KI Trading Model service."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # EasyInsight TimescaleDB API
    easyinsight_api_url: str = Field(
        default="http://localhost:3000",
        description="Base URL for EasyInsight TimescaleDB API"
    )
    easyinsight_api_key: str = Field(
        default="",
        description="API key for EasyInsight authentication"
    )

    # Ollama Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )
    ollama_model: str = Field(
        default="llama3.1:70b",
        description="Ollama model to use for analysis"
    )

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chromadb",
        description="Directory for ChromaDB persistence"
    )
    chroma_collection_name: str = Field(
        default="trading_history",
        description="ChromaDB collection name for trading data"
    )

    # Embedding Model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model for generating embeddings"
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
