"""
Embedder Service Configuration
===============================

Service-spezifische Konfiguration, erbt von der zentralen Microservices-Config.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class EmbedderSettings(BaseSettings):
    """Konfiguration für den Embedder Service."""

    # Service Identity
    service_name: str = "embedder"
    display_name: str = "Embedder Service"
    description: str = "Central Embedding Service for ML Models"
    version: str = "2.0.0"

    # Server
    port: int = Field(default=3007, alias="PORT")
    root_path: str = Field(default="", alias="ROOT_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Model Loading
    model_load_timeout: float = 180.0  # Embedder braucht länger
    lazy_load_models: bool = True  # Lazy Loading default an
    warmup_on_startup: bool = Field(default=True, alias="EMBEDDER_WARMUP")  # Alle Modelle beim Start laden

    # GPU Configuration
    use_gpu: bool = Field(default=True, alias="USE_GPU")
    gpu_device_id: int = 0

    # Text Embedding Models
    text_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model für Text Embeddings"
    )
    finbert_model: str = Field(
        default="ProsusAI/finbert",
        description="Model für Finance Text Embeddings"
    )

    # Model Paths
    model_path: Optional[str] = Field(
        default="/app/data/models/embedder",
        alias="MODEL_PATH"
    )
    cache_dir: str = Field(
        default="/app/cache",
        description="Verzeichnis für Embedding Cache"
    )

    # Cache Configuration
    cache_enabled: bool = True
    cache_max_size: int = Field(
        default=10000,
        alias="EMBEDDING_CACHE_SIZE",
        description="Maximale Anzahl Cache-Einträge"
    )
    cache_ttl_hours: int = 24

    # Embedding Dimensions (nur lesend)
    text_dimension: int = 384
    finbert_dimension: int = 768
    timeseries_dimension: int = 320
    feature_dimension: int = 128

    # Dependencies
    data_service_url: str = Field(
        default="http://trading-data:3001",
        alias="DATA_SERVICE_URL"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True


# Singleton-Instanz
settings = EmbedderSettings()
