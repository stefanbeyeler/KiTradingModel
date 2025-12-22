"""
Central Microservices Configuration
====================================

Zentrale Konfigurationsdatei für alle Microservices.
Diese Datei definiert Ports, URLs und Service-Metadaten.

Verwendung:
- Docker Compose liest diese Konfiguration
- Services nutzen diese für Inter-Service-Kommunikation
- Dashboard zeigt Service-Status basierend auf dieser Config
"""

from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
from typing import Optional, Dict, Any
from enum import Enum


class ServiceType(str, Enum):
    """Kategorisierung der Services."""
    GATEWAY = "gateway"      # Frontend, API Gateway
    DATA = "data"            # Datenzugriff und -verwaltung
    ML = "ml"                # Machine Learning Services
    ANALYSIS = "analysis"    # Analyse-Services


class ServiceConfig(BaseSettings):
    """Basis-Konfiguration für einen einzelnen Service."""

    name: str
    display_name: str
    description: str
    port: int
    service_type: ServiceType

    # Container-Konfiguration
    container_name: str
    requires_gpu: bool = False
    memory_limit: str = "4G"

    # Health Check
    health_endpoint: str = "/health"
    health_start_period: int = 30  # Sekunden

    # Dependencies
    depends_on: list[str] = []

    # Model Loading
    model_load_timeout: int = 120  # Sekunden
    lazy_load_models: bool = False

    class Config:
        extra = "ignore"


class MicroservicesConfig(BaseSettings):
    """
    Zentrale Konfiguration aller Microservices.

    Ports werden aus Umgebungsvariablen geladen oder nutzen Defaults.
    """

    # =========================================================
    # Service Ports (aus Umgebungsvariablen oder Defaults)
    # =========================================================

    frontend_port: int = Field(default=3000, alias="FRONTEND_PORT")
    data_service_port: int = Field(default=3001, alias="DATA_SERVICE_PORT")
    nhits_service_port: int = Field(default=3002, alias="NHITS_SERVICE_PORT")
    tcn_service_port: int = Field(default=3003, alias="TCN_SERVICE_PORT")
    hmm_service_port: int = Field(default=3004, alias="HMM_SERVICE_PORT")
    embedder_service_port: int = Field(default=3005, alias="EMBEDDER_SERVICE_PORT")
    rag_service_port: int = Field(default=3008, alias="RAG_SERVICE_PORT")
    llm_service_port: int = Field(default=3009, alias="LLM_SERVICE_PORT")

    # =========================================================
    # Host Configuration
    # =========================================================

    # Docker-Netzwerk-Hostname (für Inter-Service-Kommunikation)
    docker_network: str = Field(default="trading-net")

    # Externe Hosts
    easyinsight_host: str = Field(default="10.1.19.102:3000")
    ollama_host: str = Field(default="host.docker.internal:11434")

    # =========================================================
    # Version
    # =========================================================

    version: str = Field(default="2.0.0")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True

    # =========================================================
    # Service Definitions
    # =========================================================

    @computed_field
    @property
    def services(self) -> Dict[str, Dict[str, Any]]:
        """Alle Service-Definitionen."""
        return {
            "frontend": {
                "name": "frontend",
                "display_name": "Frontend Dashboard",
                "description": "API Gateway & Dashboard",
                "port": self.frontend_port,
                "internal_port": 80,
                "service_type": ServiceType.GATEWAY,
                "container_name": "trading-frontend",
                "requires_gpu": False,
                "memory_limit": "512M",
                "health_endpoint": "/health",
                "health_start_period": 10,
                "depends_on": ["data-service", "nhits-service"],
            },
            "data": {
                "name": "data",
                "display_name": "Data Service",
                "description": "Symbol-Management, Strategien, Daten-Sync",
                "port": self.data_service_port,
                "service_type": ServiceType.DATA,
                "container_name": "trading-data",
                "requires_gpu": False,
                "memory_limit": "4G",
                "health_endpoint": "/health",
                "health_start_period": 20,
                "depends_on": [],
            },
            "nhits": {
                "name": "nhits",
                "display_name": "NHITS Service",
                "description": "Neural Hierarchical Interpolation - Preisprognosen",
                "port": self.nhits_service_port,
                "service_type": ServiceType.ML,
                "container_name": "trading-nhits",
                "requires_gpu": True,
                "memory_limit": "16G",
                "health_endpoint": "/health",
                "health_start_period": 40,
                "model_load_timeout": 180,
                "depends_on": ["data-service"],
            },
            "tcn": {
                "name": "tcn",
                "display_name": "TCN-Pattern Service",
                "description": "Temporal Convolutional Network - Chart Pattern Erkennung",
                "port": self.tcn_service_port,
                "service_type": ServiceType.ML,
                "container_name": "trading-tcn",
                "requires_gpu": True,
                "memory_limit": "8G",
                "health_endpoint": "/health",
                "health_start_period": 40,
                "model_load_timeout": 120,
                "depends_on": ["data-service", "embedder-service"],
            },
            "hmm": {
                "name": "hmm",
                "display_name": "HMM-Regime Service",
                "description": "Hidden Markov Model + LightGBM - Regime-Erkennung",
                "port": self.hmm_service_port,
                "service_type": ServiceType.ML,
                "container_name": "trading-hmm",
                "requires_gpu": False,
                "memory_limit": "4G",
                "health_endpoint": "/health",
                "health_start_period": 30,
                "model_load_timeout": 60,
                "depends_on": ["data-service"],
            },
            "embedder": {
                "name": "embedder",
                "display_name": "Embedder Service",
                "description": "Zentraler Embedding-Service - Text, FinBERT, TimeSeries",
                "port": self.embedder_service_port,
                "service_type": ServiceType.ML,
                "container_name": "trading-embedder",
                "requires_gpu": True,
                "memory_limit": "12G",
                "health_endpoint": "/health",
                "health_start_period": 120,
                "model_load_timeout": 180,
                "lazy_load_models": True,
                "depends_on": ["data-service"],
            },
            "rag": {
                "name": "rag",
                "display_name": "RAG Service",
                "description": "Vector Search & Knowledge Base",
                "port": self.rag_service_port,
                "service_type": ServiceType.ANALYSIS,
                "container_name": "trading-rag",
                "requires_gpu": True,
                "memory_limit": "8G",
                "health_endpoint": "/health",
                "health_start_period": 60,
                "model_load_timeout": 120,
                "depends_on": ["data-service"],
            },
            "llm": {
                "name": "llm",
                "display_name": "LLM Service",
                "description": "Large Language Model Analyse mit RAG",
                "port": self.llm_service_port,
                "service_type": ServiceType.ANALYSIS,
                "container_name": "trading-llm",
                "requires_gpu": True,
                "memory_limit": "32G",
                "health_endpoint": "/health",
                "health_start_period": 60,
                "depends_on": ["rag-service"],
            },
        }

    # =========================================================
    # URL Helpers
    # =========================================================

    def get_service_url(self, service_name: str, use_docker_network: bool = True) -> str:
        """
        Gibt die URL eines Services zurück.

        Args:
            service_name: Name des Services (z.B. 'data', 'nhits')
            use_docker_network: True für Docker-interne URLs, False für localhost

        Returns:
            Service URL (z.B. 'http://trading-data:3001')
        """
        service = self.services.get(service_name)
        if not service:
            raise ValueError(f"Unknown service: {service_name}")

        if use_docker_network:
            host = service["container_name"]
        else:
            host = "localhost"

        return f"http://{host}:{service['port']}"

    @computed_field
    @property
    def data_service_url(self) -> str:
        """URL des Data Service (für Docker-Netzwerk)."""
        return self.get_service_url("data")

    @computed_field
    @property
    def embedder_service_url(self) -> str:
        """URL des Embedder Service (für Docker-Netzwerk)."""
        return self.get_service_url("embedder")

    @computed_field
    @property
    def rag_service_url(self) -> str:
        """URL des RAG Service (für Docker-Netzwerk)."""
        return self.get_service_url("rag")

    @computed_field
    @property
    def nhits_service_url(self) -> str:
        """URL des NHITS Service (für Docker-Netzwerk)."""
        return self.get_service_url("nhits")

    @computed_field
    @property
    def easyinsight_api_url(self) -> str:
        """URL der EasyInsight API."""
        return f"http://{self.easyinsight_host}/api"

    @computed_field
    @property
    def ollama_url(self) -> str:
        """URL des Ollama Servers."""
        return f"http://{self.ollama_host}"

    # =========================================================
    # Export for Docker Compose
    # =========================================================

    def to_env_dict(self) -> Dict[str, str]:
        """
        Exportiert alle Port-Konfigurationen als Environment-Dict.
        Kann für docker-compose verwendet werden.
        """
        return {
            "FRONTEND_PORT": str(self.frontend_port),
            "DATA_SERVICE_PORT": str(self.data_service_port),
            "NHITS_SERVICE_PORT": str(self.nhits_service_port),
            "RAG_SERVICE_PORT": str(self.rag_service_port),
            "LLM_SERVICE_PORT": str(self.llm_service_port),
            "TCN_SERVICE_PORT": str(self.tcn_service_port),
            "HMM_SERVICE_PORT": str(self.hmm_service_port),
            "EMBEDDER_SERVICE_PORT": str(self.embedder_service_port),
            "EASYINSIGHT_API_URL": self.easyinsight_api_url,
            "OLLAMA_HOST": self.ollama_url,
        }

    def get_port_by_name(self, service_name: str) -> int:
        """Gibt den Port eines Services zurück."""
        port_map = {
            "frontend": self.frontend_port,
            "data": self.data_service_port,
            "nhits": self.nhits_service_port,
            "tcn": self.tcn_service_port,
            "hmm": self.hmm_service_port,
            "embedder": self.embedder_service_port,
            "rag": self.rag_service_port,
            "llm": self.llm_service_port,
        }
        return port_map.get(service_name, 3000)


# Singleton-Instanz
microservices_config = MicroservicesConfig()


# =========================================================
# Convenience Functions
# =========================================================

def get_service_port(service_name: str) -> int:
    """Gibt den Port eines Services zurück."""
    return microservices_config.get_port_by_name(service_name)


def get_service_url(service_name: str, use_docker_network: bool = True) -> str:
    """Gibt die URL eines Services zurück."""
    return microservices_config.get_service_url(service_name, use_docker_network)


def get_all_services() -> Dict[str, Dict[str, Any]]:
    """Gibt alle Service-Definitionen zurück."""
    return microservices_config.services
