"""Service modules for the KI Trading Model.

All imports are optional to support different container environments
with different dependencies installed.
"""

__all__ = []

# Data Gateway (minimal dependencies)
try:
    from .data_gateway_service import DataGatewayService, data_gateway
    __all__.extend(["DataGatewayService", "data_gateway"])
except ImportError:
    pass

# Symbol Service
try:
    from .symbol_service import SymbolService
    __all__.append("SymbolService")
except ImportError:
    pass

# Strategy Service
try:
    from .strategy_service import StrategyService
    __all__.append("StrategyService")
except ImportError:
    pass

# Analysis Service (requires pandas)
try:
    from .analysis_service import AnalysisService
    __all__.append("AnalysisService")
except ImportError:
    pass

# TimescaleDB Sync Service
try:
    from .timescaledb_sync_service import TimescaleDBSyncService
    __all__.append("TimescaleDBSyncService")
except ImportError:
    pass

# LLM Service (requires ollama)
try:
    from .llm_service import LLMService
    __all__.append("LLMService")
except ImportError:
    pass

# RAG Service
try:
    from .rag_service import RAGService
    __all__.append("RAGService")
except ImportError:
    pass
