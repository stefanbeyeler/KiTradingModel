from .regime_router import router as regime_router
from .scoring_router import router as scoring_router
from .training_router import router as training_router
from .system_router import router as system_router

__all__ = [
    "regime_router",
    "scoring_router",
    "training_router",
    "system_router",
]
