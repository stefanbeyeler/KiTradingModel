from .detection_router import router as detection_router
from .training_router import router as training_router
from .system_router import router as system_router

__all__ = [
    "detection_router",
    "training_router",
    "system_router",
]
