"""TCN Training Routers."""

from .training_router import router as training_router
from .system_router import router as system_router
from .feedback_router import router as feedback_router

__all__ = ["training_router", "system_router", "feedback_router"]
