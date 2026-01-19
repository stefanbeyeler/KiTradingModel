"""CNN-LSTM Routers Package."""

from .prediction_router import router as prediction_router
from .system_router import router as system_router
from .revalidation_router import router as revalidation_router

__all__ = ["prediction_router", "system_router", "revalidation_router"]
