from .detection_router import router as detection_router
from .system_router import router as system_router
from .history_router import router as history_router

# Note: training_router moved to tcn_train_app service

__all__ = [
    "detection_router",
    "system_router",
    "history_router",
]
