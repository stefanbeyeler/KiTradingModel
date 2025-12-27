"""Candlestick Pattern Service - Routers."""

from .detection_router import router as detection_router
from .history_router import router as history_router
from .system_router import router as system_router

__all__ = [
    "detection_router",
    "history_router",
    "system_router",
]
