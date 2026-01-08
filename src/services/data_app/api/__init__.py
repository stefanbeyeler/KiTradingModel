"""Data App API module."""

from .db_routes import router as db_router

__all__ = ["db_router"]
