"""
Haupt-Router für Trading Workplace Service.

Aggregiert alle Sub-Router und definiert OpenAPI-Tags.
"""

from fastapi import APIRouter

from .setups_router import router as setups_router
from .analyze_router import router as analyze_router
from .watchlist_router import router as watchlist_router
from .scan_router import router as scan_router

# Haupt-Router
router = APIRouter()

# Sub-Router einbinden
router.include_router(
    setups_router,
    prefix="/setups",
    tags=["1. Setups"]
)

router.include_router(
    analyze_router,
    prefix="/analyze",
    tags=["2. Analyse"]
)

router.include_router(
    watchlist_router,
    prefix="/watchlist",
    tags=["3. Watchlist"]
)

router.include_router(
    scan_router,
    prefix="/scan",
    tags=["4. Scanner"]
)
