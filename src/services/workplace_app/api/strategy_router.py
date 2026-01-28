"""
Strategy Router - API Endpoints für Trading-Strategien.

Verwaltet CRUD-Operationen und Export/Import von Strategien.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from loguru import logger

from ..services.strategy_service import (
    strategy_service,
    TradingStrategy,
    StrategyCreateRequest,
    StrategyUpdateRequest,
)

router = APIRouter()


@router.get("/strategies", response_model=list[TradingStrategy], tags=["4. Strategies"])
async def get_strategies(include_inactive: bool = False):
    """
    Alle Trading-Strategien abrufen.

    - **include_inactive**: Auch inaktive Strategien anzeigen
    """
    try:
        strategies = await strategy_service.get_all_strategies(include_inactive=include_inactive)
        return strategies
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/default", response_model=TradingStrategy, tags=["4. Strategies"])
async def get_default_strategy():
    """Die Standard-Strategie abrufen."""
    try:
        strategy = await strategy_service.get_default_strategy()
        if not strategy:
            raise HTTPException(status_code=404, detail="No default strategy found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get default strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}", response_model=TradingStrategy, tags=["4. Strategies"])
async def get_strategy(strategy_id: str):
    """Eine spezifische Strategie nach ID abrufen."""
    try:
        strategy = await strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies", response_model=TradingStrategy, tags=["4. Strategies"])
async def create_strategy(request: StrategyCreateRequest):
    """Eine neue Trading-Strategie erstellen."""
    try:
        strategy = await strategy_service.create_strategy(request)
        return strategy
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/strategies/{strategy_id}", response_model=TradingStrategy, tags=["4. Strategies"])
async def update_strategy(strategy_id: str, request: StrategyUpdateRequest):
    """Eine bestehende Strategie aktualisieren."""
    try:
        strategy = await strategy_service.update_strategy(strategy_id, request)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/strategies/{strategy_id}", tags=["4. Strategies"])
async def delete_strategy(strategy_id: str):
    """
    Eine Strategie loeschen.

    Nur benutzerdefinierte Strategien koennen geloescht werden.
    Standard-Strategien (default_*) sind geschuetzt.
    """
    try:
        deleted = await strategy_service.delete_strategy(strategy_id)
        if not deleted:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy '{strategy_id}' could not be deleted. Default strategies cannot be deleted."
            )
        return {"status": "deleted", "strategy_id": strategy_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{strategy_id}/set-default", response_model=TradingStrategy, tags=["4. Strategies"])
async def set_default_strategy(strategy_id: str):
    """Eine Strategie als Standard festlegen."""
    try:
        strategy = await strategy_service.set_default_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set default strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}/export", response_class=PlainTextResponse, tags=["4. Strategies"])
async def export_strategy(strategy_id: str):
    """Eine Strategie als Markdown-Datei exportieren."""
    try:
        strategy = await strategy_service.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")

        markdown = strategy_service.export_strategy_to_markdown(strategy)
        return PlainTextResponse(
            content=markdown,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{strategy.id}.md"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/import", response_model=TradingStrategy, tags=["4. Strategies"])
async def import_strategy(strategy_data: dict):
    """
    Vollständige Strategie importieren (für Config-Import).

    Akzeptiert vollständige Strategy-Daten inklusive ID.
    Wenn die Strategy existiert, wird sie aktualisiert, sonst neu erstellt.
    """
    try:
        strategy = await strategy_service.import_strategy(strategy_data)
        return strategy
    except Exception as e:
        logger.error(f"Failed to import strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
