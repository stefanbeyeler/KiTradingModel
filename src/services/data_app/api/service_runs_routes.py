"""Service Runs History API Routes.

This module provides endpoints for managing service run history across all microservices:
- Track validations, predictions, trainings, analyses, and scans
- Retrieve run history with filtering
- Get statistics per service
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from ..services.service_runs_service import (
    service_runs_service,
    ServiceRunCreate,
    ServiceRunUpdate,
    ServiceRunResult,
    ServiceRunSummary,
    ServiceRunStats,
)


router = APIRouter(prefix="/service-runs", tags=["16. Service Runs"])


# ==================== Request/Response Models ====================


class CreateRunRequest(BaseModel):
    """Request to create a new service run."""

    service: str  # data, nhits, tcn, hmm, candlestick, cnn-lstm, rag, llm, workplace, watchdog
    run_type: str  # validation, prediction, training, analysis, scan, health_check
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    triggered_by: str = "manual"  # manual, scheduled, api, auto-scan
    input_params: Optional[dict] = None


class UpdateRunRequest(BaseModel):
    """Request to update a service run."""

    status: str  # running, completed, failed, aborted
    total_items: Optional[int] = None
    items_ok: Optional[int] = None
    items_warning: Optional[int] = None
    items_error: Optional[int] = None
    success_rate: Optional[float] = None
    results: Optional[dict] = None
    metrics: Optional[dict] = None
    error_details: Optional[dict] = None


class ServiceRunListResponse(BaseModel):
    """Response for service run list."""

    total: int
    count: int
    limit: int
    offset: int
    runs: list[ServiceRunSummary]


class ServiceRunStatsResponse(BaseModel):
    """Response for service run statistics."""

    stats: ServiceRunStats
    by_service: dict[str, ServiceRunStats]


# ==================== Endpoints ====================


@router.post("/", response_model=dict)
async def create_run(request: CreateRunRequest):
    """
    Einen neuen Service-Run erstellen.

    Startet die Protokollierung eines neuen Runs (Validierung, Training, etc.).
    Gibt die run_id zurück, die zum späteren Update verwendet wird.
    """
    data = ServiceRunCreate(
        service=request.service,
        run_type=request.run_type,
        symbol=request.symbol,
        timeframe=request.timeframe,
        triggered_by=request.triggered_by,
        input_params=request.input_params,
    )

    run_id = await service_runs_service.create_run(data)

    if not run_id:
        raise HTTPException(
            status_code=503,
            detail="Service runs service not available",
        )

    return {
        "status": "ok",
        "run_id": run_id,
        "message": f"Run created for {request.service}/{request.run_type}",
    }


@router.put("/{run_id}")
async def update_run(run_id: str, request: UpdateRunRequest):
    """
    Einen Service-Run aktualisieren.

    Aktualisiert den Status und die Ergebnisse eines laufenden Runs.
    """
    update = ServiceRunUpdate(
        status=request.status,
        total_items=request.total_items,
        items_ok=request.items_ok,
        items_warning=request.items_warning,
        items_error=request.items_error,
        success_rate=request.success_rate,
        results=request.results,
        metrics=request.metrics,
        error_details=request.error_details,
    )

    success = await service_runs_service.update_run(run_id, update)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found or update failed",
        )

    return {
        "status": "ok",
        "run_id": run_id,
        "message": f"Run updated to {request.status}",
    }


@router.get("/", response_model=ServiceRunListResponse)
async def list_runs(
    service: Optional[str] = Query(default=None, description="Filter nach Service (data, nhits, tcn, etc.)"),
    run_type: Optional[str] = Query(default=None, description="Filter nach Typ (validation, training, etc.)"),
    symbol: Optional[str] = Query(default=None, description="Filter nach Symbol"),
    status: Optional[str] = Query(default=None, description="Filter nach Status (running, completed, failed)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max. Anzahl Einträge"),
    offset: int = Query(default=0, ge=0, description="Offset für Paginierung"),
):
    """
    Service-Runs auflisten.

    Gibt eine Liste aller Service-Runs zurück, sortiert nach Startzeit (neueste zuerst).
    """
    runs = await service_runs_service.list_runs(
        service=service,
        run_type=run_type,
        symbol=symbol,
        status=status,
        limit=limit,
        offset=offset,
    )

    total = await service_runs_service.count_runs(
        service=service,
        run_type=run_type,
        symbol=symbol,
        status=status,
    )

    return ServiceRunListResponse(
        total=total,
        count=len(runs),
        limit=limit,
        offset=offset,
        runs=runs,
    )


@router.get("/stats", response_model=ServiceRunStatsResponse)
async def get_stats(
    service: Optional[str] = Query(default=None, description="Filter nach Service"),
    run_type: Optional[str] = Query(default=None, description="Filter nach Typ"),
):
    """
    Statistiken über Service-Runs abrufen.

    Zeigt Gesamtzahlen, Erfolgsraten und durchschnittliche Dauern.
    """
    stats = await service_runs_service.get_stats(
        service=service,
        run_type=run_type,
    )

    by_service = await service_runs_service.get_stats_by_service()

    return ServiceRunStatsResponse(
        stats=stats,
        by_service=by_service,
    )


@router.get("/{run_id}", response_model=ServiceRunResult)
async def get_run(run_id: str):
    """
    Details eines spezifischen Service-Runs abrufen.
    """
    run = await service_runs_service.get_run(run_id)

    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found",
        )

    return run


@router.delete("/{run_id}")
async def delete_run(run_id: str):
    """
    Einen Service-Run löschen.
    """
    success = await service_runs_service.delete_run(run_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found",
        )

    return {
        "status": "ok",
        "message": f"Run {run_id} deleted",
    }


@router.post("/cleanup")
async def cleanup_old_runs(
    keep_count: int = Query(default=100, ge=10, le=1000, description="Anzahl zu behaltende Runs pro Service/Typ"),
    service: Optional[str] = Query(default=None, description="Nur für diesen Service bereinigen"),
):
    """
    Alte Service-Runs bereinigen.

    Behält die neuesten keep_count Runs pro Service/Run-Typ-Kombination.
    """
    deleted = await service_runs_service.cleanup_old_runs(
        keep_count=keep_count,
        service=service,
    )

    return {
        "status": "ok",
        "deleted_count": deleted,
        "message": f"{deleted} old runs deleted",
    }
