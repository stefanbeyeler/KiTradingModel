"""Validation History API Routes.

This module provides endpoints for managing validation run history:
- Create validation runs
- Complete validation runs with results
- List validation history
- Get individual run details
- Delete runs
- Get statistics
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from ..services.validation_history_service import (
    validation_history_service,
    ValidationRunResult,
    ValidationRunSummary,
)


router = APIRouter(prefix="/validation", tags=["6. Validation History"])


# ==================== Request Models ====================


class CreateRunRequest(BaseModel):
    """Request to create a new validation run."""

    triggered_by: str = "manual"


class CompleteRunRequest(BaseModel):
    """Request to complete a validation run with results."""

    status: str = "completed"  # completed, failed, aborted
    results: dict  # Component results: {component: {ok, warning, error}}
    error_details: Optional[list] = None  # List of error objects


class ValidationStatsResponse(BaseModel):
    """Validation history statistics."""

    available: bool
    total_runs: Optional[int] = None
    completed_runs: Optional[int] = None
    failed_runs: Optional[int] = None
    avg_success_rate: Optional[float] = None
    last_run_at: Optional[str] = None
    first_run_at: Optional[str] = None


# ==================== Endpoints ====================


@router.post("/runs", response_model=dict)
async def create_validation_run(request: CreateRunRequest):
    """
    Einen neuen Validierungslauf erstellen.

    Gibt die run_id zurück, die für spätere Updates verwendet werden kann.
    """
    run_id = await validation_history_service.create_run(triggered_by=request.triggered_by)

    if not run_id:
        raise HTTPException(
            status_code=503,
            detail="Validation history service not available",
        )

    return {
        "status": "ok",
        "run_id": run_id,
        "message": "Validation run created",
    }


@router.put("/runs/{run_id}")
async def complete_validation_run(run_id: str, request: CompleteRunRequest):
    """
    Einen Validierungslauf mit Ergebnissen abschliessen.

    Die Ergebnisse werden persistent in der Datenbank gespeichert.
    """
    success = await validation_history_service.complete_run(
        run_id=run_id,
        status=request.status,
        results=request.results,
        error_details=request.error_details,
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to complete validation run",
        )

    return {
        "status": "ok",
        "run_id": run_id,
        "message": f"Validation run {request.status}",
    }


@router.get("/runs", response_model=list[ValidationRunSummary])
async def list_validation_runs(
    limit: int = Query(default=50, ge=1, le=200, description="Max. Anzahl Einträge"),
    offset: int = Query(default=0, ge=0, description="Offset für Paginierung"),
    status: Optional[str] = Query(default=None, description="Filter nach Status"),
):
    """
    Validierungsläufe auflisten.

    Gibt eine Liste aller Validierungsläufe zurück, sortiert nach Datum (neueste zuerst).
    """
    runs = await validation_history_service.list_runs(
        limit=limit,
        offset=offset,
        status=status,
    )
    return runs


@router.get("/runs/{run_id}", response_model=ValidationRunResult)
async def get_validation_run(run_id: str):
    """
    Details eines spezifischen Validierungslaufs abrufen.

    Enthält alle Ergebnisse und Fehlerdetails.
    """
    run = await validation_history_service.get_run(run_id)

    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Validation run {run_id} not found",
        )

    return run


@router.delete("/runs/{run_id}")
async def delete_validation_run(run_id: str):
    """
    Einen Validierungslauf löschen.
    """
    success = await validation_history_service.delete_run(run_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Validation run {run_id} not found or could not be deleted",
        )

    return {
        "status": "ok",
        "message": f"Validation run {run_id} deleted",
    }


@router.get("/stats", response_model=ValidationStatsResponse)
async def get_validation_stats():
    """
    Statistiken über alle Validierungsläufe abrufen.

    Zeigt Gesamtzahlen, Erfolgsraten und Zeitraum der Historie.
    """
    stats = await validation_history_service.get_stats()
    return ValidationStatsResponse(**stats)


@router.post("/cleanup")
async def cleanup_old_runs(
    keep_count: int = Query(default=100, ge=10, le=1000, description="Anzahl zu behaltender Einträge"),
):
    """
    Alte Validierungsläufe bereinigen.

    Behält die letzten `keep_count` Einträge und löscht ältere.
    """
    deleted = await validation_history_service.cleanup_old_runs(keep_count=keep_count)

    return {
        "status": "ok",
        "deleted_count": deleted,
        "message": f"{deleted} old validation runs deleted",
    }
