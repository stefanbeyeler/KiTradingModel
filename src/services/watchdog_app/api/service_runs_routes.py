"""Service Runs API Routes for Watchdog Service.

Diese Routen bieten eine Proxy-Schnittstelle zum Data Service für Service Runs.
Sie ermöglichen das Abrufen und Verwalten von Service-Run-Historie
für alle Microservices direkt vom Watchdog Dashboard.
"""

from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from ..config import settings


router = APIRouter(prefix="/service-runs", tags=["6. Service Runs"])


# Data Service URL
DATA_SERVICE_URL = f"http://{settings.data_host}:{settings.data_port}"


# ==================== Response Models ====================


class ServiceRunSummary(BaseModel):
    """Summary of a service run for list view."""

    run_id: str
    service: str
    run_type: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    status: str
    triggered_by: str
    total_items: int = 0
    items_ok: int = 0
    items_error: int = 0
    success_rate: Optional[float] = None


class ServiceRunStats(BaseModel):
    """Statistics for service runs."""

    service: Optional[str] = None
    run_type: Optional[str] = None
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    running_runs: int = 0
    avg_success_rate: Optional[float] = None
    avg_duration_ms: Optional[float] = None


# ==================== Proxy Functions ====================


async def _proxy_get(endpoint: str, params: Optional[dict] = None) -> dict:
    """Proxy GET request to Data Service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{DATA_SERVICE_URL}/api/v1/service-runs{endpoint}"
            response = await client.get(url, params=params)

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Not found")
            elif response.status_code >= 400:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Data Service error: {response.text}"
                )

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Data Service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Data Service nicht erreichbar"
        )


async def _proxy_post(endpoint: str, data: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """Proxy POST request to Data Service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{DATA_SERVICE_URL}/api/v1/service-runs{endpoint}"
            response = await client.post(url, json=data, params=params)

            if response.status_code >= 400:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Data Service error: {response.text}"
                )

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Data Service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Data Service nicht erreichbar"
        )


async def _proxy_delete(endpoint: str) -> dict:
    """Proxy DELETE request to Data Service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{DATA_SERVICE_URL}/api/v1/service-runs{endpoint}"
            response = await client.delete(url)

            if response.status_code >= 400:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Data Service error: {response.text}"
                )

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Data Service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Data Service nicht erreichbar"
        )


# ==================== Endpoints ====================


@router.get("/")
async def list_runs(
    service: Optional[str] = Query(default=None, description="Filter nach Service (data, nhits, tcn, hmm, etc.)"),
    run_type: Optional[str] = Query(default=None, description="Filter nach Typ (validation, training, scan, etc.)"),
    symbol: Optional[str] = Query(default=None, description="Filter nach Symbol"),
    status: Optional[str] = Query(default=None, description="Filter nach Status (running, completed, failed)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max. Anzahl Einträge"),
    offset: int = Query(default=0, ge=0, description="Offset für Paginierung"),
):
    """
    Service-Runs auflisten.

    Proxy-Endpoint zum Data Service für die Anzeige aller Service-Runs
    mit optionaler Filterung nach Service, Typ, Symbol oder Status.
    """
    params = {"limit": limit, "offset": offset}
    if service:
        params["service"] = service
    if run_type:
        params["run_type"] = run_type
    if symbol:
        params["symbol"] = symbol
    if status:
        params["status"] = status

    return await _proxy_get("/", params=params)


@router.get("/stats")
async def get_stats(
    service: Optional[str] = Query(default=None, description="Filter nach Service"),
    run_type: Optional[str] = Query(default=None, description="Filter nach Typ"),
):
    """
    Statistiken über Service-Runs abrufen.

    Zeigt aggregierte Statistiken wie Gesamtzahlen, Erfolgsraten
    und durchschnittliche Laufzeiten pro Service.
    """
    params = {}
    if service:
        params["service"] = service
    if run_type:
        params["run_type"] = run_type

    return await _proxy_get("/stats", params=params)


@router.get("/{run_id}")
async def get_run(run_id: str):
    """
    Details eines spezifischen Service-Runs abrufen.

    Gibt vollständige Informationen zu einem Run zurück,
    einschliesslich Input-Parameter, Ergebnisse und Metriken.
    """
    return await _proxy_get(f"/{run_id}")


@router.delete("/{run_id}")
async def delete_run(run_id: str):
    """
    Einen Service-Run löschen.
    """
    return await _proxy_delete(f"/{run_id}")


@router.post("/cleanup")
async def cleanup_old_runs(
    keep_count: int = Query(default=100, ge=10, le=1000, description="Anzahl zu behaltende Runs pro Service/Typ"),
    service: Optional[str] = Query(default=None, description="Nur für diesen Service bereinigen"),
):
    """
    Alte Service-Runs bereinigen.

    Behält die neuesten keep_count Runs pro Service/Run-Typ-Kombination.
    """
    params = {"keep_count": keep_count}
    if service:
        params["service"] = service

    return await _proxy_post("/cleanup", params=params)


@router.get("/by-service/{service_name}")
async def get_runs_by_service(
    service_name: str,
    run_type: Optional[str] = Query(default=None, description="Filter nach Typ"),
    status: Optional[str] = Query(default=None, description="Filter nach Status"),
    limit: int = Query(default=50, ge=1, le=500, description="Max. Anzahl"),
):
    """
    Runs für einen bestimmten Service abrufen.

    Convenience-Endpoint für schnellen Zugriff auf Service-spezifische Runs.
    """
    params = {"service": service_name, "limit": limit}
    if run_type:
        params["run_type"] = run_type
    if status:
        params["status"] = status

    return await _proxy_get("/", params=params)
