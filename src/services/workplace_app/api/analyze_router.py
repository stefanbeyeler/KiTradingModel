"""
Analyze Router - Vertiefte Analyse Endpoints.

Integration mit RAG und LLM für detaillierte Trading-Analysen.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models.schemas import (
    DeepAnalysisRequest,
    DeepAnalysisResponse,
)
from ..services.deep_analysis_service import deep_analysis_service
from ..services.scanner_service import scanner_service

router = APIRouter()


@router.post(
    "/{symbol}",
    response_model=DeepAnalysisResponse,
    summary="Vertiefte Analyse",
    description="Führt eine vertiefte Analyse mit RAG-Kontext und LLM-Generierung durch."
)
async def deep_analyze(
    symbol: str,
    request: DeepAnalysisRequest,
):
    """
    Vertiefte Analyse mit RAG + LLM.

    Diese Analyse:
    1. Holt das aktuelle Trading-Setup (oder erstellt ein neues)
    2. Fragt den RAG-Service nach historischem Kontext
    3. Generiert eine LLM-basierte Analyse mit Empfehlungen

    Die Analyse dauert länger als ein einfacher Setup-Abruf,
    liefert aber detaillierte Einschätzungen und konkrete
    Entry/Exit-Level.
    """
    symbol = symbol.upper()

    try:
        # Bestehendes Setup aus Cache holen (falls vorhanden)
        existing_setup = scanner_service.get_setup(symbol)

        # Deep Analysis durchführen
        analysis = await deep_analysis_service.analyze(
            symbol=symbol,
            request=request,
            existing_setup=existing_setup,
        )

        logger.info(
            f"Deep Analysis für {symbol} abgeschlossen: "
            f"Score {analysis.setup.composite_score:.1f}, "
            f"RAG-Quellen: {analysis.rag_sources_used}"
        )

        return analysis

    except Exception as e:
        logger.error(f"Fehler bei Deep Analysis für {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler bei der Analyse: {str(e)}"
        )


@router.post(
    "/{symbol}/quick",
    response_model=DeepAnalysisResponse,
    summary="Schnelle Analyse",
    description="Führt eine schnelle Analyse ohne LLM durch (nur RAG-Kontext)."
)
async def quick_analyze(
    symbol: str,
    request: DeepAnalysisRequest,
):
    """
    Schnelle Analyse ohne LLM.

    Holt nur den RAG-Kontext und das Setup, überspringt aber
    die zeitaufwändige LLM-Generierung.
    """
    symbol = symbol.upper()

    try:
        # LLM deaktivieren für schnelle Analyse
        quick_request = DeepAnalysisRequest(
            timeframe=request.timeframe,
            include_rag=request.include_rag,
            include_llm=False,  # Kein LLM
        )

        existing_setup = scanner_service.get_setup(symbol)

        analysis = await deep_analysis_service.analyze(
            symbol=symbol,
            request=quick_request,
            existing_setup=existing_setup,
        )

        return analysis

    except Exception as e:
        logger.error(f"Fehler bei Quick Analysis für {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler bei der Analyse: {str(e)}"
        )
