"""Trading Router - Trading analysis endpoints for the LLM Service."""

import os
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
import httpx

from src.config import settings

trading_router = APIRouter()

# Service URLs
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://trading-rag:3008")
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")

# Global LLM service reference (set via llm_router.set_llm_service)
_llm_service = None


def set_llm_service(service):
    """Set the global LLM service instance."""
    global _llm_service
    _llm_service = service


def get_llm_service():
    """Get the global LLM service instance."""
    if _llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    return _llm_service


class AnalysisRequest(BaseModel):
    """Trading analysis request."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSD)")
    timeframe: str = Field(default="H1", description="Timeframe for analysis")
    use_rag: bool = Field(default=True, description="Include RAG context")
    include_ml: bool = Field(default=True, description="Include ML service predictions")


class AnalysisResponse(BaseModel):
    """Trading analysis response."""
    symbol: str
    timeframe: str
    analysis: str
    recommendation: str
    confidence: float
    model: str
    rag_used: bool
    ml_signals: Optional[dict] = None


@trading_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbol(request: AnalysisRequest):
    """
    Perform comprehensive trading analysis using LLM with optional RAG context.

    Combines:
    - Historical price data
    - RAG knowledge base context
    - ML service predictions (optional)
    """
    llm_service = get_llm_service()

    try:
        # Gather context
        rag_context = []
        ml_signals = {}

        # Get RAG context
        if request.use_rag:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    params = {
                        "query": f"Analyse {request.symbol} {request.timeframe}",
                        "symbol": request.symbol,
                        "n_results": 5
                    }
                    response = await client.get(f"{RAG_SERVICE_URL}/api/v1/rag/query", params=params)
                    if response.status_code == 200:
                        results = response.json()
                        rag_context = [doc.get("content", "") for doc in results.get("results", [])]
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Get ML signals if requested
        if request.include_ml:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # NHITS forecast
                    try:
                        nhits_url = os.getenv("NHITS_SERVICE_URL", "http://trading-nhits:3002")
                        response = await client.get(
                            f"{nhits_url}/api/v1/forecast/{request.symbol}",
                            params={"timeframe": request.timeframe}
                        )
                        if response.status_code == 200:
                            ml_signals["nhits"] = response.json()
                    except Exception:
                        pass

                    # HMM regime
                    try:
                        hmm_url = os.getenv("HMM_SERVICE_URL", "http://trading-hmm:3004")
                        response = await client.get(
                            f"{hmm_url}/api/v1/regime/{request.symbol}",
                            params={"timeframe": request.timeframe}
                        )
                        if response.status_code == 200:
                            ml_signals["hmm"] = response.json()
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"ML signal fetch failed: {e}")

        # Build analysis prompt
        context_parts = []

        if rag_context:
            context_parts.append("Relevanter Kontext aus der Wissensbasis:\n" + "\n---\n".join(rag_context[:3]))

        if ml_signals:
            if "nhits" in ml_signals:
                context_parts.append(f"NHITS Prognose: {ml_signals['nhits']}")
            if "hmm" in ml_signals:
                context_parts.append(f"HMM Regime: {ml_signals['hmm']}")

        context_text = "\n\n".join(context_parts) if context_parts else ""

        system_prompt = f"""Du bist ein erfahrener Trading-Analyst. Analysiere das Symbol {request.symbol}
auf dem {request.timeframe} Timeframe. Gib eine präzise Analyse mit:
1. Aktuelle Marktsituation
2. Wichtige Unterstützungs-/Widerstandszonen
3. Trend-Einschätzung
4. Konkrete Handelsempfehlung (LONG/SHORT/NEUTRAL) mit Konfidenz (0-100%)

{context_text}"""

        user_prompt = f"Analysiere {request.symbol} auf {request.timeframe} und gib eine Trading-Empfehlung."

        # Generate analysis
        analysis = await llm_service.generate(
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=1500
        )

        # Extract recommendation and confidence from analysis
        recommendation = "NEUTRAL"
        confidence = 0.5

        analysis_upper = analysis.upper()
        if "LONG" in analysis_upper or "KAUFEN" in analysis_upper:
            recommendation = "LONG"
        elif "SHORT" in analysis_upper or "VERKAUFEN" in analysis_upper:
            recommendation = "SHORT"

        # Try to extract confidence percentage
        import re
        conf_match = re.search(r'(\d{1,3})%', analysis)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100

        return AnalysisResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            analysis=analysis,
            recommendation=recommendation,
            confidence=confidence,
            model=llm_service.model,
            rag_used=len(rag_context) > 0,
            ml_signals=ml_signals if ml_signals else None
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@trading_router.get("/symbols")
async def get_available_symbols():
    """Get list of available trading symbols from Data Service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{DATA_SERVICE_URL}/api/v1/db/coverage")
            if response.status_code == 200:
                data = response.json()
                return {
                    "symbols": list(data.get("symbols", {}).keys()),
                    "categories": data.get("categories", {}),
                    "count": data.get("symbols_count", 0)
                }
            return {"symbols": [], "categories": {}, "count": 0}
    except Exception as e:
        logger.error(f"Failed to get symbols: {e}")
        return {"symbols": [], "categories": {}, "count": 0, "error": str(e)}


@trading_router.get("/recommendation/{symbol}")
async def get_quick_recommendation(
    symbol: str,
    timeframe: str = "H1"
):
    """
    Get a quick trading recommendation without full analysis.

    Uses cached ML signals for fast response.
    """
    llm_service = get_llm_service()

    try:
        # Quick LLM query for recommendation
        prompt = f"Gib eine kurze Trading-Empfehlung (LONG/SHORT/NEUTRAL) für {symbol} auf {timeframe}. Antworte in max. 2 Sätzen."

        response = await llm_service.generate(
            prompt=prompt,
            system="Du bist ein präziser Trading-Assistent. Antworte kurz und direkt.",
            max_tokens=200
        )

        recommendation = "NEUTRAL"
        if "LONG" in response.upper():
            recommendation = "LONG"
        elif "SHORT" in response.upper():
            recommendation = "SHORT"

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "recommendation": recommendation,
            "explanation": response,
            "model": llm_service.model
        }

    except Exception as e:
        logger.error(f"Quick recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
