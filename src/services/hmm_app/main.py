"""
HMM-Regime Service - Market Regime Detection with Hidden Markov Models + LightGBM Scoring

Port: 3006
Features:
- Market regime detection (Bull/Bear/Sideways/High Volatility)
- Signal scoring based on regime alignment
- LightGBM-based signal evaluation
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import regime_router, scoring_router, training_router, system_router

# Configuration
VERSION = "1.0.0"
SERVICE_NAME = "hmm-regime"
SERVICE_PORT = int(os.getenv("PORT", "3006"))
ROOT_PATH = os.getenv("ROOT_PATH", "")

# Logging setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>HMM</cyan> | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger.add(
    "logs/hmm_service.log",
    rotation="10 MB",
    retention="7 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    logger.info(f"Starting HMM-Regime Service v{VERSION} on port {SERVICE_PORT}")

    # Create model directory
    os.makedirs("data/models/hmm", exist_ok=True)

    yield

    logger.info("Shutting down HMM-Regime Service...")


# Create FastAPI app
app = FastAPI(
    title="HMM-Regime Service",
    description="""
    Market regime detection and signal scoring service.

    ## Features

    ### Regime Detection (HMM)
    Uses Hidden Markov Models to classify market conditions into four regimes:
    - **Bull Trend**: Upward price movement, low volatility
    - **Bear Trend**: Downward price movement, elevated volatility
    - **Sideways**: Range-bound, low directional movement
    - **High Volatility**: Large swings, unclear direction

    ### Signal Scoring (LightGBM)
    Evaluates trading signals based on:
    - Current market regime
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price action (swing highs/lows)
    - Volume analysis

    ## Score Interpretation

    | Score | Quality | Action |
    |-------|---------|--------|
    | 80-100 | Strong | Consider taking position |
    | 60-79 | Moderate | May proceed with caution |
    | 40-59 | Weak | Use caution |
    | 0-39 | Poor | Avoid or reconsider |

    ## Regime Alignment

    - **aligned**: Signal direction matches regime (e.g., long in bull trend)
    - **neutral**: Regime is unclear or sideways
    - **contrary**: Signal opposes current regime (higher risk)
    """,
    version=VERSION,
    root_path=ROOT_PATH,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system_router, prefix="/api/v1", tags=["1. System & Monitoring"])
app.include_router(regime_router, prefix="/api/v1/regime", tags=["2. Regime Detection"])
app.include_router(scoring_router, prefix="/api/v1/scoring", tags=["3. Signal Scoring"])
app.include_router(training_router, prefix="/api/v1", tags=["4. Model Training"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "HMM-Regime Service",
        "version": VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "regime_detect": "/api/v1/regime/detect",
            "regime_history": "/api/v1/regime/history/{symbol}",
            "signal_score": "/api/v1/scoring/score",
            "evaluate_setup": "/api/v1/scoring/evaluate-setup",
            "train": "/api/v1/train",
            "models": "/api/v1/models"
        }
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "healthy", "service": SERVICE_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.hmm_app.main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False
    )
