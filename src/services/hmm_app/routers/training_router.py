"""Training endpoints for HMM and LightGBM models."""

from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from ..models.schemas import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    ModelInfoResponse,
)
from ..services.regime_detection_service import regime_detection_service
from ..services.signal_scoring_service import signal_scoring_service

router = APIRouter()


@router.post("/train", response_model=TrainingResponse)
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Train HMM and/or LightGBM scorer models.

    - **symbols**: Symbols to use for training
    - **timeframe**: Timeframe for training data
    - **lookback_days**: Days of historical data
    - **model_type**: "hmm", "scorer", or "both"
    """
    try:
        results = {
            "hmm_trained": False,
            "scorer_trained": False,
            "metrics": {}
        }

        # Train HMM models
        if request.model_type in ["hmm", "both"]:
            logger.info(f"Training HMM models for {len(request.symbols)} symbols")

            hmm_results = []
            for symbol in request.symbols:
                result = await regime_detection_service.train_model(
                    symbol=symbol,
                    timeframe=request.timeframe,
                    lookback_days=request.lookback_days
                )
                hmm_results.append(result)

            results["hmm_trained"] = all(r.get("status") == "completed" for r in hmm_results)
            results["metrics"]["hmm"] = {
                "symbols_trained": len([r for r in hmm_results if r.get("status") == "completed"]),
                "total_symbols": len(request.symbols)
            }

        # Train scorer
        if request.model_type in ["scorer", "both"]:
            logger.info("Training signal scorer")

            scorer_result = await signal_scoring_service.train_scorer(
                symbols=request.symbols,
                timeframe=request.timeframe,
                lookback_days=request.lookback_days
            )

            results["scorer_trained"] = scorer_result.get("status") == "completed"
            results["metrics"]["scorer"] = scorer_result

        return TrainingResponse(
            status=TrainingStatus.COMPLETED,
            message="Training completed",
            hmm_trained=results["hmm_trained"],
            scorer_trained=results["scorer_trained"],
            metrics=results["metrics"]
        )

    except Exception as e:
        logger.error(f"Training error: {e}")
        return TrainingResponse(
            status=TrainingStatus.FAILED,
            message=str(e)
        )


@router.post("/train/hmm/{symbol}")
async def train_hmm_single(
    symbol: str,
    timeframe: str = "1h",
    lookback_days: int = 365
):
    """
    Train HMM model for a single symbol.
    """
    try:
        result = await regime_detection_service.train_model(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        return result

    except Exception as e:
        logger.error(f"HMM training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/scorer")
async def train_scorer(
    symbols: List[str],
    timeframe: str = "1h",
    lookback_days: int = 365
):
    """
    Train the LightGBM signal scorer.
    """
    try:
        result = await signal_scoring_service.train_scorer(
            symbols=symbols,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        return result

    except Exception as e:
        logger.error(f"Scorer training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about loaded models.
    """
    from ..models.hmm_regime_model import HMMRegimeModel
    from ..models.lightgbm_scorer import LightGBMSignalScorer

    # Get HMM model info
    hmm_info = {
        "loaded_symbols": list(regime_detection_service._models.keys()),
        "n_components": 4,
        "regimes": ["bull_trend", "bear_trend", "sideways", "high_volatility"]
    }

    # Get scorer info
    scorer_info = {
        "is_fitted": signal_scoring_service._scorer.is_fitted(),
        "feature_count": len(LightGBMSignalScorer.FEATURE_COLUMNS),
        "features": LightGBMSignalScorer.FEATURE_COLUMNS
    }

    return ModelInfoResponse(
        hmm_model=hmm_info,
        scorer_model=scorer_info,
        device="cpu"
    )


@router.get("/models/features")
async def get_model_features():
    """
    Get list of features used by the scorer.
    """
    from ..models.lightgbm_scorer import LightGBMSignalScorer

    features = LightGBMSignalScorer.FEATURE_COLUMNS

    return {
        "features": features,
        "count": len(features),
        "categories": {
            "regime": features[:5],
            "technical": features[5:10],
            "trend": features[10:13],
            "price_action": features[13:16],
            "volume": features[16:]
        }
    }
