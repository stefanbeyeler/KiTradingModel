"""
System Router für CNN-LSTM Inference Service.

Endpoints für Health-Checks, Service-Info und Modell-Verwaltung.
"""

import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models.schemas import HealthResponse, ModelInfo, ModelsListResponse

router = APIRouter()

# =============================================================================
# Configuration
# =============================================================================

SERVICE_NAME = os.getenv("SERVICE_NAME", "cnn-lstm")
SERVICE_VERSION = "1.0.0"
MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/cnn-lstm")


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["1. System"])
async def detailed_health_check():
    """
    Detaillierter Health-Check mit allen Service-Informationen.

    Gibt Informationen ueber:
    - Service-Status
    - Geladenes Modell
    - GPU-Verfuegbarkeit
    - Uptime
    """
    from ..main import get_uptime, is_model_loaded, get_model_version

    # GPU Check
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    return HealthResponse(
        service=SERVICE_NAME,
        status="healthy" if is_model_loaded() else "degraded",
        version=SERVICE_VERSION,
        model_loaded=is_model_loaded(),
        model_version=get_model_version(),
        gpu_available=gpu_available,
        uptime_seconds=get_uptime(),
        timestamp=datetime.now(timezone.utc)
    )


@router.get("/info", tags=["1. System"])
async def service_info():
    """
    Allgemeine Service-Informationen.

    Gibt technische Details zum Service zurueck.
    """
    from ..main import get_uptime, is_model_loaded, get_model_version

    # PyTorch Info
    pytorch_version = None
    cuda_version = None
    gpu_name = None
    try:
        import torch
        pytorch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "description": "CNN-LSTM Multi-Task Service fuer Preis-, Pattern- und Regime-Vorhersagen",
        "port": int(os.getenv("PORT", "3007")),
        "model": {
            "loaded": is_model_loaded(),
            "version": get_model_version(),
            "directory": MODEL_DIR
        },
        "runtime": {
            "pytorch_version": pytorch_version,
            "cuda_version": cuda_version,
            "gpu_name": gpu_name,
            "uptime_seconds": get_uptime()
        },
        "endpoints": {
            "predict": "/api/v1/predict/{symbol}",
            "price": "/api/v1/price/{symbol}",
            "patterns": "/api/v1/patterns/{symbol}",
            "regime": "/api/v1/regime/{symbol}",
            "batch": "/api/v1/batch"
        },
        "supported_timeframes": [
            "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"
        ],
        "tasks": {
            "price": "Preis-Vorhersage (Regression)",
            "patterns": "Pattern-Klassifikation (16 Patterns)",
            "regime": "Regime-Vorhersage (4 Klassen)"
        }
    }


# =============================================================================
# Model Management Endpoints
# =============================================================================

@router.get("/models", response_model=ModelsListResponse, tags=["3. Models"])
async def list_models():
    """
    Liste aller verfuegbaren Modelle.

    Zeigt trainierte Modelle mit Metadaten an.
    """
    try:
        from ..services.inference_service import inference_service
        models = await inference_service.list_models()
        active_model = inference_service.get_model_version()

        return ModelsListResponse(
            models=models,
            active_model=active_model,
            total_models=len(models)
        )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/reload", tags=["3. Models"])
async def reload_model(model_id: str | None = None):
    """
    Modell neu laden.

    Laedt das aktuelle oder ein spezifisches Modell neu.
    Wird typischerweise nach dem Training aufgerufen.
    """
    try:
        from ..services.inference_service import inference_service
        from ..main import _model_loaded, _model_version

        success = await inference_service.load_model(model_id)

        if success:
            # Update global state
            import src.services.cnn_lstm_app.main as main_module
            main_module._model_loaded = True
            main_module._model_version = inference_service.get_model_version()

            return {
                "success": True,
                "model_version": inference_service.get_model_version(),
                "message": "Model successfully reloaded"
            }
        else:
            return {
                "success": False,
                "model_version": None,
                "message": "No model found to load"
            }

    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=ModelInfo, tags=["3. Models"])
async def get_model_info(model_id: str):
    """
    Detaillierte Informationen zu einem spezifischen Modell.
    """
    try:
        from ..services.inference_service import inference_service
        model_info = await inference_service.get_model_info(model_id)

        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return model_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Debug Endpoints
# =============================================================================

@router.get("/debug/config", tags=["1. System"])
async def get_config():
    """
    Service-Konfiguration anzeigen (nur fuer Debugging).
    """
    return {
        "service_name": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "port": int(os.getenv("PORT", "3007")),
        "root_path": os.getenv("ROOT_PATH", "/cnn-lstm"),
        "model_dir": MODEL_DIR,
        "data_service_url": os.getenv("DATA_SERVICE_URL", "http://trading-data:3001"),
        "environment": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "not set"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", "not set"),
        }
    }
