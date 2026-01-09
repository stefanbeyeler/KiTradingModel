"""
System Router für CNN-LSTM Training Service.

Endpoints für Health-Checks, Service-Info und Modell-Verwaltung.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ..models.training_schemas import ModelCleanupResponse

router = APIRouter()

# =============================================================================
# Configuration
# =============================================================================

SERVICE_NAME = os.getenv("SERVICE_NAME", "cnn-lstm-train")
SERVICE_VERSION = "1.0.0"
MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/cnn-lstm")
MAX_MODELS_TO_KEEP = 3


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@router.get("/health", tags=["1. System"])
async def detailed_health_check():
    """
    Detaillierter Health-Check mit allen Service-Informationen.
    """
    from ..main import get_uptime, is_training_in_progress

    # GPU Check
    gpu_available = False
    gpu_name = None
    pytorch_version = None
    try:
        import torch
        pytorch_version = torch.__version__
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": SERVICE_VERSION,
        "training_in_progress": is_training_in_progress(),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "pytorch_version": pytorch_version,
        "uptime_seconds": get_uptime(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/info", tags=["1. System"])
async def service_info():
    """
    Allgemeine Service-Informationen.
    """
    from ..main import get_uptime, is_training_in_progress

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

    # Model count
    model_count = 0
    try:
        model_path = Path(MODEL_DIR)
        if model_path.exists():
            model_count = len(list(model_path.glob("*.pt")))
    except Exception:
        pass

    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "description": "CNN-LSTM Multi-Task Training Service",
        "port": int(os.getenv("PORT", "3017")),
        "model_directory": MODEL_DIR,
        "models_available": model_count,
        "training_in_progress": is_training_in_progress(),
        "runtime": {
            "pytorch_version": pytorch_version,
            "cuda_version": cuda_version,
            "gpu_name": gpu_name,
            "uptime_seconds": get_uptime()
        },
        "endpoints": {
            "train": "POST /api/v1/train",
            "status": "GET /api/v1/train/status",
            "cancel": "POST /api/v1/train/cancel",
            "history": "GET /api/v1/train/history"
        },
        "configuration": {
            "data_service_url": os.getenv("DATA_SERVICE_URL", "http://trading-data:3001"),
            "cnn_lstm_service_url": os.getenv("CNN_LSTM_SERVICE_URL", "http://trading-cnn-lstm:3007"),
            "nice_priority": os.getenv("NICE_PRIORITY", "19")
        }
    }


# =============================================================================
# Model Management Endpoints
# =============================================================================

@router.get("/models", tags=["3. Models"])
async def list_models():
    """
    Liste aller trainierten Modelle.
    """
    try:
        model_path = Path(MODEL_DIR)
        models = []

        if model_path.exists():
            for model_file in sorted(model_path.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                stat = model_file.stat()
                models.append({
                    "model_id": model_file.stem,
                    "filename": model_file.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "is_latest": model_file.name == "latest.pt" or model_file.stem.endswith("_latest")
                })

        return {
            "models": models,
            "total_models": len(models),
            "model_directory": MODEL_DIR
        }

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/cleanup", response_model=ModelCleanupResponse, tags=["3. Models"])
async def cleanup_old_models(
    keep_count: int = Query(default=MAX_MODELS_TO_KEEP, ge=1, le=10, description="Anzahl Modelle behalten")
):
    """
    Alte Modelle aufraumen.

    Behalt die neuesten `keep_count` Modelle und loescht den Rest.
    """
    try:
        model_path = Path(MODEL_DIR)
        deleted_count = 0
        freed_space = 0
        kept_models = []

        if model_path.exists():
            # Sortiere nach Modifizierungszeit (neueste zuerst)
            model_files = sorted(
                model_path.glob("*.pt"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Behalte die neuesten
            for i, model_file in enumerate(model_files):
                if i < keep_count:
                    kept_models.append(model_file.stem)
                else:
                    # Loesche aeltere Modelle
                    size = model_file.stat().st_size
                    model_file.unlink()
                    deleted_count += 1
                    freed_space += size
                    logger.info(f"Deleted old model: {model_file.name}")

        return ModelCleanupResponse(
            deleted_models=deleted_count,
            freed_space_mb=round(freed_space / (1024 * 1024), 2),
            remaining_models=len(kept_models),
            kept_models=kept_models
        )

    except Exception as e:
        logger.error(f"Error cleaning up models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}", tags=["3. Models"])
async def delete_model(model_id: str):
    """
    Spezifisches Modell loeschen.
    """
    try:
        model_path = Path(MODEL_DIR) / f"{model_id}.pt"

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        size = model_path.stat().st_size
        model_path.unlink()

        logger.info(f"Deleted model: {model_id}")

        return {
            "success": True,
            "deleted_model": model_id,
            "freed_space_mb": round(size / (1024 * 1024), 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
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
        "port": int(os.getenv("PORT", "3017")),
        "root_path": os.getenv("ROOT_PATH", "/cnn-lstm-train"),
        "model_dir": MODEL_DIR,
        "data_service_url": os.getenv("DATA_SERVICE_URL", "http://trading-data:3001"),
        "cnn_lstm_service_url": os.getenv("CNN_LSTM_SERVICE_URL", "http://trading-cnn-lstm:3007"),
        "environment": {
            "NICE_PRIORITY": os.getenv("NICE_PRIORITY", "19"),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "not set"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", "not set"),
        }
    }
