"""
System Router für CNN-LSTM Inference Service.

Endpoints für Health-Checks, Service-Info und Modell-Verwaltung.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models.schemas import GPUMetricsResponse, HealthResponse, ModelInfo, ModelsListResponse

# Optional GPU monitoring via nvidia-ml-py
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.info("pynvml not available - GPU metrics will use torch fallback")


def _read_gpu_temperature_from_thermal_zone() -> float:
    """
    Read GPU temperature from Linux thermal zones (Tegra/Jetson fallback).

    Returns temperature in Celsius, or 0.0 if unavailable.
    """
    thermal_base = Path("/sys/class/thermal")
    if not thermal_base.exists():
        return 0.0

    try:
        for zone_dir in thermal_base.iterdir():
            if not zone_dir.is_dir():
                continue
            type_file = zone_dir / "type"
            temp_file = zone_dir / "temp"
            if type_file.exists() and temp_file.exists():
                zone_type = type_file.read_text().strip()
                if zone_type == "gpu-thermal":
                    temp_millidegrees = int(temp_file.read_text().strip())
                    return temp_millidegrees / 1000.0
    except Exception as e:
        logger.debug(f"Could not read GPU thermal zone: {e}")

    return 0.0

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


@router.get("/gpu", response_model=GPUMetricsResponse, tags=["1. System"])
async def get_gpu_metrics():
    """
    Detaillierte GPU-Metriken fuer Watchdog-Integration.

    Dieser Endpoint wird vom Watchdog Service verwendet, um GPU-Ressourcen
    zu ueberwachen, wenn der Watchdog selbst keinen direkten GPU-Zugang hat.

    Verwendet pynvml (nvidia-ml-py) fuer detaillierte Metriken, mit PyTorch
    als Fallback fuer grundlegende Informationen.
    """
    timestamp = datetime.now(timezone.utc)

    # Try pynvml first for detailed metrics
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total_mb = mem_info.total / (1024 * 1024)
            memory_used_mb = mem_info.used / (1024 * 1024)
            memory_free_mb = mem_info.free / (1024 * 1024)
            memory_percent = (mem_info.used / mem_info.total) * 100

            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
            except pynvml.NVMLError:
                gpu_utilization = 0.0

            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temperature = 0.0

            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
            except pynvml.NVMLError:
                power = 0.0

            pynvml.nvmlShutdown()

            return GPUMetricsResponse(
                available=True,
                index=0,
                name=name,
                memory_total_mb=round(memory_total_mb, 2),
                memory_used_mb=round(memory_used_mb, 2),
                memory_free_mb=round(memory_free_mb, 2),
                memory_percent=round(memory_percent, 1),
                utilization_percent=round(gpu_utilization, 1),
                temperature_celsius=round(temperature, 1),
                power_usage_watts=round(power, 1),
                is_healthy=True,
                error_message=None,
                timestamp=timestamp,
                source="pynvml"
            )

        except Exception as e:
            logger.warning(f"pynvml error, falling back to torch: {e}")
            # Fall through to torch fallback

    # PyTorch fallback - limited metrics but better than nothing
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            memory_total_mb = props.total_memory / (1024 * 1024)

            # Get current memory usage via torch
            memory_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            memory_reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
            memory_free_mb = memory_total_mb - memory_reserved_mb
            memory_percent = (memory_reserved_mb / memory_total_mb) * 100

            # Try to get temperature from thermal zones (Tegra/Jetson)
            temperature = _read_gpu_temperature_from_thermal_zone()

            return GPUMetricsResponse(
                available=True,
                index=0,
                name=gpu_name,
                memory_total_mb=round(memory_total_mb, 2),
                memory_used_mb=round(memory_reserved_mb, 2),
                memory_free_mb=round(memory_free_mb, 2),
                memory_percent=round(memory_percent, 1),
                utilization_percent=0.0,  # Not available via torch
                temperature_celsius=round(temperature, 1),
                power_usage_watts=0.0,  # Not available via torch
                is_healthy=True,
                error_message=None if temperature > 0 else "Limited metrics (torch fallback)",
                timestamp=timestamp,
                source="torch+sysfs" if temperature > 0 else "torch"
            )
    except Exception as e:
        logger.warning(f"GPU metrics unavailable: {e}")

    # No GPU available
    return GPUMetricsResponse(
        available=False,
        index=0,
        name="None",
        memory_total_mb=0.0,
        memory_used_mb=0.0,
        memory_free_mb=0.0,
        memory_percent=0.0,
        utilization_percent=0.0,
        temperature_celsius=0.0,
        power_usage_watts=0.0,
        is_healthy=False,
        error_message="No GPU detected",
        timestamp=timestamp,
        source="none"
    )


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
