"""
Inference Service für CNN-LSTM Multi-Task Model.

Laedt trainierte Modelle und fuehrt Vorhersagen durch.
Enthaelt robustes CUDA Error Handling fuer GPU-basierte Inferenz.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..models.schemas import (
    MarketRegime,
    ModelInfo,
    MultiTaskPrediction,
    PatternPrediction,
    PatternType,
    PriceDirection,
    PricePrediction,
    PredictionResponse,
    RegimePrediction,
    RegimeTransitionProbs,
)
from .feature_service import feature_service

# Lazy import fuer PyTorch
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch


class CUDAError(Exception):
    """Custom exception for CUDA-related errors."""
    pass


def _handle_cuda_error(func_name: str, error: Exception) -> None:
    """
    Handles CUDA errors with proper logging and cleanup.

    Args:
        func_name: Name of the function where error occurred
        error: The exception that was raised
    """
    error_str = str(error).lower()

    # Check for known CUDA error patterns
    is_cuda_error = any(pattern in error_str for pattern in [
        "cuda", "gpu", "illegal memory access", "out of memory",
        "device-side assert", "cublas", "cudnn", "nccl"
    ])

    if is_cuda_error:
        logger.error(f"CUDA error in {func_name}: {error}")

        # Attempt recovery
        _ensure_torch()
        if torch.cuda.is_available():
            try:
                # Synchronize to ensure error is captured
                torch.cuda.synchronize()
                # Clear GPU memory cache
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared after CUDA error")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup GPU after CUDA error: {cleanup_error}")
    else:
        logger.error(f"Error in {func_name}: {error}")


def _safe_cuda_operation(operation_name: str):
    """
    Decorator for safe CUDA operations with automatic error handling.

    Wraps GPU operations to catch CUDA errors and attempt recovery.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except RuntimeError as e:
                _handle_cuda_error(operation_name, e)
                raise CUDAError(f"CUDA operation failed in {operation_name}: {e}") from e
            except Exception as e:
                _handle_cuda_error(operation_name, e)
                raise
        return wrapper
    return decorator


def _get_gpu_memory_info() -> dict:
    """
    Get current GPU memory usage information.

    Returns:
        Dictionary with GPU memory stats or empty dict if not available
    """
    _ensure_torch()

    if not torch.cuda.is_available():
        return {}

    try:
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        }
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")
        return {}


# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = os.getenv("MODEL_DIR", "/app/data/models/cnn-lstm")
SERVICE_VERSION = "1.0.0"

# Pattern Mapping
PATTERN_TYPES_LIST = [
    PatternType.HEAD_AND_SHOULDERS,
    PatternType.INVERSE_HEAD_AND_SHOULDERS,
    PatternType.DOUBLE_TOP,
    PatternType.DOUBLE_BOTTOM,
    PatternType.TRIPLE_TOP,
    PatternType.TRIPLE_BOTTOM,
    PatternType.ASCENDING_TRIANGLE,
    PatternType.DESCENDING_TRIANGLE,
    PatternType.SYMMETRICAL_TRIANGLE,
    PatternType.BULL_FLAG,
    PatternType.BEAR_FLAG,
    PatternType.CUP_AND_HANDLE,
    PatternType.RISING_WEDGE,
    PatternType.FALLING_WEDGE,
    PatternType.CHANNEL_UP,
    PatternType.CHANNEL_DOWN,
]

# Regime Mapping
REGIME_TYPES_LIST = [
    MarketRegime.BULL_TREND,
    MarketRegime.BEAR_TREND,
    MarketRegime.SIDEWAYS,
    MarketRegime.HIGH_VOLATILITY,
]


class InferenceService:
    """
    Inference Service fuer CNN-LSTM Modell.

    Verwaltet Modell-Loading und Vorhersagen fuer alle Tasks.
    """

    def __init__(self):
        self._model = None
        self._model_version: Optional[str] = None
        self._model_metadata: Optional[dict] = None
        self._device = "cpu"

    def get_model_version(self) -> Optional[str]:
        """Gibt aktuelle Modell-Version zurueck."""
        return self._model_version

    def is_model_loaded(self) -> bool:
        """Prueft ob Modell geladen ist."""
        return self._model is not None

    # =========================================================================
    # Model Loading
    # =========================================================================

    async def load_model(self, model_id: Optional[str] = None) -> bool:
        """
        Laedt ein Modell mit robustem CUDA Error Handling.

        Args:
            model_id: Spezifische Modell-ID oder None fuer latest

        Returns:
            True wenn erfolgreich geladen
        """
        _ensure_torch()

        try:
            model_path = Path(MODEL_DIR)

            if model_id:
                # Lade spezifisches Modell
                target_path = model_path / f"{model_id}.pt"
            else:
                # Lade latest
                latest_path = model_path / "latest.pt"
                if latest_path.exists():
                    target_path = latest_path
                else:
                    # Finde neuestes Modell
                    pt_files = list(model_path.glob("*.pt"))
                    if not pt_files:
                        logger.warning("No models found")
                        return False
                    target_path = max(pt_files, key=lambda x: x.stat().st_mtime)

            if not target_path.exists():
                logger.warning(f"Model not found: {target_path}")
                return False

            # Bestimme Device mit CUDA-Check
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                try:
                    # Test CUDA accessibility before using it
                    torch.cuda.synchronize()
                    self._device = "cuda"
                    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                except RuntimeError as cuda_err:
                    logger.warning(f"CUDA available but not usable: {cuda_err}")
                    self._device = "cpu"
            else:
                self._device = "cpu"

            # Clear GPU cache before loading new model
            if self._device == "cuda":
                torch.cuda.empty_cache()

            # Lade Modell mit CUDA Error Handling
            try:
                from ..models.cnn_lstm_model import load_model
                self._model = load_model(str(target_path), device=self._device)
                self._model.eval()

                # Synchronize to catch any deferred CUDA errors
                if self._device == "cuda":
                    torch.cuda.synchronize()

            except RuntimeError as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "gpu" in error_str or "illegal memory" in error_str:
                    logger.warning(f"CUDA error during model load, falling back to CPU: {e}")
                    # Fallback to CPU
                    self._device = "cpu"
                    torch.cuda.empty_cache()
                    from ..models.cnn_lstm_model import load_model
                    self._model = load_model(str(target_path), device="cpu")
                    self._model.eval()
                else:
                    raise

            # Extrahiere Metadaten
            checkpoint = torch.load(str(target_path), map_location=self._device, weights_only=False)
            self._model_metadata = checkpoint.get("metadata", {})
            self._model_version = self._model_metadata.get(
                "job_id",
                target_path.stem
            )

            gpu_info = _get_gpu_memory_info()
            gpu_mem_str = f" (GPU mem: {gpu_info.get('allocated_mb', 0)}MB)" if gpu_info else ""
            logger.info(f"Model loaded: {self._model_version} on {self._device}{gpu_mem_str}")
            return True

        except RuntimeError as e:
            _handle_cuda_error("load_model", e)
            # Try CPU fallback on CUDA failure
            if self._device == "cuda":
                logger.warning("Attempting CPU fallback after CUDA error")
                self._device = "cpu"
                return await self.load_model(model_id)
            return False

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    async def list_models(self) -> list[ModelInfo]:
        """Listet alle verfuegbaren Modelle."""
        models = []
        model_path = Path(MODEL_DIR)

        if not model_path.exists():
            return models

        for pt_file in sorted(model_path.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                stat = pt_file.stat()
                metadata = {}

                # Versuche Metadaten zu laden
                try:
                    _ensure_torch()
                    checkpoint = torch.load(str(pt_file), map_location="cpu")
                    metadata = checkpoint.get("metadata", {})
                except Exception:
                    pass

                models.append(ModelInfo(
                    model_id=pt_file.stem,
                    version=metadata.get("job_id", pt_file.stem),
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    timeframes=metadata.get("timeframes", ["H1"]),
                    input_features=25,
                    total_parameters=self._model.get_num_parameters() if self._model else 0,
                    training_samples=metadata.get("training_samples"),
                    metrics={"loss": metadata.get("final_loss")},
                    is_active=pt_file.stem == self._model_version
                ))
            except Exception as e:
                logger.warning(f"Error reading model {pt_file}: {e}")

        return models

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Gibt Info fuer spezifisches Modell zurueck."""
        models = await self.list_models()
        for model in models:
            if model.model_id == model_id:
                return model
        return None

    # =========================================================================
    # Predictions
    # =========================================================================

    async def predict(
        self,
        symbol: str,
        timeframe: str = "H1"
    ) -> Optional[PredictionResponse]:
        """
        Fuehrt Multi-Task Vorhersage durch mit robustem CUDA Error Handling.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe

        Returns:
            PredictionResponse oder None bei Fehler
        """
        if not self.is_model_loaded():
            logger.error("No model loaded")
            return None

        _ensure_torch()
        start_time = time.time()

        try:
            # Hole Features und aktuellen Preis
            result = await feature_service.prepare_features_with_price(symbol, timeframe)
            if result is None:
                logger.error(f"Could not prepare features for {symbol}")
                return None

            features, current_price = result

            # Konvertiere zu Tensor mit CUDA Error Handling
            try:
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self._device)

                # Inference mit expliziter Synchronisation
                with torch.no_grad():
                    outputs = self._model(features_tensor)

                    # Synchronize GPU before accessing results
                    if self._device == "cuda":
                        torch.cuda.synchronize()

                # Parse Outputs - explizite Kopie auf CPU
                price_pred = outputs['price'].detach().cpu().numpy()[0]
                pattern_logits = outputs['patterns'].detach().cpu().numpy()[0]
                regime_logits = outputs['regime'].detach().cpu().numpy()[0]

            except RuntimeError as cuda_err:
                error_str = str(cuda_err).lower()
                if "cuda" in error_str or "illegal memory" in error_str or "out of memory" in error_str:
                    logger.error(f"CUDA error during inference for {symbol}: {cuda_err}")
                    # Attempt recovery
                    self._handle_cuda_recovery()
                    return None
                raise

            # Erstelle Price Prediction mit echtem aktuellem Preis
            price_prediction = self._create_price_prediction(
                current_price, price_pred, timeframe
            )

            # Pattern Predictions
            pattern_predictions = self._create_pattern_predictions(pattern_logits)

            # Regime Prediction
            regime_prediction = self._create_regime_prediction(regime_logits)

            inference_time = (time.time() - start_time) * 1000

            # Periodic GPU memory cleanup (every 100 inferences)
            if self._device == "cuda" and hasattr(self, '_inference_count'):
                self._inference_count += 1
                if self._inference_count % 100 == 0:
                    torch.cuda.empty_cache()
            elif self._device == "cuda":
                self._inference_count = 1

            return PredictionResponse(
                symbol=symbol,
                timeframe=timeframe.upper(),
                timestamp=datetime.now(timezone.utc),
                predictions=MultiTaskPrediction(
                    price=price_prediction,
                    patterns=pattern_predictions,
                    regime=regime_prediction
                ),
                model_version=self._model_version or "unknown",
                inference_time_ms=round(inference_time, 2)
            )

        except CUDAError as e:
            logger.error(f"CUDA error during prediction for {symbol}: {e}")
            self._handle_cuda_recovery()
            return None

        except Exception as e:
            _handle_cuda_error("predict", e)
            return None

    def _handle_cuda_recovery(self) -> None:
        """
        Attempts to recover from CUDA errors.

        Clears GPU cache and optionally reloads model on CPU.
        """
        _ensure_torch()
        logger.warning("Attempting CUDA recovery...")

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except Exception as e:
            logger.error(f"CUDA recovery failed: {e}")
            # Mark device as CPU for future inferences
            self._device = "cpu"
            logger.warning("Switching to CPU mode due to CUDA errors")

    def get_gpu_status(self) -> dict:
        """
        Returns current GPU status information.

        Useful for health checks and monitoring.
        """
        _ensure_torch()

        status = {
            "device": self._device,
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": self.is_model_loaded(),
        }

        if torch.cuda.is_available():
            try:
                status.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "memory": _get_gpu_memory_info(),
                    "cuda_healthy": True,
                })
                # Test CUDA accessibility
                torch.cuda.synchronize()
            except RuntimeError as e:
                status["cuda_healthy"] = False
                status["cuda_error"] = str(e)

        return status

    def _create_price_prediction(
        self,
        current_price: float,
        predictions: np.ndarray,
        timeframe: str
    ) -> PricePrediction:
        """Erstellt PricePrediction aus Modell-Output."""
        # predictions enthält [1h, 4h, 1d, 1w] prozentuale Aenderungen

        # Handle NaN values
        predictions = np.nan_to_num(predictions, nan=0.0)

        # Berechne Forecast-Preise
        forecast_1h = current_price * (1 + predictions[0]) if len(predictions) > 0 else None
        forecast_4h = current_price * (1 + predictions[1]) if len(predictions) > 1 else None
        forecast_1d = current_price * (1 + predictions[2]) if len(predictions) > 2 else None
        forecast_1w = current_price * (1 + predictions[3]) if len(predictions) > 3 else None

        # Bestimme Richtung basierend auf 1d Forecast
        avg_change = np.mean(predictions)
        if np.isnan(avg_change):
            avg_change = 0.0
        if avg_change > 0.005:
            direction = PriceDirection.BULLISH
        elif avg_change < -0.005:
            direction = PriceDirection.BEARISH
        else:
            direction = PriceDirection.NEUTRAL

        # Konfidenz basierend auf Konsistenz der Vorhersagen
        signs = np.sign(predictions)
        consistency = np.mean(signs == signs[0]) if len(signs) > 0 else 0.5
        if np.isnan(consistency):
            consistency = 0.5
        confidence = min(0.95, max(0.3, consistency * 0.8 + abs(avg_change) * 2))

        return PricePrediction(
            current=round(current_price, 2),
            forecast_1h=round(forecast_1h, 2) if forecast_1h else None,
            forecast_4h=round(forecast_4h, 2) if forecast_4h else None,
            forecast_1d=round(forecast_1d, 2) if forecast_1d else None,
            forecast_1w=round(forecast_1w, 2) if forecast_1w else None,
            direction=direction,
            confidence=round(confidence, 3),
            change_percent_1h=round(predictions[0] * 100, 2) if len(predictions) > 0 else None,
            change_percent_1d=round(predictions[2] * 100, 2) if len(predictions) > 2 else None
        )

    def _create_pattern_predictions(
        self,
        logits: np.ndarray,
        threshold: float = 0.5
    ) -> list[PatternPrediction]:
        """Erstellt PatternPredictions aus Modell-Output."""
        # Handle NaN values
        logits = np.nan_to_num(logits, nan=0.0)

        # Sigmoid auf Logits
        probs = 1 / (1 + np.exp(-logits))
        probs = np.nan_to_num(probs, nan=0.0)

        predictions = []
        for i, prob in enumerate(probs):
            if prob >= threshold and i < len(PATTERN_TYPES_LIST):
                pattern_type = PATTERN_TYPES_LIST[i]

                # Bestimme erwartete Richtung basierend auf Pattern
                bullish_patterns = [
                    PatternType.INVERSE_HEAD_AND_SHOULDERS,
                    PatternType.DOUBLE_BOTTOM,
                    PatternType.TRIPLE_BOTTOM,
                    PatternType.ASCENDING_TRIANGLE,
                    PatternType.BULL_FLAG,
                    PatternType.CUP_AND_HANDLE,
                    PatternType.FALLING_WEDGE,
                    PatternType.CHANNEL_UP,
                ]

                if pattern_type in bullish_patterns:
                    direction = PriceDirection.BULLISH
                else:
                    direction = PriceDirection.BEARISH

                predictions.append(PatternPrediction(
                    type=pattern_type,
                    confidence=round(float(prob), 3),
                    direction=direction
                ))

        # Sortiere nach Konfidenz
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        return predictions[:5]  # Top 5

    def _create_regime_prediction(self, logits: np.ndarray) -> RegimePrediction:
        """Erstellt RegimePrediction aus Modell-Output."""
        # Handle NaN values in logits
        if np.any(np.isnan(logits)):
            logits = np.nan_to_num(logits, nan=0.0)

        # Softmax auf Logits
        exp_logits = np.exp(logits - np.max(logits))
        sum_exp = np.sum(exp_logits)
        if sum_exp == 0 or np.isnan(sum_exp):
            probs = np.array([0.25, 0.25, 0.25, 0.25])  # Fallback: equal distribution
        else:
            probs = exp_logits / sum_exp

        # Handle any remaining NaN values
        probs = np.nan_to_num(probs, nan=0.25)

        # Bestimme Regime mit hoechster Wahrscheinlichkeit
        regime_idx = int(np.argmax(probs))
        current_regime = REGIME_TYPES_LIST[regime_idx]
        probability = float(probs[regime_idx])

        return RegimePrediction(
            current=current_regime,
            probability=round(probability, 3),
            transition_probs=RegimeTransitionProbs(
                bull_trend=round(float(probs[0]), 3),
                bear_trend=round(float(probs[1]), 3),
                sideways=round(float(probs[2]), 3),
                high_volatility=round(float(probs[3]), 3)
            )
        )

    # =========================================================================
    # Task-Specific Predictions
    # =========================================================================

    async def predict_price(self, symbol: str, timeframe: str = "H1") -> Optional[PricePrediction]:
        """Nur Preis-Vorhersage."""
        result = await self.predict(symbol, timeframe)
        if result:
            return result.predictions.price
        return None

    async def predict_patterns(self, symbol: str, timeframe: str = "H1") -> Optional[list[PatternPrediction]]:
        """Nur Pattern-Klassifikation."""
        result = await self.predict(symbol, timeframe)
        if result:
            return result.predictions.patterns
        return None

    async def predict_regime(self, symbol: str, timeframe: str = "H1") -> Optional[RegimePrediction]:
        """Nur Regime-Vorhersage."""
        result = await self.predict(symbol, timeframe)
        if result:
            return result.predictions.regime
        return None

    async def close(self):
        """Schliesst Service und gibt GPU-Ressourcen frei."""
        _ensure_torch()

        # Release model from GPU
        if self._model is not None:
            del self._model
            self._model = None

        # Clear GPU cache
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("GPU resources released")
            except Exception as e:
                logger.warning(f"Error releasing GPU resources: {e}")

        await feature_service.close()


# Singleton Instance
inference_service = InferenceService()
