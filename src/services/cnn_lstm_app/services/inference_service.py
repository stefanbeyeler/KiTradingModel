"""
Inference Service für CNN-LSTM Multi-Task Model.

Laedt trainierte Modelle und fuehrt Vorhersagen durch.
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
        Laedt ein Modell.

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

            # Lade Modell
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            from ..models.cnn_lstm_model import load_model
            self._model = load_model(str(target_path), device=self._device)
            self._model.eval()

            # Extrahiere Metadaten
            checkpoint = torch.load(str(target_path), map_location=self._device)
            self._model_metadata = checkpoint.get("metadata", {})
            self._model_version = self._model_metadata.get(
                "job_id",
                target_path.stem
            )

            logger.info(f"Model loaded: {self._model_version} on {self._device}")
            return True

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
        Fuehrt Multi-Task Vorhersage durch.

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
            # Hole Features
            features = await feature_service.prepare_features(symbol, timeframe)
            if features is None:
                logger.error(f"Could not prepare features for {symbol}")
                return None

            # Konvertiere zu Tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self._device)

            # Inference
            with torch.no_grad():
                outputs = self._model(features_tensor)

            # Parse Outputs
            price_pred = outputs['price'].cpu().numpy()[0]
            pattern_logits = outputs['patterns'].cpu().numpy()[0]
            regime_logits = outputs['regime'].cpu().numpy()[0]

            # Hole aktuellen Preis aus Features (close ist Index 3)
            current_price = float(features[-1, 3])  # Letzte Close

            # Denormalisiere (vereinfacht - in Produktion waere Skalierung noetig)
            # Hier nehmen wir an dass price_pred prozentuale Aenderungen sind
            price_prediction = self._create_price_prediction(
                current_price, price_pred, timeframe
            )

            # Pattern Predictions
            pattern_predictions = self._create_pattern_predictions(pattern_logits)

            # Regime Prediction
            regime_prediction = self._create_regime_prediction(regime_logits)

            inference_time = (time.time() - start_time) * 1000

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

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None

    def _create_price_prediction(
        self,
        current_price: float,
        predictions: np.ndarray,
        timeframe: str
    ) -> PricePrediction:
        """Erstellt PricePrediction aus Modell-Output."""
        # predictions enthält [1h, 4h, 1d, 1w] prozentuale Aenderungen

        # Berechne Forecast-Preise
        forecast_1h = current_price * (1 + predictions[0]) if len(predictions) > 0 else None
        forecast_4h = current_price * (1 + predictions[1]) if len(predictions) > 1 else None
        forecast_1d = current_price * (1 + predictions[2]) if len(predictions) > 2 else None
        forecast_1w = current_price * (1 + predictions[3]) if len(predictions) > 3 else None

        # Bestimme Richtung basierend auf 1d Forecast
        avg_change = np.mean(predictions)
        if avg_change > 0.005:
            direction = PriceDirection.BULLISH
        elif avg_change < -0.005:
            direction = PriceDirection.BEARISH
        else:
            direction = PriceDirection.NEUTRAL

        # Konfidenz basierend auf Konsistenz der Vorhersagen
        signs = np.sign(predictions)
        consistency = np.mean(signs == signs[0]) if len(signs) > 0 else 0.5
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
        # Sigmoid auf Logits
        probs = 1 / (1 + np.exp(-logits))

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
        # Softmax auf Logits
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

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
        """Schliesst Service."""
        await feature_service.close()


# Singleton Instance
inference_service = InferenceService()
