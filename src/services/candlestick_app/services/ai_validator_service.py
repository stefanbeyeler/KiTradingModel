"""
AI Validator Service for Candlestick Pattern Recognition.

This service loads trained TCN models and validates/adjusts pattern confidence
scores detected by the rule-based system.

Architecture:
    1. Rule-based detection → Initial patterns with rule_confidence
    2. AI validation → Adjusts confidence based on trained model
    3. Final output → Combined confidence score
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Model directory (shared volume with training service)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))

# Pattern type to index mapping (must match training service)
PATTERN_TYPES = [
    "hammer", "inverted_hammer", "shooting_star", "hanging_man",
    "doji", "dragonfly_doji", "gravestone_doji",
    "bullish_engulfing", "bearish_engulfing",
    "morning_star", "evening_star",
    "piercing_line", "dark_cloud_cover",
    "three_white_soldiers", "three_black_crows",
    "rising_three_methods", "falling_three_methods",
    "spinning_top", "bullish_harami", "bearish_harami", "harami_cross",
    "no_pattern"  # For negative samples
]

PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_TYPES)}
IDX_TO_PATTERN = {i: p for i, p in enumerate(PATTERN_TYPES)}


@dataclass
class AIValidationResult:
    """Result of AI validation for a single pattern."""
    pattern_type: str
    rule_confidence: float
    ai_confidence: float
    final_confidence: float
    ai_prediction: str  # What the AI thinks the pattern is
    ai_agreement: bool  # Does AI agree with rule-based detection?
    validation_method: str  # "model" or "fallback"


class TCNPatternModel:
    """
    Temporal Convolutional Network for pattern classification.

    This is a lightweight inference-only model that loads pre-trained weights.
    Training happens in the candlestick-train service.
    """

    def __init__(self):
        self.model = None
        self.model_info: Optional[Dict] = None
        self.device = "cpu"  # CPU-only for inference (GPU used for training)
        self._torch_available = False

        try:
            import torch
            import torch.nn as nn
            self._torch_available = True
            self._torch = torch
            self._nn = nn
            logger.info("PyTorch available for AI validation")
        except ImportError:
            logger.warning("PyTorch not available - AI validation will use fallback mode")

    def _build_model(self, num_classes: int = len(PATTERN_TYPES)):
        """Build the TCN model architecture (must match training)."""
        if not self._torch_available:
            return None

        nn = self._nn
        torch = self._torch

        class TemporalBlock(nn.Module):
            """Single temporal block with dilated causal convolution."""

            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
                super().__init__()
                padding = (kernel_size - 1) * dilation

                self.conv1 = nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=padding, dilation=dilation
                )
                self.conv2 = nn.Conv1d(
                    out_channels, out_channels, kernel_size,
                    padding=padding, dilation=dilation
                )
                self.dropout = nn.Dropout(dropout)
                self.relu = nn.ReLU()

                # Residual connection
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                    if in_channels != out_channels else None

            def forward(self, x):
                out = self.conv1(x)
                out = out[:, :, :x.size(2)]  # Causal trim
                out = self.relu(out)
                out = self.dropout(out)

                out = self.conv2(out)
                out = out[:, :, :x.size(2)]  # Causal trim
                out = self.relu(out)
                out = self.dropout(out)

                res = x if self.downsample is None else self.downsample(x)
                return self.relu(out + res)

        class TCN(nn.Module):
            """Temporal Convolutional Network for pattern classification."""

            def __init__(self, input_channels=5, num_classes=num_classes,
                         num_channels=[32, 64, 64], kernel_size=3, dropout=0.2):
                super().__init__()

                layers = []
                for i, out_channels in enumerate(num_channels):
                    in_channels = input_channels if i == 0 else num_channels[i-1]
                    dilation = 2 ** i
                    layers.append(
                        TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
                    )

                self.tcn = nn.Sequential(*layers)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(num_channels[-1], num_classes)

            def forward(self, x):
                # x shape: (batch, channels, sequence_length)
                out = self.tcn(x)
                out = self.global_pool(out)
                out = out.squeeze(-1)
                out = self.fc(out)
                return out

        return TCN(num_classes=num_classes)

    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.

        If no path is provided, loads the latest model from MODEL_DIR.
        """
        if not self._torch_available:
            logger.warning("Cannot load model - PyTorch not available")
            return False

        torch = self._torch

        try:
            # Find model file
            if model_path is None:
                model_path = self._find_latest_model()

            if model_path is None:
                logger.info("No trained model found - AI validation will use fallback")
                return False

            model_path = Path(model_path)

            # Load model info
            info_path = Path(str(model_path) + ".json")
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                logger.info(f"Loaded model info: {self.model_info}")

            # Build and load model
            self.model = self._build_model()
            if self.model is None:
                return False

            # Load weights
            if model_path.exists() and model_path.suffix == '.pt':
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info(f"Loaded model from {model_path}")
                return True
            else:
                logger.warning(f"Model file not found or invalid: {model_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _find_latest_model(self) -> Optional[str]:
        """Find the most recently created model in MODEL_DIR."""
        if not MODEL_DIR.exists():
            return None

        model_files = list(MODEL_DIR.glob("candlestick_model_*.pt"))
        if not model_files:
            return None

        # Sort by modification time, newest first
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(model_files[0])

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict pattern type from OHLCV features.

        Args:
            features: numpy array of shape (sequence_length, 5) for OHLCV

        Returns:
            Tuple of (predicted_pattern_type, confidence)
        """
        if self.model is None or not self._torch_available:
            return ("unknown", 0.0)

        torch = self._torch

        try:
            with torch.no_grad():
                # Normalize features
                features = self._normalize_features(features)

                # Convert to tensor: (batch=1, channels=5, sequence)
                x = torch.tensor(features.T, dtype=torch.float32).unsqueeze(0)

                # Forward pass
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)

                # Get prediction
                confidence, pred_idx = probs.max(dim=1)
                pattern_type = IDX_TO_PATTERN.get(pred_idx.item(), "unknown")

                return (pattern_type, confidence.item())

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ("unknown", 0.0)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize OHLCV features for model input."""
        # Normalize each column independently
        normalized = np.zeros_like(features, dtype=np.float32)

        for i in range(features.shape[1]):
            col = features[:, i]
            col_min = col.min()
            col_max = col.max()
            if col_max > col_min:
                normalized[:, i] = (col - col_min) / (col_max - col_min)
            else:
                normalized[:, i] = 0.5

        return normalized

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded and ready for inference."""
        return self.model is not None


class AIValidatorService:
    """
    Service for AI-based pattern validation.

    Combines rule-based detection with AI model predictions to produce
    more accurate confidence scores.
    """

    def __init__(self):
        self.model = TCNPatternModel()
        self._model_loaded = False
        self._last_model_check = None
        self._model_check_interval = 300  # Check for new model every 5 minutes

        # Weighting for final confidence calculation
        self.rule_weight = 0.4  # Weight for rule-based confidence
        self.ai_weight = 0.6   # Weight for AI confidence (when available)

    def initialize(self) -> bool:
        """Initialize the AI validator by loading the model."""
        self._model_loaded = self.model.load()
        self._last_model_check = datetime.now(timezone.utc)

        if self._model_loaded:
            logger.info("AI Validator initialized with trained model")
        else:
            logger.info("AI Validator initialized in fallback mode (no model)")

        return self._model_loaded

    def _maybe_reload_model(self):
        """Check if model should be reloaded (new training completed)."""
        if self._last_model_check is None:
            self.initialize()
            return

        elapsed = (datetime.now(timezone.utc) - self._last_model_check).total_seconds()
        if elapsed > self._model_check_interval:
            # Check for newer model
            latest = self.model._find_latest_model()
            if latest and self.model.model_info:
                latest_info_path = Path(latest + ".json")
                if latest_info_path.exists():
                    with open(latest_info_path, 'r') as f:
                        latest_info = json.load(f)

                    current_created = self.model.model_info.get("created_at", "")
                    latest_created = latest_info.get("created_at", "")

                    if latest_created > current_created:
                        logger.info("New model detected, reloading...")
                        self._model_loaded = self.model.load(latest)

            self._last_model_check = datetime.now(timezone.utc)

    def validate_pattern(
        self,
        pattern_type: str,
        rule_confidence: float,
        ohlcv_data: List[Dict[str, float]],
        lookback: int = 20
    ) -> AIValidationResult:
        """
        Validate a detected pattern using the AI model.

        Args:
            pattern_type: The pattern type detected by rules
            rule_confidence: Confidence from rule-based detection
            ohlcv_data: List of OHLCV dicts (most recent last)
            lookback: Number of candles to use for AI prediction

        Returns:
            AIValidationResult with combined confidence
        """
        self._maybe_reload_model()

        # Fallback mode if model not available
        if not self._model_loaded:
            return AIValidationResult(
                pattern_type=pattern_type,
                rule_confidence=rule_confidence,
                ai_confidence=0.0,
                final_confidence=rule_confidence,
                ai_prediction="unknown",
                ai_agreement=True,  # Assume agreement in fallback
                validation_method="fallback"
            )

        try:
            # Prepare features from OHLCV data
            features = self._prepare_features(ohlcv_data, lookback)

            # Get AI prediction
            ai_prediction, ai_confidence = self.model.predict(features)

            # Check if AI agrees with rule-based detection
            ai_agreement = (ai_prediction == pattern_type)

            # Calculate final confidence
            if ai_agreement:
                # AI agrees: boost confidence
                final_confidence = (
                    self.rule_weight * rule_confidence +
                    self.ai_weight * ai_confidence
                )
                # Bonus for agreement
                final_confidence = min(1.0, final_confidence * 1.1)
            else:
                # AI disagrees: reduce confidence
                if ai_confidence > 0.7:
                    # Strong AI disagreement: significantly reduce
                    final_confidence = rule_confidence * 0.5
                else:
                    # Weak AI disagreement: slight reduction
                    final_confidence = rule_confidence * 0.8

            return AIValidationResult(
                pattern_type=pattern_type,
                rule_confidence=rule_confidence,
                ai_confidence=ai_confidence,
                final_confidence=round(final_confidence, 3),
                ai_prediction=ai_prediction,
                ai_agreement=ai_agreement,
                validation_method="model"
            )

        except Exception as e:
            logger.error(f"AI validation failed: {e}")
            return AIValidationResult(
                pattern_type=pattern_type,
                rule_confidence=rule_confidence,
                ai_confidence=0.0,
                final_confidence=rule_confidence,
                ai_prediction="error",
                ai_agreement=True,
                validation_method="fallback"
            )

    def validate_patterns_batch(
        self,
        patterns: List[Dict[str, Any]],
        ohlcv_data: List[Dict[str, float]],
        lookback: int = 20
    ) -> List[AIValidationResult]:
        """Validate multiple patterns in batch."""
        results = []

        for pattern in patterns:
            result = self.validate_pattern(
                pattern_type=pattern.get("pattern_type", "unknown"),
                rule_confidence=pattern.get("confidence", 0.5),
                ohlcv_data=ohlcv_data,
                lookback=lookback
            )
            results.append(result)

        return results

    def _prepare_features(
        self,
        ohlcv_data: List[Dict[str, float]],
        lookback: int
    ) -> np.ndarray:
        """Convert OHLCV data to numpy features for model."""
        # Take last 'lookback' candles
        data = ohlcv_data[-lookback:] if len(ohlcv_data) >= lookback else ohlcv_data

        # Convert to numpy array
        features = np.array([
            [
                d.get("open", 0),
                d.get("high", 0),
                d.get("low", 0),
                d.get("close", 0),
                d.get("volume", 0)
            ]
            for d in data
        ], dtype=np.float32)

        # Pad if necessary
        if len(features) < lookback:
            padding = np.zeros((lookback - len(features), 5), dtype=np.float32)
            features = np.vstack([padding, features])

        return features

    def get_status(self) -> Dict[str, Any]:
        """Get AI validator status."""
        return {
            "model_loaded": self._model_loaded,
            "model_info": self.model.model_info if self._model_loaded else None,
            "validation_method": "model" if self._model_loaded else "fallback",
            "last_model_check": self._last_model_check.isoformat() if self._last_model_check else None,
            "weights": {
                "rule_weight": self.rule_weight,
                "ai_weight": self.ai_weight
            }
        }


# Global singleton
ai_validator_service = AIValidatorService()
