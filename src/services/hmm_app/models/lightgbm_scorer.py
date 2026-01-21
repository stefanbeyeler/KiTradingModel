"""LightGBM Signal Scorer for trading signals.

This is a compatibility wrapper that imports from the shared module.
The actual implementation is in src/shared/lightgbm_scorer.py
"""

import sys
import os

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

# Import everything from shared module
from src.shared.lightgbm_scorer import (
    LightGBMSignalScorer,
    SignalScore,
    SignalType,
    MarketRegime,
    _load_lightgbm,
)

# Re-export for backward compatibility
__all__ = [
    'LightGBMSignalScorer',
    'SignalScore',
    'SignalType',
    'MarketRegime',
    '_load_lightgbm',
]
