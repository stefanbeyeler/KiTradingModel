"""CRT (Candle Range Theory) Detection Module.

This module implements ICT/SMC-style Candle Range Theory detection
with integration into the existing TCN-Pattern service.

Components:
- SessionManager: Handles EST session times and H4 candle boundaries
- RangeTracker: Manages CRT range state machine
- PurgeDetector: Detects liquidity sweeps and re-entries
- CRTSignalService: Generates trading signals with service integration
"""

from .session_manager import SessionManager, session_manager
from .range_tracker import RangeTracker, CRTRange, CRTState, range_tracker
from .purge_detector import PurgeDetector, PurgeEvent, ReEntryEvent, purge_detector
from .crt_signal_service import CRTSignalService, CRTSignal, crt_signal_service

__all__ = [
    # Session Manager
    "SessionManager",
    "session_manager",
    # Range Tracker
    "RangeTracker",
    "CRTRange",
    "CRTState",
    "range_tracker",
    # Purge Detector
    "PurgeDetector",
    "PurgeEvent",
    "ReEntryEvent",
    "purge_detector",
    # Signal Service
    "CRTSignalService",
    "CRTSignal",
    "crt_signal_service",
]
