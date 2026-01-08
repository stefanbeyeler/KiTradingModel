"""
TCN Pattern History Service - Persists and manages detected chart patterns.

Stores TCN pattern detections (Head & Shoulders, Double Top, Triangles, etc.)
with timestamps for historical analysis.
"""

import os
import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List
from loguru import logger


@dataclass
class PatternPoint:
    """A key point in a pattern (pivot, neckline intersection, etc.)."""
    index: int           # Candle index in the data
    price: float         # Price at this point
    point_type: str      # Type: "pivot_high", "pivot_low", "neckline", "support", "resistance"
    timestamp: str = ""  # ISO 8601 timestamp


@dataclass
class TCNPatternHistoryEntry:
    """A historical TCN pattern detection entry."""
    id: str                          # Unique ID (symbol_timeframe_type_timestamp)
    symbol: str                      # Trading symbol (e.g., BTCUSD)
    timeframe: str                   # Timeframe (1h, 4h, 1d)
    pattern_type: str                # Pattern type (e.g., double_top, head_and_shoulders)
    confidence: float                # Detection confidence (0-1)
    detected_at: str                 # ISO 8601 UTC - When the pattern was detected
    pattern_start_time: str          # ISO 8601 - When the pattern started forming
    pattern_end_time: str            # ISO 8601 - When the pattern completed
    direction: str                   # bullish, bearish, or neutral
    price_at_detection: float        # Price when pattern was detected
    price_target: Optional[float]    # Projected price target
    invalidation_level: Optional[float]  # Level where pattern is invalidated
    pattern_height: Optional[float]  # Height of the pattern
    category: str                    # reversal, continuation, or trend
    # Pattern geometry - key points for visualization
    pattern_points: Optional[List[dict]] = None  # List of PatternPoint as dicts
    # OHLCV data used for pattern detection (for retrospective analysis)
    ohlcv_data: Optional[List[dict]] = None  # List of OHLCV candles with timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TCNPatternHistoryEntry":
        """Create from dictionary."""
        return cls(**data)


class TCNPatternHistoryService:
    """
    Service for persisting and managing TCN pattern detections.

    Features:
    - JSON file-based persistence
    - Auto-scan with configurable interval
    - Duplicate detection
    - History cleanup (max entries, max age)
    """

    # Pattern categories
    REVERSAL_PATTERNS = [
        "head_and_shoulders", "inverse_head_and_shoulders",
        "double_top", "double_bottom",
        "triple_top", "triple_bottom",
        "cup_and_handle",
        "rising_wedge", "falling_wedge"
    ]

    CONTINUATION_PATTERNS = [
        "ascending_triangle", "descending_triangle", "symmetrical_triangle",
        "bull_flag", "bear_flag"
    ]

    TREND_PATTERNS = [
        "channel_up", "channel_down"
    ]

    def __init__(
        self,
        history_file: str = "data/tcn_pattern_history.json",
        max_entries: int = 2000,
        max_age_hours: int = 168,  # 7 days
        scan_interval_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize the TCN Pattern History Service.

        Args:
            history_file: Path to the JSON file for persistence
            max_entries: Maximum number of entries to keep
            max_age_hours: Maximum age of entries in hours
            scan_interval_seconds: Auto-scan interval in seconds
        """
        self._history_file = Path(history_file)
        self._max_entries = max_entries
        self._max_age_hours = max_age_hours
        self._scan_interval = scan_interval_seconds

        self._history: list[TCNPatternHistoryEntry] = []
        self._scan_task: Optional[asyncio.Task] = None
        self._scan_running = False
        self._last_scan: Optional[datetime] = None

        # Ensure data directory exists
        self._history_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history
        self._load_history()

        logger.info(f"TCN Pattern History Service initialized with {len(self._history)} entries")

    def _load_history(self) -> None:
        """Load history from JSON file."""
        if self._history_file.exists():
            try:
                with open(self._history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._history = [
                        TCNPatternHistoryEntry.from_dict(entry)
                        for entry in data
                    ]
                logger.info(f"Loaded {len(self._history)} entries from {self._history_file}")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                self._history = []
        else:
            self._history = []

    def _save_history(self) -> None:
        """Save history to JSON file."""
        try:
            with open(self._history_file, "w", encoding="utf-8") as f:
                json.dump(
                    [entry.to_dict() for entry in self._history],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def _get_category(self, pattern_type: str) -> str:
        """Get the category for a pattern type."""
        if pattern_type in self.REVERSAL_PATTERNS:
            return "reversal"
        elif pattern_type in self.CONTINUATION_PATTERNS:
            return "continuation"
        elif pattern_type in self.TREND_PATTERNS:
            return "trend"
        return "unknown"

    def _is_duplicate(self, entry: TCNPatternHistoryEntry) -> bool:
        """
        Check if an entry is a duplicate.

        A pattern is considered duplicate if:
        - Same symbol, timeframe, and pattern_type
        - Pattern end time is within 4 hours of an existing entry
        """
        for existing in self._history:
            if (existing.symbol == entry.symbol and
                existing.timeframe == entry.timeframe and
                existing.pattern_type == entry.pattern_type):

                # Check if pattern end times are close
                try:
                    existing_end = datetime.fromisoformat(existing.pattern_end_time.replace("Z", "+00:00"))
                    new_end = datetime.fromisoformat(entry.pattern_end_time.replace("Z", "+00:00"))

                    if abs((existing_end - new_end).total_seconds()) < 4 * 3600:  # 4 hours
                        return True
                except Exception:
                    pass

        return False

    def _cleanup_old_entries(self) -> int:
        """Remove old entries. Returns number of entries removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_age_hours)
        original_count = len(self._history)

        self._history = [
            entry for entry in self._history
            if datetime.fromisoformat(entry.detected_at.replace("Z", "+00:00")) > cutoff
        ]

        # Also enforce max entries
        if len(self._history) > self._max_entries:
            # Sort by detected_at (newest first) and keep only max_entries
            self._history.sort(
                key=lambda x: x.detected_at,
                reverse=True
            )
            self._history = self._history[:self._max_entries]

        removed = original_count - len(self._history)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old pattern entries")

        return removed

    def add_pattern(
        self,
        symbol: str,
        timeframe: str,
        pattern_type: str,
        confidence: float,
        pattern_start_time: str,
        pattern_end_time: str,
        direction: str,
        price_at_detection: float,
        price_target: Optional[float] = None,
        invalidation_level: Optional[float] = None,
        pattern_height: Optional[float] = None,
        pattern_points: Optional[List[dict]] = None,
        ohlcv_data: Optional[List[dict]] = None
    ) -> Optional[TCNPatternHistoryEntry]:
        """
        Add a detected pattern to history.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            pattern_type: Pattern type
            confidence: Detection confidence (0-1)
            pattern_start_time: When the pattern started forming (ISO 8601)
            pattern_end_time: When the pattern completed (ISO 8601)
            direction: bullish, bearish, or neutral
            price_at_detection: Price when pattern was detected
            price_target: Projected price target
            invalidation_level: Level where pattern is invalidated
            pattern_height: Height of the pattern
            pattern_points: Key points for visualization
            ohlcv_data: OHLCV candles used for detection (for retrospective analysis)

        Returns the entry if added, None if duplicate.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Generate unique ID
        entry_id = f"{symbol}_{timeframe}_{pattern_type}_{pattern_end_time}"

        # Get category
        category = self._get_category(pattern_type)

        entry = TCNPatternHistoryEntry(
            id=entry_id,
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            confidence=confidence,
            detected_at=now,
            pattern_start_time=pattern_start_time,
            pattern_end_time=pattern_end_time,
            direction=direction,
            price_at_detection=price_at_detection,
            price_target=price_target,
            invalidation_level=invalidation_level,
            pattern_height=pattern_height,
            category=category,
            pattern_points=pattern_points,
            ohlcv_data=ohlcv_data
        )

        # Check for duplicates
        if self._is_duplicate(entry):
            return None

        # Add entry
        self._history.append(entry)

        # Cleanup and save
        self._cleanup_old_entries()
        self._save_history()

        logger.debug(f"Added TCN pattern: {symbol} {pattern_type} ({confidence:.2%})")

        return entry

    def get_history(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern_type: Optional[str] = None,
        category: Optional[str] = None,
        direction: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> list[TCNPatternHistoryEntry]:
        """
        Get filtered pattern history.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            pattern_type: Filter by pattern type
            category: Filter by category (reversal, continuation, trend)
            direction: Filter by direction (bullish, bearish)
            min_confidence: Minimum confidence threshold
            limit: Maximum number of entries to return

        Returns:
            List of matching entries (newest first)
        """
        filtered = self._history.copy()

        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]

        if timeframe:
            filtered = [e for e in filtered if e.timeframe == timeframe]

        if pattern_type:
            filtered = [e for e in filtered if e.pattern_type == pattern_type]

        if category:
            filtered = [e for e in filtered if e.category == category]

        if direction:
            filtered = [e for e in filtered if e.direction == direction]

        if min_confidence > 0:
            filtered = [e for e in filtered if e.confidence >= min_confidence]

        # Sort by pattern_end_time (newest pattern end dates first)
        # This shows patterns with the most recent "Bis Datum" at the top
        filtered.sort(key=lambda x: x.pattern_end_time or x.detected_at, reverse=True)

        return filtered[:limit]

    def get_history_by_symbol(self, limit_per_symbol: int = 10) -> dict[str, list[dict]]:
        """Get history grouped by symbol."""
        result: dict[str, list[dict]] = {}

        for entry in self._history:
            if entry.symbol not in result:
                result[entry.symbol] = []

            if len(result[entry.symbol]) < limit_per_symbol:
                result[entry.symbol].append(entry.to_dict())

        # Sort each symbol's patterns by pattern_end_time (newest first)
        for symbol in result:
            result[symbol].sort(
                key=lambda x: x.get("pattern_end_time") or x.get("detected_at", ""),
                reverse=True
            )

        return result

    def get_statistics(self) -> dict:
        """Get pattern history statistics."""
        if not self._history:
            return {
                "total_patterns": 0,
                "symbols_with_patterns": 0,
                "by_pattern_type": {},
                "by_category": {},
                "by_direction": {},
                "by_timeframe": {},
                "last_scan": self._last_scan.isoformat() if self._last_scan else None,
                "scan_running": self._scan_running,
                "scan_interval_seconds": self._scan_interval
            }

        # Count by various dimensions
        by_pattern_type: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_direction: dict[str, int] = {}
        by_timeframe: dict[str, int] = {}
        symbols = set()

        for entry in self._history:
            symbols.add(entry.symbol)

            by_pattern_type[entry.pattern_type] = by_pattern_type.get(entry.pattern_type, 0) + 1
            by_category[entry.category] = by_category.get(entry.category, 0) + 1
            by_direction[entry.direction] = by_direction.get(entry.direction, 0) + 1
            by_timeframe[entry.timeframe] = by_timeframe.get(entry.timeframe, 0) + 1

        return {
            "total_patterns": len(self._history),
            "symbols_with_patterns": len(symbols),
            "by_pattern_type": dict(sorted(by_pattern_type.items(), key=lambda x: x[1], reverse=True)),
            "by_category": by_category,
            "by_direction": by_direction,
            "by_timeframe": by_timeframe,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "scan_running": self._scan_running,
            "scan_interval_seconds": self._scan_interval
        }

    def clear_history(self) -> int:
        """Clear all history. Returns number of entries removed."""
        count = len(self._history)
        self._history = []
        self._save_history()
        logger.info(f"Cleared {count} pattern history entries")
        return count

    async def scan_all_symbols(
        self,
        timeframes: Optional[list[str]] = None,
        threshold: float = 0.5
    ) -> dict:
        """
        Scan all symbols for patterns and add to history.

        Args:
            timeframes: Timeframes to scan (default: ["1h", "4h", "1d"])
            threshold: Minimum confidence threshold

        Returns:
            Scan result summary
        """
        from .data_service_client import data_service_client
        from .pattern_detection_service import pattern_detection_service

        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        self._last_scan = datetime.now(timezone.utc)

        try:
            # Get available symbols from Data Service (not directly from EasyInsight)
            symbols_data = await data_service_client.get_available_symbols()
            symbols = [s.get("symbol", s) if isinstance(s, dict) else s for s in symbols_data]

            if not symbols:
                logger.warning("No symbols available for scanning")
                return {"symbols_scanned": 0, "patterns_found": 0, "patterns_added": 0}

            patterns_found = 0
            patterns_added = 0

            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Detect patterns
                        response = await pattern_detection_service.detect_patterns(
                            symbol=symbol,
                            timeframe=timeframe,
                            lookback=200,
                            threshold=threshold
                        )

                        for pattern in response.patterns:
                            patterns_found += 1

                            # Get current price from market context
                            price = response.market_context.get("price", 0.0)

                            # Convert pattern_points to dicts if they are Pydantic objects
                            pattern_points_dicts = None
                            if pattern.pattern_points:
                                pattern_points_dicts = [
                                    p.model_dump() if hasattr(p, 'model_dump') else (p.dict() if hasattr(p, 'dict') else p)
                                    for p in pattern.pattern_points
                                ]

                            # Add to history (including OHLCV data for retrospective analysis)
                            entry = self.add_pattern(
                                symbol=symbol,
                                timeframe=timeframe,
                                pattern_type=pattern.pattern_type,
                                confidence=pattern.confidence,
                                pattern_start_time=pattern.start_time or "",
                                pattern_end_time=pattern.end_time or "",
                                direction=pattern.direction or "neutral",
                                price_at_detection=price,
                                price_target=pattern.price_target,
                                invalidation_level=pattern.invalidation_level,
                                pattern_height=pattern.pattern_height,
                                pattern_points=pattern_points_dicts,
                                ohlcv_data=response.ohlcv_data
                            )

                            if entry:
                                patterns_added += 1

                    except Exception as e:
                        logger.debug(f"Error scanning {symbol}/{timeframe}: {e}")

            logger.info(
                f"TCN Pattern scan complete: {len(symbols)} symbols, "
                f"{patterns_found} patterns found, {patterns_added} new patterns added"
            )

            return {
                "symbols_scanned": len(symbols),
                "timeframes": timeframes,
                "patterns_found": patterns_found,
                "patterns_added": patterns_added,
                "scan_time": self._last_scan.isoformat()
            }

        except Exception as e:
            logger.error(f"Error during pattern scan: {e}")
            return {"error": str(e)}

    async def scan_single_symbol(
        self,
        symbol: str,
        timeframes: Optional[list[str]] = None,
        threshold: float = 0.5
    ) -> dict:
        """
        Scan a single symbol for patterns and add to history.

        Args:
            symbol: Symbol to scan
            timeframes: Timeframes to scan (default: ["1h", "4h", "1d"])
            threshold: Minimum confidence threshold

        Returns:
            Scan result summary
        """
        from .pattern_detection_service import pattern_detection_service

        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        patterns_found = 0
        patterns_added = 0

        for timeframe in timeframes:
            try:
                # Detect patterns
                response = await pattern_detection_service.detect_patterns(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback=200,
                    threshold=threshold
                )

                for pattern in response.patterns:
                    patterns_found += 1

                    # Get current price from market context
                    price = response.market_context.get("price", 0.0)

                    # Convert pattern_points to dicts if they are Pydantic objects
                    pattern_points_dicts = None
                    if pattern.pattern_points:
                        pattern_points_dicts = [
                            p.model_dump() if hasattr(p, 'model_dump') else (p.dict() if hasattr(p, 'dict') else p)
                            for p in pattern.pattern_points
                        ]

                    # Add to history (including OHLCV data for retrospective analysis)
                    entry = self.add_pattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type=pattern.pattern_type,
                        confidence=pattern.confidence,
                        pattern_start_time=pattern.start_time or "",
                        pattern_end_time=pattern.end_time or "",
                        direction=pattern.direction or "neutral",
                        price_at_detection=price,
                        price_target=pattern.price_target,
                        invalidation_level=pattern.invalidation_level,
                        pattern_height=pattern.pattern_height,
                        pattern_points=pattern_points_dicts,
                        ohlcv_data=response.ohlcv_data
                    )

                    if entry:
                        patterns_added += 1

            except Exception as e:
                logger.debug(f"Error scanning {symbol}/{timeframe}: {e}")

        return {
            "symbol": symbol,
            "timeframes": timeframes,
            "patterns_found": patterns_found,
            "patterns_added": patterns_added
        }

    async def _auto_scan_loop(self) -> None:
        """Background loop for automatic scanning."""
        logger.info(f"Starting TCN Pattern auto-scan (interval: {self._scan_interval}s)")

        while self._scan_running:
            try:
                await self.scan_all_symbols()
            except Exception as e:
                logger.error(f"Error in auto-scan: {e}")

            # Wait for next scan
            await asyncio.sleep(self._scan_interval)

    async def start_auto_scan(self) -> bool:
        """Start automatic pattern scanning."""
        if self._scan_running:
            logger.warning("Auto-scan already running")
            return False

        self._scan_running = True
        self._scan_task = asyncio.create_task(self._auto_scan_loop())
        logger.info("TCN Pattern auto-scan started")
        return True

    async def stop_auto_scan(self) -> bool:
        """Stop automatic pattern scanning."""
        if not self._scan_running:
            logger.warning("Auto-scan not running")
            return False

        self._scan_running = False

        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
            self._scan_task = None

        logger.info("TCN Pattern auto-scan stopped")
        return True

    def is_scan_running(self) -> bool:
        """Check if auto-scan is running."""
        return self._scan_running


# Singleton instance
tcn_pattern_history_service = TCNPatternHistoryService()
