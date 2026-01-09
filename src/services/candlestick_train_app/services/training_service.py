"""Candlestick Pattern Training Service.

Handles model training for candlestick pattern recognition using PyTorch TCN.
"""

import os
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from loguru import logger

# Try to import PyTorch (may not be available in all environments)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - training will be disabled")


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job information."""
    job_id: str
    status: TrainingStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    symbols: List[str] = None
    timeframes: List[str] = None
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    progress: float = 0.0
    current_epoch: int = 0
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    error_message: Optional[str] = None
    model_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# Data Service URL
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")
CANDLESTICK_SERVICE_URL = os.getenv("CANDLESTICK_SERVICE_URL", "http://trading-candlestick:3006")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

# Shared data directory (mounted from candlestick service)
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
FEEDBACK_FILE = DATA_DIR / "pattern_feedback.json"


class TrainingService:
    """
    Service for training candlestick pattern recognition models.

    Uses PyTorch TCN (Temporal Convolutional Network) for pattern classification.
    """

    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._current_job: Optional[TrainingJob] = None
        self._training_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._model_path = Path(MODEL_DIR)
        self._model_path.mkdir(parents=True, exist_ok=True)

        # Load training history
        self._history_file = self._model_path / "training_history.json"
        self._load_history()

        logger.info(f"TrainingService initialized (PyTorch available: {TORCH_AVAILABLE})")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _load_history(self):
        """Load training history from file."""
        try:
            if self._history_file.exists():
                with open(self._history_file, 'r') as f:
                    data = json.load(f)
                    for job_data in data:
                        job = TrainingJob(**job_data)
                        self._jobs[job.job_id] = job
                logger.info(f"Loaded {len(self._jobs)} training jobs from history")
        except Exception as e:
            logger.error(f"Failed to load training history: {e}")

    def _save_history(self):
        """Save training history to file."""
        try:
            with open(self._history_file, 'w') as f:
                json.dump([job.to_dict() for job in self._jobs.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        return f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Pattern type mapping (must match inference service)
    PATTERN_TYPES = [
        # Single candle patterns
        "hammer", "inverted_hammer", "shooting_star", "hanging_man",
        "doji", "dragonfly_doji", "gravestone_doji",
        "spinning_top",
        "bullish_belt_hold", "bearish_belt_hold",
        # Two candle patterns
        "bullish_engulfing", "bearish_engulfing",
        "piercing_line", "dark_cloud_cover",
        "bullish_harami", "bearish_harami", "harami_cross",
        "bullish_counterattack", "bearish_counterattack",
        # Three candle patterns
        "morning_star", "evening_star",
        "three_white_soldiers", "three_black_crows",
        "rising_three_methods", "falling_three_methods",
        "three_inside_up", "three_inside_down",
        "bullish_abandoned_baby", "bearish_abandoned_baby",
        "tower_bottom", "tower_top",
        "advance_block",
        # Island reversal patterns (multi-candle)
        "bearish_island", "bullish_island",
        # No pattern
        "no_pattern"
    ]

    def _build_tcn_model(self, num_classes: int = None):
        """Build the TCN model for pattern classification."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        if num_classes is None:
            num_classes = len(self.PATTERN_TYPES)

        class TemporalBlock(nn.Module):
            """Temporal block with dilated causal convolution."""

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

                self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                    if in_channels != out_channels else None

            def forward(self, x):
                out = self.conv1(x)
                out = out[:, :, :x.size(2)]
                out = self.relu(out)
                out = self.dropout(out)

                out = self.conv2(out)
                out = out[:, :, :x.size(2)]
                out = self.relu(out)
                out = self.dropout(out)

                res = x if self.downsample is None else self.downsample(x)
                return self.relu(out + res)

        class TCN(nn.Module):
            """Temporal Convolutional Network."""

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
                out = self.tcn(x)
                out = self.global_pool(out)
                out = out.squeeze(-1)
                out = self.fc(out)
                return out

        return TCN(num_classes=num_classes)

    def _prepare_training_dataset(
        self,
        training_data: Dict[str, Any],
        patterns: List[Dict]
    ) -> List[Dict]:
        """
        Prepare training dataset from OHLCV data and pattern labels.

        Args:
            training_data: Dict of symbol -> timeframe -> candle data
            patterns: List of pattern history entries with labels

        Returns:
            List of training samples with features and labels
        """
        if not TORCH_AVAILABLE:
            return []

        import numpy as np

        dataset = []
        pattern_to_idx = {p: i for i, p in enumerate(self.PATTERN_TYPES)}
        sequence_length = 20  # Lookback for each sample

        for pattern in patterns:
            symbol = pattern.get("symbol", "")
            timeframe = pattern.get("timeframe", "H1")
            pattern_type = pattern.get("pattern_type", "no_pattern")
            weight = pattern.get("training_weight", 1.0)

            # Get OHLCV data for this symbol/timeframe
            symbol_data = training_data.get(symbol, {})
            candles = symbol_data.get(timeframe, [])

            if len(candles) < sequence_length:
                continue

            # Find pattern timestamp in candle data
            pattern_ts = pattern.get("timestamp", "")
            candle_idx = self._find_candle_index(candles, pattern_ts)

            if candle_idx < sequence_length:
                continue

            # Extract features for the sequence ending at pattern
            features = self._extract_features(candles[candle_idx - sequence_length:candle_idx])

            if features is None:
                continue

            # Get label
            label = pattern_to_idx.get(pattern_type, pattern_to_idx["no_pattern"])

            dataset.append({
                "features": torch.tensor(features, dtype=torch.float32),
                "label": label,
                "weight": weight,
                "symbol": symbol,
                "pattern_type": pattern_type
            })

        logger.info(f"Prepared {len(dataset)} training samples from pattern history")
        return dataset

    def _generate_synthetic_training_data(
        self,
        training_data: Dict[str, Any],
        samples_per_symbol: int = 50
    ) -> List[Dict]:
        """
        Generate synthetic training data from OHLCV when pattern history is insufficient.

        Uses sliding window approach to create training samples.
        """
        if not TORCH_AVAILABLE:
            return []

        import numpy as np

        dataset = []
        sequence_length = 20
        no_pattern_idx = self.PATTERN_TYPES.index("no_pattern")

        for symbol, timeframe_data in training_data.items():
            for timeframe, candles in timeframe_data.items():
                if len(candles) < sequence_length + samples_per_symbol:
                    continue

                # Create sliding window samples
                for i in range(samples_per_symbol):
                    start_idx = sequence_length + i
                    if start_idx >= len(candles):
                        break

                    features = self._extract_features(
                        candles[start_idx - sequence_length:start_idx]
                    )

                    if features is None:
                        continue

                    # For synthetic data, use "no_pattern" label
                    # The model learns base patterns, and confirmed patterns
                    # will be weighted higher during actual training
                    dataset.append({
                        "features": torch.tensor(features, dtype=torch.float32),
                        "label": no_pattern_idx,
                        "weight": 0.5,  # Lower weight for synthetic data
                        "symbol": symbol,
                        "pattern_type": "no_pattern"
                    })

        logger.info(f"Generated {len(dataset)} synthetic training samples")
        return dataset

    def _extract_features(self, candles: List[Dict]) -> Optional[Any]:
        """Extract normalized OHLCV features from candles."""
        import numpy as np

        try:
            features = np.array([
                [
                    float(c.get("open", c.get("o", 0))),
                    float(c.get("high", c.get("h", 0))),
                    float(c.get("low", c.get("l", 0))),
                    float(c.get("close", c.get("c", 0))),
                    float(c.get("volume", c.get("v", 0)))
                ]
                for c in candles
            ], dtype=np.float32)

            # Normalize per-column
            for i in range(features.shape[1]):
                col = features[:, i]
                col_min, col_max = col.min(), col.max()
                if col_max > col_min:
                    features[:, i] = (col - col_min) / (col_max - col_min)
                else:
                    features[:, i] = 0.5

            # Transpose for Conv1d: (channels, sequence_length)
            return features.T

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def _find_candle_index(self, candles: List[Dict], timestamp: str) -> int:
        """Find index of candle matching timestamp."""
        from dateutil import parser as date_parser

        try:
            target_ts = date_parser.isoparse(timestamp)

            for i, candle in enumerate(candles):
                candle_ts_str = candle.get("datetime") or candle.get("timestamp") or candle.get("snapshot_time")
                if candle_ts_str:
                    candle_ts = date_parser.isoparse(candle_ts_str)
                    if abs((candle_ts - target_ts).total_seconds()) < 3600:  # Within 1 hour
                        return i
        except Exception:
            pass

        return -1

    async def _fetch_training_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        lookback: int = 500
    ) -> Dict[str, Any]:
        """Fetch training data from Data Service.

        Uses TwelveData API via Data Service for OHLCV data.
        Falls back to EasyInsight if TwelveData fails.
        """
        client = await self._get_client()
        training_data = {}

        # Map standard timeframes to TwelveData intervals
        tf_mapping = {
            "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1h", "H4": "4h",
            "D1": "1day", "W1": "1week", "MN": "1month",
        }

        for symbol in symbols:
            symbol_data = {}
            for tf in timeframes:
                try:
                    # Try TwelveData first (primary source)
                    interval = tf_mapping.get(tf, tf.lower())
                    url = f"{DATA_SERVICE_URL}/api/v1/twelvedata/time_series/{symbol}"
                    params = {
                        "interval": interval,
                        "outputsize": lookback,
                    }
                    response = await client.get(url, params=params)

                    if response.status_code == 200:
                        result = response.json()
                        # TwelveData returns data in "values" array
                        values = result.get("values", [])
                        if values:
                            symbol_data[tf] = values
                            logger.debug(f"Fetched {len(values)} candles for {symbol} {tf} from TwelveData")
                            continue

                    # Fallback to EasyInsight for H1 (only timeframe supported)
                    if tf == "H1":
                        url = f"{DATA_SERVICE_URL}/api/v1/easyinsight/ohlcv/{symbol}"
                        params = {"limit": lookback}
                        response = await client.get(url, params=params)
                        if response.status_code == 200:
                            result = response.json()
                            data = result.get("data", [])
                            if data:
                                symbol_data[tf] = data
                                logger.debug(f"Fetched {len(data)} candles for {symbol} {tf} from EasyInsight")
                                continue

                    logger.warning(f"No data available for {symbol} {tf}")

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol} {tf}: {e}")
                    continue

            if symbol_data:
                training_data[symbol] = symbol_data

        return training_data

    def _load_user_feedback(self) -> Dict[str, Any]:
        """Load user feedback for pattern corrections.

        Returns a dictionary with:
        - corrections: Map of pattern_id -> corrected pattern info
        - rejected_ids: Set of pattern IDs marked as false positives
        - confirmed_ids: Set of pattern IDs confirmed as correct
        - statistics: Summary of feedback data
        - reason_analysis: Analysis of feedback reasons for training insights
        """
        feedback_data = {
            "corrections": {},
            "rejected_ids": set(),
            "confirmed_ids": set(),
            "statistics": {
                "total": 0,
                "confirmed": 0,
                "corrected": 0,
                "rejected": 0,
            },
            # New: Reason-based analysis for smarter training
            "reason_analysis": {
                "by_pattern_and_reason": {},  # pattern:reason -> count
                "problematic_patterns": [],    # Patterns with many corrections
                "training_recommendations": [] # Specific training adjustments
            }
        }

        if not FEEDBACK_FILE.exists():
            logger.info("No user feedback file found - training without corrections")
            return feedback_data

        try:
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                raw_feedback = json.load(f)

            # Count reasons by pattern
            pattern_reason_counts = {}
            pattern_correction_counts = {}

            for entry in raw_feedback:
                pattern_id = entry.get("id", "")
                feedback_type = entry.get("feedback_type", "")
                original_pattern = entry.get("original_pattern", "unknown")
                reason_category = entry.get("reason_category")
                reason_text = entry.get("reason_text")

                if feedback_type == "confirmed":
                    feedback_data["confirmed_ids"].add(pattern_id)
                    feedback_data["statistics"]["confirmed"] += 1

                elif feedback_type == "corrected":
                    feedback_data["corrections"][pattern_id] = {
                        "original": original_pattern,
                        "corrected": entry.get("corrected_pattern"),
                        "symbol": entry.get("symbol"),
                        "timeframe": entry.get("timeframe"),
                        "ohlc_data": entry.get("ohlc_data"),
                        "reason_category": reason_category,
                        "reason_text": reason_text,
                    }
                    feedback_data["statistics"]["corrected"] += 1

                    # Track correction counts by pattern
                    pattern_correction_counts[original_pattern] = \
                        pattern_correction_counts.get(original_pattern, 0) + 1

                elif feedback_type == "rejected":
                    feedback_data["rejected_ids"].add(pattern_id)
                    feedback_data["statistics"]["rejected"] += 1

                    # Track rejection counts by pattern
                    pattern_correction_counts[original_pattern] = \
                        pattern_correction_counts.get(original_pattern, 0) + 1

                # Track reason categories
                if reason_category:
                    key = f"{original_pattern}:{reason_category}"
                    pattern_reason_counts[key] = pattern_reason_counts.get(key, 0) + 1

                feedback_data["statistics"]["total"] += 1

            # Store reason analysis
            feedback_data["reason_analysis"]["by_pattern_and_reason"] = pattern_reason_counts

            # Identify problematic patterns (high correction rate)
            problematic = [
                {"pattern": p, "corrections": c}
                for p, c in sorted(pattern_correction_counts.items(), key=lambda x: -x[1])
                if c >= 3
            ]
            feedback_data["reason_analysis"]["problematic_patterns"] = problematic

            # Generate training recommendations based on feedback reasons
            recommendations = self._generate_training_recommendations(pattern_reason_counts)
            feedback_data["reason_analysis"]["training_recommendations"] = recommendations

            logger.info(
                f"Loaded user feedback: {feedback_data['statistics']['total']} total, "
                f"{feedback_data['statistics']['confirmed']} confirmed, "
                f"{feedback_data['statistics']['corrected']} corrected, "
                f"{feedback_data['statistics']['rejected']} rejected, "
                f"{len(pattern_reason_counts)} reason entries"
            )

            if recommendations:
                logger.info(f"Generated {len(recommendations)} training recommendations from feedback")

        except Exception as e:
            logger.error(f"Failed to load user feedback: {e}")

        return feedback_data

    def _generate_training_recommendations(
        self,
        pattern_reason_counts: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Generate training recommendations based on feedback reason patterns.

        Analyzes common feedback reasons to suggest:
        - Sample weighting adjustments
        - Data augmentation strategies
        - Feature importance hints
        """
        recommendations = []

        # Mapping from reason categories to training adjustments
        REASON_ADJUSTMENTS = {
            "body_too_large": {
                "action": "augment_body_variations",
                "description": "Füge Samples mit unterschiedlichen Körpergrössen hinzu",
                "weight_adjustment": 1.5,
            },
            "body_too_small": {
                "action": "augment_body_variations",
                "description": "Trainiere mehr auf minimale Körpergrössen",
                "weight_adjustment": 1.5,
            },
            "upper_shadow_too_short": {
                "action": "augment_shadow_variations",
                "description": "Erweitere Shadow-Längen-Variationen",
                "weight_adjustment": 1.3,
            },
            "lower_shadow_too_short": {
                "action": "augment_shadow_variations",
                "description": "Erweitere Shadow-Längen-Variationen",
                "weight_adjustment": 1.3,
            },
            "wrong_trend_context": {
                "action": "include_trend_context",
                "description": "Erhöhe Gewichtung von Trend-Kontext-Features",
                "weight_adjustment": 2.0,
            },
            "not_fully_engulfing": {
                "action": "strict_engulfing_samples",
                "description": "Trainiere nur auf klare Engulfing-Beispiele",
                "weight_adjustment": 1.5,
            },
            "false_positive": {
                "action": "add_negative_samples",
                "description": "Füge mehr Negativ-Beispiele für dieses Pattern hinzu",
                "weight_adjustment": 0.5,  # Reduce weight of positive samples
            },
        }

        for key, count in sorted(pattern_reason_counts.items(), key=lambda x: -x[1]):
            if count < 3:  # Minimum threshold
                continue

            parts = key.split(":", 1)
            if len(parts) != 2:
                continue

            pattern, reason = parts

            if reason in REASON_ADJUSTMENTS:
                adjustment = REASON_ADJUSTMENTS[reason]
                recommendations.append({
                    "pattern": pattern,
                    "reason": reason,
                    "count": count,
                    "action": adjustment["action"],
                    "description": adjustment["description"],
                    "weight_adjustment": adjustment["weight_adjustment"],
                    "priority": "high" if count >= 5 else "medium"
                })

        return recommendations[:10]  # Top 10 recommendations

    def _apply_feedback_to_labels(
        self,
        pattern_history: List[Dict],
        feedback: Dict[str, Any]
    ) -> List[Dict]:
        """Apply user feedback corrections to pattern labels.

        - Rejected patterns are excluded from training
        - Corrected patterns have their labels updated
        - Confirmed patterns are kept with boosted weight
        """
        corrected_patterns = []
        excluded_count = 0
        corrected_count = 0

        for pattern in pattern_history:
            pattern_id = pattern.get("id", "")

            # Skip rejected patterns (false positives)
            if pattern_id in feedback["rejected_ids"]:
                excluded_count += 1
                continue

            # Apply corrections
            if pattern_id in feedback["corrections"]:
                correction = feedback["corrections"][pattern_id]
                original_type = pattern.get("pattern_type")
                corrected_type = correction.get("corrected")

                if corrected_type and corrected_type != "no_pattern":
                    # Update pattern type to corrected value
                    pattern = pattern.copy()
                    pattern["pattern_type"] = corrected_type
                    pattern["original_pattern_type"] = original_type
                    pattern["is_corrected"] = True
                    pattern["training_weight"] = 2.0  # Higher weight for corrected samples
                    corrected_count += 1
                    corrected_patterns.append(pattern)
                # If corrected to "no_pattern", skip it
                continue

            # Boost weight for confirmed patterns
            if pattern_id in feedback["confirmed_ids"]:
                pattern = pattern.copy()
                pattern["is_confirmed"] = True
                pattern["training_weight"] = 1.5  # Slightly higher weight

            corrected_patterns.append(pattern)

        logger.info(
            f"Applied feedback: {excluded_count} excluded, "
            f"{corrected_count} corrected, "
            f"{len(corrected_patterns)} patterns for training"
        )

        return corrected_patterns

    async def _fetch_pattern_history(self, limit: int = 1000) -> List[Dict]:
        """Fetch pattern history from Candlestick Service for training labels."""
        try:
            client = await self._get_client()
            url = f"{CANDLESTICK_SERVICE_URL}/api/v1/history"
            params = {"limit": limit}

            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                patterns = data.get("patterns", [])
                logger.info(f"Fetched {len(patterns)} patterns from history")
                return patterns
            else:
                logger.warning(f"Failed to fetch pattern history: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching pattern history: {e}")
            return []

    async def _run_training(self, job: TrainingJob):
        """Execute the training job."""
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(f"Starting training job {job.job_id}")

            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available")

            # Fetch training data
            logger.info(f"Fetching training data for {len(job.symbols)} symbols...")
            training_data = await self._fetch_training_data(
                symbols=job.symbols,
                timeframes=job.timeframes,
                lookback=500
            )

            if not training_data:
                raise RuntimeError("No training data available")

            logger.info(f"Fetched data for {len(training_data)} symbols")

            # Load user feedback for pattern corrections
            logger.info("Loading user feedback for training corrections...")
            user_feedback = self._load_user_feedback()

            # Fetch pattern history from Candlestick Service
            pattern_history = await self._fetch_pattern_history()

            # Apply user feedback corrections to pattern labels
            if pattern_history:
                corrected_patterns = self._apply_feedback_to_labels(
                    pattern_history, user_feedback
                )
                logger.info(f"Training with {len(corrected_patterns)} patterns "
                           f"(feedback applied: {user_feedback['statistics']['total']} entries)")
            else:
                corrected_patterns = []
                logger.warning("No pattern history available for supervised training")

            # Build and train TCN model
            # The corrected_patterns list contains:
            # - pattern_type: The (possibly corrected) pattern label
            # - training_weight: Higher weight for confirmed/corrected patterns
            # - is_corrected: True if user corrected the label
            # - is_confirmed: True if user confirmed the pattern

            # Prepare training data
            train_dataset = self._prepare_training_dataset(
                training_data, corrected_patterns
            )

            if len(train_dataset) < 10:
                logger.warning("Insufficient training data, using fallback mode")
                # Fallback: simple training with synthetic data
                train_dataset = self._generate_synthetic_training_data(training_data)

            logger.info(f"Training dataset size: {len(train_dataset)} samples")

            # Build TCN model
            model = self._build_tcn_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=job.learning_rate)
            criterion = nn.CrossEntropyLoss(reduction='none')  # For weighted loss

            # Training loop
            model.train()
            best_model_state = None

            for epoch in range(job.epochs):
                if job.status == TrainingStatus.CANCELLED:
                    logger.info(f"Training job {job.job_id} cancelled")
                    return

                epoch_loss = 0.0
                num_batches = 0

                # Mini-batch training
                for batch_start in range(0, len(train_dataset), job.batch_size):
                    batch = train_dataset[batch_start:batch_start + job.batch_size]

                    if len(batch) == 0:
                        continue

                    # Prepare batch tensors
                    features = torch.stack([item['features'] for item in batch])
                    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
                    weights = torch.tensor([item.get('weight', 1.0) for item in batch], dtype=torch.float32)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    # Apply sample weights
                    weighted_loss = (loss * weights).mean()
                    weighted_loss.backward()
                    optimizer.step()

                    epoch_loss += weighted_loss.item()
                    num_batches += 1

                # Calculate average loss
                avg_loss = epoch_loss / max(num_batches, 1)

                # Update job progress
                job.current_epoch = epoch + 1
                job.progress = (epoch + 1) / job.epochs * 100
                job.current_loss = avg_loss

                if job.best_loss is None or avg_loss < job.best_loss:
                    job.best_loss = avg_loss
                    best_model_state = model.state_dict().copy()

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Job {job.job_id}: Epoch {epoch + 1}/{job.epochs}, "
                        f"Loss: {avg_loss:.4f}, Best: {job.best_loss:.4f}"
                    )

                # Allow other tasks to run
                await asyncio.sleep(0.01)

            # Save the best model
            model_filename = f"candlestick_model_{job.job_id}.pt"
            job.model_path = str(self._model_path / model_filename)

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            torch.save(model.state_dict(), job.model_path)
            logger.info(f"Model saved to {job.model_path}")

            # Save model metadata
            model_info = {
                "job_id": job.job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "symbols": job.symbols,
                "timeframes": job.timeframes,
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "learning_rate": job.learning_rate,
                "final_loss": job.current_loss,
                "best_loss": job.best_loss,
                "training_samples": len(train_dataset),
                "training_patterns": len(corrected_patterns),
                "feedback_applied": {
                    "total": user_feedback["statistics"]["total"],
                    "confirmed": user_feedback["statistics"]["confirmed"],
                    "corrected": user_feedback["statistics"]["corrected"],
                    "rejected": user_feedback["statistics"]["rejected"],
                },
            }
            with open(job.model_path + ".json", 'w') as f:
                json.dump(model_info, f, indent=2)

            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_history()

            logger.info(f"Training job {job.job_id} completed successfully")

            # Notify Candlestick Service to reload model (optional)
            try:
                client = await self._get_client()
                await client.post(
                    f"{CANDLESTICK_SERVICE_URL}/api/v1/model/reload",
                    json={"model_path": job.model_path}
                )
            except Exception as e:
                logger.warning(f"Failed to notify Candlestick Service: {e}")

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_history()
            logger.error(f"Training job {job.job_id} failed: {e}")

        finally:
            self._current_job = None

    async def start_training(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> TrainingJob:
        """
        Start a new training job.

        Args:
            symbols: List of symbols to train on (default: all available)
            timeframes: List of timeframes to use (default: M15, H1, H4, D1)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate

        Returns:
            TrainingJob with job information
        """
        if self._current_job and self._current_job.status == TrainingStatus.RUNNING:
            raise RuntimeError("A training job is already running")

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available - training disabled")

        # Get available symbols if not specified
        if symbols is None:
            try:
                client = await self._get_client()
                response = await client.get(f"{DATA_SERVICE_URL}/api/v1/symbols")
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list):
                    symbols = [s.get("symbol", s) if isinstance(s, dict) else s for s in result[:10]]
                else:
                    symbols = ["BTCUSD", "EURUSD", "XAUUSD"]
            except Exception:
                symbols = ["BTCUSD", "EURUSD", "XAUUSD"]

        if timeframes is None:
            timeframes = ["M15", "H1", "H4", "D1"]

        # Create job
        job = TrainingJob(
            job_id=self._generate_job_id(),
            status=TrainingStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            timeframes=timeframes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        self._jobs[job.job_id] = job
        self._current_job = job
        self._save_history()

        # Start training in background
        self._training_task = asyncio.create_task(self._run_training(job))

        logger.info(f"Created training job {job.job_id}")
        return job

    async def cancel_training(self, job_id: str) -> bool:
        """Cancel a running training job."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        if job.status != TrainingStatus.RUNNING:
            return False

        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        self._save_history()

        if self._training_task and not self._training_task.done():
            self._training_task.cancel()

        logger.info(f"Cancelled training job {job_id}")
        return True

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get currently running job."""
        return self._current_job

    def get_all_jobs(self, limit: int = 50) -> List[TrainingJob]:
        """Get all training jobs."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def get_latest_model(self) -> Optional[str]:
        """Get path to the latest trained model."""
        # Find most recent completed job with model
        completed_jobs = [
            j for j in self._jobs.values()
            if j.status == TrainingStatus.COMPLETED and j.model_path
        ]
        if not completed_jobs:
            return None

        completed_jobs.sort(key=lambda j: j.completed_at, reverse=True)
        return completed_jobs[0].model_path

    def is_training(self) -> bool:
        """Check if a training job is currently running."""
        return self._current_job is not None and self._current_job.status == TrainingStatus.RUNNING

    def factory_reset(self, delete_models: bool = False) -> Dict[str, Any]:
        """
        Reset training data to factory defaults.

        Args:
            delete_models: If True, also delete all trained model files

        Returns:
            Result of factory reset operation
        """
        import shutil

        deleted_files = []
        errors = []

        # Delete training_history.json
        if self._history_file.exists():
            try:
                self._history_file.unlink()
                deleted_files.append("training_history.json")
                logger.info("Factory reset: Deleted training_history.json")
            except Exception as e:
                errors.append(f"training_history.json: {str(e)}")
                logger.error(f"Factory reset: Failed to delete training_history.json: {e}")

        # Clear in-memory jobs
        self._jobs.clear()
        self._current_job = None
        deleted_files.append("in-memory jobs (cleared)")

        # Optionally delete model files
        if delete_models:
            model_dir = self._model_path
            if model_dir.exists():
                model_count = 0
                for model_file in model_dir.glob("candlestick_model_*.pt*"):
                    try:
                        model_file.unlink()
                        model_count += 1
                    except Exception as e:
                        errors.append(f"{model_file.name}: {str(e)}")
                        logger.error(f"Factory reset: Failed to delete {model_file.name}: {e}")

                if model_count > 0:
                    deleted_files.append(f"models/ ({model_count} files)")
                    logger.info(f"Factory reset: Deleted {model_count} model files")

        if errors:
            return {
                "success": False,
                "deleted": deleted_files,
                "errors": errors,
                "message": "Factory reset completed with errors"
            }

        return {
            "success": True,
            "deleted": deleted_files,
            "message": "Factory reset completed successfully. Training history has been cleared."
        }


# Global singleton
training_service = TrainingService()
