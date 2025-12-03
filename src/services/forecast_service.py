"""
NHITS-inspired Neural Forecast Service for price predictions.

This is a simplified implementation using PyTorch directly,
avoiding the Ray dependency issues with NeuralForecast on Windows.
"""

import logging
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.forecast_data import (
    ForecastResult,
    ForecastConfig,
    ForecastTrainingResult,
    ForecastModelInfo,
)
from src.models.trading_data import TimeSeriesData
from src.config.settings import settings

logger = logging.getLogger(__name__)


class NHITSBlock(nn.Module):
    """Single NHITS block with pooling and interpolation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        pool_kernel_size: int = 2,
    ):
        super().__init__()
        self.pool_kernel_size = pool_kernel_size
        self.output_size = output_size

        # Pooling layer
        self.pooling = nn.AvgPool1d(
            kernel_size=pool_kernel_size,
            stride=pool_kernel_size,
            ceil_mode=True
        )

        # Calculate pooled input size
        pooled_size = (input_size + pool_kernel_size - 1) // pool_kernel_size

        # MLP layers
        self.fc1 = nn.Linear(pooled_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_size)
        # Add channel dim for pooling: (batch, 1, input_size)
        x = x.unsqueeze(1)
        x = self.pooling(x)
        x = x.squeeze(1)

        # MLP
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class SimpleNHITS(nn.Module):
    """
    Simplified NHITS model for time series forecasting.

    Architecture:
    - Multiple stacked blocks with different pooling sizes
    - Each block captures patterns at different scales
    - Outputs are summed for final prediction
    """

    def __init__(
        self,
        input_size: int = 168,
        output_size: int = 24,
        hidden_size: int = 256,
        n_pool_kernel_sizes: List[int] = [2, 2, 1],
        n_quantiles: int = 3,  # For probabilistic forecasts
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_quantiles = n_quantiles

        # Create blocks with different pooling sizes
        self.blocks = nn.ModuleList([
            NHITSBlock(
                input_size=input_size,
                output_size=output_size * n_quantiles,
                hidden_size=hidden_size,
                pool_kernel_size=kernel_size
            )
            for kernel_size in n_pool_kernel_sizes
        ])

        # Final layer to combine block outputs
        self.final = nn.Linear(
            len(n_pool_kernel_sizes) * output_size * n_quantiles,
            output_size * n_quantiles
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_size)
        block_outputs = []

        for block in self.blocks:
            out = block(x)
            block_outputs.append(out)

        # Concatenate all block outputs
        combined = torch.cat(block_outputs, dim=1)

        # Final projection
        output = self.final(combined)

        # Reshape to (batch, output_size, n_quantiles)
        output = output.view(-1, self.output_size, self.n_quantiles)

        return output


class ForecastService:
    """Service for NHITS-based time series forecasting."""

    def __init__(self):
        self.models: Dict[str, SimpleNHITS] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.model_path = Path(settings.nhits_model_path)
        self.horizon = settings.nhits_horizon
        self.input_size = settings.nhits_input_size

        # Device selection
        self.device = torch.device(
            "cuda" if settings.nhits_use_gpu and torch.cuda.is_available() else "cpu"
        )

        # ThreadPoolExecutor for running blocking PyTorch operations
        # without blocking the async event loop
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nhits")

        # Ensure model directory exists
        self.model_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ForecastService initialized - "
            f"horizon={self.horizon}h, input_size={self.input_size}, "
            f"device={self.device}, model_path={self.model_path}"
        )

    def _prepare_data(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
    ) -> pd.DataFrame:
        """Convert TimeSeriesData to DataFrame."""
        if not time_series:
            raise ValueError("Empty time series data")

        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(ts.timestamp),
                'close': float(ts.close),
            }
            for ts in time_series
        ])

        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        return df

    def _create_sequences(
        self,
        data: np.ndarray,
        input_size: int,
        output_size: int
    ) -> tuple:
        """Create input/output sequences for training."""
        X, y = [], []

        for i in range(len(data) - input_size - output_size + 1):
            X.append(data[i:i + input_size])
            y.append(data[i + input_size:i + input_size + output_size])

        return np.array(X), np.array(y)

    def _normalize(self, data: np.ndarray) -> tuple:
        """Normalize data and return scaler parameters."""
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        normalized = (data - mean) / std
        return normalized, mean, std

    def _denormalize(self, data: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Denormalize data."""
        return data * std + mean

    def _create_model(self) -> SimpleNHITS:
        """Create a new NHITS model."""
        model = SimpleNHITS(
            input_size=self.input_size,
            output_size=self.horizon,
            hidden_size=settings.nhits_hidden_size,
            n_pool_kernel_sizes=settings.nhits_n_pool_kernel_size,
            n_quantiles=3,  # 10%, 50%, 90%
        )
        return model.to(self.device)

    def _get_metadata_path(self, symbol: str) -> Path:
        """Get path to model metadata file."""
        return self.model_path / f"{symbol}_metadata.json"

    def _get_model_path(self, symbol: str) -> Path:
        """Get path to model file."""
        return self.model_path / f"{symbol}_model.pt"

    def _save_metadata(self, symbol: str, metadata: Dict) -> None:
        """Save model metadata to disk."""
        path = self._get_metadata_path(symbol)
        meta_copy = metadata.copy()
        if 'trained_at' in meta_copy and isinstance(meta_copy['trained_at'], datetime):
            meta_copy['trained_at'] = meta_copy['trained_at'].isoformat()
        with open(path, 'w') as f:
            json.dump(meta_copy, f, indent=2)

    def _load_metadata(self, symbol: str) -> Optional[Dict]:
        """Load model metadata from disk."""
        path = self._get_metadata_path(symbol)
        if not path.exists():
            return None

        try:
            with open(path, 'r') as f:
                meta = json.load(f)
                if 'trained_at' in meta:
                    meta['trained_at'] = datetime.fromisoformat(meta['trained_at'])
                return meta
        except Exception as e:
            logger.warning(f"Failed to load metadata for {symbol}: {e}")
            return None

    def _should_retrain(self, symbol: str) -> bool:
        """Check if model should be retrained based on age."""
        metadata = self._load_metadata(symbol)
        if not metadata:
            return True

        trained_at = metadata.get('trained_at')
        if not trained_at:
            return True

        age_days = (datetime.utcnow() - trained_at).days
        return age_days >= settings.nhits_auto_retrain_days

    def _train_sync(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
    ) -> ForecastTrainingResult:
        """
        Synchronous training function to run in ThreadPoolExecutor.
        This prevents blocking the async event loop.
        """
        logger.info(f"Training NHITS model for {symbol} (sync)")
        start_time = datetime.utcnow()

        try:
            df = self._prepare_data(time_series, symbol)
            prices = df['close'].values

            min_required = self.input_size + self.horizon
            if len(prices) < min_required:
                raise ValueError(
                    f"Insufficient data: {len(prices)} rows, need {min_required}"
                )

            # Normalize data
            normalized, mean, std = self._normalize(prices)

            # Create sequences
            X, y = self._create_sequences(normalized, self.input_size, self.horizon)

            if len(X) < 10:
                raise ValueError(f"Not enough sequences for training: {len(X)}")

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Create dataset and loader
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=settings.nhits_batch_size,
                shuffle=True
            )

            # Create model
            model = self._create_model()

            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=settings.nhits_learning_rate)

            # Quantile loss for probabilistic forecasting
            quantiles = torch.tensor([0.1, 0.5, 0.9], device=self.device)

            def quantile_loss(pred, target):
                # pred shape: (batch, horizon, n_quantiles)
                # target shape: (batch, horizon)
                # Expand target to match pred shape
                target_expanded = target.unsqueeze(-1)  # (batch, horizon, 1)
                errors = target_expanded - pred  # (batch, horizon, n_quantiles)

                # Calculate quantile loss for each quantile
                losses = torch.max(
                    (quantiles - 1) * errors,
                    quantiles * errors
                )
                return losses.mean()

            # Training loop
            model.train()
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(settings.nhits_max_steps):
                epoch_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    output = model(batch_X)

                    # Calculate quantile loss
                    loss = quantile_loss(output, batch_y)

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= len(loader)

                # Early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 50:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if epoch % 100 == 0:
                    logger.debug(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

            # Save model
            model_path = self._get_model_path(symbol)
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std,
            }, model_path)

            self.models[symbol] = model

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Save metadata
            metadata = {
                'symbol': symbol,
                'trained_at': datetime.utcnow(),
                'training_samples': len(X),
                'horizon': self.horizon,
                'input_size': self.input_size,
                'duration_seconds': duration,
                'final_loss': best_loss,
                'mean': float(mean),
                'std': float(std),
            }
            self._save_metadata(symbol, metadata)
            self.model_metadata[symbol] = metadata

            logger.info(
                f"NHITS model trained for {symbol}: "
                f"{len(X)} samples, {duration:.1f}s, loss={best_loss:.6f}"
            )

            return ForecastTrainingResult(
                symbol=symbol,
                trained_at=datetime.utcnow(),
                training_samples=len(X),
                training_duration_seconds=duration,
                model_path=str(model_path),
                metrics={'final_loss': best_loss},
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to train NHITS model for {symbol}: {e}")
            return ForecastTrainingResult(
                symbol=symbol,
                trained_at=datetime.utcnow(),
                training_samples=0,
                training_duration_seconds=0,
                model_path="",
                metrics={},
                success=False,
                error_message=str(e),
            )

    async def train(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
    ) -> ForecastTrainingResult:
        """
        Train NHITS model on historical data.
        Runs the blocking PyTorch operations in a ThreadPoolExecutor.
        """
        logger.info(f"Training NHITS model for {symbol}")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._train_sync,
            time_series,
            symbol,
        )

    def _load_model(self, symbol: str) -> Optional[SimpleNHITS]:
        """Load a saved model from disk."""
        if symbol in self.models:
            return self.models[symbol]

        model_path = self._get_model_path(symbol)
        if not model_path.exists():
            return None

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = self._create_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.models[symbol] = model
            metadata = self._load_metadata(symbol) or {}
            metadata['mean'] = checkpoint.get('mean', 0)
            metadata['std'] = checkpoint.get('std', 1)
            self.model_metadata[symbol] = metadata

            logger.info(f"Loaded NHITS model for {symbol}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model for {symbol}: {e}")
            return None

    def _forecast_sync(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
        model: SimpleNHITS,
    ) -> ForecastResult:
        """
        Synchronous forecast function to run in ThreadPoolExecutor.
        This prevents blocking the async event loop.
        """
        try:
            df = self._prepare_data(time_series, symbol)
            prices = df['close'].values

            if len(prices) < self.input_size:
                logger.warning(
                    f"Insufficient data for prediction: {len(prices)} < {self.input_size}"
                )
                return self._create_empty_forecast(symbol, time_series)

            # Get normalization parameters
            metadata = self.model_metadata.get(symbol, {})
            mean = metadata.get('mean', np.mean(prices))
            std = metadata.get('std', np.std(prices) + 1e-8)

            # Prepare input
            input_data = (prices[-self.input_size:] - mean) / std
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

            # Generate forecast
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            # output shape: (1, horizon, 3) for 3 quantiles
            output = output.cpu().numpy()[0]

            # Denormalize
            predicted_low = output[:, 0] * std + mean  # 10th percentile
            predicted_median = output[:, 1] * std + mean  # 50th percentile (main prediction)
            predicted_high = output[:, 2] * std + mean  # 90th percentile

            predicted_prices = predicted_median.tolist()
            confidence_low = predicted_low.tolist()
            confidence_high = predicted_high.tolist()

            current_price = prices[-1]

            # Calculate trend probabilities
            price_changes = np.diff([current_price] + predicted_prices)
            trend_up_prob = float(np.mean(price_changes > 0))

            # Calculate volatility
            volatilities = [
                (hi - lo) / mid if mid > 0 else 0
                for hi, lo, mid in zip(confidence_high, confidence_low, predicted_prices)
            ]
            avg_volatility = float(np.mean(volatilities)) if volatilities else 0.0

            result = ForecastResult(
                symbol=symbol,
                forecast_timestamp=datetime.utcnow(),
                horizon_hours=self.horizon,

                predicted_prices=predicted_prices,
                confidence_low=confidence_low,
                confidence_high=confidence_high,

                predicted_price_1h=predicted_prices[0] if len(predicted_prices) > 0 else None,
                predicted_price_4h=predicted_prices[3] if len(predicted_prices) > 3 else predicted_prices[-1] if predicted_prices else None,
                predicted_price_24h=predicted_prices[-1] if predicted_prices else None,

                current_price=current_price,
                predicted_change_percent_1h=self._calc_change_pct(current_price, predicted_prices[0]) if predicted_prices else None,
                predicted_change_percent_4h=self._calc_change_pct(current_price, predicted_prices[3]) if len(predicted_prices) > 3 else None,
                predicted_change_percent_24h=self._calc_change_pct(current_price, predicted_prices[-1]) if predicted_prices else None,

                trend_up_probability=trend_up_prob,
                trend_down_probability=1 - trend_up_prob,

                predicted_volatility=avg_volatility,
                model_confidence=max(0.0, min(1.0, 1 - avg_volatility)),

                last_training_date=metadata.get('trained_at'),
                training_samples=metadata.get('training_samples'),
            )

            logger.info(
                f"NHITS forecast for {symbol}: "
                f"{current_price:.5f} -> {predicted_prices[-1]:.5f} "
                f"({result.predicted_change_percent_24h:+.2f}%) "
                f"[confidence: {result.model_confidence:.1%}]"
            )

            return result

        except Exception as e:
            logger.error(f"Forecast failed for {symbol}: {e}")
            return self._create_empty_forecast(symbol, time_series)

    async def forecast(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
        config: Optional[ForecastConfig] = None,
    ) -> ForecastResult:
        """
        Generate price forecast using NHITS.
        Runs the blocking PyTorch operations in a ThreadPoolExecutor.
        """
        config = config or ForecastConfig(symbol=symbol)

        # Check if we need to train or retrain
        model = self._load_model(symbol)

        if model is None or config.retrain or self._should_retrain(symbol):
            logger.info(f"Training/retraining NHITS model for {symbol}")
            result = await self.train(time_series, symbol)
            if not result.success:
                return self._create_empty_forecast(symbol, time_series)
            model = self.models[symbol]

        # Run inference in thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._forecast_sync,
            time_series,
            symbol,
            model,
        )

    def _calc_change_pct(self, current: float, predicted: float) -> float:
        """Calculate percentage change."""
        if current == 0:
            return 0.0
        return ((predicted - current) / current) * 100

    def _create_empty_forecast(
        self,
        symbol: str,
        time_series: List[TimeSeriesData],
    ) -> ForecastResult:
        """Create an empty forecast result when prediction fails."""
        current_price = time_series[-1].close if time_series else 0.0

        return ForecastResult(
            symbol=symbol,
            forecast_timestamp=datetime.utcnow(),
            horizon_hours=self.horizon,
            predicted_prices=[],
            confidence_low=[],
            confidence_high=[],
            current_price=current_price,
            trend_up_probability=0.5,
            trend_down_probability=0.5,
            model_confidence=0.0,
        )

    def get_model_info(self, symbol: str) -> ForecastModelInfo:
        """Get information about a trained model."""
        model_path = self._get_model_path(symbol)
        metadata = self._load_metadata(symbol)

        return ForecastModelInfo(
            symbol=symbol,
            model_exists=model_path.exists(),
            model_path=str(model_path) if model_path.exists() else None,
            last_trained=metadata.get('trained_at') if metadata else None,
            training_samples=metadata.get('training_samples') if metadata else None,
            horizon=self.horizon,
            input_size=self.input_size,
            metrics=metadata.get('metrics', {}) if metadata else {},
        )

    def list_models(self) -> List[ForecastModelInfo]:
        """List all trained models."""
        models = []
        for path in self.model_path.glob("*_model.pt"):
            symbol = path.stem.replace("_model", "")
            models.append(self.get_model_info(symbol))
        return models


# Singleton instance
forecast_service = ForecastService()
