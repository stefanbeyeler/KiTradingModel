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
        n_features: int = 1,  # Number of input features (multi-variate support)
    ):
        super().__init__()
        self.pool_kernel_size = pool_kernel_size
        self.output_size = output_size
        self.n_features = n_features

        # Pooling layer
        self.pooling = nn.AvgPool1d(
            kernel_size=pool_kernel_size,
            stride=pool_kernel_size,
            ceil_mode=True
        )

        # Calculate pooled input size (now includes features)
        pooled_size = (input_size + pool_kernel_size - 1) // pool_kernel_size
        total_input_size = pooled_size * n_features

        # MLP layers
        self.fc1 = nn.Linear(total_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_features, input_size) for multi-variate
        # or (batch, input_size) for univariate
        if x.dim() == 2:
            # Univariate: add feature dimension
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        # Apply pooling to each feature
        pooled_features = []
        for i in range(x.shape[1]):
            feat = x[:, i:i+1, :]  # (batch, 1, input_size)
            pooled = self.pooling(feat)  # (batch, 1, pooled_size)
            pooled_features.append(pooled.squeeze(1))

        # Concatenate all pooled features
        x = torch.cat(pooled_features, dim=1)  # (batch, n_features * pooled_size)

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
    - Supports multi-variate input (price + technical indicators)
    """

    def __init__(
        self,
        input_size: int = 168,
        output_size: int = 24,
        hidden_size: int = 256,
        n_pool_kernel_sizes: List[int] = [2, 2, 1],
        n_quantiles: int = 3,  # For probabilistic forecasts
        n_features: int = 1,  # Number of input features (1 = univariate, >1 = multi-variate)
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_quantiles = n_quantiles
        self.n_features = n_features

        # Create blocks with different pooling sizes
        self.blocks = nn.ModuleList([
            NHITSBlock(
                input_size=input_size,
                output_size=output_size * n_quantiles,
                hidden_size=hidden_size,
                pool_kernel_size=kernel_size,
                n_features=n_features,
            )
            for kernel_size in n_pool_kernel_sizes
        ])

        # Final layer to combine block outputs
        self.final = nn.Linear(
            len(n_pool_kernel_sizes) * output_size * n_quantiles,
            output_size * n_quantiles
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_features, input_size) for multi-variate
        # or (batch, input_size) for univariate
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

    # Feature names for multi-variate forecasting
    INDICATOR_FEATURES = [
        'rsi', 'macd_main', 'macd_signal', 'adx', 'adx_plus_di', 'adx_minus_di',
        'atr', 'cci', 'stoch_k', 'stoch_d', 'bb_upper', 'bb_middle', 'bb_lower',
        'ma100', 'ichimoku_tenkan', 'ichimoku_kijun', 'strength_4h', 'strength_1d', 'strength_1w'
    ]

    def _prepare_data(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
    ) -> pd.DataFrame:
        """Convert TimeSeriesData to DataFrame with all features."""
        if not time_series:
            raise ValueError("Empty time series data")

        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(ts.timestamp),
                'close': float(ts.close),
                'open': float(ts.open),
                'high': float(ts.high),
                'low': float(ts.low),
                # Technical Indicators
                'rsi': ts.rsi,
                'macd_main': ts.macd_main,
                'macd_signal': ts.macd_signal,
                'adx': ts.adx,
                'adx_plus_di': ts.adx_plus_di,
                'adx_minus_di': ts.adx_minus_di,
                'atr': ts.atr,
                'cci': ts.cci,
                'stoch_k': ts.stoch_k,
                'stoch_d': ts.stoch_d,
                'bb_upper': ts.bb_upper,
                'bb_middle': ts.bb_middle,
                'bb_lower': ts.bb_lower,
                'ma100': ts.ma100,
                'ichimoku_tenkan': ts.ichimoku_tenkan,
                'ichimoku_kijun': ts.ichimoku_kijun,
                'strength_4h': ts.strength_4h,
                'strength_1d': ts.strength_1d,
                'strength_1w': ts.strength_1w,
            }
            for ts in time_series
        ])

        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        return df

    def _get_available_features(self, df: pd.DataFrame) -> List[str]:
        """Determine which indicator features have sufficient non-null data."""
        available = []
        min_coverage = 0.5  # At least 50% non-null values required

        for feature in self.INDICATOR_FEATURES:
            if feature in df.columns:
                coverage = df[feature].notna().mean()
                if coverage >= min_coverage:
                    available.append(feature)

        return available

    def _prepare_multivariate_data(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
    ) -> tuple:
        """
        Prepare multi-variate data for training/inference.

        Returns:
            Tuple of (features_array, feature_means, feature_stds, n_features)
            features_array shape: (n_samples, n_features) where features are in order:
            [close, indicator1, indicator2, ...]
        """
        # Always include close price as first feature
        all_features = ['close'] + feature_names

        # Fill missing values with forward fill, then backward fill
        for col in all_features:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # Extract feature values
        feature_arrays = []
        feature_means = {}
        feature_stds = {}

        for feat in all_features:
            if feat in df.columns:
                values = df[feat].values.astype(np.float32)
                mean = np.mean(values)
                std = np.std(values) + 1e-8
                normalized = (values - mean) / std
                feature_arrays.append(normalized)
                feature_means[feat] = mean
                feature_stds[feat] = std

        # Stack features: shape (n_features, n_samples)
        features = np.stack(feature_arrays, axis=0)

        return features, feature_means, feature_stds, len(all_features)

    def _create_sequences(
        self,
        data: np.ndarray,
        input_size: int,
        output_size: int
    ) -> tuple:
        """
        Create input/output sequences for training.

        For univariate data: data shape (n_samples,)
        For multivariate data: data shape (n_features, n_samples)
        """
        X, y = [], []

        if data.ndim == 1:
            # Univariate case
            for i in range(len(data) - input_size - output_size + 1):
                X.append(data[i:i + input_size])
                y.append(data[i + input_size:i + input_size + output_size])
        else:
            # Multivariate case: data shape (n_features, n_samples)
            n_features, n_samples = data.shape
            for i in range(n_samples - input_size - output_size + 1):
                # X: (n_features, input_size)
                X.append(data[:, i:i + input_size])
                # y: only predict close price (first feature)
                y.append(data[0, i + input_size:i + input_size + output_size])

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

    def _create_model(self, n_features: int = 1) -> SimpleNHITS:
        """Create a new NHITS model with specified number of features."""
        model = SimpleNHITS(
            input_size=self.input_size,
            output_size=self.horizon,
            hidden_size=settings.nhits_hidden_size,
            n_pool_kernel_sizes=settings.nhits_n_pool_kernel_size,
            n_quantiles=3,  # 10%, 50%, 90%
            n_features=n_features,  # Multi-variate support
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
        Supports multi-variate training with technical indicators.
        """
        logger.info(f"Training NHITS model for {symbol} (sync)")
        start_time = datetime.utcnow()

        try:
            df = self._prepare_data(time_series, symbol)

            min_required = self.input_size + self.horizon
            if len(df) < min_required:
                raise ValueError(
                    f"Insufficient data: {len(df)} rows, need {min_required}"
                )

            # Check which indicator features are available
            available_features = self._get_available_features(df)
            n_features = 1 + len(available_features)  # close + indicators

            logger.info(
                f"NHITS training for {symbol}: Using {n_features} features "
                f"(close + {len(available_features)} indicators: {available_features[:5]}...)"
            )

            # Prepare multi-variate data
            if available_features:
                features, feature_means, feature_stds, n_features = self._prepare_multivariate_data(
                    df, available_features
                )
                # Create sequences for multi-variate data
                X, y = self._create_sequences(features, self.input_size, self.horizon)
            else:
                # Fallback to univariate (close only)
                prices = df['close'].values
                normalized, mean, std = self._normalize(prices)
                X, y = self._create_sequences(normalized, self.input_size, self.horizon)
                feature_means = {'close': mean}
                feature_stds = {'close': std}
                n_features = 1

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

            # Create model with correct number of features
            model = self._create_model(n_features=n_features)

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

            # Save model with feature information
            model_path = self._get_model_path(symbol)
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_features': n_features,
                'feature_names': ['close'] + available_features,
                'feature_means': {k: float(v) for k, v in feature_means.items()},
                'feature_stds': {k: float(v) for k, v in feature_stds.items()},
                # Legacy support
                'mean': float(feature_means.get('close', 0)),
                'std': float(feature_stds.get('close', 1)),
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
                'n_features': n_features,
                'feature_names': ['close'] + available_features,
                'mean': float(feature_means.get('close', 0)),
                'std': float(feature_stds.get('close', 1)),
                'feature_means': {k: float(v) for k, v in feature_means.items()},
                'feature_stds': {k: float(v) for k, v in feature_stds.items()},
            }
            self._save_metadata(symbol, metadata)
            self.model_metadata[symbol] = metadata

            logger.info(
                f"NHITS model trained for {symbol}: "
                f"{len(X)} samples, {n_features} features, {duration:.1f}s, loss={best_loss:.6f}"
            )

            return ForecastTrainingResult(
                symbol=symbol,
                trained_at=datetime.utcnow(),
                training_samples=len(X),
                training_duration_seconds=duration,
                model_path=str(model_path),
                metrics={
                    'final_loss': best_loss,
                    'n_features': n_features,
                    'features_used': ['close'] + available_features,
                },
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to train NHITS model for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        """Load a saved model from disk with multi-variate support."""
        if symbol in self.models:
            return self.models[symbol]

        model_path = self._get_model_path(symbol)
        if not model_path.exists():
            return None

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Get number of features (default to 1 for legacy models)
            n_features = checkpoint.get('n_features', 1)

            model = self._create_model(n_features=n_features)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.models[symbol] = model
            metadata = self._load_metadata(symbol) or {}

            # Load multi-variate metadata
            metadata['mean'] = checkpoint.get('mean', 0)
            metadata['std'] = checkpoint.get('std', 1)
            metadata['n_features'] = n_features
            metadata['feature_names'] = checkpoint.get('feature_names', ['close'])
            metadata['feature_means'] = checkpoint.get('feature_means', {'close': metadata['mean']})
            metadata['feature_stds'] = checkpoint.get('feature_stds', {'close': metadata['std']})

            self.model_metadata[symbol] = metadata

            logger.info(
                f"Loaded NHITS model for {symbol} "
                f"({n_features} features: {metadata['feature_names'][:3]}...)"
            )
            return model
        except Exception as e:
            logger.warning(f"Failed to load model for {symbol}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return None

    def _forecast_sync(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
        model: SimpleNHITS,
    ) -> ForecastResult:
        """
        Synchronous forecast function to run in ThreadPoolExecutor.
        Supports multi-variate forecasting with technical indicators.
        """
        try:
            df = self._prepare_data(time_series, symbol)

            if len(df) < self.input_size:
                logger.warning(
                    f"Insufficient data for prediction: {len(df)} < {self.input_size}"
                )
                return self._create_empty_forecast(symbol, time_series)

            # Get metadata including feature information
            metadata = self.model_metadata.get(symbol, {})
            n_features = metadata.get('n_features', 1)
            feature_names = metadata.get('feature_names', ['close'])
            feature_means = metadata.get('feature_means', {'close': metadata.get('mean', 0)})
            feature_stds = metadata.get('feature_stds', {'close': metadata.get('std', 1)})

            # Get close price normalization params
            close_mean = feature_means.get('close', metadata.get('mean', np.mean(df['close'].values)))
            close_std = feature_stds.get('close', metadata.get('std', np.std(df['close'].values) + 1e-8))

            # Prepare input based on model type
            if n_features > 1 and len(feature_names) > 1:
                # Multi-variate: prepare all features
                indicator_features = [f for f in feature_names if f != 'close']

                # Fill missing values
                for col in feature_names:
                    if col in df.columns:
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

                # Build feature tensor
                feature_arrays = []
                for feat in feature_names:
                    if feat in df.columns:
                        values = df[feat].values[-self.input_size:].astype(np.float32)
                        mean = feature_means.get(feat, np.mean(values))
                        std = feature_stds.get(feat, np.std(values) + 1e-8)
                        normalized = (values - mean) / std
                        feature_arrays.append(normalized)

                # Stack features: (n_features, input_size)
                input_data = np.stack(feature_arrays, axis=0)
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

                logger.debug(f"Multi-variate forecast for {symbol} with {n_features} features")
            else:
                # Univariate: close price only
                prices = df['close'].values
                input_data = (prices[-self.input_size:] - close_mean) / close_std
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

            # Generate forecast
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            # output shape: (1, horizon, 3) for 3 quantiles
            output = output.cpu().numpy()[0]

            # Denormalize using close price parameters
            predicted_low = output[:, 0] * close_std + close_mean  # 10th percentile
            predicted_median = output[:, 1] * close_std + close_mean  # 50th percentile (main prediction)
            predicted_high = output[:, 2] * close_std + close_mean  # 90th percentile

            predicted_prices = predicted_median.tolist()
            confidence_low = predicted_low.tolist()
            confidence_high = predicted_high.tolist()

            current_price = df['close'].values[-1]

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
        """Get information about a trained model including multi-variate features."""
        model_path = self._get_model_path(symbol)
        metadata = self._load_metadata(symbol)

        # Build metrics including feature information
        metrics = {}
        if metadata:
            metrics = metadata.get('metrics', {})
            metrics['n_features'] = metadata.get('n_features', 1)
            metrics['feature_names'] = metadata.get('feature_names', ['close'])
            metrics['final_loss'] = metadata.get('final_loss')

        return ForecastModelInfo(
            symbol=symbol,
            model_exists=model_path.exists(),
            model_path=str(model_path) if model_path.exists() else None,
            last_trained=metadata.get('trained_at') if metadata else None,
            training_samples=metadata.get('training_samples') if metadata else None,
            horizon=self.horizon,
            input_size=self.input_size,
            metrics=metrics,
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
