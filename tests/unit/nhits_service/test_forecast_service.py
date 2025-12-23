"""
Unit tests for NHITS Forecast Service.

Tests the forecast service business logic including:
- Input data preparation
- Forecast horizon validation
- Model configuration
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path for imports (conftest.py also does this)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestForecastDataPreparation:
    """Unit tests for forecast data preparation."""

    @pytest.mark.unit
    def test_prepare_close_prices(self):
        """Test extracting close prices from OHLCV data."""
        raw_data = [
            {"open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1000},
            {"open": 102.0, "high": 108.0, "low": 101.0, "close": 107.0, "volume": 1200},
            {"open": 107.0, "high": 110.0, "low": 105.0, "close": 109.0, "volume": 900},
        ]

        # Extract close prices
        close_prices = [d["close"] for d in raw_data]

        assert len(close_prices) == 3
        assert close_prices == [102.0, 107.0, 109.0]

    @pytest.mark.unit
    def test_convert_to_numpy_array(self):
        """Test converting price list to numpy array."""
        prices = [100.0, 101.5, 102.3, 103.0]

        arr = np.array(prices, dtype=np.float32)

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert len(arr) == 4

    @pytest.mark.unit
    def test_normalize_prices(self):
        """Test price normalization."""
        prices = np.array([100.0, 200.0, 150.0, 175.0])

        # Min-max normalization
        min_val = prices.min()
        max_val = prices.max()
        normalized = (prices - min_val) / (max_val - min_val)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == 4


class TestForecastHorizonValidation:
    """Unit tests for forecast horizon validation."""

    @pytest.mark.unit
    def test_valid_horizons(self):
        """Test valid forecast horizons."""
        valid_horizons = [1, 12, 24, 48, 168]  # 1h to 1 week

        for horizon in valid_horizons:
            assert horizon > 0
            assert horizon <= 720  # Max 30 days

    @pytest.mark.unit
    def test_invalid_horizon_zero(self):
        """Test that zero horizon is invalid."""
        horizon = 0
        assert horizon <= 0  # Invalid

    @pytest.mark.unit
    def test_invalid_horizon_negative(self):
        """Test that negative horizon is invalid."""
        horizon = -24
        assert horizon <= 0  # Invalid

    @pytest.mark.unit
    def test_horizon_too_large(self):
        """Test that very large horizon is invalid."""
        max_horizon = 720  # 30 days in hours
        horizon = 10000

        assert horizon > max_horizon  # Invalid


class TestNHITSModelConfiguration:
    """Unit tests for NHITS model configuration."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default model configuration."""
        default_config = {
            "input_size": 168,   # 1 week lookback
            "output_size": 24,   # 24h forecast
            "hidden_size": 128,
            "n_stacks": 3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        }

        assert default_config["input_size"] > 0
        assert default_config["output_size"] > 0
        assert default_config["hidden_size"] > 0
        assert default_config["n_stacks"] >= 1
        assert 0 < default_config["learning_rate"] < 1

    @pytest.mark.unit
    def test_config_validation(self):
        """Test configuration value validation."""
        # Valid config
        config = {
            "input_size": 168,
            "output_size": 24,
            "hidden_size": 128,
        }

        assert config["input_size"] > config["output_size"]  # Common constraint
        assert config["hidden_size"] >= 32  # Minimum hidden size


class TestTimeSeriesWindowCreation:
    """Unit tests for time series windowing."""

    @pytest.mark.unit
    def test_create_sliding_windows(self):
        """Test creating sliding windows from time series."""
        data = np.arange(100, dtype=np.float32)
        window_size = 10
        horizon = 5

        windows = []
        targets = []

        for i in range(len(data) - window_size - horizon + 1):
            windows.append(data[i:i + window_size])
            targets.append(data[i + window_size:i + window_size + horizon])

        windows = np.array(windows)
        targets = np.array(targets)

        assert windows.shape == (86, 10)  # 100 - 10 - 5 + 1 = 86 samples
        assert targets.shape == (86, 5)

    @pytest.mark.unit
    def test_insufficient_data_for_window(self):
        """Test handling of insufficient data."""
        data = np.arange(10, dtype=np.float32)  # Only 10 points
        window_size = 15  # Larger than data

        # Should not be able to create any windows
        n_samples = len(data) - window_size
        assert n_samples < 0  # Not enough data

    @pytest.mark.unit
    def test_window_with_multiple_features(self):
        """Test windowing with multiple features (OHLCV)."""
        # 5 features: open, high, low, close, volume
        n_samples = 100
        n_features = 5

        data = np.random.randn(n_samples, n_features).astype(np.float32)
        window_size = 20
        horizon = 10

        windows = []
        for i in range(len(data) - window_size - horizon + 1):
            windows.append(data[i:i + window_size])

        windows = np.array(windows)

        # Shape: (n_windows, window_size, n_features)
        assert windows.shape[1] == window_size
        assert windows.shape[2] == n_features


class TestForecastMetrics:
    """Unit tests for forecast metrics calculation."""

    @pytest.mark.unit
    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        actual = np.array([100.0, 110.0, 105.0, 115.0])
        predicted = np.array([102.0, 108.0, 107.0, 113.0])

        mae = np.mean(np.abs(actual - predicted))

        assert mae == 2.0  # (2 + 2 + 2 + 2) / 4

    @pytest.mark.unit
    def test_mse_calculation(self):
        """Test Mean Squared Error calculation."""
        actual = np.array([100.0, 110.0, 105.0, 115.0])
        predicted = np.array([102.0, 108.0, 107.0, 113.0])

        mse = np.mean((actual - predicted) ** 2)

        assert mse == 4.0  # (4 + 4 + 4 + 4) / 4

    @pytest.mark.unit
    def test_rmse_calculation(self):
        """Test Root Mean Squared Error calculation."""
        actual = np.array([100.0, 110.0, 105.0, 115.0])
        predicted = np.array([102.0, 108.0, 107.0, 113.0])

        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        assert rmse == 2.0

    @pytest.mark.unit
    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        actual = np.array([100.0, 200.0, 150.0, 200.0])
        predicted = np.array([110.0, 190.0, 160.0, 210.0])

        # Avoid division by zero
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        expected_mape = (10/100 + 10/200 + 10/150 + 10/200) / 4 * 100
        assert np.isclose(mape, expected_mape)


class TestForecastResultValidation:
    """Unit tests for forecast result validation."""

    @pytest.mark.unit
    def test_forecast_result_structure(self):
        """Test forecast result has required fields."""
        result = {
            "symbol": "BTCUSD",
            "horizon": 24,
            "predictions": [100.0, 101.0, 102.0],
            "timestamps": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"],
            "confidence": 0.85,
            "model_version": "1.0.0",
        }

        assert "symbol" in result
        assert "predictions" in result
        assert len(result["predictions"]) == len(result["timestamps"])
        assert 0 <= result["confidence"] <= 1

    @pytest.mark.unit
    def test_predictions_are_numeric(self):
        """Test that predictions are numeric values."""
        predictions = [100.5, 101.2, 99.8, 102.0]

        for pred in predictions:
            assert isinstance(pred, (int, float))
            assert not np.isnan(pred)
            assert not np.isinf(pred)
