"""
Data Pipeline für CNN-LSTM Training.

Laedt Trainingsdaten ueber den Data Service mit 3-Layer-Caching.
Erstellt Training-Labels fuer Multi-Task Learning.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

# =============================================================================
# Configuration
# =============================================================================

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")

# Sequenzlaengen pro Timeframe
SEQUENCE_LENGTHS = {
    "M1": 480,
    "M5": 288,
    "M15": 192,
    "M30": 192,
    "H1": 168,
    "H4": 168,
    "D1": 120,
    "W1": 52,
    "MN": 24,
}

# Forecast-Horizonte (in Bars)
FORECAST_HORIZONS = {
    "M1": [60, 240, 1440, 10080],     # 1h, 4h, 1d, 1w in M1 bars
    "M5": [12, 48, 288, 2016],        # 1h, 4h, 1d, 1w in M5 bars
    "M15": [4, 16, 96, 672],          # 1h, 4h, 1d, 1w in M15 bars
    "M30": [2, 8, 48, 336],           # 1h, 4h, 1d, 1w in M30 bars
    "H1": [1, 4, 24, 168],            # 1h, 4h, 1d, 1w in H1 bars
    "H4": [1, 1, 6, 42],              # 4h, 4h, 1d, 1w in H4 bars (min 1)
    "D1": [1, 1, 1, 7],               # 1d, 1d, 1d, 1w in D1 bars
    "W1": [1, 1, 1, 1],               # All 1w in W1 bars
    "MN": [1, 1, 1, 1],               # All 1m in MN bars
}

# Pattern-Typen (16)
PATTERN_TYPES = [
    "head_and_shoulders", "inverse_head_and_shoulders",
    "double_top", "double_bottom",
    "triple_top", "triple_bottom",
    "ascending_triangle", "descending_triangle", "symmetrical_triangle",
    "bull_flag", "bear_flag",
    "cup_and_handle",
    "rising_wedge", "falling_wedge",
    "channel_up", "channel_down",
]

# Regime-Typen (4)
REGIME_TYPES = ["bull_trend", "bear_trend", "sideways", "high_volatility"]


class DataPipeline:
    """
    Data Pipeline fuer Training-Daten.

    Laedt OHLCV-Daten vom Data Service und erstellt Labels fuer:
    - Preis-Vorhersage (Future Returns)
    - Pattern-Klassifikation (Swing-basierte Erkennung)
    - Regime-Vorhersage (Volatilitaets-/Trend-basiert)
    """

    def __init__(self):
        self._http_client = None

    async def _get_http_client(self):
        """Lazy initialization des HTTP-Clients."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    def get_sequence_length(self, timeframe: str) -> int:
        """Gibt die Sequenzlaenge fuer einen Timeframe zurueck."""
        return SEQUENCE_LENGTHS.get(timeframe.upper(), 168)

    def get_forecast_horizons(self, timeframe: str) -> list[int]:
        """Gibt die Forecast-Horizonte fuer einen Timeframe zurueck."""
        return FORECAST_HORIZONS.get(timeframe.upper(), [1, 4, 24, 168])

    # =========================================================================
    # Data Fetching
    # =========================================================================

    async def fetch_training_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        days: int = 365
    ) -> list[dict]:
        """
        Holt Training-Daten vom Data Service.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            days: Anzahl Tage an Daten

        Returns:
            Liste von OHLCV-Dictionaries
        """
        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{DATA_SERVICE_URL}/api/v1/training-data/{symbol}",
                params={"timeframe": timeframe, "days": days}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching training data for {symbol}: {e}")
            return []

    async def fetch_available_symbols(self) -> list[str]:
        """Holt Liste verfuegbarer Symbole."""
        try:
            client = await self._get_http_client()
            response = await client.get(f"{DATA_SERVICE_URL}/api/v1/symbols")
            response.raise_for_status()
            return response.json().get("symbols", [])
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def extract_ohlcv_arrays(self, data: list[dict]) -> dict[str, np.ndarray]:
        """Extrahiert OHLCV-Arrays aus Daten."""
        return {
            "opens": np.array([d.get("open", 0) for d in data], dtype=np.float32),
            "highs": np.array([d.get("high", 0) for d in data], dtype=np.float32),
            "lows": np.array([d.get("low", 0) for d in data], dtype=np.float32),
            "closes": np.array([d.get("close", 0) for d in data], dtype=np.float32),
            "volumes": np.array([d.get("volume", 0) for d in data], dtype=np.float32),
        }

    def calculate_features(self, ohlcv: dict[str, np.ndarray], timeframe: str) -> np.ndarray:
        """
        Berechnet alle 25 Features aus OHLCV-Daten.

        Returns:
            Feature-Array der Shape (len, 25)
        """
        closes = ohlcv["closes"]
        highs = ohlcv["highs"]
        lows = ohlcv["lows"]
        opens = ohlcv["opens"]
        volumes = ohlcv["volumes"]
        n = len(closes)

        # Returns und Volatilitaet
        log_returns = np.zeros(n)
        log_returns[1:] = np.log(closes[1:] / (closes[:-1] + 1e-8))

        volatility = np.zeros(n)
        for i in range(20, n):
            volatility[i] = np.std(log_returns[i-20:i])
        volatility[:20] = volatility[20] if n > 20 else 0.01

        # Moving Averages
        sma_20 = self._rolling_mean(closes, 20)
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        macd = ema_12 - ema_26
        macd_signal = self._ema(macd, 9)

        # RSI
        rsi = self._rsi(closes, 14)

        # Stochastic
        stoch_k, stoch_d = self._stochastic(highs, lows, closes, 14, 3)

        # CCI
        cci = self._cci(highs, lows, closes, 14)

        # ATR
        atr = self._atr(highs, lows, closes, 14)

        # Bollinger Bands
        bb_middle = sma_20
        bb_std = np.zeros(n)
        for i in range(20, n):
            bb_std[i] = np.std(closes[i-20:i])
        bb_std[:20] = bb_std[20] if n > 20 else 0.01
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std

        # Volume Indicators
        obv = np.cumsum(np.where(log_returns > 0, volumes, -volumes))
        obv_norm = (obv - np.mean(obv)) / (np.std(obv) + 1e-8)

        clv = np.where(
            highs != lows,
            ((closes - lows) - (highs - closes)) / (highs - lows + 1e-8),
            0
        )
        ad_line = np.cumsum(clv * volumes)
        ad_norm = (ad_line - np.mean(ad_line)) / (np.std(ad_line) + 1e-8)

        # Position Features
        price_vs_sma = (closes - sma_20) / (sma_20 + 1e-8)
        bb_range = bb_upper - bb_lower
        bb_position = np.where(bb_range > 0, (closes - bb_lower) / bb_range, 0.5)

        # Timeframe Encoding
        tf_encoding = {
            "M1": 0.0, "M5": 0.1, "M15": 0.2, "M30": 0.3,
            "H1": 0.4, "H4": 0.5, "D1": 0.6, "W1": 0.8, "MN": 1.0
        }
        tf_code = np.full(n, tf_encoding.get(timeframe.upper(), 0.4))

        # Kombiniere Features
        features = np.column_stack([
            self._normalize(opens),
            self._normalize(highs),
            self._normalize(lows),
            self._normalize(closes),
            self._normalize(volumes),
            log_returns,
            volatility,
            self._normalize(sma_20),
            self._normalize(ema_12),
            self._normalize(ema_26),
            macd / (closes + 1e-8),  # Normalisiere MACD
            macd_signal / (closes + 1e-8),
            rsi / 100,
            stoch_k / 100,
            stoch_d / 100,
            cci / 200,
            atr / (closes + 1e-8),  # Normalisiere ATR
            self._normalize(bb_upper),
            self._normalize(bb_middle),
            self._normalize(bb_lower),
            obv_norm,
            ad_norm,
            price_vs_sma,
            bb_position,
            tf_code
        ])

        return features.astype(np.float32)

    # =========================================================================
    # Label Generation
    # =========================================================================

    def create_price_labels(
        self,
        closes: np.ndarray,
        horizons: list[int]
    ) -> np.ndarray:
        """
        Erstellt Preis-Labels (Future Returns).

        Args:
            closes: Close-Preise
            horizons: Forecast-Horizonte in Bars

        Returns:
            Array der Shape (len - max_horizon, len(horizons))
        """
        max_horizon = max(horizons)
        n = len(closes) - max_horizon

        labels = np.zeros((n, len(horizons)), dtype=np.float32)

        for i in range(n):
            for j, h in enumerate(horizons):
                # Prozentuale Aenderung
                labels[i, j] = (closes[i + h] - closes[i]) / (closes[i] + 1e-8)

        return labels

    def create_pattern_labels(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int = 50
    ) -> np.ndarray:
        """
        Erstellt Pattern-Labels (Multi-Label).

        Vereinfachte Erkennung basierend auf Swing-Punkten und Preis-Struktur.

        Returns:
            Array der Shape (len, 16) mit binaeren Labels
        """
        n = len(closes)
        labels = np.zeros((n, len(PATTERN_TYPES)), dtype=np.float32)

        # Berechne Swing-Punkte
        swing_highs = self._find_swing_points(highs, window=5, is_high=True)
        swing_lows = self._find_swing_points(lows, window=5, is_high=False)

        for i in range(window, n):
            # Hole relevante Swings im Fenster
            recent_highs = [(idx, highs[idx]) for idx in swing_highs if i - window < idx <= i]
            recent_lows = [(idx, lows[idx]) for idx in swing_lows if i - window < idx <= i]

            # Einfache Pattern-Erkennung
            labels[i] = self._detect_patterns(
                recent_highs, recent_lows, closes[i-window:i+1], i
            )

        return labels

    def create_regime_labels(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Erstellt Regime-Labels (4-Class).

        Regimes:
        0: Bull Trend (steigende Preise, niedrige Volatilitaet)
        1: Bear Trend (fallende Preise, niedrige Volatilitaet)
        2: Sideways (seitwaerts, niedrige Volatilitaet)
        3: High Volatility (hohe Volatilitaet)

        Returns:
            Array der Shape (len,) mit Klassen-Indizes
        """
        n = len(closes)
        labels = np.zeros(n, dtype=np.int64)

        for i in range(window, n):
            window_closes = closes[i-window:i+1]
            window_highs = highs[i-window:i+1]
            window_lows = lows[i-window:i+1]

            # Berechne Metriken
            returns = (window_closes[-1] - window_closes[0]) / (window_closes[0] + 1e-8)
            volatility = np.std(np.diff(np.log(window_closes + 1e-8)))
            range_pct = (np.max(window_highs) - np.min(window_lows)) / (window_closes[0] + 1e-8)

            # Schwellenwerte
            vol_threshold = 0.02  # 2% Volatilitaet
            trend_threshold = 0.03  # 3% Bewegung

            # Klassifizierung
            if volatility > vol_threshold or range_pct > 0.15:
                labels[i] = 3  # High Volatility
            elif returns > trend_threshold:
                labels[i] = 0  # Bull Trend
            elif returns < -trend_threshold:
                labels[i] = 1  # Bear Trend
            else:
                labels[i] = 2  # Sideways

        # Fuehre erste Labels
        labels[:window] = labels[window]

        return labels

    # =========================================================================
    # Dataset Creation
    # =========================================================================

    async def create_training_dataset(
        self,
        symbol: str,
        timeframe: str = "H1",
        days: int = 365
    ) -> Optional[dict]:
        """
        Erstellt einen kompletten Training-Datensatz.

        Returns:
            Dictionary mit 'features', 'price_labels', 'pattern_labels', 'regime_labels'
            oder None bei Fehler
        """
        # Hole Daten
        data = await self.fetch_training_data(symbol, timeframe, days)
        if not data or len(data) < 200:
            logger.warning(f"Not enough data for {symbol} {timeframe}: got {len(data)}")
            return None

        # Extrahiere OHLCV
        ohlcv = self.extract_ohlcv_arrays(data)

        # Berechne Features
        features = self.calculate_features(ohlcv, timeframe)

        # Erstelle Labels
        horizons = self.get_forecast_horizons(timeframe)
        max_horizon = max(horizons)

        price_labels = self.create_price_labels(ohlcv["closes"], horizons)
        pattern_labels = self.create_pattern_labels(
            ohlcv["highs"], ohlcv["lows"], ohlcv["closes"]
        )
        regime_labels = self.create_regime_labels(
            ohlcv["closes"], ohlcv["highs"], ohlcv["lows"]
        )

        # Schneide auf gleiche Laenge
        n = len(price_labels)
        seq_len = self.get_sequence_length(timeframe)

        # Erstelle Sequenzen
        num_samples = n - seq_len
        if num_samples < 10:
            logger.warning(f"Not enough samples for {symbol}: {num_samples}")
            return None

        X = np.zeros((num_samples, seq_len, features.shape[1]), dtype=np.float32)
        y_price = np.zeros((num_samples, len(horizons)), dtype=np.float32)
        y_pattern = np.zeros((num_samples, len(PATTERN_TYPES)), dtype=np.float32)
        y_regime = np.zeros(num_samples, dtype=np.int64)

        for i in range(num_samples):
            X[i] = features[i:i+seq_len]
            y_price[i] = price_labels[i + seq_len - 1]
            y_pattern[i] = pattern_labels[i + seq_len - 1]
            y_regime[i] = regime_labels[i + seq_len - 1]

        return {
            "features": X,
            "price_labels": y_price,
            "pattern_labels": y_pattern,
            "regime_labels": y_regime,
            "symbol": symbol,
            "timeframe": timeframe,
            "num_samples": num_samples
        }

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Berechnet Rolling Mean."""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.mean(data[start:i+1])
        return result

    def _ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """Berechnet EMA."""
        result = np.zeros_like(data)
        mult = 2 / (window + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = data[i] * mult + result[i-1] * (1 - mult)
        return result

    def _rsi(self, closes: np.ndarray, window: int = 14) -> np.ndarray:
        """Berechnet RSI."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))

        if len(gains) >= window:
            avg_gain[window] = np.mean(gains[:window])
            avg_loss[window] = np.mean(losses[:window])

        for i in range(window + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i-1]) / window
            avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i-1]) / window

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        return 100 - (100 / (1 + rs))

    def _stochastic(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        k_window: int, d_window: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Berechnet Stochastic."""
        n = len(closes)
        k = np.zeros(n)

        for i in range(k_window - 1, n):
            h = np.max(highs[i-k_window+1:i+1])
            l = np.min(lows[i-k_window+1:i+1])
            if h != l:
                k[i] = 100 * (closes[i] - l) / (h - l)
            else:
                k[i] = 50

        d = self._rolling_mean(k, d_window)
        return k, d

    def _cci(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int
    ) -> np.ndarray:
        """Berechnet CCI."""
        tp = (highs + lows + closes) / 3
        sma_tp = self._rolling_mean(tp, window)

        mad = np.zeros(len(tp))
        for i in range(window, len(tp)):
            mad[i] = np.mean(np.abs(tp[i-window:i] - sma_tp[i]))

        return np.where(mad != 0, (tp - sma_tp) / (0.015 * mad), 0)

    def _atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int
    ) -> np.ndarray:
        """Berechnet ATR."""
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )

        return self._ema(tr, window)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-Score Normalisierung."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std

    def _find_swing_points(
        self, data: np.ndarray, window: int = 5, is_high: bool = True
    ) -> list[int]:
        """Findet Swing-Punkte."""
        swings = []
        for i in range(window, len(data) - window):
            if is_high:
                if data[i] == max(data[i-window:i+window+1]):
                    swings.append(i)
            else:
                if data[i] == min(data[i-window:i+window+1]):
                    swings.append(i)
        return swings

    def _detect_patterns(
        self,
        swing_highs: list[tuple[int, float]],
        swing_lows: list[tuple[int, float]],
        closes: np.ndarray,
        current_idx: int
    ) -> np.ndarray:
        """
        Erkennt Patterns basierend auf Swing-Punkten.

        Vereinfachte Implementierung - gibt binaere Labels zurueck.
        """
        labels = np.zeros(len(PATTERN_TYPES), dtype=np.float32)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return labels

        # Sortiere nach Index
        highs = sorted(swing_highs, key=lambda x: x[0])
        lows = sorted(swing_lows, key=lambda x: x[0])

        # Double Top/Bottom Detection
        if len(highs) >= 2:
            h1, h2 = highs[-2], highs[-1]
            if abs(h1[1] - h2[1]) / h1[1] < 0.02:  # Innerhalb 2%
                labels[PATTERN_TYPES.index("double_top")] = 1.0

        if len(lows) >= 2:
            l1, l2 = lows[-2], lows[-1]
            if abs(l1[1] - l2[1]) / l1[1] < 0.02:
                labels[PATTERN_TYPES.index("double_bottom")] = 1.0

        # Trend-basierte Patterns
        if len(closes) > 10:
            trend = (closes[-1] - closes[0]) / (closes[0] + 1e-8)

            # Ascending/Descending Triangle (vereinfacht)
            if len(highs) >= 2 and len(lows) >= 2:
                high_slope = (highs[-1][1] - highs[-2][1]) / (highs[-1][0] - highs[-2][0] + 1)
                low_slope = (lows[-1][1] - lows[-2][1]) / (lows[-1][0] - lows[-2][0] + 1)

                if abs(high_slope) < 0.001 and low_slope > 0:
                    labels[PATTERN_TYPES.index("ascending_triangle")] = 1.0
                elif abs(low_slope) < 0.001 and high_slope < 0:
                    labels[PATTERN_TYPES.index("descending_triangle")] = 1.0

            # Channel Detection
            if trend > 0.03:
                labels[PATTERN_TYPES.index("channel_up")] = 0.5
            elif trend < -0.03:
                labels[PATTERN_TYPES.index("channel_down")] = 0.5

        return labels

    async def close(self):
        """Schliesst den HTTP-Client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Singleton Instance
data_pipeline = DataPipeline()
