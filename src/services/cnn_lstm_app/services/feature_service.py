"""
Feature Engineering Service für CNN-LSTM.

Bereitet Features aus OHLCV-Daten und technischen Indikatoren vor.
Integriert mit dem DataGateway Service fuer Datenabruf.
"""

import os
from typing import Optional

import numpy as np
from loguru import logger

from src.config.microservices import microservices_config

# =============================================================================
# Configuration
# =============================================================================

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", microservices_config.data_service_url)

# Feature Definitionen (25 Features)
FEATURE_NAMES = [
    # OHLCV (5)
    "open", "high", "low", "close", "volume",
    # Returns (2)
    "log_return", "volatility_20",
    # Trend (5)
    "sma_20", "ema_12", "ema_26", "macd", "macd_signal",
    # Momentum (4)
    "rsi_14", "stoch_k", "stoch_d", "cci_14",
    # Volatility (4)
    "atr_14", "bb_upper", "bb_middle", "bb_lower",
    # Volume (2)
    "obv_normalized", "ad_line",
    # Position (2)
    "price_vs_sma", "bb_position",
    # Meta (1)
    "timeframe_encoding"
]

# Timeframe zu numerischer Kodierung
TIMEFRAME_ENCODING = {
    "M1": 0.0,
    "M5": 0.1,
    "M15": 0.2,
    "M30": 0.3,
    "H1": 0.4,
    "H4": 0.5,
    "D1": 0.6,
    "W1": 0.8,
    "MN": 1.0
}

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


class FeatureService:
    """
    Service fuer Feature-Engineering.

    Bereitet OHLCV-Daten und technische Indikatoren als Input fuer CNN-LSTM vor.
    """

    def __init__(self):
        self._http_client = None

    async def _get_http_client(self):
        """Lazy initialization des HTTP-Clients."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def get_sequence_length(self, timeframe: str) -> int:
        """Gibt die Sequenzlaenge fuer einen Timeframe zurueck."""
        return SEQUENCE_LENGTHS.get(timeframe.upper(), 168)

    def get_timeframe_encoding(self, timeframe: str) -> float:
        """Gibt die numerische Kodierung fuer einen Timeframe zurueck."""
        return TIMEFRAME_ENCODING.get(timeframe.upper(), 0.4)

    async def fetch_ohlcv_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        limit: int = 500
    ) -> list[dict]:
        """
        Holt OHLCV-Daten vom Data Service.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe (H1, D1, etc.)
            limit: Anzahl der Datenpunkte

        Returns:
            Liste von OHLCV-Dictionaries
        """
        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{DATA_SERVICE_URL}/api/v1/db/ohlcv/{symbol}",
                params={"timeframe": timeframe, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return []

    async def fetch_indicators(
        self,
        symbol: str,
        timeframe: str = "H1",
        limit: int = 500
    ) -> dict:
        """
        Holt technische Indikatoren vom Data Service.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            limit: Anzahl der Datenpunkte

        Returns:
            Dictionary mit Indikator-Arrays
        """
        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{DATA_SERVICE_URL}/api/v1/indicators/all/{symbol}",
                params={"timeframe": timeframe, "limit": limit}
            )
            response.raise_for_status()
            return response.json().get("indicators", {})
        except Exception as e:
            logger.warning(f"Error fetching indicators for {symbol}: {e}")
            return {}

    def calculate_returns(self, closes: np.ndarray) -> np.ndarray:
        """Berechnet logarithmische Returns."""
        returns = np.zeros_like(closes)
        # Vermeide Division by Zero und Log von 0
        prev_closes = closes[:-1]
        curr_closes = closes[1:]
        # Nur berechnen wo beide Werte > 0
        valid_mask = (prev_closes > 0) & (curr_closes > 0)
        returns[1:] = np.where(valid_mask, np.log(curr_closes / prev_closes), 0.0)
        returns[0] = 0  # Erster Wert ist 0
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    def calculate_volatility(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Berechnet Rolling Volatility."""
        volatility = np.zeros_like(returns)
        for i in range(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i])
        # Fuehre erste Werte mit erstem berechneten Wert
        if len(returns) > window:
            volatility[:window] = volatility[window]
        return np.nan_to_num(volatility, nan=0.0)

    def calculate_sma(self, data: np.ndarray, window: int) -> np.ndarray:
        """Berechnet Simple Moving Average."""
        sma = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            sma[i] = np.mean(data[start:i+1])
        return sma

    def calculate_ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """Berechnet Exponential Moving Average."""
        ema = np.zeros_like(data)
        multiplier = 2 / (window + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema

    def calculate_rsi(self, closes: np.ndarray, window: int = 14) -> np.ndarray:
        """Berechnet Relative Strength Index."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))

        # Erste Durchschnitte
        if len(gains) >= window:
            avg_gain[window] = np.mean(gains[:window])
            avg_loss[window] = np.mean(losses[:window])

        # Smoothed averages
        for i in range(window + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i-1]) / window
            avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i-1]) / window

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))

        return np.nan_to_num(rsi, nan=50.0)

    def calculate_macd(
        self,
        closes: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Berechnet MACD und Signal Line."""
        ema_fast = self.calculate_ema(closes, fast)
        ema_slow = self.calculate_ema(closes, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        return macd, macd_signal

    def calculate_bollinger_bands(
        self,
        closes: np.ndarray,
        window: int = 20,
        num_std: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Berechnet Bollinger Bands."""
        middle = self.calculate_sma(closes, window)
        std = np.zeros_like(closes)
        for i in range(window, len(closes)):
            std[i] = np.std(closes[i-window:i])
        std[:window] = std[window] if len(closes) > window else 0

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return upper, middle, lower

    def calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int = 14
    ) -> np.ndarray:
        """Berechnet Average True Range."""
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        atr = self.calculate_ema(tr, window)
        return atr

    def calculate_stochastic(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        k_window: int = 14,
        d_window: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Berechnet Stochastic Oscillator (K und D)."""
        stoch_k = np.zeros(len(closes))

        for i in range(k_window - 1, len(closes)):
            highest = np.max(highs[i-k_window+1:i+1])
            lowest = np.min(lows[i-k_window+1:i+1])
            if highest != lowest:
                stoch_k[i] = 100 * (closes[i] - lowest) / (highest - lowest)
            else:
                stoch_k[i] = 50

        stoch_d = self.calculate_sma(stoch_k, d_window)
        return stoch_k, stoch_d

    def calculate_cci(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int = 14
    ) -> np.ndarray:
        """Berechnet Commodity Channel Index."""
        tp = (highs + lows + closes) / 3
        sma_tp = self.calculate_sma(tp, window)

        mad = np.zeros(len(tp))
        for i in range(window, len(tp)):
            mad[i] = np.mean(np.abs(tp[i-window:i] - sma_tp[i]))

        cci = np.where(mad != 0, (tp - sma_tp) / (0.015 * mad), 0)
        return np.nan_to_num(cci, nan=0.0)

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalisiert Features mit Z-Score Normalisierung.

        Args:
            features: Array der Shape (seq_len, num_features)

        Returns:
            Normalisierte Features
        """
        # Ersetze NaN/Inf vor Normalisierung
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Vermeide Division durch 0
        normalized = (features - mean) / std
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    async def prepare_features(
        self,
        symbol: str,
        timeframe: str = "H1",
        sequence_length: Optional[int] = None,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Bereitet alle Features fuer ein Symbol vor.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            sequence_length: Sequenzlaenge (optional, nutzt Default)
            normalize: Features normalisieren

        Returns:
            Feature-Array der Shape (sequence_length, 25) oder None bei Fehler
        """
        result = await self.prepare_features_with_price(symbol, timeframe, sequence_length, normalize)
        if result is None:
            return None
        return result[0]

    async def prepare_features_with_price(
        self,
        symbol: str,
        timeframe: str = "H1",
        sequence_length: Optional[int] = None,
        normalize: bool = True
    ) -> Optional[tuple[np.ndarray, float]]:
        """
        Bereitet alle Features fuer ein Symbol vor und gibt auch den aktuellen Preis zurueck.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe
            sequence_length: Sequenzlaenge (optional, nutzt Default)
            normalize: Features normalisieren

        Returns:
            Tuple von (Feature-Array, aktueller Preis) oder None bei Fehler
        """
        if sequence_length is None:
            sequence_length = self.get_sequence_length(timeframe)

        # Hole mehr Daten als benoetigt fuer Indikator-Berechnung
        extra_buffer = 50
        limit = sequence_length + extra_buffer

        # Hole OHLCV-Daten
        ohlcv = await self.fetch_ohlcv_data(symbol, timeframe, limit)
        if not ohlcv or len(ohlcv) < sequence_length:
            logger.warning(f"Not enough OHLCV data for {symbol}: got {len(ohlcv)}, need {sequence_length}")
            return None

        # Extrahiere Arrays
        opens = np.array([d.get("open", 0) for d in ohlcv], dtype=np.float32)
        highs = np.array([d.get("high", 0) for d in ohlcv], dtype=np.float32)
        lows = np.array([d.get("low", 0) for d in ohlcv], dtype=np.float32)
        closes = np.array([d.get("close", 0) for d in ohlcv], dtype=np.float32)
        volumes = np.array([d.get("volume", 0) for d in ohlcv], dtype=np.float32)

        # Berechne Features
        log_returns = self.calculate_returns(closes)
        volatility = self.calculate_volatility(log_returns, 20)
        sma_20 = self.calculate_sma(closes, 20)
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        macd, macd_signal = self.calculate_macd(closes)
        rsi = self.calculate_rsi(closes)
        stoch_k, stoch_d = self.calculate_stochastic(highs, lows, closes)
        cci = self.calculate_cci(highs, lows, closes)
        atr = self.calculate_atr(highs, lows, closes)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)

        # OBV (simplified)
        obv = np.cumsum(np.where(log_returns > 0, volumes, -volumes))
        obv_normalized = (obv - np.mean(obv)) / (np.std(obv) + 1e-8)

        # A/D Line (simplified)
        clv = np.where(
            highs != lows,
            ((closes - lows) - (highs - closes)) / (highs - lows),
            0
        )
        ad_line = np.cumsum(clv * volumes)
        ad_line = (ad_line - np.mean(ad_line)) / (np.std(ad_line) + 1e-8)

        # Position Features
        price_vs_sma = (closes - sma_20) / (sma_20 + 1e-8)
        bb_range = bb_upper - bb_lower
        bb_position = np.where(
            bb_range != 0,
            (closes - bb_lower) / bb_range,
            0.5
        )

        # Timeframe Encoding
        tf_encoding = np.full(len(closes), self.get_timeframe_encoding(timeframe))

        # Kombiniere alle Features
        features = np.column_stack([
            opens, highs, lows, closes, volumes,
            log_returns, volatility,
            sma_20, ema_12, ema_26, macd, macd_signal,
            rsi / 100,  # Normalisiere RSI zu [0, 1]
            stoch_k / 100, stoch_d / 100,  # Normalisiere Stochastic
            cci / 200,  # Skaliere CCI
            atr,
            bb_upper, bb_middle, bb_lower,
            obv_normalized, ad_line,
            price_vs_sma, bb_position,
            tf_encoding
        ])

        # Speichere aktuellen Preis VOR Normalisierung
        current_price = float(closes[-1])

        # Schneide auf Sequenzlaenge
        features = features[-sequence_length:]

        # Normalisiere wenn gewuenscht
        if normalize:
            # Normalisiere nur OHLCV und preis-basierte Features
            # Behalte bereits normalisierte Features
            features[:, :5] = self.normalize_features(features[:, :5])  # OHLCV
            features[:, 7:12] = self.normalize_features(features[:, 7:12])  # MAs, MACD
            features[:, 16:20] = self.normalize_features(features[:, 16:20])  # ATR, BBands

        # Finales NaN/Inf Cleanup
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features.astype(np.float32), current_price

    async def prepare_batch_features(
        self,
        symbols: list[str],
        timeframe: str = "H1",
        sequence_length: Optional[int] = None
    ) -> tuple[np.ndarray, list[str]]:
        """
        Bereitet Features fuer mehrere Symbole vor.

        Args:
            symbols: Liste von Symbolen
            timeframe: Timeframe
            sequence_length: Sequenzlaenge

        Returns:
            Tuple von (features_array, successful_symbols)
        """
        features_list = []
        successful_symbols = []

        for symbol in symbols:
            features = await self.prepare_features(symbol, timeframe, sequence_length)
            if features is not None:
                features_list.append(features)
                successful_symbols.append(symbol)

        if not features_list:
            return np.array([]), []

        return np.stack(features_list), successful_symbols

    async def close(self):
        """Schliesst den HTTP-Client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Singleton Instance
feature_service = FeatureService()
