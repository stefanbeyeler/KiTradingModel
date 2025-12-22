"""Hidden Markov Model for Market Regime Detection."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
from loguru import logger

# Lazy imports
hmmlearn_hmm = None
joblib = None


def _load_hmmlearn():
    global hmmlearn_hmm, joblib
    if hmmlearn_hmm is None:
        try:
            from hmmlearn import hmm as h
            import joblib as jl
            hmmlearn_hmm = h
            joblib = jl
            return True
        except ImportError:
            logger.warning("hmmlearn not installed. HMM will use fallback.")
            return False
    return True


class MarketRegime(str, Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    regime: MarketRegime
    probability: float
    duration: int  # Candles in current regime
    transition_probs: Dict[MarketRegime, float]


class HMMRegimeModel:
    """
    Hidden Markov Model for Market Regime Detection.

    Uses Gaussian HMM with 4 hidden states:
    - Bull Trend: Positive returns, low volatility
    - Bear Trend: Negative returns, elevated volatility
    - Sideways: Low returns, low volatility
    - High Volatility: High volatility, undirected movement
    """

    REGIMES = [
        MarketRegime.BULL_TREND,
        MarketRegime.BEAR_TREND,
        MarketRegime.SIDEWAYS,
        MarketRegime.HIGH_VOLATILITY,
    ]

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the HMM model.

        Args:
            n_components: Number of hidden states
            covariance_type: Type of covariance ('full', 'diag', 'spherical')
            n_iter: Number of EM iterations
            random_state: Random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = None
        self._is_fitted = False
        self._regime_mapping: Dict[int, MarketRegime] = {}

    def _create_model(self):
        """Create the HMM model."""
        if not _load_hmmlearn():
            return None

        return hmmlearn_hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )

    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract features for HMM.

        Features:
        - Log returns
        - Rolling volatility (20 periods)
        - Trend strength (SMA deviation)
        """
        # Log returns
        returns = np.diff(np.log(prices))

        # Rolling volatility
        volatility = np.array([
            np.std(returns[max(0, i-20):i+1])
            for i in range(len(returns))
        ])

        # Trend strength (deviation from SMA)
        sma_period = 20
        sma = np.convolve(prices, np.ones(sma_period)/sma_period, mode='valid')
        sma_padded = np.concatenate([np.full(sma_period - 1, sma[0]), sma])
        trend_strength = (prices - sma_padded) / (sma_padded + 1e-8)

        # Combine features (align arrays)
        features = np.column_stack([
            returns,
            volatility,
            trend_strength[1:]  # Align with returns
        ])

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def _map_states_to_regimes(self, features: np.ndarray):
        """
        Map HMM states to market regimes based on state characteristics.
        """
        if self._model is None:
            return

        means = self._model.means_

        # Analyze state characteristics
        state_chars = []
        for i in range(self.n_components):
            return_mean = means[i, 0] if means.shape[1] > 0 else 0
            vol_mean = means[i, 1] if means.shape[1] > 1 else 0

            state_chars.append({
                'state': i,
                'return': return_mean,
                'volatility': vol_mean
            })

        # Sort by return and volatility
        by_return = sorted(state_chars, key=lambda x: x['return'])
        by_vol = sorted(state_chars, key=lambda x: x['volatility'])

        # Assign regimes (heuristic)
        assigned = set()

        # Highest volatility = HIGH_VOLATILITY
        high_vol_state = by_vol[-1]['state']
        self._regime_mapping[high_vol_state] = MarketRegime.HIGH_VOLATILITY
        assigned.add(high_vol_state)

        # Highest returns (not high vol) = BULL_TREND
        for s in reversed(by_return):
            if s['state'] not in assigned:
                self._regime_mapping[s['state']] = MarketRegime.BULL_TREND
                assigned.add(s['state'])
                break

        # Lowest returns (not assigned) = BEAR_TREND
        for s in by_return:
            if s['state'] not in assigned:
                self._regime_mapping[s['state']] = MarketRegime.BEAR_TREND
                assigned.add(s['state'])
                break

        # Remaining = SIDEWAYS
        for i in range(self.n_components):
            if i not in assigned:
                self._regime_mapping[i] = MarketRegime.SIDEWAYS

    def fit(self, prices: np.ndarray) -> "HMMRegimeModel":
        """
        Train the HMM on historical price data.

        Args:
            prices: Array of closing prices

        Returns:
            self
        """
        if len(prices) < 100:
            logger.warning("Insufficient data for HMM training")
            return self

        self._model = self._create_model()
        if self._model is None:
            self._is_fitted = False
            return self

        try:
            features = self._extract_features(prices)
            self._model.fit(features)
            self._map_states_to_regimes(features)
            self._is_fitted = True
            logger.info("HMM model fitted successfully")
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            self._is_fitted = False

        return self

    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def predict_regime(self, prices: np.ndarray) -> RegimeState:
        """
        Predict the current market regime.

        Args:
            prices: Array of recent prices

        Returns:
            RegimeState with current regime and probabilities
        """
        if not self._is_fitted:
            # Fallback: simple trend detection
            return self._fallback_predict(prices)

        features = self._extract_features(prices)

        try:
            # Viterbi for most likely state sequence
            log_prob, states = self._model.decode(features, algorithm="viterbi")
            current_state = states[-1]
            current_regime = self._regime_mapping.get(current_state, MarketRegime.SIDEWAYS)

            # State probabilities
            posteriors = self._model.predict_proba(features)
            current_probs = posteriors[-1]

            # Calculate duration
            duration = 1
            for i in range(len(states) - 2, -1, -1):
                if states[i] == current_state:
                    duration += 1
                else:
                    break

            # Transition probabilities
            trans_probs = {}
            for regime in self.REGIMES:
                for state, mapped_regime in self._regime_mapping.items():
                    if mapped_regime == regime:
                        trans_probs[regime] = float(
                            self._model.transmat_[current_state, state]
                        )
                        break

            return RegimeState(
                regime=current_regime,
                probability=float(current_probs[current_state]),
                duration=duration,
                transition_probs=trans_probs
            )

        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return self._fallback_predict(prices)

    def _fallback_predict(self, prices: np.ndarray) -> RegimeState:
        """Fallback prediction using simple rules."""
        if len(prices) < 20:
            return RegimeState(
                regime=MarketRegime.SIDEWAYS,
                probability=0.5,
                duration=1,
                transition_probs={r: 0.25 for r in self.REGIMES}
            )

        # Simple metrics
        returns = np.diff(np.log(prices[-50:])) if len(prices) >= 50 else np.diff(np.log(prices))
        mean_return = np.mean(returns)
        volatility = np.std(returns)

        # High volatility threshold
        if volatility > np.percentile(np.abs(returns), 80):
            regime = MarketRegime.HIGH_VOLATILITY
            prob = 0.7
        elif mean_return > 0.001:
            regime = MarketRegime.BULL_TREND
            prob = 0.6
        elif mean_return < -0.001:
            regime = MarketRegime.BEAR_TREND
            prob = 0.6
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.5

        return RegimeState(
            regime=regime,
            probability=prob,
            duration=1,
            transition_probs={r: 0.25 for r in self.REGIMES}
        )

    def get_regime_history(
        self,
        prices: np.ndarray
    ) -> List[Tuple[MarketRegime, float]]:
        """
        Get regime history for all time points.

        Args:
            prices: Price array

        Returns:
            List of (regime, probability) tuples
        """
        if not self._is_fitted:
            # Fallback: all sideways
            return [(MarketRegime.SIDEWAYS, 0.5)] * (len(prices) - 1)

        features = self._extract_features(prices)

        try:
            _, states = self._model.decode(features, algorithm="viterbi")
            posteriors = self._model.predict_proba(features)

            history = []
            for i, state in enumerate(states):
                regime = self._regime_mapping.get(state, MarketRegime.SIDEWAYS)
                prob = float(posteriors[i, state])
                history.append((regime, prob))

            return history

        except Exception as e:
            logger.error(f"History generation failed: {e}")
            return [(MarketRegime.SIDEWAYS, 0.5)] * (len(prices) - 1)

    def save(self, path: str):
        """Save model to disk."""
        if not _load_hmmlearn():
            return

        data = {
            'model': self._model,
            'regime_mapping': self._regime_mapping,
            'is_fitted': self._is_fitted,
            'n_components': self.n_components
        }
        joblib.dump(data, path)
        logger.info(f"HMM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HMMRegimeModel":
        """Load model from disk."""
        if not _load_hmmlearn():
            return cls()

        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return cls()

        try:
            data = joblib.load(path)
            instance = cls(n_components=data.get('n_components', 4))
            instance._model = data.get('model')
            instance._regime_mapping = data.get('regime_mapping', {})
            instance._is_fitted = data.get('is_fitted', False)
            logger.info(f"HMM model loaded from {path}")
            return instance
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return cls()
