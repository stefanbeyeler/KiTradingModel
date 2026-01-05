"""Signal scoring service."""

from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import os
from loguru import logger

from ..models.hmm_regime_model import MarketRegime
from ..models.lightgbm_scorer import LightGBMSignalScorer, SignalType, SignalScore
from ..models.schemas import SignalScoringResponse
from .regime_detection_service import regime_detection_service


class SignalScoringService:
    """
    Service for scoring trading signals using LightGBM.

    Evaluates signals based on:
    - Current market regime
    - Technical indicators
    - Price action
    - Volume analysis
    """

    MODEL_DIR = "data/models/hmm"

    def __init__(self):
        """Initialize the service."""
        self._scorer = LightGBMSignalScorer()

        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # Try to load saved model
        scorer_path = os.path.join(self.MODEL_DIR, "signal_scorer.pkl")
        if os.path.exists(scorer_path):
            self._scorer = LightGBMSignalScorer.load(scorer_path)

    async def score_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        timeframe: str = "1h",
        entry_price: Optional[float] = None
    ) -> SignalScoringResponse:
        """
        Score a trading signal.

        Args:
            symbol: Trading symbol
            signal_type: Signal type (long/short)
            timeframe: Timeframe
            entry_price: Optional proposed entry price

        Returns:
            SignalScoringResponse
        """
        try:
            from src.services.data_gateway_service import data_gateway

            # Fetch OHLCV data
            data = await data_gateway.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=100
            )

            if not data or len(data) < 50:
                return self._error_response(symbol, signal_type, "Insufficient data")

            ohlcv = np.array([
                [d['open'], d['high'], d['low'], d['close'], d.get('volume', 0)]
                for d in data
            ], dtype=np.float64)

            # Get regime info
            closes = ohlcv[:, 3]
            regime_model = regime_detection_service.get_model(symbol)

            if not regime_model.is_fitted():
                regime_model.fit(closes)

            regime_state = regime_model.predict_regime(closes)

            # Convert transition probs to string keys
            regime_probs = {
                k.value: v
                for k, v in regime_state.transition_probs.items()
            }
            # Add current regime probability
            regime_probs[regime_state.regime.value] = regime_state.probability

            # Score signal
            score_result = self._scorer.score_signal(
                ohlcv=ohlcv,
                regime_probs=regime_probs,
                regime_duration=regime_state.duration,
                signal_type=signal_type
            )

            # Generate recommendation
            recommendation = self._generate_recommendation(
                score_result, regime_state.regime, signal_type
            )

            # Risk assessment
            risk_assessment = self._assess_risk(
                ohlcv, signal_type, entry_price, score_result
            )

            return SignalScoringResponse(
                symbol=symbol,
                signal_type=signal_type,
                score=score_result.score,
                confidence=score_result.confidence,
                regime_alignment=score_result.regime_alignment,
                current_regime=regime_state.regime,
                recommendation=recommendation,
                feature_importance=score_result.feature_importance,
                risk_assessment=risk_assessment
            )

        except Exception as e:
            logger.error(f"Signal scoring error for {symbol}: {e}")
            return self._error_response(symbol, signal_type, str(e))

    async def score_batch(
        self,
        signals: List[Dict]
    ) -> List[SignalScoringResponse]:
        """
        Score multiple signals.

        Args:
            signals: List of signal dicts with symbol, signal_type, timeframe

        Returns:
            List of SignalScoringResponse
        """
        results = []

        for signal in signals:
            result = await self.score_signal(
                symbol=signal.get("symbol", ""),
                signal_type=SignalType(signal.get("signal_type", "neutral")),
                timeframe=signal.get("timeframe", "1h"),
                entry_price=signal.get("entry_price")
            )
            results.append(result)

        return results

    def _generate_recommendation(
        self,
        score: SignalScore,
        regime: MarketRegime,
        signal_type: SignalType
    ) -> str:
        """Generate trading recommendation based on score."""
        if score.score >= 80:
            strength = "Strong"
            action = "consider taking"
        elif score.score >= 60:
            strength = "Moderate"
            action = "may consider"
        elif score.score >= 40:
            strength = "Weak"
            action = "use caution with"
        else:
            strength = "Poor"
            action = "avoid"

        alignment_text = ""
        if score.regime_alignment == "aligned":
            alignment_text = " Signal aligns well with current regime."
        elif score.regime_alignment == "contrary":
            alignment_text = " Warning: Signal contradicts current regime."

        direction = "long" if signal_type == SignalType.LONG else "short"

        return f"{strength} signal ({score.score:.0f}/100). You {action} this {direction} position.{alignment_text}"

    def _assess_risk(
        self,
        ohlcv: np.ndarray,
        signal_type: SignalType,
        entry_price: Optional[float],
        score: SignalScore
    ) -> Dict:
        """Assess risk for the signal."""
        closes = ohlcv[:, 3]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]

        current_price = closes[-1]
        entry = entry_price or current_price

        # ATR for stop loss calculation
        tr_values = []
        for i in range(1, min(15, len(highs))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]) if i < len(closes) else 0,
                abs(lows[-i] - closes[-i-1]) if i < len(closes) else 0
            )
            tr_values.append(tr)
        atr = np.mean(tr_values) if tr_values else current_price * 0.02

        # Suggested stop loss (2 ATR)
        if signal_type == SignalType.LONG:
            stop_loss = entry - 2 * atr
            take_profit = entry + 3 * atr
        else:
            stop_loss = entry + 2 * atr
            take_profit = entry - 3 * atr

        # Risk/reward ratio
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        # Risk level based on score and alignment
        if score.score >= 70 and score.regime_alignment == "aligned":
            risk_level = "low"
        elif score.score >= 50:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "risk_level": risk_level,
            "suggested_stop_loss": round(stop_loss, 4),
            "suggested_take_profit": round(take_profit, 4),
            "risk_reward_ratio": round(rr_ratio, 2),
            "atr_14": round(atr, 4),
            "entry_price": round(entry, 4)
        }

    def _error_response(
        self,
        symbol: str,
        signal_type: SignalType,
        error: str
    ) -> SignalScoringResponse:
        """Create error response."""
        return SignalScoringResponse(
            symbol=symbol,
            signal_type=signal_type,
            score=0.0,
            confidence=0.0,
            regime_alignment="unknown",
            current_regime=MarketRegime.SIDEWAYS,
            recommendation=f"Error: {error}",
            feature_importance={},
            risk_assessment={"error": error}
        )

    async def train_scorer(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        lookback_days: int = 365
    ) -> Dict:
        """
        Train the signal scorer.

        Note: This is a simplified training that generates synthetic labels
        based on future price movements.
        """
        try:
            from src.services.data_gateway_service import data_gateway

            all_features = []
            all_labels = []

            for symbol in symbols:
                logger.info(f"Processing {symbol} for scorer training")

                candles_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
                limit = lookback_days * candles_per_day.get(timeframe, 24)

                data = await data_gateway.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )

                if not data or len(data) < 200:
                    continue

                ohlcv = np.array([
                    [d['open'], d['high'], d['low'], d['close'], d.get('volume', 0)]
                    for d in data
                ], dtype=np.float64)

                closes = ohlcv[:, 3]

                # Fit regime model
                regime_model = regime_detection_service.get_model(symbol)
                if not regime_model.is_fitted():
                    regime_model.fit(closes)

                # Generate training samples
                samples, labels = self._generate_training_samples(
                    ohlcv, regime_model
                )

                all_features.extend(samples)
                all_labels.extend(labels)

            if not all_features:
                return {"status": "failed", "message": "No training data generated"}

            X = np.array(all_features)
            y = np.array(all_labels)

            # Split for validation
            n_samples = len(X)
            n_val = int(n_samples * 0.2)
            indices = np.random.permutation(n_samples)

            train_X = X[indices[n_val:]]
            train_y = y[indices[n_val:]]
            val_X = X[indices[:n_val]]
            val_y = y[indices[:n_val]]

            # Train scorer
            self._scorer.fit(train_X, train_y, eval_set=(val_X, val_y))

            # Save scorer
            scorer_path = os.path.join(self.MODEL_DIR, "signal_scorer.pkl")
            self._scorer.save(scorer_path)

            return {
                "status": "completed",
                "samples": len(X),
                "model_path": scorer_path
            }

        except Exception as e:
            logger.error(f"Scorer training failed: {e}")
            return {"status": "failed", "message": str(e)}

    def _generate_training_samples(
        self,
        ohlcv: np.ndarray,
        regime_model
    ) -> tuple[List[np.ndarray], List[float]]:
        """Generate training samples with synthetic labels."""
        closes = ohlcv[:, 3]
        features = []
        labels = []

        # Predict regime history
        regime_history = regime_model.get_regime_history(closes)

        for i in range(100, len(ohlcv) - 20):
            window = ohlcv[i-100:i]

            # Get regime info at this point
            if i - 1 < len(regime_history):
                regime, prob = regime_history[i - 1]
                regime_probs = {
                    regime.value: prob,
                    "bull_trend": 0.25,
                    "bear_trend": 0.25,
                    "sideways": 0.25,
                    "high_volatility": 0.25
                }
                regime_probs[regime.value] = prob
            else:
                regime_probs = {r.value: 0.25 for r in MarketRegime}

            # Calculate features
            feature_vec = self._scorer._calculate_features(
                window, regime_probs, 10
            )

            # Calculate label based on future returns
            future_return = (closes[i + 20] - closes[i]) / closes[i]

            # Score: 0-100 based on how profitable a long signal would be
            if future_return > 0.02:
                label = 80 + min(future_return * 500, 20)
            elif future_return > 0:
                label = 50 + future_return * 1500
            elif future_return > -0.02:
                label = 50 + future_return * 1500
            else:
                label = max(0, 20 + future_return * 500)

            features.append(feature_vec)
            labels.append(label)

        return features, labels


# Singleton instance
signal_scoring_service = SignalScoringService()
