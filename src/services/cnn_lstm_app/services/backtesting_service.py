"""
Backtesting Service - Vergleicht CNN-LSTM Predictions mit echten Marktdaten.

Ermoeglicht:
- Automatischen Vergleich von Predictions gegen reale Preisbewegungen
- Berechnung von Genauigkeitsmetriken pro Task
- Historische Performance-Analyse
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path

import httpx
from loguru import logger


DATA_DIR = os.getenv("DATA_DIR", "/app/data")


@dataclass
class BacktestResult:
    """Ergebnis eines Backtests fuer eine einzelne Prediction."""
    prediction_id: str
    symbol: str
    timeframe: str
    prediction_timestamp: str
    backtest_timestamp: str

    # Price Backtest
    price_predicted_direction: str
    price_actual_direction: str
    price_direction_correct: bool
    price_predicted_change_1h: Optional[float] = None
    price_actual_change_1h: Optional[float] = None
    price_error_1h: Optional[float] = None
    price_predicted_change_1d: Optional[float] = None
    price_actual_change_1d: Optional[float] = None
    price_error_1d: Optional[float] = None

    # Regime Backtest
    regime_predicted: str = ""
    regime_actual: Optional[str] = None  # Basierend auf Volatilitaet/Trend
    regime_correct: bool = False

    # Pattern Backtest (schwieriger zu validieren)
    patterns_predicted: list = field(default_factory=list)
    pattern_outcome: Optional[str] = None  # success, failure, inconclusive

    # Metriken
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestSummary:
    """Zusammenfassung aller Backtests."""
    total_predictions: int
    backtested: int
    pending: int

    # Price Metriken
    price_direction_accuracy: float
    price_mae_1h: float  # Mean Absolute Error
    price_mae_1d: float

    # Regime Metriken
    regime_accuracy: float

    # Pattern Metriken
    pattern_success_rate: float

    # Nach Symbol/Timeframe
    by_symbol: dict = field(default_factory=dict)
    by_timeframe: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BacktestingService:
    """
    Service fuer das Backtesting von CNN-LSTM Predictions.

    Vergleicht gespeicherte Predictions mit den tatsaechlichen Marktdaten
    um die Modell-Genauigkeit zu messen.
    """

    def __init__(self, results_file: str = None):
        if results_file is None:
            results_file = os.path.join(DATA_DIR, "cnn_lstm_backtest_results.json")
        self._results_file = Path(results_file)
        self._results: list[BacktestResult] = []

        self._data_service_url = os.getenv("DATA_SERVICE_URL", "http://trading-data:3001")

        self._load_results()
        logger.info(f"BacktestingService initialized - {len(self._results)} results loaded")

    def _load_results(self):
        """Lade Backtest-Ergebnisse aus Datei."""
        try:
            if self._results_file.exists():
                with open(self._results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._results = [BacktestResult(**r) for r in data]
        except Exception as e:
            logger.error(f"Failed to load backtest results: {e}")
            self._results = []

    def _save_results(self):
        """Speichere Backtest-Ergebnisse."""
        try:
            self._results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._results_file, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in self._results], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")

    def _get_timeframe_hours(self, timeframe: str) -> int:
        """Gibt die Stunden fuer einen Timeframe zurueck."""
        tf_hours = {
            "M1": 0, "M5": 0, "M15": 0, "M30": 1,
            "H1": 1, "H4": 4, "D1": 24, "W1": 168
        }
        return tf_hours.get(timeframe.upper(), 1)

    async def _fetch_price_at_time(
        self,
        symbol: str,
        timeframe: str,
        target_time: datetime
    ) -> Optional[float]:
        """Hole den Preis zu einem bestimmten Zeitpunkt."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Hole OHLCV-Daten um den Zeitpunkt herum
                response = await client.get(
                    f"{self._data_service_url}/api/v1/db/ohlcv/{symbol}",
                    params={
                        "timeframe": timeframe,
                        "limit": 10,
                        "end_time": target_time.isoformat()
                    }
                )

                if response.status_code != 200:
                    return None

                data = response.json().get("data", [])
                if not data:
                    return None

                # Finde die naechste Kerze zum Zeitpunkt
                for candle in reversed(data):
                    return float(candle.get("close", 0))

                return None

        except Exception as e:
            logger.warning(f"Error fetching price: {e}")
            return None

    def _determine_actual_direction(
        self,
        price_before: float,
        price_after: float,
        threshold: float = 0.001
    ) -> str:
        """Bestimmt die tatsaechliche Richtung basierend auf Preisaenderung."""
        if price_before == 0:
            return "neutral"

        change = (price_after - price_before) / price_before

        if change > threshold:
            return "bullish"
        elif change < -threshold:
            return "bearish"
        else:
            return "neutral"

    def _calculate_actual_regime(
        self,
        ohlcv_data: list,
        lookback: int = 20
    ) -> str:
        """
        Berechnet das tatsaechliche Regime basierend auf OHLCV-Daten.

        Einfache Heuristik basierend auf:
        - Trend-Richtung (Reihe von hoeheren/tieferen Schlusskursen)
        - Volatilitaet (ATR im Vergleich zum Durchschnitt)
        """
        if not ohlcv_data or len(ohlcv_data) < lookback:
            return "sideways"

        recent = ohlcv_data[-lookback:]

        # Berechne Trend
        closes = [float(c.get("close", 0)) for c in recent]
        first_half_avg = sum(closes[:lookback // 2]) / (lookback // 2)
        second_half_avg = sum(closes[lookback // 2:]) / (lookback // 2)

        trend_change = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg else 0

        # Berechne Volatilitaet (vereinfachter ATR)
        ranges = []
        for c in recent:
            h = float(c.get("high", 0))
            l = float(c.get("low", 0))
            if h > 0:
                ranges.append((h - l) / h)

        avg_range = sum(ranges) / len(ranges) if ranges else 0

        # Klassifizierung
        if avg_range > 0.03:  # Hohe Volatilitaet (> 3%)
            return "high_volatility"
        elif trend_change > 0.02:  # Starker Aufwaertstrend (> 2%)
            return "bull_trend"
        elif trend_change < -0.02:  # Starker Abwaertstrend (< -2%)
            return "bear_trend"
        else:
            return "sideways"

    async def backtest_prediction(
        self,
        prediction: dict
    ) -> Optional[BacktestResult]:
        """
        Fuehrt Backtest fuer eine einzelne Prediction durch.

        Args:
            prediction: Prediction-Eintrag aus der History

        Returns:
            BacktestResult oder None wenn nicht moeglich
        """
        try:
            symbol = prediction.get("symbol", "")
            timeframe = prediction.get("timeframe", "H1")
            prediction_timestamp = prediction.get("timestamp", "")

            # Parse Timestamp
            pred_time = datetime.fromisoformat(prediction_timestamp.replace('Z', '+00:00'))

            # Pruefe ob genug Zeit vergangen ist (mind. 1 Tag fuer aussagekraeftige Ergebnisse)
            min_wait = timedelta(hours=24)
            if datetime.now(timezone.utc) - pred_time < min_wait:
                logger.debug(f"Prediction {prediction.get('id')} too recent for backtest")
                return None

            # Hole aktuelle Preise
            price_at_prediction = prediction.get("price_current", 0)
            predicted_direction = prediction.get("price_direction", "neutral")
            predicted_change_1h = prediction.get("price_change_percent_1h")
            predicted_change_1d = prediction.get("price_change_percent_1d")

            # Hole tatsaechliche Preise
            price_1h_later = await self._fetch_price_at_time(
                symbol, timeframe, pred_time + timedelta(hours=1)
            )
            price_1d_later = await self._fetch_price_at_time(
                symbol, timeframe, pred_time + timedelta(days=1)
            )

            # Berechne tatsaechliche Aenderungen
            actual_change_1h = None
            actual_change_1d = None
            price_error_1h = None
            price_error_1d = None

            if price_1h_later and price_at_prediction:
                actual_change_1h = ((price_1h_later - price_at_prediction) / price_at_prediction) * 100
                if predicted_change_1h is not None:
                    price_error_1h = abs(predicted_change_1h - actual_change_1h)

            if price_1d_later and price_at_prediction:
                actual_change_1d = ((price_1d_later - price_at_prediction) / price_at_prediction) * 100
                if predicted_change_1d is not None:
                    price_error_1d = abs(predicted_change_1d - actual_change_1d)

            # Bestimme tatsaechliche Richtung (basierend auf 1d)
            actual_direction = "neutral"
            if price_1d_later and price_at_prediction:
                actual_direction = self._determine_actual_direction(
                    price_at_prediction, price_1d_later, threshold=0.005
                )

            # Direction Accuracy
            direction_correct = predicted_direction == actual_direction

            # Regime Backtest
            regime_predicted = prediction.get("regime_current", "sideways")
            ohlcv_snapshot = prediction.get("ohlcv_snapshot", [])
            regime_actual = self._calculate_actual_regime(ohlcv_snapshot)
            regime_correct = regime_predicted == regime_actual

            # Pattern Outcome (vereinfacht)
            patterns = prediction.get("patterns", [])
            pattern_outcome = "inconclusive"
            if patterns:
                # Wenn bullish pattern und Preis gestiegen -> success
                # Wenn bearish pattern und Preis gefallen -> success
                main_pattern = patterns[0] if patterns else {}
                pattern_direction = main_pattern.get("direction", "neutral")

                if actual_direction == pattern_direction:
                    pattern_outcome = "success"
                elif actual_direction != "neutral" and pattern_direction != "neutral":
                    pattern_outcome = "failure"

            # Overall Score (gewichtet)
            score_parts = []
            if direction_correct:
                score_parts.append(0.4)  # 40% fuer Richtung
            if regime_correct:
                score_parts.append(0.3)  # 30% fuer Regime
            if pattern_outcome == "success":
                score_parts.append(0.3)  # 30% fuer Pattern
            elif pattern_outcome == "inconclusive":
                score_parts.append(0.15)  # 15% wenn unklar

            overall_score = sum(score_parts)

            result = BacktestResult(
                prediction_id=prediction.get("id", ""),
                symbol=symbol,
                timeframe=timeframe,
                prediction_timestamp=prediction_timestamp,
                backtest_timestamp=datetime.now(timezone.utc).isoformat(),
                price_predicted_direction=predicted_direction,
                price_actual_direction=actual_direction,
                price_direction_correct=direction_correct,
                price_predicted_change_1h=predicted_change_1h,
                price_actual_change_1h=actual_change_1h,
                price_error_1h=price_error_1h,
                price_predicted_change_1d=predicted_change_1d,
                price_actual_change_1d=actual_change_1d,
                price_error_1d=price_error_1d,
                regime_predicted=regime_predicted,
                regime_actual=regime_actual,
                regime_correct=regime_correct,
                patterns_predicted=[p.get("type") for p in patterns],
                pattern_outcome=pattern_outcome,
                overall_score=round(overall_score, 3)
            )

            # Speichere Ergebnis
            self._results.append(result)
            self._save_results()

            logger.info(
                f"Backtest complete for {symbol}/{timeframe}: "
                f"direction={'✓' if direction_correct else '✗'}, "
                f"regime={'✓' if regime_correct else '✗'}, "
                f"score={overall_score:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return None

    async def run_backtests(
        self,
        predictions: list[dict],
        max_backtests: int = 50
    ) -> dict:
        """
        Fuehrt Backtests fuer mehrere Predictions durch.

        Args:
            predictions: Liste von Predictions
            max_backtests: Maximale Anzahl Backtests

        Returns:
            Zusammenfassung
        """
        new_backtests = 0
        skipped = 0
        errors = 0

        # Finde bereits getestete Predictions
        tested_ids = {r.prediction_id for r in self._results}

        for prediction in predictions[:max_backtests]:
            pred_id = prediction.get("id", "")

            if pred_id in tested_ids:
                skipped += 1
                continue

            result = await self.backtest_prediction(prediction)
            if result:
                new_backtests += 1
            else:
                errors += 1

        return {
            "new_backtests": new_backtests,
            "skipped": skipped,
            "errors": errors,
            "total_results": len(self._results)
        }

    def get_results(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Hole Backtest-Ergebnisse mit Filtern."""
        results = []

        for r in reversed(self._results):
            if symbol and r.symbol != symbol.upper():
                continue
            if timeframe and r.timeframe != timeframe.upper():
                continue

            results.append(r.to_dict())

            if len(results) >= limit:
                break

        return results

    def get_summary(self) -> dict:
        """Berechne Zusammenfassung aller Backtests."""
        if not self._results:
            return BacktestSummary(
                total_predictions=0,
                backtested=0,
                pending=0,
                price_direction_accuracy=0.0,
                price_mae_1h=0.0,
                price_mae_1d=0.0,
                regime_accuracy=0.0,
                pattern_success_rate=0.0
            ).to_dict()

        # Berechne Metriken
        direction_correct = sum(1 for r in self._results if r.price_direction_correct)
        regime_correct = sum(1 for r in self._results if r.regime_correct)
        pattern_success = sum(1 for r in self._results if r.pattern_outcome == "success")
        pattern_total = sum(1 for r in self._results if r.pattern_outcome != "inconclusive")

        # MAE
        errors_1h = [r.price_error_1h for r in self._results if r.price_error_1h is not None]
        errors_1d = [r.price_error_1d for r in self._results if r.price_error_1d is not None]

        mae_1h = sum(errors_1h) / len(errors_1h) if errors_1h else 0.0
        mae_1d = sum(errors_1d) / len(errors_1d) if errors_1d else 0.0

        # Nach Symbol/Timeframe
        by_symbol = {}
        by_timeframe = {}

        for r in self._results:
            # Symbol
            if r.symbol not in by_symbol:
                by_symbol[r.symbol] = {"total": 0, "direction_correct": 0}
            by_symbol[r.symbol]["total"] += 1
            if r.price_direction_correct:
                by_symbol[r.symbol]["direction_correct"] += 1

            # Timeframe
            if r.timeframe not in by_timeframe:
                by_timeframe[r.timeframe] = {"total": 0, "direction_correct": 0}
            by_timeframe[r.timeframe]["total"] += 1
            if r.price_direction_correct:
                by_timeframe[r.timeframe]["direction_correct"] += 1

        # Berechne Accuracy pro Symbol/Timeframe
        for sym, data in by_symbol.items():
            data["accuracy"] = round(data["direction_correct"] / data["total"] * 100, 1) if data["total"] else 0

        for tf, data in by_timeframe.items():
            data["accuracy"] = round(data["direction_correct"] / data["total"] * 100, 1) if data["total"] else 0

        total = len(self._results)

        return BacktestSummary(
            total_predictions=total,
            backtested=total,
            pending=0,
            price_direction_accuracy=round(direction_correct / total * 100, 1) if total else 0.0,
            price_mae_1h=round(mae_1h, 3),
            price_mae_1d=round(mae_1d, 3),
            regime_accuracy=round(regime_correct / total * 100, 1) if total else 0.0,
            pattern_success_rate=round(pattern_success / pattern_total * 100, 1) if pattern_total else 0.0,
            by_symbol=by_symbol,
            by_timeframe=by_timeframe
        ).to_dict()

    def clear_results(self):
        """Loesche alle Backtest-Ergebnisse."""
        self._results = []
        self._save_results()
        logger.info("Backtest results cleared")

    def clear_memory(self):
        """Loesche In-Memory Daten."""
        count = len(self._results)
        self._results = []
        logger.info(f"Backtest memory cleared: {count} results")
        return {"cleared": True, "results_cleared": count}


# Global singleton
backtesting_service = BacktestingService()
