"""
Lokale Pivot Points Berechnung aus OHLCV-Daten.

Berechnet Standard, Fibonacci und Camarilla Pivot Points
ohne externe API-Aufrufe.
"""

from datetime import datetime, timezone
from typing import Optional
from loguru import logger


def calculate_pivot_points(
    high: float,
    low: float,
    close: float,
    timestamp: Optional[datetime] = None,
) -> dict:
    """
    Berechnet alle Pivot Point Varianten aus High, Low, Close.

    Standard Pivot Points (Floor Trader Pivots):
        Pivot = (High + Low + Close) / 3
        R1 = 2 * Pivot - Low
        S1 = 2 * Pivot - High
        R2 = Pivot + (High - Low)
        S2 = Pivot - (High - Low)
        R3 = High + 2 * (Pivot - Low)
        S3 = Low - 2 * (High - Pivot)

    Fibonacci Pivot Points:
        Pivot = (High + Low + Close) / 3
        R1 = Pivot + 0.382 * (High - Low)
        R2 = Pivot + 0.618 * (High - Low)
        R3 = Pivot + 1.000 * (High - Low)
        S1 = Pivot - 0.382 * (High - Low)
        S2 = Pivot - 0.618 * (High - Low)
        S3 = Pivot - 1.000 * (High - Low)

    Camarilla Pivot Points:
        R1 = Close + (High - Low) * 1.1 / 12
        R2 = Close + (High - Low) * 1.1 / 6
        R3 = Close + (High - Low) * 1.1 / 4
        R4 = Close + (High - Low) * 1.1 / 2
        S1 = Close - (High - Low) * 1.1 / 12
        S2 = Close - (High - Low) * 1.1 / 6
        S3 = Close - (High - Low) * 1.1 / 4
        S4 = Close - (High - Low) * 1.1 / 2

    Args:
        high: Höchstkurs der Vorperiode
        low: Tiefstkurs der Vorperiode
        close: Schlusskurs der Vorperiode
        timestamp: Zeitstempel für die Pivot Points

    Returns:
        Dictionary mit allen Pivot Point Werten
    """
    # Range für Berechnungen
    range_hl = high - low

    # Standard Pivot Points
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + range_hl
    s2 = pivot - range_hl
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    # Fibonacci Pivot Points
    fib_r1 = pivot + 0.382 * range_hl
    fib_r2 = pivot + 0.618 * range_hl
    fib_r3 = pivot + range_hl
    fib_s1 = pivot - 0.382 * range_hl
    fib_s2 = pivot - 0.618 * range_hl
    fib_s3 = pivot - range_hl

    # Camarilla Pivot Points
    cam_factor = range_hl * 1.1
    cam_r1 = close + cam_factor / 12
    cam_r2 = close + cam_factor / 6
    cam_r3 = close + cam_factor / 4
    cam_r4 = close + cam_factor / 2
    cam_s1 = close - cam_factor / 12
    cam_s2 = close - cam_factor / 6
    cam_s3 = close - cam_factor / 4
    cam_s4 = close - cam_factor / 2

    result = {
        # Standard Pivots
        "pivot": round(pivot, 8),
        "r1": round(r1, 8),
        "r2": round(r2, 8),
        "r3": round(r3, 8),
        "s1": round(s1, 8),
        "s2": round(s2, 8),
        "s3": round(s3, 8),
        # Fibonacci Pivots
        "fib_r1": round(fib_r1, 8),
        "fib_r2": round(fib_r2, 8),
        "fib_r3": round(fib_r3, 8),
        "fib_s1": round(fib_s1, 8),
        "fib_s2": round(fib_s2, 8),
        "fib_s3": round(fib_s3, 8),
        # Camarilla Pivots
        "cam_r1": round(cam_r1, 8),
        "cam_r2": round(cam_r2, 8),
        "cam_r3": round(cam_r3, 8),
        "cam_r4": round(cam_r4, 8),
        "cam_s1": round(cam_s1, 8),
        "cam_s2": round(cam_s2, 8),
        "cam_s3": round(cam_s3, 8),
        "cam_s4": round(cam_s4, 8),
    }

    if timestamp:
        result["timestamp"] = timestamp

    return result


def calculate_pivot_points_from_ohlcv(ohlcv_data: list[dict]) -> list[dict]:
    """
    Berechnet Pivot Points für eine Liste von OHLCV-Daten.

    Die Pivot Points für jeden Zeitpunkt werden aus der VORHERIGEN
    Kerze berechnet (z.B. Daily Pivots aus Yesterday's HLC).

    Args:
        ohlcv_data: Liste von OHLCV-Dictionaries, sortiert nach Timestamp ASC
                   Erwartet: timestamp/datetime, high, low, close

    Returns:
        Liste von Pivot Point Dictionaries mit Timestamps
    """
    if not ohlcv_data or len(ohlcv_data) < 2:
        return []

    results = []

    # Daten sollten chronologisch sortiert sein (älteste zuerst)
    for i in range(1, len(ohlcv_data)):
        prev_candle = ohlcv_data[i - 1]
        curr_candle = ohlcv_data[i]

        # Timestamp der aktuellen Kerze (für die die Pivots gelten)
        ts = curr_candle.get("timestamp") or curr_candle.get("datetime")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue

        # HLC der vorherigen Kerze
        try:
            high = float(prev_candle.get("high", 0))
            low = float(prev_candle.get("low", 0))
            close = float(prev_candle.get("close", 0))
        except (ValueError, TypeError):
            continue

        if high <= 0 or low <= 0 or close <= 0:
            continue

        if high < low:
            # Ungültige Daten
            continue

        pivots = calculate_pivot_points(high, low, close, ts)
        results.append(pivots)

    return results


class PivotCalculatorService:
    """
    Service für lokale Pivot Point Berechnung und Speicherung.
    """

    def __init__(self):
        self._stats = {
            "calculations": 0,
            "errors": 0,
        }

    async def calculate_and_store_pivots(
        self,
        symbol: str,
        timeframe: str,
        ohlcv_data: list[dict],
        timescaledb_service,
    ) -> int:
        """
        Berechnet Pivot Points aus OHLCV-Daten und speichert sie in TimescaleDB.

        Args:
            symbol: Trading-Symbol
            timeframe: Timeframe (z.B. "H1", "D1")
            ohlcv_data: OHLCV-Daten (chronologisch sortiert)
            timescaledb_service: TimescaleDB Service Instanz

        Returns:
            Anzahl der gespeicherten Pivot Point Einträge
        """
        if not ohlcv_data or len(ohlcv_data) < 2:
            return 0

        try:
            # Pivot Points berechnen
            pivots = calculate_pivot_points_from_ohlcv(ohlcv_data)

            if not pivots:
                return 0

            self._stats["calculations"] += len(pivots)

            # In TimescaleDB speichern
            count = await timescaledb_service.upsert_levels_indicators(
                symbol=symbol,
                timeframe=timeframe,
                data=pivots,
                source="calculated",
            )

            logger.debug(
                f"Calculated and stored {count} pivot points for {symbol}/{timeframe}"
            )
            return count

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to calculate pivots for {symbol}/{timeframe}: {e}")
            return 0

    def get_stats(self) -> dict:
        """Gibt Statistiken zurück."""
        return self._stats.copy()


# Singleton-Instanz
pivot_calculator = PivotCalculatorService()
