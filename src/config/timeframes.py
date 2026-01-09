"""
Zentrale Timeframe-Konfiguration für alle Services.

Diese Datei ist die EINZIGE Quelle für Timeframe-Definitionen.
Alle Services MÜSSEN diese Konfiguration verwenden.

Standard-Format: Großbuchstaben mit Präfix (M1, M5, H1, D1, etc.)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import timedelta


class Timeframe(str, Enum):
    """
    Standard-Timeframes für das gesamte System.

    Format: {Einheit}{Anzahl}
    - M = Minuten (M1, M5, M15, M30)
    - H = Stunden (H1, H4)
    - D = Tage (D1)
    - W = Wochen (W1)
    - MN = Monat (MN)
    """
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN = "MN"


# Mapping von verschiedenen Formaten auf das Standard-Format
TIMEFRAME_ALIASES: Dict[str, Timeframe] = {
    # Standard-Format (bereits normalisiert)
    "M1": Timeframe.M1,
    "M5": Timeframe.M5,
    "M15": Timeframe.M15,
    "M30": Timeframe.M30,
    "H1": Timeframe.H1,
    "H4": Timeframe.H4,
    "D1": Timeframe.D1,
    "W1": Timeframe.W1,
    "MN": Timeframe.MN,

    # Kleinbuchstaben-Varianten
    "m1": Timeframe.M1,
    "m5": Timeframe.M5,
    "m15": Timeframe.M15,
    "m30": Timeframe.M30,
    "h1": Timeframe.H1,
    "h4": Timeframe.H4,
    "d1": Timeframe.D1,
    "w1": Timeframe.W1,
    "mn": Timeframe.MN,

    # TwelveData-Format
    "1min": Timeframe.M1,
    "5min": Timeframe.M5,
    "15min": Timeframe.M15,
    "30min": Timeframe.M30,
    "1h": Timeframe.H1,
    "4h": Timeframe.H4,
    "1day": Timeframe.D1,
    "1week": Timeframe.W1,
    "1month": Timeframe.MN,

    # Yahoo Finance-Format
    "1m": Timeframe.M1,
    "5m": Timeframe.M5,
    "15m": Timeframe.M15,
    "30m": Timeframe.M30,
    "60m": Timeframe.H1,
    "1d": Timeframe.D1,
    "1wk": Timeframe.W1,
    "1mo": Timeframe.MN,

    # Alternative Schreibweisen
    "1minute": Timeframe.M1,
    "5minute": Timeframe.M5,
    "5minutes": Timeframe.M5,
    "15minute": Timeframe.M15,
    "15minutes": Timeframe.M15,
    "30minute": Timeframe.M30,
    "30minutes": Timeframe.M30,
    "1hour": Timeframe.H1,
    "4hour": Timeframe.H4,
    "daily": Timeframe.D1,
    "day": Timeframe.D1,
    "weekly": Timeframe.W1,
    "week": Timeframe.W1,
    "monthly": Timeframe.MN,
    "month": Timeframe.MN,
}

# Mapping zu TwelveData-Format
TIMEFRAME_TO_TWELVEDATA: Dict[Timeframe, str] = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1day",
    Timeframe.W1: "1week",
    Timeframe.MN: "1month",
}

# Mapping zu Yahoo Finance-Format
TIMEFRAME_TO_YFINANCE: Dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.H4: "1h",    # Yahoo hat kein 4h, Fallback auf 1h
    Timeframe.D1: "1d",
    Timeframe.W1: "1wk",
    Timeframe.MN: "1mo",
}

# Yahoo Finance Fallback-Hinweise (für Logging/Warnungen)
YFINANCE_FALLBACK_INFO: Dict[Timeframe, Tuple[str, str]] = {
    Timeframe.H4: ("1h", "Yahoo Finance unterstützt kein 4-Stunden-Intervall"),
}

# Mapping zu EasyInsight-Format (Feldpräfixe)
TIMEFRAME_TO_EASYINSIGHT: Dict[Timeframe, str] = {
    Timeframe.M1: "m1",
    Timeframe.M5: "m5",
    Timeframe.M15: "m15",
    Timeframe.M30: "m30",
    Timeframe.H1: "h1",
    Timeframe.H4: "h4",
    Timeframe.D1: "d1",
    Timeframe.W1: "w1",
    Timeframe.MN: "mn",
}

# Kerzen pro Tag für jedes Timeframe
TIMEFRAME_CANDLES_PER_DAY: Dict[Timeframe, float] = {
    Timeframe.M1: 1440.0,    # 60 * 24
    Timeframe.M5: 288.0,     # 12 * 24
    Timeframe.M15: 96.0,     # 4 * 24
    Timeframe.M30: 48.0,     # 2 * 24
    Timeframe.H1: 24.0,      # 1 * 24
    Timeframe.H4: 6.0,       # 0.25 * 24
    Timeframe.D1: 1.0,       # 1 pro Tag
    Timeframe.W1: 0.142857,  # 1/7 pro Tag
    Timeframe.MN: 0.033333,  # ~1/30 pro Tag
}

# Timeframe-Dauer als timedelta
TIMEFRAME_DURATION: Dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M30: timedelta(minutes=30),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
    Timeframe.W1: timedelta(weeks=1),
    Timeframe.MN: timedelta(days=30),  # Approximation
}

# Timeframe in Minuten
TIMEFRAME_MINUTES: Dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
    Timeframe.MN: 43200,  # Approximation (30 Tage)
}

# Standard-Timeframes für verschiedene Anwendungsfälle
DEFAULT_TRAINING_TIMEFRAMES: List[Timeframe] = [
    Timeframe.M15,
    Timeframe.H1,
    Timeframe.D1,
]

DEFAULT_INFERENCE_TIMEFRAMES: List[Timeframe] = [
    Timeframe.H1,
    Timeframe.D1,
]

DEFAULT_PATTERN_TIMEFRAMES: List[Timeframe] = [
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
]

# Sortierreihenfolge (kleinste zu größte)
TIMEFRAME_ORDER: List[Timeframe] = [
    Timeframe.M1,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
    Timeframe.W1,
    Timeframe.MN,
]


def normalize_timeframe(timeframe: str) -> Timeframe:
    """
    Normalisiert einen Timeframe-String zum Standard-Format.

    Args:
        timeframe: Timeframe in beliebigem Format (z.B. "1h", "H1", "1hour", "60m")

    Returns:
        Timeframe: Normalisierter Timeframe als Enum

    Raises:
        ValueError: Wenn der Timeframe nicht erkannt wird

    Examples:
        >>> normalize_timeframe("1h")
        Timeframe.H1
        >>> normalize_timeframe("H1")
        Timeframe.H1
        >>> normalize_timeframe("1hour")
        Timeframe.H1
        >>> normalize_timeframe("daily")
        Timeframe.D1
    """
    if isinstance(timeframe, Timeframe):
        return timeframe

    # Leerzeichen entfernen und lowercase für Lookup
    tf_clean = str(timeframe).strip()

    # Direkte Suche im Alias-Dict
    if tf_clean in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[tf_clean]

    # Lowercase-Suche
    tf_lower = tf_clean.lower()
    if tf_lower in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[tf_lower]

    # Uppercase-Suche
    tf_upper = tf_clean.upper()
    if tf_upper in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[tf_upper]

    raise ValueError(
        f"Unbekannter Timeframe: '{timeframe}'. "
        f"Gültige Werte: {', '.join(t.value for t in Timeframe)}"
    )


def normalize_timeframe_safe(timeframe: str, default: Timeframe = Timeframe.H1) -> Timeframe:
    """
    Normalisiert einen Timeframe-String sicher mit Fallback.

    Args:
        timeframe: Timeframe in beliebigem Format
        default: Fallback-Timeframe bei ungültiger Eingabe

    Returns:
        Timeframe: Normalisierter Timeframe oder default
    """
    try:
        return normalize_timeframe(timeframe)
    except ValueError:
        return default


def to_twelvedata(timeframe: str | Timeframe) -> str:
    """
    Konvertiert einen Timeframe zum TwelveData-Format.

    Args:
        timeframe: Timeframe in beliebigem Format

    Returns:
        str: TwelveData-kompatibles Format (z.B. "1h", "1day")
    """
    tf = normalize_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    return TIMEFRAME_TO_TWELVEDATA[tf]


def to_yfinance(timeframe: str | Timeframe) -> str:
    """
    Konvertiert einen Timeframe zum Yahoo Finance-Format.

    Args:
        timeframe: Timeframe in beliebigem Format

    Returns:
        str: Yahoo Finance-kompatibles Format (z.B. "1h", "1d")
    """
    tf = normalize_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    return TIMEFRAME_TO_YFINANCE[tf]


def to_easyinsight(timeframe: str | Timeframe) -> str:
    """
    Konvertiert einen Timeframe zum EasyInsight-Präfix-Format.

    Args:
        timeframe: Timeframe in beliebigem Format

    Returns:
        str: EasyInsight-Präfix (z.B. "h1", "d1")
    """
    tf = normalize_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    return TIMEFRAME_TO_EASYINSIGHT[tf]


def get_candles_per_day(timeframe: str | Timeframe) -> float:
    """
    Gibt die Anzahl der Kerzen pro Tag für einen Timeframe zurück.

    Args:
        timeframe: Timeframe in beliebigem Format

    Returns:
        float: Anzahl Kerzen pro Tag
    """
    tf = normalize_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    return TIMEFRAME_CANDLES_PER_DAY[tf]


def get_duration(timeframe: str | Timeframe) -> timedelta:
    """
    Gibt die Dauer eines Timeframes als timedelta zurück.

    Args:
        timeframe: Timeframe in beliebigem Format

    Returns:
        timedelta: Dauer des Timeframes
    """
    tf = normalize_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    return TIMEFRAME_DURATION[tf]


def get_minutes(timeframe: str | Timeframe) -> int:
    """
    Gibt die Dauer eines Timeframes in Minuten zurück.

    Args:
        timeframe: Timeframe in beliebigem Format

    Returns:
        int: Dauer in Minuten
    """
    tf = normalize_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    return TIMEFRAME_MINUTES[tf]


def calculate_limit_for_days(timeframe: str | Timeframe, days: int, max_limit: int = 5000) -> int:
    """
    Berechnet die benötigte Anzahl Datenpunkte für eine bestimmte Anzahl Tage.

    Args:
        timeframe: Timeframe in beliebigem Format
        days: Anzahl der Tage
        max_limit: Maximales Limit (z.B. TwelveData max 5000)

    Returns:
        int: Berechnetes Limit (mindestens 100, maximal max_limit)
    """
    candles_per_day = get_candles_per_day(timeframe)
    calculated = int(days * candles_per_day)
    return min(max(calculated, 100), max_limit)


def sort_timeframes(timeframes: List[str | Timeframe]) -> List[Timeframe]:
    """
    Sortiert eine Liste von Timeframes von kleinster zu größter Einheit.

    Args:
        timeframes: Liste von Timeframes in beliebigem Format

    Returns:
        List[Timeframe]: Sortierte Liste von Timeframes
    """
    normalized = [
        normalize_timeframe(tf) if isinstance(tf, str) else tf
        for tf in timeframes
    ]
    return sorted(normalized, key=lambda tf: TIMEFRAME_ORDER.index(tf))


def is_valid_timeframe(timeframe: str) -> bool:
    """
    Prüft, ob ein Timeframe-String gültig ist.

    Args:
        timeframe: Zu prüfender Timeframe-String

    Returns:
        bool: True wenn gültig, False sonst
    """
    try:
        normalize_timeframe(timeframe)
        return True
    except ValueError:
        return False


def get_yfinance_fallback_warning(timeframe: str | Timeframe) -> Optional[str]:
    """
    Gibt eine Warnung zurück, wenn Yahoo Finance für diesen Timeframe
    einen Fallback verwendet.

    Args:
        timeframe: Timeframe in beliebigem Format

    Returns:
        Optional[str]: Warnmeldung oder None
    """
    tf = normalize_timeframe(timeframe) if isinstance(timeframe, str) else timeframe
    if tf in YFINANCE_FALLBACK_INFO:
        _, warning = YFINANCE_FALLBACK_INFO[tf]
        return warning
    return None
