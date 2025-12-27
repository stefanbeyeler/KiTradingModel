"""
Zentrale Indikator-Konfiguration für alle Services.

Diese Datei ist die EINZIGE Quelle für Indikator-Definitionen.
Alle Services MÜSSEN diese Konfiguration verwenden.

Jeder Indikator enthält:
- Bezeichnungen für TwelveData und EasyInsight APIs
- Kategorie und Beschreibung
- Stärken, Schwächen und Anwendungsgebiete
- Status (aktiv/inaktiv)
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class IndicatorCategory(str, Enum):
    """Kategorien für technische Indikatoren."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND_FILTER = "trend_filter"
    PRICE_TRANSFORM = "price_transform"
    ML_FEATURES = "ml_features"
    PATTERN = "pattern"


class IndicatorConfig(BaseModel):
    """Konfiguration für einen einzelnen Indikator."""

    # Identifikation
    id: str = Field(..., description="Eindeutige ID des Indikators (lowercase)")
    name: str = Field(..., description="Anzeigename des Indikators")

    # API-Bezeichnungen
    twelvedata_name: Optional[str] = Field(None, description="Bezeichnung in TwelveData API")
    easyinsight_name: Optional[str] = Field(None, description="Bezeichnung in EasyInsight API")

    # Klassifikation
    category: IndicatorCategory = Field(..., description="Kategorie des Indikators")

    # Status
    enabled: bool = Field(True, description="Ob der Indikator aktiv ist")

    # Dokumentation
    description: str = Field(..., description="Kurze Beschreibung der Funktionsweise")
    calculation: str = Field("", description="Berechnungsformel/Methode")
    strengths: List[str] = Field(default_factory=list, description="Stärken des Indikators")
    weaknesses: List[str] = Field(default_factory=list, description="Schwächen des Indikators")
    use_cases: List[str] = Field(default_factory=list, description="Typische Anwendungsgebiete")

    # Parameter (Standard-Werte)
    default_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Standard-Parameter für den Indikator"
    )

    # Metadaten
    created_at: Optional[datetime] = Field(None, description="Erstellungszeitpunkt")
    updated_at: Optional[datetime] = Field(None, description="Letzte Aktualisierung")


# =============================================================================
# TREND / MOVING AVERAGES
# =============================================================================

SMA = IndicatorConfig(
    id="sma",
    name="Simple Moving Average (SMA)",
    twelvedata_name="sma",
    easyinsight_name="sma",
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Berechnet den arithmetischen Durchschnitt der Schlusskurse über einen definierten Zeitraum.",
    calculation="SMA = (P1 + P2 + ... + Pn) / n",
    strengths=[
        "Einfach zu verstehen und zu berechnen",
        "Glättet Preisschwankungen effektiv",
        "Guter Indikator für langfristige Trends",
        "Häufig als dynamische Unterstützung/Widerstand genutzt"
    ],
    weaknesses=[
        "Reagiert langsam auf Preisänderungen (Lagging)",
        "Gewichtet alle Perioden gleich",
        "Kann bei Seitwärtsmärkten falsche Signale liefern"
    ],
    use_cases=[
        "Trendidentifikation (Preis über/unter SMA)",
        "Crossover-Strategien (SMA 50/200)",
        "Dynamische Support/Resistance-Zonen",
        "Glättung von Preisdaten für weitere Analyse"
    ],
    default_params={"period": 20}
)

EMA = IndicatorConfig(
    id="ema",
    name="Exponential Moving Average (EMA)",
    twelvedata_name="ema",
    easyinsight_name="ema",
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Gleitender Durchschnitt mit exponentieller Gewichtung neuerer Kurse.",
    calculation="EMA = (Close - EMA_prev) × Multiplier + EMA_prev; Multiplier = 2/(Period+1)",
    strengths=[
        "Reagiert schneller auf Preisänderungen als SMA",
        "Gewichtet aktuelle Preise stärker",
        "Besser für kurzfristige Trends geeignet",
        "Weniger Lag als SMA"
    ],
    weaknesses=[
        "Kann zu mehr Fehlsignalen führen",
        "Empfindlicher gegenüber Marktrauschen",
        "Komplexere Berechnung"
    ],
    use_cases=[
        "Kurzfristige Trendanalyse",
        "MACD-Berechnung (12/26 EMA)",
        "Schnellere Crossover-Signale",
        "Trailing-Stop-Berechnung"
    ],
    default_params={"period": 20}
)

WMA = IndicatorConfig(
    id="wma",
    name="Weighted Moving Average (WMA)",
    twelvedata_name="wma",
    easyinsight_name=None,
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Gleitender Durchschnitt mit linearer Gewichtung, wobei neuere Werte stärker gewichtet werden.",
    calculation="WMA = (P1×n + P2×(n-1) + ... + Pn×1) / (n×(n+1)/2)",
    strengths=[
        "Gewichtet neuere Preise stärker als ältere",
        "Glatter als EMA bei ähnlicher Reaktionszeit",
        "Guter Kompromiss zwischen SMA und EMA"
    ],
    weaknesses=[
        "Komplexere Berechnung als SMA",
        "Weniger verbreitet, daher weniger Referenzwerte"
    ],
    use_cases=[
        "Alternative zu EMA für Trendfolge",
        "Preisglättung für Indikatoren",
        "Mittelfristige Trendanalyse"
    ],
    default_params={"period": 20}
)

DEMA = IndicatorConfig(
    id="dema",
    name="Double Exponential Moving Average (DEMA)",
    twelvedata_name="dema",
    easyinsight_name=None,
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Doppelter EMA zur Reduzierung des Lags bei gleichzeitiger Glättung.",
    calculation="DEMA = 2 × EMA(n) - EMA(EMA(n))",
    strengths=[
        "Deutlich reduzierter Lag gegenüber EMA",
        "Reagiert schneller auf Trendwechsel",
        "Weniger Verzögerung bei Signalen"
    ],
    weaknesses=[
        "Kann bei volatilen Märkten überschiessen",
        "Mehr Fehlsignale in Seitwärtsmärkten",
        "Komplexere Interpretation"
    ],
    use_cases=[
        "Schnelle Trendwechsel-Erkennung",
        "Scalping und Day-Trading",
        "Kombination mit langsameren MAs"
    ],
    default_params={"period": 20}
)

TEMA = IndicatorConfig(
    id="tema",
    name="Triple Exponential Moving Average (TEMA)",
    twelvedata_name="tema",
    easyinsight_name=None,
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Dreifacher EMA für minimalen Lag und maximale Glättung.",
    calculation="TEMA = 3×EMA - 3×EMA(EMA) + EMA(EMA(EMA))",
    strengths=[
        "Minimaler Lag aller Moving Averages",
        "Sehr schnelle Reaktion auf Preisänderungen",
        "Effektive Trendfolge"
    ],
    weaknesses=[
        "Sehr empfindlich gegenüber Marktrauschen",
        "Kann zu viele Signale generieren",
        "Benötigt Filterung"
    ],
    use_cases=[
        "Sehr kurzfristiges Trading",
        "Schnelle Trendwechsel-Signale",
        "Kombination mit Oszillatoren"
    ],
    default_params={"period": 20}
)

KAMA = IndicatorConfig(
    id="kama",
    name="Kaufman Adaptive Moving Average (KAMA)",
    twelvedata_name="kama",
    easyinsight_name=None,
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Adaptiver MA, der seine Sensitivität automatisch an die Marktvolatilität anpasst.",
    calculation="KAMA = KAMA_prev + SC × (Price - KAMA_prev); SC basiert auf Efficiency Ratio",
    strengths=[
        "Passt sich automatisch an Marktbedingungen an",
        "Weniger Whipsaws in Seitwärtsmärkten",
        "Folgt starken Trends eng"
    ],
    weaknesses=[
        "Komplexe Berechnung",
        "Kann bei plötzlichen Trendwechseln langsam reagieren",
        "Parameter-Optimierung erforderlich"
    ],
    use_cases=[
        "Adaptive Trendfolge-Systeme",
        "Volatilitätsangepasste Strategien",
        "Langfristige Positionierung"
    ],
    default_params={"period": 10}
)

MAMA = IndicatorConfig(
    id="mama",
    name="MESA Adaptive Moving Average (MAMA)",
    twelvedata_name="mama",
    easyinsight_name=None,
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Von John Ehlers entwickelter adaptiver MA basierend auf Hilbert-Transformation.",
    calculation="Nutzt Hilbert-Transformation zur Phasenbestimmung und passt Rate of Change an",
    strengths=[
        "Sehr schnelle Adaption an Zyklen",
        "Minimiert False Signals",
        "Wissenschaftlich fundiert (DSP)"
    ],
    weaknesses=[
        "Komplexe Mathematik (Signalverarbeitung)",
        "Schwer zu optimieren",
        "Weniger intuitiv"
    ],
    use_cases=[
        "Zyklus-basierte Analyse",
        "Professionelle Trading-Systeme",
        "Kombination mit FAMA (Following Adaptive MA)"
    ],
    default_params={"fast_limit": 0.5, "slow_limit": 0.05}
)

T3 = IndicatorConfig(
    id="t3",
    name="T3 Moving Average",
    twelvedata_name="t3",
    easyinsight_name=None,
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Geglätteter Moving Average mit minimalem Lag, entwickelt von Tim Tillson.",
    calculation="T3 = c1×e6 + c2×e5 + c3×e4 + c4×e3; basierend auf GD-MA",
    strengths=[
        "Sehr glatt mit wenig Lag",
        "Gute Balance zwischen Reaktion und Glättung",
        "Reduziert Whipsaws effektiv"
    ],
    weaknesses=[
        "Kann bei schnellen Bewegungen hinterherhinken",
        "Komplexe Berechnung",
        "Weniger verbreitet"
    ],
    use_cases=[
        "Mittelfristige Trendfolge",
        "Glättung für andere Indikatoren",
        "Swing-Trading"
    ],
    default_params={"period": 5, "v_factor": 0.7}
)

TRIMA = IndicatorConfig(
    id="trima",
    name="Triangular Moving Average (TRIMA)",
    twelvedata_name="trima",
    easyinsight_name=None,
    category=IndicatorCategory.TREND,
    enabled=True,
    description="Doppelt geglätteter MA, der die mittleren Werte am stärksten gewichtet.",
    calculation="TRIMA = SMA(SMA(Price, n/2), n/2)",
    strengths=[
        "Sehr glatt, minimiert Rauschen",
        "Gut für langfristige Trends",
        "Stabile Signale"
    ],
    weaknesses=[
        "Erheblicher Lag",
        "Langsame Reaktion auf Trendwechsel"
    ],
    use_cases=[
        "Langfristige Trendbestätigung",
        "Glättung stark volatiler Daten",
        "Positionstrading"
    ],
    default_params={"period": 20}
)

# =============================================================================
# TREND FILTER
# =============================================================================

VWAP = IndicatorConfig(
    id="vwap",
    name="Volume Weighted Average Price (VWAP)",
    twelvedata_name="vwap",
    easyinsight_name=None,
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Volumengewichteter Durchschnittspreis, wichtiger institutioneller Benchmark.",
    calculation="VWAP = Σ(Preis × Volumen) / Σ(Volumen)",
    strengths=[
        "Zeigt den wahren Durchschnittspreis",
        "Wichtiger institutioneller Benchmark",
        "Kombiniert Preis und Volumen"
    ],
    weaknesses=[
        "Nur für Intraday sinnvoll (wird täglich zurückgesetzt)",
        "Kann in illiquiden Märkten verzerrt sein",
        "Lagging Indicator"
    ],
    use_cases=[
        "Institutionelle Order-Ausführung",
        "Intraday Support/Resistance",
        "Fair Value Bestimmung",
        "Algorithmic Trading"
    ],
    default_params={}
)

SUPERTREND = IndicatorConfig(
    id="supertrend",
    name="Supertrend",
    twelvedata_name="supertrend",
    easyinsight_name=None,
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Trendfolge-Indikator basierend auf ATR, zeigt Trendrichtung und Stop-Levels.",
    calculation="SuperTrend = HL2 ± (Multiplier × ATR)",
    strengths=[
        "Klare Kauf/Verkauf-Signale",
        "Integrierter Trailing-Stop",
        "Passt sich an Volatilität an"
    ],
    weaknesses=[
        "Whipsaws in Seitwärtsmärkten",
        "Verzögerung bei Trendwechseln",
        "Parameter-sensitiv"
    ],
    use_cases=[
        "Trendfolge-Trading",
        "Automatische Stop-Loss-Platzierung",
        "Position-Sizing basierend auf Trend"
    ],
    default_params={"period": 10, "multiplier": 3.0}
)

ICHIMOKU = IndicatorConfig(
    id="ichimoku",
    name="Ichimoku Cloud (Ichimoku Kinko Hyo)",
    twelvedata_name="ichimoku",
    easyinsight_name=None,
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Komplettes Trading-System mit Trend, Momentum, Support/Resistance in einem Indikator.",
    calculation="5 Linien: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span",
    strengths=[
        "Vollständiges Trading-System",
        "Zeigt Trend, Momentum und S/R gleichzeitig",
        "Visuelle Wolke für Trendstärke",
        "Projiziert zukünftige S/R-Zonen"
    ],
    weaknesses=[
        "Komplex zu interpretieren",
        "Benötigt Platz auf Charts",
        "Nicht für alle Märkte geeignet (ursprünglich für Aktien)"
    ],
    use_cases=[
        "Multi-Faktor-Trendanalyse",
        "Swing-Trading Setups",
        "Langfristige Positionierung",
        "Crossover-Strategien"
    ],
    default_params={
        "conversion_line_period": 9,
        "base_line_period": 26,
        "lagging_span_period": 52,
        "displacement": 26
    }
)

SAR = IndicatorConfig(
    id="sar",
    name="Parabolic SAR",
    twelvedata_name="sar",
    easyinsight_name=None,
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Stop and Reverse System von Welles Wilder für Trendfolge mit eingebautem Stop.",
    calculation="SAR = SAR_prev + AF × (EP - SAR_prev); AF erhöht sich mit Trend",
    strengths=[
        "Klare visuelle Signale (Punkte über/unter Preis)",
        "Integriertes Stop-Loss-System",
        "Gute Performance in Trendmärkten"
    ],
    weaknesses=[
        "Viele Fehlsignale in Seitwärtsmärkten",
        "Kann zu früh aus Trends aussteigen",
        "Acceleration Factor kann zu aggressiv werden"
    ],
    use_cases=[
        "Trailing-Stop-Management",
        "Trendfolge-Systeme",
        "Entry/Exit-Signale"
    ],
    default_params={"acceleration": 0.02, "maximum": 0.2}
)

# =============================================================================
# MOMENTUM
# =============================================================================

RSI = IndicatorConfig(
    id="rsi",
    name="Relative Strength Index (RSI)",
    twelvedata_name="rsi",
    easyinsight_name="rsi",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Misst die Geschwindigkeit und Stärke von Preisbewegungen auf einer Skala von 0-100.",
    calculation="RSI = 100 - (100 / (1 + RS)); RS = Avg Gain / Avg Loss",
    strengths=[
        "Zeigt überkaufte/überverkaufte Zustände",
        "Divergenzen identifizieren Trendwechsel",
        "Universell einsetzbar",
        "Gut für Mean-Reversion-Strategien"
    ],
    weaknesses=[
        "Kann lange in Extremzonen verbleiben (Trending Markets)",
        "Einzelner RSI reicht nicht für Handelsentscheidungen",
        "Whipsaws in volatilen Märkten"
    ],
    use_cases=[
        "Overbought/Oversold Identifikation (70/30)",
        "Divergenz-Analyse",
        "Trendbestätigung (Mittellinie 50)",
        "Mean-Reversion-Strategien"
    ],
    default_params={"period": 14}
)

MACD = IndicatorConfig(
    id="macd",
    name="Moving Average Convergence Divergence (MACD)",
    twelvedata_name="macd",
    easyinsight_name="macd",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Zeigt die Beziehung zwischen zwei EMAs und deren Dynamik.",
    calculation="MACD = EMA(12) - EMA(26); Signal = EMA(9) von MACD; Histogramm = MACD - Signal",
    strengths=[
        "Kombiniert Trend und Momentum",
        "Vielseitige Signale (Crossover, Divergenz, Histogramm)",
        "Funktioniert in verschiedenen Märkten",
        "Leicht zu interpretieren"
    ],
    weaknesses=[
        "Lagging Indicator",
        "Kann falsche Signale in Seitwärtsmärkten geben",
        "Parameter können optimiert werden müssen"
    ],
    use_cases=[
        "Signal-Line Crossovers",
        "Zero-Line Crossovers",
        "Divergenz-Analyse",
        "Momentum-Bestätigung"
    ],
    default_params={"fast_period": 12, "slow_period": 26, "signal_period": 9}
)

STOCH = IndicatorConfig(
    id="stoch",
    name="Stochastic Oscillator",
    twelvedata_name="stoch",
    easyinsight_name="stoch",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Vergleicht den aktuellen Schlusskurs mit der Preisspanne über einen Zeitraum.",
    calculation="%K = (Close - Low_n) / (High_n - Low_n) × 100; %D = SMA(%K)",
    strengths=[
        "Frühe Warnsignale für Trendwechsel",
        "Gut für Range-Trading",
        "Zeigt Momentum-Verlust an",
        "Divergenzen sehr aussagekräftig"
    ],
    weaknesses=[
        "Viele Fehlsignale in Trendmärkten",
        "Kann lange in Extremzonen bleiben",
        "Erfordert Bestätigung durch andere Indikatoren"
    ],
    use_cases=[
        "Overbought/Oversold (80/20)",
        "Crossover-Signale (%K/%D)",
        "Divergenz-Analyse",
        "Swing-Trading"
    ],
    default_params={"k_period": 14, "d_period": 3, "smooth_k": 3}
)

STOCHRSI = IndicatorConfig(
    id="stochrsi",
    name="Stochastic RSI",
    twelvedata_name="stochrsi",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Wendet den Stochastic auf RSI-Werte an für erhöhte Sensitivität.",
    calculation="StochRSI = (RSI - RSI_Low) / (RSI_High - RSI_Low)",
    strengths=[
        "Sehr sensibel für kurzfristige Bewegungen",
        "Mehr Signale als RSI allein",
        "Gut für Scalping und Day-Trading"
    ],
    weaknesses=[
        "Sehr empfindlich (mehr Fehlsignale)",
        "Erfordert strenge Filterung",
        "Nicht für längerfristiges Trading"
    ],
    use_cases=[
        "Kurzfristiges Momentum-Trading",
        "Timing für Einstiege",
        "Confirmation für andere Signale"
    ],
    default_params={"period": 14, "stoch_period": 14}
)

WILLIAMS_R = IndicatorConfig(
    id="willr",
    name="Williams %R",
    twelvedata_name="willr",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Momentum-Oszillator ähnlich Stochastic, aber mit invertierter Skala (-100 bis 0).",
    calculation="%R = (High_n - Close) / (High_n - Low_n) × -100",
    strengths=[
        "Schnelle Reaktion auf Preisänderungen",
        "Einfache Interpretation",
        "Gut für kurzfristige Trades"
    ],
    weaknesses=[
        "Sehr volatil in Seitwärtsmärkten",
        "Invertierte Skala kann verwirrend sein",
        "Benötigt Bestätigung"
    ],
    use_cases=[
        "Overbought/Oversold (-20/-80)",
        "Momentum-Bestätigung",
        "Failure Swings erkennen"
    ],
    default_params={"period": 14}
)

CCI = IndicatorConfig(
    id="cci",
    name="Commodity Channel Index (CCI)",
    twelvedata_name="cci",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Misst die Abweichung des Preises vom statistischen Durchschnitt.",
    calculation="CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation); TP = (H+L+C)/3",
    strengths=[
        "Zeigt statistische Extreme an",
        "Universell einsetzbar (nicht nur Commodities)",
        "Gut für Zyklen-Erkennung"
    ],
    weaknesses=[
        "Keine festen Overbought/Oversold-Grenzen",
        "Kann in Trends lange extrem bleiben",
        "Erfordert Erfahrung bei Interpretation"
    ],
    use_cases=[
        "Overbought/Oversold (±100 oder ±200)",
        "Trend-Trading (Zero-Line Crossover)",
        "Divergenz-Analyse",
        "Commodity-Zyklen"
    ],
    default_params={"period": 20}
)

CMO = IndicatorConfig(
    id="cmo",
    name="Chande Momentum Oscillator (CMO)",
    twelvedata_name="cmo",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Ähnlich RSI, aber verwendet Rohwerte statt geglättete Durchschnitte.",
    calculation="CMO = (Sum_Up - Sum_Down) / (Sum_Up + Sum_Down) × 100",
    strengths=[
        "Direkte Messung des Momentums",
        "Reagiert schneller als RSI",
        "Symmetrische Skala (-100 bis +100)"
    ],
    weaknesses=[
        "Volatiler als RSI",
        "Weniger verbreitet (weniger Referenzwerte)",
        "Erfordert zusätzliche Filterung"
    ],
    use_cases=[
        "Momentum-Analyse",
        "Overbought/Oversold (±50)",
        "Divergenz-Erkennung"
    ],
    default_params={"period": 14}
)

ROC = IndicatorConfig(
    id="roc",
    name="Rate of Change (ROC)",
    twelvedata_name="roc",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Zeigt die prozentuale Veränderung des Preises über einen Zeitraum.",
    calculation="ROC = ((Close - Close_n) / Close_n) × 100",
    strengths=[
        "Einfach zu verstehen (prozentuale Änderung)",
        "Gut für Momentum-Analyse",
        "Unbegrenzte Skala zeigt starke Bewegungen"
    ],
    weaknesses=[
        "Volatil bei kurzen Perioden",
        "Keine festen Grenzen für Extreme",
        "Kann durch einzelne Ausreisser verzerrt werden"
    ],
    use_cases=[
        "Momentum-Messung",
        "Trendstärke-Analyse",
        "Zero-Line Crossovers"
    ],
    default_params={"period": 10}
)

MOM = IndicatorConfig(
    id="mom",
    name="Momentum",
    twelvedata_name="mom",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Einfachster Momentum-Indikator, zeigt absolute Preisänderung.",
    calculation="MOM = Close - Close_n",
    strengths=[
        "Sehr einfach und direkt",
        "Keine Normalisierung (zeigt absolute Werte)",
        "Führender Indikator"
    ],
    weaknesses=[
        "Abhängig vom Preisniveau des Assets",
        "Nicht vergleichbar zwischen Assets",
        "Keine festen Grenzen"
    ],
    use_cases=[
        "Basales Momentum-Signal",
        "Trend-Beschleunigung erkennen",
        "Input für andere Indikatoren"
    ],
    default_params={"period": 10}
)

PPO = IndicatorConfig(
    id="ppo",
    name="Percentage Price Oscillator (PPO)",
    twelvedata_name="ppo",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="MACD in Prozent ausgedrückt, ermöglicht Vergleich zwischen Assets.",
    calculation="PPO = ((EMA_fast - EMA_slow) / EMA_slow) × 100",
    strengths=[
        "Vergleichbar zwischen verschiedenen Assets",
        "Normalisierte Version von MACD",
        "Gut für Screening"
    ],
    weaknesses=[
        "Lagging wie MACD",
        "Weniger verbreitet als MACD"
    ],
    use_cases=[
        "Asset-Vergleich",
        "Crossover-Signale",
        "Screening nach Momentum"
    ],
    default_params={"fast_period": 12, "slow_period": 26}
)

APO = IndicatorConfig(
    id="apo",
    name="Absolute Price Oscillator (APO)",
    twelvedata_name="apo",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Differenz zwischen zwei EMAs in absoluten Preiseinheiten.",
    calculation="APO = EMA_fast - EMA_slow",
    strengths=[
        "Identisch mit MACD-Linie",
        "Zeigt absolute Preisdifferenz",
        "Einfache Interpretation"
    ],
    weaknesses=[
        "Nicht vergleichbar zwischen Assets",
        "Abhängig vom Preisniveau"
    ],
    use_cases=[
        "Momentum-Analyse",
        "Zero-Line Crossovers",
        "Trendstärke"
    ],
    default_params={"fast_period": 12, "slow_period": 26}
)

AROON = IndicatorConfig(
    id="aroon",
    name="Aroon",
    twelvedata_name="aroon",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Misst die Zeit seit dem letzten Hoch/Tief für Trendstärke.",
    calculation="Aroon Up = ((n - Bars since High) / n) × 100; Aroon Down analog",
    strengths=[
        "Zeigt Trendstärke und -richtung",
        "Erkennt neue Trends früh",
        "Einfache Interpretation (0-100)"
    ],
    weaknesses=[
        "Kann in Range-Märkten falsche Signale geben",
        "Reagiert sprunghaft bei neuen Highs/Lows"
    ],
    use_cases=[
        "Trendidentifikation (Aroon Up > 70)",
        "Crossovers für Trendwechsel",
        "Konsolidierungserkennung"
    ],
    default_params={"period": 25}
)

AROON_OSC = IndicatorConfig(
    id="aroonosc",
    name="Aroon Oscillator",
    twelvedata_name="aroonosc",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Differenz zwischen Aroon Up und Aroon Down als einzelne Linie.",
    calculation="Aroon Osc = Aroon Up - Aroon Down",
    strengths=[
        "Einfache Darstellung (eine Linie)",
        "Klare Trendrichtung (+/- Werte)",
        "Weniger komplex als Aroon-Paar"
    ],
    weaknesses=[
        "Verliert Information über absolute Levels",
        "Extreme Werte können lange anhalten"
    ],
    use_cases=[
        "Schnelle Trendrichtung-Bestimmung",
        "Zero-Line Crossovers",
        "Trendstärke-Messung"
    ],
    default_params={"period": 25}
)

BOP = IndicatorConfig(
    id="bop",
    name="Balance of Power (BOP)",
    twelvedata_name="bop",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Misst die Stärke von Käufern vs. Verkäufern basierend auf Kerzenstruktur.",
    calculation="BOP = (Close - Open) / (High - Low)",
    strengths=[
        "Zeigt Kauf-/Verkaufsdruck",
        "Kein Parameter erforderlich",
        "Frühe Warnung für Trendwechsel"
    ],
    weaknesses=[
        "Volatil bei kleinen Kerzen",
        "Kann durch Gaps verzerrt werden",
        "Weniger verbreitet"
    ],
    use_cases=[
        "Kauf-/Verkaufsdruck-Analyse",
        "Divergenz-Erkennung",
        "Volumen-Bestätigung"
    ],
    default_params={}
)

MFI = IndicatorConfig(
    id="mfi",
    name="Money Flow Index (MFI)",
    twelvedata_name="mfi",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="RSI mit Volumengewichtung - Volume-Weighted RSI.",
    calculation="MFI = 100 - (100 / (1 + Money Ratio)); Money Ratio = Pos. Flow / Neg. Flow",
    strengths=[
        "Kombiniert Preis und Volumen",
        "Zeigt Smart Money Akkumulation/Distribution",
        "Divergenzen sehr aussagekräftig"
    ],
    weaknesses=[
        "Benötigt zuverlässige Volumendaten",
        "Nicht für Forex ohne echtes Volumen",
        "Kann durch grosse Orders verzerrt werden"
    ],
    use_cases=[
        "Overbought/Oversold (80/20)",
        "Divergenz-Analyse (wichtiger als bei RSI)",
        "Akkumulation/Distribution erkennen"
    ],
    default_params={"period": 14}
)

DX = IndicatorConfig(
    id="dx",
    name="Directional Movement Index (DX)",
    twelvedata_name="dx",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Basis für ADX, misst die Richtungsstärke des Trends.",
    calculation="DX = |+DI - -DI| / (+DI + -DI) × 100",
    strengths=[
        "Zeigt Trendstärke ohne Richtung",
        "Basis für ADX-Berechnung",
        "Objektive Trendbewertung"
    ],
    weaknesses=[
        "Volatiler als ADX",
        "Weniger geglättet",
        "Meist wird ADX bevorzugt"
    ],
    use_cases=[
        "Trendstärke-Messung",
        "Input für ADX",
        "Kurzfristige Trend-Analyse"
    ],
    default_params={"period": 14}
)

ADX = IndicatorConfig(
    id="adx",
    name="Average Directional Index (ADX)",
    twelvedata_name="adx",
    easyinsight_name="adx",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Misst die Trendstärke unabhängig von der Richtung (0-100).",
    calculation="ADX = Smoothed Moving Average von DX",
    strengths=[
        "Zeigt Trendstärke objektiv",
        "Hilft Trending vs. Ranging Markets unterscheiden",
        "Gut für Strategie-Auswahl"
    ],
    weaknesses=[
        "Zeigt keine Trendrichtung",
        "Lagging Indicator",
        "Funktioniert besser für starke Trends"
    ],
    use_cases=[
        "Trendstärke-Messung (>25 = Trend, <20 = Range)",
        "Strategie-Filter (Trend vs. Mean-Reversion)",
        "Entry-Timing bei steigendem ADX"
    ],
    default_params={"period": 14}
)

ADXR = IndicatorConfig(
    id="adxr",
    name="Average Directional Movement Rating (ADXR)",
    twelvedata_name="adxr",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Geglättete Version von ADX für stabilere Signale.",
    calculation="ADXR = (ADX + ADX_n) / 2",
    strengths=[
        "Glatter als ADX",
        "Weniger Fehlsignale",
        "Gut für längerfristige Analyse"
    ],
    weaknesses=[
        "Noch mehr Lag als ADX",
        "Langsame Reaktion"
    ],
    use_cases=[
        "Langfristige Trendstärke",
        "Bestätigung für ADX-Signale",
        "Position-Trading"
    ],
    default_params={"period": 14}
)

PLUS_DI = IndicatorConfig(
    id="plus_di",
    name="Plus Directional Indicator (+DI)",
    twelvedata_name="plus_di",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Misst die Stärke der Aufwärtsbewegung.",
    calculation="+DI = Smoothed(+DM) / ATR × 100",
    strengths=[
        "Zeigt Stärke der Bullen",
        "Vergleich mit -DI zeigt Dominanz",
        "Objektive Messung"
    ],
    weaknesses=[
        "Isoliert weniger aussagekräftig",
        "Benötigt -DI für Kontext"
    ],
    use_cases=[
        "Teil des DMI-Systems",
        "Crossover mit -DI für Signale",
        "Trendrichtung bestätigen"
    ],
    default_params={"period": 14}
)

MINUS_DI = IndicatorConfig(
    id="minus_di",
    name="Minus Directional Indicator (-DI)",
    twelvedata_name="minus_di",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Misst die Stärke der Abwärtsbewegung.",
    calculation="-DI = Smoothed(-DM) / ATR × 100",
    strengths=[
        "Zeigt Stärke der Bären",
        "Vergleich mit +DI zeigt Dominanz",
        "Objektive Messung"
    ],
    weaknesses=[
        "Isoliert weniger aussagekräftig",
        "Benötigt +DI für Kontext"
    ],
    use_cases=[
        "Teil des DMI-Systems",
        "Crossover mit +DI für Signale",
        "Abwärtstrend bestätigen"
    ],
    default_params={"period": 14}
)

CRSI = IndicatorConfig(
    id="crsi",
    name="Connors RSI",
    twelvedata_name="crsi",
    easyinsight_name=None,
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Kombination aus 3 Komponenten für Mean-Reversion-Strategien.",
    calculation="CRSI = (RSI + Streak RSI + Percent Rank) / 3",
    strengths=[
        "Speziell für Mean-Reversion optimiert",
        "Kombiniert mehrere Aspekte",
        "Hohe Trefferquote bei Extremwerten"
    ],
    weaknesses=[
        "Komplexe Berechnung",
        "Nicht für Trendfolge geeignet",
        "Weniger verbreitet"
    ],
    use_cases=[
        "Mean-Reversion Trading",
        "Extreme Überkauft/Überverkauft-Zustände",
        "Short-Term Reversals"
    ],
    default_params={"rsi_period": 3, "streak_period": 2, "rank_period": 100}
)

# =============================================================================
# VOLATILITÄT
# =============================================================================

BBANDS = IndicatorConfig(
    id="bbands",
    name="Bollinger Bands",
    twelvedata_name="bbands",
    easyinsight_name="bbands",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Volatilitätsbänder um einen Moving Average basierend auf Standardabweichung.",
    calculation="Upper = SMA + (k × StdDev); Lower = SMA - (k × StdDev); k meist 2",
    strengths=[
        "Zeigt Volatilität visuell",
        "Dynamische Support/Resistance",
        "Erkennt Squeeze/Expansion",
        "Vielseitig einsetzbar"
    ],
    weaknesses=[
        "Bänder selbst sind keine Kauf/Verkauf-Signale",
        "Preis kann an Bändern entlanglaufen",
        "Erfordert Bestätigung durch andere Indikatoren"
    ],
    use_cases=[
        "Volatilitäts-Analyse",
        "Squeeze-Erkennung (bevorstehende Bewegung)",
        "Mean-Reversion (Rückkehr zur Mitte)",
        "Breakout-Trading"
    ],
    default_params={"period": 20, "std_dev": 2}
)

ATR = IndicatorConfig(
    id="atr",
    name="Average True Range (ATR)",
    twelvedata_name="atr",
    easyinsight_name="atr",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Misst die durchschnittliche Volatilität unter Berücksichtigung von Gaps.",
    calculation="ATR = Smoothed Average von True Range; TR = Max(H-L, |H-C_prev|, |L-C_prev|)",
    strengths=[
        "Objektive Volatilitätsmessung",
        "Berücksichtigt Gaps",
        "Universell für Stop-Loss/Position-Sizing",
        "Standard für Risk Management"
    ],
    weaknesses=[
        "Zeigt keine Richtung",
        "Absolute Werte (nicht normalisiert)",
        "Reagiert verzögert auf Volatilitätsänderungen"
    ],
    use_cases=[
        "Stop-Loss Berechnung (z.B. 2× ATR)",
        "Position Sizing",
        "Volatilitäts-Breakout-Strategien",
        "Trailing Stops"
    ],
    default_params={"period": 14}
)

NATR = IndicatorConfig(
    id="natr",
    name="Normalized Average True Range (NATR)",
    twelvedata_name="natr",
    easyinsight_name=None,
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="ATR als Prozentsatz des Preises für Vergleichbarkeit zwischen Assets.",
    calculation="NATR = (ATR / Close) × 100",
    strengths=[
        "Vergleichbar zwischen verschiedenen Assets",
        "Zeigt relative Volatilität",
        "Gut für Screening"
    ],
    weaknesses=[
        "Weniger verbreitet als ATR",
        "Prozent-Werte weniger intuitiv für Stops"
    ],
    use_cases=[
        "Asset-Vergleich nach Volatilität",
        "Volatilitäts-Screening",
        "Normalisierte Risk-Analyse"
    ],
    default_params={"period": 14}
)

TRANGE = IndicatorConfig(
    id="trange",
    name="True Range",
    twelvedata_name="trange",
    easyinsight_name=None,
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Die wahre Spanne einer Kerze inkl. Gaps zur Vorgängerkerze.",
    calculation="TR = Max(High-Low, |High-Close_prev|, |Low-Close_prev|)",
    strengths=[
        "Zeigt aktuelle Volatilität genau",
        "Berücksichtigt Gaps",
        "Basis für ATR"
    ],
    weaknesses=[
        "Kein Durchschnitt (einzelne Werte)",
        "Sehr volatil",
        "Meist wird ATR bevorzugt"
    ],
    use_cases=[
        "Aktuelle Kerzen-Volatilität",
        "Gap-Erkennung",
        "Input für ATR"
    ],
    default_params={}
)

PERCENT_B = IndicatorConfig(
    id="percent_b",
    name="Percent B (%B)",
    twelvedata_name="percent_b",
    easyinsight_name=None,
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Zeigt Position des Preises innerhalb der Bollinger Bands (0-1 normalisiert).",
    calculation="%B = (Close - Lower Band) / (Upper Band - Lower Band)",
    strengths=[
        "Normalisierte Darstellung (0-1)",
        "Zeigt relative Position zu Bändern",
        "Gut für Vergleiche und ML"
    ],
    weaknesses=[
        "Kann >1 oder <0 werden",
        "Erfordert BBands-Verständnis"
    ],
    use_cases=[
        "Overbought/Oversold (>1 / <0)",
        "ML-Feature (normalisiert)",
        "Crossover mit 0.5 Linie"
    ],
    default_params={"period": 20, "std_dev": 2}
)

# =============================================================================
# VOLUMEN
# =============================================================================

OBV = IndicatorConfig(
    id="obv",
    name="On-Balance Volume (OBV)",
    twelvedata_name="obv",
    easyinsight_name=None,
    category=IndicatorCategory.VOLUME,
    enabled=True,
    description="Kumulativer Volumenindikator, addiert/subtrahiert Volumen basierend auf Preisrichtung.",
    calculation="OBV = OBV_prev + Volume (wenn Close > Close_prev) oder - Volume",
    strengths=[
        "Zeigt Akkumulation/Distribution",
        "Führender Indikator (Volumen vor Preis)",
        "Einfaches Konzept"
    ],
    weaknesses=[
        "Absolute Werte schwer interpretierbar",
        "Kann durch einzelne Volumen-Spikes verzerrt werden",
        "Trend des OBV wichtiger als Wert"
    ],
    use_cases=[
        "Trendbestätigung (OBV-Trend = Preis-Trend)",
        "Divergenz-Analyse",
        "Breakout-Bestätigung"
    ],
    default_params={}
)

AD = IndicatorConfig(
    id="ad",
    name="Accumulation/Distribution Line (A/D)",
    twelvedata_name="ad",
    easyinsight_name=None,
    category=IndicatorCategory.VOLUME,
    enabled=True,
    description="Volumen-basierter Indikator, gewichtet nach Kerzenposition.",
    calculation="A/D = A/D_prev + ((Close-Low) - (High-Close)) / (High-Low) × Volume",
    strengths=[
        "Berücksichtigt Kerzenstruktur",
        "Zeigt Smart Money Flow",
        "Divergenzen sehr aussagekräftig"
    ],
    weaknesses=[
        "Kann durch Gaps verzerrt werden",
        "Absolute Werte weniger wichtig als Trend"
    ],
    use_cases=[
        "Akkumulation/Distribution erkennen",
        "Divergenz-Analyse",
        "Trendbestätigung"
    ],
    default_params={}
)

ADOSC = IndicatorConfig(
    id="adosc",
    name="Chaikin A/D Oscillator (ADOSC)",
    twelvedata_name="adosc",
    easyinsight_name=None,
    category=IndicatorCategory.VOLUME,
    enabled=True,
    description="MACD der A/D-Linie für Momentum des Geldflusses.",
    calculation="ADOSC = EMA(A/D, 3) - EMA(A/D, 10)",
    strengths=[
        "Zeigt Momentum des Geldflusses",
        "Oszillator-Format leichter lesbar",
        "Führende Signale möglich"
    ],
    weaknesses=[
        "Kann falsche Signale in Seitwärtsmärkten geben",
        "Abhängig von A/D-Qualität"
    ],
    use_cases=[
        "Zero-Line Crossovers",
        "Divergenz-Analyse",
        "Momentum des Geldflusses"
    ],
    default_params={"fast_period": 3, "slow_period": 10}
)

# =============================================================================
# PRICE TRANSFORM
# =============================================================================

AVGPRICE = IndicatorConfig(
    id="avgprice",
    name="Average Price",
    twelvedata_name="avgprice",
    easyinsight_name=None,
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="Durchschnitt von Open, High, Low, Close.",
    calculation="AvgPrice = (O + H + L + C) / 4",
    strengths=[
        "Einfache Preisglättung",
        "Berücksichtigt alle OHLC-Werte",
        "Schnelle Berechnung"
    ],
    weaknesses=[
        "Alle Komponenten gleich gewichtet",
        "Weniger aussagekräftig als andere Transforms"
    ],
    use_cases=[
        "Input für andere Indikatoren",
        "Alternative zu Close",
        "Preisglättung"
    ],
    default_params={}
)

MEDPRICE = IndicatorConfig(
    id="medprice",
    name="Median Price",
    twelvedata_name="medprice",
    easyinsight_name=None,
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="Durchschnitt von High und Low (Mittelpunkt der Kerze).",
    calculation="MedPrice = (H + L) / 2",
    strengths=[
        "Zeigt Kerzen-Mittelpunkt",
        "Unabhängig von Open/Close",
        "Gut für Range-Analyse"
    ],
    weaknesses=[
        "Ignoriert Open/Close",
        "Weniger aussagekräftig für Richtung"
    ],
    use_cases=[
        "Input für Indikatoren (z.B. CCI)",
        "Pivot-Berechnungen",
        "Range-Mittelpunkt"
    ],
    default_params={}
)

TYPPRICE = IndicatorConfig(
    id="typprice",
    name="Typical Price",
    twelvedata_name="typprice",
    easyinsight_name=None,
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="Durchschnitt von High, Low und Close (typischer Preis).",
    calculation="TypPrice = (H + L + C) / 3",
    strengths=[
        "Standard für viele Indikatoren",
        "Close-gewichtet durch Mittelung",
        "Weit verbreitet"
    ],
    weaknesses=[
        "Ignoriert Open",
        "Kann bei Gaps irreführend sein"
    ],
    use_cases=[
        "Input für CCI, MFI etc.",
        "VWAP-Berechnung",
        "Standard-Preisrepräsentant"
    ],
    default_params={}
)

WCLPRICE = IndicatorConfig(
    id="wclprice",
    name="Weighted Close Price",
    twelvedata_name="wclprice",
    easyinsight_name=None,
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="Gewichteter Durchschnitt mit doppelter Close-Gewichtung.",
    calculation="WCLPrice = (H + L + 2×C) / 4",
    strengths=[
        "Betont Schlusskurs",
        "Stabiler als nur Close",
        "Berücksichtigt Range"
    ],
    weaknesses=[
        "Willkürliche Gewichtung",
        "Weniger verbreitet"
    ],
    use_cases=[
        "Alternative zu Close",
        "Input für Indikatoren",
        "Gewichtete Preisdarstellung"
    ],
    default_params={}
)

# =============================================================================
# ML FEATURES
# =============================================================================

LINEARREG_SLOPE = IndicatorConfig(
    id="linearregslope",
    name="Linear Regression Slope",
    twelvedata_name="linearregslope",
    easyinsight_name=None,
    category=IndicatorCategory.ML_FEATURES,
    enabled=True,
    description="Steigung der linearen Regression als numerisches Trend-Mass.",
    calculation="Slope der Least-Squares-Regression über n Perioden",
    strengths=[
        "Numerische Trendstärke und -richtung",
        "Ideal für ML-Modelle",
        "Statistisch fundiert"
    ],
    weaknesses=[
        "Reagiert verzögert auf Trendwechsel",
        "Empfindlich gegenüber Ausreissern",
        "Weniger intuitiv"
    ],
    use_cases=[
        "ML-Feature für Trendrichtung",
        "Quantitative Trendanalyse",
        "Screening nach Trendstärke"
    ],
    default_params={"period": 14}
)

HT_TRENDMODE = IndicatorConfig(
    id="ht_trendmode",
    name="Hilbert Transform - Trend vs. Cycle Mode",
    twelvedata_name="ht_trendmode",
    easyinsight_name=None,
    category=IndicatorCategory.ML_FEATURES,
    enabled=True,
    description="Klassifiziert Markt als Trend (1) oder Range/Zyklus (0).",
    calculation="Basiert auf Hilbert-Transformation der Preisdaten",
    strengths=[
        "Binäre Klassifikation Trend/Range",
        "Automatische Erkennung",
        "Wissenschaftlich fundiert (DSP)"
    ],
    weaknesses=[
        "Komplexe Mathematik",
        "Kann bei Übergängen flackern",
        "Nicht intuitiv zu verstehen"
    ],
    use_cases=[
        "Strategie-Auswahl (Trend vs. Mean-Reversion)",
        "ML-Feature",
        "Automatische Marktklassifikation"
    ],
    default_params={}
)

# =============================================================================
# EASYINSIGHT SPECIFIC
# =============================================================================

STRENGTH = IndicatorConfig(
    id="strength",
    name="Multi-Timeframe Strength",
    twelvedata_name=None,
    easyinsight_name="strength",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Proprietärer EasyInsight-Indikator, der die Trendstärke über mehrere Timeframes (4H, D1, W1) misst.",
    calculation="Proprietäre Berechnung basierend auf Preis-Momentum und Trend-Konsistenz über H4, D1, W1",
    strengths=[
        "Multi-Timeframe-Analyse in einem Wert",
        "Zeigt Trend-Alignment über Zeitebenen",
        "Gut für Confluence-Trading",
        "Proprietärer EasyInsight-Algorithmus"
    ],
    weaknesses=[
        "Nur über EasyInsight API verfügbar",
        "Proprietäre Berechnung (Black-Box)",
        "Abhängig von EasyInsight-Infrastruktur"
    ],
    use_cases=[
        "Multi-Timeframe Trend-Bestätigung",
        "Confluence-Signale",
        "Filter für Trade-Qualität",
        "Positionsgrössen-Anpassung"
    ],
    default_params={"timeframes": ["4h", "1d", "1w"]}
)

CCI_EASYINSIGHT = IndicatorConfig(
    id="cci_easyinsight",
    name="Commodity Channel Index (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="cci",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="EasyInsight-Version des CCI mit vorberechneten Werten.",
    calculation="CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation)",
    strengths=[
        "Vorberechnet und optimiert",
        "Konsistent mit anderen EasyInsight-Indikatoren",
        "Schneller Zugriff ohne API-Limits"
    ],
    weaknesses=[
        "Nur H1 Timeframe",
        "Feste Parameter (nicht anpassbar)"
    ],
    use_cases=[
        "Overbought/Oversold (±100)",
        "Trend-Trading",
        "Divergenz-Analyse"
    ],
    default_params={"period": 20}
)

# Strength Timeframe-spezifisch
STRENGTH_4H = IndicatorConfig(
    id="strength_4h",
    name="Currency Strength 4H",
    twelvedata_name=None,
    easyinsight_name="strength_4h",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="EasyInsight Currency Strength auf 4-Stunden-Basis.",
    calculation="Proprietäre Berechnung der relativen Währungsstärke über 4H Timeframe",
    strengths=[
        "Kurzfristige Stärke-Analyse",
        "Gut für Intraday-Trading",
        "Zeigt relative Währungsstärke"
    ],
    weaknesses=[
        "Nur über EasyInsight verfügbar",
        "Höheres Rauschen als längere Timeframes"
    ],
    use_cases=[
        "Intraday Währungspaar-Auswahl",
        "Kurzfristige Trend-Bestätigung",
        "Momentum-Filter"
    ],
    default_params={"timeframe": "4h"}
)

STRENGTH_1D = IndicatorConfig(
    id="strength_1d",
    name="Currency Strength Daily",
    twelvedata_name=None,
    easyinsight_name="strength_1d",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="EasyInsight Currency Strength auf Tagesbasis.",
    calculation="Proprietäre Berechnung der relativen Währungsstärke über D1 Timeframe",
    strengths=[
        "Stabilere Signale als 4H",
        "Gute Balance zwischen Reaktion und Stabilität",
        "Standard für Swing-Trading"
    ],
    weaknesses=[
        "Nur über EasyInsight verfügbar",
        "Tägliche Aktualisierung"
    ],
    use_cases=[
        "Swing-Trading Währungsauswahl",
        "Daily Trend-Bestätigung",
        "Multi-Timeframe-Analyse"
    ],
    default_params={"timeframe": "1d"}
)

STRENGTH_1W = IndicatorConfig(
    id="strength_1w",
    name="Currency Strength Weekly",
    twelvedata_name=None,
    easyinsight_name="strength_1w",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="EasyInsight Currency Strength auf Wochenbasis.",
    calculation="Proprietäre Berechnung der relativen Währungsstärke über W1 Timeframe",
    strengths=[
        "Langfristige Trendrichtung",
        "Wenig Rauschen",
        "Ideal für Positions-Trading"
    ],
    weaknesses=[
        "Nur über EasyInsight verfügbar",
        "Langsame Reaktion auf Änderungen"
    ],
    use_cases=[
        "Langfristige Positionierung",
        "Makro-Trend-Analyse",
        "Strategische Währungsauswahl"
    ],
    default_params={"timeframe": "1w"}
)

# ATR/Range EasyInsight
ATR_D1 = IndicatorConfig(
    id="atr_d1",
    name="ATR Daily (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="atr_d1",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="EasyInsight vorberechneter ATR auf Tagesbasis.",
    calculation="ATR = Wilder's Smoothed Average of True Range über 14 Tage",
    strengths=[
        "Vorberechnet und optimiert",
        "Tägliche Volatilitätsreferenz",
        "Gut für Position Sizing"
    ],
    weaknesses=[
        "Nur D1 Timeframe",
        "Feste 14-Perioden-Einstellung"
    ],
    use_cases=[
        "Stop-Loss-Berechnung",
        "Position Sizing",
        "Volatilitäts-Filter"
    ],
    default_params={"period": 14}
)

RANGE_D1 = IndicatorConfig(
    id="range_d1",
    name="Range Daily (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="range_d1",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Tägliche Handelsspanne (High - Low) von EasyInsight.",
    calculation="Range = High_D1 - Low_D1",
    strengths=[
        "Einfache Volatilitätsmessung",
        "Direkter Marktüberblick",
        "Gut für Breakout-Strategien"
    ],
    weaknesses=[
        "Nur absoluter Wert",
        "Nicht normalisiert"
    ],
    use_cases=[
        "Breakout-Erkennung",
        "Volatilitäts-Screening",
        "Range-Trading-Setup"
    ],
    default_params={}
)

ATR_PCT_D1 = IndicatorConfig(
    id="atr_pct_d1",
    name="ATR Prozent Daily (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="atr_pct_d1",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="ATR als Prozentsatz des Preises für Vergleichbarkeit zwischen Assets.",
    calculation="ATR% = (ATR / Close) × 100",
    strengths=[
        "Vergleichbar zwischen Assets",
        "Normalisierte Volatilität",
        "Gut für Cross-Asset-Analyse"
    ],
    weaknesses=[
        "Nur D1 Timeframe",
        "Abhängig von ATR-Berechnung"
    ],
    use_cases=[
        "Asset-Vergleich nach Volatilität",
        "Risiko-normiertes Position Sizing",
        "Volatilitäts-Ranking"
    ],
    default_params={}
)

# Support/Resistance EasyInsight
S1_LEVEL_M5 = IndicatorConfig(
    id="s1_level_m5",
    name="Support Level M5 (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="s1_level_m5",
    category=IndicatorCategory.PATTERN,
    enabled=True,
    description="EasyInsight berechneter Support-Level auf M5 Basis.",
    calculation="Proprietäre S/R-Berechnung basierend auf Preiscluster",
    strengths=[
        "Automatisch aktualisiert",
        "Kurzfristig relevant",
        "Für Scalping geeignet"
    ],
    weaknesses=[
        "Nur M5 Timeframe",
        "Kann schnell invalidiert werden"
    ],
    use_cases=[
        "Scalping Entry-Points",
        "Kurzfristiger Support",
        "Stop-Loss-Platzierung"
    ],
    default_params={}
)

R1_LEVEL_M5 = IndicatorConfig(
    id="r1_level_m5",
    name="Resistance Level M5 (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="r1_level_m5",
    category=IndicatorCategory.PATTERN,
    enabled=True,
    description="EasyInsight berechneter Resistance-Level auf M5 Basis.",
    calculation="Proprietäre S/R-Berechnung basierend auf Preiscluster",
    strengths=[
        "Automatisch aktualisiert",
        "Kurzfristig relevant",
        "Für Scalping geeignet"
    ],
    weaknesses=[
        "Nur M5 Timeframe",
        "Kann schnell invalidiert werden"
    ],
    use_cases=[
        "Scalping Take-Profit",
        "Kurzfristiger Widerstand",
        "Breakout-Ziele"
    ],
    default_params={}
)

# OHLC EasyInsight
OHLC_M15 = IndicatorConfig(
    id="ohlc_m15",
    name="OHLC M15 (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="m15_ohlc",
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="EasyInsight M15 Kerzendaten (Open, High, Low, Close).",
    calculation="Aggregierte OHLC-Werte über 15-Minuten-Intervalle",
    strengths=[
        "Schneller Zugriff auf M15 Daten",
        "Vorberechnet und konsistent",
        "Keine zusätzliche Aggregation nötig"
    ],
    weaknesses=[
        "Fester Timeframe",
        "Keine historische Tiefe"
    ],
    use_cases=[
        "M15 Chart-Analyse",
        "Multi-Timeframe-Daten",
        "Schnelle Kursabfrage"
    ],
    default_params={"fields": ["m15_open", "m15_high", "m15_low", "m15_close"]}
)

OHLC_H1 = IndicatorConfig(
    id="ohlc_h1",
    name="OHLC H1 (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="h1_ohlc",
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="EasyInsight H1 Kerzendaten (Open, High, Low, Close).",
    calculation="Aggregierte OHLC-Werte über 1-Stunden-Intervalle",
    strengths=[
        "Standard-Timeframe für Intraday",
        "Vorberechnet und konsistent",
        "Gut für Swing-Trading"
    ],
    weaknesses=[
        "Fester Timeframe",
        "Keine historische Tiefe"
    ],
    use_cases=[
        "H1 Chart-Analyse",
        "Intraday-Trading",
        "Trend-Identifikation"
    ],
    default_params={"fields": ["h1_open", "h1_high", "h1_low", "h1_close"]}
)

OHLC_D1 = IndicatorConfig(
    id="ohlc_d1",
    name="OHLC D1 (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="d1_ohlc",
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="EasyInsight D1 Kerzendaten (Open, High, Low, Close).",
    calculation="Aggregierte OHLC-Werte über Tages-Intervalle",
    strengths=[
        "Tägliche Referenzwerte",
        "Wichtig für Gap-Analyse",
        "Stabile Signale"
    ],
    weaknesses=[
        "Nur End-of-Day Werte",
        "Keine Intraday-Details"
    ],
    use_cases=[
        "Daily Chart-Analyse",
        "Gap-Trading",
        "Swing-Trading"
    ],
    default_params={"fields": ["d1_open", "d1_high", "d1_low", "d1_close"]}
)

# Preis EasyInsight
BID_ASK = IndicatorConfig(
    id="bid_ask",
    name="Bid/Ask (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="bid_ask",
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="Aktuelle Bid und Ask Kurse von EasyInsight.",
    calculation="Echtzeit Bid/Ask aus MT5 Feed",
    strengths=[
        "Echtzeit-Kurse",
        "Für präzise Order-Ausführung",
        "Zeigt Marktliquidität"
    ],
    weaknesses=[
        "Nur Snapshot (nicht historisch)",
        "Kann sich schnell ändern"
    ],
    use_cases=[
        "Order-Ausführung",
        "Spread-Monitoring",
        "Entry/Exit-Timing"
    ],
    default_params={"fields": ["bid", "ask"]}
)

SPREAD = IndicatorConfig(
    id="spread",
    name="Spread (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="spread",
    category=IndicatorCategory.PRICE_TRANSFORM,
    enabled=True,
    description="Aktueller Spread (Ask - Bid) absolut und prozentual.",
    calculation="Spread = Ask - Bid; Spread% = Spread / Mid × 100",
    strengths=[
        "Liquiditätsindikator",
        "Trading-Kosten-Übersicht",
        "Volatilitäts-Proxy"
    ],
    weaknesses=[
        "Variiert stark nach Tageszeit",
        "Broker-abhängig"
    ],
    use_cases=[
        "Kostenanalyse",
        "Liquiditäts-Check",
        "Beste Trading-Zeiten finden"
    ],
    default_params={"fields": ["spread", "spread_pct"]}
)

# ADX Komponenten EasyInsight
ADX_PLUSDI = IndicatorConfig(
    id="adx_plusdi",
    name="+DI (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="adx_plusdi",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Positive Directional Indicator von EasyInsight ADX.",
    calculation="+DI = 100 × Smoothed +DM / ATR",
    strengths=[
        "Zeigt Aufwärtsdruck",
        "Teil des ADX-Systems",
        "Crossover-Signale mit -DI"
    ],
    weaknesses=[
        "Nur mit -DI sinnvoll",
        "Lagging Indicator"
    ],
    use_cases=[
        "Bullish Trend-Stärke",
        "+DI/-DI Crossovers",
        "Trend-Richtung"
    ],
    default_params={}
)

ADX_MINUSDI = IndicatorConfig(
    id="adx_minusdi",
    name="-DI (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="adx_minusdi",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Negative Directional Indicator von EasyInsight ADX.",
    calculation="-DI = 100 × Smoothed -DM / ATR",
    strengths=[
        "Zeigt Abwärtsdruck",
        "Teil des ADX-Systems",
        "Crossover-Signale mit +DI"
    ],
    weaknesses=[
        "Nur mit +DI sinnvoll",
        "Lagging Indicator"
    ],
    use_cases=[
        "Bearish Trend-Stärke",
        "+DI/-DI Crossovers",
        "Trend-Richtung"
    ],
    default_params={}
)

# Bollinger Bands Komponenten EasyInsight
BB_BASE = IndicatorConfig(
    id="bb_base",
    name="Bollinger Mittellinie (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="bb_base",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Mittellinie (SMA) der EasyInsight Bollinger Bands.",
    calculation="BB_Base = SMA(Close, 20)",
    strengths=[
        "Dynamischer Durchschnitt",
        "Mean-Reversion-Referenz",
        "Trend-Indikator"
    ],
    weaknesses=[
        "Feste 20-Perioden",
        "Keine Parameter-Anpassung"
    ],
    use_cases=[
        "Mean-Reversion-Ziel",
        "Trend-Richtung",
        "Mittelfristiger Durchschnitt"
    ],
    default_params={}
)

BB_UPPER = IndicatorConfig(
    id="bb_upper",
    name="Bollinger Oberes Band (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="bb_upper",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Oberes Band der EasyInsight Bollinger Bands.",
    calculation="BB_Upper = SMA + (2 × StdDev)",
    strengths=[
        "Dynamischer Widerstand",
        "Volatilitäts-angepasst",
        "Overbought-Referenz"
    ],
    weaknesses=[
        "Feste Parameter (20, 2)",
        "Kein eigenständiges Signal"
    ],
    use_cases=[
        "Overbought-Zone",
        "Breakout-Ziele",
        "Volatilitäts-Expansion"
    ],
    default_params={}
)

BB_LOWER = IndicatorConfig(
    id="bb_lower",
    name="Bollinger Unteres Band (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="bb_lower",
    category=IndicatorCategory.VOLATILITY,
    enabled=True,
    description="Unteres Band der EasyInsight Bollinger Bands.",
    calculation="BB_Lower = SMA - (2 × StdDev)",
    strengths=[
        "Dynamischer Support",
        "Volatilitäts-angepasst",
        "Oversold-Referenz"
    ],
    weaknesses=[
        "Feste Parameter (20, 2)",
        "Kein eigenständiges Signal"
    ],
    use_cases=[
        "Oversold-Zone",
        "Support-Level",
        "Mean-Reversion-Entry"
    ],
    default_params={}
)

# Ichimoku Komponenten EasyInsight
ICHIMOKU_TENKAN = IndicatorConfig(
    id="ichimoku_tenkan",
    name="Ichimoku Tenkan-sen (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="ichimoku_tenkan",
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Conversion Line des EasyInsight Ichimoku (9-Perioden Mittelwert).",
    calculation="Tenkan = (Highest High + Lowest Low) / 2 über 9 Perioden",
    strengths=[
        "Schnelle Trend-Linie",
        "Kurzfristiges Momentum",
        "Crossover mit Kijun"
    ],
    weaknesses=[
        "Kurze Periode (volatil)",
        "Nur Teil des Systems"
    ],
    use_cases=[
        "Kurzfristiger Trend",
        "TK-Crossover-Signale",
        "Momentum-Indikator"
    ],
    default_params={"period": 9}
)

ICHIMOKU_KIJUN = IndicatorConfig(
    id="ichimoku_kijun",
    name="Ichimoku Kijun-sen (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="ichimoku_kijun",
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Base Line des EasyInsight Ichimoku (26-Perioden Mittelwert).",
    calculation="Kijun = (Highest High + Lowest Low) / 2 über 26 Perioden",
    strengths=[
        "Mittelfristige Trend-Linie",
        "Starker Support/Resistance",
        "Pullback-Ziel"
    ],
    weaknesses=[
        "Langsamere Reaktion",
        "Nur Teil des Systems"
    ],
    use_cases=[
        "Mittelfristiger Trend",
        "Pullback-Entry",
        "Stop-Loss-Referenz"
    ],
    default_params={"period": 26}
)

ICHIMOKU_SENKOUA = IndicatorConfig(
    id="ichimoku_senkoua",
    name="Ichimoku Senkou Span A (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="ichimoku_senkoua",
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Leading Span A der EasyInsight Ichimoku Cloud.",
    calculation="Senkou A = (Tenkan + Kijun) / 2, 26 Perioden voraus projiziert",
    strengths=[
        "Vorausschauend",
        "Cloud-Grenze (schnell)",
        "Trend-Wechsel-Signal"
    ],
    weaknesses=[
        "Projiziert, nicht aktuell",
        "Nur mit Senkou B sinnvoll"
    ],
    use_cases=[
        "Cloud-Obergrenze",
        "Zukünftiger Support",
        "Trend-Projektion"
    ],
    default_params={}
)

ICHIMOKU_SENKOUB = IndicatorConfig(
    id="ichimoku_senkoub",
    name="Ichimoku Senkou Span B (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="ichimoku_senkoub",
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Leading Span B der EasyInsight Ichimoku Cloud.",
    calculation="Senkou B = (Highest + Lowest) / 2 über 52 Perioden, 26 voraus",
    strengths=[
        "Langfristige Cloud-Grenze",
        "Starker S/R-Level",
        "Flacher = starker Trend"
    ],
    weaknesses=[
        "Sehr langsam",
        "Nur mit Senkou A sinnvoll"
    ],
    use_cases=[
        "Cloud-Untergrenze",
        "Langfristiger S/R",
        "Trend-Stärke"
    ],
    default_params={}
)

ICHIMOKU_CHIKOU = IndicatorConfig(
    id="ichimoku_chikou",
    name="Ichimoku Chikou Span (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="ichimoku_chikou",
    category=IndicatorCategory.TREND_FILTER,
    enabled=True,
    description="Lagging Span des EasyInsight Ichimoku (Close 26 Perioden zurück).",
    calculation="Chikou = Close, 26 Perioden zurückversetzt",
    strengths=[
        "Momentum-Bestätigung",
        "Trend-Validierung",
        "Historischer Vergleich"
    ],
    weaknesses=[
        "Rückwärts blickend",
        "Zusätzliche Komplexität"
    ],
    use_cases=[
        "Trend-Bestätigung",
        "Signal-Validierung",
        "Momentum-Check"
    ],
    default_params={}
)

# MA EasyInsight
MA_10 = IndicatorConfig(
    id="ma_10",
    name="Moving Average 10 (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="ma_10",
    category=IndicatorCategory.TREND,
    enabled=True,
    description="EasyInsight 10-Perioden Moving Average.",
    calculation="MA = SMA(Close, 10)",
    strengths=[
        "Schneller Trend-Indikator",
        "Gut für kurzfristiges Trading",
        "Dynamischer Support"
    ],
    weaknesses=[
        "Kurze Periode (volatil)",
        "Viele Whipsaws"
    ],
    use_cases=[
        "Kurzfristiger Trend",
        "Schnelle Crossovers",
        "Intraday-Trading"
    ],
    default_params={"period": 10}
)

# MACD Komponenten EasyInsight
MACD_SIGNAL = IndicatorConfig(
    id="macd_signal",
    name="MACD Signal Line (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="macd_signal",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Signal-Linie des EasyInsight MACD (9-Perioden EMA).",
    calculation="MACD_Signal = EMA(MACD, 9)",
    strengths=[
        "Crossover-Trigger",
        "Glättet MACD-Signale",
        "Timing-Indikator"
    ],
    weaknesses=[
        "Lagging zum MACD",
        "Kann verspätet reagieren"
    ],
    use_cases=[
        "MACD-Crossover",
        "Signal-Bestätigung",
        "Entry/Exit-Timing"
    ],
    default_params={}
)

# Stochastic Komponenten EasyInsight
STO_SIGNAL = IndicatorConfig(
    id="sto_signal",
    name="Stochastic Signal (%D) (EasyInsight)",
    twelvedata_name=None,
    easyinsight_name="sto_signal",
    category=IndicatorCategory.MOMENTUM,
    enabled=True,
    description="Signal-Linie (%D) des EasyInsight Stochastic.",
    calculation="%D = SMA(%K, 3)",
    strengths=[
        "Crossover-Trigger",
        "Glättet %K-Signale",
        "Bestätigt Momentum"
    ],
    weaknesses=[
        "Lagging zu %K",
        "Kann verspätet reagieren"
    ],
    use_cases=[
        "%K/%D-Crossover",
        "Signal-Glättung",
        "Entry-Timing"
    ],
    default_params={}
)

# =============================================================================
# PATTERN
# =============================================================================

PIVOT_POINTS_HL = IndicatorConfig(
    id="pivot_points_hl",
    name="Pivot Points High/Low",
    twelvedata_name="pivot_points_hl",
    easyinsight_name=None,
    category=IndicatorCategory.PATTERN,
    enabled=True,
    description="Identifiziert signifikante Hochs und Tiefs für Support/Resistance.",
    calculation="Erkennt lokale Extrempunkte basierend auf umgebenden Kerzen",
    strengths=[
        "Objektive S/R-Identifikation",
        "Historische Pivots als Referenz",
        "Automatische Erkennung"
    ],
    weaknesses=[
        "Lagging (Bestätigung benötigt)",
        "Parameter-abhängig",
        "Nicht alle Pivots sind gleich wichtig"
    ],
    use_cases=[
        "Support/Resistance-Levels",
        "Entry/Exit-Punkte",
        "Chart-Pattern-Analyse"
    ],
    default_params={"left_bars": 2, "right_bars": 2}
)

# =============================================================================
# REGISTRY
# =============================================================================

# Alle Indikatoren in einem Dictionary
INDICATORS_REGISTRY: Dict[str, IndicatorConfig] = {
    # Trend / Moving Averages
    "sma": SMA,
    "ema": EMA,
    "wma": WMA,
    "dema": DEMA,
    "tema": TEMA,
    "kama": KAMA,
    "mama": MAMA,
    "t3": T3,
    "trima": TRIMA,

    # Trend Filter
    "vwap": VWAP,
    "supertrend": SUPERTREND,
    "ichimoku": ICHIMOKU,
    "sar": SAR,

    # Momentum
    "rsi": RSI,
    "macd": MACD,
    "stoch": STOCH,
    "stochrsi": STOCHRSI,
    "willr": WILLIAMS_R,
    "cci": CCI,
    "cmo": CMO,
    "roc": ROC,
    "mom": MOM,
    "ppo": PPO,
    "apo": APO,
    "aroon": AROON,
    "aroonosc": AROON_OSC,
    "bop": BOP,
    "mfi": MFI,
    "dx": DX,
    "adx": ADX,
    "adxr": ADXR,
    "plus_di": PLUS_DI,
    "minus_di": MINUS_DI,
    "crsi": CRSI,

    # Volatilität
    "bbands": BBANDS,
    "atr": ATR,
    "natr": NATR,
    "trange": TRANGE,
    "percent_b": PERCENT_B,

    # Volumen
    "obv": OBV,
    "ad": AD,
    "adosc": ADOSC,

    # Price Transform
    "avgprice": AVGPRICE,
    "medprice": MEDPRICE,
    "typprice": TYPPRICE,
    "wclprice": WCLPRICE,

    # ML Features
    "linearregslope": LINEARREG_SLOPE,
    "ht_trendmode": HT_TRENDMODE,

    # EasyInsight Specific - General
    "strength": STRENGTH,
    "cci_easyinsight": CCI_EASYINSIGHT,

    # EasyInsight Specific - Strength Timeframes
    "strength_4h": STRENGTH_4H,
    "strength_1d": STRENGTH_1D,
    "strength_1w": STRENGTH_1W,

    # EasyInsight Specific - ATR/Range
    "atr_d1": ATR_D1,
    "range_d1": RANGE_D1,
    "atr_pct_d1": ATR_PCT_D1,

    # EasyInsight Specific - Support/Resistance
    "s1_level_m5": S1_LEVEL_M5,
    "r1_level_m5": R1_LEVEL_M5,

    # EasyInsight Specific - OHLC
    "ohlc_m15": OHLC_M15,
    "ohlc_h1": OHLC_H1,
    "ohlc_d1": OHLC_D1,

    # EasyInsight Specific - Price
    "bid_ask": BID_ASK,
    "spread": SPREAD,

    # EasyInsight Specific - ADX Components
    "adx_plusdi": ADX_PLUSDI,
    "adx_minusdi": ADX_MINUSDI,

    # EasyInsight Specific - Bollinger Components
    "bb_base": BB_BASE,
    "bb_upper": BB_UPPER,
    "bb_lower": BB_LOWER,

    # EasyInsight Specific - Ichimoku Components
    "ichimoku_tenkan": ICHIMOKU_TENKAN,
    "ichimoku_kijun": ICHIMOKU_KIJUN,
    "ichimoku_senkoua": ICHIMOKU_SENKOUA,
    "ichimoku_senkoub": ICHIMOKU_SENKOUB,
    "ichimoku_chikou": ICHIMOKU_CHIKOU,

    # EasyInsight Specific - MA
    "ma_10": MA_10,

    # EasyInsight Specific - MACD/Stochastic Components
    "macd_signal": MACD_SIGNAL,
    "sto_signal": STO_SIGNAL,

    # Pattern
    "pivot_points_hl": PIVOT_POINTS_HL,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_indicator(indicator_id: str) -> Optional[IndicatorConfig]:
    """Gibt einen Indikator anhand seiner ID zurück."""
    return INDICATORS_REGISTRY.get(indicator_id.lower())


def get_all_indicators() -> List[IndicatorConfig]:
    """Gibt alle registrierten Indikatoren zurück."""
    return list(INDICATORS_REGISTRY.values())


def get_enabled_indicators() -> List[IndicatorConfig]:
    """Gibt alle aktiven Indikatoren zurück."""
    return [ind for ind in INDICATORS_REGISTRY.values() if ind.enabled]


def get_indicators_by_category(category: IndicatorCategory) -> List[IndicatorConfig]:
    """Gibt alle Indikatoren einer Kategorie zurück."""
    return [ind for ind in INDICATORS_REGISTRY.values() if ind.category == category]


def get_twelvedata_indicators() -> List[IndicatorConfig]:
    """Gibt alle Indikatoren mit TwelveData-Unterstützung zurück."""
    return [ind for ind in INDICATORS_REGISTRY.values() if ind.twelvedata_name]


def get_easyinsight_indicators() -> List[IndicatorConfig]:
    """Gibt alle Indikatoren mit EasyInsight-Unterstützung zurück."""
    return [ind for ind in INDICATORS_REGISTRY.values() if ind.easyinsight_name]


def search_indicators(query: str) -> List[IndicatorConfig]:
    """Sucht Indikatoren nach Name, ID oder Beschreibung."""
    query_lower = query.lower()
    results = []
    for ind in INDICATORS_REGISTRY.values():
        if (query_lower in ind.id.lower() or
            query_lower in ind.name.lower() or
            query_lower in ind.description.lower()):
            results.append(ind)
    return results


def get_indicator_stats() -> Dict[str, Any]:
    """Gibt Statistiken über die registrierten Indikatoren zurück."""
    all_indicators = get_all_indicators()
    enabled = get_enabled_indicators()

    category_counts = {}
    for cat in IndicatorCategory:
        category_counts[cat.value] = len(get_indicators_by_category(cat))

    return {
        "total": len(all_indicators),
        "enabled": len(enabled),
        "disabled": len(all_indicators) - len(enabled),
        "twelvedata_supported": len(get_twelvedata_indicators()),
        "easyinsight_supported": len(get_easyinsight_indicators()),
        "by_category": category_counts
    }


# Kategorie-Beschreibungen für UI
CATEGORY_DESCRIPTIONS: Dict[IndicatorCategory, str] = {
    IndicatorCategory.TREND: "Gleitende Durchschnitte und Trend-Indikatoren",
    IndicatorCategory.MOMENTUM: "Oszillatoren und Momentum-Messer",
    IndicatorCategory.VOLATILITY: "Volatilitäts- und Bandbreiten-Indikatoren",
    IndicatorCategory.VOLUME: "Volumen-basierte Indikatoren",
    IndicatorCategory.TREND_FILTER: "Komplexe Trendfilter und -systeme",
    IndicatorCategory.PRICE_TRANSFORM: "Preis-Transformationen",
    IndicatorCategory.ML_FEATURES: "Spezielle Features für Machine Learning",
    IndicatorCategory.PATTERN: "Pattern-Erkennungs-Indikatoren",
}
