# NHITS Integration Plan für KITradingModel

## Status: IMPLEMENTIERT

Die NHITS-Integration wurde erfolgreich implementiert mit einer eigenen PyTorch-basierten Implementierung
(anstatt NeuralForecast wegen Ray-Kompatibilitätsproblemen auf Windows).

## Übersicht

Integration von NHITS (N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting)
als Ergänzung zum bestehenden LLaMA 3.1:8b System.

## Architektur: Hybrid-Ansatz

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEUE HYBRID-ARCHITEKTUR                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TimescaleDB ──► TimeSeriesData (OHLC)                                     │
│                        │                                                    │
│                        ├──────────────────────────┐                        │
│                        │                          │                        │
│                        ▼                          ▼                        │
│              ┌─────────────────┐        ┌─────────────────┐                │
│              │  NHITS Service  │        │ Technical       │                │
│              │  (NeuralForecast)│        │ Indicators      │                │
│              └────────┬────────┘        └────────┬────────┘                │
│                       │                          │                         │
│                       ▼                          ▼                         │
│              ┌─────────────────┐        ┌─────────────────┐                │
│              │ ForecastResult  │        │ TradingSignals  │                │
│              │ - predicted_price│        │ - RSI, MACD    │                │
│              │ - confidence_lo │        │ - BB, ADX      │                │
│              │ - confidence_hi │        └────────┬────────┘                │
│              │ - trend_prob    │                 │                         │
│              └────────┬────────┘                 │                         │
│                       │                          │                         │
│                       └──────────┬───────────────┘                         │
│                                  │                                         │
│                                  ▼                                         │
│                       ┌─────────────────┐                                  │
│                       │ MarketAnalysis  │◄────── RAG Context              │
│                       │ (Enhanced)      │                                  │
│                       └────────┬────────┘                                  │
│                                │                                           │
│                                ▼                                           │
│                       ┌─────────────────┐                                  │
│                       │ LLaMA 3.1:8b    │                                  │
│                       │ + NHITS Forecast│                                  │
│                       └────────┬────────┘                                  │
│                                │                                           │
│                                ▼                                           │
│                       ┌─────────────────┐                                  │
│                       │ Trading         │                                  │
│                       │ Recommendation  │                                  │
│                       └─────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementierungsschritte

### Phase 1: Dependencies & Konfiguration

**1.1 Neue Dependencies (requirements.txt)**
```
neuralforecast>=1.7.0
pytorch>=2.0.0  # Falls nicht vorhanden
```

**1.2 Settings erweitern (src/config/settings.py)**
```python
# NHITS Konfiguration
NHITS_ENABLED: bool = True
NHITS_HORIZON: int = 24           # Vorhersage-Horizont in Stunden
NHITS_INPUT_SIZE: int = 168       # Input-Fenster (7 Tage * 24h)
NHITS_HIDDEN_SIZE: int = 512      # Netzwerk-Größe
NHITS_N_POOLS: list = [2, 2, 1]   # Hierarchische Pooling-Kernel
NHITS_BATCH_SIZE: int = 32
NHITS_MAX_STEPS: int = 1000       # Training Steps
NHITS_LEARNING_RATE: float = 1e-3
NHITS_USE_GPU: bool = True
NHITS_MODEL_PATH: str = "./models/nhits"  # Persistenz
```

---

### Phase 2: Neue Modelle (src/models/forecast_data.py)

```python
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class ForecastResult(BaseModel):
    """NHITS Vorhersage-Ergebnis"""
    symbol: str
    forecast_timestamp: datetime
    horizon_hours: int

    # Preisvorhersagen
    predicted_prices: List[float]          # Preis pro Stunde
    confidence_low: List[float]            # Untere Konfidenzgrenze (10%)
    confidence_high: List[float]           # Obere Konfidenzgrenze (90%)

    # Aggregierte Metriken
    predicted_price_1h: float
    predicted_price_4h: float
    predicted_price_24h: float

    predicted_change_percent_1h: float
    predicted_change_percent_24h: float

    # Trend-Wahrscheinlichkeiten
    trend_up_probability: float            # P(Preis steigt)
    trend_down_probability: float          # P(Preis fällt)

    # Volatilitätsschätzung
    predicted_volatility: float            # Erwartete Volatilität

    # Modell-Metriken
    model_confidence: float                # Modell-Konfidenz (0-1)
    last_training_date: Optional[datetime]
    training_mape: Optional[float]         # Mean Absolute Percentage Error

class ForecastConfig(BaseModel):
    """Konfiguration für Forecast-Request"""
    symbol: str
    horizon: int = 24                      # Stunden
    include_confidence: bool = True
    retrain: bool = False                  # Modell neu trainieren?
```

---

### Phase 3: Neuer Service (src/services/forecast_service.py)

```python
"""
NHITS Forecast Service für Preisvorhersagen
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

from src.models.forecast_data import ForecastResult, ForecastConfig
from src.models.trading_data import TimeSeriesData
from src.config.settings import settings

logger = logging.getLogger(__name__)

class ForecastService:
    """Service für NHITS-basierte Zeitreihenvorhersagen"""

    def __init__(self):
        self.models: Dict[str, NeuralForecast] = {}
        self.model_path = settings.NHITS_MODEL_PATH
        self.horizon = settings.NHITS_HORIZON
        self.input_size = settings.NHITS_INPUT_SIZE

        os.makedirs(self.model_path, exist_ok=True)

    def _prepare_data(
        self,
        time_series: List[TimeSeriesData],
        symbol: str
    ) -> pd.DataFrame:
        """
        Konvertiert TimeSeriesData zu NeuralForecast-Format.

        NeuralForecast erwartet:
        - unique_id: Identifier für die Zeitreihe
        - ds: Datetime
        - y: Zielwert (Close-Preis)
        """
        df = pd.DataFrame([
            {
                'unique_id': symbol,
                'ds': pd.to_datetime(ts.timestamp),
                'y': ts.close,
                # Exogene Variablen (optional)
                'volume': getattr(ts, 'volume', 0),
                'high': ts.high,
                'low': ts.low,
            }
            for ts in time_series
        ])

        df = df.sort_values('ds').reset_index(drop=True)
        return df

    def _create_model(self) -> NeuralForecast:
        """Erstellt ein neues NHITS-Modell"""

        # MQLoss für probabilistische Vorhersagen
        # Quantile: 10%, 50%, 90%
        loss = MQLoss(quantiles=[0.1, 0.5, 0.9])

        model = NHITS(
            h=self.horizon,
            input_size=self.input_size,
            hidden_size=settings.NHITS_HIDDEN_SIZE,
            n_pool_kernel_size=settings.NHITS_N_POOLS,
            batch_size=settings.NHITS_BATCH_SIZE,
            max_steps=settings.NHITS_MAX_STEPS,
            learning_rate=settings.NHITS_LEARNING_RATE,
            loss=loss,
            accelerator='gpu' if settings.NHITS_USE_GPU else 'cpu',
            random_seed=42,
        )

        nf = NeuralForecast(
            models=[model],
            freq='H'  # Stündliche Daten
        )

        return nf

    async def train(
        self,
        time_series: List[TimeSeriesData],
        symbol: str
    ) -> Dict:
        """
        Trainiert das NHITS-Modell auf historischen Daten.

        Args:
            time_series: Liste von TimeSeriesData (mind. input_size + horizon)
            symbol: Symbol-Identifier

        Returns:
            Training-Metriken
        """
        logger.info(f"Training NHITS model for {symbol}")

        df = self._prepare_data(time_series, symbol)

        if len(df) < self.input_size + self.horizon:
            raise ValueError(
                f"Insufficient data: {len(df)} rows, "
                f"need {self.input_size + self.horizon}"
            )

        # Modell erstellen und trainieren
        nf = self._create_model()
        nf.fit(df)

        # Modell speichern
        model_file = os.path.join(self.model_path, f"{symbol}.pkl")
        nf.save(model_file)
        self.models[symbol] = nf

        logger.info(f"NHITS model trained and saved for {symbol}")

        return {
            "symbol": symbol,
            "training_rows": len(df),
            "model_path": model_file,
            "trained_at": datetime.utcnow().isoformat()
        }

    def _load_model(self, symbol: str) -> Optional[NeuralForecast]:
        """Lädt ein gespeichertes Modell"""
        if symbol in self.models:
            return self.models[symbol]

        model_file = os.path.join(self.model_path, f"{symbol}.pkl")
        if os.path.exists(model_file):
            nf = NeuralForecast.load(model_file)
            self.models[symbol] = nf
            return nf

        return None

    async def forecast(
        self,
        time_series: List[TimeSeriesData],
        symbol: str,
        config: Optional[ForecastConfig] = None
    ) -> ForecastResult:
        """
        Generiert Preisvorhersagen mit NHITS.

        Args:
            time_series: Aktuelle Zeitreihendaten (mind. input_size)
            symbol: Symbol-Identifier
            config: Forecast-Konfiguration

        Returns:
            ForecastResult mit Vorhersagen und Konfidenzintervallen
        """
        config = config or ForecastConfig(symbol=symbol)

        # Modell laden oder neu trainieren
        nf = self._load_model(symbol)

        if nf is None or config.retrain:
            await self.train(time_series, symbol)
            nf = self.models[symbol]

        # Daten vorbereiten
        df = self._prepare_data(time_series, symbol)

        # Vorhersage generieren
        forecast_df = nf.predict(df)

        # Ergebnisse extrahieren
        # Spalten: unique_id, ds, NHITS-q-0.1, NHITS-q-0.5, NHITS-q-0.9
        predictions = forecast_df[forecast_df['unique_id'] == symbol]

        predicted_prices = predictions['NHITS-median'].tolist()
        confidence_low = predictions['NHITS-lo-90'].tolist()
        confidence_high = predictions['NHITS-hi-90'].tolist()

        current_price = time_series[-1].close

        # Trend-Wahrscheinlichkeiten berechnen
        price_changes = np.diff([current_price] + predicted_prices)
        trend_up_prob = np.mean(price_changes > 0)

        # Volatilität aus Konfidenzintervallen
        volatility = np.mean([
            (hi - lo) / mid if mid > 0 else 0
            for hi, lo, mid in zip(
                confidence_high,
                confidence_low,
                predicted_prices
            )
        ])

        result = ForecastResult(
            symbol=symbol,
            forecast_timestamp=datetime.utcnow(),
            horizon_hours=self.horizon,

            predicted_prices=predicted_prices,
            confidence_low=confidence_low,
            confidence_high=confidence_high,

            predicted_price_1h=predicted_prices[0] if len(predicted_prices) > 0 else current_price,
            predicted_price_4h=predicted_prices[3] if len(predicted_prices) > 3 else predicted_prices[-1],
            predicted_price_24h=predicted_prices[-1],

            predicted_change_percent_1h=(
                (predicted_prices[0] - current_price) / current_price * 100
                if len(predicted_prices) > 0 else 0
            ),
            predicted_change_percent_24h=(
                (predicted_prices[-1] - current_price) / current_price * 100
            ),

            trend_up_probability=trend_up_prob,
            trend_down_probability=1 - trend_up_prob,

            predicted_volatility=volatility,
            model_confidence=1 - volatility,  # Inverse Volatilität als Konfidenz

            last_training_date=None,  # Aus Modell-Metadaten
            training_mape=None
        )

        logger.info(
            f"NHITS forecast for {symbol}: "
            f"{current_price:.5f} → {predicted_prices[-1]:.5f} "
            f"({result.predicted_change_percent_24h:+.2f}%)"
        )

        return result


# Singleton-Instanz
forecast_service = ForecastService()
```

---

### Phase 4: MarketAnalysis erweitern (src/models/trading_data.py)

```python
# Zu MarketAnalysis hinzufügen:

class MarketAnalysis(BaseModel):
    # ... bestehende Felder ...

    # NEU: NHITS Forecast-Daten
    nhits_forecast: Optional[Dict] = None
    predicted_price_1h: Optional[float] = None
    predicted_price_4h: Optional[float] = None
    predicted_price_24h: Optional[float] = None
    predicted_change_percent_24h: Optional[float] = None
    forecast_confidence_low: Optional[float] = None
    forecast_confidence_high: Optional[float] = None
    trend_up_probability: Optional[float] = None
    model_confidence: Optional[float] = None
```

---

### Phase 5: Analysis Service Integration (src/services/analysis_service.py)

```python
# In _create_market_analysis() hinzufügen:

async def _create_market_analysis(
    self,
    symbol: str,
    time_series: List[TimeSeriesData],
    indicators: TechnicalIndicators,
    signals: List[TradingSignal]
) -> MarketAnalysis:
    """Erstellt MarketAnalysis mit NHITS-Forecast"""

    # ... bestehender Code ...

    # NEU: NHITS Forecast
    nhits_forecast = None
    if settings.NHITS_ENABLED:
        try:
            from src.services.forecast_service import forecast_service

            forecast_result = await forecast_service.forecast(
                time_series=time_series,
                symbol=symbol
            )

            nhits_forecast = {
                "predicted_price_1h": forecast_result.predicted_price_1h,
                "predicted_price_4h": forecast_result.predicted_price_4h,
                "predicted_price_24h": forecast_result.predicted_price_24h,
                "predicted_change_percent_24h": forecast_result.predicted_change_percent_24h,
                "confidence_low_24h": forecast_result.confidence_low[-1],
                "confidence_high_24h": forecast_result.confidence_high[-1],
                "trend_up_probability": forecast_result.trend_up_probability,
                "model_confidence": forecast_result.model_confidence,
            }

            logger.info(f"NHITS forecast added for {symbol}")

        except Exception as e:
            logger.warning(f"NHITS forecast failed: {e}")

    return MarketAnalysis(
        # ... bestehende Felder ...
        nhits_forecast=nhits_forecast,
        predicted_price_24h=nhits_forecast.get("predicted_price_24h") if nhits_forecast else None,
        # ... etc ...
    )
```

---

### Phase 6: LLM Prompt erweitern (src/services/llm_service.py)

```python
# System Prompt erweitern:

ENHANCED_SYSTEM_PROMPT = """
... bestehender Prompt ...

## NHITS FORECAST DATA
If provided, use the NHITS neural network forecast to enhance your analysis:
- predicted_price_24h: AI-predicted price in 24 hours
- confidence_low/high: 90% confidence interval
- trend_up_probability: Probability that price will increase
- model_confidence: How confident the model is (0-1)

IMPORTANT:
- If NHITS predicts significant movement (>1%) with high confidence (>0.7),
  weight this heavily in your recommendation
- Use confidence intervals for stop-loss and take-profit levels
- If NHITS and technical indicators disagree, mention this discrepancy
"""

# Im User Prompt:
def _build_user_prompt(self, market_analysis: MarketAnalysis) -> str:
    prompt = f"""
    ... bestehender Prompt ...

    ## NHITS AI FORECAST
    """

    if market_analysis.nhits_forecast:
        forecast = market_analysis.nhits_forecast
        prompt += f"""
    - Predicted Price (1h): {forecast.get('predicted_price_1h', 'N/A')}
    - Predicted Price (4h): {forecast.get('predicted_price_4h', 'N/A')}
    - Predicted Price (24h): {forecast.get('predicted_price_24h', 'N/A')}
    - Predicted Change (24h): {forecast.get('predicted_change_percent_24h', 0):+.2f}%
    - 90% Confidence Range: [{forecast.get('confidence_low_24h', 'N/A')} - {forecast.get('confidence_high_24h', 'N/A')}]
    - Trend Up Probability: {forecast.get('trend_up_probability', 0):.1%}
    - Model Confidence: {forecast.get('model_confidence', 0):.1%}
    """
    else:
        prompt += "\n    - No NHITS forecast available"

    return prompt
```

---

### Phase 7: API Endpoints (src/api/routes.py)

```python
# Neue Endpoints:

@router.post("/forecast/{symbol}")
async def get_forecast(
    symbol: str,
    horizon: int = 24,
    retrain: bool = False
):
    """
    Generiert NHITS Preisvorhersage für ein Symbol.

    - horizon: Vorhersage-Horizont in Stunden (default: 24)
    - retrain: Modell neu trainieren (default: False)
    """
    from src.services.forecast_service import forecast_service
    from src.models.forecast_data import ForecastConfig

    # Daten aus TimescaleDB laden
    time_series = await analysis_service._fetch_market_data(symbol, days=30)

    config = ForecastConfig(
        symbol=symbol,
        horizon=horizon,
        retrain=retrain
    )

    result = await forecast_service.forecast(
        time_series=time_series,
        symbol=symbol,
        config=config
    )

    return result

@router.post("/forecast/{symbol}/train")
async def train_forecast_model(symbol: str):
    """Trainiert das NHITS-Modell für ein Symbol neu."""
    from src.services.forecast_service import forecast_service

    time_series = await analysis_service._fetch_market_data(symbol, days=90)
    result = await forecast_service.train(time_series, symbol)

    return result
```

---

## Datenfluss im Detail

```
1. User Request: POST /analyze?symbol=EURUSD
                        │
                        ▼
2. TimescaleDB: Fetch 30 days OHLC (H1 timeframe)
                        │
                        ▼
3. Parallel Processing:
   ┌────────────────────┼────────────────────┐
   │                    │                    │
   ▼                    ▼                    ▼
Technical          NHITS Forecast        RAG Search
Indicators         (24h horizon)         (Similar patterns)
   │                    │                    │
   └────────────────────┼────────────────────┘
                        │
                        ▼
4. MarketAnalysis: Combine all data
   - Current price, changes
   - RSI, MACD, BB signals
   - NHITS predictions + confidence
   - Historical context from RAG
                        │
                        ▼
5. LLM Analysis (LLaMA 3.1:8b):
   - Analyze technicals
   - Consider NHITS forecast
   - Use RAG context
   - Generate recommendation
                        │
                        ▼
6. TradingRecommendation:
   - Direction (LONG/SHORT/NEUTRAL)
   - Entry, SL, TP levels (informed by NHITS confidence)
   - Confidence score (boosted by NHITS agreement)
   - Rationale (includes NHITS interpretation)
```

---

## GPU/Ressourcen-Anforderungen

| Komponente | GPU Memory | CPU | RAM |
|------------|-----------|-----|-----|
| LLaMA 3.1:8b | ~6-8 GB | 16 Threads | 16 GB |
| NHITS (Training) | ~2 GB | 4 Threads | 4 GB |
| NHITS (Inference) | ~500 MB | 2 Threads | 2 GB |
| FAISS (RAG) | ~1 GB | 2 Threads | 2 GB |
| **Total** | **~10 GB** | **24 Threads** | **24 GB** |

Mit einer RTX 3080/4080 (10-16 GB VRAM) läuft alles parallel.

---

## Vorteile des Hybrid-Ansatzes

1. **Präzise numerische Vorhersagen** durch NHITS
2. **Konfidenzintervalle** für besseres Risikomanagement
3. **LLM kann NHITS-Ergebnisse interpretieren** und erklären
4. **Ensemble-Effekt**: Wenn NHITS und technische Indikatoren übereinstimmen → höhere Konfidenz
5. **Explainability**: LLM erklärt, warum es mit/gegen NHITS-Forecast handelt

---

## Nächste Schritte

1. [ ] Dependencies installieren: `pip install neuralforecast`
2. [ ] `forecast_data.py` Modelle erstellen
3. [ ] `forecast_service.py` implementieren
4. [ ] `trading_data.py` erweitern
5. [ ] `analysis_service.py` integrieren
6. [ ] `llm_service.py` Prompts erweitern
7. [ ] API Endpoints hinzufügen
8. [ ] Tests schreiben
9. [ ] Training-Daten vorbereiten (mind. 7 Tage H1-Daten pro Symbol)
