# Self-Learning CNN-LSTM Multi-Task Service - Proposal

**Version**: 1.0
**Datum**: 2026-01-26
**Status**: Entwurf
**Basiert auf**: SELF_LEARNING_TCN_PROPOSAL.md

---

## Executive Summary

Dieses Dokument beschreibt die Erweiterung des CNN-LSTM Multi-Task Service zu einem **selbstlernenden System**. Das Ziel ist, dass das System kontinuierlich aus realen Marktergebnissen lernt und seine drei Vorhersage-Tasks (Preis, Patterns, Regime) automatisch verbessert.

### Unterschiede zum TCN Self-Learning

| Aspekt | TCN | CNN-LSTM |
|--------|-----|----------|
| **Output** | Single Task (Patterns) | Multi-Task (Price, Patterns, Regime) |
| **Tracking** | Pattern-Outcome | 3 separate Outcome-Streams |
| **EWC** | Single Fisher Matrix | Task-weighted Fisher Matrices |
| **Validation** | Pattern Accuracy | Composite Score (3 Tasks) |

---

## 1. Architektur-Übersicht

### 1.1 CNN-LSTM Multi-Task Architecture

```
Input (batch, seq_len, 25 features)
         │
    CNN Encoder
    ├── Conv1d(25→64, k=3) + BN + ReLU
    ├── Conv1d(64→128, k=3) + BN + ReLU
    └── Conv1d(128→256, k=3) + BN + ReLU
         │
   BiLSTM Encoder
    └── BiLSTM(256→128, layers=2) + Attention
         │
    ┌────┼────┐
    ▼    ▼    ▼
  Price Pattern Regime
  Head   Head   Head
   (4)   (16)   (4)
```

### 1.2 Self-Learning Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      SELF-LEARNING CNN-LSTM SYSTEM                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       MULTI-TASK PREDICTION                          │    │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐                    │    │
│  │  │   PRICE   │    │  PATTERN  │    │   REGIME  │                    │    │
│  │  │ Prediction│    │ Detection │    │ Detection │                    │    │
│  │  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                    │    │
│  └────────┼────────────────┼────────────────┼──────────────────────────┘    │
│           │                │                │                                │
│           ▼                ▼                ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    OUTCOME TRACKING (per Task)                       │    │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐                    │    │
│  │  │   Price   │    │  Pattern  │    │   Regime  │                    │    │
│  │  │  Tracker  │    │  Tracker  │    │  Tracker  │                    │    │
│  │  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                    │    │
│  └────────┼────────────────┼────────────────┼──────────────────────────┘    │
│           │                │                │                                │
│           └────────────────┼────────────────┘                                │
│                            ▼                                                 │
│           ┌────────────────────────────────┐                                │
│           │       FEEDBACK AGGREGATOR       │                                │
│           │   (weighted by task priority)   │                                │
│           └───────────────┬────────────────┘                                │
│                           │                                                  │
│           ┌───────────────┼───────────────┐                                 │
│           ▼               ▼               ▼                                 │
│    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                         │
│    │   DRIFT     │ │  FEEDBACK   │ │ INCREMENTAL │                         │
│    │  DETECTOR   │ │   BUFFER    │ │   TRAINER   │                         │
│    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                         │
│           │               │               │                                  │
│           └───────────────┼───────────────┘                                 │
│                           ▼                                                  │
│                  ┌─────────────────┐                                        │
│                  │   ORCHESTRATOR   │                                        │
│                  │   + Hot-Reload   │                                        │
│                  └─────────────────┘                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Task Outcome Tracking

### 2.1 Task-spezifische Outcome-Definitionen

```python
@dataclass
class MultiTaskOutcome:
    """Outcome für alle drei CNN-LSTM Tasks."""

    prediction_id: str
    symbol: str
    timeframe: str
    predicted_at: datetime

    # ========== PRICE TASK ==========
    price_prediction: PricePredictionOutcome

    # ========== PATTERN TASK ==========
    pattern_prediction: PatternPredictionOutcome

    # ========== REGIME TASK ==========
    regime_prediction: RegimePredictionOutcome

    # ========== AGGREGATED ==========
    composite_score: float  # Gewichteter Durchschnitt
    overall_outcome: str    # WIN / PARTIAL / LOSS


@dataclass
class PricePredictionOutcome:
    """Outcome für Price Direction Task."""

    # Predictions
    predicted_direction: str        # "up" / "down" / "neutral"
    predicted_change_1h: float      # %
    predicted_change_4h: float      # %
    predicted_change_24h: float     # %
    confidence: float

    # Actual Results
    actual_direction: Optional[str]
    actual_change_1h: Optional[float]
    actual_change_4h: Optional[float]
    actual_change_24h: Optional[float]

    # Evaluation
    direction_correct: Optional[bool]
    mae_1h: Optional[float]         # Mean Absolute Error
    mae_4h: Optional[float]
    mae_24h: Optional[float]

    outcome: Optional[str]          # WIN / PARTIAL / LOSS
    outcome_score: Optional[float]  # -1.0 to +1.0


@dataclass
class PatternPredictionOutcome:
    """Outcome für Pattern Detection Task."""

    # Predictions (Multi-Label, 16 patterns)
    predicted_patterns: List[str]
    pattern_probabilities: Dict[str, float]
    top_pattern: str
    top_confidence: float

    # Validation (nach Pattern-Abschluss)
    patterns_confirmed: List[str]
    patterns_invalidated: List[str]

    # Evaluation
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]

    outcome: Optional[str]
    outcome_score: Optional[float]


@dataclass
class RegimePredictionOutcome:
    """Outcome für Regime Detection Task."""

    # Predictions
    predicted_regime: str           # bull / bear / sideways / high_vol
    regime_probabilities: Dict[str, float]
    confidence: float

    # Actual (nach Tracking-Periode)
    actual_regime: Optional[str]
    regime_duration_correct: Optional[bool]

    # Evaluation
    regime_correct: Optional[bool]
    regime_match_score: Optional[float]  # 0-1

    outcome: Optional[str]
    outcome_score: Optional[float]
```

### 2.2 Tracking-Perioden per Task

```python
class MultiTaskOutcomeTracker:
    """Verfolgt Outcomes für alle drei Tasks."""

    # Task-spezifische Tracking-Perioden
    TRACKING_PERIODS = {
        "price": {
            "H1": [1, 4, 24],           # Stunden - direkt messbar
            "H4": [4, 24, 72],
            "D1": [24, 72, 168],
        },
        "pattern": {
            "H1": [24, 72, 168],         # Patterns brauchen Zeit zur Bestätigung
            "H4": [72, 168, 336],
            "D1": [168, 336, 720],
        },
        "regime": {
            "H1": [24, 72, 168],         # Regime-Stabilität prüfen
            "H4": [72, 168, 336],
            "D1": [168, 336, 720],
        },
    }

    async def track_prediction(self, prediction: MultiTaskPrediction):
        """Startet Tracking für eine Multi-Task Vorhersage."""

        outcome = MultiTaskOutcome(
            prediction_id=prediction.id,
            symbol=prediction.symbol,
            timeframe=prediction.timeframe,
            predicted_at=datetime.now(timezone.utc),
            price_prediction=self._init_price_outcome(prediction),
            pattern_prediction=self._init_pattern_outcome(prediction),
            regime_prediction=self._init_regime_outcome(prediction),
        )

        self.active_tracking[prediction.id] = outcome

        # Schedule checkpoints für jeden Task
        await self._schedule_price_checkpoints(outcome)
        await self._schedule_pattern_checkpoints(outcome)
        await self._schedule_regime_checkpoints(outcome)
```

### 2.3 Price Task Evaluation

```python
async def evaluate_price_outcome(
    self,
    outcome: PricePredictionOutcome,
    symbol: str,
    timeframe: str,
    hours_elapsed: int,
) -> PricePredictionOutcome:
    """Evaluiert Price Prediction nach Ablauf der Tracking-Periode."""

    # Hole aktuelle Preise
    current_price = await self._get_current_price(symbol)
    initial_price = outcome.price_at_prediction

    # Berechne tatsächliche Änderung
    actual_change = (current_price - initial_price) / initial_price * 100

    # Setze actual values basierend auf elapsed time
    if hours_elapsed >= 1:
        outcome.actual_change_1h = actual_change
    if hours_elapsed >= 4:
        outcome.actual_change_4h = actual_change
    if hours_elapsed >= 24:
        outcome.actual_change_24h = actual_change

    # Direction Evaluation
    actual_direction = "up" if actual_change > 0.1 else ("down" if actual_change < -0.1 else "neutral")
    outcome.actual_direction = actual_direction
    outcome.direction_correct = (outcome.predicted_direction == actual_direction)

    # MAE Calculation
    if hours_elapsed >= 1 and outcome.predicted_change_1h is not None:
        outcome.mae_1h = abs(outcome.predicted_change_1h - outcome.actual_change_1h)

    # Outcome Classification
    if outcome.direction_correct:
        # Richtige Richtung
        magnitude_accuracy = 1.0 - min(1.0, outcome.mae_1h / 5.0)  # 5% max error
        if magnitude_accuracy > 0.7:
            outcome.outcome = "WIN"
            outcome.outcome_score = 0.5 + magnitude_accuracy * 0.5
        else:
            outcome.outcome = "PARTIAL"
            outcome.outcome_score = magnitude_accuracy * 0.5
    else:
        # Falsche Richtung
        outcome.outcome = "LOSS"
        outcome.outcome_score = -0.5 - (1.0 - outcome.confidence) * 0.5

    return outcome
```

---

## 3. Multi-Task Feedback System

### 3.1 Task-gewichtete Feedback-Aggregation

```python
@dataclass
class MultiTaskFeedbackEntry:
    """Feedback-Eintrag für Multi-Task Training."""

    # Input Data
    ohlcv_sequence: np.ndarray          # (seq_len, 25)

    # Original Labels
    price_label: np.ndarray             # (4,) - direction + changes
    pattern_label: np.ndarray           # (16,) - multi-label
    regime_label: np.ndarray            # (4,) - one-hot

    # Task-specific Feedback
    price_feedback: TaskFeedback
    pattern_feedback: TaskFeedback
    regime_feedback: TaskFeedback

    # Aggregated
    composite_weight: float             # 0.0-2.0
    timestamp: datetime


@dataclass
class TaskFeedback:
    """Feedback für einen einzelnen Task."""

    feedback_type: FeedbackType
    weight: float                       # 0.0-2.0
    outcome_score: float                # -1.0 to +1.0
    confidence_calibration: float       # War Confidence korrekt?


class MultiTaskFeedbackAggregator:
    """Aggregiert Feedback von allen Tasks."""

    # Task-Gewichte für Loss-Berechnung (aus config.py)
    TASK_WEIGHTS = {
        "price": 0.40,      # Wichtigster Task
        "pattern": 0.35,
        "regime": 0.25,
    }

    def aggregate_feedback(
        self,
        outcome: MultiTaskOutcome
    ) -> MultiTaskFeedbackEntry:
        """Erstellt gewichteten Feedback-Eintrag."""

        # Task-spezifische Feedback-Objekte
        price_fb = self._create_price_feedback(outcome.price_prediction)
        pattern_fb = self._create_pattern_feedback(outcome.pattern_prediction)
        regime_fb = self._create_regime_feedback(outcome.regime_prediction)

        # Gewichteter Composite-Score
        composite_score = (
            self.TASK_WEIGHTS["price"] * price_fb.outcome_score +
            self.TASK_WEIGHTS["pattern"] * pattern_fb.outcome_score +
            self.TASK_WEIGHTS["regime"] * regime_fb.outcome_score
        )

        # Composite Weight (höher bei klaren Signalen)
        composite_weight = self._calculate_composite_weight(
            price_fb, pattern_fb, regime_fb
        )

        return MultiTaskFeedbackEntry(
            ohlcv_sequence=outcome.ohlcv_data,
            price_label=outcome.price_prediction.original_label,
            pattern_label=outcome.pattern_prediction.original_label,
            regime_label=outcome.regime_prediction.original_label,
            price_feedback=price_fb,
            pattern_feedback=pattern_fb,
            regime_feedback=regime_fb,
            composite_weight=composite_weight,
            timestamp=datetime.now(timezone.utc),
        )

    def _calculate_composite_weight(
        self,
        price_fb: TaskFeedback,
        pattern_fb: TaskFeedback,
        regime_fb: TaskFeedback,
    ) -> float:
        """
        Berechnet Gewicht für Training.

        Höhere Gewichte bei:
        - Klaren Win/Loss (nicht neutral)
        - Übereinstimmung aller Tasks
        - Hoher Confidence-Calibration
        """

        # Basis-Gewicht
        base = 1.0

        # Bonus für klare Signale
        scores = [price_fb.outcome_score, pattern_fb.outcome_score, regime_fb.outcome_score]
        clarity = np.mean([abs(s) for s in scores])  # 0-1
        base += clarity * 0.3

        # Bonus für Task-Übereinstimmung
        signs = [np.sign(s) for s in scores if s != 0]
        if len(signs) >= 2 and len(set(signs)) == 1:
            base += 0.2  # Alle Tasks stimmen überein

        # Penalty für widersprüchliche Signale
        if len(set(signs)) > 1:
            base -= 0.2

        return min(2.0, max(0.5, base))
```

---

## 4. Multi-Task EWC Training

### 4.1 Task-spezifische Fisher Matrices

```python
class MultiTaskEWCTrainer:
    """Inkrementelles Training mit EWC für Multi-Task Learning."""

    def __init__(
        self,
        model_path: str = "data/models/cnn-lstm/latest.pt",
        learning_rate: float = 1e-5,
        ewc_lambda: float = 1000,
    ):
        self.model_path = Path(model_path)
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda

        # Separate Fisher Matrices pro Task
        self.fisher_matrices = {
            "shared": None,     # CNN + LSTM Encoder
            "price": None,      # Price Head
            "pattern": None,    # Pattern Head
            "regime": None,     # Regime Head
        }
        self.optimal_params = None

    def _compute_task_fisher(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        task: str,
    ) -> Dict[str, torch.Tensor]:
        """Berechnet Fisher Matrix für einen spezifischen Task."""

        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        model.eval()
        for batch in data_loader:
            x, price_y, pattern_y, regime_y = batch
            model.zero_grad()

            # Forward
            price_out, pattern_out, regime_out = model(x)

            # Task-spezifischer Loss
            if task == "price":
                loss = F.mse_loss(price_out, price_y)
            elif task == "pattern":
                loss = F.binary_cross_entropy_with_logits(pattern_out, pattern_y)
            elif task == "regime":
                loss = F.cross_entropy(regime_out, regime_y.argmax(dim=1))
            else:  # shared
                loss = self._multi_task_loss(
                    price_out, pattern_out, regime_out,
                    price_y, pattern_y, regime_y
                )

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

        # Normalize
        for name in fisher:
            fisher[name] /= len(data_loader)

        return fisher

    def _ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Multi-Task EWC Loss.

        Kombiniert Fisher Matrices aller Tasks gewichtet.
        """
        total_loss = 0

        for task, fisher in self.fisher_matrices.items():
            if fisher is None:
                continue

            task_weight = {
                "shared": 1.0,
                "price": 0.4,
                "pattern": 0.35,
                "regime": 0.25,
            }[task]

            for name, param in model.named_parameters():
                if name in fisher:
                    optimal = self.optimal_params[name]
                    total_loss += task_weight * (
                        fisher[name] * (param - optimal).pow(2)
                    ).sum()

        return total_loss

    async def incremental_update(
        self,
        feedback_batch: List[MultiTaskFeedbackEntry],
        validation_threshold: float = 0.02,
    ) -> IncrementalUpdateResult:
        """
        Führt inkrementelles Multi-Task Update durch.
        """

        if len(feedback_batch) < 100:
            return IncrementalUpdateResult(
                success=False,
                reason="Nicht genug Samples"
            )

        # 1. Lade Modell
        model, old_metrics = await self._load_model()

        # 2. Erstelle DataLoader
        train_loader, val_loader = self._create_dataloaders(feedback_batch)

        # 3. Initialisiere Fisher Matrices falls nötig
        if self.fisher_matrices["shared"] is None:
            for task in ["shared", "price", "pattern", "regime"]:
                self.fisher_matrices[task] = self._compute_task_fisher(
                    model, val_loader, task
                )
            self.optimal_params = {n: p.clone() for n, p in model.named_parameters()}

        # 4. Training Loop
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        model.train()
        for epoch in range(3):
            for batch in train_loader:
                x, price_y, pattern_y, regime_y, weights = batch
                optimizer.zero_grad()

                # Forward
                price_out, pattern_out, regime_out = model(x)

                # Multi-Task Loss (gewichtet nach Feedback)
                task_loss = self._weighted_multi_task_loss(
                    price_out, pattern_out, regime_out,
                    price_y, pattern_y, regime_y,
                    weights
                )

                # EWC Loss
                ewc_loss = self._ewc_loss(model)

                # Total Loss
                loss = task_loss + self.ewc_lambda * ewc_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # 5. Validation
        new_metrics = await self._validate_model(model, val_loader)

        # 6. Check Performance (für jeden Task)
        performance_ok = True
        for task in ["price", "pattern", "regime"]:
            old_score = old_metrics[f"{task}_score"]
            new_score = new_metrics[f"{task}_score"]
            if old_score - new_score > validation_threshold:
                performance_ok = False
                break

        if not performance_ok:
            return IncrementalUpdateResult(
                success=False,
                reason="Performance drop too high",
                old_metrics=old_metrics,
                new_metrics=new_metrics
            )

        # 7. Save & Reload
        await self._save_model(model, new_metrics)

        # 8. Update Fisher Matrices
        for task in ["shared", "price", "pattern", "regime"]:
            self.fisher_matrices[task] = self._compute_task_fisher(
                model, val_loader, task
            )
        self.optimal_params = {n: p.clone() for n, p in model.named_parameters()}

        # 9. Notify Hot-Reload
        await self._notify_model_reload()

        return IncrementalUpdateResult(
            success=True,
            samples_count=len(feedback_batch),
            old_metrics=old_metrics,
            new_metrics=new_metrics,
        )
```

---

## 5. Multi-Task Drift Detection

```python
class MultiTaskDriftDetector:
    """Erkennt Drift für alle drei Tasks."""

    def __init__(
        self,
        window_size: int = 100,
        thresholds: Dict[str, Dict[str, float]] = None,
    ):
        self.window_size = window_size

        # Task-spezifische Thresholds
        self.thresholds = thresholds or {
            "price": {
                "warning": 0.10,    # 10% Accuracy Drop
                "critical": 0.20,
            },
            "pattern": {
                "warning": 0.08,    # 8% F1 Drop
                "critical": 0.15,
            },
            "regime": {
                "warning": 0.08,
                "critical": 0.15,
            },
            "composite": {
                "warning": 0.08,
                "critical": 0.15,
            },
        }

        self.reference_metrics: Dict[str, Dict] = {}
        self.current_window: List[MultiTaskOutcome] = []

    async def check_drift(
        self,
        outcome: MultiTaskOutcome
    ) -> MultiTaskDriftStatus:
        """Prüft Drift für alle Tasks."""

        self.current_window.append(outcome)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

        if len(self.current_window) < self.window_size // 2:
            return MultiTaskDriftStatus(detected=False)

        # Berechne aktuelle Metriken pro Task
        current_metrics = {
            "price": self._compute_price_metrics(),
            "pattern": self._compute_pattern_metrics(),
            "regime": self._compute_regime_metrics(),
        }

        # Composite Score
        current_metrics["composite"] = self._compute_composite_metrics(current_metrics)

        # Initialisiere Referenz falls nötig
        if not self.reference_metrics:
            self.reference_metrics = current_metrics
            return MultiTaskDriftStatus(detected=False, message="Reference established")

        # Berechne Drift pro Task
        drift_results = {}
        overall_severity = "OK"

        for task in ["price", "pattern", "regime", "composite"]:
            drift_score = self._compute_drift_score(
                self.reference_metrics[task],
                current_metrics[task]
            )

            severity = "OK"
            if drift_score > self.thresholds[task]["critical"]:
                severity = "CRITICAL"
                overall_severity = "CRITICAL"
            elif drift_score > self.thresholds[task]["warning"]:
                severity = "WARNING"
                if overall_severity != "CRITICAL":
                    overall_severity = "WARNING"

            drift_results[task] = {
                "score": drift_score,
                "severity": severity,
            }

        return MultiTaskDriftStatus(
            detected=overall_severity != "OK",
            overall_severity=overall_severity,
            task_drifts=drift_results,
            recommended_action=self._get_recommended_action(overall_severity),
        )

    def _compute_price_metrics(self) -> Dict:
        """Berechnet Price-Task Metriken."""
        outcomes = [o.price_prediction for o in self.current_window
                   if o.price_prediction.outcome is not None]

        if not outcomes:
            return {}

        return {
            "direction_accuracy": np.mean([
                1 if o.direction_correct else 0 for o in outcomes
            ]),
            "avg_mae": np.mean([o.mae_1h for o in outcomes if o.mae_1h is not None]),
            "avg_outcome_score": np.mean([o.outcome_score for o in outcomes]),
        }

    def _compute_pattern_metrics(self) -> Dict:
        """Berechnet Pattern-Task Metriken."""
        outcomes = [o.pattern_prediction for o in self.current_window
                   if o.pattern_prediction.outcome is not None]

        if not outcomes:
            return {}

        return {
            "avg_precision": np.mean([o.precision for o in outcomes if o.precision]),
            "avg_recall": np.mean([o.recall for o in outcomes if o.recall]),
            "avg_f1": np.mean([o.f1_score for o in outcomes if o.f1_score]),
            "avg_outcome_score": np.mean([o.outcome_score for o in outcomes]),
        }

    def _compute_regime_metrics(self) -> Dict:
        """Berechnet Regime-Task Metriken."""
        outcomes = [o.regime_prediction for o in self.current_window
                   if o.regime_prediction.outcome is not None]

        if not outcomes:
            return {}

        return {
            "regime_accuracy": np.mean([
                1 if o.regime_correct else 0 for o in outcomes
            ]),
            "avg_outcome_score": np.mean([o.outcome_score for o in outcomes]),
        }
```

---

## 6. API Endpoints

### 6.1 CNN-LSTM Inference Service (Port 3007)

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/v1/self-learning/status` | GET | Self-Learning System Status |
| `/api/v1/outcomes/{prediction_id}` | GET | Multi-Task Outcome abrufen |
| `/api/v1/outcomes/statistics` | GET | Aggregierte Outcome-Statistiken |
| `/api/v1/drift/status` | GET | Multi-Task Drift Status |
| `/api/v1/drift/history` | GET | Drift-Verlauf |

### 6.2 CNN-LSTM Training Service (Port 3017)

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/v1/self-learning/status` | GET | Self-Learning Status |
| `/api/v1/self-learning/start` | POST | Self-Learning Loop starten |
| `/api/v1/self-learning/stop` | POST | Self-Learning Loop stoppen |
| `/api/v1/self-learning/trigger` | POST | Manuelles Training |
| `/api/v1/feedback-buffer/statistics` | GET | Feedback Buffer Stats |
| `/api/v1/feedback-buffer/clear` | POST | Buffer leeren |
| `/api/v1/model/versions` | GET | Modell-Versionen |
| `/api/v1/model/rollback/{version}` | POST | Rollback zu Version |
| `/api/v1/model/compare` | POST | A/B Modell-Vergleich |

### 6.3 Response Models

```python
class MultiTaskSelfLearningStatus(BaseModel):
    """Status des Self-Learning Systems."""

    enabled: bool

    # Tracking Status
    predictions_being_tracked: int
    outcomes_completed_today: int

    # Per-Task Status
    price_task: TaskStatus
    pattern_task: TaskStatus
    regime_task: TaskStatus

    # Feedback
    feedback_buffer_size: int
    feedback_ready_for_training: bool

    # Drift
    overall_drift_status: str
    task_drift_scores: Dict[str, float]

    # Training
    last_incremental_train: Optional[datetime]
    next_scheduled_train: Optional[datetime]
    total_incremental_updates: int

    model_version: str


class TaskStatus(BaseModel):
    """Status für einen einzelnen Task."""

    outcome_count: int
    win_rate: float
    avg_score: float
    drift_score: float
    drift_severity: str
```

---

## 7. Konfiguration

### 7.1 Environment Variables

```bash
# Self-Learning Feature Flags
CNN_LSTM_SELF_LEARNING_ENABLED=true
CNN_LSTM_OUTCOME_TRACKING_ENABLED=true
CNN_LSTM_INCREMENTAL_TRAINING_ENABLED=true
CNN_LSTM_DRIFT_DETECTION_ENABLED=true

# Task-specific Tracking Periods (comma-separated hours)
CNN_LSTM_PRICE_TRACKING_PERIODS="1,4,24"
CNN_LSTM_PATTERN_TRACKING_PERIODS="24,72,168"
CNN_LSTM_REGIME_TRACKING_PERIODS="24,72,168"

# Feedback Buffer
CNN_LSTM_FEEDBACK_BUFFER_MAX_SIZE=10000
CNN_LSTM_FEEDBACK_BUFFER_MIN_BATCH=100
CNN_LSTM_FEEDBACK_BUFFER_MAX_AGE_HOURS=168

# Incremental Training
CNN_LSTM_INCREMENTAL_LEARNING_RATE=0.00001
CNN_LSTM_INCREMENTAL_EWC_LAMBDA=1000
CNN_LSTM_INCREMENTAL_VALIDATION_THRESHOLD=0.02

# Task Weights (for loss calculation)
CNN_LSTM_PRICE_WEIGHT=0.40
CNN_LSTM_PATTERN_WEIGHT=0.35
CNN_LSTM_REGIME_WEIGHT=0.25

# Drift Detection
CNN_LSTM_DRIFT_WINDOW_SIZE=100
CNN_LSTM_DRIFT_PRICE_THRESHOLD=0.10
CNN_LSTM_DRIFT_PATTERN_THRESHOLD=0.08
CNN_LSTM_DRIFT_REGIME_THRESHOLD=0.08

# Auto-Training
CNN_LSTM_AUTO_TRAIN_INTERVAL_HOURS=24
CNN_LSTM_AUTO_TRAIN_MIN_FEEDBACK=100
```

---

## 8. Implementierungsplan

### Phase 1: Multi-Task Outcome Tracking (1 Woche)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 1.1 | MultiTaskOutcome Datenstrukturen | 1 Tag |
| 1.2 | Price Outcome Tracker | 1 Tag |
| 1.3 | Pattern Outcome Tracker | 1 Tag |
| 1.4 | Regime Outcome Tracker | 1 Tag |
| 1.5 | API Endpoints + Tests | 1 Tag |

### Phase 2: Feedback Aggregation (1 Woche)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 2.1 | MultiTaskFeedbackEntry | 1 Tag |
| 2.2 | Feedback Aggregator | 2 Tage |
| 2.3 | Feedback Buffer Service | 1 Tag |
| 2.4 | Tests | 1 Tag |

### Phase 3: Multi-Task EWC Training (2 Wochen)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 3.1 | Task-spezifische Fisher Matrices | 2 Tage |
| 3.2 | Multi-Task EWC Loss | 2 Tage |
| 3.3 | Incremental Training Service | 3 Tage |
| 3.4 | Validation & Rollback | 2 Tage |
| 3.5 | Hot-Reload Integration | 1 Tag |

### Phase 4: Drift Detection (1 Woche)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 4.1 | Multi-Task Drift Detector | 2 Tage |
| 4.2 | Per-Task Thresholds | 1 Tag |
| 4.3 | Alert System | 1 Tag |
| 4.4 | Tests | 1 Tag |

### Phase 5: Orchestration & UI (1 Woche)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 5.1 | Self-Learning Orchestrator | 2 Tage |
| 5.2 | API Endpoints | 1 Tag |
| 5.3 | Frontend Dashboard | 2 Tage |

### Phase 6: Testing & Rollout (1 Woche)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 6.1 | Integration Tests | 2 Tage |
| 6.2 | Shadow Mode | 2 Tage |
| 6.3 | Documentation | 1 Tag |

**Gesamtaufwand: ~7 Wochen**

---

## 9. Unterschiede zum TCN Self-Learning

| Aspekt | TCN | CNN-LSTM |
|--------|-----|----------|
| **Tracking** | Single Pattern Outcome | 3 Task-Outcomes parallel |
| **Evaluation** | Pattern Win/Loss | Composite Score aus 3 Tasks |
| **Fisher Matrix** | Single Matrix | 4 Matrices (shared + 3 tasks) |
| **EWC Loss** | Standard EWC | Task-weighted EWC |
| **Drift Detection** | Global Drift | Per-Task + Composite Drift |
| **Validation** | Pattern Accuracy | Multi-Metric Validation |
| **Feedback Weight** | Outcome-based | Task-Consistency Bonus |

---

## 10. Frontend Integration

### 10.1 Lernfortschritt-Chart Erweiterung

Das bestehende Lernfortschritt-Chart in `config-cnn-lstm.html` sollte erweitert werden um:

1. **Task-spezifische Tabs**: Price / Pattern / Regime / Composite
2. **Outcome-Tracking Visualisierung**: Win/Loss Rate über Zeit
3. **Drift-Anzeige**: Aktueller Drift-Status mit Farbcodierung
4. **Self-Learning Controls**: Start/Stop/Trigger Buttons

### 10.2 Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠 Self-Learning Status                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Status: ✅ Aktiv    Drift: 🟢 OK (3.2%)                      ││
│  │ Tracking: 47 Predictions    Buffer: 234/10000                ││
│  │ Last Training: vor 18h      Next: in 6h                      ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  📊 Lernfortschritt                                              │
│  [Price] [Pattern] [Regime] [Composite]                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │     ╭──────────────────────────────────────╮                ││
│  │  80%│                    ___.-──╮          │                ││
│  │     │              _.──'        │          │                ││
│  │  60%│         _.──'             │          │                ││
│  │     │    _.──'                  │          │                ││
│  │  40%│ ──'                       │          │                ││
│  │     ╰─────────────────────────────────────╯                 ││
│  │        Jan 20   Jan 22   Jan 24   Jan 26                    ││
│  └─────────────────────────────────────────────────────────────┘│
│  Summary: 8 Trainings | Ø Accuracy: 67.3% | Trend: ↑ +4.2%      │
├─────────────────────────────────────────────────────────────────┤
│  [🔄 Training Starten]  [⏹️ Stoppen]  [📊 Details]              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| **Task Interference** | Mittel | Hoch | Separate Fisher Matrices pro Task |
| **Unbalanced Learning** | Mittel | Mittel | Task-Gewichtung in Loss + Feedback |
| **Slow Feedback Loop** | Hoch | Niedrig | Price-Task gibt schnelles Feedback |
| **Memory Overhead** | Mittel | Mittel | 4 Fisher Matrices statt 1 |
| **Complexity** | Hoch | Mittel | Schrittweise Implementierung |

---

## 12. Fazit

Das CNN-LSTM Self-Learning System ist komplexer als das TCN-Pendant, da drei Tasks parallel getrackt und optimiert werden müssen. Die Hauptvorteile sind:

1. **Schnelleres Feedback**: Price-Task gibt innerhalb von Stunden Feedback
2. **Robustere Validierung**: Mehrere Tasks als Cross-Check
3. **Differenzierte Drift-Detection**: Task-spezifische Probleme erkennbar

Die Implementierung sollte **nach dem TCN Self-Learning** erfolgen, um Erfahrungen aus dem einfacheren System zu nutzen.

---

## Anhang: Referenz-Implementierung

Die vollständige Referenz-Implementierung basiert auf dem TCN Self-Learning Proposal und erweitert dieses um Multi-Task-Fähigkeiten. Siehe:

- `SELF_LEARNING_TCN_PROPOSAL.md` - Basis-Architektur
- `src/services/cnn_lstm_app/` - Bestehende CNN-LSTM Implementierung
- `src/services/cnn_lstm_train_app/` - Bestehender Training Service
