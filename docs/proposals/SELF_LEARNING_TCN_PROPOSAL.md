# Self-Learning TCN Pattern Service - Proposal

**Version**: 1.0
**Datum**: 2026-01-19
**Status**: Entwurf

---

## Executive Summary

Dieses Dokument beschreibt einen Vorschlag zur Erweiterung des TCN Pattern Service zu einem **selbstlernenden System**. Das Ziel ist, dass das System kontinuierlich aus realen Marktergebnissen lernt und seine Pattern-Erkennung automatisch verbessert, ohne manuelles Retraining.

---

## 1. Problemstellung

### Aktueller Zustand

```
Training (offline) ──► Modell ──► Inference (statisch)
      │                               │
      │                               ▼
      │                        Pattern Detection
      │                               │
      └─────── manuelles Retraining ◄─┘ (kein Feedback)
```

**Limitierungen:**
- Modell bleibt statisch bis zum nächsten manuellen Training
- Kein automatisches Lernen aus Markt-Feedback
- Pattern-Qualität verschlechtert sich bei Marktveränderungen
- Claude-Validierungen werden nicht für Retraining genutzt
- Outcome-Tracking (Pattern → Kursbewegung) fehlt

---

## 2. Vision: Self-Learning Architecture

### Ziel-Architektur

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SELF-LEARNING TCN SYSTEM                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │   DETECT    │────►│   TRACK     │────►│  EVALUATE   │                    │
│  │  Patterns   │     │  Outcomes   │     │   Quality   │                    │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                    │
│         ▲                                        │                           │
│         │                                        ▼                           │
│  ┌──────┴──────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │   DEPLOY    │◄────│   TRAIN     │◄────│  FEEDBACK   │                    │
│  │  Hot-Reload │     │ Incremental │     │   Buffer    │                    │
│  └─────────────┘     └─────────────┘     └─────────────┘                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Kernprinzipien

1. **Outcome-basiertes Lernen**: Patterns werden nach tatsächlichem Kursverlauf bewertet
2. **Continuous Feedback**: Validierungsergebnisse fliessen automatisch ins Training
3. **Incremental Learning**: Modell wird inkrementell angepasst, nicht komplett neu trainiert
4. **Human-in-the-Loop**: Claude-Validierungen als zusätzliche Qualitätskontrolle
5. **Drift Detection**: Automatische Erkennung von Concept Drift

---

## 3. Komponenten-Design

### 3.1 Outcome Tracker Service

Verfolgt was nach einer Pattern-Erkennung tatsächlich passiert ist.

```python
@dataclass
class PatternOutcome:
    """Ergebnis eines erkannten Patterns nach Ablauf der Halteperiode."""

    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str                    # bullish/bearish

    # Detection Context
    detected_at: datetime
    price_at_detection: float
    confidence: float

    # Outcome Tracking
    tracking_started: datetime
    tracking_ended: datetime

    # Price Movement
    price_after_1h: Optional[float]
    price_after_4h: Optional[float]
    price_after_24h: Optional[float]
    price_after_7d: Optional[float]

    max_favorable_move: float         # Max Bewegung in erwartete Richtung
    max_adverse_move: float           # Max Bewegung gegen erwartete Richtung
    final_move: float                 # Schlusskurs vs. Detection

    # Outcome Classification
    outcome: PatternOutcomeType       # WIN / LOSS / NEUTRAL / INVALIDATED
    outcome_score: float              # -1.0 bis +1.0

    # Target Achievement
    target_hit: bool
    target_hit_time: Optional[datetime]
    stop_hit: bool
    stop_hit_time: Optional[datetime]

    # External Validation
    claude_validated: bool
    claude_agreed: bool
    claude_confidence: float


class PatternOutcomeType(Enum):
    WIN = "win"                       # Pattern erfolgreich (≥ 60% Target erreicht)
    PARTIAL_WIN = "partial_win"       # Pattern teilweise erfolgreich (30-60%)
    NEUTRAL = "neutral"               # Keine signifikante Bewegung
    LOSS = "loss"                     # Pattern falsch (Stop erreicht)
    INVALIDATED = "invalidated"       # Pattern durch Gap/News invalidiert
```

**Tracking-Logik:**

```python
class OutcomeTrackerService:
    """Verfolgt Pattern-Outcomes über konfigurierbare Zeiträume."""

    def __init__(self):
        self.tracking_periods = {
            "M15": [1, 4, 12, 48],        # Stunden
            "H1": [4, 24, 72, 168],       # Stunden
            "H4": [24, 72, 168, 336],     # Stunden
            "D1": [168, 336, 720, 2160],  # Stunden (7d, 14d, 30d, 90d)
        }
        self.active_tracking: Dict[str, PatternOutcome] = {}

    async def start_tracking(self, pattern: TCNPatternHistoryEntry):
        """Startet Outcome-Tracking für ein erkanntes Pattern."""
        outcome = PatternOutcome(
            pattern_id=pattern.id,
            symbol=pattern.symbol,
            timeframe=pattern.timeframe,
            pattern_type=pattern.pattern_type,
            direction=pattern.direction,
            detected_at=datetime.fromisoformat(pattern.detected_at),
            price_at_detection=pattern.price_at_detection,
            confidence=pattern.confidence,
            tracking_started=datetime.now(timezone.utc),
            # ... weitere Initialisierung
        )
        self.active_tracking[pattern.id] = outcome

        # Schedule Checkpoints
        for hours in self.tracking_periods.get(pattern.timeframe, [24]):
            await self._schedule_checkpoint(pattern.id, hours)

    async def _check_outcome(self, pattern_id: str) -> PatternOutcome:
        """Evaluiert den aktuellen Stand eines Patterns."""
        outcome = self.active_tracking[pattern_id]

        # Hole aktuelle Preise
        current_price = await self._get_current_price(outcome.symbol)

        # Berechne Moves
        move_pct = (current_price - outcome.price_at_detection) / outcome.price_at_detection * 100

        # Richtungs-adjustierte Bewertung
        if outcome.direction == "bullish":
            favorable_move = max(0, move_pct)
            adverse_move = min(0, move_pct)
        else:
            favorable_move = max(0, -move_pct)
            adverse_move = min(0, -move_pct)

        outcome.max_favorable_move = max(outcome.max_favorable_move, favorable_move)
        outcome.max_adverse_move = min(outcome.max_adverse_move, adverse_move)

        return outcome

    def classify_outcome(self, outcome: PatternOutcome) -> PatternOutcomeType:
        """Klassifiziert das finale Outcome."""

        # Berechne Risk-Reward basierte Scores
        if outcome.target_hit:
            return PatternOutcomeType.WIN

        if outcome.stop_hit:
            return PatternOutcomeType.LOSS

        # Prozentuale Bewegung in erwartete Richtung
        expected_target = outcome.price_target or (
            outcome.price_at_detection * (1.02 if outcome.direction == "bullish" else 0.98)
        )
        target_distance = abs(expected_target - outcome.price_at_detection)
        achieved_distance = outcome.max_favorable_move * outcome.price_at_detection / 100

        achievement_ratio = achieved_distance / target_distance if target_distance > 0 else 0

        if achievement_ratio >= 0.6:
            return PatternOutcomeType.WIN
        elif achievement_ratio >= 0.3:
            return PatternOutcomeType.PARTIAL_WIN
        elif outcome.max_adverse_move < -2.0:  # > 2% gegen uns
            return PatternOutcomeType.LOSS
        else:
            return PatternOutcomeType.NEUTRAL
```

### 3.2 Feedback Buffer

Sammelt Feedback-Daten für inkrementelles Training.

```python
@dataclass
class FeedbackEntry:
    """Ein Feedback-Eintrag für das Training."""

    # Input Data (OHLCV Sequence)
    ohlcv_sequence: np.ndarray        # (seq_len, 5)

    # Original Label (vom Training)
    original_label: np.ndarray        # (16,) Multi-label

    # Feedback Signal
    feedback_type: FeedbackType
    feedback_weight: float            # 0.0-2.0 (> 1.0 = verstärkt)

    # Zusätzliche Signale
    outcome_score: float              # -1.0 bis +1.0
    claude_confidence: Optional[float]
    market_regime: Optional[str]      # bull/bear/sideways

    timestamp: datetime


class FeedbackType(Enum):
    OUTCOME_POSITIVE = "outcome_positive"    # Pattern war korrekt
    OUTCOME_NEGATIVE = "outcome_negative"    # Pattern war falsch
    CLAUDE_CONFIRMED = "claude_confirmed"    # Claude hat zugestimmt
    CLAUDE_REJECTED = "claude_rejected"      # Claude hat abgelehnt
    MANUAL_CORRECTION = "manual_correction"  # Manuelles Feedback
    DRIFT_DETECTED = "drift_detected"        # Concept Drift erkannt


class FeedbackBuffer:
    """Puffert Feedback-Daten für Batch-Training."""

    def __init__(
        self,
        max_size: int = 10000,
        min_batch_size: int = 100,
        max_age_hours: int = 168,  # 7 Tage
    ):
        self.buffer: List[FeedbackEntry] = []
        self.max_size = max_size
        self.min_batch_size = min_batch_size
        self.max_age_hours = max_age_hours
        self.persistence_path = Path("data/feedback/tcn_feedback_buffer.json")

    async def add_feedback(self, entry: FeedbackEntry):
        """Fügt Feedback zum Buffer hinzu."""
        self.buffer.append(entry)

        # Cleanup alte Einträge
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        self.buffer = [e for e in self.buffer if e.timestamp > cutoff]

        # LRU wenn zu voll
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

        # Persistieren
        await self._persist()

    def get_training_batch(self) -> Optional[List[FeedbackEntry]]:
        """Holt einen Batch für Training wenn genug Daten vorhanden."""
        if len(self.buffer) < self.min_batch_size:
            return None

        # Stratified Sampling nach Feedback-Typ
        batch = self._stratified_sample(self.min_batch_size)
        return batch

    def _stratified_sample(self, n: int) -> List[FeedbackEntry]:
        """Stratified Sampling um Balance zu gewährleisten."""
        by_type: Dict[FeedbackType, List[FeedbackEntry]] = {}
        for entry in self.buffer:
            by_type.setdefault(entry.feedback_type, []).append(entry)

        # Proportional sampling
        result = []
        for feedback_type, entries in by_type.items():
            count = max(1, int(n * len(entries) / len(self.buffer)))
            result.extend(random.sample(entries, min(count, len(entries))))

        return result[:n]
```

### 3.3 Incremental Training Service

Führt inkrementelles Training durch ohne das Modell komplett neu zu trainieren.

```python
class IncrementalTrainingService:
    """Inkrementelles Training basierend auf Feedback."""

    def __init__(
        self,
        model_path: str = "data/models/tcn/latest.pt",
        learning_rate: float = 1e-5,      # Sehr kleine LR für Fine-Tuning
        max_samples_per_update: int = 500,
        min_samples_threshold: int = 100,
    ):
        self.model_path = Path(model_path)
        self.learning_rate = learning_rate
        self.max_samples = max_samples_per_update
        self.min_samples = min_samples_threshold

        # Elastic Weight Consolidation für Catastrophic Forgetting Prevention
        self.ewc_lambda = 1000
        self.fisher_matrix: Optional[Dict[str, torch.Tensor]] = None
        self.optimal_params: Optional[Dict[str, torch.Tensor]] = None

    async def incremental_update(
        self,
        feedback_batch: List[FeedbackEntry],
        validation_threshold: float = 0.02,  # Max 2% Performance-Verlust
    ) -> IncrementalUpdateResult:
        """
        Führt ein inkrementelles Update durch.

        Returns:
            IncrementalUpdateResult mit Metriken und Status
        """
        if len(feedback_batch) < self.min_samples:
            return IncrementalUpdateResult(
                success=False,
                reason="Nicht genug Samples",
                samples_count=len(feedback_batch)
            )

        # 1. Lade aktuelles Modell
        model, old_metrics = await self._load_model()

        # 2. Erstelle Training-Dataset aus Feedback
        train_loader, val_loader = self._create_dataloaders(feedback_batch)

        # 3. Speichere Fisher Matrix für EWC (falls nicht vorhanden)
        if self.fisher_matrix is None:
            self.fisher_matrix = self._compute_fisher_matrix(model, val_loader)
            self.optimal_params = {n: p.clone() for n, p in model.named_parameters()}

        # 4. Fine-Tune mit EWC Regularization
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        model.train()
        for epoch in range(3):  # Wenige Epochen für Fine-Tuning
            for batch in train_loader:
                optimizer.zero_grad()

                # Forward Pass
                x, y, weights = batch
                predictions = model(x)

                # Task Loss (gewichtet nach Feedback)
                task_loss = self._weighted_bce_loss(predictions, y, weights)

                # EWC Loss (verhindert Catastrophic Forgetting)
                ewc_loss = self._ewc_loss(model)

                # Total Loss
                loss = task_loss + self.ewc_lambda * ewc_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # 5. Validierung
        new_metrics = await self._validate_model(model, val_loader)

        # 6. Performance Check
        performance_drop = old_metrics["val_auc"] - new_metrics["val_auc"]

        if performance_drop > validation_threshold:
            logger.warning(
                f"Performance drop {performance_drop:.3f} > threshold {validation_threshold}. "
                "Update rejected."
            )
            return IncrementalUpdateResult(
                success=False,
                reason=f"Performance drop too high: {performance_drop:.3f}",
                old_metrics=old_metrics,
                new_metrics=new_metrics
            )

        # 7. Speichere neues Modell
        await self._save_model(model, new_metrics)

        # 8. Update Fisher Matrix
        self.fisher_matrix = self._compute_fisher_matrix(model, val_loader)
        self.optimal_params = {n: p.clone() for n, p in model.named_parameters()}

        # 9. Notify Inference Service
        await self._notify_model_reload()

        return IncrementalUpdateResult(
            success=True,
            samples_count=len(feedback_batch),
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            improvement=new_metrics["val_auc"] - old_metrics["val_auc"]
        )

    def _ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Elastic Weight Consolidation Loss."""
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal).pow(2)).sum()
        return loss

    def _compute_fisher_matrix(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """Berechnet Fisher Information Matrix für EWC."""
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        model.eval()
        for x, y, _ in data_loader:
            model.zero_grad()
            output = model(x)
            loss = F.binary_cross_entropy_with_logits(output, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

        # Normalize
        for name in fisher:
            fisher[name] /= len(data_loader)

        return fisher
```

### 3.4 Drift Detection Service

Erkennt wenn das Modell nicht mehr zur aktuellen Marktlage passt.

```python
class DriftDetectionService:
    """Erkennt Concept Drift im Pattern Recognition."""

    def __init__(
        self,
        window_size: int = 100,       # Anzahl Patterns für Drift-Berechnung
        drift_threshold: float = 0.15, # 15% Accuracy Drop = Drift
        warning_threshold: float = 0.08,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold

        self.reference_distribution: Optional[Dict] = None
        self.current_window: List[PatternOutcome] = []

    async def check_drift(self, outcome: PatternOutcome) -> DriftStatus:
        """Prüft auf Concept Drift nach jedem neuen Outcome."""

        self.current_window.append(outcome)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

        if len(self.current_window) < self.window_size // 2:
            return DriftStatus(detected=False, message="Not enough data")

        # Berechne aktuelle Performance-Metriken
        current_metrics = self._compute_metrics(self.current_window)

        if self.reference_distribution is None:
            self.reference_distribution = current_metrics
            return DriftStatus(detected=False, message="Reference established")

        # Vergleiche mit Referenz
        drift_score = self._compute_drift_score(
            self.reference_distribution,
            current_metrics
        )

        if drift_score > self.drift_threshold:
            return DriftStatus(
                detected=True,
                severity="HIGH",
                drift_score=drift_score,
                message=f"Significant drift detected: {drift_score:.2%}",
                recommended_action="FULL_RETRAIN"
            )
        elif drift_score > self.warning_threshold:
            return DriftStatus(
                detected=True,
                severity="WARNING",
                drift_score=drift_score,
                message=f"Moderate drift detected: {drift_score:.2%}",
                recommended_action="INCREMENTAL_UPDATE"
            )

        return DriftStatus(detected=False, drift_score=drift_score)

    def _compute_metrics(self, outcomes: List[PatternOutcome]) -> Dict:
        """Berechnet Performance-Metriken für ein Outcome-Window."""

        win_count = sum(1 for o in outcomes if o.outcome == PatternOutcomeType.WIN)
        loss_count = sum(1 for o in outcomes if o.outcome == PatternOutcomeType.LOSS)
        total = len(outcomes)

        # Per-Pattern-Type Accuracy
        by_pattern: Dict[str, List[PatternOutcome]] = {}
        for o in outcomes:
            by_pattern.setdefault(o.pattern_type, []).append(o)

        pattern_accuracies = {
            pt: sum(1 for o in outcomes if o.outcome in [PatternOutcomeType.WIN, PatternOutcomeType.PARTIAL_WIN]) / len(outcomes)
            for pt, outcomes in by_pattern.items()
            if len(outcomes) >= 5
        }

        return {
            "win_rate": win_count / total if total > 0 else 0,
            "loss_rate": loss_count / total if total > 0 else 0,
            "avg_outcome_score": np.mean([o.outcome_score for o in outcomes]),
            "pattern_accuracies": pattern_accuracies,
            "confidence_calibration": self._compute_calibration(outcomes),
        }

    def _compute_drift_score(self, reference: Dict, current: Dict) -> float:
        """Berechnet einen Drift-Score zwischen 0 und 1."""

        # Win Rate Drift
        win_rate_drift = abs(reference["win_rate"] - current["win_rate"])

        # Outcome Score Drift
        score_drift = abs(reference["avg_outcome_score"] - current["avg_outcome_score"])

        # Pattern-wise Drift (falls genug Daten)
        pattern_drifts = []
        for pt in reference.get("pattern_accuracies", {}):
            if pt in current.get("pattern_accuracies", {}):
                drift = abs(
                    reference["pattern_accuracies"][pt] -
                    current["pattern_accuracies"][pt]
                )
                pattern_drifts.append(drift)

        pattern_drift = np.mean(pattern_drifts) if pattern_drifts else 0

        # Gewichteter Gesamtscore
        return 0.4 * win_rate_drift + 0.3 * score_drift + 0.3 * pattern_drift
```

### 3.5 Self-Learning Orchestrator

Koordiniert alle Komponenten zu einem Gesamtsystem.

```python
class SelfLearningOrchestrator:
    """Orchestriert das gesamte Self-Learning System."""

    def __init__(self):
        self.outcome_tracker = OutcomeTrackerService()
        self.feedback_buffer = FeedbackBuffer()
        self.incremental_trainer = IncrementalTrainingService()
        self.drift_detector = DriftDetectionService()

        # Konfiguration
        self.auto_train_enabled = True
        self.auto_train_interval_hours = 24
        self.min_feedback_for_train = 100
        self.last_auto_train: Optional[datetime] = None

    async def on_pattern_detected(self, pattern: TCNPatternHistoryEntry):
        """Callback wenn ein neues Pattern erkannt wurde."""

        # Starte Outcome-Tracking
        await self.outcome_tracker.start_tracking(pattern)
        logger.info(f"Started tracking pattern {pattern.id}: {pattern.pattern_type}")

    async def on_outcome_complete(self, outcome: PatternOutcome):
        """Callback wenn ein Pattern-Outcome abgeschlossen ist."""

        # 1. Klassifiziere Outcome
        outcome.outcome = self.outcome_tracker.classify_outcome(outcome)
        outcome.outcome_score = self._compute_outcome_score(outcome)

        # 2. Erstelle Feedback-Entry
        feedback = await self._create_feedback_entry(outcome)
        await self.feedback_buffer.add_feedback(feedback)

        # 3. Check Drift
        drift_status = await self.drift_detector.check_drift(outcome)
        if drift_status.detected:
            await self._handle_drift(drift_status)

        # 4. Check Auto-Train Trigger
        if self._should_auto_train():
            await self._trigger_incremental_training()

        logger.info(
            f"Pattern {outcome.pattern_id} outcome: {outcome.outcome.value} "
            f"(score: {outcome.outcome_score:.2f})"
        )

    async def on_claude_validation(self, validation: TCNClaudeValidationResult):
        """Callback wenn Claude eine Validierung abgeschlossen hat."""

        # Erstelle Feedback aus Claude-Validierung
        feedback = FeedbackEntry(
            ohlcv_sequence=await self._get_pattern_sequence(validation.pattern_id),
            original_label=await self._get_pattern_label(validation.pattern_type),
            feedback_type=(
                FeedbackType.CLAUDE_CONFIRMED if validation.claude_agrees
                else FeedbackType.CLAUDE_REJECTED
            ),
            feedback_weight=1.0 + validation.claude_confidence,  # Höheres Gewicht bei hoher Confidence
            outcome_score=1.0 if validation.claude_agrees else -1.0,
            claude_confidence=validation.claude_confidence,
            timestamp=datetime.now(timezone.utc)
        )

        await self.feedback_buffer.add_feedback(feedback)

    async def _trigger_incremental_training(self):
        """Triggert inkrementelles Training."""

        batch = self.feedback_buffer.get_training_batch()
        if batch is None:
            logger.info("Not enough feedback for training")
            return

        logger.info(f"Starting incremental training with {len(batch)} samples")

        result = await self.incremental_trainer.incremental_update(batch)

        if result.success:
            logger.info(
                f"Incremental training successful. "
                f"Improvement: {result.improvement:+.4f}"
            )
            # Clear used feedback
            self.feedback_buffer.mark_as_used(batch)
        else:
            logger.warning(f"Incremental training failed: {result.reason}")

        self.last_auto_train = datetime.now(timezone.utc)

    async def _handle_drift(self, drift_status: DriftStatus):
        """Reagiert auf erkannten Concept Drift."""

        if drift_status.severity == "HIGH":
            # Bei schwerem Drift: Volles Retraining triggern
            logger.warning(f"High drift detected! Triggering full retrain.")
            await self._request_full_retrain()
        else:
            # Bei moderatem Drift: Sofortiges inkrementelles Update
            logger.info(f"Moderate drift detected. Triggering immediate update.")
            await self._trigger_incremental_training()

    def _should_auto_train(self) -> bool:
        """Prüft ob Auto-Training ausgelöst werden soll."""

        if not self.auto_train_enabled:
            return False

        if len(self.feedback_buffer.buffer) < self.min_feedback_for_train:
            return False

        if self.last_auto_train is None:
            return True

        hours_since_last = (
            datetime.now(timezone.utc) - self.last_auto_train
        ).total_seconds() / 3600

        return hours_since_last >= self.auto_train_interval_hours
```

---

## 4. Datenfluss

### 4.1 Kompletter Feedback-Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FEEDBACK LOOP                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PATTERN DETECTION                                                       │
│     │                                                                       │
│     ├──► TCN Model predicts patterns                                        │
│     │    └──► Pattern stored in History                                     │
│     │                                                                       │
│  2. OUTCOME TRACKING                                                        │
│     │                                                                       │
│     ├──► OutcomeTracker starts monitoring                                   │
│     │    ├──► Checkpoint 1: 1 hour later                                    │
│     │    ├──► Checkpoint 2: 4 hours later                                   │
│     │    ├──► Checkpoint 3: 24 hours later                                  │
│     │    └──► Final: 7 days later                                           │
│     │                                                                       │
│  3. OUTCOME CLASSIFICATION                                                  │
│     │                                                                       │
│     ├──► WIN: Target reached (≥60%)                                         │
│     ├──► PARTIAL_WIN: Partial target (30-60%)                               │
│     ├──► NEUTRAL: No significant move                                       │
│     └──► LOSS: Stop hit or wrong direction                                  │
│                                                                             │
│  4. FEEDBACK GENERATION                                                     │
│     │                                                                       │
│     ├──► Positive Feedback: WIN/PARTIAL_WIN → weight 1.0-1.5               │
│     ├──► Negative Feedback: LOSS → weight 1.0-2.0 (verstärkt)              │
│     └──► Neutral: Skip or low weight                                        │
│                                                                             │
│  5. DRIFT DETECTION                                                         │
│     │                                                                       │
│     ├──► Compare current window vs reference                                │
│     ├──► WARNING: 8-15% drop → Incremental Update                          │
│     └──► HIGH: >15% drop → Full Retrain                                    │
│                                                                             │
│  6. INCREMENTAL TRAINING                                                    │
│     │                                                                       │
│     ├──► Batch from Feedback Buffer (≥100 samples)                         │
│     ├──► Fine-tune with EWC regularization                                  │
│     ├──► Validate: Max 2% performance drop                                  │
│     └──► Hot-reload new model                                               │
│                                                                             │
│  7. CONTINUOUS IMPROVEMENT                                                  │
│     │                                                                       │
│     └──► Loop back to Step 1 with improved model                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Claude Integration

```
Pattern Detection
      │
      ▼
┌─────────────────────┐
│ Pattern Confidence  │
│    < 0.7 ?          │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼ Yes       ▼ No
 Request      Skip Claude
 Claude       Validation
 Validation
     │
     ▼
┌─────────────────────┐
│ Claude Vision API   │
│ "Is this a valid    │
│  {pattern_type}?"   │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
 Agrees       Disagrees
     │           │
     ▼           ▼
 Positive    Negative
 Feedback    Feedback
 (weight     (weight
  +0.5)       +1.0)
```

---

## 5. Konfiguration

### 5.1 Environment Variables

```bash
# Self-Learning Feature Flags
SELF_LEARNING_ENABLED=true
OUTCOME_TRACKING_ENABLED=true
INCREMENTAL_TRAINING_ENABLED=true
DRIFT_DETECTION_ENABLED=true

# Outcome Tracking
OUTCOME_TRACKING_PERIODS_H1="4,24,72,168"  # Stunden
OUTCOME_WIN_THRESHOLD=0.6                    # 60% Target = WIN
OUTCOME_PARTIAL_THRESHOLD=0.3                # 30% Target = PARTIAL

# Feedback Buffer
FEEDBACK_BUFFER_MAX_SIZE=10000
FEEDBACK_BUFFER_MIN_BATCH=100
FEEDBACK_BUFFER_MAX_AGE_HOURS=168            # 7 Tage

# Incremental Training
INCREMENTAL_LEARNING_RATE=0.00001           # Sehr kleine LR
INCREMENTAL_EWC_LAMBDA=1000                  # EWC Regularisierung
INCREMENTAL_VALIDATION_THRESHOLD=0.02        # Max 2% Drop erlaubt

# Drift Detection
DRIFT_WINDOW_SIZE=100
DRIFT_THRESHOLD_HIGH=0.15                    # 15% = Full Retrain
DRIFT_THRESHOLD_WARNING=0.08                 # 8% = Incremental Update

# Auto-Training
AUTO_TRAIN_INTERVAL_HOURS=24
AUTO_TRAIN_MIN_FEEDBACK=100
```

### 5.2 Pydantic Settings

```python
class SelfLearningSettings(BaseSettings):
    """Self-Learning Konfiguration."""

    # Feature Flags
    enabled: bool = Field(default=True, env="SELF_LEARNING_ENABLED")
    outcome_tracking_enabled: bool = Field(default=True)
    incremental_training_enabled: bool = Field(default=True)
    drift_detection_enabled: bool = Field(default=True)

    # Outcome Tracking
    outcome_win_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    outcome_partial_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Feedback
    feedback_max_size: int = Field(default=10000, ge=100)
    feedback_min_batch: int = Field(default=100, ge=10)
    feedback_max_age_hours: int = Field(default=168, ge=24)

    # Training
    incremental_learning_rate: float = Field(default=1e-5, gt=0)
    incremental_ewc_lambda: float = Field(default=1000, ge=0)
    incremental_validation_threshold: float = Field(default=0.02, ge=0, le=0.1)

    # Drift
    drift_window_size: int = Field(default=100, ge=20)
    drift_threshold_high: float = Field(default=0.15, ge=0, le=1)
    drift_threshold_warning: float = Field(default=0.08, ge=0, le=1)

    # Auto-Train
    auto_train_interval_hours: int = Field(default=24, ge=1)
    auto_train_min_feedback: int = Field(default=100, ge=10)

    class Config:
        env_prefix = "SELF_LEARNING_"
```

---

## 6. API Erweiterungen

### 6.1 Neue Endpoints

| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/api/v1/self-learning/status` | GET | Status des Self-Learning Systems |
| `/api/v1/self-learning/config` | GET/PUT | Konfiguration lesen/ändern |
| `/api/v1/self-learning/feedback/stats` | GET | Feedback-Buffer Statistiken |
| `/api/v1/self-learning/drift/status` | GET | Aktueller Drift-Status |
| `/api/v1/self-learning/training/trigger` | POST | Manuell Training triggern |
| `/api/v1/outcomes/{pattern_id}` | GET | Outcome für Pattern abrufen |
| `/api/v1/outcomes/statistics` | GET | Aggregierte Outcome-Statistiken |

### 6.2 Response Models

```python
class SelfLearningStatus(BaseModel):
    """Status des Self-Learning Systems."""

    enabled: bool
    outcome_tracking_active: bool
    patterns_being_tracked: int

    feedback_buffer_size: int
    feedback_ready_for_training: bool

    drift_status: str                    # "OK" | "WARNING" | "HIGH"
    drift_score: float

    last_incremental_train: Optional[datetime]
    next_scheduled_train: Optional[datetime]

    model_version: str
    total_incremental_updates: int


class OutcomeStatistics(BaseModel):
    """Aggregierte Outcome-Statistiken."""

    total_patterns_tracked: int
    total_completed: int

    win_rate: float
    partial_win_rate: float
    loss_rate: float
    neutral_rate: float

    avg_outcome_score: float

    by_pattern_type: Dict[str, PatternTypeStats]
    by_timeframe: Dict[str, TimeframeStats]

    period_start: datetime
    period_end: datetime
```

---

## 7. Datenbankschema

### 7.1 Neue Tabellen (TimescaleDB)

```sql
-- Pattern Outcomes
CREATE TABLE pattern_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,

    -- Detection
    detected_at TIMESTAMPTZ NOT NULL,
    price_at_detection DECIMAL(20, 8) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,

    -- Tracking
    tracking_started TIMESTAMPTZ NOT NULL,
    tracking_ended TIMESTAMPTZ,

    -- Price Movement
    price_after_1h DECIMAL(20, 8),
    price_after_4h DECIMAL(20, 8),
    price_after_24h DECIMAL(20, 8),
    price_after_7d DECIMAL(20, 8),

    max_favorable_move DECIMAL(10, 4),
    max_adverse_move DECIMAL(10, 4),
    final_move DECIMAL(10, 4),

    -- Outcome
    outcome VARCHAR(20),
    outcome_score DECIMAL(5, 4),

    -- Target
    target_hit BOOLEAN DEFAULT FALSE,
    target_hit_time TIMESTAMPTZ,
    stop_hit BOOLEAN DEFAULT FALSE,
    stop_hit_time TIMESTAMPTZ,

    -- Claude
    claude_validated BOOLEAN DEFAULT FALSE,
    claude_agreed BOOLEAN,
    claude_confidence DECIMAL(5, 4),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hypertable für Time-Series Optimierung
SELECT create_hypertable('pattern_outcomes', 'detected_at');

-- Indices
CREATE INDEX idx_outcomes_symbol_time ON pattern_outcomes (symbol, detected_at DESC);
CREATE INDEX idx_outcomes_pattern_type ON pattern_outcomes (pattern_type, detected_at DESC);
CREATE INDEX idx_outcomes_outcome ON pattern_outcomes (outcome, detected_at DESC);


-- Feedback Buffer (optional, kann auch Redis sein)
CREATE TABLE feedback_buffer (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id VARCHAR(64),

    ohlcv_sequence BYTEA NOT NULL,        -- Pickled numpy array
    original_label BYTEA NOT NULL,

    feedback_type VARCHAR(30) NOT NULL,
    feedback_weight DECIMAL(5, 4) NOT NULL,
    outcome_score DECIMAL(5, 4),
    claude_confidence DECIMAL(5, 4),
    market_regime VARCHAR(20),

    used_for_training BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_unused ON feedback_buffer (used_for_training, created_at)
    WHERE NOT used_for_training;


-- Training History
CREATE TABLE incremental_training_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    training_type VARCHAR(20) NOT NULL,  -- "incremental" | "full"
    samples_count INT NOT NULL,

    -- Metrics
    old_val_auc DECIMAL(6, 5),
    new_val_auc DECIMAL(6, 5),
    improvement DECIMAL(6, 5),

    -- Config used
    learning_rate DECIMAL(10, 8),
    ewc_lambda DECIMAL(10, 2),

    -- Status
    success BOOLEAN NOT NULL,
    failure_reason TEXT,

    model_version_before VARCHAR(100),
    model_version_after VARCHAR(100),

    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);


-- Drift Detection Log
CREATE TABLE drift_detection_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    drift_score DECIMAL(6, 5) NOT NULL,
    severity VARCHAR(10),                -- "OK" | "WARNING" | "HIGH"

    window_size INT NOT NULL,
    win_rate DECIMAL(5, 4),
    loss_rate DECIMAL(5, 4),
    avg_outcome_score DECIMAL(5, 4),

    action_taken VARCHAR(30),            -- "NONE" | "INCREMENTAL" | "FULL_RETRAIN"

    detected_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('drift_detection_log', 'detected_at');
```

---

## 8. Implementierungsplan

### Phase 1: Foundation (2 Wochen)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 1.1 | Outcome Tracker Service implementieren | 3 Tage |
| 1.2 | Database Schema Migration | 1 Tag |
| 1.3 | API Endpoints für Outcomes | 2 Tage |
| 1.4 | Pattern History Integration | 2 Tage |
| 1.5 | Unit Tests | 2 Tage |

### Phase 2: Feedback System (2 Wochen)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 2.1 | Feedback Buffer Service | 2 Tage |
| 2.2 | Claude Validation Integration | 2 Tage |
| 2.3 | Feedback Weight Berechnung | 1 Tag |
| 2.4 | Persistierung (Redis/DB) | 2 Tage |
| 2.5 | API Endpoints | 1 Tag |
| 2.6 | Unit Tests | 2 Tage |

### Phase 3: Incremental Training (2 Wochen)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 3.1 | EWC Implementation | 3 Tage |
| 3.2 | Incremental Training Service | 3 Tage |
| 3.3 | Validation & Rollback | 2 Tage |
| 3.4 | Hot-Reload Integration | 1 Tag |
| 3.5 | Unit + Integration Tests | 3 Tage |

### Phase 4: Drift Detection (1 Woche)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 4.1 | Drift Detection Service | 2 Tage |
| 4.2 | Alert System | 1 Tag |
| 4.3 | Auto-Retrain Trigger | 1 Tag |
| 4.4 | Tests | 1 Tag |

### Phase 5: Orchestration (1 Woche)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 5.1 | Self-Learning Orchestrator | 3 Tage |
| 5.2 | Watchdog Integration | 1 Tag |
| 5.3 | Frontend Dashboard | 1 Tag |

### Phase 6: Testing & Rollout (2 Wochen)

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 6.1 | End-to-End Tests | 3 Tage |
| 6.2 | Shadow Mode (parallel zu Prod) | 3 Tage |
| 6.3 | Performance Testing | 2 Tage |
| 6.4 | Documentation | 2 Tage |

**Gesamtaufwand: ~10 Wochen**

---

## 9. Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| **Catastrophic Forgetting** | Mittel | Hoch | EWC Regularisierung, Validation Threshold |
| **Feedback Noise** | Hoch | Mittel | Gewichtung, Stratified Sampling, Claude QA |
| **Drift False Positives** | Mittel | Niedrig | Konservative Thresholds, Manual Override |
| **Training Instabilität** | Niedrig | Hoch | Kleine Learning Rate, Gradient Clipping |
| **Speicherverbrauch** | Mittel | Mittel | Buffer Limits, LRU Eviction |
| **API Latenz** | Niedrig | Mittel | Async Processing, Background Training |

---

## 10. Monitoring & Alerting

### 10.1 Key Metrics

```python
# Prometheus Metrics
METRICS = {
    # Outcome Tracking
    "tcn_patterns_tracked_total": Counter,
    "tcn_outcomes_completed_total": Counter,
    "tcn_outcome_win_rate": Gauge,
    "tcn_outcome_loss_rate": Gauge,

    # Feedback
    "tcn_feedback_buffer_size": Gauge,
    "tcn_feedback_entries_total": Counter,

    # Training
    "tcn_incremental_updates_total": Counter,
    "tcn_incremental_update_success_rate": Gauge,
    "tcn_model_version": Info,

    # Drift
    "tcn_drift_score": Gauge,
    "tcn_drift_alerts_total": Counter,
}
```

### 10.2 Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| `TCNDriftHigh` | drift_score > 0.15 für > 1h | Critical |
| `TCNWinRateDrop` | win_rate < 0.4 für > 24h | Warning |
| `TCNTrainingFailed` | 3 consecutive failures | Warning |
| `TCNFeedbackBufferFull` | buffer_size > 9000 | Info |

---

## 11. Fazit

### Vorteile

1. **Kontinuierliche Verbesserung**: System lernt automatisch aus Markt-Feedback
2. **Adaptiv**: Reagiert auf Marktveränderungen (Concept Drift)
3. **Robust**: EWC verhindert Catastrophic Forgetting
4. **Transparent**: Vollständiges Outcome-Tracking und Logging
5. **Human-in-the-Loop**: Claude-Validierung als Qualitätskontrolle

### Nachteile

1. **Komplexität**: Signifikant mehr Code und Komponenten
2. **Latenz**: Outcome-Tracking benötigt Zeit (Tage bis Wochen)
3. **Ressourcen**: Zusätzlicher Speicher für Feedback Buffer
4. **Risiko**: Potenzielle Modell-Degradation bei schlechtem Feedback

### Empfehlung

Das Self-Learning System sollte **schrittweise** eingeführt werden:

1. **Phase 1**: Nur Outcome-Tracking (passiv, keine Modell-Änderungen)
2. **Phase 2**: Drift Detection + Alerting (passiv, nur Monitoring)
3. **Phase 3**: Incremental Training im Shadow Mode (parallel zu Prod)
4. **Phase 4**: Full Activation nach erfolgreicher Validierung

---

## Anhang A: Glossar

| Begriff | Beschreibung |
|---------|--------------|
| **Catastrophic Forgetting** | Verlust von gelerntem Wissen beim Lernen neuer Aufgaben |
| **Concept Drift** | Veränderung der statistischen Eigenschaften der Zieldaten über Zeit |
| **EWC** | Elastic Weight Consolidation - Methode zur Vermeidung von Catastrophic Forgetting |
| **Fisher Information** | Mass für die Information, die eine Beobachtung über einen Parameter enthält |
| **Hot-Reload** | Aktualisierung eines Modells ohne Service-Neustart |
| **Incremental Learning** | Lernen aus neuen Daten ohne komplettes Retraining |

---

## Anhang B: Referenzen

1. Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS.
2. Zenke, F., Poole, B., & Ganguli, S. (2017). "Continual Learning Through Synaptic Intelligence." ICML.
3. Lu, J., et al. (2018). "Learning under Concept Drift: A Review." IEEE TKDE.
