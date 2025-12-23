# Multi-Model LLM Implementierungsplan

## Übersicht

Integration einer Modell-Hierarchie für optimierte Latenz, Kosten und Ressourcennutzung.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL ROUTER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   FAST      │  │   MEDIUM    │  │       COMPLEX           │  │
│  │ llama3.1:8b │  │ llama3.1:8b │  │    llama3.1:70b         │  │
│  │  ~200ms     │  │  ~500ms     │  │      ~2-3s              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│        │                │                      │                 │
│        ▼                ▼                      ▼                 │
│  - Sentiment      - RAG Queries        - Trading Strategien     │
│  - Klassifikation - Zusammenfassungen  - Komplexe Analysen      │
│  - Entity Extract - Pattern-Beschreib. - Multi-Step Reasoning   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Grundlagen (Woche 1)

### 1.1 Model Registry Service

**Neue Datei:** `src/services/model_registry.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import asyncio
from ollama import AsyncClient

class ModelTier(Enum):
    FAST = "fast"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class ModelConfig:
    name: str
    tier: ModelTier
    context_window: int
    avg_latency_ms: int
    cost_factor: float  # Relativ zu 8B = 1.0
    max_tokens: int
    temperature: float
    capabilities: list[str]

class ModelRegistry:
    """Zentrale Registry für alle verfügbaren LLM-Modelle."""

    MODELS: dict[str, ModelConfig] = {
        "llama3.1:8b": ModelConfig(
            name="llama3.1:8b",
            tier=ModelTier.FAST,
            context_window=8192,
            avg_latency_ms=200,
            cost_factor=1.0,
            max_tokens=2048,
            temperature=0.1,
            capabilities=["sentiment", "classification", "extraction", "summary_short"]
        ),
        "llama3.1:70b": ModelConfig(
            name="llama3.1:70b",
            tier=ModelTier.COMPLEX,
            context_window=8192,
            avg_latency_ms=2500,
            cost_factor=10.0,
            max_tokens=4096,
            temperature=0.1,
            capabilities=["strategy", "analysis", "reasoning", "code", "summary_long"]
        ),
        # Optional: Für Jetson Orin oder Cloud
        "qwen2.5:7b": ModelConfig(
            name="qwen2.5:7b",
            tier=ModelTier.FAST,
            context_window=32768,
            avg_latency_ms=180,
            cost_factor=0.8,
            max_tokens=2048,
            temperature=0.1,
            capabilities=["sentiment", "classification", "extraction", "multilingual"]
        ),
    }

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self._available_models: set[str] = set()
        self._model_stats: dict[str, dict] = {}

    async def refresh_available_models(self) -> list[str]:
        """Aktualisiert Liste verfügbarer Modelle von Ollama."""
        client = AsyncClient(host=self.ollama_host)
        response = await client.list()
        self._available_models = {
            m.get("name") or m.get("model")
            for m in response.get("models", [])
        }
        return list(self._available_models)

    def get_model_for_tier(self, tier: ModelTier) -> Optional[ModelConfig]:
        """Gibt das beste verfügbare Modell für einen Tier zurück."""
        candidates = [
            config for name, config in self.MODELS.items()
            if config.tier == tier and name in self._available_models
        ]
        if not candidates:
            return None
        # Sortiere nach Latenz (schnellstes zuerst)
        return min(candidates, key=lambda c: c.avg_latency_ms)

    def get_fallback_model(self, tier: ModelTier) -> Optional[ModelConfig]:
        """Fallback: Nächst-niedrigerer Tier wenn gewünschter nicht verfügbar."""
        tier_order = [ModelTier.COMPLEX, ModelTier.MEDIUM, ModelTier.FAST]
        start_idx = tier_order.index(tier)

        for fallback_tier in tier_order[start_idx:]:
            model = self.get_model_for_tier(fallback_tier)
            if model:
                return model
        return None
```

### 1.2 Task Classifier

**Neue Datei:** `src/services/task_classifier.py`

```python
import re
from enum import Enum
from dataclasses import dataclass
from .model_registry import ModelTier

class TaskType(Enum):
    SENTIMENT = "sentiment"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARY = "summary"
    RAG_QUERY = "rag_query"
    TRADING_ANALYSIS = "trading_analysis"
    STRATEGY = "strategy"
    CHAT = "chat"

@dataclass
class ClassifiedTask:
    task_type: TaskType
    model_tier: ModelTier
    confidence: float
    reasoning: str

class TaskClassifier:
    """Klassifiziert eingehende Anfragen nach Komplexität."""

    # Keyword-basierte Heuristiken (Phase 1)
    TASK_PATTERNS: dict[TaskType, tuple[list[str], ModelTier]] = {
        TaskType.SENTIMENT: (
            ["sentiment", "stimmung", "bullish", "bearish", "fear", "greed"],
            ModelTier.FAST
        ),
        TaskType.CLASSIFICATION: (
            ["klassifizier", "categorize", "einordnen", "typ", "art"],
            ModelTier.FAST
        ),
        TaskType.EXTRACTION: (
            ["extrahier", "extract", "finde", "liste", "nenne"],
            ModelTier.FAST
        ),
        TaskType.SUMMARY: (
            ["zusammenfass", "summarize", "überblick", "kurz"],
            ModelTier.MEDIUM
        ),
        TaskType.RAG_QUERY: (
            ["was ist", "erkläre", "warum", "wie funktioniert"],
            ModelTier.MEDIUM
        ),
        TaskType.TRADING_ANALYSIS: (
            ["analyse", "analyze", "prognose", "forecast", "einschätzung"],
            ModelTier.COMPLEX
        ),
        TaskType.STRATEGY: (
            ["strategie", "strategy", "plan", "vorgehen", "handelssystem"],
            ModelTier.COMPLEX
        ),
    }

    # Komplexitäts-Indikatoren
    COMPLEXITY_INDICATORS = {
        "high": [
            "detailliert", "ausführlich", "step-by-step", "begründe",
            "vergleiche", "trade-offs", "risiken", "szenarien"
        ],
        "low": [
            "kurz", "schnell", "einfach", "nur", "ja/nein", "one-liner"
        ]
    }

    def classify(self, query: str, context: dict | None = None) -> ClassifiedTask:
        """Klassifiziert eine Query nach Task-Typ und Modell-Tier."""
        query_lower = query.lower()

        # 1. Pattern Matching
        for task_type, (patterns, default_tier) in self.TASK_PATTERNS.items():
            if any(p in query_lower for p in patterns):
                tier = self._adjust_tier_by_complexity(query_lower, default_tier)
                return ClassifiedTask(
                    task_type=task_type,
                    model_tier=tier,
                    confidence=0.8,
                    reasoning=f"Matched pattern for {task_type.value}"
                )

        # 2. Längen-basierte Heuristik
        if len(query) < 50:
            return ClassifiedTask(
                task_type=TaskType.CHAT,
                model_tier=ModelTier.FAST,
                confidence=0.5,
                reasoning="Short query, using fast model"
            )
        elif len(query) > 500:
            return ClassifiedTask(
                task_type=TaskType.TRADING_ANALYSIS,
                model_tier=ModelTier.COMPLEX,
                confidence=0.6,
                reasoning="Long query with context, using complex model"
            )

        # 3. Default: Medium
        return ClassifiedTask(
            task_type=TaskType.RAG_QUERY,
            model_tier=ModelTier.MEDIUM,
            confidence=0.4,
            reasoning="No specific pattern matched, defaulting to medium"
        )

    def _adjust_tier_by_complexity(
        self, query: str, base_tier: ModelTier
    ) -> ModelTier:
        """Passt Tier basierend auf Komplexitäts-Indikatoren an."""
        tier_order = [ModelTier.FAST, ModelTier.MEDIUM, ModelTier.COMPLEX]
        current_idx = tier_order.index(base_tier)

        # Upgrade bei Komplexitäts-Indikatoren
        if any(ind in query for ind in self.COMPLEXITY_INDICATORS["high"]):
            current_idx = min(current_idx + 1, 2)

        # Downgrade bei Einfachheits-Indikatoren
        if any(ind in query for ind in self.COMPLEXITY_INDICATORS["low"]):
            current_idx = max(current_idx - 1, 0)

        return tier_order[current_idx]
```

### 1.3 Konfiguration erweitern

**Änderung in:** `src/config/settings.py`

```python
# Neue Einstellungen für Multi-Model Support
class Settings(BaseSettings):
    # ... bestehende Einstellungen ...

    # Multi-Model Konfiguration
    model_fast: str = Field(
        default="llama3.1:8b",
        description="Modell für schnelle/einfache Tasks"
    )
    model_medium: str = Field(
        default="llama3.1:8b",  # Kann auf 13B/32B geändert werden
        description="Modell für mittlere Komplexität"
    )
    model_complex: str = Field(
        default="llama3.1:70b",
        description="Modell für komplexe Analysen"
    )

    # Auto-Routing
    enable_auto_routing: bool = Field(
        default=True,
        description="Automatisches Model-Routing aktivieren"
    )
    routing_log_enabled: bool = Field(
        default=True,
        description="Routing-Entscheidungen loggen"
    )
```

---

## Phase 2: Model Router Integration (Woche 2)

### 2.1 Model Router Service

**Neue Datei:** `src/services/model_router.py`

```python
import time
import logging
from typing import Any
from dataclasses import dataclass
from ollama import AsyncClient

from .model_registry import ModelRegistry, ModelTier, ModelConfig
from .task_classifier import TaskClassifier, ClassifiedTask

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    selected_model: ModelConfig
    original_tier: ModelTier
    was_fallback: bool
    classification: ClassifiedTask
    routing_time_ms: float

@dataclass
class ModelResponse:
    content: str
    model_used: str
    tier_used: ModelTier
    latency_ms: float
    tokens_generated: int
    routing_decision: RoutingDecision

class ModelRouter:
    """Intelligenter Router für Multi-Model LLM Anfragen."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        enable_fallback: bool = True
    ):
        self.ollama_host = ollama_host
        self.enable_fallback = enable_fallback
        self.registry = ModelRegistry(ollama_host)
        self.classifier = TaskClassifier()
        self._client = AsyncClient(host=ollama_host)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialisiert Router und lädt verfügbare Modelle."""
        await self.registry.refresh_available_models()
        self._initialized = True
        logger.info(f"ModelRouter initialized with models: {self.registry._available_models}")

    async def route_and_generate(
        self,
        query: str,
        system_prompt: str | None = None,
        context: dict | None = None,
        force_tier: ModelTier | None = None,
        **kwargs
    ) -> ModelResponse:
        """Routet Query zum passenden Modell und generiert Antwort."""
        if not self._initialized:
            await self.initialize()

        start_time = time.perf_counter()

        # 1. Klassifiziere Task
        if force_tier:
            classification = ClassifiedTask(
                task_type=None,
                model_tier=force_tier,
                confidence=1.0,
                reasoning="Forced tier override"
            )
        else:
            classification = self.classifier.classify(query, context)

        # 2. Wähle Modell
        model_config = self.registry.get_model_for_tier(classification.model_tier)
        was_fallback = False

        if not model_config and self.enable_fallback:
            model_config = self.registry.get_fallback_model(classification.model_tier)
            was_fallback = True

        if not model_config:
            raise ValueError(f"No model available for tier {classification.model_tier}")

        routing_time = (time.perf_counter() - start_time) * 1000

        routing_decision = RoutingDecision(
            selected_model=model_config,
            original_tier=classification.model_tier,
            was_fallback=was_fallback,
            classification=classification,
            routing_time_ms=routing_time
        )

        logger.info(
            f"Routing: {classification.task_type} -> {model_config.name} "
            f"(tier={classification.model_tier.value}, fallback={was_fallback})"
        )

        # 3. Generiere Antwort
        gen_start = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        response = await self._client.chat(
            model=model_config.name,
            messages=messages,
            options={
                "temperature": model_config.temperature,
                "num_ctx": model_config.context_window,
                "num_predict": kwargs.get("max_tokens", model_config.max_tokens),
            }
        )

        gen_time = (time.perf_counter() - gen_start) * 1000

        return ModelResponse(
            content=response["message"]["content"],
            model_used=model_config.name,
            tier_used=classification.model_tier,
            latency_ms=gen_time,
            tokens_generated=response.get("eval_count", 0),
            routing_decision=routing_decision
        )

    # Convenience Methods für spezifische Task-Typen
    async def sentiment(self, text: str) -> ModelResponse:
        """Schnelle Sentiment-Analyse."""
        system = "Analysiere das Sentiment. Antworte nur mit: BULLISH, BEARISH oder NEUTRAL"
        return await self.route_and_generate(
            query=text,
            system_prompt=system,
            force_tier=ModelTier.FAST
        )

    async def classify(self, text: str, categories: list[str]) -> ModelResponse:
        """Schnelle Klassifikation."""
        cats = ", ".join(categories)
        system = f"Klassifiziere in eine dieser Kategorien: {cats}. Antworte nur mit der Kategorie."
        return await self.route_and_generate(
            query=text,
            system_prompt=system,
            force_tier=ModelTier.FAST
        )

    async def trading_analysis(
        self,
        market_data: dict,
        rag_context: list[str]
    ) -> ModelResponse:
        """Komplexe Trading-Analyse (immer 70B)."""
        # Bestehende generate_analysis Logik aus llm_service.py
        return await self.route_and_generate(
            query=self._build_analysis_prompt(market_data, rag_context),
            system_prompt=self._get_trading_system_prompt(),
            force_tier=ModelTier.COMPLEX
        )
```

### 2.2 LLM Service Anpassung

**Änderung in:** `src/services/llm_service.py`

```python
from .model_router import ModelRouter, ModelTier

class LLMService:
    """LLM Service mit Multi-Model Support."""

    def __init__(self):
        # ... bestehende Initialisierung ...
        self.router = ModelRouter(
            ollama_host=settings.ollama_host,
            enable_fallback=True
        )

    async def initialize(self) -> None:
        """Erweiterte Initialisierung mit Router."""
        await self.router.initialize()
        # ... bestehende Logik ...

    async def generate_analysis(
        self,
        market_analysis: "MarketAnalysis",
        rag_context: list[str] | None = None,
        nhits_forecast: dict | None = None,
    ) -> "TradingRecommendation":
        """Trading-Analyse mit automatischem Model-Routing."""

        # Für komplexe Analysen: Immer 70B
        response = await self.router.trading_analysis(
            market_data=market_analysis.model_dump(),
            rag_context=rag_context or []
        )

        # Parse und return wie bisher
        return self._parse_recommendation(response.content)

    async def quick_sentiment(self, text: str) -> dict:
        """Schnelle Sentiment-Abfrage (8B)."""
        response = await self.router.sentiment(text)
        return {
            "sentiment": response.content.strip(),
            "model": response.model_used,
            "latency_ms": response.latency_ms
        }

    async def chat(
        self,
        message: str,
        auto_route: bool = True
    ) -> dict:
        """Chat mit optionalem Auto-Routing."""
        if auto_route:
            response = await self.router.route_and_generate(query=message)
        else:
            # Legacy: Immer 70B
            response = await self.router.route_and_generate(
                query=message,
                force_tier=ModelTier.COMPLEX
            )

        return {
            "response": response.content,
            "model": response.model_used,
            "tier": response.tier_used.value,
            "latency_ms": response.latency_ms,
            "routing": {
                "was_fallback": response.routing_decision.was_fallback,
                "confidence": response.routing_decision.classification.confidence
            }
        }
```

---

## Phase 3: API Erweiterung (Woche 3)

### 3.1 Neue API Endpoints

**Änderung in:** `src/services/llm_app/main.py`

```python
from pydantic import BaseModel
from typing import Optional

class QuickSentimentRequest(BaseModel):
    text: str

class QuickSentimentResponse(BaseModel):
    sentiment: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float
    model_used: str
    latency_ms: float

class ClassifyRequest(BaseModel):
    text: str
    categories: list[str]

class ChatRequest(BaseModel):
    message: str
    auto_route: bool = True
    force_model: Optional[str] = None

class ModelStatsResponse(BaseModel):
    models: dict[str, dict]
    routing_stats: dict

# Neue Endpoints
@app.post("/api/v1/llm/sentiment", response_model=QuickSentimentResponse)
async def quick_sentiment(request: QuickSentimentRequest):
    """Schnelle Sentiment-Analyse (optimiert für Latenz)."""
    result = await llm_service.quick_sentiment(request.text)
    return QuickSentimentResponse(**result)

@app.post("/api/v1/llm/classify")
async def classify(request: ClassifyRequest):
    """Schnelle Klassifikation in vordefinierte Kategorien."""
    response = await llm_service.router.classify(
        request.text,
        request.categories
    )
    return {
        "category": response.content.strip(),
        "model_used": response.model_used,
        "latency_ms": response.latency_ms
    }

@app.post("/api/v1/llm/chat")
async def chat(request: ChatRequest):
    """Chat mit optionalem Auto-Routing."""
    return await llm_service.chat(
        message=request.message,
        auto_route=request.auto_route
    )

@app.get("/api/v1/llm/models")
async def list_models():
    """Liste aller verfügbaren Modelle mit Stats."""
    await llm_service.router.registry.refresh_available_models()
    return {
        "available": list(llm_service.router.registry._available_models),
        "configured": {
            tier.value: llm_service.router.registry.get_model_for_tier(tier)
            for tier in ModelTier
        }
    }

@app.get("/api/v1/llm/routing-stats")
async def routing_stats():
    """Statistiken über Model-Routing Entscheidungen."""
    # Implementierung mit Metriken-Sammlung
    pass
```

---

## Phase 4: Monitoring & Optimierung (Woche 4)

### 4.1 Routing Metriken

**Neue Datei:** `src/services/routing_metrics.py`

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

@dataclass
class RoutingMetrics:
    """Sammelt Metriken über Model-Routing Entscheidungen."""

    # Zähler pro Modell
    requests_per_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Latenz-Tracking
    latencies_per_model: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Fallback-Tracking
    fallback_count: int = 0
    total_requests: int = 0

    # Tier-Verteilung
    requests_per_tier: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, model: str, tier: str, latency_ms: float, was_fallback: bool):
        """Zeichnet eine Routing-Entscheidung auf."""
        self.total_requests += 1
        self.requests_per_model[model] += 1
        self.requests_per_tier[tier] += 1
        self.latencies_per_model[model].append(latency_ms)

        if was_fallback:
            self.fallback_count += 1

        # Halte nur letzte 1000 Latenz-Werte
        if len(self.latencies_per_model[model]) > 1000:
            self.latencies_per_model[model] = self.latencies_per_model[model][-1000:]

    def get_stats(self) -> dict:
        """Gibt aggregierte Statistiken zurück."""
        return {
            "total_requests": self.total_requests,
            "fallback_rate": self.fallback_count / max(self.total_requests, 1),
            "requests_per_model": dict(self.requests_per_model),
            "requests_per_tier": dict(self.requests_per_tier),
            "avg_latency_per_model": {
                model: statistics.mean(latencies) if latencies else 0
                for model, latencies in self.latencies_per_model.items()
            },
            "p95_latency_per_model": {
                model: (
                    sorted(latencies)[int(len(latencies) * 0.95)]
                    if len(latencies) > 20 else None
                )
                for model, latencies in self.latencies_per_model.items()
            }
        }

# Singleton für globale Metriken
routing_metrics = RoutingMetrics()
```

### 4.2 Dashboard Integration

**Frontend-Erweiterung für Model-Monitoring:**

```javascript
// Neues Widget im Dashboard
const ModelRoutingStats = () => {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch('/api/v1/llm/routing-stats')
      .then(r => r.json())
      .then(setStats);
  }, []);

  return (
    <Card title="Model Routing">
      <PieChart data={stats?.requests_per_tier} />
      <Table
        columns={['Model', 'Requests', 'Avg Latency', 'P95']}
        data={Object.entries(stats?.avg_latency_per_model || {})}
      />
      <Metric
        label="Fallback Rate"
        value={`${(stats?.fallback_rate * 100).toFixed(1)}%`}
      />
    </Card>
  );
};
```

---

## Phase 5: Erweiterte Features (Optional)

### 5.1 Adaptive Routing

```python
class AdaptiveRouter(ModelRouter):
    """Router der sich basierend auf Performance anpasst."""

    async def route_and_generate(self, query: str, **kwargs):
        # Prüfe ob 8B Modell überlastet ist (hohe Latenz)
        if self._is_model_overloaded("llama3.1:8b"):
            # Fallback auf alternative oder Queue
            pass

        return await super().route_and_generate(query, **kwargs)

    def _is_model_overloaded(self, model: str) -> bool:
        recent_latencies = routing_metrics.latencies_per_model[model][-10:]
        if not recent_latencies:
            return False
        avg = statistics.mean(recent_latencies)
        expected = self.registry.MODELS[model].avg_latency_ms
        return avg > expected * 2  # Doppelte erwartete Latenz
```

### 5.2 ML-basierter Classifier (Zukunft)

```python
class MLTaskClassifier(TaskClassifier):
    """ML-basierter Classifier trainiert auf historischen Routing-Daten."""

    def __init__(self, model_path: str = "models/task_classifier.onnx"):
        self.model = onnx.load(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def classify(self, query: str, context: dict | None = None) -> ClassifiedTask:
        # Embedding + Classification
        embedding = self._embed(query)
        tier_probs = self._predict(embedding)

        return ClassifiedTask(
            task_type=self._infer_task_type(query),
            model_tier=ModelTier(tier_probs.argmax()),
            confidence=tier_probs.max(),
            reasoning="ML classifier prediction"
        )
```

---

## Migrations-Checkliste

### Vorbereitungen

- [ ] Ollama: llama3.1:8b Modell herunterladen (`ollama pull llama3.1:8b`)
- [ ] Optional: qwen2.5:7b für Multilingual Support
- [ ] Monitoring-Stack erweitern (Prometheus Metriken)

### Code-Änderungen

- [ ] `src/services/model_registry.py` erstellen
- [ ] `src/services/task_classifier.py` erstellen
- [ ] `src/services/model_router.py` erstellen
- [ ] `src/services/routing_metrics.py` erstellen
- [ ] `src/config/settings.py` erweitern
- [ ] `src/services/llm_service.py` anpassen
- [ ] `src/services/llm_app/main.py` erweitern
- [ ] Tests schreiben

### Docker-Änderungen

- [ ] `docker-compose.microservices.yml`: Neue Environment Variables
- [ ] Health-Checks für beide Modelle

### Tests

- [ ] Unit Tests für TaskClassifier
- [ ] Integration Tests für ModelRouter
- [ ] Load Tests mit verschiedenen Query-Typen
- [ ] Latenz-Benchmarks vor/nach

---

## Erwartete Verbesserungen

| Metrik | Vorher (nur 70B) | Nachher (Multi-Model) |
|--------|------------------|----------------------|
| Sentiment-Latenz | ~2500ms | ~200ms |
| Klassifikation-Latenz | ~2500ms | ~200ms |
| RAG-Query-Latenz | ~2500ms | ~500ms |
| Trading-Analyse | ~2500ms | ~2500ms (unverändert) |
| GPU-Auslastung | 100% bei jeder Query | ~30% für einfache Queries |
| Durchsatz (Queries/min) | ~24 | ~100+ (für gemischte Workloads) |

---

## Risiken & Mitigationen

| Risiko | Mitigation |
|--------|------------|
| 8B-Modell liefert schlechte Qualität | Fallback auf 70B, Confidence-Threshold |
| Routing-Overhead | Caching, <5ms Routing-Zeit |
| Komplexe Debugging | Ausführliches Logging, Routing-Traces |
| Inkonsistente Antworten | Temperatur=0.1, deterministische Prompts |

---

## Nächste Schritte

1. **Jetzt:** Review dieses Plans
2. **Phase 1:** Model Registry + TaskClassifier implementieren
3. **Benchmark:** Aktuelle Latenz-Baseline messen
4. **Rollout:** Schrittweise mit Feature-Flag
