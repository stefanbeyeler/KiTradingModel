"""LLM Service for Llama 3.1 70B via Ollama with GPU optimization."""

import json
from datetime import datetime
from typing import Optional
import ollama
from loguru import logger

from ..config import settings
from ..models.trading_data import (
    MarketAnalysis,
    TradingRecommendation,
    SignalType,
    ConfidenceLevel,
)


class LLMService:
    """Service for interacting with Llama 3.1 70B via Ollama with optimized settings."""

    def __init__(self):
        self.model = settings.ollama_model
        self.host = settings.ollama_host
        self._client = None

        # Performance options optimized for i9-13900K + RTX 3070
        self._options = {
            "temperature": 0.3,
            "num_ctx": settings.ollama_num_ctx,        # Context window
            "num_gpu": settings.ollama_num_gpu,        # GPU layers (-1 = auto)
            "num_thread": settings.ollama_num_thread,  # CPU threads (16 for i9-13900K)
            "num_batch": 512,                          # Batch size for prompt processing
            "num_keep": 24,                            # Tokens to keep from initial prompt
        }

        logger.info(f"LLM Service initialized - Model: {self.model}, Options: {self._options}")

    def _get_client(self):
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    async def check_model_available(self) -> bool:
        try:
            client = self._get_client()
            models = client.list()
            available_models = []
            models_data = models.get("models", []) if isinstance(models, dict) else models
            for m in models_data:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or str(m)
                else:
                    name = getattr(m, "model", None) or getattr(m, "name", None) or str(m)
                available_models.append(name)
            is_available = any(self.model in m for m in available_models)
            if not is_available:
                logger.warning(f"Model {self.model} not found. Available: {available_models}")
            return is_available
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    async def get_model_info(self) -> dict:
        """Get information about the currently configured model."""
        try:
            client = self._get_client()
            info = client.show(self.model)
            return {
                "model": self.model,
                "parameters": info.get("parameters", {}),
                "template": info.get("template", ""),
                "details": info.get("details", {}),
                "options": self._options
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"model": self.model, "error": str(e)}

    async def pull_model(self) -> bool:
        try:
            client = self._get_client()
            logger.info(f"Pulling model {self.model}...")
            client.pull(self.model)
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False

    async def generate_analysis(self, market_data: MarketAnalysis, rag_context: list[str], custom_prompt: Optional[str] = None) -> TradingRecommendation:
        try:
            client = self._get_client()

            # Build context from RAG
            context_str = ""
            if rag_context:
                context_str = "\n\nHistorischer Kontext:\n" + "\n---\n".join(rag_context[:5])

            # Build the prompt
            system_prompt = """Du bist ein erfahrener Trading-Analyst. Analysiere die Marktdaten und gib eine strukturierte Empfehlung.
Antworte IMMER im folgenden JSON-Format:
{
    "signal": "buy" | "sell" | "hold",
    "confidence": "high" | "medium" | "low",
    "entry_price": <number or null>,
    "stop_loss": <number or null>,
    "take_profit": <number or null>,
    "reasoning": "<kurze Begründung>",
    "key_factors": ["<factor1>", "<factor2>"],
    "risks": ["<risk1>", "<risk2>"],
    "timeframe": "short_term" | "medium_term" | "long_term"
}"""

            user_prompt = f"{str(market_data)}{context_str}"
            if custom_prompt:
                user_prompt += f"\n\nZusätzliche Anweisungen: {custom_prompt}"

            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options=self._options
            )
            return self._parse_recommendation(response["message"]["content"], market_data.symbol, market_data.current_price)
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    def _parse_recommendation(self, llm_response: str, symbol: str, current_price: float) -> TradingRecommendation:
        try:
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start == -1:
                raise ValueError("No JSON")
            data = json.loads(llm_response[json_start:json_end])
            return TradingRecommendation(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal=SignalType(data.get("signal", "hold")),
                confidence=ConfidenceLevel(data.get("confidence", "low")),
                entry_price=data.get("entry_price"),
                stop_loss=data.get("stop_loss"),
                take_profit=data.get("take_profit"),
                reasoning=data.get("reasoning", ""),
                key_factors=data.get("key_factors", []),
                risks=data.get("risks", []),
                timeframe=data.get("timeframe", "short_term")
            )
        except Exception as e:
            return TradingRecommendation(
                symbol=symbol,
                signal=SignalType.HOLD,
                confidence=ConfidenceLevel.LOW,
                reasoning=str(e),
                key_factors=[],
                risks=[],
                timeframe="short_term"
            )