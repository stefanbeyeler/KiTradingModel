"""LLM Service for Llama 3.1 70B via Ollama."""

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
    """Service for interacting with Llama 3.1 70B via Ollama."""

    def __init__(self):
        self.model = settings.ollama_model
        self.host = settings.ollama_host
        self._client = None

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
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Du bist ein Trading-Experte. Antworte im JSON-Format."},
                    {"role": "user", "content": str(market_data)}
                ],
                options={"temperature": 0.3}
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