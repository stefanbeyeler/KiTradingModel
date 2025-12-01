"""LLM Service for Llama 3.1 70B via Ollama with GPU optimization."""

import json
import time
from datetime import datetime
from typing import Optional, Tuple
import ollama
from loguru import logger

from ..config import settings
from ..models.trading_data import (
    MarketAnalysis,
    TradingRecommendation,
    TradeDirection,
    SignalType,
    ConfidenceLevel,
)
from ..config import get_output_schema_prompt
from .query_log_service import query_log_service, TimescaleDBDataLog, RAGContextLog


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

    async def generate_analysis(
        self,
        market_data: MarketAnalysis,
        rag_context: list[str],
        custom_prompt: Optional[str] = None,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        # Neue Parameter für detaillierte Protokollierung
        timescaledb_data: Optional[TimescaleDBDataLog] = None,
        rag_context_details: Optional[RAGContextLog] = None,
    ) -> TradingRecommendation:
        start_time = time.time()
        system_prompt = ""
        user_prompt = ""
        llm_response = ""

        try:
            client = self._get_client()

            # Build context from RAG
            context_str = ""
            if rag_context:
                context_str = "\n\nHistorischer Kontext:\n" + "\n---\n".join(rag_context[:5])

            # Build the system prompt with output schema
            output_schema_prompt = get_output_schema_prompt()

            system_prompt = f"""Du bist ein erfahrener Trading-Analyst. Analysiere die Marktdaten und gib eine strukturierte Empfehlung.

{output_schema_prompt}

Antworte IMMER nur mit dem JSON-Objekt, ohne zusätzlichen Text davor oder danach."""

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

            llm_response = response["message"]["content"]
            processing_time = (time.time() - start_time) * 1000

            recommendation = self._parse_recommendation(llm_response, market_data.symbol, market_data.current_price)

            # Log the query with detailed data source information
            query_log_service.add_log(
                query_type="analysis",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_response=llm_response,
                model_used=self.model,
                processing_time_ms=processing_time,
                symbol=market_data.symbol,
                rag_context=rag_context[:5] if rag_context else [],
                parsed_response=recommendation.model_dump() if recommendation else None,
                success=True,
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                # Detaillierte Datenquellenprotokollierung
                timescaledb_data=timescaledb_data,
                rag_context_details=rag_context_details,
            )

            return recommendation

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error: {e}")

            # Log the failed query with detailed data source information
            query_log_service.add_log(
                query_type="analysis",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_response=llm_response,
                model_used=self.model,
                processing_time_ms=processing_time,
                symbol=market_data.symbol if market_data else None,
                rag_context=rag_context[:5] if rag_context else [],
                success=False,
                error_message=str(e),
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                # Detaillierte Datenquellenprotokollierung auch bei Fehlern
                timescaledb_data=timescaledb_data,
                rag_context_details=rag_context_details,
            )

            raise

    def _parse_recommendation(self, llm_response: str, symbol: str, current_price: float) -> TradingRecommendation:
        try:
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start == -1:
                raise ValueError("No JSON found in LLM response")
            data = json.loads(llm_response[json_start:json_end])

            # Parse direction (new schema)
            direction_str = data.get("direction", "NEUTRAL").upper()
            try:
                direction = TradeDirection(direction_str)
            except ValueError:
                direction = TradeDirection.NEUTRAL

            # Map direction to legacy signal
            signal_map = {
                TradeDirection.LONG: SignalType.BUY,
                TradeDirection.SHORT: SignalType.SELL,
                TradeDirection.NEUTRAL: SignalType.HOLD
            }
            signal = signal_map.get(direction, SignalType.HOLD)

            # Parse confidence_score (0-100) to confidence level
            confidence_score = int(data.get("confidence_score", 50))
            if confidence_score >= 80:
                confidence = ConfidenceLevel.VERY_HIGH
            elif confidence_score >= 60:
                confidence = ConfidenceLevel.HIGH
            elif confidence_score >= 40:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW

            # Get take profit values
            take_profit_1 = data.get("take_profit_1")
            take_profit_2 = data.get("take_profit_2")
            take_profit_3 = data.get("take_profit_3")

            # Build risk factors string from list if needed
            risk_factors = data.get("risk_factors", "")
            if isinstance(risk_factors, list):
                risk_factors = "; ".join(risk_factors)

            return TradingRecommendation(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                # New schema fields
                direction=direction,
                confidence_score=confidence_score,
                setup_recommendation=data.get("setup_recommendation", ""),
                entry_price=data.get("entry_price"),
                stop_loss=data.get("stop_loss"),
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                risk_reward_ratio=data.get("risk_reward_ratio"),
                recommended_position_size=data.get("recommended_position_size"),
                max_risk_percent=data.get("max_risk_percent"),
                trend_analysis=data.get("trend_analysis", ""),
                support_resistance=data.get("support_resistance", ""),
                key_levels=data.get("key_levels", ""),
                risk_factors=risk_factors,
                trade_rationale=data.get("trade_rationale", ""),
                # Legacy fields for backward compatibility
                signal=signal,
                confidence=confidence,
                take_profit=take_profit_1,  # Use first TP as legacy take_profit
                reasoning=data.get("setup_recommendation", ""),
                key_factors=[],
                risks=[risk_factors] if risk_factors else [],
                timeframe="short_term"
            )
        except Exception as e:
            logger.error(f"Error parsing LLM recommendation: {e}")
            return TradingRecommendation(
                symbol=symbol,
                direction=TradeDirection.NEUTRAL,
                confidence_score=25,
                setup_recommendation=f"Parsing error: {str(e)}",
                signal=SignalType.HOLD,
                confidence=ConfidenceLevel.LOW,
                reasoning=str(e),
                key_factors=[],
                risks=[],
                timeframe="short_term"
            )