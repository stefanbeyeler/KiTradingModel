"""Service for logging and retrieving AI query history."""

import json
import os
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


class QueryLogEntry(BaseModel):
    """Single query log entry."""
    id: str
    timestamp: datetime
    query_type: str  # "analysis", "quick_recommendation", "rag_query"
    symbol: Optional[str] = None

    # Prompt information
    system_prompt: str
    user_prompt: str

    # RAG context used
    rag_context: list[str] = Field(default_factory=list)
    rag_document_count: int = 0

    # LLM response
    llm_response: str
    parsed_response: Optional[dict] = None

    # Metadata
    model_used: str
    processing_time_ms: float
    success: bool = True
    error_message: Optional[str] = None

    # Strategy info
    strategy_id: Optional[str] = None
    strategy_name: Optional[str] = None


class QueryLogService:
    """Service for managing AI query logs."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._logs: deque[QueryLogEntry] = deque(maxlen=1000)  # Keep last 1000 entries
        self._persist_path = "./data/query_logs.json"
        self._save_lock = threading.Lock()
        self._initialized = True

        # Load existing logs
        self._load_logs()
        logger.info(f"QueryLogService initialized with {len(self._logs)} existing entries")

    def _load_logs(self) -> None:
        """Load logs from disk."""
        try:
            if os.path.exists(self._persist_path):
                with open(self._persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry in data:
                        try:
                            # Convert timestamp string back to datetime
                            if isinstance(entry.get("timestamp"), str):
                                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                            self._logs.append(QueryLogEntry(**entry))
                        except Exception as e:
                            logger.warning(f"Failed to load log entry: {e}")
                logger.info(f"Loaded {len(self._logs)} query logs from disk")
        except Exception as e:
            logger.warning(f"Failed to load query logs: {e}")

    def _save_logs(self) -> None:
        """Persist logs to disk."""
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with self._save_lock:
                with open(self._persist_path, "w", encoding="utf-8") as f:
                    # Convert to list of dicts with serializable timestamps
                    logs_data = []
                    for log in self._logs:
                        log_dict = log.model_dump()
                        log_dict["timestamp"] = log.timestamp.isoformat()
                        logs_data.append(log_dict)
                    json.dump(logs_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query logs: {e}")

    def add_log(
        self,
        query_type: str,
        system_prompt: str,
        user_prompt: str,
        llm_response: str,
        model_used: str,
        processing_time_ms: float,
        symbol: Optional[str] = None,
        rag_context: Optional[list[str]] = None,
        parsed_response: Optional[dict] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> QueryLogEntry:
        """Add a new query log entry."""

        entry = QueryLogEntry(
            id=f"log_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            query_type=query_type,
            symbol=symbol,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            rag_context=rag_context or [],
            rag_document_count=len(rag_context) if rag_context else 0,
            llm_response=llm_response,
            parsed_response=parsed_response,
            model_used=model_used,
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error_message,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
        )

        self._logs.append(entry)
        self._save_logs()

        logger.info(f"Added query log: {entry.id} ({query_type}, {symbol})")
        return entry

    def get_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        query_type: Optional[str] = None,
        symbol: Optional[str] = None,
        success_only: bool = False,
    ) -> list[QueryLogEntry]:
        """Get query logs with optional filtering."""

        # Convert deque to list and reverse (newest first)
        logs = list(reversed(self._logs))

        # Apply filters
        if query_type:
            logs = [l for l in logs if l.query_type == query_type]
        if symbol:
            logs = [l for l in logs if l.symbol and symbol.lower() in l.symbol.lower()]
        if success_only:
            logs = [l for l in logs if l.success]

        # Apply pagination
        return logs[offset:offset + limit]

    def get_log_by_id(self, log_id: str) -> Optional[QueryLogEntry]:
        """Get a specific log entry by ID."""
        for log in self._logs:
            if log.id == log_id:
                return log
        return None

    def get_stats(self) -> dict:
        """Get query log statistics."""
        logs = list(self._logs)

        if not logs:
            return {
                "total_queries": 0,
                "success_rate": 0,
                "avg_processing_time_ms": 0,
                "queries_by_type": {},
                "queries_by_symbol": {},
            }

        success_count = sum(1 for l in logs if l.success)
        total_time = sum(l.processing_time_ms for l in logs)

        # Count by type
        by_type: dict[str, int] = {}
        for log in logs:
            by_type[log.query_type] = by_type.get(log.query_type, 0) + 1

        # Count by symbol (top 10)
        by_symbol: dict[str, int] = {}
        for log in logs:
            if log.symbol:
                by_symbol[log.symbol] = by_symbol.get(log.symbol, 0) + 1
        by_symbol = dict(sorted(by_symbol.items(), key=lambda x: x[1], reverse=True)[:10])

        return {
            "total_queries": len(logs),
            "success_rate": round(success_count / len(logs) * 100, 1),
            "avg_processing_time_ms": round(total_time / len(logs), 1),
            "queries_by_type": by_type,
            "queries_by_symbol": by_symbol,
        }

    def clear_logs(self) -> int:
        """Clear all logs. Returns number of deleted entries."""
        count = len(self._logs)
        self._logs.clear()
        self._save_logs()
        logger.info(f"Cleared {count} query logs")
        return count


# Global singleton instance
query_log_service = QueryLogService()
