"""Conversation Memory Service - Sliding Window Memory for LLM Chat."""

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Conversation:
    """A conversation session with sliding window memory."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    max_messages: int = 10  # Sliding window size

    def add_message(self, role: str, content: str) -> None:
        """Add a message and maintain sliding window."""
        self.messages.append(Message(role=role, content=content))
        self.last_activity = time.time()

        # Keep only last N messages (sliding window)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self) -> str:
        """Get conversation history as formatted context string."""
        if not self.messages:
            return ""

        context_lines = ["Bisheriger Gesprächsverlauf:"]
        for msg in self.messages:
            role_label = "User" if msg.role == "user" else "Assistent"
            # Truncate long messages in context
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            context_lines.append(f"{role_label}: {content}")

        return "\n".join(context_lines)

    def get_message_count(self) -> int:
        """Get number of messages in conversation."""
        return len(self.messages)


class ConversationMemoryService:
    """Manages conversation sessions with automatic cleanup."""

    def __init__(
        self,
        max_sessions: int = 100,
        session_ttl_seconds: int = 3600,  # 1 hour
        max_messages_per_session: int = 10
    ):
        """Initialize the conversation memory service.

        Args:
            max_sessions: Maximum number of sessions to keep in memory
            session_ttl_seconds: Time-to-live for inactive sessions
            max_messages_per_session: Sliding window size per session
        """
        self._sessions: OrderedDict[str, Conversation] = OrderedDict()
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl_seconds
        self._max_messages = max_messages_per_session
        logger.info(f"ConversationMemory initialized: max_sessions={max_sessions}, "
                   f"ttl={session_ttl_seconds}s, window={max_messages_per_session}")

    def create_session(self) -> str:
        """Create a new conversation session."""
        self._cleanup_expired()

        session_id = str(uuid.uuid4())[:8]  # Short ID for convenience
        self._sessions[session_id] = Conversation(
            session_id=session_id,
            max_messages=self._max_messages
        )

        # Enforce max sessions (LRU eviction)
        while len(self._sessions) > self._max_sessions:
            oldest_id = next(iter(self._sessions))
            del self._sessions[oldest_id]
            logger.debug(f"Evicted oldest session: {oldest_id}")

        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Conversation]:
        """Get a conversation session by ID."""
        session = self._sessions.get(session_id)
        if session:
            # Check if expired
            if time.time() - session.last_activity > self._session_ttl:
                del self._sessions[session_id]
                logger.debug(f"Session expired: {session_id}")
                return None
            # Move to end (LRU)
            self._sessions.move_to_end(session_id)
        return session

    def get_or_create_session(self, session_id: Optional[str] = None) -> Conversation:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        # Create new session
        new_id = self.create_session()
        return self._sessions[new_id]

    def add_exchange(self, session_id: str, user_message: str, assistant_response: str) -> None:
        """Add a user-assistant exchange to the conversation."""
        session = self.get_session(session_id)
        if session:
            session.add_message("user", user_message)
            session.add_message("assistant", assistant_response)
            logger.debug(f"Added exchange to session {session_id}, "
                        f"total messages: {session.get_message_count()}")

    def get_context(self, session_id: str) -> str:
        """Get conversation context for a session."""
        session = self.get_session(session_id)
        if session:
            return session.get_context()
        return ""

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False

    def get_stats(self) -> dict:
        """Get memory statistics."""
        self._cleanup_expired()
        total_messages = sum(s.get_message_count() for s in self._sessions.values())
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self._max_sessions,
            "total_messages": total_messages,
            "max_messages_per_session": self._max_messages,
            "session_ttl_seconds": self._session_ttl
        }

    def _cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_activity > self._session_ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)


# Singleton instance
_conversation_memory: Optional[ConversationMemoryService] = None


def get_conversation_memory() -> ConversationMemoryService:
    """Get or create the singleton ConversationMemoryService instance."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemoryService()
    return _conversation_memory
