"""
Service Registry - Global service instance management for microservices.

This module provides a centralized way to register and access service instances
across different parts of the application. It's used by route handlers to access
services that are initialized in the main application modules.
"""

from typing import Optional, Any

# Global service registry
_services: dict[str, Any] = {}


def register_service(name: str, instance: Any) -> None:
    """
    Register a service instance.

    Args:
        name: Service name (e.g., 'sync_service', 'rag_service')
        instance: The service instance to register
    """
    _services[name] = instance


def get_service(name: str) -> Optional[Any]:
    """
    Get a registered service instance.

    Args:
        name: Service name

    Returns:
        The service instance or None if not registered
    """
    return _services.get(name)


def get_sync_service():
    """Get the sync service instance."""
    return _services.get('sync_service')


def get_rag_service():
    """Get the RAG service instance."""
    return _services.get('rag_service')


def get_llm_service():
    """Get the LLM service instance."""
    return _services.get('llm_service')


def get_nhits_service():
    """Get the NHITS service instance."""
    return _services.get('nhits_service')
