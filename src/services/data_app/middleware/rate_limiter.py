"""
Rate Limiter Middleware for Data Service.

Provides sliding window rate limiting with Redis backend for distributed systems.
Falls back to in-memory storage when Redis is unavailable.

Rate Limits:
- Per-IP: 100 requests/minute (configurable)
- Training endpoints: 10 requests/minute
- Health/docs endpoints: No limit (whitelisted)
"""

import time
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False


class RateLimiter:
    """
    Sliding window rate limiter with Redis backend.

    Supports both distributed (Redis) and local (in-memory) rate limiting.
    """

    # Endpoints that should not be rate limited
    WHITELIST_PATHS = {
        "/health",
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/data/health",
        "/data/docs",
        "/data/redoc",
    }

    # IP prefixes that bypass rate limiting (internal Docker network)
    WHITELIST_IP_PREFIXES = (
        "172.",      # Docker bridge networks
        "10.",       # Internal networks
        "192.168.",  # Local networks
        "127.",      # Localhost
    )

    # Paths with stricter limits
    STRICT_PATHS = {"/train", "/training", "/sync"}

    def __init__(
        self,
        redis_url: str = "redis://trading-redis:6379",
        per_ip_limit: int = 100,
        strict_limit: int = 10,
        window_seconds: int = 60,
    ):
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
            per_ip_limit: Requests per minute per IP for normal endpoints
            strict_limit: Requests per minute for strict endpoints (training, sync)
            window_seconds: Time window for rate limiting (default 60s)
        """
        self.redis_url = redis_url
        self.per_ip_limit = per_ip_limit
        self.strict_limit = strict_limit
        self.window_seconds = window_seconds
        self._redis: Optional[aioredis.Redis] = None
        self._fallback_counters: dict = {}
        self._connected = False

    async def connect(self) -> bool:
        """
        Connect to Redis.

        Returns:
            True if connected, False if falling back to in-memory
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available - using in-memory rate limiting")
            return False

        try:
            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            self._connected = True
            logger.info(f"Rate limiter connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) - using in-memory rate limiting")
            self._redis = None
            self._connected = False
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _get_limit_for_path(self, path: str) -> int:
        """Get the rate limit for a given path."""
        for strict_path in self.STRICT_PATHS:
            if strict_path in path:
                return self.strict_limit
        return self.per_ip_limit

    def _is_whitelisted(self, path: str) -> bool:
        """Check if path is whitelisted (no rate limiting)."""
        return path in self.WHITELIST_PATHS or path.rstrip("/") in self.WHITELIST_PATHS

    def _is_internal_ip(self, client_ip: str) -> bool:
        """Check if IP is from internal/Docker network (bypass rate limiting)."""
        return client_ip.startswith(self.WHITELIST_IP_PREFIXES)

    async def _increment_redis(self, key: str) -> int:
        """Increment counter in Redis."""
        try:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window_seconds + 1)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.warning(f"Redis error in rate limiter: {e}")
            return 1  # Allow on error

    def _increment_memory(self, key: str) -> int:
        """Increment counter in memory (fallback)."""
        now = int(time.time())
        window_start = now - (now % self.window_seconds)
        full_key = f"{key}:{window_start}"

        # Increment counter
        self._fallback_counters[full_key] = self._fallback_counters.get(full_key, 0) + 1
        current = self._fallback_counters[full_key]

        # Cleanup old entries (older than 2 windows)
        cutoff = window_start - (2 * self.window_seconds)
        self._fallback_counters = {
            k: v for k, v in self._fallback_counters.items() if int(k.split(":")[-1]) >= cutoff
        }

        return current

    async def is_allowed(self, client_ip: str, path: str) -> tuple[bool, int, dict]:
        """
        Check if request is allowed under rate limit.

        Args:
            client_ip: Client IP address
            path: Request path

        Returns:
            Tuple of (allowed: bool, remaining: int, headers: dict)
        """
        # Whitelisted paths bypass rate limiting
        if self._is_whitelisted(path):
            return True, -1, {}

        # Internal Docker/network IPs bypass rate limiting
        if self._is_internal_ip(client_ip):
            return True, -1, {}

        # Determine limit and create key
        limit = self._get_limit_for_path(path)
        now = int(time.time())
        window_key = f"ratelimit:{client_ip}:{now // self.window_seconds}"

        # Increment counter
        if self._redis and self._connected:
            current = await self._increment_redis(window_key)
        else:
            current = self._increment_memory(window_key)

        # Check limit
        remaining = max(0, limit - current)
        allowed = current <= limit

        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(self.window_seconds - (now % self.window_seconds)),
        }

        return allowed, remaining, headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting requests."""

    def __init__(self, app, rate_limiter: RateLimiter):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            rate_limiter: RateLimiter instance
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Get client IP (handle proxies)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"

        path = request.url.path

        # Check rate limit
        allowed, remaining, headers = await self.rate_limiter.is_allowed(client_ip, path)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}: {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please slow down.",
                    "retry_after": self.rate_limiter.window_seconds,
                },
                headers=headers,
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to successful responses
        if headers:
            for key, value in headers.items():
                response.headers[key] = value

        return response
