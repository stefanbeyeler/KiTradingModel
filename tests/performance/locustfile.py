"""
Locust load testing configuration for KI Trading Microservices.

Run with:
    locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s

Or with web UI:
    locust -f tests/performance/locustfile.py
    Then open http://localhost:8089
"""
from locust import HttpUser, task, between, tag
import random
import os

# Get service host from environment or use localhost
SERVICE_HOST = os.getenv("TEST_SERVICE_HOST", "localhost")


class DataServiceUser(HttpUser):
    """Load test for Data Service."""

    host = f"http://{SERVICE_HOST}:3001"
    wait_time = between(1, 3)

    symbols = ["BTCUSD", "ETHUSD", "EURUSD", "XAUUSD", "GER40"]
    intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]

    @tag("smoke")
    @task(10)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @tag("api")
    @task(5)
    def get_symbols(self):
        """Get list of symbols."""
        self.client.get("/api/v1/symbols")

    @tag("api")
    @task(8)
    def get_ohlcv(self):
        """Get OHLCV data for a random symbol."""
        symbol = random.choice(self.symbols)
        interval = random.choice(self.intervals)
        self.client.get(
            f"/api/v1/ohlcv/{symbol}",
            params={"interval": interval, "limit": 100}
        )

    @tag("api")
    @task(3)
    def get_technical_indicator(self):
        """Get technical indicator for a symbol."""
        symbol = random.choice(self.symbols)
        indicator = random.choice(["rsi", "macd", "sma"])
        self.client.get(f"/api/v1/twelvedata/{indicator}/{symbol}")

    @tag("api")
    @task(2)
    def get_managed_symbols(self):
        """Get managed symbols."""
        self.client.get("/api/v1/managed-symbols")


class NHITSServiceUser(HttpUser):
    """Load test for NHITS Service."""

    host = f"http://{SERVICE_HOST}:3002"
    wait_time = between(2, 5)

    symbols = ["BTCUSD", "ETHUSD", "EURUSD"]

    @tag("smoke")
    @task(5)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @tag("api")
    @task(2)
    def get_models(self):
        """Get list of trained models."""
        self.client.get("/api/v1/models")

    @tag("api")
    @task(1)
    def request_forecast(self):
        """Request a forecast (may be slow)."""
        symbol = random.choice(self.symbols)
        horizon = random.choice([12, 24, 48])
        with self.client.post(
            "/api/v1/forecast",
            json={"symbol": symbol, "horizon": horizon},
            catch_response=True
        ) as response:
            # Accept both success and "model not found" responses
            if response.status_code in [200, 404, 503]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class RAGServiceUser(HttpUser):
    """Load test for RAG Service."""

    host = f"http://{SERVICE_HOST}:3008"
    wait_time = between(2, 5)

    queries = [
        "Bitcoin price analysis",
        "EUR/USD trend",
        "Gold market outlook",
        "Trading signals today",
        "Market sentiment",
        "Technical analysis patterns"
    ]

    @tag("smoke")
    @task(5)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @tag("api")
    @task(3)
    def semantic_search(self):
        """Perform semantic search."""
        query = random.choice(self.queries)
        top_k = random.choice([3, 5, 10])
        with self.client.post(
            "/api/v1/query",
            json={"query": query, "top_k": top_k},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404, 503]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @tag("api")
    @task(2)
    def get_stats(self):
        """Get RAG statistics."""
        self.client.get("/api/v1/stats")


class LLMServiceUser(HttpUser):
    """Load test for LLM Service."""

    host = f"http://{SERVICE_HOST}:3009"
    wait_time = between(5, 10)  # Longer wait due to LLM latency

    symbols = ["BTCUSD", "ETHUSD", "EURUSD"]

    @tag("smoke")
    @task(5)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @tag("api")
    @task(1)
    def request_analysis(self):
        """Request trading analysis (slow operation)."""
        symbol = random.choice(self.symbols)
        with self.client.post(
            "/api/v1/analyze",
            json={"symbol": symbol, "use_rag": False},
            catch_response=True,
            timeout=120
        ) as response:
            if response.status_code in [200, 404, 503]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class TCNServiceUser(HttpUser):
    """Load test for TCN Pattern Service."""

    host = f"http://{SERVICE_HOST}:3003"
    wait_time = between(2, 5)

    symbols = ["BTCUSD", "ETHUSD", "EURUSD", "XAUUSD"]

    @tag("smoke")
    @task(5)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @tag("api")
    @task(2)
    def detect_patterns(self):
        """Detect chart patterns."""
        symbol = random.choice(self.symbols)
        with self.client.post(
            "/api/v1/detect",
            json={"symbol": symbol, "interval": "1h"},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404, 422, 503]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class HMMServiceUser(HttpUser):
    """Load test for HMM Regime Service."""

    host = f"http://{SERVICE_HOST}:3004"
    wait_time = between(2, 5)

    symbols = ["BTCUSD", "ETHUSD", "EURUSD"]

    @tag("smoke")
    @task(5)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @tag("api")
    @task(2)
    def detect_regime(self):
        """Detect market regime."""
        symbol = random.choice(self.symbols)
        with self.client.post(
            "/api/v1/regime",
            json={"symbol": symbol},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404, 422, 503]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class EmbedderServiceUser(HttpUser):
    """Load test for Embedder Service."""

    host = f"http://{SERVICE_HOST}:3005"
    wait_time = between(1, 3)

    texts = [
        "Bitcoin is showing bullish momentum",
        "The market is in a bearish trend",
        "Technical indicators suggest consolidation",
        "Strong support level at current price"
    ]

    @tag("smoke")
    @task(5)
    def health_check(self):
        """Check service health."""
        self.client.get("/health")

    @tag("api")
    @task(3)
    def embed_text(self):
        """Generate text embedding."""
        text = random.choice(self.texts)
        with self.client.post(
            "/api/v1/embed/text",
            json={"text": text},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404, 422, 503]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


# Mixed workload simulating real usage patterns
class MixedWorkloadUser(HttpUser):
    """Mixed workload simulating realistic usage patterns."""

    wait_time = between(1, 5)

    def on_start(self):
        """Set up initial state."""
        self.symbols = ["BTCUSD", "ETHUSD", "EURUSD", "XAUUSD", "GER40"]
        self.current_symbol = random.choice(self.symbols)

    @task(10)
    def check_data_health(self):
        """Check data service health."""
        self.client.get(f"http://{SERVICE_HOST}:3001/health")

    @task(5)
    def get_market_data(self):
        """Get market data - most common operation."""
        self.client.get(
            f"http://{SERVICE_HOST}:3001/api/v1/ohlcv/{self.current_symbol}",
            params={"interval": "1h", "limit": 100}
        )

    @task(3)
    def get_symbols_list(self):
        """Get available symbols."""
        self.client.get(f"http://{SERVICE_HOST}:3001/api/v1/symbols")

    @task(1)
    def request_forecast(self):
        """Request price forecast - less frequent."""
        with self.client.post(
            f"http://{SERVICE_HOST}:3002/api/v1/forecast",
            json={"symbol": self.current_symbol, "horizon": 24},
            catch_response=True
        ) as response:
            if response.status_code in [200, 404, 503]:
                response.success()

    @task(1)
    def switch_symbol(self):
        """Simulate user switching to different symbol."""
        self.current_symbol = random.choice(self.symbols)
