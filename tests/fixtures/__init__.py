"""
Test fixtures for KI Trading Model tests.

Provides pre-defined test data for consistent testing.
"""
import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def load_fixture(name: str) -> dict | list:
    """Load test fixture from JSON file."""
    path = FIXTURES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


# Pre-loaded fixtures
SAMPLE_SYMBOLS = load_fixture("symbols")
SAMPLE_OHLCV = load_fixture("ohlcv_data")
SAMPLE_FORECASTS = load_fixture("forecasts")
