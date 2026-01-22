"""Pytest configuration for integration tests."""

import pytest


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset LlamaIndex settings before each test."""
    # This ensures tests don't interfere with each other
    yield
