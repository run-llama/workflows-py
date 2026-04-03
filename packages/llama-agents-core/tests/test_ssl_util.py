"""Tests for SSL/TLS utility functions."""

import ssl

import httpx
import pytest
from llama_agents.core.client.ssl_util import get_httpx_verify_param, get_ssl_context


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean SSL-related environment variables before each test."""
    monkeypatch.delenv("LLAMA_DEPLOY_USE_TRUSTSTORE", raising=False)


def test_get_ssl_context_default() -> None:
    """Test that get_ssl_context returns True when no env var is set."""
    result = get_ssl_context()
    assert result is True


def test_get_ssl_context_truststore_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_ssl_context returns SSLContext when truststore is enabled."""
    monkeypatch.setenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "1")
    result = get_ssl_context()
    assert isinstance(result, ssl.SSLContext)


def test_get_httpx_verify_param_delegates() -> None:
    """Test that get_httpx_verify_param delegates to get_ssl_context."""
    # Default case - should return True
    result = get_httpx_verify_param()
    assert result is True


def test_get_httpx_verify_param_delegates_truststore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_httpx_verify_param returns SSLContext when truststore enabled."""
    monkeypatch.setenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "1")
    result = get_httpx_verify_param()
    assert isinstance(result, ssl.SSLContext)


def test_ssl_context_type_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that truststore SSLContext has expected protocol."""
    monkeypatch.setenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "1")
    result = get_ssl_context()
    assert isinstance(result, ssl.SSLContext)
    # Verify it's configured with TLS client protocol
    assert result.protocol == ssl.PROTOCOL_TLS_CLIENT


def test_httpx_clients_accept_ssl_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that httpx.AsyncClient accepts the SSL context from get_httpx_verify_param."""
    monkeypatch.setenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "1")
    verify = get_httpx_verify_param()

    # Should not raise an error - just verify client creation succeeds
    client = httpx.AsyncClient(verify=verify)
    assert client is not None


def test_httpx_clients_accept_default_verify() -> None:
    """Test that httpx.AsyncClient accepts the default True verify param."""
    verify = get_httpx_verify_param()
    assert verify is True

    # Should not raise an error - just verify client creation succeeds
    client = httpx.AsyncClient(verify=verify)
    assert client is not None
