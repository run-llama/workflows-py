"""Integration tests for SSL context usage in HTTP clients."""

from unittest.mock import MagicMock, patch

import pytest
from llama_agents.cli.auth.client import ClientContextManager
from llama_agents.core.client.manage_client import BaseClient


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean SSL-related environment variables before each test."""
    monkeypatch.delenv("LLAMA_DEPLOY_USE_TRUSTSTORE", raising=False)


@pytest.mark.asyncio
async def test_manage_client_uses_ssl_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that BaseClient passes verify parameter to httpx clients."""
    monkeypatch.setenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "1")

    mock_client = MagicMock()
    mock_httpx_async_client = MagicMock(return_value=mock_client)

    with patch("httpx.AsyncClient", mock_httpx_async_client):
        BaseClient(base_url="http://localhost:8000")

        # Verify httpx.AsyncClient was called twice (client and hookless_client)
        assert mock_httpx_async_client.call_count == 2

        # Check that both calls included the verify parameter
        for call in mock_httpx_async_client.call_args_list:
            kwargs = call[1]
            assert "verify" in kwargs
            # Should be an SSLContext when truststore is enabled
            verify_param = kwargs["verify"]
            # Just verify it's not True (it should be an SSLContext)
            assert verify_param is not True


@pytest.mark.asyncio
async def test_manage_client_uses_default_verify() -> None:
    """Test that BaseClient uses default True verify when truststore disabled."""
    mock_client = MagicMock()
    mock_httpx_async_client = MagicMock(return_value=mock_client)

    with patch("httpx.AsyncClient", mock_httpx_async_client):
        BaseClient(base_url="http://localhost:8000")

        # Verify httpx.AsyncClient was called
        assert mock_httpx_async_client.call_count == 2

        # Check that verify=True was passed
        for call in mock_httpx_async_client.call_args_list:
            kwargs = call[1]
            assert kwargs.get("verify") is True


@pytest.mark.asyncio
async def test_auth_client_uses_ssl_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ClientContextManager passes verify parameter to httpx client."""
    monkeypatch.setenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "1")

    mock_client = MagicMock()
    mock_httpx_async_client = MagicMock(return_value=mock_client)

    with patch("httpx.AsyncClient", mock_httpx_async_client):
        async with ClientContextManager(base_url="http://localhost:8000"):
            # Verify httpx.AsyncClient was called
            assert mock_httpx_async_client.call_count == 1

            # Check that verify parameter was passed
            call_kwargs = mock_httpx_async_client.call_args[1]
            assert "verify" in call_kwargs
            # Should be an SSLContext when truststore is enabled
            verify_param = call_kwargs["verify"]
            assert verify_param is not True


@pytest.mark.asyncio
async def test_auth_client_uses_default_verify() -> None:
    """Test that ClientContextManager uses default True verify when disabled."""
    mock_client = MagicMock()
    mock_httpx_async_client = MagicMock(return_value=mock_client)

    with patch("httpx.AsyncClient", mock_httpx_async_client):
        async with ClientContextManager(base_url="http://localhost:8000"):
            # Check that verify=True was passed
            call_kwargs = mock_httpx_async_client.call_args[1]
            assert call_kwargs.get("verify") is True
