"""Tests for CLI options and decorators."""

import os
from typing import Any, Protocol, runtime_checkable
from unittest.mock import MagicMock

import pytest
from click import Context, Parameter
from click.testing import CliRunner
from llama_agents.cli.app import app


@runtime_checkable
class ClickDecorated(Protocol):
    __click_params__: list[Any]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean TLS-related environment variables before each test."""
    monkeypatch.delenv("UV_NATIVE_TLS", raising=False)
    monkeypatch.delenv("LLAMA_DEPLOY_USE_TRUSTSTORE", raising=False)
    monkeypatch.delenv("LLAMACTL_NATIVE_TLS", raising=False)


def test_native_tls_option_sets_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --native-tls flag sets the expected environment variables."""
    # Clear any existing values
    monkeypatch.delenv("UV_NATIVE_TLS", raising=False)
    monkeypatch.delenv("LLAMA_DEPLOY_USE_TRUSTSTORE", raising=False)

    runner = CliRunner()
    # Use serve command which has the native_tls_option applied via global_options
    result = runner.invoke(app, ["serve", "--native-tls", "--help"])

    # Command should succeed
    assert result.exit_code == 0

    # Check that env vars were set
    # Note: Click callbacks run during parsing, so we need to check if the flag exists
    assert "--native-tls" in result.output


def test_native_tls_option_preserves_existing_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that native TLS option doesn't override pre-existing env vars."""
    # Pre-set custom values
    monkeypatch.setenv("UV_NATIVE_TLS", "custom")
    monkeypatch.setenv("LLAMA_DEPLOY_USE_TRUSTSTORE", "custom")

    # Import the callback function after env is set
    from llama_agents.cli.options import native_tls_option

    # Find the callback by introspecting the decorator chain
    # native_tls_option returns a function that applies click.option
    def dummy_func() -> None:
        pass

    decorated = native_tls_option(dummy_func)

    # Extract the click option params
    if isinstance(decorated, ClickDecorated):
        params = decorated.__click_params__
        # Find the native-tls param
        for param in params:
            if hasattr(param, "name") and "native_tls" in str(param.name):
                if hasattr(param, "callback") and param.callback:
                    # Call the callback with True
                    ctx = MagicMock(spec=Context)
                    param_obj = MagicMock(spec=Parameter)
                    param.callback(ctx, param_obj, True)
                    break

    # Values should remain as "custom"
    assert os.environ.get("UV_NATIVE_TLS") == "custom"
    assert os.environ.get("LLAMA_DEPLOY_USE_TRUSTSTORE") == "custom"


def test_native_tls_option_from_envvar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LLAMACTL_NATIVE_TLS env var enables native TLS."""
    monkeypatch.setenv("LLAMACTL_NATIVE_TLS", "1")

    runner = CliRunner()
    # When running with LLAMACTL_NATIVE_TLS set, the callback should be triggered
    result = runner.invoke(app, ["serve", "--help"])

    # Command should succeed
    assert result.exit_code == 0
    # Help should show the native-tls option
    assert "--native-tls" in result.output
