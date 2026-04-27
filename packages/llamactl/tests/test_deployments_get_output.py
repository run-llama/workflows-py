# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``deployments get`` output modes, ``--project`` override, and the
``deployments list`` hidden alias."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import yaml
from click.testing import CliRunner
from conftest import make_deployment, patch_project_client
from llama_agents.cli.app import app
from llama_agents.core.schema.deployments import (
    DeploymentHistoryResponse,
    DeploymentResponse,
    ReleaseHistoryItem,
)


def _make_client_mock(deployments: list[DeploymentResponse]) -> MagicMock:
    """A mock ProjectClient stand-in with the methods the commands hit."""

    async def _list() -> list[DeploymentResponse]:
        return list(deployments)

    async def _get(
        deployment_id: str, include_events: bool = False
    ) -> DeploymentResponse:
        for d in deployments:
            if d.id == deployment_id:
                return d
        raise RuntimeError(f"deployment not found: {deployment_id}")

    client = MagicMock()
    client.list_deployments.side_effect = _list
    client.get_deployment.side_effect = _get
    client.project_id = "proj_default"
    client.base_url = "http://test:8011"
    return client


def test_deployments_get_text_no_args_lists(patched_auth: Any) -> None:
    runner = CliRunner()
    deployments = [make_deployment("app-a"), make_deployment("app-b")]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock):
        result = runner.invoke(app, ["deployments", "get", "--no-interactive"])
    assert result.exit_code == 0, result.output
    assert "app-a" in result.output
    assert "app-b" in result.output
    # Plain-table headers, no Rich markup, no truncation ellipsis.
    assert "NAME" in result.output
    assert "PHASE" in result.output
    assert "\x1b[" not in result.output  # no ANSI escapes
    assert "…" not in result.output


def test_deployments_get_json_array(patched_auth: Any) -> None:
    runner = CliRunner()
    deployments = [make_deployment("app-a"), make_deployment("app-b")]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock):
        result = runner.invoke(
            app, ["deployments", "get", "--no-interactive", "-o", "json"]
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert {d["name"] for d in data} == {"app-a", "app-b"}
    # All deployments returned should expose ``status``.
    assert all("status" in d for d in data)
    assert all("phase" in d["status"] for d in data)
    # Deprecated aliases / leaked flags must not appear.
    for d in data:
        assert "id" not in d
        assert "llama_deploy_version" not in d
        assert "has_personal_access_token" not in d
        assert "secret_names" not in d


def test_deployments_get_yaml_list(patched_auth: Any) -> None:
    runner = CliRunner()
    deployments = [make_deployment("only-one")]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock):
        result = runner.invoke(
            app, ["deployments", "get", "--no-interactive", "-o", "yaml"]
        )
    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load(result.output)
    assert isinstance(parsed, list)
    assert parsed[0]["name"] == "only-one"


def test_deployments_get_single_text_no_tui(patched_auth: Any) -> None:
    """``deployments get <name>`` should never launch the Textual monitor."""
    runner = CliRunner()
    deployments = [make_deployment("my-app")]
    client_mock = _make_client_mock(deployments)
    with (
        patch_project_client(client_mock),
        patch(
            "llama_agents.cli.textual.deployment_monitor.monitor_deployment_screen"
        ) as mock_monitor,
    ):
        # Even if interactive=True, get must print a table — TUI is gone.
        result = runner.invoke(app, ["deployments", "get", "my-app", "--interactive"])
    assert result.exit_code == 0, result.output
    assert mock_monitor.call_count == 0
    assert "my-app" in result.output
    # Single-row uses the same column layout as the list view.
    assert "NAME" in result.output
    assert "PHASE" in result.output


def test_deployments_get_single_json(patched_auth: Any) -> None:
    runner = CliRunner()
    deployments = [make_deployment("my-app")]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock):
        result = runner.invoke(
            app, ["deployments", "get", "my-app", "--no-interactive", "-o", "json"]
        )
    assert result.exit_code == 0, result.output
    obj = json.loads(result.output)
    assert isinstance(obj, dict)
    assert obj["name"] == "my-app"
    assert obj["status"]["phase"] == "Running"
    assert obj["status"]["project_id"] == "proj_default"
    # warning is always-explicit-null
    assert obj["status"]["warning"] is None
    # No deprecated aliases / leaks
    assert "id" not in obj
    assert "llama_deploy_version" not in obj
    assert "has_personal_access_token" not in obj
    assert "secret_names" not in obj
    # Empty secrets / no PAT means the keys are omitted entirely.
    assert "secrets" not in obj
    assert "personal_access_token" not in obj


def test_deployments_get_single_yaml(patched_auth: Any) -> None:
    runner = CliRunner()
    deployments = [make_deployment("my-app")]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock):
        result = runner.invoke(
            app, ["deployments", "get", "my-app", "--no-interactive", "-o", "yaml"]
        )
    assert result.exit_code == 0, result.output
    obj = yaml.safe_load(result.output)
    assert isinstance(obj, dict)
    assert obj["name"] == "my-app"
    assert obj["status"]["phase"] == "Running"


def test_deployments_get_secrets_and_pat_masked(patched_auth: Any) -> None:
    runner = CliRunner()
    deployments = [
        make_deployment(
            "secret-app",
            secret_names=["LLAMA_CLOUD_API_KEY", "OPENAI_API_KEY"],
            has_personal_access_token=True,
        )
    ]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock):
        result = runner.invoke(
            app, ["deployments", "get", "secret-app", "--no-interactive", "-o", "json"]
        )
    assert result.exit_code == 0, result.output
    obj = json.loads(result.output)
    assert obj["secrets"] == {
        "LLAMA_CLOUD_API_KEY": "********",
        "OPENAI_API_KEY": "********",
    }
    assert obj["personal_access_token"] == "********"


def test_deployments_get_empty_json_is_array(patched_auth: Any) -> None:
    runner = CliRunner()
    client_mock = _make_client_mock([])
    with patch_project_client(client_mock):
        result = runner.invoke(
            app, ["deployments", "get", "--no-interactive", "-o", "json"]
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_deployments_list_hidden_alias_works(patched_auth: Any) -> None:
    """The hidden ``deployments list`` alias should still produce JSON output."""
    runner = CliRunner()
    deployments = [make_deployment("app-a")]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock):
        result = runner.invoke(
            app, ["deployments", "list", "--no-interactive", "-o", "json"]
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [d["name"] for d in data] == ["app-a"]


def test_deployments_list_hidden_in_help(patched_auth: Any) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["deployments", "--help"])
    assert result.exit_code == 0
    # `list` should be hidden (not surfaced in `--help`), but `get` should be.
    assert "  list " not in result.output
    assert "  get " in result.output


def test_deployments_get_project_override_threads_to_client(
    patched_auth: Any,
) -> None:
    """``--project foo`` should construct a ProjectClient with project_id='foo'."""
    runner = CliRunner()
    deployments = [make_deployment("app-a", project_id="proj_other")]
    client_mock = _make_client_mock(deployments)
    with patch_project_client(client_mock) as ctor:
        result = runner.invoke(
            app,
            [
                "deployments",
                "get",
                "--no-interactive",
                "--project",
                "proj_other",
                "-o",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    # ProjectClient was called with project_id='proj_other'
    args, kwargs = ctor.call_args
    # Positional: (api_url, project_id, api_key, auth_middleware)
    assert args[1] == "proj_other"


def test_deployments_history_json_output(patched_auth: Any) -> None:
    runner = CliRunner()
    items = [
        ReleaseHistoryItem(
            git_sha="aaaaaaa1111", released_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
        ReleaseHistoryItem(
            git_sha="bbbbbbb2222", released_at=datetime(2026, 2, 1, tzinfo=timezone.utc)
        ),
    ]

    async def _hist(deployment_id: str) -> DeploymentHistoryResponse:
        return DeploymentHistoryResponse(deployment_id=deployment_id, history=items)

    client_mock = _make_client_mock([make_deployment("my-app")])
    client_mock.get_deployment_history.side_effect = _hist
    client_mock.aclose = MagicMock()

    async def _aclose() -> None:
        return None

    client_mock.aclose.side_effect = _aclose
    with patch_project_client(client_mock):
        result = runner.invoke(
            app,
            [
                "deployments",
                "history",
                "my-app",
                "--no-interactive",
                "-o",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    # Newest first
    assert data[0]["git_sha"] == "bbbbbbb2222"
    assert data[1]["git_sha"] == "aaaaaaa1111"


def _make_auth_profile() -> Any:
    """Build a real ``Auth`` dataclass; helpers like SimpleNamespace fail
    equality checks used by the active-profile indicator."""

    from llama_agents.cli.config.schema import Auth

    return Auth(
        id="1",
        name="prof",
        api_url="http://test:8011",
        project_id="proj_default",
        api_key="secret-key",
        api_key_id=None,
        device_oidc=None,
    )


def test_auth_list_json_omits_secrets(patched_auth: Any) -> None:
    runner = CliRunner()
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        profile = _make_auth_profile()
        mock_auth_svc.list_profiles.return_value = [profile]
        mock_auth_svc.get_current_profile.return_value = profile
        mock_auth_svc.env = SimpleNamespace(requires_auth=True)
        mock_service.current_auth_service.return_value = mock_auth_svc
        result = runner.invoke(app, ["auth", "list", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["name"] == "prof"
    # API-key profile reports auth_type=token.
    assert data[0]["auth_type"] == "token"
    # Secrets must not leak through.
    assert "api_key" not in data[0]


def test_auth_list_json_no_credential_is_none() -> None:
    """A profile with no api_key and no device_oidc reports auth_type=none."""
    runner = CliRunner()
    from llama_agents.cli.config.schema import Auth

    profile = Auth(
        id="2",
        name="noauth",
        api_url="http://localhost:8011",
        project_id="proj_default",
        api_key=None,
        api_key_id=None,
        device_oidc=None,
    )
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.list_profiles.return_value = [profile]
        mock_auth_svc.get_current_profile.return_value = profile
        mock_auth_svc.env = SimpleNamespace(requires_auth=False)
        mock_service.current_auth_service.return_value = mock_auth_svc
        result = runner.invoke(app, ["auth", "list", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["auth_type"] == "none"


def test_auth_env_list_json(patched_auth: Any) -> None:
    runner = CliRunner()
    from llama_agents.cli.config.schema import Environment

    env_a = Environment(api_url="https://api.a", requires_auth=True)
    env_b = Environment(api_url="https://api.b", requires_auth=False)
    with patch("llama_agents.cli.commands.env._env_service") as mock_env_service:
        svc = MagicMock()
        svc.list_environments.return_value = [env_a, env_b]
        svc.get_current_environment.return_value = env_a
        mock_env_service.return_value = svc
        result = runner.invoke(app, ["auth", "env", "list", "-o", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert {e["api_url"] for e in data} == {"https://api.a", "https://api.b"}
    assert next(e for e in data if e["api_url"] == "https://api.a")["active"] is True
    # ``min_llamactl_version`` is not part of the public env list contract.
    for entry in data:
        assert "min_llamactl_version" not in entry
