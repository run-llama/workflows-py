# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``deployments logs`` and ``deployments status`` (Slice A Phase 2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from llama_agents.cli.app import app
from llama_agents.core.schema import LogEvent
from llama_agents.core.schema.deployments import DeploymentResponse


def _make_deployment(deployment_id: str = "my-app", **overrides: Any) -> DeploymentResponse:
    base: dict[str, Any] = {
        "id": deployment_id,
        "display_name": deployment_id,
        "repo_url": "https://github.com/example/repo",
        "deployment_file_path": "llama_deploy.yaml",
        "git_ref": "main",
        "git_sha": "abc1234567890",
        "project_id": "proj_default",
        "secret_names": [],
        "apiserver_url": None,
        "status": "Running",
    }
    base.update(overrides)
    return DeploymentResponse.model_validate(base)


@pytest.fixture
def fake_profile() -> SimpleNamespace:
    return SimpleNamespace(
        api_url="http://test:8011",
        project_id="proj_default",
        api_key="key",
        device_oidc=None,
        name="prof",
    )


@pytest.fixture
def patched_auth(fake_profile: SimpleNamespace):
    with patch("llama_agents.cli.config.env_service.service") as mock_service:
        mock_auth_svc = MagicMock()
        mock_auth_svc.get_current_profile.return_value = fake_profile
        mock_auth_svc.list_profiles.return_value = [fake_profile]
        mock_auth_svc.env = SimpleNamespace(requires_auth=True)
        mock_auth_svc.auth_middleware.return_value = None
        mock_service.current_auth_service.return_value = mock_auth_svc
        yield mock_service


def _make_log_events(n: int = 3) -> list[LogEvent]:
    base_ts = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
    return [
        LogEvent(
            pod="pod-x",
            container="app",
            text=(
                f'{{"event": "msg-{i}", "level": "info", '
                f'"timestamp": "2026-04-26T12:00:0{i}.000Z"}}'
            ),
            timestamp=base_ts.replace(second=i),
        )
        for i in range(n)
    ]


def _patch_project_client(client_mock: MagicMock):
    return patch(
        "llama_agents.core.client.manage_client.ProjectClient",
        return_value=client_mock,
    )


def _make_logs_client(events: list[LogEvent]) -> MagicMock:
    """Project-client mock with stream_deployment_logs + aclose."""

    async def _stream(*args: Any, **kwargs: Any) -> AsyncIterator[LogEvent]:
        for ev in events:
            yield ev

    async def _aclose() -> None:
        return None

    client = MagicMock()
    # SAM: side_effect on a non-async returning an async iterator works because
    # the command does `async for ev in client.stream_deployment_logs(...)`.
    # The mock returns the async generator object directly when called.
    client.stream_deployment_logs = MagicMock(side_effect=_stream)
    client.get_deployment = MagicMock()
    client.aclose = MagicMock(side_effect=_aclose)
    client.project_id = "proj_default"
    client.base_url = "http://test:8011"
    return client


def test_logs_default_prints_recent_and_exits(patched_auth: Any) -> None:
    runner = CliRunner()
    events = _make_log_events(3)
    client = _make_logs_client(events)
    with _patch_project_client(client):
        result = runner.invoke(
            app, ["deployments", "logs", "my-app", "--no-interactive"]
        )
    assert result.exit_code == 0, result.output
    # Three log lines, one per event.
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 3
    assert "msg-0" in result.output
    assert "msg-2" in result.output
    # Verify follow=False was passed.
    kwargs = client.stream_deployment_logs.call_args.kwargs
    assert kwargs["follow"] is False


def test_logs_follow_passes_follow_true(patched_auth: Any) -> None:
    runner = CliRunner()
    events = _make_log_events(2)
    client = _make_logs_client(events)
    with _patch_project_client(client):
        result = runner.invoke(
            app,
            ["deployments", "logs", "my-app", "--no-interactive", "--follow"],
        )
    assert result.exit_code == 0, result.output
    kwargs = client.stream_deployment_logs.call_args.kwargs
    assert kwargs["follow"] is True


def test_logs_json_outputs_jsonl(patched_auth: Any) -> None:
    runner = CliRunner()
    events = _make_log_events(2)
    client = _make_logs_client(events)
    with _patch_project_client(client):
        result = runner.invoke(
            app,
            ["deployments", "logs", "my-app", "--no-interactive", "--json"],
        )
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    # Each line is a LogEvent envelope.
    for obj in parsed:
        assert obj["pod"] == "pod-x"
        assert obj["container"] == "app"
        assert "text" in obj
        assert "timestamp" in obj


def test_logs_no_events_emits_stderr_note(patched_auth: Any) -> None:
    runner = CliRunner()
    client = _make_logs_client([])
    with _patch_project_client(client):
        # mix_stderr=False so we can inspect stderr separately.
        result = runner.invoke(
            app,
            ["deployments", "logs", "my-app", "--no-interactive"],
        )
    assert result.exit_code == 0, result.output
    # stderr message present in combined output (CliRunner default).
    assert "no logs available yet" in result.output


def test_status_text_one_liner(patched_auth: Any) -> None:
    runner = CliRunner()

    async def _get(deployment_id: str, include_events: bool = False) -> DeploymentResponse:
        return _make_deployment(deployment_id, git_sha="deadbeef1234")

    client = MagicMock()
    client.get_deployment = MagicMock(side_effect=_get)
    client.project_id = "proj_default"
    client.base_url = "http://test:8011"
    with _patch_project_client(client):
        result = runner.invoke(
            app, ["deployments", "status", "my-app", "--no-interactive"]
        )
    assert result.exit_code == 0, result.output
    out = result.output.strip()
    assert "my-app" in out
    assert "Running" in out
    assert "deadbee" in out  # short sha (7 chars)


def test_status_json_full_payload(patched_auth: Any) -> None:
    runner = CliRunner()

    async def _get(deployment_id: str, include_events: bool = False) -> DeploymentResponse:
        return _make_deployment(deployment_id)

    client = MagicMock()
    client.get_deployment = MagicMock(side_effect=_get)
    client.project_id = "proj_default"
    client.base_url = "http://test:8011"
    with _patch_project_client(client):
        result = runner.invoke(
            app,
            ["deployments", "status", "my-app", "--no-interactive", "-o", "json"],
        )
    assert result.exit_code == 0, result.output
    obj = json.loads(result.output)
    assert obj["id"] == "my-app"
    assert obj["status"] == "Running"
    assert obj["project_id"] == "proj_default"


def test_status_yaml(patched_auth: Any) -> None:
    runner = CliRunner()

    async def _get(deployment_id: str, include_events: bool = False) -> DeploymentResponse:
        return _make_deployment(deployment_id)

    client = MagicMock()
    client.get_deployment = MagicMock(side_effect=_get)
    client.project_id = "proj_default"
    client.base_url = "http://test:8011"
    with _patch_project_client(client):
        result = runner.invoke(
            app,
            ["deployments", "status", "my-app", "--no-interactive", "-o", "yaml"],
        )
    assert result.exit_code == 0, result.output
    obj = yaml.safe_load(result.output)
    assert obj["id"] == "my-app"
