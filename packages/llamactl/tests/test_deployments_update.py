# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``deployments update``.

The headline regression here: ``deployments update <id>`` for an internal-code
deployment used to call ``asyncio.run`` twice with the same ``ProjectClient``.
The shared httpx pool was bound to the first (closed) event loop, and the
second run raised ``RuntimeError: Event loop is closed`` — surfacing in CI as
``Error: Event loop is closed`` after the "Continuing with update using last
pushed code" fallback when the internal git mirror returned a transient 500.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from conftest import make_deployment, patch_project_client
from llama_agents.cli.app import app
from llama_agents.core.schema.deployments import (
    INTERNAL_CODE_REPO_SCHEME,
    DeploymentResponse,
)


def _client_mock(
    current: DeploymentResponse, updated: DeploymentResponse
) -> MagicMock:
    """ProjectClient stand-in for the update flow."""

    async def _get(
        deployment_id: str, include_events: bool = False
    ) -> DeploymentResponse:
        return current

    async def _update(deployment_id: str, update: Any) -> DeploymentResponse:
        return updated

    client = MagicMock()
    client.get_deployment.side_effect = _get
    client.update_deployment.side_effect = _update
    client.project_id = "proj_default"
    client.base_url = "http://test:8011"
    return client


def _completed_process(
    args: list[str], returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=args, returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_deployments_update_external_repo(patched_auth: Any) -> None:
    """External-repo update: no push step, single asyncio.run, exit 0."""
    runner = CliRunner()
    current = make_deployment("my-app", git_sha="a" * 40)
    updated = make_deployment("my-app", git_sha="b" * 40)
    client = _client_mock(current, updated)
    with patch_project_client(client):
        result = runner.invoke(
            app, ["deployments", "update", "my-app", "--no-interactive"]
        )
    assert result.exit_code == 0, result.output
    assert "Updated:" in result.output
    client.get_deployment.assert_called_once()
    client.update_deployment.assert_called_once()


def test_deployments_update_internal_repo_push_failure_does_not_abort(
    patched_auth: Any,
) -> None:
    """Push 500 → fallback warning, deployment still updates, exit 0.

    Reproduces the GitHub Actions failure mode where a transient 500 from the
    internal git mirror caused ``Event loop is closed`` and a non-zero exit.
    The fix routes both ``get_deployment`` and ``update_deployment`` through a
    single ``asyncio.run`` so the httpx pool isn't reused across loops.
    """
    runner = CliRunner()
    current = make_deployment(
        "my-app", repo_url=INTERNAL_CODE_REPO_SCHEME, git_sha="a" * 40
    )
    updated = make_deployment(
        "my-app", repo_url=INTERNAL_CODE_REPO_SCHEME, git_sha="b" * 40
    )
    client = _client_mock(current, updated)

    def fake_subprocess_run(
        args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        # ``git rev-parse --git-dir`` succeeds (we're in a repo).
        if args[:2] == ["git", "rev-parse"]:
            return _completed_process(args, returncode=0, stdout=b".git\n")
        # Other git plumbing (config, remote get-url/set-url/add) succeeds.
        if args[:2] == ["git", "config"] or args[:2] == ["git", "remote"]:
            return _completed_process(args, returncode=0)
        # The actual push fails with a 500 (the originally reported flake).
        if args[:2] == ["git", "push"]:
            return _completed_process(
                args,
                returncode=1,
                stderr=(
                    b"error: RPC failed; HTTP 500\n"
                    b"fatal: the remote end hung up unexpectedly\n"
                ),
            )
        return _completed_process(args, returncode=0)

    with (
        patch_project_client(client),
        patch(
            "llama_agents.cli.commands.deployment.subprocess.run",
            side_effect=fake_subprocess_run,
        ),
        patch(
            "llama_agents.cli.utils.git_push.subprocess.run",
            side_effect=fake_subprocess_run,
        ),
    ):
        result = runner.invoke(
            app, ["deployments", "update", "my-app", "--no-interactive"]
        )

    assert result.exit_code == 0, result.output
    assert "Event loop is closed" not in result.output
    assert "Aborted" not in result.output
    assert "Push failed" in result.output
    assert "Continuing with update using last pushed code" in result.output
    assert "Updated:" in result.output
    client.get_deployment.assert_called_once()
    client.update_deployment.assert_called_once()


def test_deployments_update_internal_repo_push_success(patched_auth: Any) -> None:
    """Internal repo, push succeeds: warning suppressed, exit 0."""
    runner = CliRunner()
    current = make_deployment(
        "my-app", repo_url=INTERNAL_CODE_REPO_SCHEME, git_sha="a" * 40
    )
    updated = make_deployment(
        "my-app", repo_url=INTERNAL_CODE_REPO_SCHEME, git_sha="b" * 40
    )
    client = _client_mock(current, updated)

    def fake_subprocess_run(
        args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        return _completed_process(args, returncode=0, stdout=b".git\n")

    with (
        patch_project_client(client),
        patch(
            "llama_agents.cli.commands.deployment.subprocess.run",
            side_effect=fake_subprocess_run,
        ),
        patch(
            "llama_agents.cli.utils.git_push.subprocess.run",
            side_effect=fake_subprocess_run,
        ),
    ):
        result = runner.invoke(
            app, ["deployments", "update", "my-app", "--no-interactive"]
        )

    assert result.exit_code == 0, result.output
    assert "Push failed" not in result.output
    assert "Updated:" in result.output
    client.update_deployment.assert_called_once()
