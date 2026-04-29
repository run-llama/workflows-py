# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``llamactl deployments apply`` and ``deployments delete -f``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml as pyyaml
from click.testing import CliRunner
from conftest import make_deployment, patch_project_client
from llama_agents.cli.app import app
from llama_agents.core.schema.deployments import (
    DeploymentApply,
    DeploymentCreate,
    DeploymentResponse,
)


def _client_mock(
    *,
    apply_result: tuple[DeploymentResponse, bool] | None = None,
    create_result: DeploymentResponse | None = None,
    delete_ok: bool = True,
) -> MagicMock:
    async def _apply(
        deployment_id: str, apply_data: DeploymentApply
    ) -> tuple[DeploymentResponse, bool]:
        assert apply_result is not None, "apply not expected"
        return apply_result

    async def _create(deployment_data: DeploymentCreate) -> DeploymentResponse:
        assert create_result is not None, "create not expected"
        return create_result

    async def _delete(deployment_id: str) -> None:
        if not delete_ok:
            raise RuntimeError("delete not expected")

    client = MagicMock()
    client.apply_deployment.side_effect = _apply
    client.create_deployment.side_effect = _create
    client.delete_deployment.side_effect = _delete
    client.project_id = "proj_default"
    client.base_url = "http://test:8011"
    return client


def test_apply_with_name_calls_apply_endpoint_and_prints_updated(
    patched_auth: Any, tmp_path: Path
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("name: my-app\nrepo_url: https://example.com/r.git\n")
    deployment = make_deployment("my-app")
    client = _client_mock(apply_result=(deployment, False))

    with patch_project_client(client):
        result = runner.invoke(app, ["deployments", "apply", "-f", str(yaml_file)])

    assert result.exit_code == 0, result.output
    assert "updated" in result.output
    assert "my-app" in result.output
    client.apply_deployment.assert_called_once()
    args, _ = client.apply_deployment.call_args
    assert args[0] == "my-app"
    assert isinstance(args[1], DeploymentApply)
    assert args[1].repo_url == "https://example.com/r.git"


def test_apply_create_prints_created_when_server_signals_201(
    patched_auth: Any, tmp_path: Path
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("name: my-app\ndisplay_name: My App\n")
    deployment = make_deployment("my-app", display_name="My App")
    client = _client_mock(apply_result=(deployment, True))

    with patch_project_client(client):
        result = runner.invoke(app, ["deployments", "apply", "-f", str(yaml_file)])

    assert result.exit_code == 0, result.output
    assert "created" in result.output
    assert "my-app" in result.output


def test_apply_without_name_falls_back_to_create_with_display_name(
    patched_auth: Any, tmp_path: Path
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("display_name: My App\nrepo_url: https://example.com/r.git\n")
    deployment = make_deployment("auto-id", display_name="My App")
    client = _client_mock(create_result=deployment)

    with patch_project_client(client):
        result = runner.invoke(app, ["deployments", "apply", "-f", str(yaml_file)])

    assert result.exit_code == 0, result.output
    assert "created" in result.output
    client.create_deployment.assert_called_once()
    create_arg = client.create_deployment.call_args[0][0]
    assert isinstance(create_arg, DeploymentCreate)
    assert create_arg.display_name == "My App"
    assert create_arg.repo_url == "https://example.com/r.git"
    client.apply_deployment.assert_not_called()


def test_apply_without_name_or_display_name_errors(
    patched_auth: Any, tmp_path: Path
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("git_ref: main\n")
    client = _client_mock()

    with patch_project_client(client):
        result = runner.invoke(app, ["deployments", "apply", "-f", str(yaml_file)])

    assert result.exit_code != 0
    assert "name" in result.output and "display_name" in result.output


def test_apply_unresolved_var_errors_with_grouped_names(
    patched_auth: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text(
        "name: my-app\ngit_ref: ${BRANCH}\nsecrets:\n  A: ${MISS_A}\n  B: ${MISS_B}\n"
    )
    monkeypatch.delenv("BRANCH", raising=False)
    monkeypatch.delenv("MISS_A", raising=False)
    monkeypatch.delenv("MISS_B", raising=False)
    client = _client_mock()

    with patch_project_client(client):
        result = runner.invoke(app, ["deployments", "apply", "-f", str(yaml_file)])

    assert result.exit_code != 0
    assert "BRANCH" in result.output
    assert "MISS_A" in result.output
    assert "MISS_B" in result.output
    client.apply_deployment.assert_not_called()


def test_apply_dry_run_prints_resolved_payload_and_skips_api(
    patched_auth: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text(
        "name: my-app\n"
        "display_name: My App\n"
        "git_ref: ${BRANCH}\n"
        "secrets:\n"
        "  KEY: ${KEY_VAL}\n"
        "  MASKED: '********'\n"
    )
    monkeypatch.setenv("BRANCH", "v2")
    monkeypatch.setenv("KEY_VAL", "sk-x")
    client = _client_mock()

    with patch_project_client(client):
        result = runner.invoke(
            app, ["deployments", "apply", "-f", str(yaml_file), "--dry-run"]
        )

    assert result.exit_code == 0, result.output
    parsed = pyyaml.safe_load(result.output)
    assert parsed["name"] == "my-app"
    assert parsed["git_ref"] == "v2"
    # ``********`` is dropped; resolved key remains.
    assert parsed["secrets"] == {"KEY": "sk-x"}
    client.apply_deployment.assert_not_called()
    client.create_deployment.assert_not_called()


def test_apply_dry_run_server_is_reserved(patched_auth: Any, tmp_path: Path) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("name: my-app\n")
    client = _client_mock()

    with patch_project_client(client):
        result = runner.invoke(
            app,
            ["deployments", "apply", "-f", str(yaml_file), "--dry-run=server"],
        )

    assert result.exit_code != 0
    assert "server-side dry-run not yet supported" in result.output


def test_apply_stdin(patched_auth: Any) -> None:
    runner = CliRunner()
    deployment = make_deployment("my-app")
    client = _client_mock(apply_result=(deployment, False))

    with patch_project_client(client):
        result = runner.invoke(
            app,
            ["deployments", "apply", "-f", "-"],
            input="name: my-app\ngit_ref: main\n",
        )

    assert result.exit_code == 0, result.output
    assert "updated" in result.output
    client.apply_deployment.assert_called_once()


def test_delete_with_filename_reads_name_from_yaml(
    patched_auth: Any, tmp_path: Path
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("name: my-app\ngit_ref: main\n")
    client = _client_mock()

    with patch_project_client(client):
        result = runner.invoke(
            app,
            ["deployments", "delete", "-f", str(yaml_file), "--no-interactive"],
        )

    assert result.exit_code == 0, result.output
    assert "Deleted deployment: my-app" in result.output
    client.delete_deployment.assert_called_once_with("my-app")


def test_delete_filename_and_argument_are_mutually_exclusive(
    patched_auth: Any, tmp_path: Path
) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("name: my-app\n")

    result = runner.invoke(
        app,
        [
            "deployments",
            "delete",
            "other-app",
            "-f",
            str(yaml_file),
            "--no-interactive",
        ],
    )

    assert result.exit_code != 0
    assert "either" in result.output.lower() or "not both" in result.output.lower()


def test_delete_filename_without_name_errors(patched_auth: Any, tmp_path: Path) -> None:
    runner = CliRunner()
    yaml_file = tmp_path / "d.yaml"
    yaml_file.write_text("display_name: My App\n")

    result = runner.invoke(
        app,
        ["deployments", "delete", "-f", str(yaml_file), "--no-interactive"],
    )

    assert result.exit_code != 0
    assert "name" in result.output
