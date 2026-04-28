# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for ``llamactl deployments template``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml as pyyaml
from click.testing import CliRunner
from llama_agents.cli.app import app


def _patch_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_repo: bool,
    branch: str = "main",
    remotes: list[str] | None = None,
    git_root: Path | None = None,
) -> None:
    monkeypatch.setattr(
        "llama_agents.cli.local_context.is_git_repo", lambda: is_repo
    )
    monkeypatch.setattr(
        "llama_agents.cli.local_context.list_remotes",
        lambda: remotes if remotes is not None else [],
    )
    monkeypatch.setattr(
        "llama_agents.cli.local_context.get_current_branch", lambda: branch
    )
    if git_root is not None:
        monkeypatch.setattr(
            "llama_agents.cli.local_context.get_git_root", lambda: git_root
        )


def test_template_in_git_repo_emits_expected_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "llama_deploy.yaml").write_text(
        """
name: my-app
workflows:
  svc: "module.workflow:flow"
required_env_vars: ["API_KEY", "DB_URL"]
""".strip()
    )
    (tmp_path / ".env").write_text("API_KEY=k\n")

    _patch_git(
        monkeypatch,
        is_repo=True,
        branch="develop",
        remotes=["git@github.com:user/repo.git"],
        git_root=tmp_path,
    )

    runner = CliRunner()
    result = runner.invoke(app, ["deployments", "template"])
    assert result.exit_code == 0, result.output

    out = result.output
    # Push-mode signal: empty repo_url, double-quoted.
    assert 'repo_url: ""' in out
    # Detected branch.
    assert "git_ref: develop" in out
    # Required secrets rendered as ${VAR}.
    assert "API_KEY: ${API_KEY}" in out
    assert "DB_URL: ${DB_URL}" in out
    # Per-secret comment for the one with a matching .env entry only.
    api_idx = out.index("API_KEY:")
    assert "#! from your .env" in out[:api_idx]
    db_idx = out.index("DB_URL:")
    # The "from your .env" comment should not precede DB_URL (no .env match).
    assert "#! from your .env" not in out[api_idx + len("API_KEY:") : db_idx]
    # Output parses as YAML when comments are stripped.
    parsed = pyyaml.safe_load(out)
    assert parsed["spec"]["repo_url"] == ""
    assert parsed["spec"]["git_ref"] == "develop"


def test_template_outside_git_repo_emits_note_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_git(monkeypatch, is_repo=False)

    runner = CliRunner()
    result = runner.invoke(app, ["deployments", "template"])
    assert result.exit_code == 0, result.output

    out = result.output
    head_lines = [line for line in out.splitlines() if line.startswith("#! ")]
    assert any("NOTE: not in a git repo" in line for line in head_lines), out
    # Outside git repo: repo_url and git_ref are not set in the rendered spec.
    parsed = pyyaml.safe_load(out)
    assert "repo_url" not in parsed["spec"]
    assert "git_ref" not in parsed["spec"]


def test_template_does_not_require_auth_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command must work without an auth profile configured.

    The conftest's per-worker LLAMACTL_CONFIG_DIR is empty by default — no
    ``patched_auth`` fixture is applied here, so a successful invocation is
    proof the command did not call ``validate_authenticated_profile``.
    """
    monkeypatch.chdir(tmp_path)
    _patch_git(monkeypatch, is_repo=False)

    runner = CliRunner()
    result = runner.invoke(app, ["deployments", "template"])
    assert result.exit_code == 0, result.output


def test_template_has_no_display_name_or_name_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["deployments", "template", "--help"])
    assert result.exit_code == 0
    assert "--display-name" not in result.output
    assert "--name" not in result.output
