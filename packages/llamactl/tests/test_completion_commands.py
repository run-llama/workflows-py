# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

from click.testing import CliRunner
from llama_agents.cli.app import app


def test_completion_generate_zsh() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["completion", "generate", "zsh"])
    assert result.exit_code == 0
    assert "_LLAMACTL_COMPLETE" in result.output or "compdef" in result.output


def test_completion_generate_bash() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["completion", "generate", "bash"])
    assert result.exit_code == 0
    assert "_LLAMACTL_COMPLETE" in result.output or "complete" in result.output


def test_completion_generate_fish() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["completion", "generate", "fish"])
    assert result.exit_code == 0
    assert "llamactl" in result.output


def test_completion_generate_invalid_shell() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["completion", "generate", "powershell"])
    assert result.exit_code != 0


def test_completion_install_dry_run() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["completion", "install", "--shell", "zsh", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Would write" in result.output


def test_completion_install_dry_run_bash() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["completion", "install", "--shell", "bash", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Would write" in result.output


def test_completion_install_dry_run_fish() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["completion", "install", "--shell", "fish", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Would write" in result.output


def test_completion_group_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["completion", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "install" in result.output
