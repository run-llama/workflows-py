# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import os
from pathlib import Path

import click
from click.shell_completion import get_completion_class
from llama_agents.cli.app import app
from llama_agents.cli.completion_cache import (
    _env_hash,
    refresh_cache,
)
from llama_agents.cli.options import global_options
from llama_agents.cli.param_types import _fetch_deployments, _fetch_projects
from rich import print as rprint


@app.group(help="Shell completion helpers.", no_args_is_help=True)
@global_options
def completion() -> None:
    pass


@completion.command("generate")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@global_options
def generate(shell: str) -> None:
    """Print shell completion script to stdout.

    Example: llamactl completion generate zsh > ~/.zfunc/_llamactl
    """
    ctx = click.get_current_context()
    root_cmd = ctx.find_root().command
    # Click uses env var format: _{PROG_NAME}_COMPLETE
    shell_map = {"bash": "bash_source", "zsh": "zsh_source", "fish": "fish_source"}
    cls = get_completion_class(shell)
    if cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    comp = cls(root_cmd, {}, "llamactl", shell_map[shell])
    click.echo(comp.source())


@completion.command("install")
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default=None,
    help="Override auto-detected shell.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without modifying anything.",
)
@global_options
def install(shell: str | None, dry_run: bool) -> None:
    """Auto-detect your shell and install completions.

    Example: llamactl completion install
    """
    if shell is None:
        shell = _detect_shell()

    ctx = click.get_current_context()
    root_cmd = ctx.find_root().command
    shell_map = {"bash": "bash_source", "zsh": "zsh_source", "fish": "fish_source"}
    cls = get_completion_class(shell)
    if cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    comp = cls(root_cmd, {}, "llamactl", shell_map[shell])
    source = comp.source()

    if shell == "bash":
        _install_bash(source, dry_run)
    elif shell == "zsh":
        _install_zsh(source, dry_run)
    elif shell == "fish":
        _install_fish(source, dry_run)

    if not dry_run:
        rprint(f"\nDetected shell: [bold]{shell}[/bold]")
        rprint("Restart your shell or source the config to activate completions.")


@completion.command("update")
@global_options
def update() -> None:
    """Refresh the completion cache for API-backed resources.

    Fetches the current deployment and project lists from the API and
    writes them to the local completion cache so Tab shows up-to-date results.

    Example: llamactl completion update
    """
    eh = _env_hash()

    dep_items = refresh_cache("deployments", _fetch_deployments, eh, timeout=10.0)
    rprint(f"Cached {len(dep_items)} deployments")

    proj_items = refresh_cache("projects", _fetch_projects, eh, timeout=10.0)
    rprint(f"Cached {len(proj_items)} projects")


# ---------------------------------------------------------------------------
# Shell-specific install helpers
# ---------------------------------------------------------------------------

_MARKER = "# llamactl shell completion"


def _detect_shell() -> str:
    shell_env = os.environ.get("SHELL", "")
    name = os.path.basename(shell_env)
    if name in ("bash", "zsh", "fish"):
        return name
    return "bash"


def _install_bash(source: str, dry_run: bool) -> None:
    # Preferred: ~/.local/share/bash-completion/completions/llamactl
    comp_dir = Path.home() / ".local" / "share" / "bash-completion" / "completions"
    if not comp_dir.exists():
        comp_dir = Path.home() / ".bash_completion.d"

    target = comp_dir / "llamactl"

    if dry_run:
        rprint(f"Would write completion script to [cyan]{target}[/cyan]")
        return

    comp_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    rprint(f"Wrote completions to [cyan]{target}[/cyan]")

    # Ensure .bashrc sources the completion dir
    bashrc = Path.home() / ".bashrc"
    _ensure_source_line(
        bashrc,
        f"source {target}",
        dry_run,
    )


def _install_zsh(source: str, dry_run: bool) -> None:
    zfunc = Path.home() / ".zfunc"
    target = zfunc / "_llamactl"

    if dry_run:
        rprint(f"Would write completion script to [cyan]{target}[/cyan]")
        rprint(
            "Would ensure [cyan]~/.zfunc[/cyan] is in fpath in [cyan]~/.zshrc[/cyan]"
        )
        return

    zfunc.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    rprint(f"Wrote completions to [cyan]{target}[/cyan]")

    zshrc = Path.home() / ".zshrc"
    # Ensure fpath includes ~/.zfunc
    _ensure_source_line(
        zshrc,
        "fpath=(~/.zfunc $fpath)",
        dry_run,
    )
    # Ensure compinit is called
    _ensure_source_line(
        zshrc,
        "autoload -Uz compinit && compinit",
        dry_run,
    )


def _install_fish(source: str, dry_run: bool) -> None:
    comp_dir = Path.home() / ".config" / "fish" / "completions"
    target = comp_dir / "llamactl.fish"

    if dry_run:
        rprint(f"Would write completion script to [cyan]{target}[/cyan]")
        return

    comp_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    rprint(f"Wrote completions to [cyan]{target}[/cyan]")


def _ensure_source_line(rc_file: Path, line: str, dry_run: bool) -> None:
    """Append line to rc_file if not already present."""
    if rc_file.exists():
        content = rc_file.read_text()
        # Check if line (or something equivalent) is already there
        if line in content:
            return
    else:
        content = ""

    if dry_run:
        rprint(f"Would add to [cyan]{rc_file}[/cyan]: {line}")
        return

    with rc_file.open("a") as f:
        f.write(f"\n{line}  {_MARKER}\n")
    rprint(f"Added to [cyan]{rc_file}[/cyan]: {line}")
