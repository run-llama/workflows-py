# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import os
import re
from pathlib import Path

import click
from click.shell_completion import get_completion_class
from llama_agents.cli.app import app
from llama_agents.cli.options import global_options
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
    cls = get_completion_class(shell)
    if cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    comp = cls(root_cmd, {}, "llamactl", "_LLAMACTL_COMPLETE")
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
    cls = get_completion_class(shell)
    if cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    comp = cls(root_cmd, {}, "llamactl", "_LLAMACTL_COMPLETE")
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
    _ensure_zsh_fpath(zshrc, dry_run)


def _install_fish(source: str, dry_run: bool) -> None:
    comp_dir = Path.home() / ".config" / "fish" / "completions"
    target = comp_dir / "llamactl.fish"

    if dry_run:
        rprint(f"Would write completion script to [cyan]{target}[/cyan]")
        return

    comp_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    rprint(f"Wrote completions to [cyan]{target}[/cyan]")


def _ensure_zsh_fpath(zshrc: Path, dry_run: bool) -> None:
    """Ensure ~/.zfunc is in fpath *before* the first compinit call in .zshrc.

    If compinit already exists somewhere in the file we insert the fpath line
    right before it (so completions are discoverable).  If there is no
    compinit we append both lines at the end.
    """
    fpath_line = "fpath=(~/.zfunc $fpath)"
    compinit_line = "autoload -Uz compinit && compinit"

    if zshrc.exists():
        content = zshrc.read_text()
    else:
        content = ""

    has_fpath = "~/.zfunc" in content and "fpath" in content
    has_compinit = bool(re.search(r"^\s*compinit\b", content, re.MULTILINE)) or bool(
        re.search(r"autoload.*compinit", content)
    )

    if has_fpath and has_compinit:
        return

    if dry_run:
        if not has_fpath:
            rprint(f"Would add to [cyan]{zshrc}[/cyan]: {fpath_line}")
        if not has_compinit:
            rprint(f"Would add to [cyan]{zshrc}[/cyan]: {compinit_line}")
        return

    if not has_fpath:
        if has_compinit:
            # Insert fpath line right before the first compinit occurrence
            lines = content.splitlines(keepends=True)
            out: list[str] = []
            inserted = False
            for ln in lines:
                if (
                    not inserted
                    and ("compinit" in ln)
                    and not ln.lstrip().startswith("#")
                ):
                    out.append(f"{fpath_line}  {_MARKER}\n")
                    inserted = True
                out.append(ln)
            zshrc.write_text("".join(out))
            rprint(f"Added to [cyan]{zshrc}[/cyan]: {fpath_line} (before compinit)")
        else:
            # No compinit at all — append both
            with zshrc.open("a") as f:
                f.write(f"\n{fpath_line}  {_MARKER}\n")
                f.write(f"{compinit_line}  {_MARKER}\n")
            rprint(f"Added to [cyan]{zshrc}[/cyan]: {fpath_line}")
            rprint(f"Added to [cyan]{zshrc}[/cyan]: {compinit_line}")
    elif not has_compinit:
        # Has fpath but no compinit
        with zshrc.open("a") as f:
            f.write(f"\n{compinit_line}  {_MARKER}\n")
        rprint(f"Added to [cyan]{zshrc}[/cyan]: {compinit_line}")


def _ensure_source_line(rc_file: Path, line: str, dry_run: bool) -> None:
    """Append line to rc_file if not already present."""
    if rc_file.exists():
        content = rc_file.read_text()
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
