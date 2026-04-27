# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Sync repo-local agent skills into Claude Code's gitignored skills dir.

Skills are checked in under `.agents/skills/<name>/` (the Codex-compatible
location, see https://developers.openai.com/codex/skills). Claude Code reads
from `.claude/skills/<name>/`, which is gitignored. This command symlinks the
two so editing the checked-in copy is what Claude Code sees.
"""

from __future__ import annotations

import os
from pathlib import Path

import click


def _repo_root() -> Path:
    # cli is invoked from anywhere in the repo; resolve relative to this file.
    return Path(__file__).resolve().parents[3]


@click.command("sync-skills")
@click.option(
    "--check",
    is_flag=True,
    help="Exit non-zero if any link is missing or stale, without modifying anything.",
)
def sync_skills(check: bool) -> None:
    """Symlink `.agents/skills/<name>` into `.claude/skills/<name>`.

    Idempotent. Refuses to overwrite an existing non-symlink entry.
    """
    root = _repo_root()
    src_dir = root / ".agents" / "skills"
    dst_dir = root / ".claude" / "skills"

    if not src_dir.is_dir():
        click.echo(
            f"no skills source dir at {src_dir.relative_to(root)}, nothing to do"
        )
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    drift: list[str] = []
    created: list[str] = []
    ok: list[str] = []

    for entry in sorted(src_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        link = dst_dir / name
        # Symlink target is relative to the link's parent dir, so the link
        # survives moving the repo around.
        target = Path(os.path.relpath(entry, dst_dir))

        if link.is_symlink():
            current = Path(os.readlink(link))
            if current == target:
                ok.append(name)
                continue
            if check:
                drift.append(f"{name}: stale link -> {current}, want {target}")
                continue
            link.unlink()
            link.symlink_to(target)
            created.append(f"{name} (relinked)")
            continue

        if link.exists():
            drift.append(
                f"{name}: {link.relative_to(root)} exists and is not a symlink, "
                f"refusing to overwrite"
            )
            continue

        if check:
            drift.append(f"{name}: missing link")
            continue

        link.symlink_to(target)
        created.append(name)

    for name in ok:
        click.echo(f"ok       {name}")
    for entry in created:
        click.echo(f"linked   {entry}")
    for entry in drift:
        click.echo(f"drift    {entry}", err=True)

    if drift:
        raise SystemExit(1)
