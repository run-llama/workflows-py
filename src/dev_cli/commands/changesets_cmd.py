# SPDX-FileCopyrightText: 2025 LlamaIndex Authors
# SPDX-License-Identifier: MIT
"""Changeset commands for versioning and publishing."""

from __future__ import annotations

import os
from pathlib import Path

import click

from .. import changesets


@click.command("changeset-version")
def changeset_version() -> None:
    """Apply changeset versions, then sync versions for co-located Python packages.

    - Runs changesets to bump package.json versions.
    - Discovers all workspace packages via pnpm.
    - For any directory containing both package.json and pyproject.toml, and with
      package.json private: false, set pyproject [project].version to match the JS version.
    - If a pyproject is updated, run `uv sync` in the root directory to update the lock file.
    """
    # Ensure we're at the repo root
    repo_root = Path(__file__).parents[3]
    os.chdir(repo_root)

    # First, run changeset version to update all package.json files
    changesets.run_command(["npx", "@changesets/cli", "version"])

    # Enumerate workspace packages and perform syncs
    packages = changesets.get_pnpm_workspace_packages()
    version_map = {pkg.name: pkg for pkg in packages}
    any_changed = False
    for pkg in packages:
        changed = changesets.sync_package_version_with_pyproject(
            pkg.path, version_map, pkg.name
        )
        any_changed = any_changed or changed

    # If any pyproject.toml was updated, run uv sync in the root to update the lock file
    if any_changed:
        click.echo("Running uv sync to update lock file...")
        changesets.run_command(["uv", "sync"], cwd=repo_root)


@click.command("changeset-publish")
@click.option("--tag", is_flag=True, help="Tag the packages after publishing")
@click.option("--dry-run", is_flag=True, help="Dry run the publish")
def changeset_publish(tag: bool, dry_run: bool) -> None:
    """Publish all packages."""
    # move to the root
    os.chdir(Path(__file__).parents[3])

    changesets.maybe_publish_pypi(dry_run)

    if tag:
        if dry_run:
            click.echo("Dry run, skipping tag. Would run:")
            click.echo("  npx @changesets/cli tag")
            click.echo("  git push --tags")
        else:
            # Let changesets create JS-related tags as usual
            changesets.run_command(["npx", "@changesets/cli", "tag"])
            changesets.run_command(["git", "push", "--tags"])
