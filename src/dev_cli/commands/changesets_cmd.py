# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
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
    repo_root = Path(__file__).parents[3]
    os.chdir(repo_root)

    changesets.run_command(["npx", "@changesets/cli", "version"])

    packages = changesets.get_pnpm_workspace_packages()
    version_map = {pkg.name: pkg for pkg in packages}
    any_changed = False
    for pkg in packages:
        changed = changesets.apply_sync_values(pkg, version_map)
        any_changed = any_changed or changed
        if changed and pkg.postVersion:
            for cmd in pkg.postVersion:
                click.echo(f"Running postVersion script: {cmd}")
                changesets.run_command(["sh", "-c", cmd], cwd=pkg.path)

    if any_changed:
        click.echo("Running uv sync to update lock file...")
        changesets.run_command(["uv", "sync"], cwd=repo_root)


@click.command("changeset-publish")
@click.option("--tag", is_flag=True, help="Tag the packages after publishing")
@click.option("--dry-run", is_flag=True, help="Dry run the publish")
def changeset_publish(tag: bool, dry_run: bool) -> None:
    """Plan and publish all packages locally.

    Builds a publish plan from the current workspace, then runs every
    action sequentially. Same code path the CI fan-out uses, just with
    no parallelism.
    """
    os.chdir(Path(__file__).parents[3])

    plan = changesets.build_publish_plan(changesets.get_pnpm_workspace_packages())
    for action in plan.pypi:
        changesets.execute_pypi_action(action, dry_run=dry_run)
    for build in plan.docker_builds:
        changesets.execute_docker_build_action(build, dry_run=dry_run)
    for manifest in plan.docker_manifests:
        changesets.execute_docker_manifest_action(manifest, dry_run=dry_run)
    for chart in plan.helm:
        changesets.execute_helm_action(chart, dry_run=dry_run)

    if tag:
        if dry_run:
            click.echo("Dry run, skipping tag. Would run:")
            click.echo("  npx @changesets/cli tag")
            click.echo("  git push --tags")
        else:
            changesets.run_command(["npx", "@changesets/cli", "tag"])
            changesets.run_command(["git", "push", "--tags"])
