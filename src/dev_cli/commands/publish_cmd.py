# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Per-action publish commands used by the CI fan-out workflow.

The ``changeset-plan`` command scans the workspace, decides what needs to
be published, and emits a single ``publish-plan.json`` file. Each
``publish-*`` command below consumes that plan and executes exactly one
action (one PyPI package, one single-arch docker build, one manifest
merge, one helm chart, or the final git tag step). The GitHub Actions
workflow wires these up as a matrix so independent actions run in
parallel — most importantly, amd64 and arm64 docker builds run on their
native runners concurrently instead of emulating arm64 on amd64.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from .. import changesets, gha


def _repo_root() -> Path:
    return Path(__file__).parents[3]


def _load_plan(plan_path: Path) -> changesets.PublishPlan:
    data = json.loads(plan_path.read_text())
    return changesets.PublishPlan.from_dict(data)


@click.command("changeset-plan")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("publish-plan.json"),
    show_default=True,
    help="Where to write the publish plan JSON file.",
)
def changeset_plan(output: Path) -> None:
    """Scan the workspace and emit a publish plan JSON file.

    Also writes GitHub Actions step outputs (``pypi``, ``docker_builds``,
    ``docker_manifests``, ``helm``, ``has_work``) so downstream jobs can
    expand them into ``strategy.matrix``. An empty plan is valid and
    expected when there is nothing to publish — downstream jobs use
    ``if: has_work == 'true'`` (or check the individual list) to no-op.
    """
    os.chdir(_repo_root())
    packages = changesets.get_pnpm_workspace_packages()
    plan = changesets.build_publish_plan(packages)

    payload = plan.to_dict()
    output.write_text(json.dumps(payload, indent=2))
    click.echo(f"Wrote publish plan to {output}")

    # Summary table for the job log.
    click.echo("")
    click.echo("=== Publish plan ===")
    click.echo(f"  pypi:             {len(plan.pypi)}")
    for a in plan.pypi:
        click.echo(f"    - {a.package}@{a.version}")
    click.echo(f"  docker builds:    {len(plan.docker_builds)}")
    for b in plan.docker_builds:
        click.echo(f"    - {b.image}:{b.version} ({b.platform})")
    click.echo(f"  docker manifests: {len(plan.docker_manifests)}")
    for m in plan.docker_manifests:
        click.echo(f"    - {m.image}:{m.version} -> {m.final_tags}")
    click.echo(f"  helm:             {len(plan.helm)}")
    for h in plan.helm:
        click.echo(f"    - {h.package}@{h.version}")
    click.echo("====================")

    has_work = bool(
        plan.pypi or plan.docker_builds or plan.docker_manifests or plan.helm
    )

    gha.write_outputs(
        {
            "pypi": json.dumps(payload["pypi"]),
            "docker_builds": json.dumps(payload["docker_builds"]),
            "docker_manifests": json.dumps(payload["docker_manifests"]),
            "helm": json.dumps(payload["helm"]),
            "has_work": "true" if has_work else "false",
        }
    )


_PLAN_OPTION = click.option(
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to publish-plan.json produced by ``changeset-plan``.",
)
_DRY_RUN_OPTION = click.option(
    "--dry-run", is_flag=True, help="Log the action instead of running it."
)


@click.command("publish-pypi")
@_PLAN_OPTION
@click.option("--package", required=True, help="PyPI package name to publish.")
@_DRY_RUN_OPTION
def publish_pypi(plan_path: Path, package: str, dry_run: bool) -> None:
    """Publish a single PyPI package named in the plan."""
    os.chdir(_repo_root())
    plan = _load_plan(plan_path)
    matches = [a for a in plan.pypi if a.package == package]
    if not matches:
        raise click.ClickException(
            f"PyPI package {package!r} not found in plan (available: "
            f"{[a.package for a in plan.pypi]})"
        )
    for action in matches:
        changesets.execute_pypi_action(action, dry_run=dry_run)


@click.command("publish-docker")
@_PLAN_OPTION
@click.option("--image", required=True, help="Full image name (e.g. llamaindex/foo).")
@click.option("--platform", required=True, help="Docker platform, e.g. linux/amd64.")
@_DRY_RUN_OPTION
def publish_docker(
    plan_path: Path, image: str, platform: str, dry_run: bool
) -> None:
    """Build and push a single-arch docker image described in the plan."""
    os.chdir(_repo_root())
    plan = _load_plan(plan_path)
    matches = [
        a
        for a in plan.docker_builds
        if a.image == image and a.platform == platform
    ]
    if not matches:
        raise click.ClickException(
            f"No docker build for image={image} platform={platform} in plan."
        )
    for action in matches:
        changesets.execute_docker_build_action(action, dry_run=dry_run)


@click.command("publish-docker-manifest")
@_PLAN_OPTION
@click.option("--image", required=True, help="Full image name (e.g. llamaindex/foo).")
@_DRY_RUN_OPTION
def publish_docker_manifest(plan_path: Path, image: str, dry_run: bool) -> None:
    """Create the multi-arch manifest tags for an image."""
    os.chdir(_repo_root())
    plan = _load_plan(plan_path)
    matches = [a for a in plan.docker_manifests if a.image == image]
    if not matches:
        raise click.ClickException(
            f"No docker manifest for image={image} in plan."
        )
    for action in matches:
        changesets.execute_docker_manifest_action(action, dry_run=dry_run)


@click.command("publish-helm")
@_PLAN_OPTION
@click.option("--package", required=True, help="Helm chart package name.")
@_DRY_RUN_OPTION
def publish_helm(plan_path: Path, package: str, dry_run: bool) -> None:
    """Package and push a single Helm chart."""
    os.chdir(_repo_root())
    plan = _load_plan(plan_path)
    matches = [a for a in plan.helm if a.package == package]
    if not matches:
        raise click.ClickException(
            f"Helm chart {package!r} not found in plan."
        )
    for action in matches:
        changesets.execute_helm_action(action, dry_run=dry_run)


@click.command("publish-git-tags")
@_DRY_RUN_OPTION
def publish_git_tags(dry_run: bool) -> None:
    """Run ``changesets tag`` and push tags to the remote.

    This is the final step of the fan-out workflow and runs after all
    publish actions have succeeded. It is safe to run when nothing was
    published (``changesets tag`` is a no-op in that case).
    """
    os.chdir(_repo_root())
    if dry_run:
        click.echo("dry run: npx @changesets/cli tag")
        click.echo("dry run: git push --tags")
        return
    changesets.run_command(["npx", "@changesets/cli", "tag"])
    changesets.run_command(["git", "push", "--tags"])


