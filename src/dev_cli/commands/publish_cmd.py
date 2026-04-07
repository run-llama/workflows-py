# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CI publish commands.

``changeset-plan`` scans the workspace and emits ``publish-plan.json``;
``publish-action`` consumes that plan and runs exactly one entry from
it. The CI workflow expands the plan's lists into a matrix and calls
``publish-action`` once per shard so independent work — most importantly
amd64 and arm64 docker builds — runs in parallel on native runners.
For local releases, ``dev changeset-publish`` runs the same actions
sequentially without going through this command.
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
    return changesets.PublishPlan.from_dict(json.loads(plan_path.read_text()))


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

    Writes GitHub Actions step outputs (``pypi``, ``docker_builds``,
    ``docker_manifests``, ``helm``, ``has_work``) so downstream jobs can
    expand them into ``strategy.matrix``. An empty plan is valid and
    expected when there is nothing to publish.
    """
    os.chdir(_repo_root())
    plan = changesets.build_publish_plan(changesets.get_pnpm_workspace_packages())

    payload = plan.to_dict()
    output.write_text(json.dumps(payload, indent=2))
    click.echo(f"Wrote publish plan to {output}\n")
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


_KIND_PYPI = "pypi"
_KIND_DOCKER = "docker"
_KIND_DOCKER_MANIFEST = "docker-manifest"
_KIND_HELM = "helm"
_KINDS = [_KIND_PYPI, _KIND_DOCKER, _KIND_DOCKER_MANIFEST, _KIND_HELM]


def _select_action(
    plan: changesets.PublishPlan,
    kind: str,
    package: str | None,
    image: str | None,
    platform: str | None,
) -> object:
    """Pick the single plan entry that matches the given selectors."""
    if kind == _KIND_PYPI:
        if not package:
            raise click.UsageError("--package is required for kind=pypi")
        matches = [a for a in plan.pypi if a.package == package]
    elif kind == _KIND_DOCKER:
        if not image or not platform:
            raise click.UsageError(
                "--image and --platform are required for kind=docker"
            )
        matches = [
            a for a in plan.docker_builds if a.image == image and a.platform == platform
        ]
    elif kind == _KIND_DOCKER_MANIFEST:
        if not image:
            raise click.UsageError("--image is required for kind=docker-manifest")
        matches = [a for a in plan.docker_manifests if a.image == image]
    elif kind == _KIND_HELM:
        if not package:
            raise click.UsageError("--package is required for kind=helm")
        matches = [a for a in plan.helm if a.package == package]
    else:
        raise click.UsageError(f"Unknown kind: {kind}")

    if not matches:
        raise click.ClickException(
            f"No {kind} action matched selectors "
            f"(package={package!r}, image={image!r}, platform={platform!r})"
        )
    if len(matches) > 1:
        raise click.ClickException(
            f"Ambiguous selectors matched {len(matches)} {kind} actions"
        )
    return matches[0]


_EXECUTORS = {
    _KIND_PYPI: changesets.execute_pypi_action,
    _KIND_DOCKER: changesets.execute_docker_build_action,
    _KIND_DOCKER_MANIFEST: changesets.execute_docker_manifest_action,
    _KIND_HELM: changesets.execute_helm_action,
}


@click.command("publish-action")
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to publish-plan.json produced by ``changeset-plan``.",
)
@click.option("--kind", type=click.Choice(_KINDS), required=True)
@click.option("--package", default=None, help="Package name (pypi, helm).")
@click.option("--image", default=None, help="Image name (docker, docker-manifest).")
@click.option("--platform", default=None, help="Docker platform (docker).")
@click.option("--dry-run", is_flag=True, help="Log the action instead of running it.")
def publish_action(
    plan_path: Path,
    kind: str,
    package: str | None,
    image: str | None,
    platform: str | None,
    dry_run: bool,
) -> None:
    """Execute one entry from a publish plan."""
    os.chdir(_repo_root())
    plan = _load_plan(plan_path)
    action = _select_action(plan, kind, package, image, platform)
    _EXECUTORS[kind](action, dry_run=dry_run)
