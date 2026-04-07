# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CI publish commands.

``changeset-plan`` scans the workspace and emits ``publish-plan.json``;
``publish-action`` consumes that plan and runs exactly one entry from
it, selected by its ``action.id``. The CI workflow expands the plan's
lists into a matrix and calls ``publish-action`` once per shard so
independent work — most importantly amd64 and arm64 docker builds —
runs in parallel on native runners. For local releases, ``dev
changeset-publish`` runs the same actions sequentially without going
through this command.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from .. import changesets, gha


def _repo_root() -> Path:
    return Path(__file__).parents[3]


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

    output.write_text(plan.model_dump_json(indent=2))
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

    gha.write_outputs(plan.model_dump(mode="json"))


@click.command("publish-action")
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to publish-plan.json produced by ``changeset-plan``.",
)
@click.option(
    "--id",
    "action_id",
    required=True,
    help="Action id from the plan, e.g. 'pypi:my-pkg' or 'docker:foo/bar|linux/amd64'.",
)
@click.option("--dry-run", is_flag=True, help="Log the action instead of running it.")
def publish_action(plan_path: Path, action_id: str, dry_run: bool) -> None:
    """Execute one entry from a publish plan, selected by its action id."""
    os.chdir(_repo_root())
    plan = changesets.PublishPlan.model_validate_json(plan_path.read_text())
    try:
        action = plan.find(action_id)
    except KeyError as e:
        raise click.ClickException(str(e)) from e
    changesets.execute_action(action, dry_run=dry_run)
