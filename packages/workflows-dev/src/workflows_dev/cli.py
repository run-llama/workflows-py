from __future__ import annotations

import os
from pathlib import Path

import click

from . import changesets, gha, git_utils, index_html, versioning


@click.group()
def cli() -> None:
    """Developer tooling for the workflows repository."""


@cli.command("compute-tag-metadata")
@click.option(
    "--tag",
    required=True,
    help="Full git tag to inspect (e.g. llama-index-workflows@v1.2.3).",
)
@click.option("--output", type=click.Path(), default=None)
def compute_tag_metadata(tag: str, output: Path | None) -> None:
    """Compute semantic metadata and change classification for a tag.

    Writes tag_suffix, semver, change_type, and change_description to outputs.
    """
    try:
        metadata = versioning.infer_tag_metadata(tag)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    suffix, semver = versioning.compute_suffix_and_version(tag, metadata.tag_prefix)

    tags = git_utils.list_tags(Path.cwd(), metadata.tag_glob)
    previous = git_utils.previous_tag(metadata.normalized, tags)
    previous_version = (
        versioning.extract_semver(previous, metadata.tag_prefix) if previous else None
    )
    change_type = versioning.detect_change_type(semver, previous_version)
    change_description = ""

    click.echo(f"Current tag: {metadata.normalized}")
    if previous:
        click.echo(f"Previous tag: {previous}")
    else:
        click.echo("No previous tag found")
    click.echo(f"Version: {semver}")
    click.echo(f"Change type: {change_type}")

    gha.write_outputs(
        {
            "tag_suffix": suffix,
            "semver": semver,
            "change_type": change_type,
            "change_description": change_description,
        },
        output_path=output,
    )


@cli.command("update-index-html")
@click.option("--js-url", required=True, help="URL for the JavaScript bundle.")
@click.option("--css-url", required=True, help="URL for the CSS bundle.")
@click.option(
    "--index-path",
    type=click.Path(),
    default=None,
    help="Optional custom index.html path.",
)
def update_index_html_cmd(js_url: str, css_url: str, index_path: str | None) -> None:
    """Update debugger asset URLs in the server index.html file."""
    try:
        index_html.update_index_html(js_url, css_url, index_path)
    except Exception as exc:  # pragma: no cover - Click renders traceback
        raise click.ClickException(str(exc)) from exc
    click.echo("✅ Updated index.html")
    click.echo(f"   JavaScript: {js_url}")
    click.echo(f"   CSS: {css_url}")


@cli.command("changeset-version")
def changeset_version() -> None:
    """Apply changeset versions, then sync versions for co-located Python packages.

    - Runs changesets to bump package.json versions.
    - Discovers all workspace packages via pnpm.
    - For any directory containing both package.json and pyproject.toml, and with
      package.json private: false, set pyproject [project].version to match the JS version.
    - If a pyproject is updated, run `uv sync` in that directory to update its lock file.
    """
    # Ensure we're at the repo root
    os.chdir(Path(__file__).parents[4])

    # First, run changeset version to update all package.json files
    changesets.run_command(["npx", "@changesets/cli", "version"])

    # Enumerate workspace packages and perform syncs
    packages = changesets.get_pnpm_workspace_packages()
    version_map = {pkg.name: pkg for pkg in packages}
    for pkg in packages:
        changesets.sync_package_version_with_pyproject(pkg.path, version_map, pkg.name)


@cli.command("changeset-publish")
@click.option("--tag", is_flag=True, help="Tag the packages after publishing")
@click.option("--dry-run", is_flag=True, help="Dry run the publish")
def publish(tag: bool, dry_run: bool) -> None:
    """Publish all packages."""
    # move to the root
    os.chdir(Path(__file__).parents[4])

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
