# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""CLI entry point for dev."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import gha, git_utils, index_html, versioning
from .commands.changesets_cmd import changeset_publish, changeset_version
from .commands.pytest_cmd import pytest_cmd

# Known subcommands for argument injection
_KNOWN_SUBCOMMANDS = {
    "pytest",
    "changeset-version",
    "changeset-publish",
    "compute-tag-metadata",
    "update-index-html",
}


def _maybe_inject_pytest_subcommand() -> None:
    """Inject 'pytest' subcommand if no known subcommand is present.

    This enables `dev -k test_foo` to work as shorthand for `dev pytest -k test_foo`.
    """
    if len(sys.argv) < 2:
        # No args: `dev` -> `dev pytest`
        sys.argv.insert(1, "pytest")
        return

    # Check if first arg is a known subcommand or --help
    first_arg = sys.argv[1]
    if first_arg in _KNOWN_SUBCOMMANDS or first_arg in ("--help", "-h"):
        return

    # Not a known subcommand, inject pytest
    sys.argv.insert(1, "pytest")


@click.group()
def cli() -> None:
    """Developer tooling for the llama-agents repository.

    Run without a subcommand to execute pytest across all packages.
    """


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


cli.add_command(changeset_version)
cli.add_command(changeset_publish)
cli.add_command(pytest_cmd)
