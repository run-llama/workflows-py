from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from . import git_utils, gha, index_html, release_notes, versioning


def _resolve_tag(explicit_tag: Optional[str], github_ref: Optional[str]) -> str:
    if explicit_tag:
        return versioning.strip_refs_prefix(explicit_tag)
    if github_ref and github_ref.startswith("refs/tags/"):
        return versioning.strip_refs_prefix(github_ref)
    raise click.BadParameter(
        "Unable to determine tag. Provide --tag or set GITHUB_REF/GITHUB_REF_NAME to a tag value."
    )


@click.group()
def cli() -> None:
    """Developer tooling for the workflows repository."""


@cli.command("validate-version")
@click.option(
    "--pyproject",
    type=click.Path(exists=True, dir_okay=False),
    default="pyproject.toml",
    show_default=True,
)
@click.option("--tag-prefix", default="", show_default=True)
@click.option("--tag", envvar="GITHUB_REF_NAME")
@click.option("--github-ref", envvar="GITHUB_REF")
def validate_version(
    pyproject: str, tag_prefix: str, tag: Optional[str], github_ref: Optional[str]
) -> None:
    """Ensure the git tag matches the pyproject version."""
    target_tag = _resolve_tag(tag, github_ref)
    pyproject_version = versioning.read_pyproject_version(pyproject)
    tag_version = versioning.extract_semver(target_tag, tag_prefix)
    try:
        versioning.ensure_versions_match(pyproject_version, tag_version, target_tag)
    except versioning.VersionMismatchError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✅ Version validated: {pyproject_version} (tag: {target_tag})")


@cli.command("detect-change-type")
@click.option("--repo", type=click.Path(exists=True, file_okay=False), default=".")
@click.option("--tag-glob", default="v*", show_default=True)
@click.option("--tag-prefix", default="", show_default=True)
@click.option("--current-tag", envvar="GITHUB_REF_NAME")
@click.option("--github-ref", envvar="GITHUB_REF")
@click.option("--output", type=click.Path(), default=None)
def detect_change_type(
    repo: str,
    tag_glob: str,
    tag_prefix: str,
    current_tag: Optional[str],
    github_ref: Optional[str],
    output: Optional[Path],
) -> None:
    """Compute the semantic change between the current tag and the previous release."""
    target_tag = _resolve_tag(current_tag, github_ref)
    tags = git_utils.list_tags(repo, tag_glob)
    previous = git_utils.previous_tag(target_tag, tags)

    current_version = versioning.extract_semver(target_tag, tag_prefix)
    previous_version = (
        versioning.extract_semver(previous, tag_prefix) if previous else None
    )
    change_type = versioning.detect_change_type(current_version, previous_version)

    click.echo(f"Current tag: {target_tag}")
    if previous:
        click.echo(f"Previous tag: {previous}")
    else:
        click.echo("No previous tag found")
    click.echo(f"Change type: {change_type}")

    gha.write_outputs({"change_type": change_type}, output_path=output)


@cli.command("extract-tag-info")
@click.option(
    "--tag", required=True, help="Full git tag (e.g. llama-index-workflows@v1.2.3)."
)
@click.option("--tag-prefix", required=True, help="Expected prefix for the tag.")
@click.option("--output", type=click.Path(), default=None)
def extract_tag_info(tag: str, tag_prefix: str, output: Optional[str]) -> None:
    """Extract suffix and semantic version metadata from a tag."""
    suffix, semver = versioning.compute_suffix_and_version(tag, tag_prefix)
    gha.write_outputs({"tag_suffix": suffix, "semver": semver}, output_path=output)


@cli.command("find-previous-tag")
@click.option("--repo", type=click.Path(exists=True, file_okay=False), default=".")
@click.option("--tag-prefix", required=True)
@click.option("--current-tag", required=True)
@click.option("--output", type=click.Path(), default=None)
def find_previous_tag(
    repo: str, tag_prefix: str, current_tag: str, output: Optional[str]
) -> None:
    """Find the most recent tag for a package other than the current tag."""
    previous = git_utils.find_previous_tag(repo, tag_prefix, current_tag)
    gha.write_outputs({"previous": previous}, output_path=output)


@cli.command("generate-release-notes")
@click.option("--repository", envvar="GITHUB_REPOSITORY", required=True)
@click.option("--github-token", envvar="GITHUB_TOKEN", required=True)
@click.option("--package-label", required=True)
@click.option("--package-name", required=True)
@click.option("--current-tag", required=True)
@click.option("--previous-tag", default="", show_default=True)
@click.option("--semver", required=True)
@click.option("--output", type=click.Path(), default=None)
def generate_release_notes(
    repository: str,
    github_token: str,
    package_label: str,
    package_name: str,
    current_tag: str,
    previous_tag: str,
    semver: str,
    output: Optional[str],
) -> None:
    """Generate release notes by aggregating labeled pull requests."""
    body = release_notes.generate_release_notes(
        token=github_token,
        repository=repository,
        package_label=package_label,
        package_name=package_name,
        current_tag=current_tag,
        previous_tag=previous_tag,
        semver=semver,
    )
    gha.write_outputs({"body": body}, output_path=output)


@cli.command("needs-release")
@click.option("--repo", type=click.Path(exists=True, file_okay=False), default=".")
@click.option(
    "--pyproject",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the package's pyproject.toml file.",
)
@click.option(
    "--tag-prefix",
    required=True,
    help="Git tag prefix for this package (e.g. llama-index-workflows@).",
)
@click.option("--output", type=click.Path(), default=None)
def needs_release(
    repo: str, pyproject: str, tag_prefix: str, output: Optional[Path]
) -> None:
    """Determine whether a package needs a new release tag."""
    current_version = versioning.read_pyproject_version(pyproject)
    tags = git_utils.list_tags(repo, f"{tag_prefix}v*")
    previous_tag = tags[0] if tags else ""
    previous_version = (
        versioning.extract_semver(previous_tag, tag_prefix) if previous_tag else None
    )
    change_type = versioning.detect_change_type(current_version, previous_version)
    release_needed = "true" if change_type != "none" else "false"

    click.echo(f"Detected version: {current_version}")
    if previous_tag:
        click.echo(f"Previous tag: {previous_tag} (version {previous_version})")
    else:
        click.echo("No previous tag found; treating as first release.")
    click.echo(f"Release needed: {release_needed}")

    gha.write_outputs(
        {
            "version": current_version,
            "previous_tag": previous_tag,
            "change_type": change_type,
            "release": release_needed,
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
def update_index_html_cmd(js_url: str, css_url: str, index_path: Optional[str]) -> None:
    """Update debugger asset URLs in the server index.html file."""
    try:
        index_html.update_index_html(js_url, css_url, index_path)
    except Exception as exc:  # pragma: no cover - Click renders traceback
        raise click.ClickException(str(exc)) from exc
    click.echo("✅ Updated index.html")
    click.echo(f"   JavaScript: {js_url}")
    click.echo(f"   CSS: {css_url}")
