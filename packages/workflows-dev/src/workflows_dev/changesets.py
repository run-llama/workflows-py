"""
This is a script called by the changeset bot. Normally changeset can do the following things, but this is a mixed ts and python repo, so we need to do some extra things.

There's 2 things this does:
- Versioning: Makes changes that may be committed with the newest version.
- Releasing/Tagging: After versions are changed, we check each package to see if its released, and if not, we release it and tag it.

"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, List, cast

import click
import tomlkit
from packaging.version import Version
from pydantic import BaseModel, Field


def run_command(
    cmd: List[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    """Run a command, streaming output to the console, and raise on failure."""
    subprocess.run(cmd, check=True, text=True, cwd=cwd or Path.cwd(), env=env)


def run_and_capture(
    cmd: List[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    """Run a command and return stdout as text, raising on failure."""
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        cwd=cwd or Path.cwd(),
        env=env,
        capture_output=True,
    )
    return result.stdout


@dataclass
class PackageJson:
    name: str
    version: str
    path: Path
    private: bool


def get_pnpm_workspace_packages() -> list[PackageJson]:
    """Return directories for all workspace packages from pnpm list JSON output."""
    output = run_and_capture(["pnpm", "list", "-r", "--depth=-1", "--json"])

    package_json = cast(list[dict[str, Any]], json.loads(output))
    packages: list[PackageJson] = [
        PackageJson(
            name=data["name"],
            version=data["version"],
            path=Path(data["path"]),
            private=data.get("private", True),
        )
        for data in package_json
    ]
    return packages


def sync_package_version_with_pyproject(
    package_dir: Path, packages: dict[str, PackageJson], js_package_name: str
) -> None:
    """Sync version from package.json to pyproject.toml.

    Returns True if pyproject was changed, else False.
    """
    pyproject_path = package_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return

    package_version = packages[js_package_name].version
    toml_doc, py_doc = PyProjectContainer.parse(pyproject_path.read_text())
    current_version = py_doc.project.version

    # update workspace dependency strings by replacing the first version after == or >=
    # Perhaps sync dependency versions some day
    changed = False
    if current_version != package_version:
        toml_doc["project"]["version"] = package_version
        changed = True

    if changed:
        pyproject_path.write_text(tomlkit.dumps(toml_doc))
        click.echo(
            f"Updated {pyproject_path} version to {package_version} and synced dependency specs"
        )


def _publishable_packages() -> Generator[Path, None, None]:
    """Finds all paths to pyproject.toml that also have a package.json with private: false."""
    packages = get_pnpm_workspace_packages()
    for package in packages:
        if not package.private:
            pyproject = package.path / "pyproject.toml"
            if pyproject.exists():
                yield pyproject


def lock_python_dependencies() -> None:
    """Lock Python dependencies."""
    try:
        run_command(["uv", "lock"])
        click.echo("Locked Python dependencies")
    except subprocess.CalledProcessError as e:
        click.echo(f"Warning: Failed to lock Python dependencies: {e}", err=True)


@click.group()
def cli() -> None:
    """Changeset-based version management for llama-cloud-services."""
    pass


def maybe_publish_pypi(dry_run: bool) -> None:
    """Publish the py packages if they need to be published."""
    any = False
    for package in _publishable_packages():
        name, version = current_version(package)
        if is_published(name, version):
            click.echo(f"PyPI package {name}@{version} already published, skipping")
            continue
        any = True
        click.echo(f"Publishing PyPI package {name}@{version}")

        token = os.environ["UV_PUBLISH_TOKEN"]
        if dry_run:
            summary = (token[:3] + "***") if len(token) <= 6 else token[:6] + "****"
            click.echo(
                f"Dry run, skipping publish. Would run with publish token {summary}:"
            )
            click.echo("  uv publish")
        else:
            run_command(["uv", "build"], cwd=package.parent)
    if any:
        if dry_run:
            click.echo("Dry run, skipping publish. Would run:")
            click.echo("  uv publish")
        else:
            run_command(["uv", "publish"])


def current_version(pyproject: Path) -> tuple[str, str]:
    """Return (package_name, version_str) taken from the given pyproject.toml."""
    toml_doc, py_doc = PyProjectContainer.parse(pyproject.read_text())
    name = py_doc.project.name
    version = str(Version(py_doc.project.version))  # normalise
    return name, version


def is_published(
    name: str, version: str, index_url: str = "https://pypi.org/pypi"
) -> bool:
    """
    True  → `<name>==<version>` exists on the given index
    False → package missing *or* version missing
    """
    url = f"{index_url.rstrip('/')}/{name}/json"
    try:
        data = json.load(urllib.request.urlopen(url))
    except urllib.error.HTTPError as e:  # 404 → package not published at all
        if e.code == 404:
            return False
        raise  # any other error should surface
    return version in data["releases"]  # keys are version strings


if __name__ == "__main__":
    cli()


class PyProjectContainer(BaseModel):
    project: PyProject

    @classmethod
    def parse(cls, text: str) -> tuple[Any, PyProjectContainer]:
        doc = tomlkit.parse(text)
        return doc, PyProjectContainer.model_validate(doc)


class PyProject(BaseModel):
    name: str
    version: str
    dependencies: list[str] = Field(default_factory=list)
